import numpy as np
import matplotlib.cm as cm
import imageio
import torch
import os
from tqdm import tqdm
from IPython import embed
import laspy
from pdb import set_trace
from PIL.ImageOps import exif_transpose
from PIL import Image
import torchvision.transforms as tvf
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from time import time, strftime, localtime
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import cv2
from typing import List, Tuple
from save import save_video, save_images, visualize_poses, _resize_pil_image, load_images, save_las, enhanced_visualize_tracks

def project_points_with_depth_ordering(pts_and_colors, 
                                    img_board,
                                    extrinsics, 
                                    intrinsics, 
                                    H, W, orig_H, orig_W,
                                    device='cpu'):
    """
    支持两种输入格式的投影函数：
    1. 背景点云：{'points': [1,N,3], 'colors': [1,N,3]}
    2. 前景点云：{box_id: {'points': [T,Ni,3], 'colors': [T,Ni,3]}}
    
    对于前景点云，会自动按时间帧匹配相机投影（前3张相机用时间帧1的点云...）
    """
    if isinstance(extrinsics, list):
        extrinsics = torch.stack(extrinsics, dim=0).to(device)  # [num_cams, 4, 4]
    else:
        extrinsics = extrinsics.to(device)
        
    if isinstance(intrinsics, list):
        intrinsics = torch.stack(intrinsics, dim=0).to(device)  # [num_cams, 3, 3]
    else:
        intrinsics = intrinsics.to(device)

    dtype = torch.float64
    extrinsics = extrinsics.to(dtype)
    intrinsics = intrinsics.to(dtype)
    
    num_cams = extrinsics.shape[0]
    num_time_frames = num_cams // 3

    if 'points' in pts_and_colors: 
        all_points = pts_and_colors['points'].squeeze(0).to(dtype)  # [N, 3]
        all_colors = pts_and_colors['colors'].squeeze(0)  # [N, 3]
        
        for cam_idx in range(num_cams):
            pts_hom = torch.cat([all_points, torch.ones_like(all_points[:, :1])], dim=-1)
            pts_cam = (pts_hom @ extrinsics[cam_idx].T)[:, :3]
            
            z = pts_cam[:, 2]
            valid = z > 1e-3
            u = intrinsics[cam_idx, 0, 0] * (pts_cam[:, 0]/z) + intrinsics[cam_idx, 0, 2]
            v = intrinsics[cam_idx, 1, 1] * (pts_cam[:, 1]/z) + intrinsics[cam_idx, 1, 2]
            
            u = u.round().long()
            v = v.round().long()
            valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            
            if valid.any():
                u_valid = u[valid]
                v_valid = v[valid]
                colors_valid = all_colors[valid]
                
                depth_order = torch.argsort(z[valid])
                img_board[cam_idx].index_put_(
                    (v_valid[depth_order], u_valid[depth_order]), 
                    colors_valid[depth_order], 
                    accumulate=False
                )

    else:  # {box_id: {'points': [T,Ni,3], 'colors': [T,Ni,3]}}
        for time_idx in range(num_time_frames):
            cam_start = time_idx * 3
            cam_end = cam_start + 3

            combined_points = []
            combined_colors = []
            
            for box_id, data in pts_and_colors.items():
                if time_idx < len(data['points']):
                    frame_points = data['points'][time_idx].squeeze(0)  # [Ni, 3]
                    frame_colors = data['colors'][time_idx].squeeze(0)  # [Ni, 3]
                    combined_points.append(frame_points)
                    combined_colors.append(frame_colors)
            
            if not combined_points:
                continue
                
            all_points = torch.cat(combined_points, dim=0).to(dtype)  # [N_total, 3]
            all_colors = torch.cat(combined_colors, dim=0)  # [N_total, 3]

            for cam_idx in range(cam_start, min(cam_end, num_cams)):
                pts_hom = torch.cat([all_points, torch.ones_like(all_points[:, :1])], dim=-1)
                pts_cam = (pts_hom @ extrinsics[cam_idx].T)[:, :3]
                
                z = pts_cam[:, 2]
                valid = z > 1e-3
                u = intrinsics[cam_idx, 0, 0] * (pts_cam[:, 0]/z) + intrinsics[cam_idx, 0, 2]
                v = intrinsics[cam_idx, 1, 1] * (pts_cam[:, 1]/z) + intrinsics[cam_idx, 1, 2]
                
                u = u.round().long()
                v = v.round().long()
                valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                
                if valid.any():
                    depth_order = torch.argsort(z[valid])
                    u_sorted = u[valid][depth_order]
                    v_sorted = v[valid][depth_order]
                    colors_sorted = all_colors[valid][depth_order]

                    img_board[cam_idx].index_put_(
                        (v_sorted, u_sorted),
                        colors_sorted,
                        accumulate=False
                    )
    
    return img_board

def adjust_depth_scale(depth_map, extra_depth):
    adjusted_depth_map = []

    for i in range(len(extra_depth)):
        current_depth_map = depth_map[i]  # [H, W, 1]
        current_extra_depth = extra_depth[i]  # [1, H', W']

        current_depth_map = current_depth_map.squeeze(-1)  # [H, W]

        depth_min = current_depth_map.min()
        depth_max = current_depth_map.max()
        extra_min = current_extra_depth.min()
        extra_max = current_extra_depth.max()

        scale_factor = (extra_max - extra_min) / (depth_max - depth_min)

        adjusted_depth = (current_depth_map - depth_min) * scale_factor + extra_min

        adjusted_depth = adjusted_depth.unsqueeze(-1)  # [H, W, 1]

        adjusted_depth_map.append(adjusted_depth)

    adjusted_depth_map = torch.stack(adjusted_depth_map, dim=0)  # [N, H', W', 1]
    return adjusted_depth_map

def extend_pose(pose):
    N = pose.shape[0]
    last_row = torch.tensor([0, 0, 0, 1], dtype=pose.dtype, device=pose.device).repeat(N, 1, 1)
    extended_pose = torch.cat([pose, last_row], dim=1)
    return extended_pose

def apply_masks_to_images(images, seg_list, alpha=0.5):
    """
    Args:
        images: Tensor of shape [1, N, 3, H_new, W_new]
        seg_list: List of length N, each element contains seg_data (mask, (box_id,))
                 where mask is [1, H_orig, W_orig]
        alpha: blending factor (0 = original image, 1 = full color)
    Returns:
        colored_images: Tensor of same shape as images
    """
    device = images.device
    color_palette = torch.tensor([
        [1,0,0], [0,1,0], [0,0,1],          # 红绿蓝
        [1,1,0], [1,0,1], [0,1,1],          # 黄品青
        [1,0.5,0], [0.5,0,1], [0,0.5,1],    # 橙紫蓝
        [0.5,1,0], [1,0,0.5], [0.5,0.5,1]   # 浅绿粉蓝
    ], device=device)

    images = images.squeeze(0)
    colored_images = []
    
    for img_idx, (img_tensor, seg_data) in enumerate(zip(images, seg_list)):
        colored_img = img_tensor.clone() # img_tensor: [3, H_new, W_new]
        
        if seg_data:
            _, _, H_new, W_new = images.shape
            for mask, (box_id,) in seg_data:
                mask = mask.unsqueeze(0)  # [1, 1, H_orig, W_orig]
                mask = F.interpolate(mask, 
                                   size=(H_new, W_new), 
                                   mode='bilinear',  # or 'nearest'
                                   align_corners=False)
                mask = mask.squeeze(0).squeeze(0)  # [H_new, W_new]
                
                color_idx = hash(box_id) % len(color_palette)
                color = color_palette[color_idx].view(3, 1, 1)  # [3,1,1]
                
                colored_mask = color.expand(-1, H_new, W_new) # colored mask [3, H_new, W_new]
                
                # Blend with original image
                mask_bool = mask > 0.5
                colored_img[:, mask_bool] = (1-alpha)*colored_img[:, mask_bool] + alpha*colored_mask[:, mask_bool]
        
        colored_images.append(colored_img)
    
    return torch.stack(colored_images, dim=0).unsqueeze(0)

def split_point_cloud_and_colors(images, point_maps, seg_list):
    device = images.device
    num_frames = len(point_maps) // 3
    images = images.squeeze(0)  # [N, 3, H, W]
    
    persistent_ids = set()

    if num_frames > 1:
        frame_id_sets = [set() for _ in range(num_frames)]
        
        for time_idx in range(num_frames):
            frame_start = time_idx * 3
            frame_end = frame_start + 3
            
            for idx in range(frame_start, min(frame_end, len(seg_list))):
                if seg_list[idx]:
                    for _, (box_id,) in seg_list[idx]:
                        frame_id_sets[time_idx].add(box_id)
        
        if frame_id_sets:
            persistent_ids = set(frame_id_sets[0])
            for t in range(1, num_frames):
                persistent_ids.intersection_update(frame_id_sets[t])
    
    bg_points = []
    bg_colors = []
    fg_data = {}  # {id: {'points': [], 'colors': []}}
    
    for time_idx in range(num_frames):
        frame_start = time_idx * 3
        frame_end = frame_start + 3
        
        temp_bg_points = []
        temp_bg_colors = []
        
        for idx in range(frame_start, frame_end):
            points = point_maps[idx].reshape(-1, 3)  # [H*W, 3]
            image = images[idx].permute(1,2,0).reshape(-1,3)  # [H*W, 3]
            
            if idx >= len(seg_list) or not seg_list[idx]:
                temp_bg_points.append(points.unsqueeze(0))
                temp_bg_colors.append(image.unsqueeze(0))
                continue
                
            bg_mask = torch.ones(points.shape[0], dtype=torch.bool, device=device)
            
            for mask, (box_id,) in seg_list[idx]:
                mask = mask.squeeze(0)
                if mask.shape != point_maps[idx].shape[:2]:
                    mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0),
                                       size=point_maps[idx].shape[:2],
                                       mode='nearest').bool().squeeze()
                
                mask_flat = mask.view(-1)
                masked_points = points[mask_flat]
                masked_colors = image[mask_flat]
                
                if box_id in persistent_ids:
                    if box_id not in fg_data:
                        fg_data[box_id] = {
                            'points': [[] for _ in range(num_frames)],
                            'colors': [[] for _ in range(num_frames)]
                        }
                    fg_data[box_id]['points'][time_idx].append(masked_points.unsqueeze(0))
                    fg_data[box_id]['colors'][time_idx].append(masked_colors.unsqueeze(0))
                
                bg_mask = bg_mask & ~mask_flat
            
            temp_bg_points.append(points[bg_mask].unsqueeze(0))
            temp_bg_colors.append(image[bg_mask].unsqueeze(0))
        
        bg_points.append(torch.cat(temp_bg_points, dim=1))
        bg_colors.append(torch.cat(temp_bg_colors, dim=1))

    bg_result = {
        'points': torch.cat(bg_points, dim=1),
        'colors': torch.cat(bg_colors, dim=1)
    }
    
    for box_id in fg_data:
        fg_data[box_id]['points'] = [
            torch.cat(frame_points, dim=1) 
            for frame_points in fg_data[box_id]['points']
        ]
        fg_data[box_id]['colors'] = [
            torch.cat(frame_colors, dim=1) 
            for frame_colors in fg_data[box_id]['colors']
        ]
    
    return bg_result, fg_data

def convert_seg_list_to_coords(
    seg_list: List[Tuple[torch.Tensor, Tuple[int]]],
    new_size: Tuple[int, int],
    threshold: float = 0.5
) -> List[Tuple[torch.Tensor, Tuple[int]]]:
    """
    Convert masks in seg_list from [1, H_orig, W_orig] to [N, 2] coordinates (y, x).
    
    Args:
        seg_list: List of (mask, (box_id,)), where mask is [1, H_orig, W_orig].
        new_size: Target size (H_new, W_new).
        threshold: Mask value threshold to consider a pixel as valid.
    
    Returns:
        List of (coords, (box_id,)), where coords is [N, 2] (y, x) coordinates.
    """
    H_new, W_new = new_size
    new_seg_list = []
    
    for mask, (box_id,) in seg_list:
        # Step 1: Resize mask to [1, H_new, W_new]
        mask = mask.unsqueeze(0)  # [1, 1, H_orig, W_orig]
        mask = F.interpolate(
            mask, 
            size=(H_new, W_new), 
            mode='bilinear',  # or 'nearest' for hard masks
            align_corners=False
        )
        mask = mask.squeeze(0).squeeze(0)  # [H_new, W_new]
        
        # Step 2: Get coordinates where mask > threshold
        y_indices, x_indices = torch.where(mask > threshold)
        coords = torch.stack([x_indices, y_indices], dim=1)  # [N, 2]
        
        new_seg_list.append((coords, (box_id,)))
    
    return new_seg_list

def vggt_forward_true_pose(
    model, 
    pose_encoding_to_extri_intri, 
    unproject_depth_map_to_point_map, 
    gt_images_list, 
    extra_images_list, 
    gt_pose=None, 
    gt_intrinsic=None, 
    extra_pose=None, 
    extra_intrinsic=None, 
    extra_depth=None, 
    extra_seg_list=None,
    scene_name="noname",
    vis_dir=None, 
    device='cpu'):

    conversion_matrix = torch.tensor([[0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]], device=device, dtype=torch.float64)

    C, orig_H, orig_W = extra_images_list[0].shape

    images = load_images(
        extra_images_list,
        device=device
    )
    _, C, H, W = images.shape   
    frame_num = len(extra_images_list)  
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        depth_map = depth_map.squeeze(0)

        # for dust3r frames
        extrinsic = torch.cat(extra_pose, dim=0)  # [n, 3, 4]      
        intrinsic = []
        for i in range(frame_num):
            K = extra_intrinsic[i].squeeze(0)         
            K[0, :] = K[0, :] * W
            K[1, :] = K[1, :] * H    
            # K[0, 2] = W / 2
            # K[1, 2] = H / 2 
            K[0, 1] = 0
            K[1, 0] = 0 
            intrinsic.append(K)
        intrinsic = torch.stack(intrinsic, dim=0)
        
        # for predicted frames
        gt_K = []
        for n in range(len(gt_pose)):
            gt_pose[n] = gt_pose[n].squeeze(0) # @ torch.inverse(conversion_matrix)
            K = gt_intrinsic[n].squeeze(0)         
            K[0, :] = K[0, :] * W
            K[1, :] = K[1, :] * H     
            # K[0, 2] = W / 2
            # K[1, 2] = H / 2 
            K[0, 1] = 0
            K[1, 0] = 0 
            gt_K.append(K)     

        # for i in range(len(gt_pose) - 1 - 3):
        #     if i % 3 == 0:
        #         delta_pose = torch.matmul(gt_pose[i], torch.inverse(gt_pose[i + 3]))
        #         print(delta_pose)

        depth_map = adjust_depth_scale(depth_map, extra_depth)

        # # Predict Point Maps
        # point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
            
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map, 
                                                                    extrinsic, 
                                                                    intrinsic)
        point_map_by_unprojection = torch.from_numpy(point_map_by_unprojection).to(device)

    # Aggregate colors
    images = apply_masks_to_images(images, extra_seg_list)
    bkg_points, frg_points_dict = split_point_cloud_and_colors(images, point_map_by_unprojection, extra_seg_list)  # [B, H*W, 3]

    img_board = [torch.zeros(H, W, 3, device=device) for i in range(frame_num)]
    img_board = project_points_with_depth_ordering(
        bkg_points,
        img_board,
        gt_pose, 
        gt_K, 
        H, W, orig_H, orig_W,
        device=device)
    img_board = project_points_with_depth_ordering(
        frg_points_dict,
        img_board,
        gt_pose, 
        gt_K, 
        H, W, orig_H, orig_W,
        device=device)
    # Resize
    for i in range(len(img_board)):
        img_board[i] = img_board[i].permute(2, 0, 1).unsqueeze(0)
        img_board[i] = F.interpolate(img_board[i], size=(orig_H, orig_W), mode='nearest')

    first_frame_image = torch.cat(img_board, dim=0) # [num_frame, C, H, W]

    if vis_dir is not None:
        timestamp = time()
        gt_imgs = torch.stack(gt_images_list, dim=0)
        scene_folder = os.path.join(vis_dir, f"{scene_name}_{timestamp}")
        gt_folder = os.path.join(scene_folder, "gt")
        os.makedirs(gt_folder,exist_ok=True)
        save_video(gt_imgs[0::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(gt_folder, f'gt0.mp4'), fps=5)
        save_video(gt_imgs[1::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(gt_folder, f'gt1.mp4'), fps=5)
        save_video(gt_imgs[2::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(gt_folder, f'gt2.mp4'), fps=5)

        src_imgs = torch.stack(extra_images_list, dim=0)
        src_folder = os.path.join(scene_folder, "src")
        os.makedirs(src_folder,exist_ok=True)

        save_video(src_imgs[0::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(src_folder, f'src0.mp4'), fps=5, seg_data=extra_seg_list[0::3])
        save_video(src_imgs[1::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(src_folder, f'src1.mp4'), fps=5, seg_data=extra_seg_list[1::3])
        save_video(src_imgs[2::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(src_folder, f'src2.mp4'), fps=5, seg_data=extra_seg_list[2::3])

        pseudo_folder = os.path.join(scene_folder, "pseudo")
        os.makedirs(pseudo_folder,exist_ok=True)
        save_video(first_frame_image[0::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(pseudo_folder, f'pseudo0.mp4'), fps=5)
        save_video(first_frame_image[1::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(pseudo_folder, f'pseudo1.mp4'), fps=5)
        save_video(first_frame_image[2::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(pseudo_folder, f'pseudo2.mp4'), fps=5)
        save_images(src_imgs.permute(0,2,3,1).detach().cpu().numpy(), src_folder)
        save_las([bkg_points['points']], [bkg_points['colors']], os.path.join(src_folder, f"pts.las"))

        seg_folder = os.path.join(scene_folder, "seged")
        os.makedirs(seg_folder,exist_ok=True)
        save_video(images[0, 0::3], os.path.join(seg_folder, f'seg0.mp4'), fps=5)
        save_video(images[0, 1::3], os.path.join(seg_folder, f'seg1.mp4'), fps=5)
        save_video(images[0, 2::3], os.path.join(seg_folder, f'seg2.mp4'), fps=5)
        
        print(f"Saved vis in {src_folder}, {pseudo_folder}")
        
    return first_frame_image



def vggt_forward(
    model, 
    pose_encoding_to_extri_intri, 
    unproject_depth_map_to_point_map, 
    gt_images_list, 
    extra_images_list, 
    extra_seg_list=None,
    scene_name="noname",
    vis_dir=None, 
    device='cpu'):

    C, orig_H, orig_W = extra_images_list[0].shape

    images = load_images(
        extra_images_list + gt_images_list,
        device=device
    )
    all_frame_num, C, H, W = images.shape   
    frame_num = len(extra_images_list) 

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        extrinsic = extrinsic.squeeze(0)
        intrinsic = intrinsic.squeeze(0)

        extrinsic_src = extrinsic[:-len(gt_images_list)]
        extrinsic_src = extend_pose(extrinsic_src)
        intrinsic_src = intrinsic[:-len(gt_images_list)]

        extrinsic_gt = extrinsic[-len(gt_images_list):]
        extrinsic_gt = extend_pose(extrinsic_gt) #w2c
        # extrinsic_gt = torch.inverse(extrinsic_gt) #c2w
        intrinsic_gt = intrinsic[-len(gt_images_list):]

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        depth_map_src = depth_map.squeeze(0)[:-len(gt_images_list)]

        if not extra_seg_list[0]:
            query_points = torch.zeros(0, 2, device=device)
            track_list = torch.zeros(1, all_frame_num, 0, 2, device=device)
            vis_score = torch.zeros(1, all_frame_num, 0, device=device)
            conf_score =torch.zeros(1, all_frame_num, 0, device=device)
        else:
            query_points = convert_seg_list_to_coords(extra_seg_list[0], [H, W])[0][0]
            # choose your own points to track, with shape (N, 2) for one scene
            track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])
            track_list = track_list[-1]
            enhanced_visualize_tracks(
                images=images,                    # 输入图像 [1,18,3,H,W]
                tracks=track_list,                # 轨迹点 [1,18,N,2]
                vis_scores=vis_score,             # 可见性分数 [1,18,N]
                conf_scores=conf_score,           # 置信度分数 [1,18,N]
                query_points=query_points,        # 查询点 [1,M,2]
                output_dir=vis_dir,
                vis_threshold=0.2,
                conf_threshold=0.2,
                cmap_name="rainbow",              # 使用彩虹色映射
                frames_per_row=6,                  # 每行6帧
                device=device
            )
        
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map_src, 
                                                                    extrinsic_src, 
                                                                    intrinsic_src)
        point_map_by_unprojection = torch.from_numpy(point_map_by_unprojection).to(device)

    # Aggregate colors
    images = apply_masks_to_images(images, extra_seg_list)
    bkg_points, frg_points_dict = split_point_cloud_and_colors(images, point_map_by_unprojection, extra_seg_list)  # [B, H*W, 3]

    img_board = [torch.zeros(H, W, 3, device=device) for i in range(frame_num)]
    img_board = project_points_with_depth_ordering(
        bkg_points,
        img_board,
        extrinsic_gt, 
        intrinsic_gt, 
        H, W, orig_H, orig_W,
        device=device)
    img_board = project_points_with_depth_ordering(
        frg_points_dict,
        img_board,
        extrinsic_gt, 
        intrinsic_gt, 
        H, W, orig_H, orig_W,
        device=device)
    # Resize
    for i in range(len(img_board)):
        img_board[i] = img_board[i].permute(2, 0, 1).unsqueeze(0)
        img_board[i] = F.interpolate(img_board[i], size=(orig_H, orig_W), mode='nearest')

    first_frame_image = torch.cat(img_board, dim=0) # [num_frame, C, H, W]

    if vis_dir is not None:
        timestamp = time()
        gt_imgs = torch.stack(gt_images_list, dim=0)
        scene_folder = os.path.join(vis_dir, f"{scene_name}_{timestamp}")
        gt_folder = os.path.join(scene_folder, "gt")
        os.makedirs(gt_folder,exist_ok=True)
        save_video(gt_imgs[0::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(gt_folder, f'gt0.mp4'), fps=5)
        save_video(gt_imgs[1::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(gt_folder, f'gt1.mp4'), fps=5)
        save_video(gt_imgs[2::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(gt_folder, f'gt2.mp4'), fps=5)

        src_imgs = torch.stack(extra_images_list, dim=0)
        src_folder = os.path.join(scene_folder, "src")
        os.makedirs(src_folder,exist_ok=True)

        save_video(src_imgs[0::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(src_folder, f'src0.mp4'), fps=5, seg_data=extra_seg_list[0::3])
        save_video(src_imgs[1::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(src_folder, f'src1.mp4'), fps=5, seg_data=extra_seg_list[1::3])
        save_video(src_imgs[2::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(src_folder, f'src2.mp4'), fps=5, seg_data=extra_seg_list[2::3])

        pseudo_folder = os.path.join(scene_folder, "pseudo")
        os.makedirs(pseudo_folder,exist_ok=True)
        save_video(first_frame_image[0::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(pseudo_folder, f'pseudo0.mp4'), fps=5)
        save_video(first_frame_image[1::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(pseudo_folder, f'pseudo1.mp4'), fps=5)
        save_video(first_frame_image[2::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(pseudo_folder, f'pseudo2.mp4'), fps=5)
        save_images(src_imgs.permute(0,2,3,1).detach().cpu().numpy(), src_folder)
        save_las([bkg_points['points']], [bkg_points['colors']], os.path.join(src_folder, f"pts.las"))

        seg_folder = os.path.join(scene_folder, "seged")
        os.makedirs(seg_folder,exist_ok=True)
        save_video(images[0, 0::3], os.path.join(seg_folder, f'seg0.mp4'), fps=5)
        save_video(images[0, 1::3], os.path.join(seg_folder, f'seg1.mp4'), fps=5)
        save_video(images[0, 2::3], os.path.join(seg_folder, f'seg2.mp4'), fps=5)

        print(f"Saved vis in {src_folder}, {pseudo_folder}")

        
    return first_frame_image





