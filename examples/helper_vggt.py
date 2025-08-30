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

def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False, seg_data=None):
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().float()
        frames = frames.permute(0, 2, 3, 1)  # -> [N, H, W, C]
        frames = frames.numpy()

    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])

    id_to_color = {}
    colormap = np.array(cm.get_cmap("tab20").colors) 

    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            frame = frames[i].copy()
            h, w = frame.shape[:2]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 1)
                frame = (frame * 255).astype(np.uint8)
            if seg_data is not None:
                current_seg = seg_data[i]
                overlay = frame.copy()
                
                for j, (mask, (box_id,)) in enumerate(current_seg):
                    if box_id not in id_to_color:
                        color_idx = hash(box_id) % len(colormap)
                        id_to_color[box_id] = (colormap[color_idx] * 255).astype(np.uint8)
                    
                    color = id_to_color[box_id]
                    mask_np = mask.squeeze(0).cpu().numpy()
                    
                    if mask_np.shape != frame.shape[:2]:
                        mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0])) > 0.5
                    
                    alpha = 0.97
                    overlay[mask_np > 0.5] = (1 - alpha) * overlay[mask_np > 0.5] + alpha * color

                    contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w_box, h_box = cv2.boundingRect(contours[0])
                        cv2.putText(overlay, box_id[:6], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1)
                
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            writer.append_data(frame)

    writer.close()

def save_images(frames, output_folder, is_depths=False, grayscale=False):
    os.makedirs(output_folder, exist_ok=True)

    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            imageio.imwrite(os.path.join(output_folder, f"frame_{i:04d}.png"), depth_vis)
    else:
        for i in range(frames.shape[0]):
            frame = frames[i].copy()
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 1)
                frame = (frame * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_folder, f"frame_{i:04d}.png"), frame)

def visualize_poses(poses_list, save_path):
    c2w_poses = torch.stack(poses_list).detach().cpu().numpy()

    positions = c2w_poses[:, :3, 3]
    right_axes = c2w_poses[:, :3, 0]  # 右向轴（X）
    up_axes = c2w_poses[:, :3, 1]     # 上向轴（Y）
    forward_axes = c2w_poses[:, :3, 2] # 前向轴（Z）

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2], 
        'k-',  # 黑色实线
        linewidth=1, 
        alpha=0.5,  # 半透明
        label='Camera Path'
    )
    for i, (x, y, z) in enumerate(positions):
        ax.text(x, y, z, str(i), color='red', fontsize=8)  # 在点上标数字
    # 2. 绘制相机位置（散点）
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2], 
        c='b', 
        s=20,  # 点大小
        label='Camera Positions'
    )
    for i in range(0, len(positions)):
        # ax.quiver(
        #     positions[i, 0], positions[i, 1], positions[i, 2],
        #     right_axes[i, 0], right_axes[i, 1], right_axes[i, 2],
        #     length=0.05, arrow_length_ratio=0.1, color='r', label='Right (X)' if i == 0 else None
        # )
        # ax.quiver(
        #     positions[i, 0], positions[i, 1], positions[i, 2],
        #     up_axes[i, 0], up_axes[i, 1], up_axes[i, 2],
        #     length=0.05, arrow_length_ratio=0.1, color='g', label='Up (Y)' if i == 0 else None
        # )
        ax.quiver(
            positions[i, 0], positions[i, 1], positions[i, 2],
            forward_axes[i, 0], forward_axes[i, 1], forward_axes[i, 2],
            length=0.05, arrow_length_ratio=0.1, color='b', label='Forward (Z)' if i == 0 else None
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(pil_images, device='cpu'):
    images = []
    shapes = set()
    to_tensor = tvf.ToTensor()

    # First process all images and collect their shapes
    for img in pil_images:

        img = to_pil_image(img).convert("RGB")

        width, height = img.size
        new_width = 518

        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img).to(device)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518

        if new_height > 518:
            start_y = (new_height - 518) // 2
            img = img[:, start_y : start_y + 518, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(pil_images) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def save_las(all_pts3d_world, colored, filename):
    points = []
    for i in all_pts3d_world:
        if i is not None:
            points.append(i)
    points = torch.cat(points, dim=1).squeeze(0).detach().cpu().numpy()
    colors = []
    for i in colored:
        if i is not None:
            colors.append(i)
    colors = torch.cat(colors, dim=1).squeeze(0).detach().cpu().numpy()

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    points = np.array(points)
    colors = np.array(colors)

    if colors.dtype == np.float64 or colors.dtype == np.float32:
        colors = (colors * 255).astype(np.uint8)

    header = laspy.LasHeader(point_format=2)
    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]

    las.write(filename)


def project_points_with_depth_ordering(pts_and_colors, 
                                    img_board,
                                    extrinsics, 
                                    intrinsics, 
                                    H, W, orig_H, orig_W,
                                    device='cpu'):

    # 1. Batch all points and colors
    all_points = pts_and_colors['points'].squeeze(0)  # [N, 3]
    all_colors = pts_and_colors['colors'].squeeze(0)  # [N, 3]

    if isinstance(extrinsics, list):
        extrinsics = torch.stack(extrinsics, dim=0).to(device)  # [num_cams, 4, 4]
    else:
        extrinsics = extrinsics.to(device)
        
    if isinstance(intrinsics, list):
        intrinsics = torch.stack(intrinsics, dim=0).to(device)  # [num_cams, 3, 3]
    else:
        intrinsics = intrinsics.to(device)

    dtype = torch.float64
    all_points = all_points.to(dtype)
    extrinsics = extrinsics.to(dtype)
    intrinsics = intrinsics.to(dtype)

    # 2. Process each camera
    num_cams = extrinsics.shape[0]
    
    for cam_idx in range(num_cams):
        # Transform to camera space [N, 3]
        pts_hom = torch.cat([all_points, torch.ones_like(all_points[:, :1])], dim=-1)
        pts_cam = (pts_hom @ extrinsics[cam_idx].T)[:, :3]
        
        # Projection [N]
        z = pts_cam[:, 2]
        valid = z > 1e-3
        u = intrinsics[cam_idx, 0, 0] * (pts_cam[:, 0]/z) + intrinsics[cam_idx, 0, 2]
        v = intrinsics[cam_idx, 1, 1] * (pts_cam[:, 1]/z) + intrinsics[cam_idx, 1, 2]
        
        # Pixel coordinates and validity
        u = u.round().long()
        v = v.round().long()
        valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        
        if not valid.any():
            continue
            
        # Get valid elements
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = z[valid]
        colors_valid = all_colors[valid]
        
        # Sort by depth (nearest first)
        depth_order = torch.argsort(z_valid)
        u_sorted = u_valid[depth_order]
        v_sorted = v_valid[depth_order]
        colors_sorted = colors_valid[depth_order]
        
        # Create image - using index_put for correct ordering
        img_board[cam_idx].index_put_((v_sorted, u_sorted), colors_sorted, accumulate=False)
    
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
    """
    Split point clouds and colors with temporal awareness.
    
    Args:
        images: Tensor of shape [1, N, 3, H, W] 
                (N should be multiples of 3 for 3 time frames)
        point_maps: List of N point clouds [H, W, 3]
        seg_list: List of N segmentation data
    
    Returns:
        tuple: (bg_data, fg_data) where:
        - bg_data: {'points': [1, M, 3], 'colors': [1, M, 3]}
        - fg_data: {
            box_id: {
                'points': [T, Ni, 3],  # T时间帧的点云 
                'colors': [T, Ni, 3]    # 对应颜色
            }
          }
    """
    device = images.device
    num_frames = len(point_maps) // 3  # Assuming 3 images per time frame
    images = images.squeeze(0)  # [N, 3, H, W]
    
    # First pass: Identify persistent IDs (appear in all time frames)
    persistent_ids = set()
    if num_frames > 1:
        id_counter = {}
        for idx in range(len(seg_list)):
            if idx < len(seg_list) and seg_list[idx]:
                for _, (box_id,) in seg_list[idx]:
                    id_counter[box_id] = id_counter.get(box_id, 0) + 1
        
        persistent_ids = {bid for bid, cnt in id_counter.items() 
                        if cnt >= num_frames * 0.8}  # 80%出现阈值
    
    # Second pass: Process data
    bg_points = [[] for _ in range(num_frames)]  # Background per frame
    bg_colors = [[] for _ in range(num_frames)]
    fg_data = {}  # {id: {'points': [], 'colors': []}}
    
    for time_idx in range(num_frames):
        frame_start = time_idx * 3
        frame_end = frame_start + 3
        
        for idx in range(frame_start, frame_end):
            points = point_maps[idx].reshape(-1, 3)  # [H*W, 3]
            image = images[idx].permute(1,2,0).reshape(-1,3)  # [H*W, 3]
            
            if idx >= len(seg_list) or not seg_list[idx]:
                bg_points[time_idx].append(points.unsqueeze(0))
                bg_colors[time_idx].append(image.unsqueeze(0))
                continue
                
            bg_mask = torch.ones(points.shape[0], dtype=torch.bool, device=device)
            
            for mask, (box_id,) in seg_list[idx]:
                # Process mask (same as before)
                mask = mask.squeeze(0)
                if mask.shape != point_maps[idx].shape[:2]:
                    mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0),
                                       size=point_maps[idx].shape[:2],
                                       mode='nearest').bool().squeeze()
                
                mask_flat = mask.view(-1)
                masked_points = points[mask_flat]
                masked_colors = image[mask_flat]
                
                # Only store if ID is persistent
                if box_id in persistent_ids:
                    if box_id not in fg_data:
                        fg_data[box_id] = {
                            'points': [[] for _ in range(num_frames)],
                            'colors': [[] for _ in range(num_frames)]
                        }
                    fg_data[box_id]['points'][time_idx].append(masked_points.unsqueeze(0))
                    fg_data[box_id]['colors'][time_idx].append(masked_colors.unsqueeze(0))
                
                bg_mask = bg_mask & ~mask_flat
            
            # Store background for this frame
            bg_points[time_idx].append(points[bg_mask].unsqueeze(0))
            bg_colors[time_idx].append(image[bg_mask].unsqueeze(0))
    
    # Process background (merge across frames)
    bg_result = {
        'points': torch.cat([torch.cat(frame_points, dim=1) 
                           for frame_points in bg_points], dim=1),
        'colors': torch.cat([torch.cat(frame_colors, dim=1) 
                           for frame_colors in bg_colors], dim=1)
    }
    
    # Process foreground (keep time separation)
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

    masked_extra_images_list = apply_masks_to_images_old(extra_images_list, extra_seg_list)
    masked_images = load_images(
        masked_extra_images_list,
        device=device
    )

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
            K[0, 2] = W / 2
            K[1, 2] = H / 2 
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
            K[0, 2] = W / 2
            K[1, 2] = H / 2 
            K[0, 1] = 0
            K[1, 0] = 0 
            gt_K.append(K)     

        depth_map = adjust_depth_scale(depth_map, extra_depth)

        # # Predict Point Maps
        # point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
            
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map, 
                                                                    extrinsic, 
                                                                    intrinsic)
        point_map_by_unprojection = torch.from_numpy(point_map_by_unprojection).to(device)

    all_pts3d_world = []
    for i in range(frame_num):
        pts3d_world = point_map_by_unprojection[i].reshape(H * W, 3).unsqueeze(0)
        all_pts3d_world.append(pts3d_world)  # [B, H*W, 3]

    # Aggregate colors
    colored = [masked_images[i].reshape(1, 3, H * W).permute(0, 2, 1) for i in range(frame_num)]   # [B, H*W, 3]
    output_img_list = project_points_with_depth_ordering(all_pts3d_world, 
                                        colored,
                                        gt_pose, 
                                        gt_K, 
                                        H, W, orig_H, orig_W,
                                        device=device)

    first_frame_image = torch.cat(output_img_list, dim=0) # [num_frame, C, H, W]

    if vis_dir is not None:
        timestamp = time()
        gt_imgs = torch.stack(gt_images_list, dim=0)
        scene_folder = os.path.join(vis_dir, f"{scene_name}_{timestamp}")
        gt_folder = os.path.join(scene_folder, "gt")
        os.makedirs(gt_folder,exist_ok=True)
        save_video(gt_imgs[0::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(gt_folder, f'gt0.mp4'), fps=5)
        save_video(gt_imgs[1::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(gt_folder, f'gt1.mp4'), fps=5)
        save_video(gt_imgs[2::3].permute(0,2,3,1).detach().cpu().numpy(), os.path.join(gt_folder, f'gt2.mp4'), fps=5)

        src_imgs = torch.stack(masked_extra_images_list, dim=0)
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
        save_las(all_pts3d_world, colored, os.path.join(src_folder, f"pts.las"))
        
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
    _, C, H, W = images.shape   
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
    for id in frg_points_dict:
        img_board = project_points_with_depth_ordering(
            frg_points_dict[id],
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





