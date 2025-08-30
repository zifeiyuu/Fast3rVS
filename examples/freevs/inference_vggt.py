import sys
import torch
import os
from src.pipelines.pipeline_vggt import StableVideoDiffusionPipeline_vggt
from diffusers.utils import load_image, export_to_video
from glob import glob
import pickle 
import random
import imageio
import numpy as np
from PIL import Image
import torch.nn.functional as F
import math

import argparse

from ipdb import set_trace
from IPython import embed
from tqdm import tqdm
from torchvision import transforms
from helper_vggt import save_video, save_images, save_las, vggt_forward
import matplotlib.cm as cm
from torchvision.transforms.functional import to_pil_image

import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
vggt_path = os.path.join(base_dir, "modules/vggt")
sys.path.insert(0, vggt_path)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
sys.path.pop(0)
from moviepy import VideoFileClip, concatenate_videoclips

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--model_path', type=str,default='work_dirs/freevs_waymo_halfreso_multiframe')
parser.add_argument('--img_pickle', type=str,default = 'waymo_process/waymo_only_front_gt.pkl')
parser.add_argument('--output_dir', type=str, default='eval321') 
parser.add_argument('--video_length', type=int, default=9) # batch frame num
parser.add_argument('--prev_nframes', type=int, default=3)
parser.add_argument("--front_only",action="store_true",default=False,help="whether to randomly flip images horizontally",)
args = parser.parse_args()

def add_camera_view(video):
    nearbycamera = {
                        'FRONT': [['FRONT_LEFT', 'FRONT_RIGHT'], [1, 3]],
                        'FRONT_RIGHT':[['FRONT', 'SIDE_RIGHT'], [-3, 1]], 
                        'FRONT_LEFT':[['FRONT', 'SIDE_LEFT'], [-1, 1]],
                        'SIDE_RIGHT':[['FRONT_RIGHT'], [-1]],
                        'SIDE_LEFT':[['FRONT_LEFT'], [-1]]
                    }
    new_video = []
    for i in range(len(video)):
        fpath = video[i]['image']
        parts = fpath.split('/')
        folder_pattern = parts[-2]
        file_pattern = parts[-1]
        file_parts = file_pattern.split('_')

        image_number = int(file_parts[0])

        is_front = True

        if file_parts[-2].isalpha():
            file_parts[-2:] = ['_'.join(file_parts[-2:])]
            is_front = False
        camera_view = file_parts[-1].split('.')[0]

        if camera_view in nearbycamera:
            replacements, offsets = nearbycamera[camera_view]
            for replacement, offset in zip(replacements, offsets):
                new_image_number = image_number + offset
                new_image_number_str = f"{new_image_number:04d}" 

                new_file_parts = file_parts.copy()
                new_file_parts[0] = new_image_number_str 
                new_file_parts[-1] = replacement + '.jpg'
                new_file_pattern = '_'.join(new_file_parts)
                new_fpath = '/'.join(parts[:-1] + [new_file_pattern])

                new_video.append({'image': new_fpath})
        new_video.append({'image': fpath})

    return new_video


def load_pose_and_depth(frame_list, device='cpu'):
    assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
    pose_clip = []
    depth_clip = []

    for frame in frame_list:
        # gt image
        fpath = frame["image"]

        parts = fpath.split('/')
        folder_pattern = parts[-2]
        file_pattern = parts[-1]

        # depth
        pattern = re.compile(r'\d+_\d+_\d+_\d+_\d+_\d+_([A-Z]+(_[A-Z]+)?)+\.jpg')
        match = pattern.search(file_pattern)
        if match:
            filename = match.group(0) + '.npy'
            npy_file_path = os.path.join(os.path.dirname(__file__), '../../waymo_process/waymo_pseudoimg_multiframe_depth', folder_pattern, filename)
            if os.path.exists(npy_file_path):
                depth_data = np.load(npy_file_path)
                depth_data = torch.from_numpy(depth_data).unsqueeze(0).to(device)


        # pose and intrinsic
        pose_file = file_pattern[:-4] + '.npy'
        pose_file_path = os.path.join(os.path.dirname(__file__), '../../waymo_process/waymo_gtpose', folder_pattern, pose_file)
        if os.path.exists(pose_file_path):
            pose_data = np.load(pose_file_path, allow_pickle=True).item()
            pose_matrix = pose_data['pose']  # camera-to-world pose
            pose_matrix = torch.from_numpy(pose_matrix).to(device)

        #load all
        # pose
        pose_clip.append(pose_matrix)
        #depth            
        depth_clip.append(depth_data)


    return pose_clip, depth_clip

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = args.model_path
    img_pickle = args.img_pickle
    output_dir = args.output_dir

    # load eval dataset
    img_pickle = args.img_pickle
    with open(img_pickle, 'rb') as f:
        eval_img_data = pickle.load(f)

    pipeline = StableVideoDiffusionPipeline_vggt.from_pretrained(
        model_path, torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()
    vggt_model = VGGT()
    vggt_state_dict = torch.load("./modules/vggt/checkpoints/model.pt", map_location=device)
    vggt_model.load_state_dict(vggt_state_dict, strict=False)
    vggt_model.eval()
    vggt_model = vggt_model.to(device)

    # scenes = sorted(list(eval_img_data.keys()))
    scenes = ['11566385337103696871_5740_000_5760_000_FRONT'] #demo case

    video_length= args.video_length // 3
    prev_video_length = args.prev_nframes

    savepath= os.path.join(output_dir, "eval")
    os.makedirs(savepath,exist_ok=True)

    for scene in tqdm(scenes, desc="evaluating..."):
        eval_data = eval_img_data[scene]
        allframes=[]

        scene_path = os.path.join(savepath,scene)
        os.makedirs(scene_path,exist_ok=True)

        for index in range(prev_video_length, math.floor(len(eval_data)/video_length)): #5hz  prev_video_length, math.floor(len(eval_data)/video_length)
            cur_videos = eval_data[index*video_length:index*video_length+video_length]
            extra_videos = eval_data[index*video_length-prev_video_length:index*video_length]

            cur_videos = add_camera_view(cur_videos)
            extra_videos = add_camera_view(extra_videos)

            # cur_pose, _ = load_pose_and_depth(cur_videos, device=device)
            # prev_pose, prev_depth = load_pose_and_depth(prev_videos, device=device)
            # extra_pose, extra_depth = load_pose_and_depth(extra_videos, device=device)

            extraimage_names = [temp['image'] for temp in extra_videos]
            origin_image_pil_list = []
            for i in range(len(extraimage_names)):
                origin_image_pil_list.append(load_image(extraimage_names[-3:][i % 3]))

            extra_images_list = []
            to_tensor = transforms.ToTensor()
            for i in range(len(extraimage_names)):
                extra_image = to_tensor(load_image(extraimage_names[i])).to(device)
                extra_images_list.append(extra_image)

            cur_images_list = []
            curimage_names = [temp['image'] for temp in cur_videos]
            for i in range(len(curimage_names)):
                cur_image = to_tensor(load_image(curimage_names[i])).to(device)
                cur_images_list.append(cur_image)

            reference_path = os.path.join(output_dir, 'reference')
            os.makedirs(reference_path, exist_ok=True)

            first_frame_image = vggt_forward(
                vggt_model, 
                pose_encoding_to_extri_intri,
                unproject_depth_map_to_point_map,
                cur_images_list,
                extra_images_list, 
                vis_dir=reference_path, 
                device=device
            )
            generator = torch.manual_seed(42)
            frames = pipeline(first_frame_image, origin_image=origin_image_pil_list, width=origin_image_pil_list[0].size[0], height=origin_image_pil_list[0].size[1],num_frames=video_length*3, num_inference_steps=25,min_guidance_scale=2.0,max_guidance_scale=2.0, noise_aug_strength=0.02, generator=generator)
            frames = frames.frames[0]

            allframes = allframes+frames

        # Define video writer
        fps = 5        
        filename = os.path.join(scene_path,"pred0.mp4")
        with imageio.get_writer(filename,fps=fps,codec='libx264', bitrate='5000k', quality=10) as video:
            for idx in range(len(allframes)):
                if idx % 3 == 0:
                    frame = np.array(allframes[idx].convert('RGB'))
                    video.append_data(frame)
        filename = os.path.join(scene_path,"pred1.mp4")
        with imageio.get_writer(filename,fps=fps,codec='libx264', bitrate='5000k', quality=10) as video:
            for idx in range(len(allframes)):
                if idx % 3 == 1:
                    frame = np.array(allframes[idx].convert('RGB'))
                    video.append_data(frame)
        filename = os.path.join(scene_path,"pred2.mp4")
        with imageio.get_writer(filename,fps=fps,codec='libx264', bitrate='5000k', quality=10) as video:
            for idx in range(len(allframes)):
                if idx % 3 == 2:
                    frame = np.array(allframes[idx].convert('RGB'))
                    video.append_data(frame)

        pseudo0_clips = []
        pseudo1_clips = []
        pseudo2_clips = []

        sub_folders = [f for f in os.listdir(reference_path) if os.path.isdir(os.path.join(reference_path, f))]
        sub_folders.sort()
        
        for sub_folder in sub_folders:
            pseudo_folder = os.path.join(reference_path, sub_folder, 'pseudo')
            if os.path.exists(pseudo_folder):
                for i in range(3):
                    video_path = os.path.join(pseudo_folder, f'pseudo{i}.mp4')
                    if os.path.exists(video_path):
                        clip = VideoFileClip(video_path)
                        if i == 0:
                            pseudo0_clips.append(clip)
                        elif i == 1:
                            pseudo1_clips.append(clip)
                        elif i == 2:
                            pseudo2_clips.append(clip)
        
        if pseudo0_clips:
            final_clip0 = concatenate_videoclips(pseudo0_clips)
            final_clip0.write_videofile(os.path.join(savepath, 'pseudo0.mp4'))
        
        if pseudo1_clips:
            final_clip1 = concatenate_videoclips(pseudo1_clips)
            final_clip1.write_videofile(os.path.join(savepath, 'pseudo1.mp4'))
        
        if pseudo2_clips:
            final_clip2 = concatenate_videoclips(pseudo2_clips)
            final_clip2.write_videofile(os.path.join(savepath, 'pseudo2.mp4'))