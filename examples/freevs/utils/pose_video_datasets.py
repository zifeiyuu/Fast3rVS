import os, re
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms as tr
import random, json
import pickle as pkl
import tqdm
from collections import defaultdict
from utils.image_datasets import combine_text_conds, get_text_cond_tokens, TEXT_CONDITIONS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .image_datasets import default_loader
from ipdb import set_trace
import re

class Pose_VideoDataset(data.Dataset):
    def __init__(self,
        data_root,
        video_length,
        prev_video_length,
        video_transforms=None,
        text_tokenizer=None,
        init_caption='',
        multi_view=False,
        ego = False,
        mismatch_aug_ratio = None,
        AR_mode = False,
        **kargs
        ):

        super().__init__()
        self.mismatch_aug_ratio=mismatch_aug_ratio
        self.loader = default_loader
        self.init_caption = init_caption 
        self.multi_view = multi_view
        self.video_length = video_length // 3
        self.img_transform = video_transforms
        self.prev_video_length = prev_video_length
        # self.camera_view = ['CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT']
        # TODO: hardcode here
        self.camera_view = ['CAM_FRONT']
        self.camera_captions = {'cam01': 'front camera view',
                                'CAM_BACK': 'back camera view',
                                'CAM_FRONT': 'front camera view',
                                'CAM_FRONT_RIGHT':'front right camera view',
                                'CAM_FRONT_LEFT':'front left camera view',
                                'CAM_BACK_RIGHT':'back right camera view',
                                'CAM_BACK_LEFT':'back left camera view'
                                }
        self.nearbycamera = {
                                'FRONT': [['FRONT_LEFT', 'FRONT_RIGHT'], [1, 3]],
                                'FRONT_RIGHT':[['FRONT', 'SIDE_RIGHT'], [-3, 1]], 
                                'FRONT_LEFT':[['FRONT', 'SIDE_LEFT'], [-1, 1]],
                                'SIDE_RIGHT':[['FRONT_RIGHT'], [-1]],
                                'SIDE_LEFT':[['FRONT_LEFT'], [-1]]
                            }
        self.ego = ego

        if self.ego:
            self.videos, self.conds = self._make_dataset_video_ego(data_root, video_length)
        else:
            self.videos, self.conds = self._make_dataset_video(data_root, video_length)

        self.default_transforms = tr.Compose(
            [
                tr.ToTensor(),
            ]
        )
    
        self.intrin_aug_range=[0.5, 1]
        self.AR_mode = AR_mode


    def add_camera_view(self, video):
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
            new_video.append({'image': fpath})
            if camera_view in self.nearbycamera:
                replacements, offsets = self.nearbycamera[camera_view]
                for replacement, offset in zip(replacements, offsets):
                    new_image_number = image_number + offset
                    new_image_number_str = f"{new_image_number:04d}" 

                    new_file_parts = file_parts.copy()
                    new_file_parts[0] = new_image_number_str 
                    new_file_parts[-1] = replacement + '.jpg'
                    new_file_pattern = '_'.join(new_file_parts)
                    new_fpath = '/'.join(parts[:-1] + [new_file_pattern])

                    new_video.append({'image': new_fpath})

        return new_video

    def __getitem__(self, index):
        extra_frame_num = self.prev_video_length
        video, conds = self.videos[index], self.conds[index]
        init_frame = random.randint(extra_frame_num, len(video)-self.video_length*3 - extra_frame_num)
        extra_video = video[init_frame - extra_frame_num:init_frame]    
        if self.AR_mode:  
            video = video[init_frame:init_frame+self.video_length*3] # video begin with a random frame
            conds = conds[init_frame:init_frame+self.video_length*3]            
        else:
            video = video[init_frame:init_frame+self.video_length] # video begin with a random frame
            conds = conds[init_frame:init_frame+self.video_length]

        extra_video = self.add_camera_view(extra_video)
        video = self.add_camera_view(video)

        # make clip tensor
        if self.multi_view:
            print('not support yet')
        else:
            frames, segs, poses, intrinsics, depths, _ = self.load_and_transform_frames(video, self.loader, self.img_transform)

            frames = torch.cat(frames, 1) # c,t,h,w
            frames = frames.transpose(0, 1) # t,c,h,w
            
            example = dict()
            example["pixel_values"] = frames
            example["images"], _, _, _, _, _ = self.load_and_transform_frames(video, self.loader)
            if self.ego:
                frames_ego = torch.tensor(conds)
                example["ego_values"] = frames_ego / (0.5)

            example["pose"] = poses
            example["intrinsic"] = intrinsics
            example["depth"] = depths

            example["extra_images"], example["extra_segs"], example["extra_pose"], example["extra_intrinsic"], example["extra_depth"], example["scene"] = self.load_and_transform_frames(extra_video, self.loader)

        return example

    def __len__(self):
        return len(self.videos) 

    def _make_dataset_video(self, info_path, nframes):
        set_scene = ['16336545122307923741_486_637_506_637'] # 11566385337103696871_5740_000_5760_000    16336545122307923741_486_637_506_637
        finished = []
        pattern = re.compile(r'segment-(\d+_\d+_\d+_\d+_\d+)_with_camera_labels\.tfrecord')
        with open('/high_perf_store/l3_deep/xiaoziyu/FreeVS/diffusers/waymo_process/waymo_pseudoimg_multiframe_depth/finished.txt', 'r') as file:
            for line in file:
                match = pattern.search(line.strip())
                if match:
                    extracted_part = match.group(1)
                    finished.append(extracted_part)  

        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        output_videos = []
        output_text_videos = []
        for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
            pattern = re.compile(r'(\d+_\d+_\d+_\d+_\d+)_[A-Z_]+')
            match = pattern.search(video_name)
            if match:
                extracted_part = match.group(1)
                # if extracted_part not in set_scene:
                #     continue
                if extracted_part not in finished:
                    continue
            else:
                continue

            for cam_v in self.camera_view:
                view_video = []
                text_video = []

                for frame in frames:
                    view_video.append(frame)
                    text_video.append(self.camera_captions[cam_v])
                
                output_videos.append(view_video)
                output_text_videos.append(text_video)

        return output_videos, output_text_videos
    
    def _make_dataset(self, info_path, nframes):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        output_videos = []
        output_text_videos = []
        if not self.multi_view:
            for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
                for cam_v in self.camera_view:
                    view_video = []
                    text_video = []
                    for frame in frames:
                        view_video.append(frame[cam_v])
                        text_video.append(self.camera_captions[cam_v])
                    
                    output_videos.append(view_video)
                    output_text_videos.append(text_video)
        else:
            for video_name, frames in video_info.items():
                output_videos.append(frames)

        return output_videos, output_text_videos

    def load_and_transform_frames(self, frame_list, loader, img_transform=None):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []
        pose_clip = []
        intrinsic_clip = []
        depth_clip = []
        bbox_clip = []
        seg_clip = []

        use_mismatch_aug=False
        if np.random.rand()<self.mismatch_aug_ratio:
            use_mismatch_aug=True
            minusaug=False
            if np.random.rand()<0.5:
                minusaug=True

        use_intrin_aug=False
        intrin_aug_scale = np.random.rand()*(self.intrin_aug_range[1]-self.intrin_aug_range[0])+self.intrin_aug_range[0]

        for frame in frame_list:
            # gt image
            fpath = frame["image"]
            #960,384
            img = loader(fpath)
            img_w, img_h = img.size[:2]

            if img_transform is not None:
                img = img_transform(img)
            else:
                img = self.default_transforms(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))

            can_load = True

            parts = fpath.split('/')
            folder_pattern = parts[-2]
            file_pattern = parts[-1]

            # depth
            pattern = re.compile(r'\d+_\d+_\d+_\d+_\d+_\d+_([A-Z]+(_[A-Z]+)?)+\.jpg')
            match = pattern.search(file_pattern)
            if match:
                filename = match.group(0) + '.npy'
                npy_file_path = os.path.join(os.path.dirname(__file__), '../../../waymo_process/waymo_pseudoimg_multiframe_depth', folder_pattern, filename)
                if os.path.exists(npy_file_path):
                    depth_data = np.load(npy_file_path)
                else:
                    can_load = False
            else:
                can_load = False

            # pose and intrinsic
            pose_file = file_pattern[:-4] + '.npy'
            pose_file_path = os.path.join(os.path.dirname(__file__), '../../../waymo_process/pose_conversion', folder_pattern, pose_file)
            if os.path.exists(pose_file_path):
                pose_data = np.load(pose_file_path, allow_pickle=True).item()
                pose_matrix = pose_data['pose']  # world-to-camera pose
                # pose_matrix = np.linalg.inv(pose_data['pose'])  # camera-to-world pose
                intrinsic_matrix = pose_data['intrinsic']  # intrinsic 3x3 matrix
                H_orig = pose_data['h_orig']
                W_orig = pose_data['w_orig']

                # intrinsic_matrix[0, 1] = 0
                # intrinsic_matrix[1, 0] = 0 
                intrinsic_matrix[0, :] = intrinsic_matrix[0, :] / W_orig
                intrinsic_matrix[1, :] = intrinsic_matrix[1, :] / H_orig
            else:
                can_load = False

            # segmentation
            seg_file = file_pattern[:-4] + '.npy'
            seg_file_path = os.path.join(os.path.dirname(__file__), '../../../waymo_process/waymo_seg', folder_pattern, seg_file)
            if os.path.exists(seg_file_path):
                seg_data = np.load(seg_file_path, allow_pickle=True).item()
                speed_arr = seg_data['speeds']
                keep_indices = np.where(speed_arr >= 1)[0]

                seg_data['masks'] = [seg_data['masks'][i] for i in keep_indices]
                seg_data['ids'] = [seg_data['ids'][i] for i in keep_indices]
                seg_data['speeds'] = speed_arr[keep_indices].tolist()

                seg_mask = seg_data['masks']
                seg_id = seg_data['ids']
            else:
                can_load = False

            #load all
            if can_load:
                #image
                clip.append(img)
                # seg
                seg_clip.append(list(zip(seg_mask, seg_id)))            
                # pose
                pose_clip.append(pose_matrix)
                #intrinsic
                intrinsic_clip.append(intrinsic_matrix)    
                #depth            
                depth_clip.append(depth_data)


        return clip, seg_clip, pose_clip, intrinsic_clip, depth_clip, folder_pattern




############################################################


    def _make_dataset_video_ego(self, info_path, nframes):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        output_videos = []
        output_ego_videos = []
        for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
            for cam_v in self.camera_view:
                view_video = []
                ego_video = []
                for frame_i in range(len(frames[cam_v])):
                    frame = frames[cam_v][frame_i]
                    frame_path = frame
                    view_video.append(frame_path)
                    ego_video.append(frames['ego'][frame_i])
                
                output_videos.append(view_video)
                output_ego_videos.append(ego_video)

        return output_videos, output_ego_videos