import os
import shutil
import cv2
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import json
import pickle as pkl
from ipdb import set_trace
import numpy as np

def process_and_save_image(img_path, output_path):
    img = cv2.imread(img_path)

    resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

    # clip sky
    if 'FRONT' in img_path.split('/')[-1]:
        resized_img=resized_img[256:]
    else:
        resized_img=resized_img[59:]

    # cv2.imwrite(output_path, resized_img)

    H, W = resized_img.shape[0], resized_img.shape[1]

    return H, W

import argparse
parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--data_root', type=str,default='waymo_process/waymo_gtimg_5hz_allseg')
parser.add_argument('--pose_root', type=str,default='waymo_process/waymo_gtpose_5hz_allseg')
parser.add_argument('--output_dir', type=str, default='waymo_process/gt') 
parser.add_argument('--output_pickle', type=str, default='waymo_process/waymo_pose_and_gt.pkl') 
args = parser.parse_args()

if __name__ == "__main__":

    data_root = args.data_root
    pose_root = args.pose_root
    output_dir = args.output_dir
    output_pickle = args.output_pickle

    os.makedirs(output_dir, exist_ok=True)

    scenes = sorted(os.listdir(data_root))

    scenes = [temp.split('segment-')[-1].split('_with_camera')[0] for temp in scenes]

    infos = defaultdict(list)

    for scene in tqdm(scenes):
        scene_path = os.path.join(data_root, scene, '')
        out_scene_path = os.path.join(output_dir, scene)
        pose_path = os.path.join(pose_root, scene, '')

        imgs_path = scene_path

        os.makedirs(out_scene_path, exist_ok=True)

        frames = sorted(os.listdir(imgs_path))
        poses = sorted(os.listdir(pose_path))

        for cam_pos in ['FRONT','FRONT_LEFT','SIDE_LEFT','FRONT_RIGHT','SIDE_RIGHT']:
            for frame, pose in zip(frames, poses):
                if (not cam_pos+'.jpg' in frame) or not (cam_pos+'.npy' in pose):
                    continue
                frame_path = os.path.join(imgs_path, frame)
                out_frame_path = os.path.join(out_scene_path, frame)


                H, W = process_and_save_image(frame_path, out_frame_path)

                single_pose_path = os.path.join(pose_path, pose)
                pose_data = np.load(single_pose_path, allow_pickle=True).item()
                pose_matrix = pose_data['pose']  # world-to-camera pose
                intrinsic_matrix = pose_data['intrinsic']  # intrinsic 3x3 matrix
                H_orig = pose_data['h_orig']
                W_orig = pose_data['w_orig']

                intrinsic_matrix[0, :] = intrinsic_matrix[0, :] * W / W_orig
                intrinsic_matrix[1, :] = intrinsic_matrix[1, :] * H / H_orig

                frame_info = dict()
                frame_info['image'] = out_frame_path  
                frame_info['pose'] = pose_matrix
                frame_info['intrinsic'] = intrinsic_matrix

                infos[scene+'_'+cam_pos].append(frame_info)


    with open(output_pickle, 'wb') as fw:
        pkl.dump(infos, fw)
