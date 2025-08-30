import os
import shutil
import cv2
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import json
import pickle as pkl
from ipdb import set_trace
import numpy as np


import argparse
parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--data_root', type=str,default='waymo_process/waymo_gtimg_5hz_allseg')
parser.add_argument('--output_dir', type=str, default='waymo_process/gt') 
parser.add_argument('--output_pickle', type=str, default='waymo_process/waymo_front_gt.pkl') 
args = parser.parse_args()

if __name__ == "__main__":

    data_root = args.data_root
    output_dir = args.output_dir
    output_pickle = args.output_pickle

    scenes = sorted(os.listdir(data_root))

    scenes = [temp.split('segment-')[-1].split('_with_camera')[0] for temp in scenes]

    infos = defaultdict(list)

    for scene in tqdm(scenes):
        scene_path = os.path.join(data_root, scene, '')
        out_scene_path = os.path.join(output_dir, scene)

        imgs_path = scene_path

        os.makedirs(out_scene_path, exist_ok=True)

        frames = sorted(os.listdir(imgs_path))

        for cam_pos in ['FRONT']:
            for frame in frames:
                if (not cam_pos+'.jpg' in frame):
                    continue
                out_frame_path = os.path.join(out_scene_path, frame)

                frame_info = dict()
                frame_info['image'] = out_frame_path  

                infos[scene+'_'+cam_pos].append(frame_info)


    with open(output_pickle, 'wb') as fw:
        pkl.dump(infos, fw)
