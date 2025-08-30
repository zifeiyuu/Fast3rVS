import os
import sys
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from PIL import Image
from io import BytesIO
import numpy as np
from pyquaternion import Quaternion
import pickle as pkl
import warnings
from IPython import embed
from pdb import set_trace
from tqdm import tqdm

warnings.filterwarnings("ignore")

def save_images_from_waymo(segment_path, output_path,interval=2):
    
    dataset = tf.data.TFRecordDataset(segment_path, compression_type='')

    for idx, data in enumerate(dataset):

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        segment_name = frame.context.name

        if idx==0:
            segment_output_dir = os.path.join(output_path, segment_name + ".npy")
        
        if not idx%interval==0: #sample each *interval frames
            continue

        scene_poses = []
        for i, image in enumerate(frame.images):
            ego_pose = np.array(image.pose.transform).reshape(4,4) # frame.pose.transform
            scene_poses.append(ego_pose)

        np.save(segment_output_dir, scene_poses)

    return segment_name

import argparse
parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--waymo_raw_dir', type=str,default='/high_perf_store/l3_deep/open-datasets/waymo/v1.4.3/ori/training')
parser.add_argument('--output_dir', type=str,default='waymo_scene_pose')
parser.add_argument('--interval', type=int, default=2)
args = parser.parse_args()

import multiprocessing
if __name__ == "__main__":
    # ith = int(sys.argv[1])

    folder_path=args.waymo_raw_dir

    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)

    scenes=os.listdir(folder_path)
    
    scenes = [temp for temp in scenes if ('.tfrecord' in temp)]
                
    pool = multiprocessing.Pool(1)

    for tfrecord_path in tqdm(scenes):
        file_path = os.path.join(folder_path, tfrecord_path)
        pool.apply_async(save_images_from_waymo, args=(file_path, output_dir,args.interval))

    pool.close()
    pool.join()
