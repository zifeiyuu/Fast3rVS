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

conversion_matrix = np.array([[0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]])
def save_images_from_waymo(segment_path, output_path,interval=2):

    count=1
    
    dataset = tf.data.TFRecordDataset(segment_path, compression_type='')
    camera_poses={}
    for idx, data in enumerate(dataset):

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        segment_name = frame.context.name

        if idx==0:
            segment_output_dir = os.path.join(output_path, segment_name)
            os.makedirs(segment_output_dir,exist_ok=True)
        
        if not idx%interval==0: #sample each *interval frames
            continue

        for i, image in enumerate(frame.images):
            camera_calibration = next(cc for cc in frame.context.camera_calibrations if cc.name == image.name)
            frame_num = idx
            camera_position = open_dataset.CameraName.Name.Name(image.name)

            extrinsic = camera_calibration.extrinsic.transform
            intrinsic = camera_calibration.intrinsic

            full_matrix=np.array(camera_calibration.extrinsic.transform).reshape(4,4)

            ego_pose = np.array(image.pose.transform).reshape(4,4)#frame.pose.transform
            # ego_pose =np.array(ego_pose).reshape(4,4)

            world_transform = ego_pose.dot(full_matrix)
            world_transform =  world_transform @ np.linalg.inv(conversion_matrix) 
            world_to_camera_transform = np.linalg.inv(world_transform)

            fx = intrinsic[0]
            fy = intrinsic[1]
            cx = intrinsic[2]
            cy = intrinsic[3]
            skew = intrinsic[4]

            image_data = tf.image.decode_jpeg(image.image).numpy()
            H_orig, W_orig = image_data.shape[0],image_data.shape[1]

            intrinsic_matrix = np.array([
                                    [fx, skew, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]
                                ])
            metadata = {
                "intrinsic": intrinsic_matrix,
                "pose": world_to_camera_transform,
                "h_orig": H_orig,
                "w_orig": W_orig
            }

            pose_filename = f"{str(count).zfill(4)}_{segment_name}_{frame_num}_{camera_position}.npy"
            pose_filepath = os.path.join(segment_output_dir, pose_filename)

            np.save(pose_filepath, metadata)

            count+=1

    return segment_name

import argparse
parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--waymo_raw_dir', type=str,default='/high_perf_store/l3_deep/open-datasets/waymo/v1.4.3/ori/training')
parser.add_argument('--output_dir', type=str,default='waymo_gtpose_5hz_allseg')
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
                
    # pool = multiprocessing.Pool(1)

    # for tfrecord_path in tqdm(scenes):
    #     file_path = os.path.join(folder_path, tfrecord_path)
    #     pool.apply_async(save_images_from_waymo, args=(file_path, output_dir,args.interval))

    # pool.close()
    # pool.join()

    for tfrecord_path in tqdm(scenes):
        file_path = os.path.join(folder_path, tfrecord_path)
        save_images_from_waymo(file_path, output_dir,args.interval)