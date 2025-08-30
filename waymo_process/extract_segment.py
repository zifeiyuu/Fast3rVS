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
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import box_utils

warnings.filterwarnings("ignore")

def save_images_from_waymo(segment_path, output_path,interval=2):
    print(segment_path)

    count=1
    
    dataset = tf.data.TFRecordDataset(segment_path, compression_type='')

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
            frame_num = idx
            camera_position = open_dataset.CameraName.Name.Name(image.name)
            image_data = tf.image.decode_jpeg(image.image).numpy()
            H_orig, W_orig = image_data.shape[0],image_data.shape[1]

            all_labels = []
            
            # 创建动态车辆ID集合
            moving_vehicle_ids = set()
            for laser_label in frame.laser_labels:
                if laser_label.type != 1:
                    continue
                
                # 计算速度大小
                speed = np.sqrt(
                    laser_label.metadata.speed_x**2 +
                    laser_label.metadata.speed_y**2 +
                    laser_label.metadata.speed_z**2
                )
                if speed > 0.05:
                    # 将3D标注投影到2D，匹配Panoptic实例
                    vertices = box_utils.get_3d_box_vertices(laser_label.box)
                    projected = box_utils.project_3d_to_2d(vertices, 
                        next(c for c in frame.context.camera_calibrations if c.name == image.name))
                    # 计算与Panoptic实例的重叠率
                    for orig_id in original_ids:
                        iou = calculate_iou(projected, (original_instance_id == orig_id))
                        if iou > 0.5:  # 阈值可调
                            moving_vehicle_ids.add(orig_id)
                            break

                
            segment_filename = f"{str(count).zfill(4)}_{segment_name}_{frame_num}_{camera_position}.npy"
            segment_filepath = os.path.join(segment_output_dir, segment_filename)
            np.save(segment_filepath, all_labels)

            count+=1

    return segment_name

import argparse
parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--waymo_raw_dir', type=str,default='/high_perf_store/l3_deep/open-datasets/waymo/v1.4.3/ori/training')
parser.add_argument('--output_dir', type=str,default='waymo_gtsegment')
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

