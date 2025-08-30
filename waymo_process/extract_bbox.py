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
    speed_threshold = 0.1   # 0.3 m/s ≈ 1 km/h
    count=1
    
    dataset = tf.data.TFRecordDataset(segment_path, compression_type='')
    vehicle_speeds = {}
    for idx, data in enumerate(dataset):

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        if not idx%interval==0: #sample each *interval frames
            continue

        for laser_label in frame.laser_labels:
            if laser_label.type == 1:  # TYPE_VEHICLE
                speed = np.sqrt(
                    laser_label.metadata.speed_x**2 +
                    laser_label.metadata.speed_y**2 +
                    laser_label.metadata.speed_z**2
                )
                # if speed > speed_threshold:
                #     if (laser_label.id not in vehicle_speed) or (speed > vehicle_speed[laser_label.id]):
                #         vehicle_speed[laser_label.id] = speed
                if laser_label.id not in vehicle_speeds:
                    vehicle_speeds[laser_label.id] = {}
                vehicle_speeds[laser_label.id][idx] = speed

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
            all_boxes = []            
            frame_num = idx
            camera_position = open_dataset.CameraName.Name.Name(image.name)
            image_data = tf.image.decode_jpeg(image.image).numpy()
            H_orig, W_orig = image_data.shape[0],image_data.shape[1]

            for camera_labels in frame.camera_labels:
                # Ignore camera labels that do not correspond to this camera.
                if camera_labels.name != image.name:
                    continue

                # Iterate over the individual labels.
                for label in camera_labels.labels:
                    if label.type != 1: ##VEHICLE
                        continue
                    box_2d = [
                        label.box.center_x - label.box.width / 2,
                        label.box.center_y - label.box.length / 2,
                        label.box.center_x + label.box.width / 2,
                        label.box.center_y + label.box.length / 2
                    ]
                    for laser_2d_label in frame.projected_lidar_labels:
                        if laser_2d_label.name != image.name:
                            continue
                        for label_2d in laser_2d_label.labels:
                            if label_2d.type != 1:
                                continue
                            box_3d_proj = [
                                label_2d.box.center_x - label_2d.box.width / 2,
                                label_2d.box.center_y - label_2d.box.length / 2,
                                label_2d.box.center_x + label_2d.box.width / 2,
                                label_2d.box.center_y + label_2d.box.length / 2
                            ]
                            iou = calculate_iou(box_2d, box_3d_proj)
                            if iou > 0.5:
                                true_id = clean_waymo_id(label_2d.id)
                                if true_id in vehicle_speeds and idx in vehicle_speeds[true_id]:
                                    all_boxes.append({
                                        "center_x": label.box.center_x,
                                        "center_y": label.box.center_y,
                                        "w": label.box.width,
                                        "h": label.box.length,
                                        "id": true_id,
                                        "h_orig": H_orig,
                                        "w_orig": W_orig,
                                        "speed": vehicle_speeds[true_id][idx]
                                    })
                                    break

            bbox_filename = f"{str(count).zfill(4)}_{segment_name}_{frame_num}_{camera_position}.npy"
            bbox_filepath = os.path.join(segment_output_dir, bbox_filename)
            np.save(bbox_filepath, all_boxes)

            count+=1

    return segment_name

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    return inter_area / (area1 + area2 - inter_area)

def clean_waymo_id(raw_id):
    camera_suffixes = [
        '_FRONT', '_FRONT_LEFT', '_FRONT_RIGHT',
        '_SIDE_LEFT', '_SIDE_RIGHT', '_BACK'
    ]
    
    for suffix in camera_suffixes:
        if raw_id.endswith(suffix):
            return raw_id[:-len(suffix)]
    
    return raw_id

import argparse
parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--waymo_raw_dir', type=str,default='/high_perf_store/l3_deep/open-datasets/waymo/v1.4.3/ori/training')
parser.add_argument('--output_dir', type=str,default='waymo_gtbbox')
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

    # for tfrecord_path in tqdm(scenes):
    #     file_path = os.path.join(folder_path, tfrecord_path)
    #     save_images_from_waymo(file_path, output_dir,args.interval)

