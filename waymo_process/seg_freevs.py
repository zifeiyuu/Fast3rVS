import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time
from pdb import set_trace
import re

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_id(box, box_id, ax, color='green', fontsize=12):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    ax.text(
        center_x, y1 - 5,
        str(box_id),
        color=color,
        fontsize=fontsize,
        ha='center',
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )
def show_speed(box, speed, ax, color='red', fontsize=12):
    x1, y1, x2, y2 = box
    ax.text(
        x2, y1,
        f"Speed: {speed} m/s",
        color=color,
        fontsize=fontsize,
        ha='right',
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

bbox_folder = '/high_perf_store/l3_deep/xiaoziyu/FreeVS/diffusers/waymo_process/waymo_moving_bbox'
image_folder = '/high_perf_store/l3_deep/xiaoziyu/FreeVS/diffusers/waymo_process/gt'
output_folder = '/high_perf_store/l3_deep/xiaoziyu/FreeVS/diffusers/waymo_process/debug_moving'
os.makedirs(output_folder, exist_ok=True) 

finished_roots_file = os.path.join(output_folder, "finished_roots.txt")
if not os.path.exists(finished_roots_file):
    with open(finished_roots_file, 'w') as f:
        pass
with open(finished_roots_file, 'r') as f:
    finished_roots = set(line.strip() for line in f.readlines())

for root, _, files in os.walk(image_folder):
    if root == image_folder:
        continue
    folder_name = os.path.basename(root)
    if folder_name in finished_roots:
        continue
    for file in files:
        if file.lower().endswith('.jpg'):
            fpath = os.path.join(root, file)
            # if fpath != "/high_perf_store/l3_deep/xiaoziyu/FreeVS/diffusers/waymo_process/gt/10596949720463106554_1933_530_1953_530/0440_10596949720463106554_1933_530_1953_530_174_SIDE_RIGHT.jpg":
            #     continue

            parts = fpath.split('/')
            folder_pattern = parts[-2]
            file_pattern = parts[-1]

            image = Image.open(fpath)
            image = np.array(image.convert("RGB"))
            img_h, img_w, c = image.shape

            bbox_file = file_pattern[:-4] + '.npy'
            bbox_file_path = os.path.join(bbox_folder, folder_pattern, bbox_file)
            bbox_list = []
            bbox_id_list = []
            bbox_speed_list = []
            bbox_data = np.load(bbox_file_path, allow_pickle=True)
            for bbox in bbox_data:
                box_id = bbox["id"]
                box_speed = bbox["speed"]
                min_x = bbox["box"][0]
                min_y = bbox["box"][1]
                max_x = bbox["box"][2]
                max_y = bbox["box"][3]
                
                h_orig = bbox["h_orig"]
                w_orig = bbox["w_orig"]

                scale = img_w / w_orig  # 因为原图被 resize(w/2, h/2)
                min_x = min_x * scale
                min_y = min_y * scale
                max_x = max_x * scale
                max_y = max_y * scale

                crop_offset = 256

                min_y = min_y - crop_offset
                max_y = max_y - crop_offset

                center_x = min_x + (max_x - min_x) / 2
                center_y = min_y + (max_y - min_y) / 2
                width = max_y - min_y
                height = max_x - min_x

                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2

                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))

                box_width = x2 - x1
                box_height = y2 - y1

                min_area = 1

                if box_width <= 0 or box_height <= 0 or (box_width * box_height) < min_area:
                    continue

                bbox_list.append([x1, y1, x2, y2])
                bbox_id_list.append(box_id)
                bbox_speed_list.append(box_speed)

            clean_ids = [re.sub(r'_(FRONT|SIDE)(?:_(LEFT|RIGHT))?$', '', id_) for id_ in bbox_id_list]
            predictor.set_image(image)

            input_boxes = np.array(bbox_list)
            output_dir = output_folder
            output_dir = os.path.join(output_dir, folder_pattern)
            os.makedirs(output_dir, exist_ok=True) 
            masks = []           
            if input_boxes.size > 0:
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

            output_path = os.path.join(output_dir, file_pattern[:-4] + '.png')
            plt.figure(figsize=(10, 10))
            plt.imshow(image)

            for box, box_id, mask, speed in zip(input_boxes, clean_ids, masks, bbox_speed_list):
                if speed > 0.1:
                    show_box(box, plt.gca())
                    show_id(box, box_id, plt.gca())  
                    show_speed(box, str(speed), plt.gca())  
                    show_mask(mask[0] if mask.shape[0] == 1 else mask, plt.gca(), random_color=True)
            plt.axis('off')

            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close() 


            # mask_data = {
            #     'masks': [],
            #     'ids': []
            # }
            # for mask, box_id in zip(masks, clean_ids):
            #     mask = mask[0] if mask.shape[0] == 1 else mask
            #     mask_data['masks'].append(mask)
            #     mask_data['ids'].append(box_id)
            
            # mask_data['masks'] = np.array(mask_data['masks'])
            # mask_data['ids'] = np.array(mask_data['ids'])
            # mask_output_path = os.path.join(output_dir, file_pattern[:-4] + '.npy')
            # np.save(mask_output_path, mask_data)

    with open(finished_roots_file, 'a') as f:
        f.write(f"{folder_name}\n")