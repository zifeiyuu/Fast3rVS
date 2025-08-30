import sys
import torch
import os
from src.pipelines.pipeline_stable_video_diffusion_custom import StableVideoDiffusionPipeline_convnb_multiframe
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

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--model_path', type=str,default='work_dirs/freevs_waymo_halfreso_multiframe')
parser.add_argument('--img_pickle', type=str,default = 'waymo_process/waymo_multiframe_subsegbycampos.pkl')
parser.add_argument('--output_dir', type=str, default='rendered_waymo_origin') 
parser.add_argument('--video_length', type=int, default=6) # batch frame num
parser.add_argument("--front_only",action="store_true",default=False,help="whether to randomly flip images horizontally",)
args = parser.parse_args()

if __name__ == "__main__":

    model_path = args.model_path
    img_pickle = args.img_pickle

    pipe = StableVideoDiffusionPipeline_convnb_multiframe.from_pretrained(
        model_path, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    with open(img_pickle, 'rb') as f:
        img_data = pickle.load(f)

    scenes = sorted(list(img_data.keys()))

    # scenes = ['12505030131868863688_1740_000_1760_000_FRONT'] #demo case

    video_length=args.video_length

    savepath=args.output_dir
    os.makedirs(savepath,exist_ok=True)

    epoch = 1
    for e in range(epoch):
        print(f"epoch {e}")
        for scene in tqdm(scenes, desc="evaluating..."):
            if True:#scene[-9:]=='CAM_FRONT':

                if args.front_only and not scene[-5:]=='FRONT':
                    continue

                imgs = img_data[scene]
                allframes=[]
                allpseudo_frames = []
                propagate_latents=None
                prev_latents=None
                count=0

                os.makedirs(os.path.join(savepath,scene),exist_ok=True)

                for rand_int in range(0, math.floor(len(imgs)/video_length)): #5hz  math.floor(len(imgs)/video_length)

                    if rand_int<math.floor(len(imgs)/video_length):
                        images = [temp['pseudo_image'] for temp in imgs[rand_int*video_length:rand_int*video_length+video_length]]
                        gtimage_names = [temp['image'] for temp in imgs[rand_int*video_length:rand_int*video_length+video_length]]
                    else:
                        images = [temp['pseudo_image'] for temp in imgs[len(imgs)-video_length:len(imgs)]]
                        gtimage_names =  [temp['image'] for temp in imgs[len(imgs)-video_length:len(imgs)]]

                    # demo case
                    # images = [temp.replace('waymo_pseudoimg_multiframe','waymo_pseudoimg_multiframe_5hz_democases') for temp in images]
                    
                    if not sum([os.path.exists(image) for image in images]) == len(images):
                        break

                    pseudo_images=[]
                    for image in images:
                        image = load_image(image)
                        image = image.resize((int(960),int(384)))
                        pseudo_images.append(image)
                    
                    origin_image = imgs[rand_int*(video_length)]['image']
                    origin_image = load_image(origin_image)
                    set_trace()

                    generator = torch.manual_seed(42)


                    frames = pipe(pseudo_images,origin_image=origin_image, width=origin_image.size[0], height=origin_image.size[1],num_frames=video_length, num_inference_steps=25,min_guidance_scale=2.0,max_guidance_scale=2.0, noise_aug_strength=0.02, generator=generator)
                    frames = frames.frames[0]

                    allframes = allframes+frames
                    allpseudo_frames = allpseudo_frames + pseudo_images

                    pseudo_folder = os.path.join(savepath, scene, "pseudo_image")
                    os.makedirs(pseudo_folder, exist_ok=True)
                    for _i,p_frame in enumerate(pseudo_images):
                        export_path=os.path.join(pseudo_folder, os.path.split(gtimage_names[_i])[-1])
                        p_frame.save(export_path)

                    predicted_folder = os.path.join(savepath, scene, "predicted")
                    os.makedirs(predicted_folder, exist_ok=True)
                    for _i,frame in enumerate(frames):
                        export_path=os.path.join(predicted_folder, os.path.split(gtimage_names[_i])[-1])
                        frame.save(export_path)
                        count+=1

                filename = os.path.join(savepath,"short",scene)+'.mp4'
                fps = 10
                
                import cv2

                # Define video writer
                frame = np.array(allframes[0].convert('RGB'))
                p_frame = np.array(allpseudo_frames[0])
                spliced =np.concatenate((frame, p_frame), axis=1)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, fps, (spliced.shape[1], spliced.shape[0]))

                for idx in range(len(allframes)):
                    frame = np.array(allframes[idx])
                    p_frame = np.array(allpseudo_frames[idx].convert('RGB'))
                    spliced = np.concatenate((frame, p_frame), axis=1)
                    out.write(spliced) 

                out.release()

                # with imageio.get_writer(filename,fps=fps,codec='libx264', bitrate='5000k', quality=10) as video:
                #     for idx in range(len(allframes)):
                #         frame = np.array(allframes[idx].convert('RGB'))
                #         p_frame = np.array(allpseudo_frames[idx])
                #         spliced =np.concatenate((frame, p_frame), axis=1)
                #         video.append_data(spliced)

