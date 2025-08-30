## Fast3rVS

Novel view synthesis for street-scene videos with free camera trajectories, leveraging LiDAR (or synthesized) point clouds and diffusion models.
<img width="1196" height="597" alt="image" src="https://github.com/user-attachments/assets/78704d52-9994-43b5-8732-664bdae433f2" />

<img width="803" height="475" alt="image" src="https://github.com/user-attachments/assets/e36d40c9-8c57-483d-a864-7295df887dfc" />
<img width="468" height="187" alt="image" src="https://github.com/user-attachments/assets/2db6b719-67b0-4e9e-ad99-c9b221d490fd" />
<img width="468" height="187" alt="image" src="https://github.com/user-attachments/assets/fdfeb521-4d9e-46db-bc35-577ae7a4f47c" />
<img width="1280" height="651" alt="image" src="https://github.com/user-attachments/assets/91ab5979-ef19-4d5c-9260-490181b4373b" />
<img width="468" height="187" alt="image" src="https://github.com/user-attachments/assets/80ff0fd7-c235-489f-ab9c-3c21d655ce99" />

<img width="468" height="187" alt="image" src="https://github.com/user-attachments/assets/485b36f9-f1b1-4757-85a3-b95a59e3111f" />




---

## Method Overview
1. **Inputs.** Multi-frame RGB sequences (+ optional GT LiDAR) and calibrated camera intrinsics/extrinsics.  
2. **Point-cloud proxy.** When GT LiDAR is missing, generate **pseudo point clouds** with Mast3r from selected frames.  
3. **Diffusion backbone.** FreeVS-style model operating on point-cloud/image features; supports a **temporal window** (adjacent + long-term memory).  
4. **Rendering.** Project predicted radiance/opacity to novel poses; compute multi-view photometric and geometry-aware losses.  
5. **Training.** RGB reconstruction loss; optional **binary voxel/occupancy** supervision to enforce 2D↔3D consistency.

---

## Datasets
- Street-scene video with poses: Waymo Open, KITTI-360.  

---

## Environment & Setup
```bash
conda create -n freevs python=3.8
conda activate freevs

cd diffusers
pip install .
pip install -r requirements.txt
```

---

## How to Run

### Prepare segmentation data
```bash
bash extract.sh 
```
### Training
```bash
bash train.sh 
```

### Inference
```bash
bash eval.sh
```
---

## License
MIT License © 2025 zifeiyuu
