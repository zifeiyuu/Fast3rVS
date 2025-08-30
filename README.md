## Fast3rVS

> Novel view synthesis for street-scene videos with free camera trajectories, leveraging LiDAR (or synthesized) point clouds and diffusion models.

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