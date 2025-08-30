export NCCL_P2P_LEVEL=NVL
export MODEL_NAME="./pretrained/stable-video-diffusion-img2vid-xt" # SVD official
# export DATASET_NAME='waymo_process/waymo_multiframe_subsegbycampos.pkl'
export DATASET_NAME="/high_perf_store/l3_deep/xiaoziyu/FreeVS/diffusers/waymo_process/waymo_only_front_gt.pkl" # 'waymo_process/waymo_multiframe_subsegbycampos_transform_simulation.pkl'
export HF_HOME="./huggingface"
WORK=freevs_waymo_halfreso_multiframe_transformation_simulate_trainunet
EXP=freevs_waymo_halfreso_multiframe_transformation_simulate_trainunet
accelerate launch --config_file fsdp.yaml --main_process_port 10012 examples/freevs/train_vggt_AR.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --enable_xformers_memory_efficient_attention \
  --max_train_steps=15000 \
  --learning_rate=5e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --dataloader_num_workers 6 \
  --nframes 6 \
  --conditioning_dropout_prob=0.2 \
  --mismatch_aug_ratio=0.5 \
  --seed_for_gen=42 \
  --ddim \
  --checkpointing_steps 2500 \
  --tracker_project_name $EXP \
  --load_from_pkl \
  --checkpoints_total_limit 3\
  --layout_encoder_path pretrained/ \
  --output_dir="/high_perf_store/l3_deep/xiaoziyu/FreeVS/diffusers/outputs/debug" \
  --prev_nframes=4 \
  # --vis_vggt \
  # --resume_from_checkpoint 'outputs/vggt_src_pose/checkpoint-15000' \
  # --initialize_pseudoimg_encoder \
  #   --max_train_steps=15000 \ 
  #   --learning_rate=5e-5 \