export NCCL_P2P_LEVEL=NVL
export MODEL_NAME="./pretrained/stable-video-diffusion-img2vid-xt" # SVD official
# export DATASET_NAME='waymo_process/waymo_multiframe_subsegbycampos.pkl'
export DATASET_NAME="/high_perf_store/l3_deep/lixuhuan/FreeVS/waymo_data/waymo_process/waymo_multiframe_subsegbycampos.pkl" # 'waymo_process/waymo_multiframe_subsegbycampos_transform_simulation.pkl'
export HF_HOME="./huggingface"
WORK=freevs_waymo_halfreso_multiframe_transformation_simulate_trainunet
EXP=freevs_waymo_halfreso_multiframe_transformation_simulate_trainunet
accelerate launch --config_file fsdp.yaml --main_process_port 10012 examples/freevs/train_svd_onlytrainunet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --enable_xformers_memory_efficient_attention \
  --max_train_steps=15000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --output_dir="/high_perf_store/l3_deep/xiaoziyu/FreeVS/diffusers/outputs" \
  --dataloader_num_workers 6 \
  --nframes 6 \
  --conditioning_dropout_prob=0.2 \
  --mismatch_aug_ratio=0.5 \
  --seed_for_gen=42 \
  --ddim \
  --checkpointing_steps 2500 \
  --tracker_project_name $EXP \
  --load_from_pkl \
  --checkpoints_total_limit 2\
  --layout_encoder_path pretrained/ #\
  # --initialize_pseudoimg_encoder #\
  # --resume_from_checkpoint "latest" 
    # --gradient_checkpointing \
  # --report_to wandb \