#!/bin/bash
#SBATCH --job-name=GCMAE_Pretrain
#SBATCH --output=gcmae_pretrain_%j.out
#SBATCH --error=gcmae_pretrain_%j.err
#SBATCH --partition=kisski-h100      
#SBATCH --gres=gpu:h100:1            
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load modules
module load cuda/11.1
module load python/3.8

# Activate your conda environment
source /user/louay.hamdi/u13592/ls/etc/profile.d/conda.sh
conda activate pytorch_env

# Set distributed training env
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 🚀 Launch pretraining
torchrun --nproc_per_node=1 --master_port=29500 main_pretrain.py \
  --data_path /user/louay.hamdi/u13592/.project/dir.project/NSCLC_pretraining_dataset/images \
  --data_val_path /user/louay.hamdi/u13592/.project/dir.project/NSCLC_pretraining_dataset/images \
  --output_dir /user/louay.hamdi/u13592/.project/dir.project/gcmae_output/checkpoints/pretrained_on_NSCLC_no_normpixloss \
  --log_dir /user/louay.hamdi/u13592/.project/dir.project/gcmae_logs \
  --batch_size 256 \
  --model gcmae_vit_base_patch16 \
  --mask_ratio 0.8 \
  --epochs 80 \
  --warmup_epochs 40 \
  --blr 1e-3 \
  --weight_decay 0.05 \
  --low_dim 768 \
  --nce_k 8192 \
  --nce_t 0.07 \
  --nce_m 0.5 \
  --norm_pix_loss False \
  --num_workers 8 \
  --world_size 1
