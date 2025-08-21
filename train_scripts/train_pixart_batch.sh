#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --time=16:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --array 1-2
#SBATCH --job-name=train_pixart_batch
#SBATCH -o /n/home12/binxuwang/Github/DiffusionObjectRelation/cluster_logs/train_pixart_batch.%N_%A_%a.out           # STDOUT
#SBATCH -e /n/home12/binxuwang/Github/DiffusionObjectRelation/cluster_logs/train_pixart_batch.%N_%A_%a.err           # STDERR

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'PixArt_B_img128_internal_objrelation2_prompt32_training_from_scratch.py  --work-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel2_DiT_B_pilot/
PixArt_B_img128_internal_objrelation2_prompt32_training_from_scratch.py  --work-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel2_DiT_B_pilot/
'
export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

module load python
mamba deactivate
mamba activate torch2
which python
echo "Running script with config: $param_name"

cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/$param_name \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"
