#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --time=16:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --array 1-6
#SBATCH --job-name=train_pixart_batch
#SBATCH -o /n/home12/hjkim/Github/DiffusionObjectRelation/cluster_logs/train_pixart_batch.%N_%A_%a.out           # STDOUT
#SBATCH -e /n/home12/hjkim/Github/DiffusionObjectRelation/cluster_logs/train_pixart_batch.%N_%A_%a.err           # STDERR

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'PixArt_B_img128_internal_objrelation_single_mini_rndemb.py  --work-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_singleobj_mini_rndemb
PixArt_B_img128_internal_objrelation_single_mini_rndembpos.py  --work-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_singleobj_mini_rndembpos
PixArt_B_img128_internal_objrelation_double_mini_rndemb.py  --work-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_doubleobj_mini_rndemb
PixArt_B_img128_internal_objrelation_double_mini_rndembpos.py  --work-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_doubleobj_mini_rndembpos
PixArt_B_img128_internal_objrelation_mixed_mini_rndemb.py  --work-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_mixedobj_mini_rndemb
PixArt_B_img128_internal_objrelation_mixed_mini_rndembpos.py  --work-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_mixedobj_mini_rndembpos
'
export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

module load python
mamba deactivate
mamba activate torch2
which python
echo "Running script with config: $param_name"
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

cd ~/Github/DiffusionObjectRelation
torchrun \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/hjkim/Github/DiffusionObjectRelation/train_scripts/train_configs/$param_name \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"
