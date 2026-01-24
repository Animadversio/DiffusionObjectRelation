#!/bin/bash
#SBATCH --job-name=eval_train_traj_parallel
#SBATCH --output=logs/eval_train_traj_parallel_%j.out
#SBATCH --error=logs/eval_train_traj_parallel_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hjkim@fas.harvard.edu

# Load modules
module load anaconda/2023a
module load cuda/11.8

# Activate conda environment
source activate /n/home12/hjkim/.conda/envs/diffusion_env

# Create logs directory
mkdir -p logs

# Change to script directory
cd /n/home12/hjkim/Github/DiffusionObjectRelation/experimental_scripts

# Run all models in parallel using 4 GPUs
python posthoc_generation_train_traj_eval_cli_cluster.py --parallel --num_workers 4

echo "Parallel job completed at $(date)" 