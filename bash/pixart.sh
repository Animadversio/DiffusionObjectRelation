#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH -c 1                         # 1 CPU core
#SBATCH --mem=16G                    # 16GB RAM
#SBATCH -t 1:00:00                   # 1 hour
#SBATCH --gres=gpu:teslaV100:1       # Request 1 Tesla V100 GPU
#SBATCH --job-name="vscodetunnel"    

# Load required modules
module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/23.1.0

# Start GPU monitoring in background
/n/cluster/bin/job_gpu_monitor.sh &

# Activate your conda environment
source activate DiT

# Keep the job running
sleep 1h