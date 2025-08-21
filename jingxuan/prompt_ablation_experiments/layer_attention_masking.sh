#!/bin/bash
#SBATCH --job-name=layer_attention_masking
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:00
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=jfan@g.harvard.edu


echo "started job"
source ~/.bashrc
module load python
conda deactivate
conda activate /n/holylabs/LABS/sompolinsky_lab/Users/xupan/envs/pixart
echo "PYTHON ENV: $(which python)"

cd /n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation/jingxuan/prompt_ablation_experiments


python head_attention_masking.py 