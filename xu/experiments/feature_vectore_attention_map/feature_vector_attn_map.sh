#!/bin/bash
#SBATCH --job-name=feature_vector_attn_map
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=0-24:00
#SBATCH --mem=200G
#SBATCH --array=0
#SBATCH --output=/n/sompolinsky_lab/Everyone/xupan/out/%x_%j.out
#SBATCH --error=/n/sompolinsky_lab/Everyone/xupan/err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=xupan@fas.harvard.edu

export HF_HOME=/n/netscratch/sompolinsky_lab/Everyone/xupan/huggingface
export HF_DATASETS_CACHE=/n/netscratch/sompolinsky_lab/Everyone/xupan/datasets


echo "started"

module purge
module load python
mamba deactivate
mamba activate /n/holylabs/LABS/sompolinsky_lab/Users/xupan/envs/pixart

which python


cd /n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation/xu/experiments/feature_vectore_attention_map

python feature_vector_attn_map.py
