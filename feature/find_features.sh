#!/bin/bash
#SBATCH --job-name=llama_format
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-24:00
#SBATCH --mem=100G
#SBATCH --output=/n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/out/%x_%j.out
#SBATCH --error=/n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=xupan@fas.harvard.edu

echo "started"

module load python

mamba activate /n/holylabs/LABS/sompolinsky_lab/Users/xupan/envs/pixart

cd /n/home13/xupan/Projects/DiffusionObjectRelation/DiffusionObjectRelation/feature

python find_features.py