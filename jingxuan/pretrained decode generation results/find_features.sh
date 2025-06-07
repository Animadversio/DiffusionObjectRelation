#!/bin/bash
#SBATCH --job-name=feature_classifier
#SBATCH --account=sompolinsky_lab
#SBATCH --partition=sapphire
#SBATCH --array=0-10%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --mail-user=jfan@g.harvard.edu

echo "started"
source ~/.bashrc
module load python
conda deactivate
conda activate /n/holylabs/LABS/sompolinsky_lab/Users/xupan/envs/pixart
echo "PYTHON ENV: $(which python)"

cd /n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation/jingxuan/latent_decoding_analysis

python find_feature_classifier_batch_process_CLI.py --layer_id $SLURM_ARRAY_TASK_ID 