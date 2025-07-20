#!/bin/bash
#SBATCH --job-name=text_prompt_ablation
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-04:00
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

 
REVERSE_MASK="--reverse_mask False"
ONE_STEP_MODE="--one_step_mode True"
MASK_FUNC="--mask_func None"

echo "Running with reverse_mask=${REVERSE_MASK} and one_step_mode=${ONE_STEP_MODE} and mask_func=${MASK_FUNC}"
python text_prompt_ablation.py ${REVERSE_MASK} ${ONE_STEP_MODE} ${MASK_FUNC}