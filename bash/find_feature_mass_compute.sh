#!/bin/bash
#SBATCH -t 24:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sapphire          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=50G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --array 1-32
#SBATCH -o pixart_mass_feat_finder_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e pixart_mass_feat_finder_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--layer_id 11 --t_start 10 --t_end 14
--layer_id 11 --t_start 7 --t_end 10
--layer_id 11 --t_start 4 --t_end 7
--layer_id 11 --t_start 0 --t_end 4
--layer_id  8 --t_start 10 --t_end 14
--layer_id  8 --t_start 7 --t_end 10
--layer_id  8 --t_start 4 --t_end 7
--layer_id  8 --t_start 0 --t_end 4
--layer_id  5 --t_start 10 --t_end 14
--layer_id  5 --t_start 7 --t_end 10
--layer_id  5 --t_start 4 --t_end 7
--layer_id  5 --t_start 0 --t_end 4
--layer_id  4 --t_start 10 --t_end 14
--layer_id  4 --t_start 7 --t_end 10
--layer_id  4 --t_start 4 --t_end 7
--layer_id  4 --t_start 0 --t_end 4
--layer_id  0 --t_start 10 --t_end 14
--layer_id  0 --t_start 7 --t_end 10
--layer_id  0 --t_start 4 --t_end 7
--layer_id  0 --t_start 0 --t_end 4
--layer_id  9 --t_start 10 --t_end 14
--layer_id  9 --t_start 7 --t_end 10
--layer_id  9 --t_start 4 --t_end 7
--layer_id  9 --t_start 0 --t_end 4
--layer_id  7 --t_start 10 --t_end 14
--layer_id  7 --t_start 7 --t_end 10
--layer_id  7 --t_start 4 --t_end 7
--layer_id  7 --t_start 0 --t_end 4
--layer_id  2 --t_start 10 --t_end 14
--layer_id  2 --t_start 7 --t_end 10
--layer_id  2 --t_start 4 --t_end 7
--layer_id  2 --t_start 0 --t_end 4
'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python
mamba deactivate
# module load cuda cudnn
mamba activate torch2
which python

# run code
cd /n/home12/binxuwang/Github/DiffusionObjectRelation
python analysis_script/find_feature_classifier_batch_process_CLI.py $param_name
