#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --array 3-9
#SBATCH --job-name=generaliz_train_traj_eval
#SBATCH -o /n/home12/binxuwang/Github/DiffusionObjectRelation/cluster_logs/generalize_train_traj_eval.%N_%A_%a.out           # STDOUT
#SBATCH -e /n/home12/binxuwang/Github/DiffusionObjectRelation/cluster_logs/generalize_train_traj_eval.%N_%A_%a.err           # STDERR

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--model_run_name objrel_T5_DiT_B_pilot 
--model_run_name objrel_rndembdposemb_DiT_B_pilot 
--model_run_name objrel_T5_DiT_mini_pilot 
--model_run_name objrel_rndembdposemb_DiT_micro_pilot 
--model_run_name objrel_rndembdposemb_DiT_nano_pilot 
--model_run_name objrel_rndembdposemb_DiT_mini_pilot 
--model_run_name objrel_T5_DiT_B_pilot_WDecay 
--model_run_name objrel_T5_DiT_mini_pilot_WDecay 
--model_run_name objrel_rndemb_DiT_B_pilot 
--model_run_name objrel_CLIPemb_DiT_B_pilot 
--model_run_name objrel_CLIPemb_DiT_mini_pilot 
'
export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

module load python
mamba deactivate
mamba activate torch2
which python
echo "Running script with config: $param_name"

cd /n/home12/binxuwang/Github/DiffusionObjectRelation/experimental_scripts
python generalization_profile_eval_cli.py  $param_name \
    --checkpoints epoch_3600_step_144000.pth epoch_3650_step_146000.pth epoch_3700_step_148000.pth epoch_3750_step_150000.pth epoch_3800_step_152000.pth epoch_3850_step_154000.pth epoch_3900_step_156000.pth epoch_3950_step_158000.pth epoch_4000_step_160000.pth


# --text_encoder_type T5
# --text_encoder_type RandomEmbeddingEncoder_wPosEmb
# --text_encoder_type T5
# --text_encoder_type RandomEmbeddingEncoder_wPosEmb
# --text_encoder_type RandomEmbeddingEncoder_wPosEmb
# --text_encoder_type RandomEmbeddingEncoder_wPosEmb
# --text_encoder_type T5
# --text_encoder_type T5
# --text_encoder_type RandomEmbeddingEncoder




# Define the parameter list as a proper bash array

# # "--model_run_name objrel_rndembdposemb_DiT_mini_pilot --ckpt_name epoch_1600_step_64000.pth --text_encoder_type RandomEmbeddingEncoder_wPosEmb --suffix '_ep1600'"
# # "--model_run_name objrel_rndembdposemb_DiT_mini_pilot --ckpt_name epoch_4000_step_160000.pth --text_encoder_type RandomEmbeddingEncoder_wPosEmb "
# # "--model_run_name objrel_rndembdposemb_DiT_micro_pilot --ckpt_name epoch_4000_step_160000.pth --text_encoder_type RandomEmbeddingEncoder_wPosEmb "
# # "--model_run_name objrel2_DiT_B_pilot --ckpt_name epoch_2000_step_80000.pth --text_encoder_type T5 "
# # "--model_run_name objrel_T5_DiT_B_pilot --ckpt_name epoch_4000_step_160000.pth --text_encoder_type T5 "
# # "--model_run_name objrel_T5_DiT_mini_pilot --ckpt_name epoch_4000_step_160000.pth --text_encoder_type T5 "
# "--model_run_name objrel_T5_DiT_B_pilot --ckpt_name epoch_4000_step_160000.pth --text_encoder_type T5 --T5_dtype bfloat16"
# "--model_run_name objrel_T5_DiT_B_pilot_WDecay --ckpt_name epoch_4000_step_160000.pth --text_encoder_type T5 --T5_dtype bfloat16"
# "--model_run_name objrel_T5_DiT_mini_pilot_WDecay --ckpt_name epoch_4000_step_160000.pth --text_encoder_type T5 --T5_dtype bfloat16"

# param_list=(
#     "--model_run_name objrel_T5_DiT_B_pilot --text_encoder_type T5"
#     "--model_run_name objrel_T5_DiT_mini_pilot --text_encoder_type T5"
#     "--model_run_name objrel_rndembdposemb_DiT_B_pilot --text_encoder_type RandomEmbeddingEncoder_wPosEmb"
#     "--model_run_name objrel_rndembdposemb_DiT_micro_pilot --text_encoder_type RandomEmbeddingEncoder_wPosEmb"
#     "--model_run_name objrel_rndembdposemb_DiT_nano_pilot --text_encoder_type RandomEmbeddingEncoder_wPosEmb"
#     "--model_run_name objrel_rndembdposemb_DiT_mini_pilot --text_encoder_type RandomEmbeddingEncoder_wPosEmb"
#     "--model_run_name objrel_rndemb_DiT_B_pilot --text_encoder_type RandomEmbeddingEncoder"
#     "--model_run_name objrel_T5_DiT_B_pilot_WDecay --text_encoder_type T5"
#     "--model_run_name objrel_T5_DiT_mini_pilot_WDecay --text_encoder_type T5"
# # ]:
# )

# cd /n/home12/binxuwang/Github/DiffusionObjectRelation/experimental_scripts
# # Loop through each parameter set and run the script
# for param in "${param_list[@]}"; do
#     echo "Running with parameters: $param"
#     time python posthoc_generation_train_traj_eval_cli.py $param
# done