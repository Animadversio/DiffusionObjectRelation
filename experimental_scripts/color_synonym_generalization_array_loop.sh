#! /bin/bash
#SBATCH --job-name=robust
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:00
#SBATCH --mem=100G
#SBATCH -o sbatch_logs/%j_%a.out    # File to which STDOUT will be written, %j inserts jobid, %a inserts array index
#SBATCH -e sbatch_logs/%j_%a.err    # File to which STDERR will be written, %j inserts jobid, %a inserts array index
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jfan@g.harvard.edu
#SBATCH --array=0-5%3 # 3 models * 2 synonym maps = 6 jobs


echo "started job"
source ~/.bashrc
module load python
conda deactivate
conda activate torch2
echo "PYTHON ENV: $(which python)"

cd /n/netscratch/konkle_lab/Everyone/Jingxuan/DiffusionObjectRelation/experimental_scripts

TASK_ID=1
MODELS=(
  "objrel_T5_DiT_B_pilot"
  "objrel_rndembdposemb_DiT_B_pilot"
  "objrel_CLIPemb_DiT_B_pilot"
)

# Must be 1-1 aligned with MODELS
TEXT_ENCODERS=(
  "T5"
)

SYN_MAPS=(
  "red_to_crimson"
  "blue_to_navy"
)




echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--model_run_name objrel_T5_DiT_B_pilot  --text_encoder_type T5  --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}"   "{rel_text} {color2} {shape2} is {color1} {shape1}"   "{rel_text} {color2} {shape2} is the {color1} {shape1}"   "{rel_text} the {color2} {shape2} is {color1} {shape1}" --color_synonym_map none
--model_run_name objrel_T5_DiT_B_pilot  --text_encoder_type T5  --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}" --color_synonym_map red_to_crimson
--model_run_name objrel_T5_DiT_B_pilot  --text_encoder_type T5 --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}" --color_synonym_map blue_to_navy
'
# --model_run_name objrel_T5_DiT_mini_pilot 

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# CKPT="epoch_4000_step_160000.pth"
# TEMPLATE="{color1} {shape1} is {rel_text} {color2} {shape2}"

# TASK_ID="${SLURM_ARRAY_TASK_ID}"
# MODEL_IDX=$(( TASK_ID / ${#SYN_MAPS[@]} ))
# SYN_IDX=$(( TASK_ID % ${#SYN_MAPS[@]} ))

# MODEL_NAME="${MODELS[$MODEL_IDX]}"
# TEXT_ENCODER_TYPE="${TEXT_ENCODERS[$MODEL_IDX]}"
# COLOR_SYNONYM_MAP="${SYN_MAPS[$SYN_IDX]}"


# echo "SLURM job: ${SLURM_JOB_ID:-NA} array_task: ${TASK_ID}"
# echo "Model: ${MODEL_NAME}"
# echo "Text encoder type: ${TEXT_ENCODER_TYPE}"
# echo "Color synonym map: ${COLOR_SYNONYM_MAP}"
# echo "Checkpoint: ${CKPT}"
# echo "Template: ${TEMPLATE}"


python3 generalization_profile_eval_cli.py ${param_name} \
  --checkpoints epoch_4000_step_160000.pth


# python3 generalization_profile_eval_cli.py \
#   --model_run_name "${MODEL_NAME}" \
#   --text_encoder_type "${TEXT_ENCODER_TYPE}" \
#   --prompt_templates "${TEMPLATE}" \
#   --color_synonym_map "${COLOR_SYNONYM_MAP}"
#   --checkpoints epoch_4000_step_1600000.pth
python3 generalization_profile_eval_cli.py --model_run_name objrel_T5_DiT_B_pilot  --text_encoder_type T5  --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}" --color_synonym_map red_to_crimson \
  --checkpoints epoch_4000_step_160000.pth


param_list='
--model_run_name objrel_T5_DiT_B_pilot  --text_encoder_type T5  --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}"   "{rel_text} {color2} {shape2} is {color1} {shape1}"   "{rel_text} {color2} {shape2} is the {color1} {shape1}"   "{rel_text} the {color2} {shape2} is {color1} {shape1}" --color_synonym_map none
--model_run_name objrel_T5_DiT_B_pilot  --text_encoder_type T5  --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}" --color_synonym_map red_to_crimson
--model_run_name objrel_T5_DiT_B_pilot  --text_encoder_type T5 --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}" --color_synonym_map blue_to_navy
--model_run_name objrel_T5_DiT_mini_pilot  --text_encoder_type T5  --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}"   "{rel_text} {color2} {shape2} is {color1} {shape1}"   "{rel_text} {color2} {shape2} is the {color1} {shape1}"   "{rel_text} the {color2} {shape2} is {color1} {shape1}" --color_synonym_map none
--model_run_name objrel_T5_DiT_mini_pilot  --text_encoder_type T5  --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}" --color_synonym_map red_to_crimson
--model_run_name objrel_T5_DiT_mini_pilot  --text_encoder_type T5 --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}" --color_synonym_map blue_to_navy
'

while IFS= read -r param_name; do
  # Skip empty lines if any
  [ -z "$param_name" ] && continue
  python3 generalization_profile_eval_cli.py $param_name \
    --checkpoints epoch_4000_step_160000.pth
done <<< "$param_list"
python3 generalization_profile_eval_cli.py --model_run_name objrel_T5_DiT_B_pilot  --text_encoder_type T5  --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}"   "{rel_text} {color2} {shape2} is {color1} {shape1}"   "{rel_text} {color2} {shape2} is the {color1} {shape1}"   "{rel_text} the {color2} {shape2} is {color1} {shape1}" --color_synonym_map none \
  --checkpoints epoch_4000_step_160000.pth



while IFS= read -r line; do
  [[ -z "$line" ]] && continue

  # turn the line into real args (honor quotes inside the line)
  eval "set -- $line"

  python3 generalization_profile_eval_cli.py "$@" \
    --checkpoints epoch_4000_step_160000.pth
done <<< "$param_list"

