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


MODELS=(
  "objrel_T5_DiT_B_pilot"
  "objrel_rndembdposemb_DiT_B_pilot"
  "objrel_CLIPemb_DiT_B_pilot"
)

# Must be 1-1 aligned with MODELS
TEXT_ENCODERS=(
  "T5"
  "RandomEmbeddingEncoder_wPosEmb"
  "openai_CLIP"
)

SYN_MAPS=(
  "red_to_crimson"
  "blue_to_navy"
)

CKPT="epoch_4000_step_1600000.pth"
TEMPLATE="{color1} {shape1} is {rel_text} {color2} {shape2}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
MODEL_IDX=$(( TASK_ID / ${#SYN_MAPS[@]} ))
SYN_IDX=$(( TASK_ID % ${#SYN_MAPS[@]} ))

MODEL_NAME="${MODELS[$MODEL_IDX]}"
TEXT_ENCODER_TYPE="${TEXT_ENCODERS[$MODEL_IDX]}"
COLOR_SYNONYM_MAP="${SYN_MAPS[$SYN_IDX]}"


echo "SLURM job: ${SLURM_JOB_ID:-NA} array_task: ${TASK_ID}"
echo "Model: ${MODEL_NAME}"
echo "Text encoder type: ${TEXT_ENCODER_TYPE}"
echo "Color synonym map: ${COLOR_SYNONYM_MAP}"
echo "Checkpoint: ${CKPT}"
echo "Template: ${TEMPLATE}"

python3 generalization_profile_eval_cli.py \
  --model_run_name "${MODEL_NAME}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}" \
  --checkpoints "${CKPT}" \
  --prompt_templates "${TEMPLATE}" \
  --color_synonym_map "${COLOR_SYNONYM_MAP}" \


