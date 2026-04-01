#!/bin/bash

# Bash script to run head_alignment_synopsis.py for all listed training runs
set -e

CKPT="epoch_4000_step_160000.pth"
FIGDIR="Figures/DiT_head_alignment"
    # objrel_CLIPemb_DiT_B_pilot \
    # objrel_CLIPemb_DiT_mini_pilot \
    # objrel_rndembdposemb_DiT_B_pilot \
    # objrel_rndembdposemb_DiT_micro_pilot \
    # objrel_rndembdposemb_DiT_mini_pilot \
    # objrel_rndembdposemb_DiT_nano_pilot
    
# Map run names to text encoder type by matching substrings
for MODEL_RUN_NAME in \
    objrel_T5_DiT_B_pilot \
    objrel_T5_DiT_B_pilot_WDecay \
    objrel_T5_DiT_mini_pilot \
    objrel_T5_DiT_mini_pilot_WDecay
do
    if [[ "$MODEL_RUN_NAME" == *"CLIPemb"* ]]; then
        TEXT_ENCODER_TYPE="CLIP"
    elif [[ "$MODEL_RUN_NAME" == *"T5"* ]]; then
        TEXT_ENCODER_TYPE="T5"
    elif [[ "$MODEL_RUN_NAME" == *"rndembdposemb"* ]]; then
        TEXT_ENCODER_TYPE="RTE"
    else
        echo "Cannot determine text encoder type for $MODEL_RUN_NAME"
        continue
    fi
    echo "Running: $MODEL_RUN_NAME (Encoder: $TEXT_ENCODER_TYPE)"
    python experimental_scripts/head_alignment_synopsis.py \
        --model_run_name "$MODEL_RUN_NAME" \
        --text_encoder_type "$TEXT_ENCODER_TYPE" \
        --ckpt_name "$CKPT" \
        --figdir "$FIGDIR"
done


