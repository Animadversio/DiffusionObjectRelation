#!/bin/bash

# Bash script to evaluate epoch 4000 checkpoints of all supported models using generalization CLI
# Generated for evaluating the last major checkpoint (epoch 4000) across all trained models

set -e  # Exit on error

# Set paths
SCRIPT_DIR="/n/home12/binxuwang/Github/DiffusionObjectRelation"
CLI_SCRIPT="experimental_scripts/generalization_profile_eval_cli.py"

# Models that have epoch 4000 checkpoints and are supported by the CLI script
MODELS=(
    "objrel_T5_DiT_B_pilot"
    "objrel_T5_DiT_mini_pilot" 
    "objrel_rndembdposemb_DiT_B_pilot"
    "objrel_rndembdposemb_DiT_micro_pilot"
    "objrel_rndembdposemb_DiT_nano_pilot"
    "objrel_rndembdposemb_DiT_mini_pilot"
    "objrel_rndemb_DiT_B_pilot"
    "objrel_T5_DiT_B_pilot_WDecay"
    "objrel_T5_DiT_mini_pilot_WDecay"
)

# Change to script directory
cd "$SCRIPT_DIR"

echo "Starting evaluation of epoch 4000 checkpoints for all models..."
echo "Date: $(date)"
echo "Models to evaluate: ${#MODELS[@]}"
echo ""

# Iterate through each model
for model in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Evaluating model: $model"
    echo "============================================================"
    
    # Check if epoch 4000 checkpoint exists
    checkpoint_path="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/$model/checkpoints/epoch_4000_step_160000.pth"
    
    if [[ -f "$checkpoint_path" ]]; then
        echo "Found checkpoint: epoch_4000_step_160000.pth"
        
        # Run the evaluation
        echo "Starting evaluation..."
        python "$CLI_SCRIPT" \
            --model_run_name "$model" \
            --checkpoints "epoch_4000_step_160000.pth" \
            --num_images 49 \
            --num_inference_steps 14 \
            --guidance_scale 4.5 \
            --generator_seed 42
        
        if [[ $? -eq 0 ]]; then
            echo "✓ Successfully evaluated $model"
        else
            echo "✗ Failed to evaluate $model"
        fi
        
    else
        echo "✗ Checkpoint not found: $checkpoint_path"
    fi
    
    echo ""
done

echo "============================================================"
echo "Evaluation complete!"
echo "Date: $(date)"
echo "============================================================"