#!/usr/bin/env python3
"""
Helper script to create a default config.py file for model evaluation.
This creates a minimal config with the required attributes.
"""

import os
import sys

def create_default_config(model_run_name, output_dir):
    """Create a default config.py file for the given model."""
    
    # Determine model type and size from model_run_name
    if "DiT_B" in model_run_name:
        model = "PixArt_B_2"
        depth = 12
        hidden_size = 768
        num_heads = 12
    elif "DiT_mini" in model_run_name:
        model = "PixArt_mini_2"
        depth = 6
        hidden_size = 384
        num_heads = 6
    elif "DiT_micro" in model_run_name:
        model = "PixArt_micro_2"
        depth = 6
        hidden_size = 192
        num_heads = 3
    elif "DiT_nano" in model_run_name:
        model = "PixArt_nano_2"
        depth = 3
        hidden_size = 192
        num_heads = 3
    else:
        # Default to B size
        model = "PixArt_B_2"
        depth = 12
        hidden_size = 768
        num_heads = 12
    
    config_content = f'''# Default config for {model_run_name}
# This is a minimal config file for evaluation purposes

# Model configuration
model = "{model}"
image_size = 512
latent_size = image_size // 8

# Precision settings
mixed_precision = "bf16"  # or "fp16", "fp32"

# Model parameters
pred_sigma = True
learn_sigma = True

# Window attention settings
window_block_indexes = []
window_size = 0
use_rel_pos = True
lewei_scale = 1.0

# Training settings
grad_checkpointing = False
fp32_attention = False
model_max_length = 20

# Sampling settings
train_sampling_steps = 14
snr_loss = 0.1

# Other settings
seed = 42
output_dir = "/tmp"
'''
    
    config_path = os.path.join(output_dir, "config.py")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created default config at: {config_path}")
    return config_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_default_config.py <model_run_name> <output_dir>")
        print("Example: python create_default_config.py objrel_T5_DiT_B_pilot /path/to/model/dir")
        sys.exit(1)
    
    model_run_name = sys.argv[1]
    output_dir = sys.argv[2]
    
    create_default_config(model_run_name, output_dir) 