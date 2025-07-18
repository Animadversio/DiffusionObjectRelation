#!/bin/bash

# Define the parameter list as a proper bash array
param_list=(
    "--model_run_name objrel_rndembdposemb_DiT_mini_pilot --ckpt_name epoch_1600_step_64000.pth --text_encoder_type RandomEmbeddingEncoder_wPosEmb --suffix '_ep1600'"
    "--model_run_name objrel_rndembdposemb_DiT_mini_pilot --ckpt_name epoch_4000_step_160000.pth --text_encoder_type RandomEmbeddingEncoder_wPosEmb "
    "--model_run_name objrel_rndembdposemb_DiT_micro_pilot --ckpt_name epoch_4000_step_160000.pth --text_encoder_type RandomEmbeddingEncoder_wPosEmb "
    "--model_run_name objrel2_DiT_B_pilot --ckpt_name epoch_2000_step_80000.pth --text_encoder_type T5 "
    "--model_run_name objrel_T5_DiT_B_pilot --ckpt_name epoch_4000_step_160000.pth --text_encoder_type T5 "
    "--model_run_name objrel_T5_DiT_mini_pilot --ckpt_name epoch_4000_step_160000.pth --text_encoder_type T5 "
)

# Loop through each parameter set and run the script
for param in "${param_list[@]}"; do
    echo "Running with parameters: $param"
    time python cross_attn_head_filtering_mass_process_multimodel.py $param
done



# Appendix: Individual parameter assignments for reference (matching param_list exactly)
# Individual parameter assignments for reference (matching param_list exactly)
# 1. objrel2_DiT_B_pilot
model_run_name="objrel2_DiT_B_pilot"
ckpt_name="epoch_2000_step_80000.pth"
text_encoder_type="T5"
suffix=""

# 2. objrel_rndembdposemb_DiT_mini_pilot (ep4000)
model_run_name="objrel_rndembdposemb_DiT_mini_pilot"
ckpt_name="epoch_4000_step_160000.pth"
text_encoder_type="RandomEmbeddingEncoder_wPosEmb"
suffix=""

# 3. objrel_rndembdposemb_DiT_micro_pilot
model_run_name="objrel_rndembdposemb_DiT_micro_pilot"
ckpt_name="epoch_4000_step_160000.pth"
text_encoder_type="RandomEmbeddingEncoder_wPosEmb"
suffix=""

# 4. objrel_T5_DiT_mini_pilot
model_run_name="objrel_T5_DiT_mini_pilot"
ckpt_name="epoch_4000_step_160000.pth"
text_encoder_type="T5"
suffix=""

# 5. objrel_rndembdposemb_DiT_mini_pilot (ep1600)
model_run_name="objrel_rndembdposemb_DiT_mini_pilot"
ckpt_name="epoch_1600_step_64000.pth"
text_encoder_type="RandomEmbeddingEncoder_wPosEmb"
suffix="_ep1600"

# 6. objrel_T5_DiT_B_pilot
model_run_name="objrel_T5_DiT_B_pilot"
ckpt_name="epoch_4000_step_160000.pth"
text_encoder_type="T5"
suffix=""