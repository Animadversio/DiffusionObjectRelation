torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train.py \
    ~/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_S_img128_internal_objrelation_training_from_scratch.py \
    --work-dir "${STORE_DIR}/DL_Projects/PixArt/results/objrel_pilot/output/trained_model" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"


mamba deactivate
mamba activate torch2

cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train.py \
    ~/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_DiT_B_pilot/output/trained_model" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"

cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation2_prompt32_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel2_DiT_B_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"



cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_diffusers.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation2_prompt32_training_from_scratch.py \
    --pipeline_load_from $HF_HOME/hub/models--PixArt-alpha--PixArt-XL-2-512x512/ \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel2_DiT_B_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"

# --pipeline_load_from $STORE_DIR"/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl/ " \
