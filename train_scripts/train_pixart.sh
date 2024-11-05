torchrun --nproc_per_node=1 \
    train_scripts/train.py \
    ~/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_S_img128_internal_objrelation_training_from_scratch.py \
    --work-dir "${STORE_DIR}/DL_Projects/PixArt/results/objrel_pilot/output/trained_model" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"

mamba deactivate
mamba activate torch2

cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    train_scripts/train.py \
    ~/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_DiT_B_pilot/output/trained_model" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"
