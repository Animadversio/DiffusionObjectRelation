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
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation2_prompt20_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel2_DiT_B_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"

# --pipeline_load_from $STORE_DIR"/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl/ " \

# train conditional diffusion with random embedding, no meaning in text encoder
cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation_rndembd_prompt20_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_rndemb_DiT_B_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"

# continue training of random embedding model above 
cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation_rndembd_prompt20_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_rndemb_DiT_B_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss" \
    --resume-from $STORE_DIR/DL_Projects/PixArt/results/objrel_rndemb_DiT_B_pilot/checkpoints/epoch_2000_step_80000.pth

# train conditional diffusion with random embedding, with positional embedding 
cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation_rndembdposemb_prompt20_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"



# train conditional diffusion with random embedding, with positional embedding 
cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_nano_img128_internal_objrelation_rndembdposemb_prompt20_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_nano_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"


# train conditional diffusion with random embedding, with positional embedding 
cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_micro_img128_internal_objrelation_rndembdposemb_prompt20_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_micro_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"


# train conditional diffusion with random embedding, with positional embedding 
cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_mini_img128_internal_objrelation_rndembdposemb_prompt20_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_mini_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"


# train conditional diffusion on dataset of single object images. POC: Hannah
cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/hjkim/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation_single_T5_prompt20_training_from_scratch.py  \
    --work-dir "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_singleobj_T5_DiT_B_pilot/" \

# train conditional diffusion with random embedding, with positional embedding 
cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_mini_img128_internal_objrelation_T5_prompt20_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_T5_DiT_mini_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"


cd ~/Github/DiffusionObjectRelation
torchrun --nproc_per_node=1 \
    PixArt-alpha/train_scripts/train_with_visualize.py \
    /n/home12/binxuwang/Github/DiffusionObjectRelation/train_scripts/train_configs/PixArt_B_img128_internal_objrelation_T5_prompt20_training_from_scratch.py \
    --work-dir $STORE_DIR"/DL_Projects/PixArt/results/objrel_T5_DiT_B_pilot/" \
    --report_to "tensorboard" \
    --loss_report_name "train_loss"

