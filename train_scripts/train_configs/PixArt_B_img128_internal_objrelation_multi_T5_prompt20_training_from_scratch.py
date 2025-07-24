_base_ = ['/n/home12/hjkim/Github/DiffusionObjectRelation/PixArt-alpha/configs/PixArt_xl2_internal.py']
data_root = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt'

image_list_json = ['data_info.json',]

data = dict(type='InternalData', root='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRelMulti_pilot1', 
            image_list_json=image_list_json, transform='default_train', load_vae_feat=True, max_length=20)
image_size = 128

# model setting
window_block_indexes = []
window_size=0
use_rel_pos=False
model = 'PixArt_mini_2' #The base model 
model_max_length = 20
caption_channels = 4096
fp32_attention = True
load_from = None

# resume_from = dict(checkpoint=None, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
resume_from = dict(checkpoint=None, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
# load_from = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/PixArt-XL-2-512x512.pth"
vae_pretrained = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/sd-vae-ft-ema"
lewei_scale = 1.0

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=10
train_batch_size = 256 # 32
num_epochs = 4000 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
# optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
# lr_schedule_args = dict(num_warmup_steps=1000)
# we use different weight decay with the official implementation since it results better result
auto_lr = dict(rule='sqrt')
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=3e-2, eps=1e-10)
lr_schedule = 'constant'
lr_schedule_args = dict(num_warmup_steps=500)

save_model_epochs = 50
save_model_steps = 2000
eval_sampling_steps = 250
log_interval = 20
do_visualize_samples = True

prompt_cache_dir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/prompt_cache_t5emb_multi"

validation_prompts = [
    "triangle",
    "square",
    "circle",
    "red",
    "blue",
    "red square",
    "blue circle",
    "blue triangle",
    "a red square",
    "a blue circle",
    "a blue triangle",
    "the blue square",
    "the red circle",
    "the triangle",
    "the square",
    "the circle",
    "the",
    "or",
    "an",
    "red",
    "blue",
    "the red square above the blue circle",
    "blue triangle to the left of red square",
    "red circle below blue triangle",
    "red circle to the right of blue triangle",
    ]



