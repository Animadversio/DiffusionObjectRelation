_base_ = ['/n/home12/binxuwang/Github/DiffusionObjectRelation/PixArt-alpha/configs/PixArt_xl2_internal.py']
data_root = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt'

image_list_json = ['data_info.json',]

data = dict(type='InternalData', root='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/pixart-pokemon', 
            image_list_json=image_list_json, transform='default_train', load_vae_feat=True)
image_size = 512

# model setting
window_block_indexes = []
window_size=0
use_rel_pos=False
model = 'PixArt_XL_2'
fp32_attention = True
load_from = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/PixArt-XL-2-512x512.pth"
vae_pretrained = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/sd-vae-ft-ema"
lewei_scale = 1.0

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=10
train_batch_size = 38 # 32
num_epochs = 200 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 200
log_interval = 20
save_model_steps=100
work_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/debug'
