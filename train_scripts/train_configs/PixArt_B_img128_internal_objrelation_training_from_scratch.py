_base_ = ['/n/home12/binxuwang/Github/DiffusionObjectRelation/PixArt-alpha/configs/PixArt_xl2_internal.py']
data_root = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt'

image_list_json = ['data_info.json',]

data = dict(type='InternalData', root='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/objectRel_pilot', 
            image_list_json=image_list_json, transform='default_train', load_vae_feat=True)
image_size = 128

# model setting
window_block_indexes = []
window_size=0
use_rel_pos=False
model = 'PixArt_B_2'
fp32_attention = True
load_from = None
resume_from = dict(checkpoint=None, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
# load_from = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/PixArt-XL-2-512x512.pth"
vae_pretrained = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/sd-vae-ft-ema"
lewei_scale = 1.0

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=10
train_batch_size = 256 # 32
num_epochs = 2000 # 3
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
save_model_steps = 1000
eval_sampling_steps = 100
log_interval = 20
work_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_pilot/output/debug'
