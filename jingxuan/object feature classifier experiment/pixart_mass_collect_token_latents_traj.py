# %%
try:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
except:
    pass

# %%
import os
from os.path import join
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import math
import torch
import pickle as pkl
from tqdm import tqdm, trange
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler
import sys
sys.path.append("/n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation/PixArt-alpha")
from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
sys.path.append("/n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation/utils")
from image_utils import pil_images_to_grid
from cv2_eval_utils import find_classify_objects, find_classify_object_masks
from layer_hook_utils import featureFetcher_module_recurrent
from pixart_utils import state_dict_convert
from pixart_sampling_utils import PixArtAlphaPipeline_custom, visualize_prompts_with_traj,visualize_single_prompt_with_traj
from pixart_utils import construct_diffuser_transformer_from_config, construct_diffuser_pipeline_from_config
from transformers import T5Tokenizer, T5EncoderModel,T5TokenizerFast
from custom_text_encoding_utils import save_prompt_embeddings_randemb, RandomEmbeddingEncoder, RandomEmbeddingEncoder_wPosEmb


# %% [markdown]
# ### Customize the PixArt pipeline to facilitate hooking

# %% [markdown]
# ### Loading directly from Diffuser transformers
savedir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot"
ckptdir = join(savedir, "checkpoints")

config = read_config(join(savedir, 'config.py'))

weight_dtype = torch.float32
if config.mixed_precision == "fp16": # accelerator.
    weight_dtype = torch.float16
elif config.mixed_precision == "bf16": # accelerator.
    weight_dtype = torch.bfloat16
print(f"weight_dtype: {weight_dtype}")
transformer = construct_diffuser_transformer_from_config(config)
pipeline = construct_diffuser_pipeline_from_config(config, pipeline_class=PixArtAlphaPipeline_custom)
ckpt = torch.load(join(ckptdir, "epoch_4000_step_160000.pth"))
pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict_ema']))
T5_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"
text_feat_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/objectRel_pilot_rndembposemb/caption_feature_wmask'
tokenizer = T5Tokenizer.from_pretrained(T5_path)
print("Fast tokenizer is_fast? ", getattr(tokenizer, "is_fast", False))
rnd_encoding = torch.load(join(text_feat_dir, "word_embedding_dict.pt"))
rndpos_encoder = RandomEmbeddingEncoder_wPosEmb(rnd_encoding["embedding_dict"],
                                              rnd_encoding["input_ids2dict_ids"],
                                              rnd_encoding["dict_ids2input_ids"],
                                              max_seq_len=20, embed_dim=4096,
                                              wpe_scale=1/6).to("cuda")
pipeline.text_encoder = rndpos_encoder
pipeline.text_encoder.dtype = weight_dtype
pipeline.tokenizer = tokenizer
device = "cuda"
pipeline = pipeline.to(device)
pipeline = pipeline.to(dtype=weight_dtype)
pipeline.set_progress_bar_config(disable=True)

# %%
import pprint
pprint.pprint(dict(config))

# %%
relations = [
    "above",
    "below",
    "to the left of",
    "to the right of",
    "to the upper left of",
    "to the upper right of",
    "to the lower left of",
    "to the lower right of",
]
# visualize prompts
# for norm1 version only had the color relation; added the shape relation for norm2 version
visualize_prompts = [f"red is {relation} blue" for relation in relations] + [f"blue is {relation} red" for relation in relations] +\
      [f"triangle is {relation} square" for relation in relations] + [f"square is {relation} triangle" for relation in relations] + \
        [f"triangle is {relation} circle" for relation in relations] + [f"circle is {relation} triangle" for relation in relations] + \
            [f"circle is {relation} square" for relation in relations] + [f"square is {relation} circle" for relation in relations] 

# %%
new_prompt_cache_dir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/red_blue_8_position_rndembposemb"
saveroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot/latent_store_norm2"

fetcher = featureFetcher_module_recurrent()
for layer_idx in range(12):
    fetcher.record_module(pipeline.transformer.transformer_blocks[layer_idx].norm2, f"block.{layer_idx}.norm2", )

for random_seed in trange(10):
    for prompt_idx in range(len(visualize_prompts)):
        print(visualize_prompts[prompt_idx])
        fetcher.clean_activations()
        image_logs, latents_traj, pred_traj, t_traj = visualize_prompts_with_traj(pipeline, 
                                                    visualize_prompts[prompt_idx:prompt_idx+1], new_prompt_cache_dir, 
                                                    max_length=config.model_max_length, weight_dtype=weight_dtype, 
                                                    num_images_per_prompt=25, random_seed=random_seed, device="cuda",)
        latents_traj = [latents_trj.cpu() for latents_trj in latents_traj]
        pred_traj = [pred_trj.cpu() for pred_trj in pred_traj]
        print(t_traj)
        t_traj = [[t.cpu() for t in t_traj[0]]]
        savedict = {"prompt": visualize_prompts[prompt_idx], 
                  "random_seed": random_seed,
                  "image_logs": image_logs,
                  "latents_traj": latents_traj,
                  "pred_traj": pred_traj,
                  "t_traj": t_traj,}
        for layer_idx in range(12):
            residual_state_traj = torch.stack(fetcher[f"block.{layer_idx}.norm2"])    
            t_steps, batch_size2, seqlen, hidden_dim = residual_state_traj.shape
            residual_spatial_state_traj = residual_state_traj.reshape(t_steps, batch_size2, int(math.sqrt(seqlen)), int(math.sqrt(seqlen)), hidden_dim)
            #print(layer_idx, residual_spatial_state_traj.shape) # (14, 50, 8, 8, 768)
            savedict[f"block_{layer_idx}_residual_spatial_state_traj"] = residual_spatial_state_traj.detach().cpu()
        
        os.makedirs(saveroot, exist_ok=True)
        pkl.dump(savedict, 
                open(join(saveroot, f"red_blue_8_pos_rndembposemb_img_latent_residual_allblocks_prompt{prompt_idx}_seed{random_seed}.pkl"), "wb"))


