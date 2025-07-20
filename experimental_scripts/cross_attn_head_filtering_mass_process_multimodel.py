#%%
%load_ext autoreload
%autoreload 2
#%%
import os
from os.path import join
import torch
import torch as th
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm.auto import trange
from contextlib import redirect_stdout

import sys
sys.path.append("/n/home12/binxuwang/Github/DiffusionObjectRelation/PixArt-alpha")
from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler
from transformers import T5Tokenizer, T5EncoderModel

sys.path.append("/n/home12/binxuwang/Github/DiffusionObjectRelation")
from utils.pixart_sampling_utils import pipeline_inference_custom, \
    PixArtAlphaPipeline_custom
from utils.pixart_utils import state_dict_convert
from utils.text_encoder_control_lib import RandomEmbeddingEncoder_wPosEmb
from utils.image_utils import pil_images_to_grid
from utils.attention_map_store_utils import replace_attn_processor, AttnProcessor2_0_Store, PixArtAttentionVisualizer_Store
from utils.cv2_eval_utils import find_classify_object_masks
from utils.attention_analysis_lib import plot_attention_layer_head_heatmaps, plot_layer_head_score_summary
from utils.attention_analysis_lib import *
from utils.cv2_eval_utils import find_classify_object_masks
from utils.obj_mask_utils import *
from circuit_toolkit.plot_utils import saveallforms

MAXIMUM = 255
# Create multi-hot masks for each category
def create_multi_hot_token_mask(tokens, target_words, seq_len=None):
    mask = th.zeros(len(tokens) if seq_len is None else seq_len, dtype=th.bool)
    for i, token in enumerate(tokens):
        # Remove the special prefix if present
        clean_token = token.replace('▁', '') if token.startswith('▁') else token
        if clean_token in target_words:
            mask[i] = True
    return mask


def create_object_based_masks(image_list, get_mask_func, map_shape=(8, 8), positive_threshold=180,):
    """
    some mask functions:
    - get_square_pos_others_neg_mask
    - get_triangle_pos_others_neg_mask
    - get_circle_pos_others_neg_mask
    - get_square_pos_others_neg_mask
    - get_triangle_pos_others_neg_mask
    - get_circle_pos_others_neg_mask
    """
    img_msks = []
    for img_idx in range(len(image_list)):
        image = image_list[img_idx]
        df, object_masks = find_classify_object_masks(image)
        obj_masks_resized = [cv2.resize(obj_mask, map_shape, interpolation=cv2.INTER_CUBIC) for obj_mask in object_masks]
        obj_masks_resized_binary = [obj_mask > positive_threshold for obj_mask in obj_masks_resized]
        obj_masks_resized_float = [obj_mask.astype(float) / MAXIMUM for obj_mask in obj_masks_resized]
        img_square_msk, _ = get_mask_func(df, obj_masks_resized_float)
        # img_triangle_msk, _ = get_triangle_pos_others_neg_mask(df, obj_masks_resized_float)
        # img_circle_msk, _ = get_circle_pos_others_neg_mask(df, obj_masks_resized_float)
        img_msks.append(th.from_numpy(img_square_msk).float())
    img_msks = th.stack(img_msks)
    # img_msks.shape # (n_samples, H, W)
    cmb_img_msks = img_msks.repeat(2,1,1)
    cmb_img_msks = cmb_img_msks.unsqueeze(1)
    # cmb_img_msks.shape # (n_samples * 2, 1, H, W)
    cmb_img_msks_vec = cmb_img_msks.flatten(start_dim=-2)
    return cmb_img_msks_vec, img_msks


def print_top_k_scores(scores, k=10, title="Top scores"):
    """
    Print the top k scores and their layer/head indices.
    
    Args:
        scores: tensor of shape (n_layers, n_heads, ...)
        k: number of top scores to print
        title: title for the output
    """
    # Flatten the scores while keeping track of layer and head indices
    scores = th.from_numpy(scores)
    n_layers, n_heads = scores.shape[:2]
    flattened_scores = scores.flatten()
    # Get the top k scores and their indices
    top_k_values, top_k_indices = flattened_scores.topk(k, largest=True)
    print(f"\n{title}:")
    print("-" * 50)
    for i, (value, idx) in enumerate(zip(top_k_values, top_k_indices)):
        # Convert flat index back to layer, head, and remaining dimensions
        remaining_size = scores.numel() // (n_layers * n_heads)
        layer_head_idx = idx // remaining_size
        layer_idx, head_idx = divmod(layer_head_idx.item(), n_heads)
        print(f"Top{i+1}: L{layer_idx}H{head_idx}, Score: {value:.2f}")
        
        
def test_product_prompt_list():
    from itertools import product
    colors = ['red', 'blue']
    target_shapes = ['square', 'triangle', 'circle']
    verticals = ['above', 'below']
    horizontals = ['to the left of', 'to the right of']
    prompts = []
    for c1, c2 in product(colors, colors):
        if c1 == c2:      # skip same‐color pairs
            continue
        for shape1, shape2 in product(target_shapes, target_shapes):
            if shape1 == shape2:
                continue
            for v, h in product(verticals, horizontals):
                prompts.append(f"{c1} {shape1} is {v} and {h} the {c2} {shape2}")
    return prompts

#%%
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Cross attention head filtering for multiple models')
    parser.add_argument('--model_run_name', type=str, required=True,
                        help='Name of the model run (e.g., objrel_T5_DiT_B_pilot)')
    parser.add_argument('--ckpt_name', type=str, required=True,
                        help='Checkpoint name (e.g., epoch_4000_step_160000.pth)')
    parser.add_argument('--text_encoder_type', type=str, required=True,
                        choices=['T5', 'RandomEmbeddingEncoder_wPosEmb'],
                        help='Type of text encoder to use')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for output directory naming')
    return parser.parse_args()

#%%
args = parse_args()
model_run_name = args.model_run_name
ckpt_name = args.ckpt_name
text_encoder_type = args.text_encoder_type
suffix = args.suffix
#%%
# savedir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot"
model_run_name = "objrel2_DiT_B_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
ckpt_name = "epoch_2000_step_80000.pth" # "epoch_4000_step_160000.pth"  
text_encoder_type = "T5"
suffix = ""

model_run_name = "objrel_rndembdposemb_DiT_mini_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
model_run_name = "objrel_rndembdposemb_DiT_micro_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth"  
text_encoder_type = "RandomEmbeddingEncoder_wPosEmb"
suffix = ""

model_run_name = "objrel_rndembdposemb_DiT_micro_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth"  
text_encoder_type = "RandomEmbeddingEncoder_wPosEmb"
suffix = ""


model_run_name = "objrel_T5_DiT_mini_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth"  
text_encoder_type = "T5"
suffix = ""

model_run_name = "objrel_rndembdposemb_DiT_mini_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
ckpt_name = "epoch_1600_step_64000.pth" # "epoch_4000_step_160000.pth" 
text_encoder_type = "RandomEmbeddingEncoder_wPosEmb" 
suffix = "_ep1600"

model_run_name = "objrel_T5_DiT_B_pilot"
ckpt_name = "epoch_4000_step_160000.pth"
ckpt_name = "epoch_1400_step_56000.pth"
ckpt_name = "epoch_1800_step_72000.pth"
ckpt_name = "epoch_2400_step_96000.pth"
text_encoder_type = "T5"
suffix = ""

#%%
text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/objectRel_pilot_rndemb/caption_feature_wmask'
T5_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"

savedir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/{model_run_name}"
figdir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/{model_run_name}{suffix}/cross_attn_vis_figs"
os.makedirs(figdir, exist_ok=True)

tokenizer = T5Tokenizer.from_pretrained(T5_path, )#subfolder="tokenizer")
if text_encoder_type == "T5":
    text_encoder = T5EncoderModel.from_pretrained(T5_path, load_in_8bit=False, torch_dtype=torch.float16, )
elif text_encoder_type == "RandomEmbeddingEncoder_wPosEmb":
    emb_data = th.load(join(text_feat_dir_old, "word_embedding_dict.pt"))
    text_encoder = RandomEmbeddingEncoder_wPosEmb(emb_data["embedding_dict"], 
                                                emb_data["input_ids2dict_ids"], 
                                                emb_data["dict_ids2input_ids"], 
                                                max_seq_len=20, embed_dim=4096,
                                                wpe_scale=1/6).to("cuda")
torch.cuda.empty_cache()
config = read_config(join(savedir, 'config.py'))
weight_dtype = torch.float32
if config.mixed_precision == "fp16": # accelerator.
    weight_dtype = torch.float16
elif config.mixed_precision == "bf16": # accelerator.
    weight_dtype = torch.bfloat16
    
image_size = config.image_size  # @param [256, 512, 1024]
latent_size = int(image_size) // 8
pred_sigma = getattr(config, 'pred_sigma', True)
learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                'model_max_length': config.model_max_length}
# train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
model = build_model(config.model,
                config.grad_checkpointing,
                config.get('fp32_attention', False),
                input_size=latent_size,
                learn_sigma=learn_sigma,
                pred_sigma=pred_sigma,
                **model_kwargs).train()
num_layers = len(model.blocks)
transformer = Transformer2DModel(
        sample_size=image_size // 8,
        num_layers=len(model.blocks),
        attention_head_dim=model.blocks[0].hidden_size // model.num_heads,
        in_channels=model.in_channels,
        out_channels=model.out_channels,
        patch_size=model.patch_size,
        attention_bias=True,
        num_attention_heads=model.num_heads,
        cross_attention_dim=model.blocks[0].hidden_size,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
)
# state_dict = state_dict_convert(all_state_dict.pop("state_dict"))
transformer.load_state_dict(state_dict_convert(model.state_dict()))
pipeline = PixArtAlphaPipeline_custom.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    transformer=transformer,
    tokenizer=tokenizer,
    text_encoder=None,
    torch_dtype=weight_dtype,
)
ckptdir = join(savedir, "checkpoints")
ckpt = torch.load(join(ckptdir, ckpt_name))
pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict_ema']))
# pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))
pipeline.tokenizer = tokenizer
pipeline.text_encoder = text_encoder
pipeline.to(device="cuda", dtype=weight_dtype);

# add attention map store hooks
pipeline.transformer = replace_attn_processor(pipeline.transformer)
attnvis_store = PixArtAttentionVisualizer_Store(pipeline)
attnvis_store.setup_hooks()
torch.cuda.empty_cache()

#%%
# prompt = "red square below and to the left of blue triangle"
for prompt in ['red square is above the blue triangle',
                'red square is below the blue triangle',
                'red square is to the left of the blue triangle',
                'red square is to the right of the blue triangle',
                'red square is above and to the left of the blue triangle',
                'red square is above and to the right of the blue triangle',
                'red square is below and to the left of the blue triangle',
                'red square is below and to the right of the blue triangle',
                'red square is above the blue circle',
                'red square is below the blue circle',
                'red square is to the left of the blue circle',
                'red square is to the right of the blue circle',
                'red square is above and to the left of the blue circle',
                'red square is above and to the right of the blue circle',
                'red square is below and to the left of the blue circle',
                'red square is below and to the right of the blue circle',
                'red triangle is above the blue square',
                'red triangle is below the blue square',
                'red triangle is to the left of the blue square',
                'red triangle is to the right of the blue square',
                'red triangle is above and to the left of the blue square',
                'red triangle is above and to the right of the blue square',
                'red triangle is below and to the left of the blue square',
                'red triangle is below and to the right of the blue square',
                'red triangle is above the blue circle',
                'red triangle is below the blue circle',
                'red triangle is to the left of the blue circle',
                'red triangle is to the right of the blue circle',
                'red triangle is above and to the left of the blue circle',
                'red triangle is above and to the right of the blue circle',
                'red triangle is below and to the left of the blue circle',
                'red triangle is below and to the right of the blue circle',
                'red circle is above the blue square',
                'red circle is below the blue square',
                'red circle is to the left of the blue square',
                'red circle is to the right of the blue square',
                'red circle is above and to the left of the blue square',
                'red circle is above and to the right of the blue square',
                'red circle is below and to the left of the blue square',
                'red circle is below and to the right of the blue square',
                'red circle is above the blue triangle',
                'red circle is below the blue triangle',
                'red circle is to the left of the blue triangle',
                'red circle is to the right of the blue triangle',
                'red circle is above and to the left of the blue triangle',
                'red circle is above and to the right of the blue triangle',
                'red circle is below and to the left of the blue triangle',
                'red circle is below and to the right of the blue triangle',
                'blue square is above the red triangle',
                'blue square is below the red triangle',
                'blue square is to the left of the red triangle',
                'blue square is to the right of the red triangle',
                'blue square is above and to the left of the red triangle',
                'blue square is above and to the right of the red triangle',
                'blue square is below and to the left of the red triangle',
                'blue square is below and to the right of the red triangle',
                'blue square is above the red circle',
                'blue square is below the red circle',
                'blue square is to the left of the red circle',
                'blue square is to the right of the red circle',
                'blue square is above and to the left of the red circle',
                'blue square is above and to the right of the red circle',
                'blue square is below and to the left of the red circle',
                'blue square is below and to the right of the red circle',
                'blue triangle is above the red square',
                'blue triangle is below the red square',
                'blue triangle is to the left of the red square',
                'blue triangle is to the right of the red square',
                'blue triangle is above and to the left of the red square',
                'blue triangle is above and to the right of the red square',
                'blue triangle is below and to the left of the red square',
                'blue triangle is below and to the right of the red square',
                'blue triangle is above the red circle',
                'blue triangle is below the red circle',
                'blue triangle is to the left of the red circle',
                'blue triangle is to the right of the red circle',
                'blue triangle is above and to the left of the red circle',
                'blue triangle is above and to the right of the red circle',
                'blue triangle is below and to the left of the red circle',
                'blue triangle is below and to the right of the red circle',
                'blue circle is above the red square',
                'blue circle is below the red square',
                'blue circle is to the left of the red square',
                'blue circle is to the right of the red square',
                'blue circle is above and to the left of the red square',
                'blue circle is above and to the right of the red square',
                'blue circle is below and to the left of the red square',
                'blue circle is below and to the right of the red square',
                'blue circle is above the red triangle',
                'blue circle is below the red triangle',
                'blue circle is to the left of the red triangle',
                'blue circle is to the right of the red triangle',
                'blue circle is above and to the left of the red triangle',
                'blue circle is above and to the right of the red triangle',
                'blue circle is below and to the left of the red triangle',
                'blue circle is below and to the right of the red triangle'
                ]:
    print(f"Processing prompt: {prompt}")
    prompt_dir = join(figdir, prompt.replace(" ", "_"))
    os.makedirs(prompt_dir, exist_ok=True)
    n_samples = 49
    attnvis_store.clear_activation()
    output = pipeline(prompt, 
            num_inference_steps=14,
            max_sequence_length=20, 
            num_images_per_prompt=n_samples,
            return_sample_pred_traj=True,
            device="cuda")
    pred_traj, latents_traj, t_traj = output[1], output[2], output[3]

    grid = pil_images_to_grid(output[0].images)
    grid.save(join(prompt_dir, "sample_images_grid.png"))

    attn_map_stacked = [th.stack(attnvis_store.activation[f'block{layer_i:02d}_self_attn_map'], dim=0) for layer_i in range(num_layers)]
    attn_map_stacked = th.stack(attn_map_stacked, dim=0)
    cross_attn_map_stacked = [th.stack(attnvis_store.activation[f'block{layer_i:02d}_cross_attn_map'], dim=0) for layer_i in range(num_layers)]
    cross_attn_map_stacked = th.stack(cross_attn_map_stacked, dim=0)
    print("attn_map_stacked.shape: ", attn_map_stacked.shape) # (num_layers, num_steps, num_images * 2, num_heads, num_tokens, num_tokens)
    print("cross_attn_map_stacked.shape: ", cross_attn_map_stacked.shape) # (num_layers, num_steps, num_images * 2, num_heads, num_tokens, num_word_tokens)

    token_splits = pipeline.tokenizer.tokenize(prompt)
    cond_slice = slice(n_samples, n_samples * 2)
    uncond_slice = slice(0, n_samples)

    square_cmb_img_msks_vec, _ = create_object_based_masks(output[0].images, get_square_pos_others_neg_mask)
    triangle_cmb_img_msks_vec, _ = create_object_based_masks(output[0].images, get_triangle_pos_others_neg_mask)
    circle_cmb_img_msks_vec, _ = create_object_based_masks(output[0].images, get_circle_pos_others_neg_mask)
    background_cmb_img_msks_vec, _ = create_object_based_masks(output[0].images, get_background_pos_obj_neg_mask)
    imgmsk_collection = {
        "square": square_cmb_img_msks_vec,
        "triangle": triangle_cmb_img_msks_vec,
        "circle": circle_cmb_img_msks_vec,
        "background": background_cmb_img_msks_vec,
    }
    for imgtoken_type in ["square", "triangle", "circle", "background"]:
        imgtoken_msk_vec = imgmsk_collection[imgtoken_type]
        if imgtoken_msk_vec.sum() == 0:
            print(f"No image mask for {imgtoken_type}, skipping...")
            continue
        for text_targets in [["square"], 
                            ["triangle"], 
                            ["circle"], 
                            ["red"], 
                            ["blue"],
                            ["below", "left", "top", "right", "above"],
                            ["below", ],
                            ["left", ],
                            ["above", ],
                            ["right", ],
                            ["and", "to", "the", "of", "on", "is"],
                            ]:
            template_type = f"image {imgtoken_type} to text {' '.join(text_targets)}"
            text_mask = create_multi_hot_token_mask(token_splits, text_targets, seq_len=20)
            if text_mask.sum() == 0:
                print(f"No text mask for {template_type}, skipping...")
                continue
            cross_attn_template = imgtoken_msk_vec[:, :, :, None] @ text_mask.float().flatten()[None, :] 
            template_similarity_scores = (cross_attn_map_stacked * cross_attn_template).sum(dim=-1).sum(dim=-1)
            fig = plot_attention_layer_head_heatmaps(template_similarity_scores[:, :, cond_slice], 
                                                    title_str=f"Attention template similarity cond pass | {template_type}\n{prompt}", 
                                                    figsize=(14, 5), sample_idx=None, num_heads=12, share_clim=False, panel_shape=(2, 6))
            saveallforms(prompt_dir, f"cross_attn_layer_head_step_cond_heatmap_{template_type.replace(' ', '_')}", figh=fig)
            figh, cond_stats, uncond_stats = plot_layer_head_score_summary(template_similarity_scores, f"{template_type}\n{prompt}", step_sum_type="max", share_clim=True);
            saveallforms(prompt_dir, f"cross_attn_layer_head_maxstep_summary_{template_type.replace(' ', '_')}", figh=figh)
            th.save({
                "cond_stats": cond_stats,
                "uncond_stats": uncond_stats,
                "template_similarity_scores": template_similarity_scores,
                "text_mask": text_mask,
                "imgtoken_msk_vec": imgtoken_msk_vec,
                "prompt": prompt,
                "template_type": template_type,
                "text_targets": text_targets,
                "imgtoken_type": imgtoken_type,
            }, join(prompt_dir, f"cross_attn_layer_head_step_stats_{template_type.replace(' ', '_')}.pt"))
            print_top_k_scores(cond_stats, k=8, title=f"Top Heads {template_type} | {prompt}");
            with open(join(prompt_dir, f"top_cross_attn_heads_{template_type.replace(' ', '_')}.txt"), "w") as f:
                with redirect_stdout(f):
                    print_top_k_scores(cond_stats, k=8, title=f"Top Heads {template_type} | {prompt}");
            print(f"Saved {template_type} to {prompt_dir}")
            plt.close("all")

# %%
