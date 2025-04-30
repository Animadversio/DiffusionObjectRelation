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
import argparse
import logging
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler
import sys
import json

# Add project paths
PROJECT_ROOT = "/n/netscratch/konkle_lab/Everyone/Jingxuan/DiffusionObjectRelation"
sys.path.append(join(PROJECT_ROOT, "PixArt-alpha"))
from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow

sys.path.append(join(PROJECT_ROOT, "utils"))
from image_utils import pil_images_to_grid
from cv2_eval_utils import find_classify_objects, find_classify_object_masks
from layer_hook_utils import featureFetcher_module_recurrent
from pixart_utils import state_dict_convert
from pixart_sampling_utils import PixArtAlphaPipeline_custom, visualize_prompts_with_traj_pretrained
from pixart_utils import construct_diffuser_transformer_from_config, construct_diffuser_pipeline_from_config

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_prompts(txt_file):
    """Load prompts from a text file."""
    with open(txt_file, 'r') as f:
        items = [item.strip() for item in f.readlines()]
    return items

def main(args):
    # Setup logging
    setup_logging()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.saveroot, exist_ok=True)
    
    # Load model
    logging.info(f"Loading model {args.model_name}")
    weight_dtype = torch.bfloat16
    pipeline = PixArtAlphaPipeline_custom.from_pretrained(
        args.model_name,
        torch_dtype=weight_dtype
    )
    
    # Load prompts
    logging.info(f"Loading prompts from {args.prompt_file}")
    visualize_prompts = load_prompts(args.prompt_file)
    
    # Load and filter prompts based on id_groups
    with open(join(PROJECT_ROOT, "analysis_script/id_groups.json"), 'r') as f:
        id_groups = json.load(f)
    selected_ids = id_groups['case2_ids'] + id_groups['case3_ids']
    selected_ids = [idx - 1 for idx in selected_ids]  # Convert to 0-based indexing
    visualize_prompts = [visualize_prompts[idx] for idx in selected_ids]
    logging.info(f"Filtered prompts to {len(visualize_prompts)} prompts from case2 and case3")
    
    # Setup feature fetcher
    fetcher = featureFetcher_module_recurrent()
    for layer_idx in range(28):
        fetcher.record_module(
            pipeline.transformer.transformer_blocks[layer_idx],
            f"block.{layer_idx}"
        )
    
    # Process each prompt
    for prompt_idx in range(len(visualize_prompts)):
        logging.info(f"Processing prompt {prompt_idx + 1}/{len(visualize_prompts)}")
        fetcher.clean_activations()
        
        try:
            image_logs, latents_traj, pred_traj, t_traj = visualize_prompts_with_traj_pretrained(
                pipeline,
                visualize_prompts[prompt_idx:prompt_idx+1],
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                device=torch.device("cuda"),
                random_seed=args.seed,
                weight_dtype=weight_dtype
            )
            
            # Move tensors to CPU
            latents_traj = [latents_trj.cpu() for latents_trj in latents_traj]
            pred_traj = [pred_trj.cpu() for pred_trj in pred_traj]
            t_traj = [[t.cpu() for t in t_traj[0]]]
            
            # Prepare save dictionary
            savedict = {
                "prompt": visualize_prompts[prompt_idx],
                "random_seed": args.seed,
                "image_logs": image_logs,
                "latents_traj": latents_traj,
                "pred_traj": pred_traj,
                "t_traj": t_traj,
            }
            
            # Process each layer
            for layer_idx in range(28):
                residual_state_traj = torch.stack(fetcher[f"block.{layer_idx}"])
                t_steps, batch_size2, seqlen, hidden_dim = residual_state_traj.shape
                residual_spatial_state_traj = residual_state_traj.reshape(
                    t_steps, batch_size2, int(math.sqrt(seqlen)), int(math.sqrt(seqlen)), hidden_dim
                )
                logging.info(f"Layer {layer_idx} shape: {residual_spatial_state_traj.shape}")
                savedict[f"block_{layer_idx}_residual_spatial_state_traj"] = residual_spatial_state_traj.detach().cpu()
            
            # Save results
            save_path = join(args.saveroot, f"spatial_img_latent_residual_allblocks_prompt{prompt_idx}_seed_{args.seed}.pkl")
            pkl.dump(savedict, open(save_path, "wb"))
            logging.info(f"Saved results to {save_path}")
            
        except Exception as e:
            logging.error(f"Error processing prompt {prompt_idx}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect latent trajectories from PixArt model")
    parser.add_argument("--model_name", type=str, default="PixArt-alpha/PixArt-XL-2-512x512",
                      help="Name of the pretrained model")
    parser.add_argument("--prompt_file", type=str,
                      default="/n/netscratch/konkle_lab/Everyone/Jingxuan/DiffusionObjectRelation/PixArt-alpha/asset/spatial.txt",
                      help="Path to the text file containing prompts")
    parser.add_argument("--saveroot", type=str, default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/pretrained/latents/",
                      help="Directory to save the results")
    parser.add_argument("--num_steps", type=int, default=14,
                      help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                      help="Guidance scale for generation")
    parser.add_argument("--seed", type=int, default=1,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)


