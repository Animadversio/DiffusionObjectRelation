#!/usr/bin/env python3
"""
CLI script for evaluating model generalization across different prompt templates and checkpoints.
Based on the notebook 20250819_DiT_T5_generalization_profile.ipynb.

This script pre-computes text embeddings once and reuses them across multiple checkpoint evaluations
for maximum efficiency.
"""

import argparse
import glob
import os
import sys
from os.path import join
import torch
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm.auto import tqdm
import time
from contextlib import redirect_stdout
from itertools import product

# Add project paths
sys.path.append("/n/home12/binxuwang/Github/DiffusionObjectRelation/PixArt-alpha")
sys.path.append("/n/home12/binxuwang/Github/DiffusionObjectRelation")

from diffusion.model.builder import build_model
from diffusion.utils.misc import read_config
from diffusers import Transformer2DModel
from transformers import T5Tokenizer, T5EncoderModel

from utils.pixart_sampling_utils import PixArtAlphaPipeline_custom
from utils.pixart_utils import state_dict_convert, construct_diffuser_pipeline_from_config
from utils.text_encoder_control_lib import RandomEmbeddingEncoder_wPosEmb, RandomEmbeddingEncoder
from utils.cv2_eval_utils import print_evaluation_summary
from utils.relation_shape_dataset_lib import ShapesDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate model generalization across prompt templates and checkpoints'
    )
    
    parser.add_argument('--model_run_name', type=str, required=True,
                        choices=[
                            "objrel_T5_DiT_B_pilot",
                            "objrel_T5_DiT_mini_pilot", 
                            "objrel_rndembdposemb_DiT_B_pilot",
                            "objrel_rndembdposemb_DiT_micro_pilot",
                            "objrel_rndembdposemb_DiT_nano_pilot",
                            "objrel_rndembdposemb_DiT_mini_pilot",
                            "objrel_rndemb_DiT_B_pilot",
                            "objrel_T5_DiT_B_pilot_WDecay",
                            "objrel_T5_DiT_mini_pilot_WDecay"
                        ],
                        help='Model run name to evaluate')
    
    parser.add_argument('--checkpoints', type=str, nargs='*',
                        help='Specific checkpoint files to evaluate (e.g., epoch_1500_step_60000.pth). If not specified, evaluates all available checkpoints.')
    
    parser.add_argument('--text_encoder_type', type=str,
                        choices=["T5", "RandomEmbeddingEncoder_wPosEmb", "RandomEmbeddingEncoder"],
                        help='Text encoder type (auto-determined from model name if not specified)')
    
    parser.add_argument('--prompt_templates', type=str, nargs='*',
                        default=[
                            "{color1} {shape1} is {rel_text} {color2} {shape2}",
                            "{color1} {shape1} {rel_text} {color2} {shape2}",
                            "{color1} {shape1} {rel_text} the {color2} {shape2}",
                            "{color1} {shape1} is {rel_text} the {color2} {shape2}",
                            "the {color1} {shape1} is {rel_text} the {color2} {shape2}",
                        ],
                        help='Prompt templates to evaluate')
    
    parser.add_argument('--num_images', type=int, default=49,
                        help='Number of images to generate per prompt')
    
    parser.add_argument('--num_inference_steps', type=int, default=14,
                        help='Number of inference steps')
    
    parser.add_argument('--guidance_scale', type=float, default=4.5,
                        help='Guidance scale for generation')
    
    parser.add_argument('--max_sequence_length', type=int, default=20,
                        help='Max sequence length for text encoding')
    
    parser.add_argument('--generator_seed', type=int, default=42,
                        help='Random seed for generation')
    
    parser.add_argument('--output_dir', type=str,
                        help='Custom output directory (default: results/{model_name}/generalization_eval/)')
    
    parser.add_argument('--single_prompt_mode', action='store_true',
                        help='Use single object pair prompts instead of full combinatorial (22 vs 264 prompts)')
    
    return parser.parse_args()


def get_text_encoder_type(model_run_name):
    """Auto-determine text encoder type from model name."""
    model_to_encoder = {
        "objrel_T5_DiT_B_pilot": "T5",
        "objrel_T5_DiT_mini_pilot": "T5", 
        "objrel_rndembdposemb_DiT_B_pilot": "RandomEmbeddingEncoder_wPosEmb",
        "objrel_rndembdposemb_DiT_micro_pilot": "RandomEmbeddingEncoder_wPosEmb",
        "objrel_rndembdposemb_DiT_nano_pilot": "RandomEmbeddingEncoder_wPosEmb",
        "objrel_rndembdposemb_DiT_mini_pilot": "RandomEmbeddingEncoder_wPosEmb",
        "objrel_rndemb_DiT_B_pilot": "RandomEmbeddingEncoder",
        "objrel_T5_DiT_B_pilot_WDecay": "T5",
        "objrel_T5_DiT_mini_pilot_WDecay": "T5",
    }
    return model_to_encoder.get(model_run_name)


def generate_prompt_collection(spatial_phrases, 
                               prompt_template="{color1} {shape1} is {rel_text} {color2} {shape2}",
                               color1="blue", shape1="circle", color2="red", shape2="square"):
    """Generate prompts for single object pair (from notebook)."""
    prompt_collection = []
    scene_info_collection = []
    for spatial_relationship, rel_text_collection in spatial_phrases.items():
        if spatial_relationship in ["in_front", "behind"]:
            continue
        for rel_text in rel_text_collection:
            prompt = prompt_template.format(color1=color1, shape1=shape1, rel_text=rel_text, color2=color2, shape2=shape2)
            scene_info = {
                "color1": color1,
                "shape1": shape1,
                "color2": color2,
                "shape2": shape2,
                "spatial_relationship": spatial_relationship
            }
            prompt_collection.append(prompt)
            scene_info_collection.append(scene_info)
    return prompt_collection, scene_info_collection


def generate_all_prompt_collection(spatial_phrases, 
                                   prompt_template="{color1} {shape1} is {rel_text} {color2} {shape2}"):
    """Generate all combinatorial prompts (from notebook)."""
    color_list = ['red', 'blue']
    shape_list = ['square', 'triangle', 'circle']
    prompt_collection = []
    scene_info_collection = []
    for color1, color2 in product(color_list, color_list):
        if color1 == color2:      # skip same‐color pairs
            continue
        for shape1, shape2 in product(shape_list, shape_list):
            if shape1 == shape2:
                continue
            for spatial_relationship, rel_text_collection in spatial_phrases.items():
                if spatial_relationship in ["in_front", "behind"]:
                    continue
                for rel_text in rel_text_collection:
                    prompt = prompt_template.format(color1=color1, shape1=shape1, rel_text=rel_text, color2=color2, shape2=shape2)
                    scene_info = {
                        "color1": color1,
                        "shape1": shape1,
                        "color2": color2,
                        "shape2": shape2,
                        "spatial_relationship": spatial_relationship
                    }
                    prompt_collection.append(prompt)
                    scene_info_collection.append(scene_info)
    return prompt_collection, scene_info_collection


def load_text_encoder(text_encoder_type):
    """Load the text encoder based on type."""
    T5_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"
    tokenizer = T5Tokenizer.from_pretrained(T5_path)
    
    if text_encoder_type == "T5":
        T5_dtype = torch.bfloat16
        text_encoder = T5EncoderModel.from_pretrained(T5_path, load_in_8bit=False, torch_dtype=T5_dtype)
    elif text_encoder_type == "RandomEmbeddingEncoder_wPosEmb":
        text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRel_pilot_rndemb/caption_feature_wmask'
        emb_data = torch.load(join(text_feat_dir_old, "word_embedding_dict.pt"))
        text_encoder = RandomEmbeddingEncoder_wPosEmb(
            emb_data["embedding_dict"], 
            emb_data["input_ids2dict_ids"], 
            emb_data["dict_ids2input_ids"], 
            max_seq_len=20, embed_dim=4096,
            wpe_scale=1/6).to("cuda")
    elif text_encoder_type == "RandomEmbeddingEncoder":
        text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRel_pilot_rndemb/caption_feature_wmask'
        emb_data = torch.load(join(text_feat_dir_old, "word_embedding_dict.pt"))
        text_encoder = RandomEmbeddingEncoder(
            emb_data["embedding_dict"], 
            emb_data["input_ids2dict_ids"], 
            emb_data["dict_ids2input_ids"]).to("cuda")
    else:
        raise ValueError(f"Unknown text encoder type: {text_encoder_type}")
    
    return tokenizer, text_encoder


def create_pipeline_from_config(config, tokenizer, text_encoder, weight_dtype):
    """Create the diffusion pipeline from config."""
    image_size = config.image_size
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs = {
        "window_block_indexes": config.window_block_indexes, 
        "window_size": config.window_size,
        "use_rel_pos": config.use_rel_pos, 
        "lewei_scale": config.lewei_scale, 
        'config': config,
        'model_max_length': config.model_max_length
    }
    
    model = build_model(
        config.model,
        config.grad_checkpointing,
        config.get('fp32_attention', False),
        input_size=latent_size,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        **model_kwargs
    ).train()
    
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
    
    transformer.load_state_dict(state_dict_convert(model.state_dict()))
    
    pipeline = PixArtAlphaPipeline_custom.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        transformer=transformer,
        tokenizer=tokenizer,
        text_encoder=None,
        torch_dtype=weight_dtype,
    )
    
    pipeline.to(device="cuda", dtype=weight_dtype)
    # note the text encoder SHALL NOT be moved to the same weight_dtype as the pipeline, it will cause error. 
    pipeline.text_encoder = text_encoder.to(device="cuda")
    
    return pipeline


def precompute_embeddings(prompt_collections, tokenizer, text_encoder, max_sequence_length, device="cuda"):
    """Pre-compute embeddings for all prompts and store in memory cache."""
    embedding_cache = {}
    
    print("Pre-computing text embeddings for all prompts...")
    total_prompts = sum(len(prompts) for prompts, _ in prompt_collections.values()) + 1  # +1 for uncond
    
    with torch.no_grad():
        # First, compute unconditional embedding (empty prompt)
        print("Computing unconditional embedding...")
        uncond_inputs = tokenizer(
            "", 
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Get uncond embeddings based on encoder type
        if hasattr(text_encoder, 'encode'):
            # For RandomEmbedding encoders
            uncond_embeddings, uncond_mask = text_encoder.encode(uncond_inputs.input_ids, uncond_inputs.attention_mask)
        else:
            # For T5 encoder
            uncond_outputs = text_encoder(uncond_inputs.input_ids, attention_mask=uncond_inputs.attention_mask)
            uncond_embeddings = uncond_outputs.last_hidden_state
            uncond_mask = uncond_inputs.attention_mask
        
        # Store unconditional embedding with empty key
        embedding_cache[""] = {
            'caption_embeds': uncond_embeddings.cpu(),
            'emb_mask': uncond_mask.cpu()
        }
        
        # Now compute all prompt embeddings
        pbar = tqdm(total=total_prompts, desc="Computing embeddings")
        pbar.update(1)  # Account for uncond embedding
        
        for template_name, (prompts, scene_infos) in prompt_collections.items():
            for prompt in prompts:
                # Create a unique key for this prompt
                prompt_key = f"{template_name}::{prompt}"
                
                if prompt_key not in embedding_cache:
                    # Tokenize and encode
                    inputs = tokenizer(
                        prompt, 
                        max_length=max_sequence_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Get embeddings based on encoder type
                    if hasattr(text_encoder, 'encode'):
                        # For RandomEmbedding encoders  
                        embeddings, attention_mask = text_encoder.encode(inputs.input_ids, inputs.attention_mask)
                    else:
                        # For T5 encoder
                        outputs = text_encoder(inputs.input_ids, attention_mask=inputs.attention_mask)
                        embeddings = outputs.last_hidden_state
                        attention_mask = inputs.attention_mask
                    
                    # Store in CPU memory to save GPU memory - use format matching existing system
                    embedding_cache[prompt_key] = {
                        'caption_embeds': embeddings.cpu(),
                        'emb_mask': attention_mask.cpu()
                    }
                
                pbar.update(1)
        pbar.close()
    
    print(f"Cached embeddings for {len(embedding_cache)} unique prompts (including uncond)")
    return embedding_cache


def evaluate_pipeline_on_prompts_with_cached_embeddings(pipeline, prompts, scene_infos, embedding_cache,
                                                       num_images=49, num_inference_steps=14, guidance_scale=4.5,
                                                       generator_seed=42, color_margin=25, spatial_threshold=5, 
                                                       device="cuda", weight_dtype=torch.float16):
    """
    Evaluate a diffusion pipeline on prompts using pre-computed cached embeddings.
    
    This function is similar to evaluate_pipeline_on_prompts but uses cached text embeddings
    instead of re-computing them, providing significant performance improvements.
    
    Args:
        pipeline: Diffusion pipeline for image generation
        prompts: List of text prompts to evaluate
        scene_infos: List of scene information dictionaries for evaluation
        embedding_cache: Dict containing pre-computed embeddings with keys like "template::prompt"
        num_images: Number of images to generate per prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        generator_seed: Random seed for generation
        color_margin: Color threshold for object classification
        spatial_threshold: Spatial relationship threshold in pixels
        device: Device for generation
        weight_dtype: Data type for computations
    
    Returns:
        tuple: (eval_df, object_df)
    """
    from utils.cv2_eval_utils import find_classify_objects, evaluate_parametric_relation
    
    # Validate inputs
    if len(prompts) != len(scene_infos):
        raise ValueError(f"Number of prompts ({len(prompts)}) must match number of scene_infos ({len(scene_infos)})")
    
    # Get unconditional embeddings from cache
    uncond_data = embedding_cache[""]
    uncond_prompt_embeds = uncond_data['caption_embeds'].to(device)
    uncond_prompt_attention_mask = uncond_data['emb_mask'].to(device)
    
    all_eval_results = []
    all_object_results = []
    
    for prompt_id, (prompt, scene_info) in tqdm(enumerate(zip(prompts, scene_infos)), 
                                               desc="Evaluating prompts", total=len(prompts)):
        
        # Find the cached embeddings for this prompt
        cached_data = None
        for cache_key in embedding_cache.keys():
            if cache_key != "" and cache_key.endswith(f"::{prompt}"):
                cached_data = embedding_cache[cache_key]
                break
        
        if cached_data is None:
            print(f"Warning: No cached embeddings found for prompt: '{prompt}'")
            continue
            
        # Get cached embeddings and move to GPU
        caption_embeds = cached_data['caption_embeds'].to(device)
        emb_mask = cached_data['emb_mask'].to(device)
        
        # Generate images using cached embeddings
        try:
            # Call pipeline with pre-computed embeddings (following visualize_prompts pattern)
            out = pipeline(
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images,
                generator=torch.Generator(device=device).manual_seed(generator_seed),
                guidance_scale=guidance_scale,
                prompt_embeds=caption_embeds,
                prompt_attention_mask=emb_mask,
                negative_prompt=None,
                negative_prompt_embeds=uncond_prompt_embeds,
                negative_prompt_attention_mask=uncond_prompt_attention_mask,
                use_resolution_binning=False,  # needed for smaller images
                prompt_dtype=weight_dtype, # torch.float16 (not bfloat16)
                verbose=False,
            )
            
            # Process each generated image
            for sample_id, image in enumerate(out.images):
                try:
                    # Detect and classify objects in the image
                    classified_objects_df = find_classify_objects(image)
                    
                    # Evaluate spatial relationships
                    eval_result = evaluate_parametric_relation(
                        classified_objects_df, scene_info, 
                        color_margin=color_margin, 
                        spatial_threshold=spatial_threshold
                    )
                    
                    # Create evaluation record
                    eval_record = {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "sample_id": sample_id,
                        **eval_result,
                        # **scene_info  # Add scene info columns
                    }
                    all_eval_results.append(eval_record)
                    
                    # Store object detection results
                    classified_objects_df["prompt_id"] = prompt_id
                    classified_objects_df["sample_id"] = sample_id
                    classified_objects_df["prompt"] = prompt
                    all_object_results.append(classified_objects_df)
                    
                except Exception as e:
                    print(f"Error evaluating sample {sample_id} for prompt '{prompt}': {e}")
                    continue
                    
        except Exception as e:
            print(f"Error generating images for prompt '{prompt}': {e}")
            continue
    
    # Convert to DataFrames
    eval_df = pd.DataFrame(all_eval_results)
    object_df = pd.concat(all_object_results, ignore_index=True) if all_object_results else pd.DataFrame()
    
    return eval_df, object_df


def evaluate_checkpoint_with_cache(pipeline, checkpoint_path, prompt_collections, embedding_cache, 
                                   args, eval_dir, use_ema=True):
    """Evaluate a single checkpoint using cached embeddings."""
    ckpt_name = os.path.basename(checkpoint_path)
    step_num = int(ckpt_name.split('_step_')[-1].split('.pth')[0]) if '_step_' in ckpt_name else 0
    
    print(f"\nEvaluating checkpoint: {ckpt_name}")
    
    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, weights_only=False)
    pipeline.transformer.load_state_dict(state_dict_convert(
        ckpt['state_dict_ema'] if use_ema else ckpt['state_dict']))
    pipeline.set_progress_bar_config(disable=True)
    
    all_results = []
    all_obj_results = []
    # Evaluate each template
    for template_name, (prompts, scene_infos) in prompt_collections.items():
        print(f"  Evaluating template: {template_name} ({len(prompts)} prompts)")
        
        try:
            # Use the new cached evaluation function for the entire template
            eval_df, object_df = evaluate_pipeline_on_prompts_with_cached_embeddings(
                pipeline, prompts, scene_infos, embedding_cache,
                num_images=args.num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator_seed=args.generator_seed,
                color_margin=25,
                spatial_threshold=5,
                device=pipeline.device,
                weight_dtype=pipeline.transformer.dtype
            )
            
            # Add checkpoint and template information to results
            eval_df['checkpoint'] = ckpt_name
            eval_df['step_num'] = step_num
            eval_df['ema'] = use_ema
            eval_df['template'] = template_name
            template_results = eval_df  # Put in list format for consistency
            
        except Exception as e:
            print(f"Error evaluating template '{template_name}': {e}")
            template_results = []
            continue
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Save template results
        template_df = eval_df # pd.concat(template_results, ignore_index=True)
        template_file = f"eval_df_{ckpt_name}{'_ema' if use_ema else '_model'}_{template_name.replace(' ', '_')}.csv"
        template_df.to_csv(join(eval_dir, template_file), index=False)
        
        template_object_file = f"object_df_{ckpt_name}{'_ema' if use_ema else '_model'}_{template_name.replace(' ', '_')}.pkl"
        object_df.to_pickle(join(eval_dir, template_object_file))
        all_results.append(template_df)
        all_obj_results.append(object_df)
    
    # Save combined results for this checkpoint
    checkpoint_df = pd.concat(all_results, ignore_index=True)
    checkpoint_filename = f"eval_df_{ckpt_name}{'_ema' if use_ema else '_model'}_all_templates.csv"
    checkpoint_df.to_csv(join(eval_dir, checkpoint_filename), index=False)
    checkpoint_object_filename = f"object_df_{ckpt_name}{'_ema' if use_ema else '_model'}_all_templates.pkl"
    checkpoint_object_df = pd.concat(all_obj_results, ignore_index=True)
    checkpoint_object_df.to_pickle(join(eval_dir, checkpoint_object_filename))
    
    return checkpoint_df, checkpoint_object_df


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    # Auto-determine text encoder type if not specified
    text_encoder_type = args.text_encoder_type or get_text_encoder_type(args.model_run_name)
    if not text_encoder_type:
        raise ValueError(f"Could not determine text encoder type for model: {args.model_run_name}")
    
    print(f"Evaluating model: {args.model_run_name}")
    print(f"Text encoder type: {text_encoder_type}")
    print(f"Prompt templates: {len(args.prompt_templates)}")
    print(f"Single prompt mode: {args.single_prompt_mode}")
    
    # Setup directories
    savedir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/{args.model_run_name}"
    if args.output_dir:
        eval_dir = args.output_dir
    else:
        eval_dir = join(savedir, "generalization_eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"Output directory: {eval_dir}")
    
    # Step 1: Load text encoder
    print("\nStep 1: Loading text encoder...")
    tokenizer, text_encoder = load_text_encoder(text_encoder_type)
    
    # Step 2: Generate all prompt collections
    print("\nStep 2: Generating prompt collections...")
    dataset_tmp = ShapesDataset(num_images=10000)
    
    prompt_collections = {}
    
    for template in args.prompt_templates:
        template_name = template.replace('{', '').replace('}', '').replace(' ', '_')
        
        if args.single_prompt_mode:
            # Single object pair (blue circle, red square)
            prompts, scene_infos = generate_prompt_collection(
                dataset_tmp.spatial_phrases,
                prompt_template=template,
                color1="blue", shape1="circle", color2="red", shape2="square"
            )
        else:
            # Full combinatorial
            prompts, scene_infos = generate_all_prompt_collection(
                dataset_tmp.spatial_phrases,
                prompt_template=template
            )
        
        prompt_collections[template_name] = (prompts, scene_infos)
        pkl.dump((prompts, scene_infos), open(join(eval_dir, f"prompt_collections_{template_name}.pkl"), "wb"))
        print(f"  Template '{template}': {len(prompts)} prompts")
    
    # Step 3: Pre-compute embeddings
    print(f"\nStep 3: Pre-computing embeddings...")
    text_encoder = text_encoder.to(device="cuda")
    embedding_cache = precompute_embeddings(
        prompt_collections, tokenizer, text_encoder, args.max_sequence_length
    )
    
    # Step 4: Setup pipeline
    print(f"\nStep 4: Setting up pipeline...")
    config = read_config(join(savedir, 'config.py'))
    weight_dtype = torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    pipeline = create_pipeline_from_config(config, tokenizer, text_encoder, weight_dtype)
    
    # Step 5: Get checkpoints to evaluate
    ckptdir = join(savedir, "checkpoints")
    if args.checkpoints:
        # Use specified checkpoints
        checkpoint_paths = [join(ckptdir, ckpt) for ckpt in args.checkpoints]
        # Verify they exist
        checkpoint_paths = [p for p in checkpoint_paths if os.path.exists(p)]
    else:
        # Use all available checkpoints
        checkpoint_paths = sorted(
            glob.glob(join(ckptdir, "*.pth")), 
            key=lambda x: int(os.path.basename(x).split('_step_')[-1].split('.pth')[0]) 
            if '_step_' in os.path.basename(x) else 0
        )
    
    print(f"\nStep 5: Found {len(checkpoint_paths)} checkpoints to evaluate")
    
    # Step 6: Evaluate each checkpoint
    all_checkpoint_results = []
    for checkpoint_path in checkpoint_paths:
        try:
            checkpoint_df, checkpoint_object_df = evaluate_checkpoint_with_cache(
                pipeline, checkpoint_path, prompt_collections, 
                embedding_cache, args, eval_dir, use_ema=True
            )
            all_checkpoint_results.append(checkpoint_df)

            print(f"Checkpoint summary: {checkpoint_path}")
            summary_by_template = checkpoint_df.groupby('template')[['overall', 'shape', 'color', 'unique_binding', 'spatial_relationship', 'spatial_relationship_loose', 'Dx', 'Dy']].mean()
            print(summary_by_template)
        except Exception as e:
            print(f"Error evaluating checkpoint {checkpoint_path}: {e}")
            continue
    
    # Step 7: Generate final summary
    if not all_checkpoint_results:
        print("No successful evaluations completed.")
    else:
        print(f"\nStep 7: Generating summary...")
        # Combine all results
        final_df = pd.concat(all_checkpoint_results, ignore_index=True)
        final_df.to_csv(join(eval_dir, "eval_df_all_checkpoints_all_templates.csv"), index=False)
        # Create summary across checkpoints and templates
        summary_df = final_df.groupby(['checkpoint', 'template']).agg({
            'overall': 'mean',
            'shape': 'mean', 
            'color': 'mean',
            'unique_binding': 'mean',
            'spatial_relationship': 'mean',
            'spatial_relationship_loose': 'mean',
            'Dx': 'mean',
            'Dy': 'mean'
        }).reset_index()
        
        summary_df.to_csv(join(eval_dir, "summary_across_checkpoints_templates.csv"), index=False)
        
        print(f"\nEvaluation complete! Results saved to: {eval_dir}")
        print(f"Total samples evaluated: {len(final_df)}")
        print("\nOverall summary:")
        for metric in ['overall', 'shape', 'color', 'spatial_relationship']:
            if metric in final_df.columns:
                mean_val = final_df[metric].mean()
                print(f"  {metric}: {mean_val:.3f}")
    
    print("\nDone!")
    print(f"Total time: {time.time() - start_time:.2f} seconds")