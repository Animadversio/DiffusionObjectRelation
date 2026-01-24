#!/usr/bin/env python3
"""
Cluster-compatible version of the training trajectory evaluation script.
Can run single model or parallel processing.
"""

import argparse
import os
import sys
import glob
from os.path import join
import torch
import torch as th
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial

# Add paths
sys.path.append("/n/home12/hjkim/Github/DiffusionObjectRelation/PixArt-alpha")
sys.path.append("/n/home12/hjkim/Github/DiffusionObjectRelation")

# Import required modules
from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler
from transformers import T5Tokenizer, T5EncoderModel

from utils.pixart_sampling_utils import pipeline_inference_custom, PixArtAlphaPipeline_custom
from utils.pixart_utils import state_dict_convert
from utils.text_encoder_control_lib import RandomEmbeddingEncoder_wPosEmb, RandomEmbeddingEncoder
from utils.image_utils import pil_images_to_grid
from utils.attention_map_store_utils import replace_attn_processor, AttnProcessor2_0_Store, PixArtAttentionVisualizer_Store
from utils.cv2_eval_utils import find_classify_object_masks, evaluate_parametric_relation
from utils.cv2_eval_utils import find_classify_objects, evaluate_parametric_relation, eval_func_factory, scene_info_collection
from utils.attention_analysis_lib import plot_attention_layer_head_heatmaps, plot_layer_head_score_summary
from utils.attention_analysis_lib import *
from utils.obj_mask_utils import *

try:
    from circuit_toolkit.plot_utils import saveallforms
except ImportError:
    def saveallforms(figdir, fname, fig=plt.gcf(), formats=["png", "pdf", "svg"]):
        import os
        os.makedirs(figdir, exist_ok=True)
        for fmt in formats:
            fig.savefig(os.path.join(figdir, f"{fname}.{fmt}"), 
                       bbox_inches='tight', dpi=300)

# Model configurations
model_to_encoder = {
    "objrel_singleobj_T5_mini_pilot1": "T5",
    "objrel_singleobj_mini_rndemb": "RndEmb",
    "objrel_singleobj_mini_rndembpos": "RndEmbPos",
    "objrel_doubleobj_T5_mini_pilot1": "T5",
    "objrel_doubleobj_mini_rndemb": "RndEmb",
    "objrel_doubleobj_mini_rndembpos": "RndEmbPos",
    "objrel_mixedobj_T5_mini_pilot1": "T5",
    "objrel_mixedobj_mini_rndemb": "RndEmb",
    "objrel_mixedobj_mini_rndembpos": "RndEmbPos",
}

model_run_names = [       
    "objrel_mixedobj_T5_mini_pilot1",
    "objrel_mixedobj_mini_rndemb",
    "objrel_mixedobj_mini_rndembpos",
    "objrel_doubleobj_T5_mini_pilot1",
    "objrel_doubleobj_mini_rndemb",
    "objrel_doubleobj_mini_rndembpos",
    "objrel_singleobj_T5_mini_pilot1",
    "objrel_singleobj_mini_rndemb",
    "objrel_singleobj_mini_rndembpos"
]

mapping2spatial_relationship = {
    "above": "above",
    "below": "below",
    "left": "left",
    "right": "right",
    "above_left": "upper_left",
    "above_right": "upper_right",
    "below_left": "lower_left",
    "below_right": "lower_right"
}

def generate_test_prompts_collection_and_parsed_words():
    """Generate test prompts for evaluation."""
    from itertools import product
    
    colors = ['red', 'blue']
    target_shapes = ['square', 'triangle', 'circle']
    verticals = ['above', 'below']
    horizontals = ['to the left of', 'to the right of']
    prompts = []
    parsed_words = []
    for c1, c2 in product(colors, colors):
        if c1 == c2:      # skip same‐color pairs
            continue
        for shape1, shape2 in product(target_shapes, target_shapes):
            if shape1 == shape2:
                continue
            for v in verticals:
                prompts.append(f"{c1} {shape1} is {v} the {c2} {shape2}")
                parsed_words.append({"color1": c1, "shape1": shape1, "relation": [v], "color2": c2, "shape2": shape2, "prop": ["is", "the"], "prompt": prompts[-1]})
            for h in horizontals:
                prompts.append(f"{c1} {shape1} is {h} the {c2} {shape2}")
                parsed_words.append({"color1": c1, "shape1": shape1, "relation": [h.split(" ")[2]], "color2": c2, "shape2": shape2, "prop": ["is", "the", "to", "of"], "prompt": prompts[-1]})
            for v, h in product(verticals, horizontals):
                prompts.append(f"{c1} {shape1} is {v} and {h} the {c2} {shape2}")
                parsed_words.append({"color1": c1, "shape1": shape1, "relation": [v, h.split(" ")[2]], "color2": c2, "shape2": shape2, "prop": ["is", "the", "to", "of", "and"], "prompt": prompts[-1]})
    return prompts, parsed_words

def load_text_encoder(text_encoder_type):
    """Load text encoder based on type."""
    T5_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"
    tokenizer = T5Tokenizer.from_pretrained(T5_path)
    
    if text_encoder_type == "T5":
        T5_dtype = torch.bfloat16
        text_encoder = T5EncoderModel.from_pretrained(T5_path, load_in_8bit=False, torch_dtype=T5_dtype)
    elif text_encoder_type == "RndEmb":
        text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRel_pilot_rndemb/caption_feature_wmask'
        emb_data = th.load(join(text_feat_dir_old, "word_embedding_dict.pt")) 
        text_encoder = RandomEmbeddingEncoder_wPosEmb(emb_data["embedding_dict"], 
                                                    emb_data["input_ids2dict_ids"], 
                                                    emb_data["dict_ids2input_ids"], 
                                                    max_seq_len=20, embed_dim=4096,
                                                    wpe_scale=1/6).to("cuda")
    elif text_encoder_type == "RndEmbPos":
        text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRel_pilot_rndemb/caption_feature_wmask'
        emb_data = th.load(join(text_feat_dir_old, "word_embedding_dict.pt")) 
        text_encoder = RandomEmbeddingEncoder(emb_data["embedding_dict"], 
                                                    emb_data["input_ids2dict_ids"], 
                                                    emb_data["dict_ids2input_ids"], 
                                                    ).to("cuda")
    return text_encoder

def create_pipeline(savedir, text_encoder):
    """Create the diffusion pipeline."""
    config = read_config(join(savedir, 'config.py'))
    weight_dtype = torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    image_size = config.image_size
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                    "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                    'model_max_length': config.model_max_length}
    
    model = build_model(config.model,
                    config.grad_checkpointing,
                    config.get('fp32_attention', False),
                    input_size=latent_size,
                    learn_sigma=learn_sigma,
                    pred_sigma=pred_sigma,
                    **model_kwargs).train()
    
    transformer = Transformer2DModel(
        num_attention_heads=config.num_attention_heads,
        attention_head_dim=config.attention_head_dim,
        in_channels=config.in_channels,
        num_layers=len(model.blocks),
        dropout=config.dropout,
        norm_num_groups=config.norm_num_groups,
        cross_attention_dim=config.cross_attention_dim,
        attention_bias=config.attention_bias,
        sample_size=latent_size,
        num_vector_embeds=config.num_vector_embeds,
        patch_size=config.patch_size,
        activation_fn=config.activation_fn,
        num_embeds_ada_norm=config.num_embeds_ada_norm,
        use_linear_projection=config.use_linear_projection,
        only_cross_attention=config.only_cross_attention,
        double_self_attention=config.double_self_attention,
        upcast_attention=config.upcast_attention,
        norm_type=config.norm_type,
        norm_elementwise_affine=config.norm_elementwise_affine,
        norm_eps=config.norm_eps,
        attention_type=config.attention_type,
        caption_channels=config.caption_channels,
        enable_freeu=config.enable_freeu,
        freeu_config=config.freeu_config,
        window_block_indexes=config.window_block_indexes,
        window_size=config.window_size,
        use_rel_pos=config.use_rel_pos,
        lewei_scale=config.lewei_scale,
    )
    
    # Load weights from the trained model
    transformer.load_state_dict(model.state_dict())
    transformer = transformer.to(device="cuda", dtype=weight_dtype)
    
    # Create pipeline
    pipeline = PixArtAlphaPipeline_custom(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
        scheduler=DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler"),
    )
    
    return pipeline

def eval_train_traj(savedir, pipeline, model_run_name):
    """Evaluate training trajectory for a single model."""
    eval_df_all_traj = []
    object_df_all_traj = []
    
    ckptdir = join(savedir, "checkpoints")
    ckpt_all = sorted(glob.glob(join(ckptdir, "*.pth")), 
                    key=lambda x: int(os.path.basename(x).split('_step_')[-1].split('.pth')[0]))

    # Generate prompts
    prompts, parsed_words = generate_test_prompts_collection_and_parsed_words()
    prompt_df = pd.DataFrame(parsed_words)
    prompt_df["relation_str"] = prompt_df["relation"].apply(lambda x: "_".join(x))
    
    # Create eval directory
    eval_dir = join(savedir, "large_scale_eval_posthoc")
    os.makedirs(eval_dir, exist_ok=True)

    for ckpt_path in ckpt_all:
        ckpt_name = os.path.basename(ckpt_path)
        step_num = int(ckpt_name.split('_step_')[-1].split('.pth')[0])
        
        for ckpt_ver in ["ema", "model"]:
            print(f"loading ckpt step {model_run_name} {step_num} {ckpt_ver} |  {ckpt_path} ...")
            ckpt = torch.load(ckpt_path)
            if ckpt_ver == "ema":
                pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict_ema']))
            else:
                pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))

            eval_df_all = []
            object_df_all = []
            
            for prompt_id, row in tqdm(prompt_df.iterrows(), total=len(prompt_df), desc=f"Step {step_num} {ckpt_ver}"):
                prompt = row["prompt"]
                scene_info = {
                    "color1": row["color1"],
                    "shape1": row["shape1"],
                    "color2": row["color2"],
                    "shape2": row["shape2"],
                    "spatial_relationship": mapping2spatial_relationship[row["relation_str"]]
                }
                
                out = pipeline(prompt, num_inference_steps=14, guidance_scale=4.5, max_sequence_length=20, 
                    num_images_per_prompt=100, generator=th.Generator(device="cuda").manual_seed(42), prompt_dtype=torch.float16)
                
                object_df_col = []
                eval_score = []
                
                for si, image_sample in enumerate(out.images):
                    classified_objects_df = find_classify_objects(image_sample)
                    classified_objects_df["sample_id"] = si
                    object_df_col.append(classified_objects_df)
                    eval_result = evaluate_parametric_relation(classified_objects_df, scene_info, color_margin=25, spatial_threshold=5)
                    
                    eval_score.append({
                        "step_num": step_num,
                        "ckpt_ver": ckpt_ver,
                        "sample_id": si,
                        "prompt": prompt,
                        "prompt_id": prompt_id,
                        "overall": eval_result["overall"],
                        "overall_loose": eval_result["overall_loose"],
                        "shape": eval_result["shape"],
                        "color": eval_result["color"],
                        "exist_binding": eval_result["exist_binding"],
                        "unique_binding": eval_result["unique_binding"],
                        "spatial_relationship": eval_result["spatial_relationship"],
                        "spatial_relationship_loose": eval_result["spatial_relationship_loose"],
                        "Dx": eval_result["Dx"],
                        "Dy": eval_result["Dy"],
                        "x1": eval_result["x1"],
                        "y1": eval_result["y1"],
                        "x2": eval_result["x2"],
                        "y2": eval_result["y2"],
                    })
                
                eval_df = pd.DataFrame(eval_score)
                object_df = pd.concat(object_df_col)
                
                # Print progress
                acc_overall = eval_df["overall"].mean()
                print(f"prompt: {prompt} (id: {prompt_id}) - overall acc: {acc_overall:.2f}")
                
                eval_df_all.append(eval_df)
                object_df_all.append(object_df)
                torch.cuda.empty_cache()
                
            eval_df_all = pd.concat(eval_df_all)
            object_df_all = pd.concat(object_df_all)
            eval_df_all.to_csv(join(eval_dir, f"eval_df_all_train_step{step_num}_{ckpt_ver}_prompts.csv"), index=False)
            object_df_all.to_pickle(join(eval_dir, f"object_df_all_train_step{step_num}_{ckpt_ver}_prompts.pkl"))
            eval_df_all_traj.append(eval_df_all)
            object_df_all_traj.append(object_df_all)
            torch.cuda.empty_cache()

    eval_df_all_traj = pd.concat(eval_df_all_traj)
    object_df_all_traj = pd.concat(object_df_all_traj)
    eval_df_all_traj.to_csv(join(eval_dir, f"eval_df_all_train_traj_prompts.csv"), index=False)
    object_df_all_traj.to_pickle(join(eval_dir, f"object_df_all_train_traj_prompts.pkl"))
    torch.cuda.empty_cache()
    
    return eval_df_all_traj, object_df_all_traj

def process_single_model(model_run_name):
    """Process a single model - designed for parallel execution."""
    print(f"Starting evaluation for {model_run_name}")
    
    suffix = "hk"
    savedir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/hannah_results/{model_run_name}"
    
    # Check if model directory exists
    if not os.path.exists(savedir):
        print(f"Model directory not found: {savedir}")
        return None
    
    text_encoder_type = model_to_encoder[model_run_name]
    print(f"text_encoder_type: {text_encoder_type}")

    # Load the model and evaluate
    text_encoder = load_text_encoder(text_encoder_type)
    pipeline = create_pipeline(savedir, text_encoder)
    eval_df_all_traj, object_df_all_traj = eval_train_traj(savedir, pipeline, model_run_name)
    
    print(f"Completed evaluation for {model_run_name}")
    return model_run_name

def main():
    parser = argparse.ArgumentParser(description='Post-hoc generation training trajectory evaluation')
    parser.add_argument('--model_run_name', type=str, 
                        choices=model_run_names,
                        help='Model run name to evaluate (if not specified, runs all models)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run models in parallel (requires multiple GPUs)')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='Number of parallel workers (default: 3)')
    
    args = parser.parse_args()
    
    if args.model_run_name:
        # Run single model
        process_single_model(args.model_run_name)
    else:
        # Run all models
        if args.parallel:
            print(f"Running {len(model_run_names)} models in parallel with {args.num_workers} workers")
            with mp.Pool(processes=args.num_workers) as pool:
                results = pool.map(process_single_model, model_run_names)
        else:
            print(f"Running {len(model_run_names)} models sequentially")
            for model_run_name in model_run_names:
                process_single_model(model_run_name)

if __name__ == "__main__":
    main() 