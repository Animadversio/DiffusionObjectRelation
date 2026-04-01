"""
Unified script for cross-attention head alignment score synopsis.

Supports three text encoder types: T5, CLIP, and RTE (RandomEmbeddingEncoder_wPosEmb).
For T5 and CLIP, spatial direction vectors are obtained via variance partition of word embeddings.
For RTE, raw encoded relation words are used directly.

Usage:
    python experimental_scripts/head_alignment_synopsis.py \
        --model_run_name objrel_T5_DiT_B_pilot \
        --text_encoder_type T5 \
        --ckpt_name epoch_4000_step_160000.pth \
        --figdir Figures/DiT_head_alignment

    python experimental_scripts/head_alignment_synopsis.py \
        --model_run_name objrel_rndembdposemb_DiT_B_pilot \
        --text_encoder_type RTE \
        --ckpt_name epoch_4000_step_160000.pth

    python experimental_scripts/head_alignment_synopsis.py \
        --model_run_name objrel_CLIPemb_DiT_mini_pilot \
        --text_encoder_type CLIP \
        --ckpt_name epoch_4000_step_160000.pth
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join
from itertools import product
from tqdm import tqdm

sys.path.append(join(os.path.dirname(__file__), "..", "PixArt-alpha"))
sys.path.append(join(os.path.dirname(__file__), ".."))

from diffusion.model.builder import build_model
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed
from diffusion.utils.misc import read_config
from diffusers import Transformer2DModel
from utils.pixart_sampling_utils import PixArtAlphaPipeline_custom, PixArtAlphaPipeline_custom_CLIP
from utils.pixart_utils import state_dict_convert, construct_diffuser_pipeline_from_config, PixArt_model_configs
from circuit_toolkit.plot_utils import saveallforms


# ---------------------------------------------------------------------------
# Alignment metrics
# ---------------------------------------------------------------------------

DIRECTION_VECTOR = {
    "above": [0, -1], "below": [0, 1],
    "left": [-1, 0], "right": [1, 0],
    "upper_left": [-1, -1], "upper_right": [1, -1],
    "lower_left": [-1, 1], "lower_right": [1, 1],
}


def ramp_alignment_metrics(M, dvec, eps=1e-12):
    H, W = M.shape
    d = np.array(dvec, dtype=float)
    d = d / np.linalg.norm(d)
    xs = np.linspace(-1, 1, W)
    ys = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(xs, ys)
    T = d[0] * X + d[1] * Y
    T = T - T.mean()
    T_norm = np.linalg.norm(T) + eps
    A = M - M.mean()
    A_norm = np.linalg.norm(A) + eps
    projection = np.sum(A * T / T_norm)
    cosine = projection / A_norm
    return dict(cosine=cosine, projection=projection, energy=A_norm, template_norm=T_norm)


def eval_ramp_alignment_all(inner_prod_mat, spatial_relationship_levels, eps=1e-12):
    align_dict_list = []
    for i_word in range(len(spatial_relationship_levels)):
        rel_name = spatial_relationship_levels[i_word]
        dvec = DIRECTION_VECTOR[rel_name]
        align_dict = ramp_alignment_metrics(
            inner_prod_mat[:, i_word:i_word+1].view(8, 8).detach().cpu().numpy(), dvec, eps=eps
        )
        align_dict["rel_name"] = rel_name
        align_dict["dvec"] = (dvec[0], dvec[1])
        align_dict["dir_idx"] = i_word
        align_dict_list.append(align_dict)
    return pd.DataFrame(align_dict_list)


# ---------------------------------------------------------------------------
# Prompt generation and word-vector extraction (for T5 / CLIP)
# ---------------------------------------------------------------------------

def generate_all_prompt_collection(spatial_phrases,
                                   prompt_template="{color1} {shape1} is {rel_text} {color2} {shape2}"):
    color_list = ["red", "blue"]
    shape_list = ["square", "triangle", "circle"]
    prompt_collection, scene_info_collection = [], []
    for color1, color2 in product(color_list, color_list):
        if color1 == color2:
            continue
        for shape1, shape2 in product(shape_list, shape_list):
            if shape1 == shape2:
                continue
            for spatial_relationship, rel_text_collection in spatial_phrases.items():
                if spatial_relationship in ["in_front", "behind"]:
                    continue
                for rel_text in rel_text_collection:
                    prompt = prompt_template.format(
                        color1=color1, shape1=shape1, rel_text=rel_text, color2=color2, shape2=shape2
                    )
                    scene_info_collection.append(dict(
                        color1=color1, shape1=shape1, color2=color2, shape2=shape2,
                        spatial_relationship=spatial_relationship,
                    ))
                    prompt_collection.append(prompt)
    return prompt_collection, scene_info_collection


def find_shape_index(tokens, shape):
    shape_clean = shape.strip().lower()
    for i, token in enumerate(tokens):
        tc = token.strip().lower()
        if tc == shape_clean or tc == f"\u2581{shape_clean}":
            return i
    for i, token in enumerate(tokens):
        if shape_clean in token.strip().lower():
            return i
    return None


def extract_wordvecs_and_project(pipeline, prompt_collection, scene_info_collection,
                                 max_sequence_length=20, device="cuda"):
    """Extract per-token word vectors at shape1/shape2 positions and project through caption_projection."""
    from experimental_scripts.generalization_profile_eval_cli import precompute_embeddings

    embedding_cache = precompute_embeddings(
        {"base": (prompt_collection, scene_info_collection)},
        pipeline.tokenizer, pipeline.text_encoder,
        max_sequence_length=max_sequence_length, device=device,
    )
    df = pd.DataFrame(scene_info_collection)
    df["prompt"] = prompt_collection

    wordvec_obj1_col, wordvec_obj2_col = [], []
    for row in df.itertuples():
        tokenized = pipeline.tokenizer(
            row.prompt, max_length=max_sequence_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        token_ids = tokenized["input_ids"][0]
        tokens = [pipeline.tokenizer.decode([tid]) for tid in token_ids]
        s1_idx = find_shape_index(tokens, row.shape1)
        s2_idx = find_shape_index(tokens, row.shape2)
        cap_embeds = embedding_cache[f"base::{row.prompt}"]["caption_embeds"]
        if s1_idx is None or s2_idx is None:
            raise ValueError(f"Shape index not found for prompt: {row.prompt}")
        wordvec_obj1_col.append(cap_embeds[0, s1_idx, :])
        wordvec_obj2_col.append(cap_embeds[0, s2_idx, :])

    wordvec_obj1_mat = torch.stack(wordvec_obj1_col, dim=0)
    wordvec_obj2_mat = torch.stack(wordvec_obj2_col, dim=0)
    with torch.no_grad():
        wordvec_obj1_mat_proj = pipeline.transformer.caption_projection(wordvec_obj1_mat.half().to(device))
        wordvec_obj2_mat_proj = pipeline.transformer.caption_projection(wordvec_obj2_mat.half().to(device))
    return df, wordvec_obj1_mat, wordvec_obj2_mat, wordvec_obj1_mat_proj, wordvec_obj2_mat_proj


def compute_variance_partition_direction_vecs(wordvec_mat_proj, scene_info_df, object_position="shape2"):
    """Run variance partition and return spatial relationship effect vectors and level names."""
    from utils.variance_partition_with_effects import variance_partition_with_effects

    scene_info_df = scene_info_df.copy()
    scene_info_df["color1shape1"] = scene_info_df["color1"] + "_" + scene_info_df["shape1"]
    scene_info_df["color2shape2"] = scene_info_df["color2"] + "_" + scene_info_df["shape2"]

    if object_position == "shape2":
        factors = {
            "spatial_relationship": scene_info_df["spatial_relationship"],
            "shape1": scene_info_df["shape1"],
            "color2shape2": scene_info_df["color2shape2"],
        }
    else:  # shape1
        factors = {
            "spatial_relationship": scene_info_df["spatial_relationship"],
            "shape2": scene_info_df["shape2"],
            "color1shape1": scene_info_df["color1shape1"],
        }

    var_part_df, intercept, effect_vecs, levels_map, R2_total = variance_partition_with_effects(
        wordvec_mat_proj.float().cpu().numpy(), factors, metric="euclidean", n_perm=100,
    )
    print(f"Variance partition (object_position={object_position}):")
    print(var_part_df)
    return effect_vecs["spatial_relationship"], levels_map["spatial_relationship"]


# ---------------------------------------------------------------------------
# Head screening loop
# ---------------------------------------------------------------------------

def screen_all_heads(pipeline, direction_vecs, spatial_relationship_levels,
                     hidden_size, head_num, layer_num, base_size, device="cuda",
                     direction_vecs_are_numpy=True):
    """
    For each (layer, head), compute Q(pos_embed) @ K(direction_vecs)^T and evaluate ramp alignment.

    direction_vecs: numpy array (n_dirs, embed_dim) or torch tensor (1, n_dirs, embed_dim)
    """
    head_dim = hidden_size // head_num
    align_df_list = []
    for layer_idx, head_idx in tqdm(list(product(range(layer_num), range(head_num)))):
        if direction_vecs_are_numpy:
            target = torch.from_numpy(direction_vecs)[None, :].half().to(device)
        else:
            target = direction_vecs.half().to(device)
            # RTE (and any 2D tensor): add batch dim so to_k matches T5/CLIP (B, S, D).
            if target.dim() == 2:
                target = target.unsqueeze(0)

        word_embed_2k = pipeline.transformer.transformer_blocks[layer_idx].attn2.to_k(target)
        word_embed_2k_h = word_embed_2k[0, :, head_idx * head_dim:(head_idx + 1) * head_dim]

        pos_embed = get_2d_sincos_pos_embed(hidden_size, (base_size, base_size), base_size=base_size)
        pos_embed = torch.from_numpy(pos_embed).unsqueeze(0).to(torch.float32).to(device)
        pos_embed_2q = pipeline.transformer.transformer_blocks[layer_idx].attn2.to_q(pos_embed.half())
        pos_embed_2q_h = pos_embed_2q[0, :, head_idx * head_dim:(head_idx + 1) * head_dim]

        inner_prod_mat = pos_embed_2q_h @ word_embed_2k_h.T
        align_df = eval_ramp_alignment_all(inner_prod_mat.float(), spatial_relationship_levels)
        align_df["layer_idx"] = layer_idx
        align_df["head_idx"] = head_idx
        align_df_list.append(align_df)
    return pd.concat(align_df_list)


# ---------------------------------------------------------------------------
# Synopsis plotting
# ---------------------------------------------------------------------------

def plot_synopsis(align_df_allheads, model_run_name, figdir, save_suffix="",
                  clims=None, fmts=None, figsize_per_metric=(5, 4)):
    """Plot the heatmap synopsis figure and save."""
    head_align_synopsis = align_df_allheads.groupby(["layer_idx", "head_idx"]).mean(numeric_only=True)
    metrics = ["cosine", "projection", "energy"]
    n_metrics = len(metrics)

    if clims is None:
        clims = [None] * n_metrics
    if fmts is None:
        fmts = [".2f"] * n_metrics

    fw, fh = figsize_per_metric
    fig, axes = plt.subplots(1, n_metrics, figsize=(fw * n_metrics, fh), squeeze=False)
    for idx, (metric, clim, fmt) in enumerate(zip(metrics, clims, fmts)):
        heatmap_data = head_align_synopsis[metric].unstack(level="head_idx")
        ax = axes[0, idx]
        kw = dict(annot=True, fmt=fmt, cmap="coolwarm", ax=ax, cbar=True)
        if clim is not None:
            kw["vmin"], kw["vmax"] = clim
        sns.heatmap(heatmap_data, **kw)
        ax.set_aspect("equal")
        ax.set_title(metric, fontsize=14)
        ax.set_xlabel("head index")
        ax.set_ylabel("layer index")

    plt.suptitle(f"{model_run_name} all heads alignment score synopsis | {save_suffix}")
    plt.tight_layout()
    saveallforms(figdir, f"{model_run_name}_all_heads_align_score_synopsis{save_suffix}")
    print(f"Saved synopsis to {figdir}/{model_run_name}_all_heads_align_score_synopsis{save_suffix}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline construction helpers
# ---------------------------------------------------------------------------

def build_pipeline_manual(config, weight_dtype, pipeline_class=PixArtAlphaPipeline_custom):
    """Build pipeline manually from config (used for CLIP and RTE)."""
    image_size = config.image_size
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, "pred_sigma", True)
    learn_sigma = getattr(config, "learn_sigma", True) and pred_sigma
    model_kwargs = {
        "window_block_indexes": config.window_block_indexes,
        "window_size": config.window_size,
        "use_rel_pos": config.use_rel_pos,
        "lewei_scale": config.lewei_scale,
        "config": config,
        "model_max_length": config.model_max_length,
        "caption_channels": getattr(config, "caption_channels", 4096),
    }
    model = build_model(
        config.model, config.grad_checkpointing, config.get("fp32_attention", False),
        input_size=latent_size, learn_sigma=learn_sigma, pred_sigma=pred_sigma, **model_kwargs,
    ).train()
    transformer = Transformer2DModel(
        sample_size=image_size // 8,
        num_layers=len(model.blocks),
        attention_head_dim=model.blocks[0].hidden_size // model.num_heads,
        in_channels=model.in_channels, out_channels=model.out_channels,
        patch_size=model.patch_size, attention_bias=True,
        num_attention_heads=model.num_heads,
        cross_attention_dim=model.blocks[0].hidden_size,
        activation_fn="gelu-approximate", num_embeds_ada_norm=1000,
        norm_type="ada_norm_single", norm_elementwise_affine=False, norm_eps=1e-6,
        caption_channels=getattr(config, "caption_channels", 4096),
    )
    transformer.load_state_dict(state_dict_convert(model.state_dict()))
    pipeline = pipeline_class.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        transformer=transformer, tokenizer=None, text_encoder=None,
        torch_dtype=weight_dtype,
    )
    return pipeline


def load_checkpoint(pipeline, ckpt_path, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "state_dict_ema" if use_ema and "state_dict_ema" in ckpt else "state_dict"
    pipeline.transformer.load_state_dict(state_dict_convert(ckpt[key]))
    print(f"Loaded checkpoint from {ckpt_path} (key={key})")


def get_model_dims(pipeline, config=None):
    """Get hidden_size, head_num, layer_num, base_size from pipeline or config."""
    # Try pipeline.transformer.config first (most reliable)
    tc = pipeline.transformer.config
    hidden_size = tc.cross_attention_dim
    head_num = tc.num_attention_heads
    layer_num = tc.num_layers
    base_size = tc.sample_size // tc.patch_size
    return hidden_size, head_num, layer_num, base_size


# ---------------------------------------------------------------------------
# Main routines per encoder type
# ---------------------------------------------------------------------------

def run_T5(args):
    from transformers import T5Tokenizer, T5EncoderModel
    from utils.relation_shape_dataset_lib import ShapesDataset

    T5_path = args.t5_path
    savedir = join(args.results_root, args.model_run_name)
    config = read_config(join(savedir, "config.py"))

    # Build pipeline
    pipeline = construct_diffuser_pipeline_from_config(config, pipeline_class=PixArtAlphaPipeline_custom)
    tokenizer = T5Tokenizer.from_pretrained(T5_path)
    text_encoder = T5EncoderModel.from_pretrained(T5_path, load_in_8bit=False, torch_dtype=torch.bfloat16)

    # Load checkpoint
    load_checkpoint(pipeline, join(savedir, "checkpoints", args.ckpt_name), use_ema=args.use_ema)
    pipeline.tokenizer = tokenizer
    pipeline.to(device="cuda", dtype=torch.float16)
    pipeline.text_encoder = text_encoder.to(device="cuda")

    # Extract word vectors and run variance partition
    dataset_tmp = ShapesDataset(num_images=10000)
    prompt_collection, scene_info_collection = generate_all_prompt_collection(dataset_tmp.spatial_phrases)
    df, wv1, wv2, wv1_proj, wv2_proj = extract_wordvecs_and_project(
        pipeline, prompt_collection, scene_info_collection, device="cuda"
    )

    hidden_size, head_num, layer_num, base_size = get_model_dims(pipeline, config)
    os.makedirs(args.figdir, exist_ok=True)

    for obj_pos, wv_proj, suffix in [("shape2", wv2_proj, "_shape2_MLP_proj_rel_factor"),
                                      ("shape1", wv1_proj, "_shape1_MLP_proj_rel_factor")]:
        direction_vecs, level_names = compute_variance_partition_direction_vecs(wv_proj, df, object_position=obj_pos)
        align_df = screen_all_heads(pipeline, direction_vecs, level_names,
                                    hidden_size, head_num, layer_num, base_size)
        csv_path = join(args.figdir, f"{args.model_run_name}_align_score_allheads{suffix}.csv")
        align_df.to_csv(csv_path, index=False)
        print(f"Saved alignment scores to {csv_path}")
        plot_synopsis(align_df, args.model_run_name, args.figdir, save_suffix=suffix,
                      clims=[(-1, 1), (-100, 100), (0, 120)],
                      fmts=[".1f", ".0f", ".0f"],
                      figsize_per_metric=(5, 4))


def run_CLIP(args):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    from utils.relation_shape_dataset_lib import ShapesDataset

    savedir = join(args.results_root, args.model_run_name)
    config = read_config(join(savedir, "config.py"))

    # Build pipeline (manual, CLIP pipeline class)
    weight_dtype = torch.float16 if config.mixed_precision == "fp16" else (
        torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32)
    pipeline = build_pipeline_manual(config, weight_dtype, pipeline_class=PixArtAlphaPipeline_custom_CLIP)

    # Load checkpoint
    load_checkpoint(pipeline, join(savedir, "checkpoints", args.ckpt_name), use_ema=args.use_ema)

    # Load CLIP encoder
    clip_model_id = args.clip_model_id
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_id, subfolder="tokenizer_2",
                                              torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_id, subfolder="text_encoder_2",
                                                               torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipeline.tokenizer = tokenizer
    pipeline.to(device="cuda", dtype=weight_dtype)
    pipeline.text_encoder = text_encoder.to(device="cuda")

    # Extract word vectors and run variance partition
    dataset_tmp = ShapesDataset(num_images=10000)
    prompt_collection, scene_info_collection = generate_all_prompt_collection(dataset_tmp.spatial_phrases)
    df, wv1, wv2, wv1_proj, wv2_proj = extract_wordvecs_and_project(
        pipeline, prompt_collection, scene_info_collection, device="cuda"
    )

    hidden_size, head_num, layer_num, base_size = get_model_dims(pipeline, config)
    os.makedirs(args.figdir, exist_ok=True)

    for obj_pos, wv_proj, suffix in [("shape2", wv2_proj, "_shape2_MLP_proj_rel_factor"),
                                      ("shape1", wv1_proj, "_shape1_MLP_proj_rel_factor")]:
        direction_vecs, level_names = compute_variance_partition_direction_vecs(wv_proj, df, object_position=obj_pos)
        align_df = screen_all_heads(pipeline, direction_vecs, level_names,
                                    hidden_size, head_num, layer_num, base_size)
        csv_path = join(args.figdir, f"{args.model_run_name}_align_score_allheads{suffix}.csv")
        align_df.to_csv(csv_path, index=False)
        print(f"Saved alignment scores to {csv_path}")
        plot_synopsis(align_df, args.model_run_name, args.figdir, save_suffix=suffix,
                      clims=[(-1, 1), None, None],
                      fmts=[".1f", ".0f", ".0f"],
                      figsize_per_metric=(5, 4))


def run_RTE(args):
    from utils.text_encoder_control_lib import RandomEmbeddingEncoder_wPosEmb
    from transformers import T5Tokenizer

    T5_path = args.t5_path
    savedir = join(args.results_root, args.model_run_name)
    config = read_config(join(savedir, "config.py"))

    # Build pipeline (manual)
    weight_dtype = torch.float16 if config.mixed_precision == "fp16" else (
        torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32)
    pipeline = build_pipeline_manual(config, weight_dtype, pipeline_class=PixArtAlphaPipeline_custom)

    # Load checkpoint
    load_checkpoint(pipeline, join(savedir, "checkpoints", args.ckpt_name), use_ema=args.use_ema)

    # Load RTE encoder
    emb_data = torch.load(join(args.rte_dict_path, "word_embedding_dict.pt"), weights_only=False)
    rndpos_encoder = RandomEmbeddingEncoder_wPosEmb(
        emb_data["embedding_dict"], emb_data["input_ids2dict_ids"],
        emb_data["dict_ids2input_ids"], max_seq_len=20, embed_dim=4096, wpe_scale=1/6,
    ).to("cuda")
    tokenizer = T5Tokenizer.from_pretrained(T5_path)

    pipeline.tokenizer = tokenizer
    pipeline.text_encoder = rndpos_encoder
    pipeline.to(device="cuda", dtype=weight_dtype)

    # For RTE: encode relation words directly (no variance partition)
    relation_words = ["above", "below", "right", "left", "up", "down", "upper", "higher", "lower"]
    spatial_relationship_levels = ["above", "below", "right", "left", "above", "below", "above", "above", "below"]
    token_ids = pipeline.tokenizer.encode(" ".join(relation_words), max_length=20, truncation=False)
    word_embeds, attn_mask = pipeline.text_encoder(token_ids)
    word_embeds_proj = pipeline.transformer.caption_projection(word_embeds)

    hidden_size, head_num, layer_num, base_size = get_model_dims(pipeline, config)
    os.makedirs(args.figdir, exist_ok=True)

    align_df = screen_all_heads(
        pipeline, word_embeds_proj, spatial_relationship_levels,
        hidden_size, head_num, layer_num, base_size,
        direction_vecs_are_numpy=False,
    )
    csv_path = join(args.figdir, f"{args.model_run_name}_all_heads_align_df.csv")
    align_df.to_csv(csv_path, index=False)
    print(f"Saved alignment scores to {csv_path}")
    plot_synopsis(align_df, args.model_run_name, args.figdir,
                  clims=[(-1, 1), (-250, 250), (0, 250)],
                  fmts=[".0f", ".0f", ".0f"],
                  figsize_per_metric=(5, 4))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Head alignment score synopsis")
    p.add_argument("--model_run_name", required=True, help="Name of the model run directory")
    p.add_argument("--text_encoder_type", required=True, choices=["T5", "CLIP", "RTE"],
                   help="Text encoder type")
    p.add_argument("--ckpt_name", default="epoch_4000_step_160000.pth",
                   help="Checkpoint filename")
    p.add_argument("--use_ema", action="store_true", default=True,
                   help="Use EMA weights from checkpoint (default: True)")
    p.add_argument("--no_ema", action="store_false", dest="use_ema")
    p.add_argument("--results_root",
                   default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results",
                   help="Root directory for model results")
    p.add_argument("--figdir", default=None,
                   help="Output figure directory (default: Figures/DiT_{type}_attn_head_finding/{model_run_name})")
    p.add_argument("--t5_path",
                   default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl",
                   help="Path to T5 model (used for T5 encoder and RTE tokenizer)")
    p.add_argument("--clip_model_id", default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="HuggingFace model ID for CLIP encoder")
    p.add_argument("--rte_dict_path",
                   default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/random_emd_dictionary",
                   help="Path to random embedding dictionary")
    return p.parse_args()


def main():
    args = parse_args()

    # Set default figdir
    if args.figdir is None:
        repo_root = join(os.path.dirname(__file__), "..")
        type_dir = {"T5": "DiT_T5_attn_head_finding",
                     "CLIP": "DiT_CLIP_attn_head_finding",
                     "RTE": "DiT_rndpos_attn_head_finding"}[args.text_encoder_type]
        args.figdir = join(repo_root, "Figures", type_dir, args.model_run_name)

    os.makedirs(args.figdir, exist_ok=True)
    print(f"Model: {args.model_run_name}")
    print(f"Encoder: {args.text_encoder_type}")
    print(f"Output: {args.figdir}")

    dispatch = {"T5": run_T5, "CLIP": run_CLIP, "RTE": run_RTE}
    dispatch[args.text_encoder_type](args)


if __name__ == "__main__":
    main()
