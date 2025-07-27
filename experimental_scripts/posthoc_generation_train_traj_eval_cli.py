# %%
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        # programmatically invoke the same things as %load_ext autoreload
        ip.run_line_magic("load_ext", "autoreload")
        ip.run_line_magic("autoreload", "2")
except ImportError:
    # not in IPython at all
    pass

# %%
import glob
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

import matplotlib.pyplot as plt
import seaborn as sns
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
from utils.text_encoder_control_lib import RandomEmbeddingEncoder_wPosEmb, RandomEmbeddingEncoder
from utils.image_utils import pil_images_to_grid
from utils.attention_map_store_utils import replace_attn_processor, AttnProcessor2_0_Store, PixArtAttentionVisualizer_Store
from utils.cv2_eval_utils import find_classify_object_masks, evaluate_parametric_relation
from utils.cv2_eval_utils import find_classify_objects, evaluate_parametric_relation, eval_func_factory, scene_info_collection
from utils.attention_analysis_lib import plot_attention_layer_head_heatmaps, plot_layer_head_score_summary
from utils.attention_analysis_lib import *
from utils.obj_mask_utils import *
from circuit_toolkit.plot_utils import saveallforms
import pandas as pd
from tqdm.auto import tqdm


def split_image_into_grid(image, grid_size=5, cell_size=128, padding=2):
    """
    Split an image into a grid of subimages.
    
    Args:
        image: PIL Image to split
        grid_size: Size of grid (grid_size x grid_size)
        cell_size: Width/height of each cell in pixels
        padding: Padding between cells in pixels
        
    Returns:
        List of subimages as PIL Images
    """
    width, height = image.size
    cell_width = cell_size
    cell_height = cell_size
    
    # Verify image dimensions match expected grid
    assert (cell_width + padding) * grid_size + padding == width and \
        (cell_height + padding) * grid_size + padding == height
        
    subimages = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * (cell_width + padding) + padding
            upper = row * (cell_height + padding) + padding
            right = left + cell_width
            lower = upper + cell_height
            subimages.append(image.crop((left, upper, right, lower)))
            
    return subimages

# %%
from itertools import product
# you can extend these lists as needed
def generate_test_prompts_collection():
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
            for v in verticals:
                prompts.append(f"{c1} {shape1} is {v} the {c2} {shape2}")
            for h in horizontals:
                prompts.append(f"{c1} {shape1} is {h} the {c2} {shape2}")
            for v, h in product(verticals, horizontals):
                prompts.append(f"{c1} {shape1} is {v} and {h} the {c2} {shape2}")
    return prompts


def generate_test_prompts_collection_and_parsed_words():
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


# %%
# model_run_name = "objrel_T5_DiT_mini_pilot_WDecay" # "objrel_rndembdposemb_DiT_B_pilot" 
model_run_name = "objrel_T5_DiT_B_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth" 
text_encoder_type = "T5" 
suffix = ""




model_run_name = "objrel_rndembdposemb_DiT_micro_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth" 
text_encoder_type = "RandomEmbeddingEncoder_wPosEmb" 
suffix = ""

model_run_name = "objrel_rndembdposemb_DiT_nano_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
# ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth" 
text_encoder_type = "RandomEmbeddingEncoder_wPosEmb" 
suffix = ""

model_run_name = "objrel_rndembdposemb_DiT_mini_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
# ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth" 
text_encoder_type = "RandomEmbeddingEncoder_wPosEmb" 
suffix = ""


model_run_name = "objrel_rndembdposemb_DiT_B_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
# ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth" 
text_encoder_type = "RandomEmbeddingEncoder_wPosEmb" 
suffix = ""

model_run_name = "objrel_T5_DiT_B_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth" 
text_encoder_type = "T5" 
suffix = ""

model_run_name = "objrel_T5_DiT_mini_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
ckpt_name = "epoch_4000_step_160000.pth" # "epoch_4000_step_160000.pth" 
text_encoder_type = "T5" 
suffix = ""

model_run_name = "objrel_rndemb_DiT_B_pilot" # "objrel_rndembdposemb_DiT_B_pilot" 
text_encoder_type = "RandomEmbeddingEncoder" 
suffix = ""
# for model_run_name, text_encoder_type in [
#     ("objrel_T5_DiT_B_pilot", "T5"),
#     ("objrel_T5_DiT_mini_pilot", "T5"),
#     ("objrel_rndembdposemb_DiT_B_pilot", "RandomEmbeddingEncoder_wPosEmb"),
#     ("objrel_rndembdposemb_DiT_micro_pilot", "RandomEmbeddingEncoder_wPosEmb"),
#     ("objrel_rndembdposemb_DiT_nano_pilot", "RandomEmbeddingEncoder_wPosEmb"),
#     ("objrel_rndembdposemb_DiT_mini_pilot", "RandomEmbeddingEncoder_wPosEmb"),
#     # ("objrel_rndemb_DiT_B_pilot", "RandomEmbeddingEncoder"),
#     ("objrel_T5_DiT_B_pilot_WDecay", "T5"),
#     ("objrel_T5_DiT_mini_pilot_WDecay", "T5"),
# ]:
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Post-hoc generation training trajectory evaluation')
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
parser.add_argument('--text_encoder_type', type=str, 
                    choices=["T5", "RandomEmbeddingEncoder_wPosEmb", "RandomEmbeddingEncoder"],
                    help='Text encoder type (will be auto-determined if not specified)')

args = parser.parse_args()

# Map model names to text encoder types
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

model_run_name = args.model_run_name
text_encoder_type = args.text_encoder_type if args.text_encoder_type else model_to_encoder[model_run_name]
# %% [markdown]
# ### DiT network at float16, T5 at bfloat16
# %%
T5_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"
tokenizer = T5Tokenizer.from_pretrained(T5_path, ) #subfolder="tokenizer")
if text_encoder_type == "T5":
    # T5_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.T5_dtype]
    T5_dtype = torch.bfloat16
    text_encoder = T5EncoderModel.from_pretrained(T5_path, load_in_8bit=False, torch_dtype=T5_dtype, )
elif text_encoder_type == "RandomEmbeddingEncoder_wPosEmb":
    text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRel_pilot_rndemb/caption_feature_wmask'
    emb_data = th.load(join(text_feat_dir_old, "word_embedding_dict.pt")) 
    text_encoder = RandomEmbeddingEncoder_wPosEmb(emb_data["embedding_dict"], 
                                                emb_data["input_ids2dict_ids"], 
                                                emb_data["dict_ids2input_ids"], 
                                                max_seq_len=20, embed_dim=4096,
                                                wpe_scale=1/6).to("cuda")
elif text_encoder_type == "RandomEmbeddingEncoder":
    text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRel_pilot_rndemb/caption_feature_wmask'
    emb_data = th.load(join(text_feat_dir_old, "word_embedding_dict.pt")) 
    text_encoder = RandomEmbeddingEncoder(emb_data["embedding_dict"], 
                                                emb_data["input_ids2dict_ids"], 
                                                emb_data["dict_ids2input_ids"], 
                                                ).to("cuda")

# %%
savedir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/{model_run_name}"
# figdir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/{model_run_name}{suffix}/cross_attn_vis_figs"
result_dir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/analysis_results/{model_run_name}{suffix}"
eval_dir = join(savedir, "large_scale_eval_posthoc")
os.makedirs(eval_dir, exist_ok=True)
#%%
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
# ckptdir = join(savedir, "checkpoints")
# ckpt = torch.load(join(ckptdir, ckpt_name))
# pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))
# pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))
pipeline.tokenizer = tokenizer
pipeline.to(device="cuda", dtype=weight_dtype);
# pipeline.to(device="cuda", dtype=torch.bfloat16);
pipeline.text_encoder = text_encoder.to(device="cuda", )
# add attention map store hooks
# pipeline.transformer = replace_attn_processor(pipeline.transformer)
# attnvis_store = PixArtAttentionVisualizer_Store(pipeline)
# attnvis_store.setup_hooks()
# torch.cuda.empty_cache()

# %%
# test one prompt works. 
out = pipeline("blue square below and to the right of red circle", num_inference_steps=14, guidance_scale=4.5, max_sequence_length=20, 
        num_images_per_prompt=4, generator=th.Generator(device="cuda").manual_seed(42), prompt_dtype=torch.float16)

# %% [markdown]
# ### Mass produce 

# %%
prompts, parsed_words = generate_test_prompts_collection_and_parsed_words()
prompt_df = pd.DataFrame(parsed_words)
prompt_df["relation_str"] = prompt_df["relation"].apply(lambda x: "_".join(x))

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

# %%
eval_df_all_traj = []
object_df_all_traj = []
# step_num = 160000
ckptdir = join(savedir, "checkpoints")
ckpt_all = sorted(glob.glob(join(ckptdir, "*.pth")), 
                key=lambda x: int(os.path.basename(x).split('_step_')[-1].split('.pth')[0]))
for ckpt_path in ckpt_all:
    ckpt_name = os.path.basename(ckpt_path)
    # Extract step number from filename (assuming format like "epoch_X_step_Y.pth")
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
        for prompt_id, row in tqdm(prompt_df.iterrows()):
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
                # eval_result = eval_func(classified_objects_df)  # Returns a dictionary
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
            # eval_df.to_csv(join(eval_dir, f"eval_df_{prompt_id}_{prompt.replace(' ', '_')}.csv"), index=False)
            # object_df.to_pickle(join(eval_dir, f"object_df_{prompt_id}_{prompt.replace(' ', '_')}.pkl"))
            # save object_df
            acc_overall = eval_df["overall"].mean()
            acc_shape = eval_df["shape"].mean()
            acc_color = eval_df["color"].mean()
            acc_exist_binding = eval_df["exist_binding"].mean()
            acc_unique_binding = eval_df["unique_binding"].mean()
            acc_spatial_relationship = eval_df["spatial_relationship"].mean()
            acc_spatial_relationship_loose = eval_df["spatial_relationship_loose"].mean()
            Dx_mean = eval_df["Dx"].mean(skipna=True)
            Dy_mean = eval_df["Dy"].mean(skipna=True)
            Dx_std = eval_df["Dx"].std(skipna=True)
            Dy_std = eval_df["Dy"].std(skipna=True)
            N_valid = eval_df["Dx"].count()
            N_total = len(eval_df)
            print(f"prompt: {prompt} (id: {prompt_id}) {scene_info}")
            print(f"accuracy color {acc_color:.2f}, shape {acc_shape:.2f}, exist_binding {acc_exist_binding:.2f}, unique_binding {acc_unique_binding:.2f}, spatial_relationship {acc_spatial_relationship:.2f}, spatial_relationship_loose {acc_spatial_relationship_loose:.2f}, overall {acc_overall:.2f}")
            print(f"Dx {Dx_mean:.2f} ± {Dx_std:.2f}, Dy {Dy_mean:.2f} ± {Dy_std:.2f} (valid N={N_valid}/{N_total})")
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

#%%
synopsis_dir = f"/n/home12/binxuwang/Github/DiffusionObjectRelation/Figures/model_eval_synopsis"
eval_df_all_traj_syn = eval_df_all_traj.groupby(["step_num", "ckpt_ver"]).mean(numeric_only=True).reset_index()
# 1) melt into long form
df_long = eval_df_all_traj_syn.reset_index().melt(
    id_vars=["step_num", "ckpt_ver"],
    value_vars=["color", "shape", "exist_binding", "unique_binding", "spatial_relationship", "spatial_relationship_loose"],
    var_name="metric",
    value_name="value",
)
# 2) make the combined label column
df_long["legend_label"] = df_long["metric"] + " " + df_long["ckpt_ver"]
# 3) plot, using legend_label for hue and metric for style (if you still want different markers/linestyles)
plt.figure(figsize=(5, 4))
sns.lineplot(
    data=df_long,
    x="step_num",
    y="value",
    hue="metric",
    style="ckpt_ver",
    markers=False,    # or dashes=True
)
plt.xlabel("Step Number")
plt.ylabel("Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="")
plt.suptitle(f"Evaluation Trajectory\n{model_run_name}")
# saveallforms(eval_dir, "eval_train_dynamics_traj_syn")
saveallforms(synopsis_dir, f"{model_run_name}_eval_train_dynamics_traj_syn")
plt.show()



# %% [markdown]
# ### Synopsis 
# # %%
# eval_syn_df = eval_df_all.groupby("prompt_id").mean(numeric_only=True)
# eval_syn_df = eval_syn_df.rename(columns={'overall': 'overall_acc', 'shape': 'shape_acc', 'color': 'color_acc', 'spatial_relationship': 'spatial_relationship_acc'})
# prompt_eval_syn_df = prompt_df.merge(eval_syn_df, left_index=True, right_index=True)
# prompt_eval_syn_df.to_pickle(join(eval_dir, f"prompt_eval_syn_df.pkl"))

# print(prompt_eval_syn_df.head())
# print("starting heatmap")
# # %%
# # Create heatmap
# # Pivot the data to create a heatmap with spatial relationships as columns
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='spatial_relationship_acc',
#     aggfunc='mean'
# )

# plt.figure(figsize=(8, 6))
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Spatial Relationship Accuracy'})
# plt.title(f'Spatial Relationship Accuracy by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
# plt.xlabel('Prompt Spatial Relationship')
# plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# saveallforms(eval_dir, f"relation_eval_heatmap_spatial_relationship_acc_{model_run_name}")
# plt.show()


# # %%
# # Pivot the data to create a heatmap with spatial relationships as columns
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='shape_acc',
#     aggfunc='mean'
# )
# plt.figure(figsize=(8, 6))
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Shape Accuracy'})
# plt.title(f'Shape Accuracy by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
# plt.xlabel('Prompt Spatial Relationship')
# plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# saveallforms(eval_dir, f"relation_eval_heatmap_shape_acc_{model_run_name}")
# plt.show()

# # %%

# # Pivot the data to create a heatmap with spatial relationships as columns
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='color_acc',
#     aggfunc='mean'
# )
# plt.figure(figsize=(8, 6))
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Color Accuracy'})
# plt.title(f'Color Accuracy by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
# plt.xlabel('Prompt Spatial Relationship')
# plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# saveallforms(eval_dir, f"relation_eval_heatmap_color_acc_{model_run_name}")
# plt.show()

# # %%

# # Pivot the data to create a heatmap with spatial relationships as columns
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='Dx',
#     aggfunc='mean'
# )
# plt.figure(figsize=(8, 6))
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', cbar_kws={'label': 'Dx'})
# plt.title(f'Dx by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
# plt.xlabel('Prompt Spatial Relationship')
# plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# saveallforms(eval_dir, f"relation_eval_heatmap_Dx_{model_run_name}")
# plt.show()

# # %%

# # Pivot the data to create a heatmap with spatial relationships as columns
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='Dy',
#     aggfunc='mean'
# )
# plt.figure(figsize=(8, 6))
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', cbar_kws={'label': 'Dy'})
# plt.title(f'Dy by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
# plt.xlabel('Prompt Spatial Relationship')
# plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# saveallforms(eval_dir, f"relation_eval_heatmap_Dy_{model_run_name}")
# plt.show()

# # %%
# # Create a combined figure with multiple subplots
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# # Plot 1: Accuracy by Prompt Spatial Relationship
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='overall_acc',
#     aggfunc='mean'
# )
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Accuracy'}, ax=axes[0, 0])
# axes[0, 0].set_title(f'Overall Accuracy')
# axes[0, 0].set_xlabel('Prompt Spatial Relationship')
# axes[0, 0].set_ylabel('Object Attributes')
# axes[0, 0].tick_params(axis='x', rotation=45)
# axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), ha='right')

# # Plot 2: Color Accuracy by Prompt Spatial Relationship
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='color_acc',
#     aggfunc='mean'
# )
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Color Accuracy'}, ax=axes[0, 1])
# axes[0, 1].set_title(f'Color Accuracy')
# axes[0, 1].set_xlabel('Prompt Spatial Relationship')
# axes[0, 1].set_ylabel('Object Attributes')
# axes[0, 1].tick_params(axis='x', rotation=45)
# axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), ha='right')
# # Get rid of the y ticks
# axes[0, 1].set_yticks([])

# # Plot 3: Shape Accuracy by Prompt Spatial Relationship
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='shape_acc',
#     aggfunc='mean'
# )
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Shape Accuracy'}, ax=axes[0, 2])
# axes[0, 2].set_title(f'Shape Accuracy')
# axes[0, 2].set_xlabel('Prompt Spatial Relationship')
# axes[0, 2].set_ylabel('Object Attributes')
# axes[0, 2].tick_params(axis='x', rotation=45)
# axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), ha='right')
# # Get rid of the y ticks
# axes[0, 2].set_yticks([])

# # Plot 4: Spatial Relationship Accuracy
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='spatial_relationship_acc',
#     aggfunc='mean'
# )
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Spatial Accuracy'}, ax=axes[1, 0])
# axes[1, 0].set_title(f'Spatial Relationship Accuracy')
# axes[1, 0].set_xlabel('Prompt Spatial Relationship')
# axes[1, 0].set_ylabel('Object Attributes')
# axes[1, 0].tick_params(axis='x', rotation=45)
# axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), ha='right')

# # Plot 5: Dx by Object Attributes
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='Dx',
#     aggfunc='mean'
# )
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', cbar_kws={'label': 'Dx'}, ax=axes[1, 1])
# axes[1, 1].set_title(f'Dx')
# axes[1, 1].set_xlabel('Prompt Spatial Relationship')
# axes[1, 1].set_ylabel('Object Attributes')
# axes[1, 1].tick_params(axis='x', rotation=45)
# axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), ha='right')
# # Get rid of the y ticks
# axes[1, 1].set_yticks([])


# # Plot 6: Dy by Object Attributes
# pivot_df = prompt_eval_syn_df.pivot_table(
#     index=['color1', 'shape1', 'color2', 'shape2'], 
#     columns='relation_str', 
#     values='Dy',
#     aggfunc='mean'
# )
# sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', cbar_kws={'label': 'Dy'}, ax=axes[1, 2])
# axes[1, 2].set_title(f'Dy')
# axes[1, 2].set_xlabel('Prompt Spatial Relationship')
# axes[1, 2].set_ylabel('Object Attributes')
# axes[1, 2].tick_params(axis='x', rotation=45)
# axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), ha='right')
# # Get rid of the y ticks
# axes[1, 2].set_yticks([])

# plt.suptitle(f'Combined Evaluation Results (mean of 100 samples per prompt)\n{model_run_name}', fontsize=16, y=0.98)
# plt.tight_layout()
# saveallforms(eval_dir, f"relation_eval_heatmap_all_synopsis_{model_run_name}")
# plt.show()

# %%


# %%


# # %%


# # %%
# acc_overall = eval_df["overall"].mean()
# acc_shape = eval_df["shape"].mean()
# acc_color = eval_df["color"].mean()
# acc_spatial_relationship = eval_df["spatial_relationship"].mean()
# Dx_mean = eval_df["Dx"].mean(skipna=True)
# Dy_mean = eval_df["Dy"].mean(skipna=True)
# Dx_std = eval_df["Dx"].std(skipna=True)
# Dy_std = eval_df["Dy"].std(skipna=True)
# N_valid = eval_df["Dx"].count()
# N_total = len(eval_df)
# print(f"accuracy color {acc_color:.2f}, shape {acc_shape:.2f}, spatial_relationship {acc_spatial_relationship:.2f}, overall {acc_overall:.2f}")
# print(f"Dx {Dx_mean:.2f} ± {Dx_std:.2f}, Dy {Dy_mean:.2f} ± {Dy_std:.2f} (valid N={N_valid}/{N_total})")

# # %%
# step_num = 160000
# prompt_id = 1
# prompt = "blue square below and to the right of a red circle"
# scene_info = {'color1': 'blue',
#   'shape1': 'Square',
#   'color2': 'red',
#   'shape2': 'Circle',
#   'spatial_relationship': 'lower_right'}
# out = pipeline(prompt, num_inference_steps=14, guidance_scale=4.5,
#          num_images_per_prompt=81, generator=th.Generator(device="cuda").manual_seed(42), prompt_dtype=torch.float16)
# # subimages = load_image_grid(step_num, prompt_id, sample_root=sample_root)


# # %%


# # %%


# # %% [markdown]
# # ### Whole network at bfloat16

# # %%
# del pipeline.text_encoder
# del text_encoder
# torch.cuda.empty_cache()
# #%%

# # %%
# text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/objectRel_pilot_rndemb/caption_feature_wmask'
# T5_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"
# tokenizer = T5Tokenizer.from_pretrained(T5_path, )#subfolder="tokenizer")
# if text_encoder_type == "T5":
#     text_encoder = T5EncoderModel.from_pretrained(T5_path, load_in_8bit=False, torch_dtype=torch.bfloat16, )

# # %%
# savedir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/{model_run_name}"
# figdir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/{model_run_name}{suffix}/cross_attn_vis_figs"
# os.makedirs(figdir, exist_ok=True)

# config = read_config(join(savedir, 'config.py'))
# weight_dtype = torch.float32
# if config.mixed_precision == "fp16": # accelerator.
#     weight_dtype = torch.float16
# elif config.mixed_precision == "bf16": # accelerator.
#     weight_dtype = torch.bfloat16
    
# image_size = config.image_size  # @param [256, 512, 1024]
# latent_size = int(image_size) // 8
# pred_sigma = getattr(config, 'pred_sigma', True)
# learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
# model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
#                 "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
#                 'model_max_length': config.model_max_length}
# # train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
# model = build_model(config.model,
#                 config.grad_checkpointing,
#                 config.get('fp32_attention', False),
#                 input_size=latent_size,
#                 learn_sigma=learn_sigma,
#                 pred_sigma=pred_sigma,
#                 **model_kwargs).train()
# num_layers = len(model.blocks)
# transformer = Transformer2DModel(
#         sample_size=image_size // 8,
#         num_layers=len(model.blocks),
#         attention_head_dim=model.blocks[0].hidden_size // model.num_heads,
#         in_channels=model.in_channels,
#         out_channels=model.out_channels,
#         patch_size=model.patch_size,
#         attention_bias=True,
#         num_attention_heads=model.num_heads,
#         cross_attention_dim=model.blocks[0].hidden_size,
#         activation_fn="gelu-approximate",
#         num_embeds_ada_norm=1000,
#         norm_type="ada_norm_single",
#         norm_elementwise_affine=False,
#         norm_eps=1e-6,
#         caption_channels=4096,
# )
# # state_dict = state_dict_convert(all_state_dict.pop("state_dict"))
# transformer.load_state_dict(state_dict_convert(model.state_dict()))
# pipeline = PixArtAlphaPipeline_custom.from_pretrained(
#     "PixArt-alpha/PixArt-XL-2-512x512",
#     transformer=transformer,
#     tokenizer=tokenizer,
#     text_encoder=None,
#     torch_dtype=weight_dtype,
# )
# ckptdir = join(savedir, "checkpoints")
# ckpt = torch.load(join(ckptdir, ckpt_name))
# pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))
# # pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))
# pipeline.tokenizer = tokenizer
# pipeline.text_encoder = text_encoder
# # pipeline.to(device="cuda", dtype=weight_dtype);
# pipeline.to(device="cuda", dtype=torch.bfloat16);
# # add attention map store hooks
# # pipeline.transformer = replace_attn_processor(pipeline.transformer)
# # attnvis_store = PixArtAttentionVisualizer_Store(pipeline)
# # attnvis_store.setup_hooks()
# # torch.cuda.empty_cache()

# # %%
# print("pipeline.text_encoder.dtype", pipeline.text_encoder.dtype)
# print("pipeline.transformer.dtype", pipeline.transformer.dtype)
# out = pipeline("a blue square below and to the right of a red circle", num_inference_steps=14, guidance_scale=4.5,
#          num_images_per_prompt=49, generator=th.Generator(device="cuda").manual_seed(42), prompt_dtype=None)
# pil_images_to_grid(out.images)

# # %% [markdown]
# # ### Both transformer and T5 Float16

# # %%
# del pipeline.text_encoder
# del text_encoder
# torch.cuda.empty_cache()
# #%%

# # %%
# text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/objectRel_pilot_rndemb/caption_feature_wmask'
# T5_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"
# tokenizer = T5Tokenizer.from_pretrained(T5_path, )#subfolder="tokenizer")
# if text_encoder_type == "T5":
#     text_encoder = T5EncoderModel.from_pretrained(T5_path, load_in_8bit=False, torch_dtype=torch.float16, )

# # %%
# savedir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/{model_run_name}"
# figdir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/{model_run_name}{suffix}/cross_attn_vis_figs"
# os.makedirs(figdir, exist_ok=True)

# config = read_config(join(savedir, 'config.py'))
# weight_dtype = torch.float32
# if config.mixed_precision == "fp16": # accelerator.
#     weight_dtype = torch.float16
# elif config.mixed_precision == "bf16": # accelerator.
#     weight_dtype = torch.bfloat16
    
# image_size = config.image_size  # @param [256, 512, 1024]
# latent_size = int(image_size) // 8
# pred_sigma = getattr(config, 'pred_sigma', True)
# learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
# model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
#                 "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
#                 'model_max_length': config.model_max_length}
# # train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
# model = build_model(config.model,
#                 config.grad_checkpointing,
#                 config.get('fp32_attention', False),
#                 input_size=latent_size,
#                 learn_sigma=learn_sigma,
#                 pred_sigma=pred_sigma,
#                 **model_kwargs).train()
# num_layers = len(model.blocks)
# transformer = Transformer2DModel(
#         sample_size=image_size // 8,
#         num_layers=len(model.blocks),
#         attention_head_dim=model.blocks[0].hidden_size // model.num_heads,
#         in_channels=model.in_channels,
#         out_channels=model.out_channels,
#         patch_size=model.patch_size,
#         attention_bias=True,
#         num_attention_heads=model.num_heads,
#         cross_attention_dim=model.blocks[0].hidden_size,
#         activation_fn="gelu-approximate",
#         num_embeds_ada_norm=1000,
#         norm_type="ada_norm_single",
#         norm_elementwise_affine=False,
#         norm_eps=1e-6,
#         caption_channels=4096,
# )
# # state_dict = state_dict_convert(all_state_dict.pop("state_dict"))
# transformer.load_state_dict(state_dict_convert(model.state_dict()))
# pipeline = PixArtAlphaPipeline_custom.from_pretrained(
#     "PixArt-alpha/PixArt-XL-2-512x512",
#     transformer=transformer,
#     tokenizer=tokenizer,
#     text_encoder=None,
#     torch_dtype=weight_dtype,
# )
# ckptdir = join(savedir, "checkpoints")
# ckpt = torch.load(join(ckptdir, ckpt_name))
# pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))
# # pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))
# pipeline.tokenizer = tokenizer
# pipeline.text_encoder = text_encoder.to(device="cuda", )
# pipeline.to(device="cuda", dtype=weight_dtype);
# # pipeline.to(device="cuda", dtype=torch.bfloat16);
# # add attention map store hooks
# # pipeline.transformer = replace_attn_processor(pipeline.transformer)
# # attnvis_store = PixArtAttentionVisualizer_Store(pipeline)
# # attnvis_store.setup_hooks()
# # torch.cuda.empty_cache()

# # %%
# print("pipeline.text_encoder.dtype", pipeline.text_encoder.dtype)
# print("pipeline.transformer.dtype", pipeline.transformer.dtype)
# out = pipeline("a blue square below and to the right of a red circle", num_inference_steps=14, guidance_scale=4.5,
#          num_images_per_prompt=49, generator=th.Generator(device="cuda").manual_seed(42), prompt_dtype=torch.float16)
# pil_images_to_grid(out.images)

# # %% [markdown]
# # ### float16 T5 embedding, bfloat16 DiT weights

# # %%
# del pipeline.text_encoder
# del text_encoder
# torch.cuda.empty_cache()

# # %%
# text_feat_dir_old = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/objectRel_pilot_rndemb/caption_feature_wmask'
# T5_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"
# tokenizer = T5Tokenizer.from_pretrained(T5_path, )#subfolder="tokenizer")
# if text_encoder_type == "T5":
#     text_encoder = T5EncoderModel.from_pretrained(T5_path, load_in_8bit=False, torch_dtype=torch.float16, )

# # %%
# savedir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/{model_run_name}"
# figdir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/{model_run_name}{suffix}/cross_attn_vis_figs"
# os.makedirs(figdir, exist_ok=True)

# config = read_config(join(savedir, 'config.py'))
# # weight_dtype = torch.float32
# # if config.mixed_precision == "fp16": # accelerator.
# #     weight_dtype = torch.float16
# # elif config.mixed_precision == "bf16": # accelerator.
# #     weight_dtype = torch.bfloat16

# weight_dtype = torch.bfloat16

# image_size = config.image_size  # @param [256, 512, 1024]
# latent_size = int(image_size) // 8
# pred_sigma = getattr(config, 'pred_sigma', True)
# learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
# model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
#                 "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
#                 'model_max_length': config.model_max_length}
# # train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
# model = build_model(config.model,
#                 config.grad_checkpointing,
#                 config.get('fp32_attention', False),
#                 input_size=latent_size,
#                 learn_sigma=learn_sigma,
#                 pred_sigma=pred_sigma,
#                 **model_kwargs).train()
# num_layers = len(model.blocks)
# transformer = Transformer2DModel(
#         sample_size=image_size // 8,
#         num_layers=len(model.blocks),
#         attention_head_dim=model.blocks[0].hidden_size // model.num_heads,
#         in_channels=model.in_channels,
#         out_channels=model.out_channels,
#         patch_size=model.patch_size,
#         attention_bias=True,
#         num_attention_heads=model.num_heads,
#         cross_attention_dim=model.blocks[0].hidden_size,
#         activation_fn="gelu-approximate",
#         num_embeds_ada_norm=1000,
#         norm_type="ada_norm_single",
#         norm_elementwise_affine=False,
#         norm_eps=1e-6,
#         caption_channels=4096,
# )
# # state_dict = state_dict_convert(all_state_dict.pop("state_dict"))
# transformer.load_state_dict(state_dict_convert(model.state_dict()))
# pipeline = PixArtAlphaPipeline_custom.from_pretrained(
#     "PixArt-alpha/PixArt-XL-2-512x512",
#     transformer=transformer,
#     tokenizer=tokenizer,
#     text_encoder=None,
#     torch_dtype=weight_dtype,
# )
# ckptdir = join(savedir, "checkpoints")
# ckpt = torch.load(join(ckptdir, ckpt_name))
# pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))
# # pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict']))
# pipeline.tokenizer = tokenizer
# pipeline.to(device="cuda", dtype=weight_dtype);
# pipeline.text_encoder = text_encoder.to(device="cuda", )
# # pipeline.to(device="cuda", dtype=torch.bfloat16);

# # %%
# print("pipeline.text_encoder.dtype", pipeline.text_encoder.dtype)
# print("pipeline.transformer.dtype", pipeline.transformer.dtype)
# out = pipeline("a blue square below and to the right of a red circle", num_inference_steps=14, guidance_scale=4.5,
#          num_images_per_prompt=49, generator=th.Generator(device="cuda").manual_seed(42), prompt_dtype=torch.bfloat16)
# pil_images_to_grid(out.images)

# # %%



