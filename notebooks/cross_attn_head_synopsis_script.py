# %%
import pandas as pd
import os
from os.path import join
import torch
import torch as th
import torch.nn.functional as F
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from circuit_toolkit.plot_utils import saveallforms

# %%
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
for prompt in generate_test_prompts_collection():
    print(prompt)
    
prompts, parsed_words = generate_test_prompts_collection_and_parsed_words()
prompt_df = pd.DataFrame(parsed_words)
# %%
# check the prompt decsompose into words in that row. 
# this is a sanity check to make sure the prompt is parsed correctly. 
prompts, parsed_words = generate_test_prompts_collection_and_parsed_words()
prompt_df = pd.DataFrame(parsed_words)
for i, row in prompt_df.iterrows():
    assert set(row["prompt"].split(" ")) == set(row["prop"]) | set(row["relation"]) | set([row["color1"], row["shape1"], row["color2"], row["shape2"]]), row

# %%
# row_idx = 0
# row = prompt_df.iloc[row_idx]
# prompt = row["prompt"]
# prompt_dir = join(figroot, prompt.replace(" ", "_"))
# img_src = row["shape1"]
# text_target = row["relation"][0]
# attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
# attn_stats = th.load(attn_stats_file)
# print(attn_stats.keys())

# %%

# sns.heatmap(attn_stats["cond_stats"])
# plt.title(f"{prompt}\n Img {img_src} to Text '{text_target}'")
# plt.show()

# # %% [markdown]
# # ### Prompt split

# # %%
# # row_idx = 0
# prompt_part_df = prompt_df.query("shape1 == 'square' and color1 == 'red'")
# fig, axs = plt.subplots(2, 8, figsize=(24, 6), dpi=300)
# for idx in range(8):
#     r_idx = np.random.randint(0, len(prompt_part_df))
#     row = prompt_part_df.iloc[r_idx]
#     prompt = row["prompt"]
#     prompt_dir = join(figroot, prompt.replace(" ", "_"))
#     img_src = row["shape1"]
#     text_target = row["shape1"]
#     attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
#     attn_stats = th.load(attn_stats_file, weights_only=False)
#     plt.sca(axs[0, idx])
#     sns.heatmap(attn_stats["cond_stats"])
#     plt.title(f"{prompt}\n Img {img_src} to Text '{text_target}'", fontsize=8)
#     plt.sca(axs[1, idx])
#     img_src = row["shape2"]
#     text_target = row["shape2"]
#     attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
#     attn_stats = th.load(attn_stats_file, weights_only=False)
#     sns.heatmap(attn_stats["cond_stats"])
#     plt.title(f"{prompt}\n Img {img_src} to Text '{text_target}'", fontsize=8)
# plt.suptitle("Cross-Attention Head Synopses | top row: Img token for shape1 -> shape1 text | bottom row: Img token for shape2 -> shape2 text", fontsize=12)
# plt.tight_layout()
# plt.show()

# # %%
# # row_idx = 0
# prompt_part_df = prompt_df.query("shape1 == 'square' and color1 == 'blue'")
# fig, axs = plt.subplots(2, 8, figsize=(24, 6), dpi=300)
# for idx in range(8):
#     r_idx = np.random.randint(0, len(prompt_part_df))
#     row = prompt_part_df.iloc[r_idx]
#     prompt = row["prompt"]
#     prompt_dir = join(figroot, prompt.replace(" ", "_"))
#     img_src = row["shape1"]
#     text_target = row["shape1"]
#     attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
#     attn_stats = th.load(attn_stats_file, weights_only=False)
#     plt.sca(axs[0, idx])
#     sns.heatmap(attn_stats["cond_stats"])
#     plt.title(f"{prompt}\n Img {img_src} to Text '{text_target}'", fontsize=8)
#     plt.sca(axs[1, idx])
#     img_src = row["shape2"]
#     text_target = row["shape2"]
#     attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
#     attn_stats = th.load(attn_stats_file, weights_only=False)
#     sns.heatmap(attn_stats["cond_stats"])
#     plt.title(f"{prompt}\n Img {img_src} to Text '{text_target}'", fontsize=8)
# plt.suptitle("Cross-Attention Head Synopses | top row: Img token for shape1 -> shape1 text | bottom row: Img token for shape2 -> shape2 text", fontsize=12)
# plt.tight_layout()
# plt.show()
# %%
# prompt_part_df = prompt_df.query("shape1 == 'square' and color1 == 'blue'")
# cond_stats_list = []
# for r_idx in range(len(prompt_part_df)):
#     row = prompt_part_df.iloc[r_idx]
#     prompt = row["prompt"]
#     prompt_dir = join(figroot, prompt.replace(" ", "_"))
#     img_src = row["shape1"]
#     text_target = row["shape1"]
#     attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
#     attn_stats = th.load(attn_stats_file, weights_only=False)
#     cond_stats = attn_stats["cond_stats"]
#     cond_stats_list.append(cond_stats)

# cond_stats_array = np.stack(cond_stats_list, axis=0)
# sns.heatmap(cond_stats_array.mean(axis=0))
# plt.show()
#%%
model_run_name = "objrel2_DiT_B_pilot"
model_run_name = "objrel_rndembdposemb_DiT_mini_pilot"
model_run_name = "objrel_rndembdposemb_DiT_micro_pilot"
figroot = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/{model_run_name}/cross_attn_vis_figs"
synfigdir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/{model_run_name}/synopsis"
os.makedirs(synfigdir, exist_ok=True)

# %% [markdown]
# ### Summary across prompt partition: Object -> corresponding shape
# %%
img_src_str = "square"
text_target_str = "square"
fig, axs = plt.subplots(2, 2, figsize=(8, 8), )
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape1 == 'square' and color1 == 'blue'",
    "shape1 == 'square' and color1 == 'red'",
    "shape2 == 'square' and color2 == 'blue'",
    "shape2 == 'square' and color2 == 'red'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for Square -> Text 'Square'", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_square_to_square_color_order_split", fig)
plt.show()

# %%
img_src_str = "circle"
text_target_str = "circle"
fig, axs = plt.subplots(2, 2, figsize=(8, 8), )
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape1 == 'circle' and color1 == 'blue'",
    "shape1 == 'circle' and color1 == 'red'",
    "shape2 == 'circle' and color2 == 'blue'",
    "shape2 == 'circle' and color2 == 'red'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for circle -> Text 'circle'", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_circle_to_circle_color_order_split", fig)
plt.show()

# %%
img_src_str = "triangle"
text_target_str = "triangle"
fig, axs = plt.subplots(2, 2, figsize=(8, 8), )
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape1 == 'triangle' and color1 == 'blue'",
    "shape1 == 'triangle' and color1 == 'red'",
    "shape2 == 'triangle' and color2 == 'blue'",
    "shape2 == 'triangle' and color2 == 'red'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for triangle -> Text 'triangle'", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_triangle_to_triangle_color_order_split", fig)
plt.show()

# %% [markdown]
# ### Object to relation words

# %%
# img_src_str = "shape1"
# text_target_str = "relation"
fig, axs = plt.subplots(2, 1, figsize=(4.5, 8), squeeze=False)
for src_idx, img_src_str in enumerate(["shape1", "shape2"]):
    for query_idx, query_str in enumerate([
        "index == index",
    ]):
        prompt_part_df = prompt_df.query(query_str)
        cond_stats_list = []
        for r_idx in range(len(prompt_part_df)):
            row = prompt_part_df.iloc[r_idx]
            prompt = row["prompt"]
            img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
            # text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
            text_target = row["relation"][-1] 
            prompt_dir = join(figroot, prompt.replace(" ", "_"))
            attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
            attn_stats = th.load(attn_stats_file, weights_only=False)
            cond_stats = attn_stats["cond_stats"]
            cond_stats_list.append(cond_stats)

        cond_stats_array = np.stack(cond_stats_list, axis=0)
        plt.sca(axs[src_idx, query_idx])
        sns.heatmap(cond_stats_array.mean(axis=0))
        if src_idx == 0:
            plt.title(f"Shape1 -> Text relation | {query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
        else:
            plt.title(f"Shape2 -> Text relation | {query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for object -> Text relation", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_object_to_relation_order_split", fig)
plt.show()

# %%
# img_src_str = "shape1"
# text_target_str = "relation"
fig, axs = plt.subplots(2, 2, figsize=(8, 8), )

for src_idx, img_src_str in enumerate(["shape1", "shape2"]):
    for query_idx, query_str in enumerate([
        "color1 == 'blue'",
        "color1 == 'red'",
    ]):
        prompt_part_df = prompt_df.query(query_str)
        cond_stats_list = []
        for r_idx in range(len(prompt_part_df)):
            row = prompt_part_df.iloc[r_idx]
            prompt = row["prompt"]
            img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
            # text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
            text_target = row["relation"][-1] 
            prompt_dir = join(figroot, prompt.replace(" ", "_"))
            attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
            attn_stats = th.load(attn_stats_file, weights_only=False)
            cond_stats = attn_stats["cond_stats"]
            cond_stats_list.append(cond_stats)

        cond_stats_array = np.stack(cond_stats_list, axis=0)
        plt.sca(axs[src_idx, query_idx])
        sns.heatmap(cond_stats_array.mean(axis=0))
        if src_idx == 0:
            plt.title(f"Shape1 -> Text relation | {query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
        else:
            plt.title(f"Shape2 -> Text relation | {query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for object -> Text relation", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_object_to_relation_color_order_split", fig)
plt.show()

# %%
# img_src_str = "shape1"
# text_target_str = "relation"
fig, axs = plt.subplots(2, 3, figsize=(12, 8), )

for src_idx, img_src_str in enumerate(["shape1", "shape2"]):
    for query_idx, query_str in enumerate([
        "shape1 == 'square'",
        "shape1 == 'circle'",
        "shape1 == 'triangle'",
    ]):
        prompt_part_df = prompt_df.query(query_str)
        cond_stats_list = []
        for r_idx in range(len(prompt_part_df)):
            row = prompt_part_df.iloc[r_idx]
            prompt = row["prompt"]
            img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
            # text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
            text_target = row["relation"][-1] 
            prompt_dir = join(figroot, prompt.replace(" ", "_"))
            attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
            attn_stats = th.load(attn_stats_file, weights_only=False)
            cond_stats = attn_stats["cond_stats"]
            cond_stats_list.append(cond_stats)

        cond_stats_array = np.stack(cond_stats_list, axis=0)
        plt.sca(axs[src_idx, query_idx])
        sns.heatmap(cond_stats_array.mean(axis=0))
        if src_idx == 0:
            plt.title(f"Shape1 -> Text relation | {query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
        else:
            plt.title(f"Shape2 -> Text relation | {query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for object -> Text relation", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_object_to_relation_shape_order_split", fig)
plt.show()

# %%
prompt_df["relation"] = prompt_df.relation.apply(lambda x: tuple(x))
prompt_df["relation_str"] = prompt_df.relation.apply(lambda x: "_".join(x))

# %%
prompt_df.relation_str.unique()

# %%
# img_src_str = "shape1"
# text_target_str = "relation"
fig, axs = plt.subplots(2, 8, figsize=(32, 8), )
for src_idx, img_src_str in enumerate(["shape1", "shape2"]):
    for query_idx, query_str in enumerate([
        "relation_str == 'above'",
        "relation_str == 'below'",
        "relation_str == 'left'",
        "relation_str == 'right'",
        "relation_str == 'above_left'",
        "relation_str == 'above_right'",
        "relation_str == 'below_left'",
        "relation_str == 'below_right'",
    ]):
        prompt_part_df = prompt_df.query(query_str)
        cond_stats_list = []
        for r_idx in range(len(prompt_part_df)):
            row = prompt_part_df.iloc[r_idx]
            prompt = row["prompt"]
            img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
            # text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
            text_target = "_".join(["below", "left", "top", "right", "above"]) #row["relation"][-1] 
            prompt_dir = join(figroot, prompt.replace(" ", "_"))
            attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
            attn_stats = th.load(attn_stats_file, weights_only=False)
            cond_stats = attn_stats["cond_stats"]
            cond_stats_list.append(cond_stats)

        cond_stats_array = np.stack(cond_stats_list, axis=0)
        plt.sca(axs[src_idx, query_idx])
        sns.heatmap(cond_stats_array.mean(axis=0))
        if src_idx == 0:
            plt.title(f"Shape1 -> Text relation | {query_str.replace(' == ', '=').replace('relation_str','rel')} | N={len(prompt_part_df)}")
        else:
            plt.title(f"Shape2 -> Text relation | {query_str.replace(' == ', '=').replace('relation_str','rel')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for object -> Text relation", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_object_to_relation_relation_order_split", fig)
plt.show()

# %%
# img_src_str = "shape1"
# text_target_str = "relation"
fig, axs = plt.subplots(2, 8, figsize=(32, 8), )
for src_idx, img_src_str in enumerate(["shape1", "shape2"]):
    for query_idx, query_str in enumerate([
        "relation_str == 'above'",
        "relation_str == 'below'",
        "relation_str == 'left'",
        "relation_str == 'right'",
        "relation_str == 'above_left'",
        "relation_str == 'above_right'",
        "relation_str == 'below_left'",
        "relation_str == 'below_right'",
    ]):
        prompt_part_df = prompt_df.query(query_str)
        cond_stats_list = []
        for r_idx in range(len(prompt_part_df)):
            row = prompt_part_df.iloc[r_idx]
            prompt = row["prompt"]
            img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
            # text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
            text_target = row["relation"][-1] # "_".join(["below", "left", "top", "right", "above"]) #
            prompt_dir = join(figroot, prompt.replace(" ", "_"))
            attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
            attn_stats = th.load(attn_stats_file, weights_only=False)
            cond_stats = attn_stats["cond_stats"]
            cond_stats_list.append(cond_stats)

        cond_stats_array = np.stack(cond_stats_list, axis=0)
        plt.sca(axs[src_idx, query_idx])
        sns.heatmap(cond_stats_array.mean(axis=0))
        if src_idx == 0:
            plt.title(f"Shape1 -> Text relation | {query_str.replace(' == ', '=').replace('relation_str','rel')} | N={len(prompt_part_df)}")
        else:
            plt.title(f"Shape2 -> Text relation | {query_str.replace(' == ', '=').replace('relation_str','rel')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for object -> Text relation", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_object_to_relation_relation_order_split_last_word", fig)
plt.show()

# %%
# img_src_str = "shape1"
# text_target_str = "relation"
fig, axs = plt.subplots(2, 6, figsize=(24, 8), )
for src_idx, img_src_str in enumerate(["shape1", "shape2"]):
    for query_idx, query_str in enumerate([
        "relation_str == 'above' and shape1 == 'square'",
        "relation_str == 'above' and shape1 == 'circle'",
        "relation_str == 'above' and shape1 == 'triangle'",
        "relation_str == 'below' and shape1 == 'square'",
        "relation_str == 'below' and shape1 == 'circle'",
        "relation_str == 'below' and shape1 == 'triangle'",
    ]):
        prompt_part_df = prompt_df.query(query_str)
        cond_stats_list = []
        for r_idx in range(len(prompt_part_df)):
            row = prompt_part_df.iloc[r_idx]
            prompt = row["prompt"]
            img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
            # text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
            text_target = "_".join(["below", "left", "top", "right", "above"]) #row["relation"][-1] # 
            prompt_dir = join(figroot, prompt.replace(" ", "_"))
            attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
            attn_stats = th.load(attn_stats_file, weights_only=False)
            cond_stats = attn_stats["cond_stats"]
            cond_stats_list.append(cond_stats)

        cond_stats_array = np.stack(cond_stats_list, axis=0)
        plt.sca(axs[src_idx, query_idx])
        sns.heatmap(cond_stats_array.mean(axis=0))
        if src_idx == 0:
            plt.title(f"Shape1 -> Text relation | {query_str.replace(' == ', '=').replace('relation_str','rel').replace('shape1','S1').replace('shape2','S2').replace('and','&')} | N={len(prompt_part_df)}")
        else:
            plt.title(f"Shape2 -> Text relation | {query_str.replace(' == ', '=').replace('relation_str','rel').replace('shape1','S1').replace('shape2','S2').replace('and','&')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for object -> Text relation", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_object_to_relation_above_below_shape_order_split", fig)
plt.show()

# %%
# img_src_str = "shape1"
# text_target_str = "relation"
fig, axs = plt.subplots(2, 6, figsize=(24, 8), )
for src_idx, img_src_str in enumerate(["shape1", "shape2"]):
    for query_idx, query_str in enumerate([
        "relation_str == 'left' and shape1 == 'square'",
        "relation_str == 'left' and shape1 == 'circle'",
        "relation_str == 'left' and shape1 == 'triangle'",
        "relation_str == 'right' and shape1 == 'square'",
        "relation_str == 'right' and shape1 == 'circle'",
        "relation_str == 'right' and shape1 == 'triangle'",
    ]):
        prompt_part_df = prompt_df.query(query_str)
        cond_stats_list = []
        for r_idx in range(len(prompt_part_df)):
            row = prompt_part_df.iloc[r_idx]
            prompt = row["prompt"]
            img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
            # text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
            text_target = row["relation"][-1] # "_".join(["below", "left", "top", "right", "above"]) #
            prompt_dir = join(figroot, prompt.replace(" ", "_"))
            attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
            attn_stats = th.load(attn_stats_file, weights_only=False)
            cond_stats = attn_stats["cond_stats"]
            cond_stats_list.append(cond_stats)

        cond_stats_array = np.stack(cond_stats_list, axis=0)
        plt.sca(axs[src_idx, query_idx])
        sns.heatmap(cond_stats_array.mean(axis=0))
        if src_idx == 0:
            plt.title(f"Shape1 -> Text relation | {query_str.replace(' == ', '=').replace('relation_str','rel').replace('shape1','S1').replace('shape2','S2').replace('and','&')} | N={len(prompt_part_df)}")
        else:
            plt.title(f"Shape2 -> Text relation | {query_str.replace(' == ', '=').replace('relation_str','rel').replace('shape1','S1').replace('shape2','S2').replace('and','&')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for object -> Text relation", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_object_to_relation_left_right_shape_order_split", fig)
plt.show()

# %% Shape1 -> Shape2
#%%
img_src_str = "shape1"
text_target_str = "shape2"
fig, axs = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape1 == 'square' and color1 == 'blue'",
    "shape1 == 'circle' and color1 == 'blue'",
    "shape1 == 'triangle' and color1 == 'blue'",
    "shape1 == 'square' and color1 == 'red'",
    "shape1 == 'circle' and color1 == 'red'",
    "shape1 == 'triangle' and color1 == 'red'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for shape 1 -> Text of shape2", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_shape1_to_shape2_color_shape_order_split", fig)
plt.show()
# %%
img_src_str = "shape2"
text_target_str = "shape1"
fig, axs = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape2 == 'square' and color2 == 'blue'",
    "shape2 == 'circle' and color2 == 'blue'",
    "shape2 == 'triangle' and color2 == 'blue'",
    "shape2 == 'square' and color2 == 'red'",
    "shape2 == 'circle' and color2 == 'red'",
    "shape2 == 'triangle' and color2 == 'red'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for shape 2 -> Text of shape1", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_shape2_to_shape1_color_shape_order_split", fig)
plt.show()
# %%
img_src_str = "shape2"
text_target_str = "shape1"
fig, axs = plt.subplots(4, 3, figsize=(12, 16), squeeze=False)
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape2 == 'square' and color2 == 'blue' and shape1 == 'circle'",
    "shape2 == 'circle' and color2 == 'blue' and shape1 == 'triangle'",
    "shape2 == 'triangle' and color2 == 'blue' and shape1 == 'square'",
    "shape2 == 'square' and color2 == 'blue' and shape1 == 'triangle'",
    "shape2 == 'circle' and color2 == 'blue' and shape1 == 'square'",
    "shape2 == 'triangle' and color2 == 'blue' and shape1 == 'circle'",
    "shape2 == 'square' and color2 == 'red' and shape1 == 'circle'",
    "shape2 == 'circle' and color2 == 'red' and shape1 == 'triangle'",
    "shape2 == 'triangle' and color2 == 'red' and shape1 == 'square'",
    "shape2 == 'square' and color2 == 'red' and shape1 == 'triangle'",
    "shape2 == 'circle' and color2 == 'red' and shape1 == 'square'",
    "shape2 == 'triangle' and color2 == 'red' and shape1 == 'circle'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=').replace(' and ', ' & ').replace('shape', 'S').replace('color', 'C')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for shape 2 -> Text of shape1", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_shape2_to_shape1_color_shape_shape_order_split", fig)
plt.show()
# %%
img_src_str = "shape1"
text_target_str = "shape2"
fig, axs = plt.subplots(4, 3, figsize=(12, 16), squeeze=False)
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape2 == 'square' and color2 == 'blue' and shape1 == 'circle'",
    "shape2 == 'circle' and color2 == 'blue' and shape1 == 'triangle'",
    "shape2 == 'triangle' and color2 == 'blue' and shape1 == 'square'",
    "shape2 == 'square' and color2 == 'blue' and shape1 == 'triangle'",
    "shape2 == 'circle' and color2 == 'blue' and shape1 == 'square'",
    "shape2 == 'triangle' and color2 == 'blue' and shape1 == 'circle'",
    "shape2 == 'square' and color2 == 'red' and shape1 == 'circle'",
    "shape2 == 'circle' and color2 == 'red' and shape1 == 'triangle'",
    "shape2 == 'triangle' and color2 == 'red' and shape1 == 'square'",
    "shape2 == 'square' and color2 == 'red' and shape1 == 'triangle'",
    "shape2 == 'circle' and color2 == 'red' and shape1 == 'square'",
    "shape2 == 'triangle' and color2 == 'red' and shape1 == 'circle'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=').replace(' and ', ' & ').replace('shape', 'S').replace('color', 'C')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for shape 1 -> Text of shape2", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_shape1_to_shape2_color_shape_shape_order_split", fig)
plt.show()
# %%
### Object to corresponding color
#%%
img_src_str = "shape2"
text_target_str = "color2"
fig, axs = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape2 == 'square' and color2 == 'blue'",
    "shape2 == 'circle' and color2 == 'blue'",
    "shape2 == 'triangle' and color2 == 'blue'",
    "shape2 == 'square' and color2 == 'red'",
    "shape2 == 'circle' and color2 == 'red'",
    "shape2 == 'triangle' and color2 == 'red'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for Shape2 -> Text Color2", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_shape2_to_color2_color_shape2_color2_order_split", fig)
plt.show()
# %%
img_src_str = "shape1"
text_target_str = "color1"
fig, axs = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape1 == 'square' and color1 == 'blue'",
    "shape1 == 'circle' and color1 == 'blue'",
    "shape1 == 'triangle' and color1 == 'blue'",
    "shape1 == 'square' and color1 == 'red'",
    "shape1 == 'circle' and color1 == 'red'",
    "shape1 == 'triangle' and color1 == 'red'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for Shape1 -> Text Color1", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_shape1_to_color1_color_shape1_color1_order_split", fig)
plt.show()
# %%
img_src_str = "shape2"
text_target_str = "color1"
fig, axs = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape2 == 'square' and color2 == 'blue'",
    "shape2 == 'circle' and color2 == 'blue'",
    "shape2 == 'triangle' and color2 == 'blue'",
    "shape2 == 'square' and color2 == 'red'",
    "shape2 == 'circle' and color2 == 'red'",
    "shape2 == 'triangle' and color2 == 'red'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for Shape2 -> Text Color1", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_shape2_to_color1_color_shape2_color1_order_split", fig)
plt.show()

# %%
img_src_str = "shape1"
text_target_str = "color2"
fig, axs = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
axs = axs.flatten()
for query_idx, query_str in enumerate([
    "shape1 == 'square' and color1 == 'blue'",
    "shape1 == 'circle' and color1 == 'blue'",
    "shape1 == 'triangle' and color1 == 'blue'",
    "shape1 == 'square' and color1 == 'red'",
    "shape1 == 'circle' and color1 == 'red'",
    "shape1 == 'triangle' and color1 == 'red'",
]):
    prompt_part_df = prompt_df.query(query_str)
    cond_stats_list = []
    for r_idx in range(len(prompt_part_df)):
        row = prompt_part_df.iloc[r_idx]
        prompt = row["prompt"]
        img_src = row[img_src_str] if img_src_str in row.keys() else img_src_str
        text_target = row[text_target_str] if text_target_str in row.keys() else text_target_str
        prompt_dir = join(figroot, prompt.replace(" ", "_"))
        attn_stats_file = join(prompt_dir, f"cross_attn_layer_head_step_stats_image_{img_src}_to_text_{text_target}.pt")
        attn_stats = th.load(attn_stats_file, weights_only=False)
        cond_stats = attn_stats["cond_stats"]
        cond_stats_list.append(cond_stats)

    cond_stats_array = np.stack(cond_stats_list, axis=0)
    plt.sca(axs[query_idx])
    sns.heatmap(cond_stats_array.mean(axis=0))
    plt.title(f"{query_str.replace(' == ', '=')} | N={len(prompt_part_df)}")
plt.suptitle("Cross-Attention Head Synopses | Img token for Shape1 -> Text Color2", fontsize=12)
plt.tight_layout()
saveallforms(synfigdir, "cross_attn_head_synopsis_shape1_to_color2_color_shape1_color2_order_split", fig)
plt.show()
