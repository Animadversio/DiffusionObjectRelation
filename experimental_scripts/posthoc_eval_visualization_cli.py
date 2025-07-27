#%%
#ipython magic to autoreload
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    print("autoreload enabled")
except NameError:
    # Not in IPython environment, skip autoreload
    pass

#%%
import os
import sys
sys.path.append("/n/home12/binxuwang/Github/DiffusionObjectRelation")
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from circuit_toolkit.plot_utils import saveallforms

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



prompts, parsed_words = generate_test_prompts_collection_and_parsed_words()
prompt_df = pd.DataFrame(parsed_words)
prompt_df["relation_str"] = prompt_df["relation"].apply(lambda x: "_".join(x))
#%%
model_run_name = "objrel_T5_DiT_mini_pilot"
for model_run_name in ["objrel_T5_DiT_mini_pilot", 
                       "objrel_T5_DiT_B_pilot",
                       "objrel_rndembdposemb_DiT_mini_pilot",
                       "objrel_rndembdposemb_DiT_B_pilot"]:
    step_num = 160000
    ckpt_ver = "ema"

    savedir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/{model_run_name}"
    synsavedir = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/{model_run_name}"
    eval_dir = join(savedir, "large_scale_eval_posthoc")
    output_dir = join(synsavedir, "large_scale_eval_posthoc_synopsis")
    os.makedirs(output_dir, exist_ok=True)

    eval_df_all = pd.read_csv(join(eval_dir, f"eval_df_all_train_step{step_num}_{ckpt_ver}_prompts.csv"))
    object_df_all = pd.read_pickle(join(eval_dir, f"object_df_all_train_step{step_num}_{ckpt_ver}_prompts.pkl"))

    # %
    eval_syn_df = eval_df_all.groupby("prompt_id").mean(numeric_only=True)
    eval_syn_df = eval_syn_df.rename(columns={'overall': 'overall_acc', 'shape': 'shape_acc', 'color': 'color_acc', 'spatial_relationship': 'spatial_relationship_acc'})
    prompt_eval_syn_df = prompt_df.merge(eval_syn_df, left_index=True, right_index=True)
    prompt_eval_syn_df.to_pickle(join(eval_dir, f"prompt_eval_syn_df.pkl"))

    print(prompt_eval_syn_df.head())
    print("starting heatmap")

    # %
    # Create a combined figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Accuracy by Prompt Spatial Relationship
    pivot_df = prompt_eval_syn_df.pivot_table(
        index=['color1', 'shape1', 'color2', 'shape2'], 
        columns='relation_str', 
        values='overall_acc',
        aggfunc='mean'
    )
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Accuracy'}, ax=axes[0, 0])
    axes[0, 0].set_title(f'Overall Accuracy')
    axes[0, 0].set_xlabel('Prompt Spatial Relationship')
    axes[0, 0].set_ylabel('Object Attributes')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), ha='right')

    # Plot 2: Color Accuracy by Prompt Spatial Relationship
    pivot_df = prompt_eval_syn_df.pivot_table(
        index=['color1', 'shape1', 'color2', 'shape2'], 
        columns='relation_str', 
        values='color_acc',
        aggfunc='mean'
    )
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Color Accuracy'}, ax=axes[0, 1])
    axes[0, 1].set_title(f'Color Accuracy')
    axes[0, 1].set_xlabel('Prompt Spatial Relationship')
    axes[0, 1].set_ylabel('Object Attributes')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), ha='right')
    # Get rid of the y ticks
    axes[0, 1].set_yticks([])

    # Plot 3: Shape Accuracy by Prompt Spatial Relationship
    pivot_df = prompt_eval_syn_df.pivot_table(
        index=['color1', 'shape1', 'color2', 'shape2'], 
        columns='relation_str', 
        values='shape_acc',
        aggfunc='mean'
    )
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Shape Accuracy'}, ax=axes[0, 2])
    axes[0, 2].set_title(f'Shape Accuracy')
    axes[0, 2].set_xlabel('Prompt Spatial Relationship')
    axes[0, 2].set_ylabel('Object Attributes')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), ha='right')
    # Get rid of the y ticks
    axes[0, 2].set_yticks([])

    # Plot 4: Spatial Relationship Accuracy
    pivot_df = prompt_eval_syn_df.pivot_table(
        index=['color1', 'shape1', 'color2', 'shape2'], 
        columns='relation_str', 
        values='spatial_relationship_acc',
        aggfunc='mean'
    )
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Spatial Accuracy'}, ax=axes[1, 0])
    axes[1, 0].set_title(f'Spatial Relationship Accuracy')
    axes[1, 0].set_xlabel('Prompt Spatial Relationship')
    axes[1, 0].set_ylabel('Object Attributes')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), ha='right')

    # Plot 5: Dx by Object Attributes
    pivot_df = prompt_eval_syn_df.pivot_table(
        index=['color1', 'shape1', 'color2', 'shape2'], 
        columns='relation_str', 
        values='Dx',
        aggfunc='mean'
    )
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', cbar_kws={'label': 'Dx'}, ax=axes[1, 1])
    axes[1, 1].set_title(f'Dx')
    axes[1, 1].set_xlabel('Prompt Spatial Relationship')
    axes[1, 1].set_ylabel('Object Attributes')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), ha='right')
    # Get rid of the y ticks
    axes[1, 1].set_yticks([])

    # Plot 6: Dy by Object Attributes
    pivot_df = prompt_eval_syn_df.pivot_table(
        index=['color1', 'shape1', 'color2', 'shape2'], 
        columns='relation_str', 
        values='Dy',
        aggfunc='mean'
    )
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', cbar_kws={'label': 'Dy'}, ax=axes[1, 2])
    axes[1, 2].set_title(f'Dy')
    axes[1, 2].set_xlabel('Prompt Spatial Relationship')
    axes[1, 2].set_ylabel('Object Attributes')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), ha='right')
    # Get rid of the y ticks
    axes[1, 2].set_yticks([])

    plt.suptitle(f'Combined Evaluation Results (mean of 100 samples per prompt)\n{model_run_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    saveallforms(output_dir, f"relation_eval_heatmap_all_synopsis_{model_run_name}_step{step_num}_{ckpt_ver}")
    plt.show()
# %%
# Create heatmap
# Pivot the data to create a heatmap with spatial relationships as columns
pivot_df = prompt_eval_syn_df.pivot_table(
    index=['color1', 'shape1', 'color2', 'shape2'], 
    columns='relation_str', 
    values='spatial_relationship_acc',
    aggfunc='mean'
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Spatial Relationship Accuracy'})
plt.title(f'Spatial Relationship Accuracy by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
plt.xlabel('Prompt Spatial Relationship')
plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
saveallforms(output_dir, f"relation_eval_heatmap_spatial_relationship_acc_{model_run_name}_step{step_num}_{ckpt_ver}")
plt.show()


# %%
# Pivot the data to create a heatmap with spatial relationships as columns
pivot_df = prompt_eval_syn_df.pivot_table(
    index=['color1', 'shape1', 'color2', 'shape2'], 
    columns='relation_str', 
    values='shape_acc',
    aggfunc='mean'
)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Shape Accuracy'})
plt.title(f'Shape Accuracy by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
plt.xlabel('Prompt Spatial Relationship')
plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
saveallforms(output_dir, f"relation_eval_heatmap_shape_acc_{model_run_name}_step{step_num}_{ckpt_ver}")
plt.show()

# %%

# Pivot the data to create a heatmap with spatial relationships as columns
pivot_df = prompt_eval_syn_df.pivot_table(
    index=['color1', 'shape1', 'color2', 'shape2'], 
    columns='relation_str', 
    values='color_acc',
    aggfunc='mean'
)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Color Accuracy'})
plt.title(f'Color Accuracy by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
plt.xlabel('Prompt Spatial Relationship')
plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
saveallforms(output_dir, f"relation_eval_heatmap_color_acc_{model_run_name}_step{step_num}_{ckpt_ver}")
plt.show()

# %%

# Pivot the data to create a heatmap with spatial relationships as columns
pivot_df = prompt_eval_syn_df.pivot_table(
    index=['color1', 'shape1', 'color2', 'shape2'], 
    columns='relation_str', 
    values='Dx',
    aggfunc='mean'
)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', cbar_kws={'label': 'Dx'})
plt.title(f'Dx by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
plt.xlabel('Prompt Spatial Relationship')
plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
saveallforms(output_dir, f"relation_eval_heatmap_Dx_{model_run_name}_step{step_num}_{ckpt_ver}")
plt.show()

# %%

# Pivot the data to create a heatmap with spatial relationships as columns
pivot_df = prompt_eval_syn_df.pivot_table(
    index=['color1', 'shape1', 'color2', 'shape2'], 
    columns='relation_str', 
    values='Dy',
    aggfunc='mean'
)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', cbar_kws={'label': 'Dy'})
plt.title(f'Dy by Object Attributes (mean of 100 samples per prompt)\n{model_run_name}')
plt.xlabel('Prompt Spatial Relationship')
plt.ylabel('Object Attributes (color1, shape1, color2, shape2)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
saveallforms(output_dir, f"relation_eval_heatmap_Dy_{model_run_name}_step{step_num}_{ckpt_ver}")
plt.show()
