# %% [markdown]
# # T5 tokenization + T5 embedding

"""
data_generation.py

Generates synthetic datasets with captions, extract features, and prepared for 
training by converting to embeddings. 

Inputs: 
- config.py: specifies dataset specifications 
- FUTURE: Command-line arguments: can override any config value. 

Outputs:
- Generated images and captions
- Feature files (VAE, T5, or random embeddings)
- Data info JSON for dataset partitioning

Usage: 
    python unified_data_gen_process.py 
    python unified_data_gen_process.py --config myconfig.json 
    python unified_data_gen_process.py --config config.json --num_images 10000 ...


Authors: Hannah Kim, Binxu Wang
Date: 2025-07-25
"""
#%%
%load_ext autoreload
%autoreload 2

# %%
import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import torch as th
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from os.path import join

from tqdm.notebook import tqdm, trange

from transformers import T5Tokenizer, T5EncoderModel

from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize, CenterCrop

from diffusers import AutoencoderKL

from datasets import load_dataset


# Set up sys.path for custom modules
sys.path.append("/n/home12/hjkim/Github/DiffusionObjectRelation/PixArt-alpha")
sys.path.append("/n/home12/hjkim/Github/DiffusionObjectRelation/")

from diffusion.model.t5 import T5Embedder
from types import SimpleNamespace
from tools.extract_features import extract_caption_t5, extract_img_vae  # , get_args
from utils.text_encoder_control_lib import save_prompt_embeddings_randemb
from utils.dataset_generation_lib import SingleShapeDataset, DoubleShapeDataset, ShapesDatasetCached, MixedShapesDataset
from utils.custom_text_encoding_utils import RandomEmbeddingEncoder_wPosEmb, RandomEmbeddingEncoder, tokenize_captions

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% Define correct dataset configuration 

from configs.dataset_generation_config import * # All information here! 
from configs.exp1_single_t5 import * # All information here! 
# from configs.exp2_single_random import * # All information here! 
from configs.exp3_double_t5 import * # All information here! 
# from configs.exp4_double_random import * # All information here! 
from configs.exp5_mixed_t5 import * # All information here! 
# from configs.exp6_mixed_random import * # All information here! 
# from configs.dataset_generation_config import dataset_type

print(f"Using dataset configuration: {dataset_name}")
print(f"Imported from config:")
print(f"- num_images: {num_images}")
print(f"- resolution: {resolution}")
print(f"- radius: {radius}")
print(f"- single_ratio: {single_ratio}")
print(f"- model_max_length: {model_max_length}")
print(f"- dataset_type: {dataset_type}")
print(f"- encoder_type: {encoder_type}")
print(f"- dataset_name: {dataset_name}")
print(f"- pixart_dir: {pixart_dir}")
print(f"- save_dir: {save_dir}")
print(f"- using_existing_img_txt: {using_existing_img_txt}")
print(f"- existing_dataset_name: {existing_dataset_name}")


dataset_class_map = {
    "Single": SingleShapeDataset,
    "Double": DoubleShapeDataset,
    "Mixed": MixedShapesDataset
}


ObjRelDataset = dataset_class_map[dataset_type]

if dataset_type == "Mixed":
    dataset_kwargs = dict(
        radius=radius, 
        resolution=resolution,
        single_ratio=single_ratio 
    )
else:
    dataset_kwargs = dict(radius=radius, resolution=resolution)

root_dir = f"{pixart_dir}/training_datasets/{dataset_name}"
prompt_cache_name = f'{dataset_name}_{encoder_type}emb_{model_max_length}token'
prompt_cache_dir = f"{pixart_dir}/output/{prompt_cache_name}"
existing_dataset_root =f"{pixart_dir}/training_datasets/{existing_dataset_name}"


# %% Generate the dataset 

if not using_existing_img_txt:
    print(f"Generating dataset from scratch")
    #Create the dataset from scratch 
    transform = transforms.Compose([
        lambda x: x.convert("RGB"),
        # Add more transforms if needed
    ])
    
    # Create dataset with unified interface
    dataset = ObjRelDataset(
        num_images=num_images, 
        **dataset_kwargs, 
        transform=transform
    )

    images_dir_absolute = join(root_dir, "images")
    captions_dir_absolute = join(root_dir, "captions")

    os.makedirs(images_dir_absolute, exist_ok=True)
    os.makedirs(captions_dir_absolute, exist_ok=True)

    image_format = "png"
    json_name = "partition/data_info.json"
    if not os.path.exists(join(root_dir, "partition")):
        os.makedirs(join(root_dir, "partition"))

    absolute_json_name = join(root_dir, json_name)
    data_info = []

    for order, (image, labels) in tqdm(enumerate(dataset)): 
        if order >= 10000:
            break
        image = image
        image.save(f"{images_dir_absolute}/{order}.{image_format}")
        with open(f"{captions_dir_absolute}/{order}.txt", "w") as text_file:
            text_file.write(labels["caption"])
        
        width, height = resolution, resolution
        ratio = 1
        data_info.append({
            "height": height,
            "width": width,
            "ratio": ratio,
            "path": f"{order}.{image_format}", 
            "prompt": labels["caption"],
        })        

    with open(absolute_json_name, "w") as json_file:
        json.dump(data_info, json_file)
else:
    print(f"Using existing dataset: {existing_dataset_name}, copying from {existing_dataset_root} to {root_dir}")
    import shutil
    # Copy from the existing img_txt_dir
    # Copy files from existing img_txt_dir to new location
    src_images_dir = join(existing_dataset_root, "images")
    src_captions_dir = join(existing_dataset_root, "captions")
    src_partition_dir = join(existing_dataset_root, "partition")

    # Create target directories
    images_dir_absolute = join(root_dir, "images") 
    captions_dir_absolute = join(root_dir, "captions")
    partition_dir_absolute = join(root_dir, "partition")

    os.makedirs(images_dir_absolute, exist_ok=True)
    os.makedirs(captions_dir_absolute, exist_ok=True) 
    os.makedirs(partition_dir_absolute, exist_ok=True)

    # Copy all files from source to target directories
    for src_dir, dst_dir in [(src_images_dir, images_dir_absolute),
                            (src_captions_dir, captions_dir_absolute),
                            (src_partition_dir, partition_dir_absolute)]:
        for filename in os.listdir(src_dir):
            src_file = join(src_dir, filename)
            dst_file = join(dst_dir, filename)
            shutil.copy2(src_file, dst_file)

    # Update absolute_json_name to point to copied location
    absolute_json_name = join(partition_dir_absolute, "data_info.json")

# %% [markdown]
# # Extract Features 
# Manually create the args object
args = SimpleNamespace(
    multi_scale=False,
    img_size=resolution,
    max_tokens=model_max_length,
    start_index=0,
    end_index=1000000,
    json_path=absolute_json_name,
    t5_save_root=os.path.join(root_dir, "caption_feature_wmask"),
    vae_save_root=os.path.join(root_dir, "img_vae_features"),
    dataset_root=root_dir,
    pretrained_models_dir=f"{pixart_dir}/output/pretrained_models",
    json_file=''
)

# Set globals expected by the script
# Set these as globals for the imported module
import tools.extract_features as extract_features

extract_features.args = args
extract_features.device = device
extract_features.image_resize = resolution

print(f'Extracting Single Image Resolution {resolution}')
extract_img_vae()
    

# %% Tokenization and Embeddings

caption_feature_dir =  "caption_feature_wmask"
word_embedding_dir = "word_embedding_dict.pt"

if encoder_type == "T5":
    # T5_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.T5_dtype]

    T5_path = f"{pixart_dir}/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"
    tokenizer = T5Tokenizer.from_pretrained(T5_path)
    T5_dtype = torch.bfloat16
    text_encoder = T5EncoderModel.from_pretrained(
                        T5_path, 
                        load_in_8bit=False, 
                        torch_dtype=T5_dtype)
elif encoder_type == "RandomEmbeddingEncoder_wPosEmb":
    text_feat_dir_old = f'{pixart_dir}/training_datasets/objectRel_pilot_rndemb/{caption_feature_dir}'
    emb_data = th.load(join(text_feat_dir_old, word_embedding_dir)) 
    text_encoder = RandomEmbeddingEncoder_wPosEmb(
                        emb_data["embedding_dict"], 
                        emb_data["input_ids2dict_ids"], 
                        emb_data["dict_ids2input_ids"], 
                        max_seq_len=20, embed_dim=4096,
                        wpe_scale=1/6).to("cuda")
elif encoder_type == "RandomEmbeddingEncoder":
    text_feat_dir_old = f'{pixart_dir}/training_datasets/objectRel_pilot_rndemb/{caption_feature_dir}'
    emb_data = th.load(join(text_feat_dir_old, word_embedding_dir)) 
    text_encoder = RandomEmbeddingEncoder(
                        emb_data["embedding_dict"], 
                        emb_data["input_ids2dict_ids"], 
                        emb_data["dict_ids2input_ids"], 
                        ).to("cuda")
else:
    raise NotImplementedError(f"Encoder type {encoder_type} not implemented")

dataset_root = root_dir
prompt_cache_dir = f"{pixart_dir}/output/{prompt_cache_name}"
caption_dir = join(root_dir, "captions")
image_dir = join(root_dir, "images")
img_feat_dir = join(root_dir, "img_vae_features_128resolution")
text_feat_dir = join(root_dir, "caption_feature_wmask")

# %% Process the embeddings 

#Tokenize the captions
if encoder_type == 'T5':
    print(f"Extracting caption t5 features for {encoder_type} encoder")
    # if text_feat_dir exists, clean it
    if os.path.exists(text_feat_dir):
        shutil.rmtree(text_feat_dir)
    os.makedirs(text_feat_dir, exist_ok=True)
    extract_caption_t5(max_tokens=model_max_length)
    
else:
    print(f"Tokenizing captions for {encoder_type} encoder")
    if not os.path.exists(text_feat_dir):
        os.makedirs(text_feat_dir, exist_ok=True)

    input_ids_tsr, attention_mask_col = tokenize_captions(
                                            caption_dir, 
                                            tokenizer, 
                                            num_captions=num_images, 
                                            model_max_length=model_max_length)
    
    for sample_idx in trange(num_images):
        input_ids = input_ids_tsr[sample_idx]
        embeddings = text_encoder(input_ids)[0].cpu()
        np.savez(join(text_feat_dir, 
                    f"{sample_idx}.npz"), 
                    caption_feature=embeddings.numpy(), 
                    attention_mask=attention_mask_col[sample_idx].numpy())


# %% Generate and save the validation prompt embeddings 
caption_embeddings = save_prompt_embeddings_randemb(
                        tokenizer, 
                        text_encoder, 
                        validation_prompts, 
                        prompt_cache_dir, 
                        device="cuda", 
                        max_length=model_max_length, 
                        t5_path=T5_path, 
                        recompute=True)

for i, embedding in enumerate(caption_embeddings):
    print(f"{i}: {embedding['prompt']} | token num:{embedding['emb_mask'].sum()}")

torch.save(caption_embeddings, 
            join(prompt_cache_dir, "caption_embeddings_list.pth"))

print(f"Saved caption embeddings to {prompt_cache_dir}")




#%% Verify the validation prompt embeddings 
# caption_embeddings = torch.load(join(prompt_cache_dir, "caption_embeddings_list.pth"))

# for i, embedding in enumerate(caption_embeddings):
#     print(f"{i}: {embedding['prompt']} | token num:{embedding['emb_mask'].sum()}")

# %%




# Define save path
# cached_datset_save_path = os.path.join(save_dir, f"{dataset_name}_cached_dataset.pth")
# os.makedirs(cached_datset_save_path, exist_ok=True)


#%% Scratch

#Get arguments from command line first. 
# parser = argparse.ArgumentParser()

# parser.add_argument('--config', type=str, default='dataset_generation_config.py')
# parser.add_argument('--num_images', type=int)
# parser.add_argument('--resolution', type=int)
# parser.add_argument('--radius', type=int)
# parser.add_argument('--ObjRelDataset', type=str)
# parser.add_argument('--encoder_type', type=str)
# parser.add_argument('--model_max_length', type=int)
# parser.add_argument('--dataset_name', type=str)

# #Collect the arguments from the CLI
# args = parser.parse_args()


#Override config file with CLI arguments if necessary
# for key in vars(args):
#     value = getattr(args, key)
#     if value is not None and key != "config":
#         key = k

#%%

# %% 
# %cd ../PixArt-alpha
# python ~/Github/DiffusionObjectRelation/PixArt-alpha/tools/extract_features.py \
#     --img_size 128 \
#     --max_tokens 20 \
#     --dataset_root "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRelSingle_randomized_pilot1" \
#     --json_path "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRelSingle_randomized_pilot1/partition/data_info.json" \
#     --t5_save_root "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRelSingle_randomized_pixel1/caption_feature_wmask" \
#     --vae_save_root "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRelSingle_randomized_pilot1/img_vae_features" \
#     --pretrained_models_dir "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models"

# %% [markdown]
# # Creating a dataset of only single object images


# %%
# %%
# Test the class to make sure the dataset images are properly formatted. 


# # Define transform
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# # Create dataset and dataloader
# dataset = SingleShapeDataset(num_images=10000, transform=transform)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
# if False:
#     # Inspect one batch
#     for images, labels in dataloader:
#         print(f"Images batch shape: {images.size()}")
#         print(f"Labels batch: {labels}")
#         print("Sample of batch captions:")
#         for i in range(min(3, len(images))):
#             print(f"Image {i}: {labels['caption'][i]}")
#         break

#     # Display a few images and captions
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#     axes = axes.ravel()

#     for i in range(4):
#         img, lbl = dataset[i]
#         img_np = img.permute(1, 2, 0).numpy()
        
#         axes[i].imshow(img_np)
#         axes[i].set_title(lbl['caption'], fontsize=10)
#         axes[i].axis('off')

#     plt.tight_layout()
#     plt.show()


# # %%
# # Create the single-object dataset

# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# dataset = ObjRelDataset(num_images=10000, transform=transform, radius=16, resolution=128)

# # Sample 10,000 images and store images + labels
# images = []
# labels = []

# for i in range(10000):
#     img, lbl = dataset[i]
#     images.append(img)
#     labels.append(lbl)

# # Stack all images into a single tensor
# image_tensors = torch.stack(images)  # shape: [10000, 3, H, W]
# # Inspect image statistics
# print("Mean pixel value:", image_tensors.mean().item())
# print("Std dev of pixel values:", image_tensors.std().item())
# # Extract single-object labels
# shapes = torch.tensor([item['shape'] for item in labels], dtype=torch.long)
# locations = torch.stack([item['location'] for item in labels], dim=0)
# # if False:
# #     # Create a TensorDataset (no need for shape2/location2)
# #     dataset = TensorDataset(image_tensors, shapes, locations)
# #     # Create DataLoader
# #     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
# #     # Inspect a batch
# #     batch = next(iter(dataloader))
# #     images_batch, shapes_batch, locations_batch = batch

# #     print("Image batch shape:", images_batch.shape)
# #     print("Shape indices:", shapes_batch)
# #     print("Locations:", locations_batch)

# ### Cache and save the dataset to disk
# # Save as a dictionary
# torch.save({
#     "images": image_tensors,     # shape: [N, 3, H, W]
#     "shapes": shapes,            # shape: [N]
#     "locations": locations       # shape: [N, 2]
# }, cached_datset_save_path)
# print(f"Saved single-object dataset to: {cached_datset_save_path}")


# # %%
# # # Load the cached dataset and encode images using VAE

# # %%
# transform = transforms.Compose([
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
# ])
# dataset = SingleShapesDatasetCached(
#     path=cached_datset_save_path,
#     transform=transform
# )
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# # %%
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
# vae = AutoencoderKL.from_pretrained(vae_model).to(device)

# # %%
# if False:
#     image_tsr = dataset[0][0].unsqueeze(0)
#     with torch.no_grad():
#         latent_dist = vae.encode(image_tsr.to(device)).latent_dist # * 2 - 1
#         latent_sample = latent_dist.sample()
#         image_rec = (vae.decode(latent_sample).sample * 0.5 + 0.5).clamp(0,1)
#         # latent_mean = latent_dist.mean
#         # image_rec = vae.decode(latent_mean).sample * 0.5 + 0.5
#     image_rec.shape

#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image_rec[0].permute(1, 2, 0).cpu().numpy())
#     plt.subplot(1, 2, 2)
#     plt.imshow((image_tsr[0] * 0.5 + 0.5).clamp(0,1).permute(1, 2, 0).cpu().numpy())

#     # %% [markdown]
#     # # Make it into a pixart trainable dataset

#     # %%

#     # %%
#     tfm = Compose([Normalize(0.5,0.5)])

#     # %%
#     tfm(ToTensor()(ToPILImage()(torch.randn(3, 128, 128))))
# %% [markdown]
# Note that with vision, the formats matter! PIL image != tensors. These are two very different formats. 


# %%
# max_length = model_max_length
# result_col = []
# recompute = True 
# os.makedirs(prompt_cache_dir, exist_ok=True)

# pretrain_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/"
# t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=f'{pretrain_path}/t5_ckpts', model_max_length=max_length)

# # Save unconditioned embedding
# uncond_prompt_embeds, uncond_attention_mask = t5.get_text_embeddings([""], )
# torch.save({'caption_embeddings': uncond_prompt_embeds, 'emb_mask': uncond_attention_mask, 'prompt': ''}, 
#             join(prompt_cache_dir,f'uncond_{max_length}token.pth'))
# result_col.append({'prompt': '', 'caption_embeddingss': uncond_prompt_embeds, 'emb_mask': uncond_attention_mask})

# print("Preparing Visualization prompt embeddings...")
# print(f"Saving visualizate prompt text embedding at {prompt_cache_dir}")

# for prompt in validation_prompts:
#     if os.path.exists(join(prompt_cache_dir,f'{prompt}_{max_length}token.pth')) and not recompute:
#         result_col.append(torch.load(join(prompt_cache_dir,f'{prompt}_{max_length}token.pth')))
#         continue
#     print(f"Mapping {prompt}...")
#     caption_emb, caption_token_attention_mask =  t5.get_text_embeddings([prompt], )
#     torch.save({'caption_embeddings': caption_emb, 'emb_mask': caption_token_attention_mask, 'prompt': prompt}, 
#                 join(prompt_cache_dir,f'{prompt}_{max_length}token.pth'))
#     result_col.append({'prompt': prompt, 'caption_embeddings': caption_emb, 'emb_mask': caption_token_attention_mask})
# print("Done!")
