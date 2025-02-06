import os
from os.path import join
import torch
import numpy as np
import torch as th
import torch.nn as nn
from tqdm import trange

def tokenize_captions(caption_dir, tokenizer, num_captions=10000, model_max_length=20):
    """Tokenize captions from text files using T5 tokenizer.
    
    Args:
        caption_dir: Directory containing caption text files
        tokenizer: T5 tokenizer instance
        num_captions: Number of captions to process
        model_max_length: Max sequence length for tokenization
        
    Returns:
        input_ids_tsr: Tensor of tokenized input IDs
        attention_mask_col: List of attention masks
    """
    input_ids_col = []
    attention_mask_col = []
    
    for i in trange(num_captions):
        text = open(join(caption_dir, f"{i}.txt")).read()
        text_tokens_and_mask = tokenizer(
            text,
            max_length=model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids_col.append(text_tokens_and_mask['input_ids'])
        attention_mask_col.append(text_tokens_and_mask['attention_mask'])

    input_ids_tsr = th.cat(input_ids_col, dim=0)
    return input_ids_tsr, attention_mask_col


# Create text encoder class
class RandomEmbeddingEncoder(nn.Module):
    def __init__(self, embedding_dict=None, input_ids2dict_ids=None, dict_ids2input_ids=None, text_feat_dir=""):
        super().__init__()
        if embedding_dict is None:
            self.embedding_dict = th.load(join(text_feat_dir, "word_embedding_dict.pt"))["embedding_dict"]
            self.input_ids2dict_ids = th.load(join(text_feat_dir, "word_embedding_dict.pt"))["input_ids2dict_ids"]
            self.dict_ids2input_ids = th.load(join(text_feat_dir, "word_embedding_dict.pt"))["dict_ids2input_ids"]
        else:
            self.embedding_dict = embedding_dict
            self.input_ids2dict_ids = input_ids2dict_ids
            self.dict_ids2input_ids = dict_ids2input_ids
        
    def __call__(self, input_ids, attention_mask=None):
        return self.encode(input_ids, attention_mask)
    
    def encode(self, input_ids, attention_mask=None):
        """Convert input ids to embeddings"""
        if isinstance(input_ids, list):
            input_ids = th.tensor(input_ids)
        # map the input_ids to dict ids 
        indices = th.tensor([self.input_ids2dict_ids[id.item()] for id in input_ids.reshape(-1)]).reshape(input_ids.shape)
        # indices = th.tensor([self.input_ids2dict_ids[id.item()] for id in input_ids])
        embeddings = self.embedding_dict[indices]
        return embeddings, attention_mask
    
    def to(self, device):
        self.embedding_dict = self.embedding_dict.to(device)
        # self.input_ids2dict_ids = self.input_ids2dict_ids.to(device)
        # self.dict_ids2input_ids = self.dict_ids2input_ids.to(device)
        return self

# text_encoder = RandomEmbeddingEncoder().to("cuda")
# # print(text_encoder.encode(input_ids_tsr[0:1]))
# text_emb = text_encoder(input_ids_tsr[0:1])[0]
# load_emb = np.load(join(text_feat_dir, f"0.npz"))["caption_feature"]
# # print(np.load(join(text_feat_dir, f"0.npz"))["caption_feature"])
# assert th.allclose(text_emb.cpu(), th.tensor(load_emb))

def get_positional_encodings(seq_len, d_model, device='cpu'):
    """
    Generate positional encodings for a sequence.

    Args:
        seq_len (int): Length of the sequence.
        d_model (int): Dimension of the model (embedding size).
        device (str): Device to place the tensor on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Positional encodings of shape (seq_len, d_model).
    """
    position = th.arange(seq_len, dtype=th.float, device=device).unsqueeze(1)
    div_term = th.exp(th.arange(0, d_model, 2, dtype=th.float, device=device) *
                         -(th.log(th.tensor(10000.0)) / d_model))
    wpe = th.zeros(seq_len, d_model, device=device)
    wpe[:, 0::2] = th.sin(position * div_term)
    wpe[:, 1::2] = th.cos(position * div_term)
    return wpe


# Create text encoder class
class RandomEmbeddingEncoder_wPosEmb(nn.Module):
    def __init__(self, embedding_dict=None, input_ids2dict_ids=None, dict_ids2input_ids=None, max_seq_len=20, embed_dim=4096, wpe_scale=1, 
                 text_feat_dir=""):
        super().__init__()
        if embedding_dict is None:
            self.embedding_dict = th.load(join(text_feat_dir, "word_embedding_dict.pt"))["embedding_dict"]
            self.input_ids2dict_ids = th.load(join(text_feat_dir, "word_embedding_dict.pt"))["input_ids2dict_ids"]
            self.dict_ids2input_ids = th.load(join(text_feat_dir, "word_embedding_dict.pt"))["dict_ids2input_ids"]
        else:
            self.embedding_dict = embedding_dict
            self.input_ids2dict_ids = input_ids2dict_ids
            self.dict_ids2input_ids = dict_ids2input_ids
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.wpe = get_positional_encodings(self.max_seq_len, self.embed_dim, device="cuda") * wpe_scale
        assert self.wpe.shape == (self.max_seq_len, self.embed_dim)
        assert self.embed_dim == self.embedding_dict.shape[1]
        self.device = self.wpe.device
        self.dtype = self.wpe.dtype
        
    def __call__(self, input_ids, attention_mask=None):
        return self.encode(input_ids, attention_mask)
    
    def encode(self, input_ids, attention_mask=None):
        """Convert input ids to embeddings"""
        if isinstance(input_ids, list):
            input_ids = th.tensor(input_ids)
        # map the input_ids to dict ids 
        indices = th.tensor([self.input_ids2dict_ids[id.item()] for id in input_ids.reshape(-1)]).reshape(input_ids.shape)
        # indices = th.tensor([self.input_ids2dict_ids[id.item()] for id in input_ids])
        embeddings = self.embedding_dict[indices]
        # add positional encoding 
        embeddings = embeddings + self.wpe[:embeddings.shape[1], :]
        return embeddings, attention_mask
    
    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.embedding_dict = self.embedding_dict.to(device=device, dtype=self.dtype)
        self.wpe = self.wpe.to(device=device, dtype=self.dtype)
        # self.input_ids2dict_ids = self.input_ids2dict_ids.to(device)
        # self.dict_ids2input_ids = self.dict_ids2input_ids.to(device)
        return self


@torch.no_grad()
def save_prompt_embeddings_randemb(tokenizer, text_encoder, validation_prompts, prompt_cache_dir="output/tmp/prompt_cache", 
                           device="cuda", max_length=20, t5_path=None, recompute=False):
    """Save T5 text embeddings for a list of prompts to cache directory.
    
    Args:
        validation_prompts (list): List of text prompts to encode
        prompt_cache_dir (str): Directory to save embeddings
        device (str): Device to run encoding on
        max_length (int): Max sequence length for tokenization
        t5_path (str): Path to T5 model. If None, uses default path
    """
    if t5_path is None:
        t5_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl"
    
    result_col = []
    os.makedirs(prompt_cache_dir, exist_ok=True)
    # Load models
    print(f"Loading text encoder and tokenizer from {t5_path} ...")
    # tokenizer = T5Tokenizer.from_pretrained(t5_path)
    # text_encoder = T5EncoderModel.from_pretrained(t5_path).to(device)
    text_encoder = text_encoder.to(device)
    # Save unconditioned embedding
    uncond = tokenizer("", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    uncond_prompt_embeds = text_encoder(uncond.input_ids, attention_mask=uncond.attention_mask)[0]
    torch.save({'caption_embeds': uncond_prompt_embeds, 'emb_mask': uncond.attention_mask, 'prompt': ''}, 
               join(prompt_cache_dir,f'uncond_{max_length}token.pth'))
    result_col.append({'prompt': '', 'caption_embeds': uncond_prompt_embeds, 'emb_mask': uncond.attention_mask})
    print("Preparing Visualization prompt embeddings...")
    print(f"Saving visualizate prompt text embedding at {prompt_cache_dir}")
    for prompt in validation_prompts:
        if os.path.exists(join(prompt_cache_dir,f'{prompt}_{max_length}token.pth')) and not recompute:
            result_col.append(torch.load(join(prompt_cache_dir,f'{prompt}_{max_length}token.pth')))
            continue
        print(f"Mapping {prompt}...")
        caption_token = tokenizer(prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
        caption_emb = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
        torch.save({'caption_embeds': caption_emb, 'emb_mask': caption_token.attention_mask, 'prompt': prompt}, 
                    join(prompt_cache_dir,f'{prompt}_{max_length}token.pth'))
        result_col.append({'prompt': prompt, 'caption_embeds': caption_emb, 'emb_mask': caption_token.attention_mask})
    print("Done!")
    # garbage collection
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    return result_col
