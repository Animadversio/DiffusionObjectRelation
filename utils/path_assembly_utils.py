import os
from os.path import join
from typing import Tuple, Union
import torch
import torch as th
from transformers import T5Tokenizer, T5EncoderModel
from utils.text_encoder_control_lib import RandomEmbeddingEncoder_wPosEmb, RandomEmbeddingEncoder



def load_text_encoder(
    text_encoder_type: str,
    text_feat_dir_path: str = None,
    t5_path: str = None,
    device: str = "cuda"
) -> Union[T5EncoderModel, RandomEmbeddingEncoder_wPosEmb, RandomEmbeddingEncoder]:
    """
    Load text encoder based on the specified type.
    
    Args:
        text_encoder_type: Type of text encoder ("T5", "RandomEmbeddingEncoder_wPosEmb", "RandomEmbeddingEncoder")
        text_feat_dir_path: Path to text feature directory (for random embedding encoders)
        t5_path: Path to T5 model directory (for T5 encoder)
        device: Device to load the encoder on
    
    Returns:
        text_encoder: Loaded text encoder
    """
    
    if text_encoder_type == "T5":
        # Set default T5 path if not provided
        
        # Load T5 tokenizer and encoder
        tokenizer = T5Tokenizer.from_pretrained(t5_path)
        T5_dtype = torch.bfloat16
        text_encoder = T5EncoderModel.from_pretrained(
            t5_path, 
            load_in_8bit=False, 
            torch_dtype=T5_dtype
        ).to(device)
        
        return text_encoder
        
    elif text_encoder_type == "RandomEmbeddingEncoder_wPosEmb":
        # Set default text feature directory path if not provided
        
        # Load random embedding data
        emb_data = th.load(join(text_feat_dir_path, "word_embedding_dict.pt"))
        
        # Create random embedding encoder with position embeddings
        text_encoder = RandomEmbeddingEncoder_wPosEmb(
            emb_data["embedding_dict"], 
            emb_data["input_ids2dict_ids"], 
            emb_data["dict_ids2input_ids"], 
            max_seq_len=20, 
            embed_dim=4096,
            wpe_scale=1/6
        ).to(device)
        
        return text_encoder
        
    elif text_encoder_type == "RandomEmbeddingEncoder":
        # Set default text feature directory path if not provided
        
        # Load random embedding data
        emb_data = th.load(join(text_feat_dir_path, "word_embedding_dict.pt"))
        
        # Create random embedding encoder without position embeddings
        text_encoder = RandomEmbeddingEncoder(
            emb_data["embedding_dict"], 
            emb_data["input_ids2dict_ids"], 
            emb_data["dict_ids2input_ids"]
        ).to(device)
        
        return text_encoder
        
    else:
        raise ValueError(f"Unknown text_encoder_type: {text_encoder_type}. "
                       f"Supported types: 'T5', 'RandomEmbeddingEncoder_wPosEmb', 'RandomEmbeddingEncoder'")


