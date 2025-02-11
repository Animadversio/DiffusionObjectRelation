import math
import torch
from typing import Optional
import torch.nn.functional as F
from einops import rearrange
from diffusers.models.attention_processor import Attention

def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    B, H, L, D = query.shape
    _, _, S, _ = key.shape

    scale_factor = 1 / math.sqrt(D) if scale is None else scale

    # Compute attention scores
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1
        )
        attn_weight = attn_weight.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Apply attention mask if provided
    if attn_mask is not None:
        attn_mask = attn_mask.to(query.device)
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask

    # Compute the attention probabilities
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # Apply dropout if specified
    attn_weight = F.dropout(attn_weight, p=dropout_p, training=True)

    # Compute the final attention output
    attn_output = torch.matmul(attn_weight, value)

    return attn_output, attn_weight


class AttnProcessor2_0_Store:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # print(attention_probs.shape, torch.isnan(attention_probs).any())
        # TODO: fix the overflow problem when using fp16, currently no problem with fp32 but fp16 will have memory issue. 
        # hidden_states, attention_probs = scaled_dot_product_attention(
        #     query.to(torch.float32), key.to(torch.float32), value.to(torch.float32), attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        # # assert no nan in attention_probs
        # assert not torch.isnan(attention_probs).any(), "Attention probabilities contain NaNs"
        self.attn_map = attention_probs

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

class PixArtAttentionVisualizer_Store:
    def __init__(self, pipe):
        self.pipe = pipe
        self.activation = defaultdict(list)
        self.hook_handles = []

    def clear_activation(self):
        self.activation = defaultdict(list)
    
    def hook_forger(self, key: str):
        """Create a hook to capture attention patterns"""
        def hook(module, input, output):
            self.activation[key].append(module.processor.attn_map.detach().cpu())
        return hook

    def hook_transformer_attention(self, module, module_id: str):
        """Hook both self-attention and cross-attention modules in PixArt"""
        hooks = []
        # For self-attention (attn1)
        if hasattr(module, 'attn1'):
            h1 = module.attn1.register_forward_hook(self.hook_forger(f"{module_id}_self_attn_map"))
            hooks.extend([h1])

        # For cross-attention (attn2)
        if hasattr(module, 'attn2'):
            h3 = module.attn2.register_forward_hook(self.hook_forger(f"{module_id}_cross_attn_map"))
            hooks.extend([h3])

        return hooks

    def setup_hooks(self):
        """Set up hooks for all transformer blocks"""
        print("Setting up hooks for PixArt attention modules:")

        for block_idx, block in enumerate(self.pipe.transformer.transformer_blocks):
            print(f"- Block {block_idx}")
            hooks = self.hook_transformer_attention(block, f"block{block_idx:02d}")
            self.hook_handles.extend(hooks)

    def cleanup_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        

def replace_attn_processor(transformer):
    for layer in transformer.transformer_blocks:
        layer.attn1.processor = AttnProcessor2_0_Store()
        layer.attn2.processor = AttnProcessor2_0_Store()
        layer.attn1.processor.store_attn_map = True
        layer.attn2.processor.store_attn_map = True
    return transformer