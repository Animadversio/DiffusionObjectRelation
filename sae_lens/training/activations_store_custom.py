from __future__ import annotations

import contextlib
import json
import os
import warnings
from collections.abc import Generator, Iterator
from collections import defaultdict
from typing import Any, Generator, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from jaxtyping import Float
from requests import HTTPError
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sae_lens import logger
from sae_lens.config import (
    DTYPE_MAP,
    CacheActivationsRunnerConfig,
    HfDataset,
    LanguageModelSAERunnerConfig,
)
from sae_lens.sae import SAE
from sae_lens.tokenization_and_batching import concat_and_batch_sequences


# TODO: finish changes
class PixArtActivationsStore:
    def __init__(self, pipe, context_size: int, d_in: int, device: torch.device):
        self.pipe = pipe
        self.context_size = context_size
        self.d_in = d_in
        self.device = device
        
        # Activation storage
        self.activation = defaultdict(list)
        self.hook_handles = []

        # Buffer for batching
        self._storage_buffer = None
        self.n_batches_in_buffer = 16
        self.train_batch_size_tokens = 4096

    def clear_activation(self):
        """Clear all collected activations."""
        self.activation = defaultdict(list)

    def hook_forger(self, key: str):
        """Create a hook to capture activations."""
        def hook(module, input, output):
            self.activation[key].append(output.detach().to(self.device))
        return hook

    def hook_transformer_attention(self, module, module_id: str):
        """Hook both self-attention and cross-attention modules."""
        hooks = []
        if hasattr(module, 'attn1'):
            h1 = module.attn1.to_q.register_forward_hook(self.hook_forger(f"{module_id}_self_Q"))
            h2 = module.attn1.to_k.register_forward_hook(self.hook_forger(f"{module_id}_self_K"))
            hooks.extend([h1, h2])

        if hasattr(module, 'attn2'):
            h3 = module.attn2.to_q.register_forward_hook(self.hook_forger(f"{module_id}_cross_Q"))
            h4 = module.attn2.to_k.register_forward_hook(self.hook_forger(f"{module_id}_cross_K"))
            hooks.extend([h3, h4])

        return hooks

    def setup_hooks(self):
        """Set up hooks for all transformer blocks in the pipeline."""
        for block_idx, block in enumerate(self.pipe.transformer.transformer_blocks):
            hooks = self.hook_transformer_attention(block, f"block{block_idx:02d}")
            self.hook_handles.extend(hooks)

    def cleanup_hooks(self):
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def get_batch_tokens(
        self, prompts: List[str], negative_prompts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Tokenize and return a batch of tokens."""
        tokens = self.pipe.tokenizer(
            prompts, padding="max_length", truncation=True, max_length=self.context_size, return_tensors="pt"
        )
        if negative_prompts:
            neg_tokens = self.pipe.tokenizer(
                negative_prompts, padding="max_length", truncation=True, max_length=self.context_size, return_tensors="pt"
            )
            tokens = torch.cat([tokens["input_ids"], neg_tokens["input_ids"]], dim=0)

        return tokens["input_ids"].to(self.device)

    def get_activations(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        """Run the model and collect activations."""
        with torch.no_grad():
            self.clear_activation()
            self.pipe.transformer(batch_tokens)
        # Gather activations for the specified hook name
        activations = []
        for key, value in self.activation.items():
            activations.append(torch.stack(value, dim=0))
        return torch.cat(activations, dim=-1)  # Combine all hooks

    def refill_buffer(self, prompts: List[str], negative_prompts: Optional[List[str]] = None):
        """Refill the storage buffer with activations from the dataset."""
        self._storage_buffer = []
        for _ in range(self.n_batches_in_buffer):
            batch_tokens = self.get_batch_tokens(prompts, negative_prompts)
            activations = self.get_activations(batch_tokens)
            self._storage_buffer.append(activations)
        self._storage_buffer = torch.cat(self._storage_buffer, dim=0)

    def get_data_loader(self, prompts: List[str], negative_prompts: Optional[List[str]] = None):
        """Return a DataLoader to iterate over activations."""
        if self._storage_buffer is None:
            self.refill_buffer(prompts, negative_prompts)
        return DataLoader(self._storage_buffer, batch_size=self.train_batch_size_tokens, shuffle=True)

    def normalize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Normalize activations (if needed)."""
        norm_factor = (self.d_in ** 0.5) / activations.norm(dim=-1).mean()
        return activations * norm_factor
