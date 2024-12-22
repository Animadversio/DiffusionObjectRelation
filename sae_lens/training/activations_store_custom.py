from __future__ import annotations

import contextlib
import json
import os
import pickle
import warnings
from collections.abc import Generator, Iterator
from typing import Any, Literal, cast

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


# TODO: Make an activation store config class to be consistent with the rest of the code.
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """
    cached_activations_path: str | None
    cached_activation_dataset: Dataset | None = None
    tokens_column: Literal["tokens", "input_ids", "text", "problem"]
    _dataloader: Iterator[Any] | None = None
    _storage_buffer: torch.Tensor | None = None
    device: torch.device

    @classmethod
    def from_cache_activations(
        cls,
        cfg: LanguageModelSAERunnerConfig,
    ) -> ActivationsStore:
        """
        Public api to create an ActivationsStore from a cached activations dataset.
        """
        return cls(
            cached_activations_path=cfg.cached_activations_path,
            dtype=cfg.dtype,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            time_step=cfg.time_step,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.model_batch_size,  # get_buffer
            train_batch_size_tokens=cfg.model_batch_size,  # dataloader
            seqpos_slice=(None,),
            device=torch.device(cfg.device),  # since we're sending these to SAE
            # NOOP
            prepend_bos=False,
            normalize_activations="none",
            autocast_lm=False,
        )

    def __init__(
        self,
        context_size: int,
        d_in: int,
        n_batches_in_buffer: int,
        total_training_tokens: int,
        store_batch_size_prompts: int,
        train_batch_size_tokens: int,
        prepend_bos: bool,
        normalize_activations: str,
        device: torch.device,
        dtype: str,
        time_step: int,
        cached_activations_path: str | None = None,
        autocast_lm: bool = False,
        seqpos_slice: tuple[int | None, ...] = (None,),
    ):
    
        self.context_size = context_size
        self.d_in = d_in
        self.n_batches_in_buffer = n_batches_in_buffer
        self.half_buffer_size = n_batches_in_buffer // 2
        self.total_training_tokens = total_training_tokens
        self.store_batch_size_prompts = store_batch_size_prompts
        self.train_batch_size_tokens = train_batch_size_tokens
        self.prepend_bos = prepend_bos
        self.normalize_activations = normalize_activations
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.cached_activations_path = cached_activations_path
        self.autocast_lm = autocast_lm
        self.seqpos_slice = seqpos_slice
        self.n_dataset_processed = 0
        self.estimated_norm_scaling_factor = None
        self.cached_activation_dataset = self.load_cached_activation_dataset()
        self.time_step = time_step

        # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)


    def load_and_reshape_latent(self):
        """
        Loads a latent tensor from a .pkl file, extracts the block_11_residual_spatial_state_traj,
        selects a specific time step, and reshapes the tensor to batch x sequence length x feature dimension.

        Parameters:
            pkl_path (str): Path to the .pkl file.
            time_step (int): The specific time step to extract from the latent tensor (first dimension).

        Returns:
            reshaped_tensor (numpy.ndarray): Reshaped tensor of shape [batch_size, sequence_len, feature_dim].
        """
        try:
            # Load the .pkl file
            with open(self.cached_activations_path, 'rb') as f:
                data = pickle.load(f)

            # Extract the latent tensor
            latent = data['block_11_residual_spatial_state_traj']  # Shape: [time_steps, batch, 8, 8, feature_dim]

            # Check if the time step is valid
            if self.time_step >= latent.shape[0]:
                raise ValueError(f"Invalid time_step {self.time_step}. Maximum allowed is {latent.shape[0] - 1}.")

            # Extract the tensor for the given time step
            time_step_tensor = latent[self.time_step]  # Shape: [batch, 8, 8, feature_dim]

            # Reshape the tensor to [batch, sequence_len (8*8), feature_dim]
            batch_size, height, width, feature_dim = time_step_tensor.shape
            sequence_len = height * width
            reshaped_tensor = time_step_tensor.reshape(batch_size, sequence_len, feature_dim)

            return reshaped_tensor

        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {self.cached_activations_path} was not found.")
        except KeyError:
            raise KeyError("The required key 'block_11_residual_spatial_state_traj' is missing in the .pkl file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the file: {e}")



    def load_cached_activation_dataset(self) -> torch.Tensor | None:
        """
        Load the cached activation dataset from disk.

        - If cached_activations_path is set, returns a PyTorch tensor
        with shape (samples, seq, embedding).
        - Assumes the dataset corresponds to a single hook/layer.
        """
        if self.cached_activations_path is None:
            return None

        assert os.path.exists(self.cached_activations_path), (
            f"Cache file {self.cached_activations_path} does not exist. "
            f"Ensure the correct path is provided."
        )

        # Load tensor from disk
        activations_tensor = self.load_and_reshape_latent()
        
        # Ensure it's the expected type and shape
        assert isinstance(activations_tensor, torch.Tensor), (
            f"Loaded data must be a PyTorch tensor. Got {type(activations_tensor)}."
        )

        expected_shape = (self.num_samples, self.context_size, self.d_in)
        assert activations_tensor.shape == expected_shape, (
            f"Loaded tensor shape {activations_tensor.shape} does not match the expected "
            f"shape {expected_shape} (samples, seq, embedding)."
        )

        self.current_row_idx = 0  # Initialize index for batch loading
        return activations_tensor


    def set_norm_scaling_factor_if_needed(self):
        if self.normalize_activations == "expected_average_only_in":
            self.estimated_norm_scaling_factor = self.estimate_norm_scaling_factor()

    def apply_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor is None:
            raise ValueError(
                "estimated_norm_scaling_factor is not set, call set_norm_scaling_factor_if_needed() first"
            )
        return activations * self.estimated_norm_scaling_factor

    def unscale(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor is None:
            raise ValueError(
                "estimated_norm_scaling_factor is not set, call set_norm_scaling_factor_if_needed() first"
            )
        return activations / self.estimated_norm_scaling_factor

    def get_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return (self.d_in**0.5) / activations.norm(dim=-1).mean()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, n_batches_for_norm_estimate: int = int(1e3)):
        norms_per_batch = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            # temporalily set estimated_norm_scaling_factor to 1.0 so the dataloader works
            self.estimated_norm_scaling_factor = 1.0
            acts = self.next_batch()
            self.estimated_norm_scaling_factor = None
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        return np.sqrt(self.d_in) / mean_norm


    @property
    def storage_buffer(self) -> torch.Tensor:
        if self._storage_buffer is None:
            self._storage_buffer = self.get_buffer(self.half_buffer_size)

        return self._storage_buffer

    @property
    def dataloader(self) -> Iterator[Any]:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return self._dataloader


    @torch.no_grad()
    def get_buffer(
        self,
        n_batches_in_buffer: int,
        raise_on_epoch_end: bool = False,
        shuffle: bool = True,
    ) -> torch.Tensor:
        """
        Loads the next `n_batches_in_buffer` batches of activations into a tensor.

        - Assumes `cached_activation_dataset` is a PyTorch tensor of shape `[batch, seq, d_in]`.
        - Supports optional shuffling and normalization.
        - Raises StopIteration when the dataset is exhausted and `raise_on_epoch_end` is True.

        Returns:
            Tensor of activations with shape `(total_size * seq, d_in)`.
        """
        assert self.cached_activation_dataset is not None, "No cached dataset found."

        # Compute total size and validate dataset shape
        batch_size, seq, d_in = self.cached_activation_dataset.shape
        expected_total_size = self.store_batch_size_prompts * n_batches_in_buffer

        if self.current_row_idx > batch_size - n_batches_in_buffer:
            self.current_row_idx = 0
            if raise_on_epoch_end:
                raise StopIteration

        # Slice the cached dataset to get the requested number of batches
        activations = self.cached_activation_dataset[
            self.current_row_idx : self.current_row_idx + n_batches_in_buffer
        ]
        self.current_row_idx += n_batches_in_buffer

        # Reshape activations: [n_batches_in_buffer, seq, d_in] -> [total_size * seq, d_in]
        activations = activations.reshape(-1, d_in)

        # Shuffle activations if required
        if shuffle:
            activations = activations[torch.randperm(activations.shape[0])]

        # Normalize activations if required
        if self.normalize_activations == "expected_average_only_in":
            activations = self.apply_norm_scaling_factor(activations)

        return activations


    @torch.no_grad()
    def get_data_loader(
        self,
    ) -> Iterator[torch.Tensor]:
        """
        Return a torch.utils.dataloader that provides batches for training.

        - Loads activations using `get_buffer`.
        - Automatically refills and shuffles the buffer for better mixing.
        - Handles dataset exhaustion by restarting a new epoch.
        
        Assumes `get_buffer` provides activations as a tensor of shape `(total_size * seq, d_in)`.

        Returns:
            Iterator of batches from the DataLoader.
        """
        batch_size = self.train_batch_size_tokens

        try:
            # Attempt to get a new buffer with activations
            new_samples = self.get_buffer(
                self.half_buffer_size, raise_on_epoch_end=True
            )
        except StopIteration:
            # Handle end-of-dataset scenario
            warnings.warn(
                "All samples in the training dataset have been exhausted, starting a new epoch."
            )
            self._storage_buffer = None  # Clear the storage buffer to prevent leakage
            try:
                new_samples = self.get_buffer(self.half_buffer_size)
            except StopIteration:
                raise ValueError(
                    "Unable to fill the buffer even after restarting a new epoch. "
                    "Consider reducing `batch_size` or `n_batches_in_buffer`."
                )

        # Combine new samples with the storage buffer if available
        if self._storage_buffer is not None:
            mixing_buffer = torch.cat([new_samples, self._storage_buffer], dim=0)
        else:
            mixing_buffer = new_samples

        # Shuffle the combined buffer for better mixing
        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # Split the buffer: 50% for storage, 50% for the DataLoader
        self._storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]
        dataloader_buffer = mixing_buffer[mixing_buffer.shape[0] // 2 :]

        # Return a DataLoader iterator for the remaining buffer
        return iter(
            DataLoader(
                dataloader_buffer,
                batch_size=batch_size,
                shuffle=False,  # Shuffling already applied to the buffer
            )
        )


    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self._dataloader = self.get_data_loader()
            return next(self.dataloader)

    def state_dict(self) -> dict[str, torch.Tensor]:
        result = {
            "n_dataset_processed": torch.tensor(self.n_dataset_processed),
        }
        if self._storage_buffer is not None:  # first time might be None
            result["storage_buffer"] = self._storage_buffer
        if self.estimated_norm_scaling_factor is not None:
            result["estimated_norm_scaling_factor"] = torch.tensor(
                self.estimated_norm_scaling_factor
            )
        return result

    def save(self, file_path: str):
        """save the state dict to a file in safetensors format"""
        save_file(self.state_dict(), file_path)


