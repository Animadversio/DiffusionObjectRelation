import sys
import os
from os.path import join
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import product
import matplotlib.pyplot as plt
import json
sys.path.append("/n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation/PixArt-alpha")
sys.path.append("/n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation")
from diffusion.utils.misc import read_config, set_random_seed, \
    init_random_seed, DebugUnderflowOverflow
from utils.cv2_eval_utils import find_classify_objects, find_classify_object_masks, evaluate_alignment, evaluate_parametric_relation
from utils.pixart_utils import construct_diffuser_pipeline_from_config, construct_pixart_transformer_from_config, state_dict_convert
from utils.attention_map_store_utils import replace_attn_processor, AttnProcessor2_0_Store, PixArtAttentionVisualizer_Store
from transformers import T5Tokenizer, T5EncoderModel
from utils.custom_text_encoding_utils import save_prompt_embeddings_randemb, RandomEmbeddingEncoder, RandomEmbeddingEncoder_wPosEmb
from utils.mask_attention_utils import get_meaningful_token_indices,mask_semantic_parts_attention, mask_all_attention, mask_padding_attention
from utils.path_assembly_utils import load_text_encoder
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler
from typing import Callable, List, Optional, Tuple, Union, Dict
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps

# Clear CUDA cache at the start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

class HeadSpecificAttentionProcessor:
    """Custom attention processor that applies different masks to specific heads within target layers"""
    
    def __init__(self, original_processor, layer_idx, target_layer_idx, target_head_indices, head_attention_mask):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.target_layer_idx = target_layer_idx  # Single target layer
        self.target_head_indices = target_head_indices  # List of head indices to mask
        self.head_attention_mask = head_attention_mask
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # If this is the target layer, apply head-specific masking
        if (self.target_layer_idx is not None and 
            self.layer_idx == self.target_layer_idx and 
            self.target_head_indices is not None and 
            self.head_attention_mask is not None):
            
            # Get the original attention processor to call it first
            # We need to modify the attention computation to mask specific heads
            
            # For head-specific masking, we need to intercept the attention computation
            # and selectively mask certain heads
            
            current_attention_mask = self.head_attention_mask
            
            # CONVERT BINARY MASK TO TRANSFORMER FORMAT
            # Binary format: 0 = cannot attend, 1 = can attend
            # Transformer format: large negative = cannot attend, ~0 = can attend
            
            # Check if mask is in binary format (values are essentially 0 or 1)
            mask_min = current_attention_mask.min().item()
            mask_max = current_attention_mask.max().item()
            is_binary_format = (mask_min >= -0.1 and mask_max <= 1.1 and 
                               torch.all((current_attention_mask <= 0.1) | (current_attention_mask >= 0.9)))
            
            if is_binary_format:
                # Convert binary mask to transformer format: 0 → -10000, 1 → 0
                current_attention_mask = (1 - current_attention_mask.to(torch.float32)) * -10000.0
            
            # Fix the shape mismatch - ensure head mask matches original mask dimensions
            if attention_mask is not None and current_attention_mask.shape != attention_mask.shape:
                # If original mask has more dimensions, we need to expand the head mask
                if len(attention_mask.shape) > len(current_attention_mask.shape):
                    # The original mask is [4, 1, 20] and head mask is [4, 20]
                    # We need to add the middle dimension (1) at position 1
                    if len(attention_mask.shape) == 3 and len(current_attention_mask.shape) == 2:
                        # Add the middle dimension: [4, 20] -> [4, 1, 20]
                        current_attention_mask = current_attention_mask.unsqueeze(1)
                    else:
                        # General case: expand to match the target shape
                        target_shape = list(attention_mask.shape)
                        current_shape = list(current_attention_mask.shape)
                        
                        # Add missing leading dimensions
                        while len(current_shape) < len(target_shape):
                            current_shape.insert(0, 1)
                        
                        # Reshape and expand to match target shape
                        current_attention_mask = current_attention_mask.view(*current_shape)
                        current_attention_mask = current_attention_mask.expand(target_shape)
            
            # Fix the data type mismatch - ensure mask has the correct dtype
            if attention_mask is not None:
                # Convert to match the original attention mask dtype
                current_attention_mask = current_attention_mask.to(dtype=attention_mask.dtype, device=attention_mask.device)
            
            # For head-specific masking, we need to modify the attention processor
            # to selectively apply masks to specific heads
            
            # Create a custom attention processor that masks specific heads
            return self._process_attention_with_head_masking(
                attn, hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=current_attention_mask,
                original_attention_mask=attention_mask,
                **kwargs
            )
        else:
            # Use the original attention mask for all other layers
            return self.original_processor(
                attn, hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask, 
                **kwargs
            )
    
    def _process_attention_with_head_masking(self, attn, hidden_states, encoder_hidden_states=None, 
                                           attention_mask=None, original_attention_mask=None, **kwargs):
        """
        Process attention with head-specific masking.
        This method applies the mask only to the specified heads.
        """
        # Get basic parameters
        batch_size, sequence_length, _ = hidden_states.shape
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            
        # Get dimensions
        inner_dim = attn.to_k.out_features
        head_dim = inner_dim // attn.heads
        
        # Calculate scale factor manually (standard scaled dot-product attention)
        scale = head_dim ** -0.5
        
        # Process query, key, value
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Reshape to separate heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        
        # Apply attention mask
        if attention_mask is not None:
            # For head-specific masking, we need to determine which heads to apply the mask to
            # and which heads to use the original mask
            
            # Create combined attention mask
            combined_attention_mask = original_attention_mask.clone() if original_attention_mask is not None else None
            
            # Apply head-specific mask only to target heads
            for head_idx in range(attn.heads):
                if head_idx in self.target_head_indices:
                    # Use the special mask for this head
                    if attention_mask.dim() == 3:
                        # attention_mask shape: [batch, 1, seq_len] -> [batch, 1, 1, seq_len]
                        head_mask = attention_mask.unsqueeze(2)
                    elif attention_mask.dim() == 4:
                        # attention_mask shape: [batch, 1, seq_len, seq_len] -> use as is
                        head_mask = attention_mask
                    else:
                        head_mask = attention_mask
                    
                    # Apply to specific head
                    attention_scores[:, head_idx:head_idx+1] += head_mask
                else:
                    # Use original mask for non-target heads
                    if combined_attention_mask is not None:
                        if combined_attention_mask.dim() == 3:
                            # original mask shape: [batch, 1, seq_len] -> [batch, 1, 1, seq_len]
                            head_mask = combined_attention_mask.unsqueeze(2)
                        elif combined_attention_mask.dim() == 4:
                            # original mask shape: [batch, 1, seq_len, seq_len] -> use as is
                            head_mask = combined_attention_mask
                        else:
                            head_mask = combined_attention_mask
                        
                        # Apply to this head
                        attention_scores[:, head_idx:head_idx+1] += head_mask
        
        # Apply softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply dropout - use F.dropout since attn.dropout is likely a float probability
        dropout_p = getattr(attn, 'dropout', 0.0)
        if hasattr(dropout_p, '__call__'):
            # If it's a dropout layer, call it directly
            attention_probs = dropout_p(attention_probs)
        else:
            # If it's a float probability, use F.dropout
            # Use the attention module's training state
            training_state = getattr(attn, 'training', True)
            attention_probs = F.dropout(attention_probs, p=dropout_p, training=training_state)
        
        # Apply attention to values
        hidden_states = torch.matmul(attention_probs, value)
        
        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        
        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states

class PixArtAlphaPipeline_HeadMask(PixArtAlphaPipeline):
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,
        return_sample_pred_traj: bool = False,
        device: str = "cuda",
        target_layer_idx: Optional[int] = None,
        target_head_indices: Optional[List[int]] = None,
        head_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation with head-specific attention masking.

        Args:
            target_layer_idx (`int`, *optional*): The specific transformer layer index at which to apply
                the head-specific attention mask. Defaults to `None`, meaning no head-specific masking.
            target_head_indices (`List[int]`, *optional*): List of attention head indices within the target
                layer to apply the mask to. For example, [0, 2, 5] would mask heads 0, 2, and 5.
                Required if `target_layer_idx` is set. Defaults to `None`.
            head_attention_mask (`torch.Tensor`, *optional*): The attention mask to be applied to the 
                specified heads in the target layer. Required if `target_layer_idx` and `target_head_indices` 
                are set. Should have the same shape as the normal attention mask. Defaults to `None`.
            
            All other parameters are the same as the base PixArtAlphaPipeline.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
        
        # Validate head masking parameters
        if target_layer_idx is not None:
            if target_head_indices is None:
                raise ValueError("target_head_indices must be provided when target_layer_idx is set")
            if head_attention_mask is None:
                raise ValueError("head_attention_mask must be provided when target_layer_idx is set")
        
        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )
        
        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            
            # Handle head-specific attention mask for CFG
            if target_layer_idx is not None and head_attention_mask is not None:
                # For CFG, we need to concatenate the negative and positive head masks
                # The negative part should use the standard negative prompt attention mask
                # The positive part should use our custom head attention mask
                
                # First ensure the head mask has the right batch size to match positive prompt
                if head_attention_mask.shape[0] != negative_prompt_attention_mask.shape[0]:
                    # If head mask batch size doesn't match, expand it
                    if head_attention_mask.shape[0] == 1:
                        # If head mask is just [1, seq_len], expand to match batch size
                        head_attention_mask = head_attention_mask.expand(negative_prompt_attention_mask.shape[0], -1)
                    else:
                        # If there's a different mismatch, use the first part that matches
                        head_attention_mask = head_attention_mask[:negative_prompt_attention_mask.shape[0]]
                
                # Create the CFG mask: [negative_attention_mask, custom_head_mask]
                # For negative prompts, use standard attention mask (no head masking)
                # For positive prompts, use the custom head attention mask
                cfg_head_attention_mask = torch.cat([negative_prompt_attention_mask, head_attention_mask], dim=0)
                head_attention_mask = cfg_head_attention_mask

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

            if do_classifier_free_guidance:
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # 7. Set up head-specific attention masking if specified
        original_processors = {}
        if target_layer_idx is not None and target_head_indices is not None and head_attention_mask is not None:
            print(f"Setting up head masking for layer {target_layer_idx}, heads: {target_head_indices}")
            
            # Store original processors and replace with head-specific ones
            for name, layer in self.transformer.transformer_blocks.named_children():
                layer_idx = int(name)
                if hasattr(layer, 'attn2'):  # Cross-attention layer
                    original_processors[layer_idx] = layer.attn2.processor
                    layer.attn2.processor = HeadSpecificAttentionProcessor(
                        original_processor=layer.attn2.processor,
                        layer_idx=layer_idx,
                        target_layer_idx=target_layer_idx,
                        target_head_indices=target_head_indices,
                        head_attention_mask=head_attention_mask
                    )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        pred_traj = []
        latents_traj = []
        t_traj = []
        denoiser_traj = []
        
        try:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    current_timestep = t
                    if not torch.is_tensor(current_timestep):
                        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                        # This would be a good case for the `match` statement (Python 3.10+)
                        is_mps = latent_model_input.device.type == "mps"
                        if isinstance(current_timestep, float):
                            dtype = torch.float32 if is_mps else torch.float64
                        else:
                            dtype = torch.int32 if is_mps else torch.int64
                        current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                    elif len(current_timestep.shape) == 0:
                        current_timestep = current_timestep[None].to(latent_model_input.device)
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    current_timestep = current_timestep.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=current_timestep,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # learned sigma
                    if self.transformer.config.out_channels // 2 == latent_channels:
                        noise_pred = noise_pred.chunk(2, dim=1)[0]
                    else:
                        noise_pred = noise_pred

                    latents_traj.append(latents.cpu())
                    pred_traj.append(noise_pred.cpu())
                    
                    # save denoiser
                    if self.scheduler.step_index is None:
                        self.scheduler._init_step_index(t)
                    denoiser_traj.append(self.scheduler.convert_model_output(model_output=noise_pred, sample=latents))
                    
                    # compute previous image: x_t -> x_t-1
                    if num_inference_steps == 1:
                        # For DMD one step sampling: https://arxiv.org/abs/2311.18828
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample
                    else:
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    
                    t_traj.append(t)
                    
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

        finally:
            # 9. Restore original attention processors
            if target_layer_idx is not None and target_head_indices is not None and head_attention_mask is not None:
                for layer_idx, original_processor in original_processors.items():
                    layer_name = str(layer_idx)
                    if hasattr(self.transformer.transformer_blocks[layer_idx], 'attn2'):
                        self.transformer.transformer_blocks[layer_idx].attn2.processor = original_processor

        latents_traj.append(latents.cpu())
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)
            
        denoiser_image_traj = []
        for i in range(len(denoiser_traj)):
            denoiser_image = self.vae.decode(denoiser_traj[i] / self.vae.config.scaling_factor, return_dict=False)[0]
            denoiser_image = self.image_processor.postprocess(denoiser_image.detach(), output_type="pil")
            denoiser_image_traj.append(denoiser_image)
            
        # Offload all models
        self.maybe_free_model_hooks()
        latents_traj = torch.stack(latents_traj)
        pred_traj = torch.stack(pred_traj)
        
        if not return_dict:
            return (image,)
        if return_sample_pred_traj:
            return ImagePipelineOutput(images=image), pred_traj, latents_traj, t_traj, denoiser_image_traj
        return ImagePipelineOutput(images=image)

def visualize_prompt_with_head_masking(pipeline, prompt, max_length=120, weight_dtype=torch.bfloat16,
                   num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25, num_image_to_plot=0, 
                   device="cuda", random_seed=0, target_layer_idx=None, target_head_indices=None, 
                   manipulated_mask_func=None, **mask_kwargs):
    """
    Generate images with head-specific attention masking
    
    Args:
        pipeline: The PixArtAlphaPipeline_HeadMask instance
        prompt: Text prompt for generation
        max_length: Maximum sequence length
        weight_dtype: Weight data type
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG guidance scale
        num_images_per_prompt: Number of images to generate
        num_image_to_plot: Which image to plot in trajectory
        device: Device to run on
        random_seed: Random seed for reproducibility
        target_layer_idx: Which transformer layer to apply head masking to (single int)
        target_head_indices: Which attention heads within the target layer to mask (list of ints)
        manipulated_mask_func: Function to generate the head-specific attention mask
        **mask_kwargs: Additional arguments for the mask function
    """
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if random_seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    
    # Tokenize the prompt
    inputs = pipeline.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=max_length).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Generate head-specific attention mask if specified
    head_attention_mask = None
    if target_layer_idx is not None and target_head_indices is not None and manipulated_mask_func is not None:
        print(f"Applying {manipulated_mask_func.__name__} to layer {target_layer_idx}, heads {target_head_indices}")
        head_attention_mask = manipulated_mask_func(prompt, pipeline.tokenizer, max_length, device, **mask_kwargs)
    
    # Get text embeddings
    text_embeddings = pipeline.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    output = pipeline(
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        prompt_embeds=text_embeddings,
        prompt_attention_mask=attention_mask,
        negative_prompt="",
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        use_resolution_binning=False,
        return_sample_pred_traj=True,
        output_type="latent",
        target_layer_idx=target_layer_idx,
        target_head_indices=target_head_indices,
        head_attention_mask=head_attention_mask,
    )
    
    latents = output[0].images
    pred_traj = output[1]
    latents_traj = output[2]
    t_traj = output[3]
    denoiser_traj = output[4]
    
    # Decode latents to images
    images = pipeline.vae.decode(latents.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    images = pipeline.image_processor.postprocess(images.detach(), output_type="pil")
    
    return {"prompt": prompt, "images": images}, latents_traj, pred_traj, t_traj, denoiser_traj



def load_custom_trained_model(model_dir, ckpt_name, t5_path, text_feat_dir, text_encoder_type, weight_dtype=torch.bfloat16):
    """Load custom trained model (copied from text_prompt_ablation.py)"""
    # Load model checkpoint and config
    ckpt = torch.load(join(model_dir, "checkpoints", ckpt_name))
    config = read_config(join(model_dir, 'config.py'))
    config.mixed_precision = "bf16"
    
    # Initialize pipeline
    pipeline = construct_diffuser_pipeline_from_config(config, pipeline_class=PixArtAlphaPipeline_HeadMask)
    pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict_ema']))
    
    # Load text encoder
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    text_encoder = load_text_encoder(text_encoder_type, text_feat_dir, t5_path)
    
    # Set up pipeline properties
    pipeline.text_encoder = text_encoder
    pipeline.text_encoder.dtype = weight_dtype
    pipeline.tokenizer = tokenizer
    pipeline = pipeline.to(dtype=weight_dtype)
    max_length = config.model_max_length
    
    return pipeline, max_length

def generate_test_prompts_collection_and_parsed_words():
    """Generate a collection of test prompts for object relation experiments"""
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

def generate_and_quantify_images_head(prompt_dir, pipeline, prompt, scene_info, target_layer_idx=None, target_head_indices=None, 
                   max_length=120, weight_dtype=torch.bfloat16, num_inference_steps=14, guidance_scale=4.5, 
                   num_images_per_prompt=25, device="cuda", random_seed=0, manipulated_mask_func=None, **mask_kwargs):
    """
    Generate images with head-specific attention masking and quantify their quality
    
    Args:
        prompt_dir: Directory to save evaluation results
        pipeline: The PixArtAlphaPipeline_HeadMask instance
        prompt: Text prompt for generation
        target_layer_idx: Which transformer layer to apply head masking to (single int)
        target_head_indices: Which attention heads within the target layer to mask (list of ints)
        max_length: Maximum sequence length
        weight_dtype: Weight data type
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG guidance scale
        num_images_per_prompt: Number of images to generate
        device: Device to run on
        random_seed: Random seed for reproducibility
        manipulated_mask_func: Function to generate the head-specific attention mask
        **mask_kwargs: Additional arguments for the mask function
    """
    image_logs, latents_traj, pred_traj, t_traj, denoiser_traj = visualize_prompt_with_head_masking(
        pipeline=pipeline, 
        prompt=prompt, 
        max_length=max_length, 
        weight_dtype=weight_dtype, 
        num_images_per_prompt=num_images_per_prompt,
        device=device, 
        random_seed=random_seed,
        target_layer_idx=target_layer_idx,
        target_head_indices=target_head_indices,
        manipulated_mask_func=manipulated_mask_func,
        **mask_kwargs
    )
    
    # Calculate scores
    avg_scores = calculate_gen_images_scores(image_logs, prompt, scene_info)
    
    # Save evaluation results  
    save_cv2_eval_head(
        prompt_dir=prompt_dir,
        prompt=prompt,
        target_layer_idx=target_layer_idx,
        target_head_indices=target_head_indices,
        manipulated_mask_func=manipulated_mask_func,
        avg_scores=avg_scores,
        mask_kwargs=mask_kwargs
    )
    
    # Clear variables to free memory
    del image_logs
    del latents_traj 
    del pred_traj
    del t_traj
    del denoiser_traj
    
    # Clear CUDA cache
    torch.cuda.empty_cache()

def save_cv2_eval_head(prompt_dir, prompt, target_layer_idx, target_head_indices, manipulated_mask_func, avg_scores, mask_kwargs):
    """
    Save evaluation results for head-specific masking experiments
    
    Args:
        prompt_dir: Directory to save results
        prompt: Text prompt
        target_layer_idx: Layer that was masked
        target_head_indices: Head(s) that were masked  
        manipulated_mask_func: Masking function used
        avg_scores: Average scores to save
        mask_kwargs: Additional mask parameters
    """
    # Create saved_metrics directory inside prompt_dir
    metrics_dir = os.path.join(prompt_dir, 'saved_metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Generate key name based on parameters
    if target_layer_idx is None and target_head_indices is None:
        key = 'baseline'
    else:
        mask_func_name = manipulated_mask_func.__name__ if manipulated_mask_func else 'no_mask'
        
        # Format head indices
        if target_head_indices is None:
            head_str = 'none'
        elif isinstance(target_head_indices, int):
            head_str = str(target_head_indices)
        elif isinstance(target_head_indices, list):
            if len(target_head_indices) == 1:
                head_str = str(target_head_indices[0])
            else:
                head_str = '_'.join(map(str, target_head_indices))
        else:
            head_str = str(target_head_indices)
        
        # Use format: f'step_{layer}_{head}_{mask_func_name}'
        key = f'step_{target_layer_idx}_{head_str}_{mask_func_name}'
        
        # Append mask_kwargs to key if they exist (e.g., part_to_mask)
        if mask_kwargs:
            # Handle the part_to_mask parameter specifically
            if 'part_to_mask' in mask_kwargs:
                key = f"{key}_{mask_kwargs['part_to_mask']}"
            # Handle other mask_kwargs
            for k, v in mask_kwargs.items():
                if k != 'part_to_mask':  # Already handled above
                    key = f"{key}_{k}_{v}"
    
    # Save to head_masking.json file
    filename = os.path.join(metrics_dir, 'head_masking.json')
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            all_scores = json.load(f)
    else:
        all_scores = {}
    
    # Add new scores under appropriate key
    all_scores[key] = avg_scores
    
    # Save the updated scores
    with open(filename, 'w') as f:
        json.dump(all_scores, f, indent=4)

def calculate_gen_images_scores(image_logs, prompt, scene_info):
    """Calculate average scores across all images (copied from text_prompt_ablation.py)"""
    # shape_match_scores = []
    # color_binding_scores = []
    # spatial_color_scores = []
    # spatial_shape_scores = []
    overall_scores = []
    overall_loose_scores = []
    shape_scores = []
    color_scores = []
    exist_binding_scores = []
    unique_binding_scores = []
    spatial_relationship_scores = []
    spatial_relationship_loose_scores = []
    Dx_scores = []
    Dy_scores = []
    x1_scores = []
    y1_scores = []
    x2_scores = []
    y2_scores = []

    for image in image_logs['images']:
        # Get objects from image
        df = find_classify_objects(image, area_threshold=100, radius=16.0)
        
        # Check if DataFrame is empty or missing required columns
        if df.empty or not all(col in df.columns for col in ['Shape', 'Color (RGB)', 'Center (x, y)']):
            # No objects detected - assign zero scores
            overall_scores.append(0)
            overall_loose_scores.append(0)
            shape_scores.append(0)
            color_scores.append(0)
            exist_binding_scores.append(0)
            unique_binding_scores.append(0)
            spatial_relationship_scores.append(0)
            spatial_relationship_loose_scores.append(0)
            Dx_scores.append(0)
            Dy_scores.append(0)
            x1_scores.append(0)
            y1_scores.append(0)
            x2_scores.append(0)
            y2_scores.append(0)
            continue
        
        # Evaluate alignment
        # result = evaluate_alignment(prompt, df)
        
        # # Add scores
        # shape_match_scores.append(int(result['shape_match']))
        # color_binding_scores.append(sum(int(x) for x in result['color_binding_match'].values()) / len(result['color_binding_match']))
        # spatial_color_scores.append(int(result['spatial_color_relation']))
        # spatial_shape_scores.append(int(result['spatial_shape_relation'])) 
        # overall_scores.append(result['overall_score'])
        
        # New evaluate alignment
        result = evaluate_parametric_relation(df, scene_info, color_margin=25, spatial_threshold=5)
        
        overall_scores.append(result['overall'])
        overall_loose_scores.append(result['overall_loose'])
        shape_scores.append(result['shape'])
        color_scores.append(result['color'])
        exist_binding_scores.append(result['exist_binding'])
        unique_binding_scores.append(result['unique_binding'])
        spatial_relationship_scores.append(result['spatial_relationship'])
        spatial_relationship_loose_scores.append(result['spatial_relationship_loose'])
        Dx_scores.append(result['Dx'])
        Dy_scores.append(result['Dy'])
        x1_scores.append(result['x1'])
        y1_scores.append(result['y1'])
        x2_scores.append(result['x2'])
        y2_scores.append(result['y2'])

    # Calculate averages
    avg_scores = {
        'shape': np.mean(shape_scores),
        'color': np.mean(color_scores),
        'exist_binding': np.mean(exist_binding_scores),
        'unique_binding': np.mean(unique_binding_scores),
        'spatial_relationship': np.mean(spatial_relationship_scores),
        'spatial_relationship_loose': np.mean(spatial_relationship_loose_scores),
        'Dx': np.mean(Dx_scores),
        'Dy': np.mean(Dy_scores),
        'x1': np.mean(x1_scores),
        'y1': np.mean(y1_scores),
        'x2': np.mean(x2_scores),
        'y2': np.mean(y2_scores)
    }
    return avg_scores

def test_single_prompt_head_ablation(prompt=None, scene_info=None, save_dir=None):
    """Test function for head ablation on a single prompt - loops through heads 0-11 on layer 2"""
    
    # Default paths
    base_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results'
    model_run_name = 'objrel_rndembdposemb_DiT_mini_pilot'
    ckpt_name = 'epoch_4000_step_160000.pth'
    model_dir = join(base_dir, model_run_name)
    t5_path = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl'
    text_feat_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRel_pilot_rndembposemb/caption_feature_wmask'
    save_dir = save_dir
    text_encoder_type = 'RandomEmbeddingEncoder_wPosEmb'
    
    
    # Head ablation parameters
    target_layer_idx = 2  # Test layer 2
    heads_to_test = list(range(0, 12))  # Test heads 0-11 (assuming 12 heads per layer)
    
    # Mask function and parameters - test each part separately
    mask_func = mask_semantic_parts_attention
    parts_to_mask = ['colors', 'spatial', 'objects']  # Test each semantic part individually
    
    print("Loading model...")
    pipeline, max_length = load_custom_trained_model(model_dir, ckpt_name, t5_path, text_feat_dir, text_encoder_type)
    
    print(f"Testing prompt: '{prompt}'")
    print(f"Testing layer: {target_layer_idx}")
    print(f"Testing heads individually: {heads_to_test}")
    print(f"Testing semantic parts individually: {parts_to_mask}")
    
    # Create save directory
    prompt_dir = os.path.join(save_dir, prompt.replace(' ', '_'))
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Clear any existing results file to start fresh
    metrics_file = os.path.join(prompt_dir, 'saved_metrics', 'head_masking.json')
    if os.path.exists(metrics_file):
        os.remove(metrics_file)
        print("Cleared existing results file")
    
    total_experiments = 1 + len(heads_to_test) * len(parts_to_mask)  # 1 baseline + head combinations
    current_experiment = 0
    
    # Generate baseline (no masking)
    current_experiment += 1
    print(f"\n[{current_experiment}/{total_experiments}] Generating baseline (no masking)")
    
    generate_and_quantify_images_head(
        prompt_dir=prompt_dir,
        pipeline=pipeline,
        prompt=prompt,
        scene_info=scene_info,
        target_layer_idx=None,  # No masking
        target_head_indices=None,
        max_length=max_length,
        weight_dtype=torch.bfloat16,
        num_inference_steps=14,
        guidance_scale=4.5,
        num_images_per_prompt=25,
        device="cuda",
        random_seed=42,
        manipulated_mask_func=None
    )
    
    # Test each combination of head and semantic part
    for part_to_mask in parts_to_mask:
        for head_idx in heads_to_test:
            current_experiment += 1
            print(f"[{current_experiment}/{total_experiments}] Layer {target_layer_idx}, Head {head_idx} + {part_to_mask} masking")
            
            generate_and_quantify_images_head(
                prompt_dir=prompt_dir,
                pipeline=pipeline,
                prompt=prompt,
                scene_info=scene_info,
                target_layer_idx=target_layer_idx,
                target_head_indices=[head_idx],  # Single head as list
                max_length=max_length,
                weight_dtype=torch.bfloat16,
                num_inference_steps=14,
                guidance_scale=4.5,
                num_images_per_prompt=25,
                device="cuda",
                random_seed=42,
                manipulated_mask_func=mask_func,
                part_to_mask=part_to_mask
            )
            
            # Clear CUDA cache to prevent memory issues
            torch.cuda.empty_cache()
    
    print(f"\n✅ Head ablation experiment completed successfully!")
    print(f"Results saved to: {metrics_file}")
        
        

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run layer ablation experiments')
    #parser.add_argument('--test', action='store_true', help='Run quick test with default settings')
   # parser.add_argument('--prompt', type=str, default="red circle is to the left of blue square", 
                       #help='Text prompt to test (for single prompt testing)')
    parser.add_argument('--save_dir', type=str, default='/n/home13/xupan/sompolinsky_lab/object_relation/head_ablation/rndposemb_mini/layer_2/',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
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

    # Generate test prompts
    print("Generating test prompts...")
    prompts, parsed_words = generate_test_prompts_collection_and_parsed_words()
    print(f"Generated {len(prompts)} test prompts")
    prompt_df = pd.DataFrame(parsed_words)
    prompt_df["relation_str"] = prompt_df["relation"].apply(lambda x: "_".join(x))
    print(f"Running layer ablation experiments:")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Save dir: {args.save_dir}")
    
    # Loop through each prompt
    for prompt_id, row in tqdm(prompt_df.iterrows()):
        prompt = row["prompt"]
        scene_info = {
                "color1": row["color1"],
                "shape1": row["shape1"],
                "color2": row["color2"],
                "shape2": row["shape2"],
                "spatial_relationship": mapping2spatial_relationship[row["relation_str"]]
            }
        print(f"\n{'='*60}")
        print(f"Processing prompt {prompt_id + 1}/{len(prompt_df)}: '{prompt}'")
        print(f"{'='*60}")
        
        test_single_prompt_head_ablation(prompt=prompt, scene_info=scene_info, save_dir=args.save_dir)
            
    print(f"\n{'='*60}")
    print(f"✅ All layer ablation experiments completed!")
    print(f"Processed {len(prompts)} prompts")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'='*60}")