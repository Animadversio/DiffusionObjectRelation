import sys
import os
from os.path import join
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
sys.path.append("/n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation/PixArt-alpha")
sys.path.append("/n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation")
from diffusion.utils.misc import read_config, set_random_seed, \
    init_random_seed, DebugUnderflowOverflow
from utils.cv2_eval_utils import find_classify_objects, find_classify_object_masks, evaluate_alignment
from utils.pixart_utils import construct_diffuser_pipeline_from_config, construct_pixart_transformer_from_config, state_dict_convert
from utils.attention_map_store_utils import replace_attn_processor, AttnProcessor2_0_Store, PixArtAttentionVisualizer_Store
from transformers import T5Tokenizer, T5EncoderModel
from utils.custom_text_encoding_utils import save_prompt_embeddings_randemb, RandomEmbeddingEncoder, RandomEmbeddingEncoder_wPosEmb
from utils.mask_attention_utils import get_meaningful_token_indices,mask_semantic_parts_attention, mask_all_attention, mask_padding_attention
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler
from typing import Callable, List, Optional, Tuple, Union, Dict
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps

# Clear CUDA cache at the start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

class LayerSpecificAttentionProcessor:
    """Custom attention processor that applies different masks to different layers"""
    
    def __init__(self, original_processor, layer_idx, target_layer_indices, layer_attention_mask):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.target_layer_indices = target_layer_indices  # Now a list of layer indices
        self.layer_attention_mask = layer_attention_mask
        self.debug_call_count = 0  # Add counter to track all calls
        self.seen_timesteps = set()  # Track which timesteps we've seen
        self.attention_printed = False  # Flag to print attention weights only once
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # If this is one of the target layers, use the layer-specific attention mask
        if self.target_layer_indices is not None and self.layer_idx in self.target_layer_indices and self.layer_attention_mask is not None:
            # Use the layer-specific mask instead of the default attention mask
            current_attention_mask = self.layer_attention_mask
            
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
            
            # Fix the shape mismatch - ensure layer mask matches original mask dimensions
            if attention_mask is not None and current_attention_mask.shape != attention_mask.shape:
                # If original mask has more dimensions, we need to expand the layer mask
                if len(attention_mask.shape) > len(current_attention_mask.shape):
                    # The original mask is [4, 1, 20] and layer mask is [4, 20]
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
            
            self.debug_call_count += 1
            
            return self.original_processor(
                attn, hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=current_attention_mask, 
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

class PixArtAlphaPipeline_LayerMask(PixArtAlphaPipeline):
    
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
        target_layer_indices: Optional[Union[int, List[int]]] = None,
        layer_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation with layer-specific attention masking.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            max_sequence_length (`int` defaults to 120): Maximum sequence length to use with the `prompt`.
            target_layer_indices (`int` or `List[int]`, *optional*): The specific transformer layer index(es) at which to apply
                the layer-specific attention mask. Can be a single integer (e.g., 5) or a list of integers (e.g., [2, 5, 8]).
                Defaults to `None`, meaning no layer-specific masking.
            layer_attention_mask (`torch.Tensor`, *optional*): The attention mask to be applied at the 
                target layer(s). Required if `target_layer_indices` is set. Should have the same shape as the 
                normal attention mask. Defaults to `None`.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            # deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)
        
        # Normalize target_layer_indices to always be a list
        if target_layer_indices is not None:
            if isinstance(target_layer_indices, int):
                target_layer_indices = [target_layer_indices]
            print(f"Target layers for masking: {target_layer_indices}")
        
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
            
            # Handle layer-specific attention mask for CFG
            if target_layer_indices is not None and layer_attention_mask is not None:
                # For CFG, we need to concatenate the negative and positive layer masks
                # First ensure the layer mask has the right batch size
                if layer_attention_mask.shape[0] != negative_prompt_attention_mask.shape[0]:
                    # If layer mask batch size doesn't match, expand it
                    if layer_attention_mask.shape[0] == 1:
                        # If layer mask is just [1, seq_len], expand to match batch size
                        layer_attention_mask = layer_attention_mask.expand(negative_prompt_attention_mask.shape[0], -1)
                    else:
                        # If there's a different mismatch, use the first part that matches
                        layer_attention_mask = layer_attention_mask[:negative_prompt_attention_mask.shape[0]]
                
                # Concatenate: [negative_mask, positive_mask]
                layer_attention_mask = torch.cat([negative_prompt_attention_mask, layer_attention_mask], dim=0)

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

        # 7. Set up layer-specific attention masking if specified
        original_processors = {}
        if target_layer_indices is not None and layer_attention_mask is not None:
            print(f"Setting up layer masking for layers: {target_layer_indices}")
            
            # Store original processors and replace with layer-specific ones
            for name, layer in self.transformer.transformer_blocks.named_children():
                layer_idx = int(name)
                if hasattr(layer, 'attn2'):  # Cross-attention layer
                    original_processors[layer_idx] = layer.attn2.processor
                    layer.attn2.processor = LayerSpecificAttentionProcessor(
                        original_processor=layer.attn2.processor,
                        layer_idx=layer_idx,
                        target_layer_indices=target_layer_indices, # Pass the list of target layer indices
                        layer_attention_mask=layer_attention_mask
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
            if target_layer_indices is not None and layer_attention_mask is not None:
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

def visualize_prompt_with_layer_masking(pipeline, prompt, max_length=120, weight_dtype=torch.bfloat16,
                   num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25, num_image_to_plot=0, 
                   device="cuda", random_seed=0, target_layer_indices=None, manipulated_mask_func=None, **mask_kwargs):
    """
    Generate images with layer-specific attention masking
    
    Args:
        pipeline: The PixArtAlphaPipeline_LayerMask instance
        prompt: Text prompt for generation
        max_length: Maximum sequence length
        weight_dtype: Weight data type
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG guidance scale
        num_images_per_prompt: Number of images to generate
        num_image_to_plot: Which image to plot in trajectory
        device: Device to run on
        random_seed: Random seed for reproducibility
        target_layer_indices: Which transformer layer(s) to apply the mask to (int or list of ints)
        manipulated_mask_func: Function to generate the layer-specific attention mask
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
    
    # Generate layer-specific attention mask if specified
    layer_attention_mask = None
    if target_layer_indices is not None and manipulated_mask_func is not None:
        print(f"Applying {manipulated_mask_func.__name__} to layer(s) {target_layer_indices}")
        layer_attention_mask = manipulated_mask_func(prompt, pipeline.tokenizer, max_length, device, **mask_kwargs)
    
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
        target_layer_indices=target_layer_indices,
        layer_attention_mask=layer_attention_mask,
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



def load_custom_trained_model(model_dir, t5_path, text_feat_dir, weight_dtype=torch.bfloat16):
    """Load custom trained model (copied from text_prompt_ablation.py)"""
    # Load model checkpoint and config
    ckpt = torch.load(join(model_dir, "checkpoints", "epoch_4000_step_160000.pth"))
    config = read_config(join(model_dir, 'config.py'))
    config.mixed_precision = "bf16"
    
    # Initialize pipeline
    pipeline = construct_diffuser_pipeline_from_config(config, pipeline_class=PixArtAlphaPipeline_LayerMask)
    pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict_ema']))
    
    # Load text encoder
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    rnd_encoding = torch.load(join(text_feat_dir, "word_embedding_dict.pt"))
    rndpos_encoder = RandomEmbeddingEncoder_wPosEmb(rnd_encoding["embedding_dict"],
                                                  rnd_encoding["input_ids2dict_ids"], 
                                                  rnd_encoding["dict_ids2input_ids"],
                                                  max_seq_len=20, embed_dim=4096,
                                                  wpe_scale=1/6).to("cuda")
    
    # Set up pipeline properties
    pipeline.text_encoder = rndpos_encoder
    pipeline.text_encoder.dtype = weight_dtype
    pipeline.tokenizer = tokenizer
    pipeline = pipeline.to(dtype=weight_dtype)
    max_length = config.model_max_length
    
    return pipeline, max_length

def generate_and_quantify_images_layer(prompt_dir, pipeline, prompt, target_layer_indices=None, max_length=120, weight_dtype=torch.bfloat16,
                   num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25, device="cuda", random_seed=0,
                   manipulated_mask_func=None, **mask_kwargs):
    """
    Generate images with layer-specific attention masking and quantify their quality
    
    Args:
        prompt_dir: Directory to save evaluation results
        pipeline: The PixArtAlphaPipeline_LayerMask instance
        prompt: Text prompt for generation
        target_layer_indices: Which transformer layer(s) to apply the mask to (int or list of ints)
        max_length: Maximum sequence length
        weight_dtype: Weight data type
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG guidance scale
        num_images_per_prompt: Number of images to generate
        device: Device to run on
        random_seed: Random seed for reproducibility
        manipulated_mask_func: Function to generate the layer-specific attention mask
        **mask_kwargs: Additional arguments for the mask function
    """
    image_logs, latents_traj, pred_traj, t_traj, denoiser_traj = visualize_prompt_with_layer_masking(
        pipeline=pipeline, 
        prompt=prompt, 
        max_length=max_length, 
        weight_dtype=weight_dtype, 
        num_images_per_prompt=num_images_per_prompt,
        device=device, 
        random_seed=random_seed,
        target_layer_indices=target_layer_indices, 
        manipulated_mask_func=manipulated_mask_func,
        **mask_kwargs
    )
    
    # Calculate scores
    avg_scores = calculate_gen_images_scores(image_logs, prompt)
    
    # Save evaluation results  
    save_cv2_eval_layer(
        prompt_dir=prompt_dir,
        prompt=prompt,
        target_layer_indices=target_layer_indices,
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

def save_cv2_eval_layer(prompt_dir, prompt, target_layer_indices, manipulated_mask_func, avg_scores, mask_kwargs):
    """
    Save evaluation results for layer-specific masking experiments
    
    Args:
        prompt_dir: Directory to save results
        prompt: Text prompt
        target_layer_indices: Layer(s) that were masked
        manipulated_mask_func: Masking function used
        avg_scores: Average scores to save
        mask_kwargs: Additional mask parameters
    """
    # Create saved_metrics directory inside prompt_dir
    metrics_dir = os.path.join(prompt_dir, 'saved_metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Generate key name based on parameters using the requested format
    if target_layer_indices is None:
        key = 'baseline'
    else:
        mask_func_name = manipulated_mask_func.__name__ if manipulated_mask_func else 'no_mask'
        if isinstance(target_layer_indices, int):
            layer_str = str(target_layer_indices)
        else:
            layer_str = '_'.join(map(str, target_layer_indices))
        
        # Use the requested format: f'step_{ablated layers}_{mask_func_name}'
        key = f'step_{layer_str}_{mask_func_name}'
        
        # Append mask_kwargs to key if they exist (e.g., part_to_mask)
        if mask_kwargs:
            # Handle the part_to_mask parameter specifically
            if 'part_to_mask' in mask_kwargs:
                key = f"{key}_{mask_kwargs['part_to_mask']}"
            # Handle other mask_kwargs
            for k, v in mask_kwargs.items():
                if k != 'part_to_mask':  # Already handled above
                    key = f"{key}_{k}_{v}"
    
    # Save to layer_masking.json file
    filename = os.path.join(metrics_dir, 'layer_masking.json')
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

def calculate_gen_images_scores(image_logs, prompt):
    """Calculate average scores across all images (copied from text_prompt_ablation.py)"""
    shape_match_scores = []
    color_binding_scores = []
    spatial_color_scores = []
    spatial_shape_scores = []
    overall_scores = []

    for image in image_logs['images']:
        # Get objects from image
        df = find_classify_objects(image, area_threshold=100, radius=16.0)
        
        # Check if DataFrame is empty or missing required columns
        if df.empty or not all(col in df.columns for col in ['Shape', 'Color (RGB)', 'Center (x, y)']):
            # No objects detected - assign zero scores
            shape_match_scores.append(0)
            color_binding_scores.append(0)
            spatial_color_scores.append(0)
            spatial_shape_scores.append(0)
            overall_scores.append(0)
            continue
        
        # Evaluate alignment
        result = evaluate_alignment(prompt, df)
        
        # Add scores
        shape_match_scores.append(int(result['shape_match']))
        color_binding_scores.append(sum(int(x) for x in result['color_binding_match'].values()) / len(result['color_binding_match']))
        spatial_color_scores.append(int(result['spatial_color_relation']))
        spatial_shape_scores.append(int(result['spatial_shape_relation'])) 
        overall_scores.append(result['overall_score'])

    # Calculate averages
    avg_scores = {
        'shape_match': np.mean(shape_match_scores),
        'color_binding': np.mean(color_binding_scores),
        'spatial_color_relation': np.mean(spatial_color_scores),
        'spatial_shape_relation': np.mean(spatial_shape_scores),
        'overall_score': np.mean(overall_scores)
    }
    return avg_scores

def test_single_prompt_layer_ablation():
    """Test function for layer ablation on a single prompt"""
    
    # Default paths (adjust as needed)
    model_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot'
    t5_path = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl'
    text_feat_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRel_pilot_rndembposemb/caption_feature_wmask'
    save_dir = '/n/home13/xupan/sompolinsky_lab/object_relation/layer_ablation/'
    
    # Test prompt
    prompt = "red circle is to the left of blue square"
    
    # Layers to test - FIXED: custom model has 12 layers (0-11), not 28 layers (0-27)
    layers_to_test = list(range(0, 12))  # Test all layers 0-11 for custom model
    
    # Mask function and parameters - we'll test each part separately
    mask_func = mask_semantic_parts_attention
    parts_to_mask = ['colors', 'objects', 'spatial']  # Test each semantic part individually
    
    print("Loading model...")
    pipeline, max_length = load_custom_trained_model(model_dir, t5_path, text_feat_dir)
    
    print(f"Testing prompt: '{prompt}'")
    print(f"Testing layers individually: {layers_to_test}")
    print(f"Testing semantic parts individually: {parts_to_mask}")
    
    # Create save directory
    prompt_dir = os.path.join(save_dir, prompt.replace(' ', '_'))
    os.makedirs(prompt_dir, exist_ok=True)
    
    try:
        # Generate baseline (no masking)
        print("\n=== Generating baseline (no masking) ===")
        generate_and_quantify_images_layer(
            prompt_dir=prompt_dir,
            pipeline=pipeline,
            prompt=prompt,
            target_layer_indices=None,  # No masking
            max_length=max_length,
            weight_dtype=torch.bfloat16,
            num_inference_steps=14,
            guidance_scale=4.5,
            num_images_per_prompt=25,
            device="cuda",
            random_seed=42,
            manipulated_mask_func=None
        )
        
        # Test each semantic part individually 
        for part_to_mask in parts_to_mask:
            print(f"\n=== Testing {part_to_mask} masking (each layer separately) ===")
            
            # Test each layer individually for this semantic part
            for layer_idx in layers_to_test:
                print(f"Processing layer {layer_idx} with {part_to_mask} masking...")
                
                generate_and_quantify_images_layer(
                    prompt_dir=prompt_dir,
                    pipeline=pipeline,
                    prompt=prompt,
                    target_layer_indices=layer_idx,  # Single layer at a time
                    max_length=max_length,
                    weight_dtype=torch.bfloat16,
                    num_inference_steps=14,
                    guidance_scale=4.5,
                    num_images_per_prompt=25,
                    device="cuda",
                    random_seed=42,
                    manipulated_mask_func=mask_func,
                    part_to_mask=part_to_mask  # Single semantic part at a time
                )
                
                # Clear CUDA cache after each layer to prevent memory issues
                torch.cuda.empty_cache()
        
        print(f"\n✅ Layer ablation completed successfully!")
        print(f"Results saved to: {prompt_dir}/saved_metrics/layer_masking.json")
        
        # Print summary of results
        metrics_file = os.path.join(prompt_dir, 'saved_metrics', 'layer_masking.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                results = json.load(f)
            print(f"\nGenerated {len(results)} different conditions:")
            print(f"  - 1 baseline condition")
            print(f"  - {len(layers_to_test)} layers × {len(parts_to_mask)} parts = {len(layers_to_test) * len(parts_to_mask)} individual layer-part combinations")
            for key in sorted(results.keys()):
                print(f"  - {key}: overall_score = {results[key]['overall_score']:.3f}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def layer_ablation_experiment(prompt, layers_to_test, parts_to_mask, save_dir, 
                             model_dir=None, t5_path=None, text_feat_dir=None):
    """
    Run layer ablation experiment on a single prompt
    
    Args:
        prompt: Text prompt to test
        layers_to_test: List of layer indices to test masking on
        parts_to_mask: List of semantic parts to mask ('colors', 'objects', 'spatial')
        save_dir: Directory to save results
        model_dir: Path to model directory (optional, uses default if None)
        t5_path: Path to T5 model (optional, uses default if None)  
        text_feat_dir: Path to text features (optional, uses default if None)
    """
    
    # Default paths
    if model_dir is None:
        model_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot'
    if t5_path is None:
        t5_path = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl'
    if text_feat_dir is None:
        text_feat_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/training_datasets/objectRel_pilot_rndembposemb/caption_feature_wmask'
    
    # Load model
    print("Loading model...")
    pipeline, max_length = load_custom_trained_model(model_dir, t5_path, text_feat_dir)
    
    print(f"Testing prompt: '{prompt}'")
    print(f"Testing layers: {layers_to_test}")
    print(f"Testing semantic parts: {parts_to_mask}")
    
    # Create save directory
    prompt_dir = os.path.join(save_dir, prompt.replace(' ', '_'))
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Clear any existing results file to start fresh
    metrics_file = os.path.join(prompt_dir, 'saved_metrics', 'layer_masking.json')
    if os.path.exists(metrics_file):
        os.remove(metrics_file)
        print("Cleared existing results file")
    
    total_experiments = 1 + len(layers_to_test) * len(parts_to_mask)  # 1 baseline + layer combinations
    current_experiment = 0
    
    try:
        # Generate baseline (no masking)
        current_experiment += 1
        print(f"\n[{current_experiment}/{total_experiments}] Generating baseline (no masking)")
        
        generate_and_quantify_images_layer(
            prompt_dir=prompt_dir,
            pipeline=pipeline,
            prompt=prompt,
            target_layer_indices=None,  # No masking
            max_length=max_length,
            weight_dtype=torch.bfloat16,
            num_inference_steps=14,
            guidance_scale=4.5,
            num_images_per_prompt=25,
            device="cuda",
            random_seed=42,
            manipulated_mask_func=None
        )
        
        # Test each combination of layer and semantic part
        for part_to_mask in parts_to_mask:
            for layer_idx in layers_to_test:
                current_experiment += 1
                print(f"[{current_experiment}/{total_experiments}] Layer {layer_idx} + {part_to_mask} masking")
                
                generate_and_quantify_images_layer(
                    prompt_dir=prompt_dir,
                    pipeline=pipeline,
                    prompt=prompt,
                    target_layer_indices=layer_idx,
                    max_length=max_length,
                    weight_dtype=torch.bfloat16,
                    num_inference_steps=14,
                    guidance_scale=4.5,
                    num_images_per_prompt=25,
                    device="cuda",
                    random_seed=42,
                    manipulated_mask_func=mask_semantic_parts_attention,
                    part_to_mask=part_to_mask
                )
                
                # Clear CUDA cache to prevent memory issues
                torch.cuda.empty_cache()
        
        print(f"\n✅ Layer ablation experiment completed successfully!")
        print(f"Results saved to: {metrics_file}")
        
        # Print summary
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                results = json.load(f)
            print(f"\nGenerated {len(results)} different conditions")
            
            # Show baseline and best/worst performing conditions
            baseline_score = results.get('baseline', {}).get('overall_score', 0)
            print(f"Baseline overall score: {baseline_score:.3f}")
            
            # Find best and worst performing layer conditions
            layer_results = {k: v for k, v in results.items() if k != 'baseline'}
            if layer_results:
                best_condition = max(layer_results.items(), key=lambda x: x[1]['overall_score'])
                worst_condition = min(layer_results.items(), key=lambda x: x[1]['overall_score'])
                
                print(f"Best performing: {best_condition[0]} (score: {best_condition[1]['overall_score']:.3f})")
                print(f"Worst performing: {worst_condition[0]} (score: {worst_condition[1]['overall_score']:.3f})")
        
        return metrics_file
        
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run layer ablation experiments')
    parser.add_argument('--test', action='store_true', help='Run quick test with default settings')
    parser.add_argument('--prompt', type=str, default="red circle is to the left of blue square", 
                       help='Text prompt to test')
    parser.add_argument('--layers', type=str, default="0-27", 
                       help='Layer indices to test (e.g., "0-27" for all layers, "5,10,15" for specific layers)')
    parser.add_argument('--parts', type=str, default="colors,objects,spatial", 
                       help='Semantic parts to mask (comma-separated)')
    parser.add_argument('--save_dir', type=str, default='/n/home13/xupan/sompolinsky_lab/object_relation/layer_ablation/',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.test:
        # Run the quick test function
        test_single_prompt_layer_ablation()
    else:
        # Parse layer indices
        if '-' in args.layers:
            start, end = map(int, args.layers.split('-'))
            layers_to_test = list(range(start, end + 1))
        else:
            layers_to_test = [int(x.strip()) for x in args.layers.split(',')]
        
        # Parse semantic parts
        parts_to_mask = [x.strip() for x in args.parts.split(',')]
        
        print(f"Running layer ablation experiment:")
        print(f"  Prompt: {args.prompt}")
        print(f"  Layers: {layers_to_test}")
        print(f"  Parts: {parts_to_mask}")
        print(f"  Save dir: {args.save_dir}")
        
        # Run the experiment
        result_file = layer_ablation_experiment(
            prompt=args.prompt,
            layers_to_test=layers_to_test,
            parts_to_mask=parts_to_mask,
            save_dir=args.save_dir
        )
        
        if result_file:
            print(f"\n✅ Experiment completed! Results saved to: {result_file}")
        else:
            print("\n❌ Experiment failed!")
    
    # Example usage:
    # python layer_attention_masking.py --test  # Quick test
    # python layer_attention_masking.py --prompt "red circle is to the left of blue square" --layers "0-10" --parts "spatial"
    # python layer_attention_masking.py --layers "5,10,15,20" --parts "colors,objects" 