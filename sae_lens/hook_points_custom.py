from collections import defaultdict
from typing import Callable, Any, Optional, List
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler

class PixArtAttentionVisualizer:
    """
    A class for capturing and analyzing attention patterns in PixArt's transformer-based model.
    """
    def __init__(self, pipe):
        self.pipe = pipe
        self.activation = defaultdict(list)  # To store activations by key
        self.hook_handles = []  # To manage hook lifecycles

    def clear_activation(self):
        """Clear all stored activations."""
        self.activation = defaultdict(list)

    def hook_forger(self, key: str, modify_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Create a hook to capture and optionally modify attention patterns.

        Args:
            key (str): Key to identify the activation.
            modify_fn (Callable): Optional function to modify activations during the forward pass.
        """
        def hook(module, input, output):
            output_data = output.detach().cpu()
            if modify_fn:
                output_data = modify_fn(output_data)
            self.activation[key].append(output_data)
            return output_data  # Return modified or original output
        return hook

    def hook_transformer_attention(self, module, module_id: str, modify_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Hook both self-attention and cross-attention modules in PixArt's transformer blocks.

        Args:
            module (torch.nn.Module): The transformer block to hook into.
            module_id (str): Unique identifier for the module.
            modify_fn (Callable): Optional function to modify activations.
        """
        hooks = []
        # For self-attention (attn1)
        if hasattr(module, 'attn1'):
            h1 = module.attn1.to_q.register_forward_hook(self.hook_forger(f"{module_id}_self_Q", modify_fn))
            h2 = module.attn1.to_k.register_forward_hook(self.hook_forger(f"{module_id}_self_K", modify_fn))
            hooks.extend([h1, h2])

        # For cross-attention (attn2)
        if hasattr(module, 'attn2'):
            h3 = module.attn2.to_q.register_forward_hook(self.hook_forger(f"{module_id}_cross_Q", modify_fn))
            h4 = module.attn2.to_k.register_forward_hook(self.hook_forger(f"{module_id}_cross_K", modify_fn))
            hooks.extend([h3, h4])

        return hooks

    def setup_hooks(self, modify_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Set up hooks for all transformer blocks in the pipeline.

        Args:
            modify_fn (Callable): Optional function to modify activations.
        """
        print("Setting up hooks for PixArt attention modules:")
        for block_idx, block in enumerate(self.pipe.transformer.transformer_blocks):
            print(f"- Hooking Block {block_idx}")
            hooks = self.hook_transformer_attention(block, f"block{block_idx:02d}", modify_fn)
            self.hook_handles.extend(hooks)

    def cleanup_hooks(self):
        """Remove all hooks."""
        print("Cleaning up hooks...")
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def compute_attention(self, module_id: str, timestep: int, attn_type: str = 'self'):
        Q = self.activation[f"{module_id}_{attn_type}_Q"][timestep]
        K = self.activation[f"{module_id}_{attn_type}_K"][timestep]
        scale = K.shape[-1] ** -0.5
        attention = scale * torch.bmm(Q.float(), K.transpose(-1, -2).float())
        attention = torch.softmax(attention, dim=-1)
        # Split attention for unconditional and conditional paths
        uncond_attention = attention[:attention.shape[0]//2]
        cond_attention = attention[attention.shape[0]//2:]
        return uncond_attention, cond_attention
        
    
    def visualize_attention(self, module_id: str, timestep: int, tokens: Tuple[List[str], List[str]],
                          attn_type: str = 'self', figsize: Tuple[int, int] = (20, 15)):
        """Visualize attention patterns for both unconditional and conditional paths"""
        neg_tokens, pos_tokens = tokens
        Q = self.activation[f"{module_id}_{attn_type}_Q"][timestep]
        K = self.activation[f"{module_id}_{attn_type}_K"][timestep]

        if Q is None or K is None:
            print(f"No attention data found for {module_id} at timestep {timestep}")
            return

        # Compute attention scores for both paths
        scale = K.shape[-1] ** -0.5
        attention = scale * torch.bmm(Q.float(), K.transpose(-1, -2).float())
        attention = torch.softmax(attention, dim=-1)

        # Split attention for unconditional and conditional paths
        uncond_attention = attention[:attention.shape[0]//2]
        cond_attention = attention[attention.shape[0]//2:]

        # Create subplot for both paths
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot unconditional attention
        n_neg_tokens = min(16, len(neg_tokens))
        uncond_map = uncond_attention[0, :, :n_neg_tokens].reshape(-1, n_neg_tokens)
        sns.heatmap(uncond_map.cpu().numpy(), ax=ax1, cmap='viridis')
        ax1.set_title(f"Unconditional Attention\n(Negative Prompt)")
        ax1.set_xticks(np.arange(n_neg_tokens) + 0.5)
        ax1.set_xticklabels(neg_tokens[:n_neg_tokens], rotation=45)

        # Plot conditional attention
        n_pos_tokens = min(16, len(pos_tokens))
        cond_map = cond_attention[0, :, :n_pos_tokens].reshape(-1, n_pos_tokens)
        sns.heatmap(cond_map.cpu().numpy(), ax=ax2, cmap='viridis')
        ax2.set_title(f"Conditional Attention\n(Positive Prompt)")
        ax2.set_xticks(np.arange(n_pos_tokens) + 0.5)
        ax2.set_xticklabels(pos_tokens[:n_pos_tokens], rotation=45)

        plt.suptitle(f'{attn_type.capitalize()} Attention: {module_id}, Step {timestep}')
        plt.tight_layout()
        plt.show()

    def analyze_attention_patterns(self, prompt: str, negative_prompt: str = "",
                                 timesteps_to_show: List[int] = None,
                                 guidance_scale: float = 4.5):
        """Generate image and analyze attention patterns with CFG"""
        print(f"Analyzing attention patterns:")
        print(f"Positive prompt: {prompt}")
        print(f"Negative prompt: {negative_prompt}")

        # Generate image and collect attention
        output, tokens = self.generate_with_attention(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale
        )

        # Show generated image
        plt.figure(figsize=(12, 12))
        plt.imshow(output.images[0])
        plt.title(f"Generated Image\nPrompt: {prompt}\nNegative Prompt: {negative_prompt}")
        plt.axis('off')
        plt.show()

        # If timesteps not specified, sample a few
        if timesteps_to_show is None:
            timesteps_to_show = list(range(0, 20, 5))

        # Visualize patterns for each transformer block
        for block_idx in range(len(self.pipe.transformer.transformer_blocks)):
            module_id = f"block{block_idx:02d}"
            print(f"\nAnalyzing Block {block_idx}")

            for timestep in timesteps_to_show:
                print(f"Timestep {timestep}")
                # Visualize self-attention
                self.visualize_attention(module_id, timestep, tokens, 'self')
                # Visualize cross-attention
                self.visualize_attention(module_id, timestep, tokens, 'cross')


class PixArtAlphaPipeline_custom(PixArtAlphaPipeline):
    
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
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
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

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

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            # deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)
        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # if use_resolution_binning:
        #     if self.transformer.config.sample_size == 128:
        #         aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        #     elif self.transformer.config.sample_size == 64:
        #         aspect_ratio_bin = ASPECT_RATIO_512_BIN
        #     elif self.transformer.config.sample_size == 32:
        #         aspect_ratio_bin = ASPECT_RATIO_256_BIN
        #     else:
        #         raise ValueError("Invalid sample size")
        #     orig_height, orig_width = height, width
        #     height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

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
        print(prompt_embeds.shape)
        print(prompt_attention_mask.shape)
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

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        pred_traj = []
        latents_traj = []
        t_traj = []
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

                # predict noise model_output
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

                latents_traj.append(latents)
                pred_traj.append(noise_pred)
                # compute previous image: x_t -> x_t-1
                if num_inference_steps == 1:
                    # For DMD one step sampling: https://arxiv.org/abs/2311.18828
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                # pred_traj.append(self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample)
                
                t_traj.append(t)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        latents_traj.append(latents)
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # if use_resolution_binning:
            #     image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        if return_sample_pred_traj:
            return ImagePipelineOutput(images=image), pred_traj, latents_traj, t_traj
        return ImagePipelineOutput(images=image)