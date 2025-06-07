import os
import torch
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler

@torch.inference_mode()
def visualize_prompts(pipeline, validation_prompts, prompt_cache_dir, max_length=120, weight_dtype=torch.float16,
                   num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25, device="cuda"):
    # logger.info("Running validation... ")
    # device = accelerator.device
    # model = accelerator.unwrap_model(model)
    if validation_prompts is None:
        validation_prompts = [
            "triangle is to the upper left of square", 
            "blue triangle is to the upper left of red square", 
            "triangle is above and to the right of square", 
            "blue circle is above and to the right of blue square", 
            "triangle is to the left of square", 
            "triangle is to the left of triangle", 
            "circle is below red square",
            "red circle is to the left of blue square",
            "blue square is to the right of red circle",
            "red circle is above square",
            "triangle is above red circle",
            "red is above blue",
            "red is to the left of red",
            "blue triangle is above red triangle", 
            "blue circle is above blue square", 
        ]
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(0)
    image_logs = []
    images = []
    latents = []
    uncond_data = torch.load(f'{prompt_cache_dir}/uncond_{max_length}token.pth', map_location='cpu')
    uncond_prompt_embeds = uncond_data['caption_embeds'].to(device)
    uncond_prompt_attention_mask = uncond_data['emb_mask'].to(device)
    visualized_prompts = []
    for _, prompt in enumerate(validation_prompts):
        if not os.path.exists(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth'):
            continue
        embed = torch.load(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth', map_location='cpu')
        caption_embs, emb_masks = embed['caption_embeds'].to(device), embed['emb_mask'].to(device)
        latents.append(pipeline(
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            prompt_embeds=caption_embs,
            prompt_attention_mask=emb_masks,
            negative_prompt=None,
            negative_prompt_embeds=uncond_prompt_embeds,
            negative_prompt_attention_mask=uncond_prompt_attention_mask,
            use_resolution_binning=False, # need this for smaller images like ours. 
            output_type="latent",
        ).images)
        visualized_prompts.append(prompt)
    # flush()
    for latent in latents:
        images.append(pipeline.vae.decode(latent.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0])
    for prompt, image in zip(visualized_prompts, images):
        image = pipeline.image_processor.postprocess(image, output_type="pil")
        image_logs.append({"validation_prompt": prompt, "images": image})

    return image_logs

# %%
@torch.inference_mode()
def load_embed_and_mask(validation_prompts, prompt_cache_dir, max_length=120, device="cuda"):
    # logger.info("Running validation... ")
    # device = accelerator.device
    # model = accelerator.unwrap_model(model)
    if validation_prompts is None:
        validation_prompts = [
            "triangle is to the upper left of square", 
            "blue triangle is to the upper left of red square", 
            "triangle is above and to the right of square", 
            "blue circle is above and to the right of blue square", 
            "triangle is to the left of square", 
            "triangle is to the left of triangle", 
            "circle is below red square",
            "red circle is to the left of blue square",
            "blue square is to the right of red circle",
            "red circle is above square",
            "triangle is above red circle",
            "red is above blue",
            "red is to the left of red",
            "blue triangle is above red triangle", 
            "blue circle is above blue square", 
        ]
    embed_infos = []
    for _, prompt in enumerate(validation_prompts):
        if not os.path.exists(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth'):
            continue
        embed = torch.load(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth', map_location='cpu')
        caption_embs, emb_masks = embed['caption_embeds'].to(device), embed['emb_mask'].to(device)
        embed_infos.append({"caption_embeds": caption_embs, "emb_mask": emb_masks, "prompt": prompt})
    uncond_data = torch.load(f'{prompt_cache_dir}/uncond_{max_length}token.pth', map_location='cpu')
    uncond_prompt_embeds = uncond_data['caption_embeds'].to(device)
    uncond_prompt_attention_mask = uncond_data['emb_mask'].to(device)
    embed_infos.append({"caption_embeds": uncond_prompt_embeds, "emb_mask": uncond_prompt_attention_mask, "prompt": ""})
    return embed_infos

# %%
@torch.inference_mode()
def visualize_prompts_with_traj(pipeline, validation_prompts, prompt_cache_dir, max_length=120, weight_dtype=torch.float16,
                   num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25, device="cuda", random_seed=0):
    # logger.info("Running validation... ")
    # device = accelerator.device
    # model = accelerator.unwrap_model(model)
    if validation_prompts is None:
        validation_prompts = [
            "triangle is to the upper left of square", 
            "blue triangle is to the upper left of red square", 
            "triangle is above and to the right of square", 
            "blue circle is above and to the right of blue square", 
            "triangle is to the left of square", 
            "triangle is to the left of triangle", 
            "circle is below red square",
            "red circle is to the left of blue square",
            "blue square is to the right of red circle",
            "red circle is above square",
            "triangle is above red circle",
            "red is above blue",
            "red is to the left of red",
            "blue triangle is above red triangle", 
            "blue circle is above blue square", 
        ]
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if random_seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    image_logs = []
    images = []
    latents = []
    pred_traj = []
    latents_traj = []
    t_traj = []
    uncond_data = torch.load(f'{prompt_cache_dir}/uncond_{max_length}token.pth', map_location='cpu')
    uncond_prompt_embeds = uncond_data['caption_embeds'].to(device)
    uncond_prompt_attention_mask = uncond_data['emb_mask'].to(device)
    visualized_prompts = []
    for _, prompt in enumerate(validation_prompts):
        if not os.path.exists(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth'):
            continue
        embed = torch.load(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth', map_location='cpu')
        caption_embs, emb_masks = embed['caption_embeds'].to(device), embed['emb_mask'].to(device)
        output = pipeline(
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            prompt_embeds=caption_embs,
            prompt_attention_mask=emb_masks,
            negative_prompt=None,
            negative_prompt_embeds=uncond_prompt_embeds,
            negative_prompt_attention_mask=uncond_prompt_attention_mask,
            use_resolution_binning=False, # need this for smaller images like ours. 
            return_sample_pred_traj=True,
            output_type="latent",
        )
        latents.append(output[0].images)
        pred_traj.append(output[1])
        latents_traj.append(output[2])
        t_traj.append(output[3])
        visualized_prompts.append(prompt)
    # flush()
    for latent in latents:
        images.append(pipeline.vae.decode(latent.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0])
    for prompt, image in zip(visualized_prompts, images):
        image = pipeline.image_processor.postprocess(image, output_type="pil")
        image_logs.append({"validation_prompt": prompt, "images": image})
    
    return image_logs, latents_traj, pred_traj, t_traj

@torch.inference_mode()
def visualize_prompts_with_traj_pretrained(pipeline, validation_prompts, num_inference_steps=14, guidance_scale=4.5, 
                                num_images_per_prompt=1, device=torch.device("cuda"), random_seed=0, weight_dtype=torch.bfloat16):

    # Move pipeline to GPU (or multi-GPU)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Set up generator
    if random_seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    #image_logs = []
    #print(generator.device.type)
    image_logs, images, latents, pred_traj, latents_traj, t_traj, visualized_prompts = [], [], [], [], [], [], []

    for prompt in validation_prompts:
        # Run pipeline
        output = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            use_resolution_binning=False,
            return_sample_pred_traj=True,
            output_type="latent",
            device=device,
        )
        # Store latents and trajectories
        latents.append(output[0].images)
        pred_traj.append(output[1])
        latents_traj.append(output[2])
        t_traj.append(output[3])
        visualized_prompts.append(prompt)

    for latent in latents:
        images.append(pipeline.vae.decode(latent.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0])

    # Postprocess images (small VRAM impact)
    for prompt, image in zip(visualized_prompts, images):
        image = pipeline.image_processor.postprocess(image, output_type="pil")
        image_logs.append({"validation_prompt": prompt, "images": image})
    print("Function execution complete!")
    return image_logs, latents_traj, pred_traj, t_traj



# subclass a new pipeline from PixArtAlphaPipeline
from typing import Callable, List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps
# from diffusers.pipelines.pixart_alpha import EXAMPLE_DOC_STRING, ImagePipelineOutput

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
        device: str = "cuda",
        inference_step_star: Optional[int] = None,
        post_prompt_attention_mask: Optional[torch.Tensor] = None,
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
            inference_step_star (`int`, *optional*): The specific inference step at which to change
                the `prompt_attention_mask`. Defaults to `None`, meaning no change.
            post_prompt_attention_mask (`torch.Tensor`, *optional*): The new attention mask for the conditional
                prompt to be used from `inference_step_star` onwards. Required if `inference_step_star` is set.
                Should have the same shape as the original conditional prompt attention mask. Defaults to `None`.

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
        #print(prompt_attention_mask.shape)
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
        original_prompt_embeds = prompt_embeds
        original_negative_prompt_embeds = negative_prompt_embeds
        original_negative_prompt_attention_mask = negative_prompt_attention_mask
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

        if inference_step_star is not None and post_prompt_attention_mask is not None:
            # Encode prompt for post operations
            (
                post_prompt_embeds,
                post_prompt_attention_mask,
                post_negative_prompt_embeds,
                post_negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt, # Use the original prompt
                do_classifier_free_guidance,
                negative_prompt=negative_prompt, # Use the original negative prompt
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=original_prompt_embeds,
                negative_prompt_embeds=original_negative_prompt_embeds,
                prompt_attention_mask=post_prompt_attention_mask,
                negative_prompt_attention_mask=original_negative_prompt_attention_mask,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
            )


        # `prompt_embeds`, `prompt_attention_mask`, etc. now hold the results from `encode_prompt`.
        uncond_mask_for_step_change = None # Initialize
        if do_classifier_free_guidance:
            # Store the unconditional mask that came from encode_prompt before concatenation,
            # as we'll need it if we change the conditional mask mid-process.
            # At this point, negative_prompt_attention_mask is the one from encode_prompt.
            uncond_mask_for_step_change = negative_prompt_attention_mask
            
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

            if inference_step_star is not None and post_prompt_attention_mask is not None:
                # Concatenate post embeddings and masks for CFG
                post_prompt_embeds = torch.cat([post_negative_prompt_embeds, post_prompt_embeds], dim=0)
                # The post_prompt_attention_mask is already the *conditional* mask.
                # We need to combine it with the *unconditional* mask (post_negative_prompt_attention_mask).
                post_prompt_attention_mask = torch.cat([post_negative_prompt_attention_mask, post_prompt_attention_mask], dim=0)

        # Original print statements referred to prompt_embeds and prompt_attention_mask
        # after potential concatenation for CFG.
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

                # Determine which embeddings and mask to use
                current_prompt_embeds = prompt_embeds
                current_prompt_attention_mask = prompt_attention_mask

                # Check if we need to update the attention mask
                if inference_step_star is not None and post_prompt_attention_mask is not None:
                    if i >= inference_step_star: 
                        current_prompt_embeds = post_prompt_embeds
                    
                        current_prompt_attention_mask = post_prompt_attention_mask
                        print(f"Step {i}: Using POST prompt_attention_mask shape: {current_prompt_attention_mask.shape}")
                    
                    elif i < inference_step_star : # or the conditions for post are not met
                        print(f"Step {i}: Using PRE prompt_attention_mask shape: {current_prompt_attention_mask.shape}")


                # predict noise model_output
                #print(current_prompt_attention_mask)
                
                # if i >= inference_step_star:
                #     noise_pred = self.transformer(
                #         latent_model_input,
                #         encoder_hidden_states=torch.zeros_like(current_prompt_embeds), # Use current_prompt_embeds
                #         encoder_attention_mask=current_prompt_attention_mask, # Use current_prompt_attention_mask
                #         timestep=current_timestep,
                #         added_cond_kwargs=added_cond_kwargs,
                #         return_dict=False,
                #     )[0]
                # else:
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=current_prompt_embeds, # Use current_prompt_embeds
                    encoder_attention_mask=current_prompt_attention_mask, # Use current_prompt_attention_mask
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

        latents_traj.append(latents.cpu())
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
        latents_traj = torch.stack(latents_traj)
        pred_traj = torch.stack(pred_traj)
        # t_traj = torch.stack(t_traj)
        if not return_dict:
            return (image,)
        if return_sample_pred_traj:
            return ImagePipelineOutput(images=image), pred_traj, latents_traj, t_traj
        return ImagePipelineOutput(images=image)


@torch.inference_mode()
def visualize_prompts_with_traj_from_embed_dict(pipeline, uncond_prompt_dict, cond_prompt_dict, weight_dtype=torch.float16,
                   num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25, device="cuda", random_seed=0):
    # logger.info("Running validation... ")
    # device = accelerator.device
    # model = accelerator.unwrap_model(model)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if random_seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    image_logs = []
    images = []
    latents = []
    pred_traj = []
    latents_traj = []
    t_traj = []
    visualized_prompts = []
    # uncond_data = torch.load(f'{prompt_cache_dir}/uncond_{max_length}token.pth', map_location='cpu')
    uncond_prompt_embeds = uncond_prompt_dict['caption_embeds'].to(device)
    uncond_prompt_attention_mask = uncond_prompt_dict['emb_mask'].to(device)
    # for _, prompt in enumerate(validation_prompts):
    caption_embs = cond_prompt_dict['caption_embeds'].to(device)
    emb_masks = cond_prompt_dict['emb_mask'].to(device)
    output = pipeline(
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        prompt_embeds=caption_embs,
        prompt_attention_mask=emb_masks,
        negative_prompt=None,
        negative_prompt_embeds=uncond_prompt_embeds,
        negative_prompt_attention_mask=uncond_prompt_attention_mask,
        use_resolution_binning=False, # need this for smaller images like ours. 
        return_sample_pred_traj=True,
        output_type="latent",
    )
    latents.append(output[0].images)
    pred_traj.append(output[1])
    latents_traj.append(output[2])
    t_traj.append(output[3])
    visualized_prompts.append(prompt)
    # flush()
    for latent in latents:
        images.append(pipeline.vae.decode(latent.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0])
    for prompt, image in zip(visualized_prompts, images):
        image = pipeline.image_processor.postprocess(image, output_type="pil")
        image_logs.append({"validation_prompt": prompt, "images": image})
    
    return image_logs, latents_traj, pred_traj, t_traj


@torch.inference_mode()
def pipeline_inference_custom(pipeline, prompt, negative_prompt="", num_inference_steps=14, num_images_per_prompt=25, guidance_scale=4.5, random_seed=0, max_sequence_length=20, 
                              return_sample_pred_traj=True, output_type="latent", device="cuda", weight_dtype=torch.float16, **kwargs):
    pipeline = pipeline.to(device)
    output = pipeline(prompt=prompt,
         negative_prompt=negative_prompt,
         num_inference_steps=num_inference_steps,
         num_images_per_prompt=num_images_per_prompt,
         generator=torch.Generator(device=device).manual_seed(random_seed),
         guidance_scale=guidance_scale,
         max_sequence_length=max_sequence_length,
         use_resolution_binning=False,
         return_sample_pred_traj=return_sample_pred_traj,
         output_type=output_type,
         **kwargs,
    )
    if return_sample_pred_traj:
        latents = output[0].images
        pred_traj = output[1]
        latents_traj = output[2]
        t_traj = output[3]
    else:
        latents = output.images
        pred_traj = None
        latents_traj = None
        t_traj = None
    images = pipeline.vae.decode(latents.clone().to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    image_logs = []
    images = pipeline.image_processor.postprocess(images, output_type="pil")
    image_logs.append({"validation_prompt": prompt, "images": images})
    return image_logs, pred_traj, latents_traj, t_traj

@torch.inference_mode()
def visualize_single_prompt_with_traj(pipeline, prompt, save_dir, num_inference_steps=14, guidance_scale=4.5, 
                                    device=torch.device("cuda"), random_seed=0, weight_dtype=torch.float16):
    """
    Generate and save trajectory data for a single prompt.
    
    Args:
        pipeline: The PixArt pipeline
        prompt (str): The prompt to generate from
        save_dir (str): Directory to save all outputs
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Guidance scale for generation
        device: Device to run generation on
        random_seed (int): Random seed for reproducibility
        weight_dtype: Data type for weights
    
    Returns:
        None (saves all outputs to disk)
    """
    # Move pipeline to device
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Set up generator
    if random_seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(random_seed)

    # Create a valid filename from the prompt
    filename_base = "".join(c if c.isalnum() else "_" for c in prompt)
    
    # Create subdirectories
    subdirs = {
        'image': os.path.join(save_dir, 'generated_image'),
        'latents': os.path.join(save_dir, 'latents_traj'),
        'pred': os.path.join(save_dir, 'pred_traj')
    }
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)

    # Run pipeline
    output = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,  # Always 1 for this function
        generator=generator,
        guidance_scale=guidance_scale,
        use_resolution_binning=False,
        return_sample_pred_traj=True,
        output_type="latent",
        device=device,
    )

    # Extract outputs
    latents = output[0].images
    pred_traj = output[1]
    latents_traj = output[2]
    t_traj = output[3]

    # Decode and save the final image
    image = pipeline.vae.decode(latents.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")
    image_path = os.path.join(subdirs['image'], f"{filename_base}.png")
    image.save(image_path)
    print(f"Generated image saved to: {image_path}")

    # Save trajectory data
    latents_path = os.path.join(subdirs['latents'], f"{filename_base}.pt")
    pred_path = os.path.join(subdirs['pred'], f"{filename_base}.pt")
    
    torch.save(latents_traj, latents_path)
    torch.save(pred_traj, pred_path)
    
    print(f"Latents trajectory saved to: {latents_path}")
    print(f"Prediction trajectory saved to: {pred_path}")
