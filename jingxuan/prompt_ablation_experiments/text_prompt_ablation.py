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
from typing import Callable, List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps

# Clear CUDA cache at the start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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
        reverse_mask: bool = False,
        one_step_mode: bool = False,
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
            reverse_mask (`bool`, *optional*, defaults to False): If True, reverses the behavior of when to apply the
                post_prompt_attention_mask - it will be applied before inference_step_star instead of after. This allows
                for switching between two different attention masks at a specific timestep.
            one_step_mode (`bool`, *optional*, defaults to False): If True, only uses normal prompt attention mask for
                inference_step_star and the step after, and post_prompt_attention_mask for all other steps.

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
        #print(prompt_embeds.shape)
        #print(prompt_attention_mask.shape)
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
        denoiser_traj = []
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
                    if one_step_mode:
                        # Use normal prompt attention mask only for inference_step_star and the step after
                        if i == inference_step_star: #or i == inference_step_star + 1:
                            current_prompt_embeds = prompt_embeds
                            current_prompt_attention_mask = prompt_attention_mask
                        else:
                            current_prompt_embeds = post_prompt_embeds
                            current_prompt_attention_mask = post_prompt_attention_mask
                    else:
                        if i >= inference_step_star: 
                            if not reverse_mask:
                                current_prompt_embeds = post_prompt_embeds
                                current_prompt_attention_mask = post_prompt_attention_mask
                            else:
                                pass
                        elif i < inference_step_star:
                            if not reverse_mask:
                                pass
                            else:
                                current_prompt_embeds = post_prompt_embeds
                                current_prompt_attention_mask = post_prompt_attention_mask

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
                # save denoiser
                # print(self.scheduler.step_index)
                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)
                denoiser_traj.append(self.scheduler.convert_model_output(model_output=noise_pred, sample=latents))
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
        denoiser_image_traj = []
        for i in range(len(denoiser_traj)):
            denoiser_image = self.vae.decode(denoiser_traj[i] / self.vae.config.scaling_factor, return_dict=False)[0]
            denoiser_image = self.image_processor.postprocess(denoiser_image.detach(), output_type="pil")
            denoiser_image_traj.append(denoiser_image)
        # Offload all models
        self.maybe_free_model_hooks()
        latents_traj = torch.stack(latents_traj)
        pred_traj = torch.stack(pred_traj)
        # t_traj = torch.stack(t_traj)
        if not return_dict:
            return (image,)
        if return_sample_pred_traj:
            return ImagePipelineOutput(images=image), pred_traj, latents_traj, t_traj, denoiser_image_traj
        return ImagePipelineOutput(images=image)

def visualize_prompt_with_traj(pipeline, prompt, max_length=120, weight_dtype=torch.bfloat16,
                   num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25, num_image_to_plot=0, device="cuda", random_seed=0,inference_step_star=None,manipulated_mask_func=None,reverse_mask=False,one_step_mode=False,**mask_kwargs):
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if random_seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    
    # load negative prompt embeds
    # uncond_data = torch.load(f'{prompt_cache_dir}/uncond_{max_length}token.pth', map_location='cpu')
    # uncond_prompt_embeds = uncond_data['caption_embeds'].to(device)
    # uncond_prompt_attention_mask = uncond_data['emb_mask'].to(device)
    
    #if not os.path.exists(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth'):
        #raise FileNotFoundError(f"Prompt cache not found for: {prompt}")
        
    #embed = torch.load(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth', map_location='cpu')
    #caption_embs, emb_masks = embed['caption_embeds'].to(device), embed['emb_mask'].to(device)
    # Tokenize the prompt
    inputs = pipeline.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=max_length).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    #print(attention_mask.shape)
    if inference_step_star is None:
        manipulated_mask = None
    else:
        manipulated_mask = manipulated_mask_func(prompt, pipeline.tokenizer, max_length, device, **mask_kwargs)
    #print(manipulated_mask)
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
        inference_step_star=inference_step_star,
        post_prompt_attention_mask=manipulated_mask,
        reverse_mask=reverse_mask,
        one_step_mode=one_step_mode
    )
    #print(output)
    latents = output[0].images
    #print(latents)
    pred_traj = output[1]
    latents_traj = output[2]
    t_traj = output[3]
    denoiser_traj = output[4]
    
    # Decode latents to images
    images = pipeline.vae.decode(latents.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    # Convert images from bfloat16 to float16 to match bias type
    images = pipeline.image_processor.postprocess(images.detach(), output_type="pil")
    
    return {"prompt": prompt, "images": images}, latents_traj, pred_traj, t_traj,denoiser_traj

def plot_denoiser_traj(axes, denoiser_traj, num_image_to_plot,inference_step_star):
    for i in range(14):
        axes[i].imshow(denoiser_traj[i][num_image_to_plot])
        # Draw vertical line to the right of each subplot
        if i == inference_step_star:
            axes[i].text(-0.2, 0.5, '*', transform=axes[i].transAxes, fontsize=15, color='black')
        axes[i].axis('off')  # Turn off axis labels

# Calculate average scores across all images
def calculate_gen_images_scores(image_logs,prompt):
    shape_match_scores = []
    color_binding_scores = []
    spatial_color_scores = []
    spatial_shape_scores = []
    overall_scores = []

    for image in image_logs['images']:
        # Get objects from image
        df = find_classify_objects(image, area_threshold=100, radius=16.0)
        #print(df.head())
        
        #Check if DataFrame is empty or missing required columns
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

def save_end_imgs(prompt,save_dir, image_logs,inference_step_star,manipulated_mask_func,mask_kwargs):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subdirectory based on prompt and generation parameters
    if inference_step_star is None:
        suffix = 'original'
    else:
        mask_func_name = manipulated_mask_func.__name__ if manipulated_mask_func else 'no_mask'
        suffix = f'step_{inference_step_star}_{mask_func_name}'
        if mask_kwargs:
            suffix = f"{suffix}_{mask_kwargs}"
    
    save_path = os.path.join(save_dir, prompt.replace(' ', '_'))
    os.makedirs(save_path, exist_ok=True)
    
    # Save grid of images as a single PNG
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes_flat = axes.flatten()
    
    for i in range(25):
        axes_flat[i].imshow(image_logs['images'][i])
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(save_path, f'image_grid_{suffix}.png')
    plt.savefig(grid_path)
    plt.close()

def save_intermediate_imgs(prompt,save_dir, denoiser_traj,inference_step_star,manipulated_mask_func,mask_kwargs):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subdirectory based on prompt and generation parameters
    if inference_step_star is None:
        suffix = 'original'
    else:
        mask_func_name = manipulated_mask_func.__name__ if manipulated_mask_func else 'no_mask'
        suffix = f'step_{inference_step_star}_{mask_func_name}'
        if mask_kwargs:
            suffix = f"{suffix}_{mask_kwargs}"
    for i in range(len(denoiser_traj)):
        save_path = os.path.join(save_dir, prompt.replace(' ', '_'),f"step_{i}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save grid of images as a single PNG
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        axes_flat = axes.flatten()
        
        for j in range(25):
            axes_flat[j].imshow(denoiser_traj[i][j])
            axes_flat[j].axis('off')
        
        plt.tight_layout()
        grid_path = os.path.join(save_path, f'image_grid_{suffix}.png')
        plt.savefig(grid_path)
        plt.close()

def save_cv2_eval(prompt_dir,prompt,inference_step_star,manipulated_mask_func,reverse_mask,one_step_mode,avg_scores,mask_kwargs):
    # Create saved_metrics directory inside prompt_dir
    metrics_dir = os.path.join(prompt_dir, 'saved_metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Generate key name based on parameters
    if inference_step_star is None:
        key = 'original'
    else:
        mask_func_name = manipulated_mask_func.__name__ if manipulated_mask_func else 'no_mask'
        key = f'step_{inference_step_star}_{mask_func_name}'
        # Append mask_kwargs to key if they exist
        if mask_kwargs:
            key = f"{key}_{mask_kwargs}"
    
    # Load existing data if file exists, otherwise create new dict
    if one_step_mode:
        filename = os.path.join(metrics_dir, 'one_step.json')
    elif reverse_mask:
        filename = os.path.join(metrics_dir, 'ablate_before.json')
    else:
        filename = os.path.join(metrics_dir, 'ablate_after.json')
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


def generate_and_quantify_images(prompt_dir,axes, pipeline, prompt, max_length=120, weight_dtype=torch.bfloat16,
                   num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25, num_image_to_plot=0, device="cuda", random_seed=0,inference_step_star=None,manipulated_mask_func=None,reverse_mask=False,one_step_mode=False,**mask_kwargs):
    image_logs, latents_traj, pred_traj, t_traj, denoiser_traj = visualize_prompt_with_traj(pipeline=pipeline, prompt=prompt, max_length=max_length, weight_dtype=weight_dtype, num_images_per_prompt=num_images_per_prompt, \
        num_image_to_plot=num_image_to_plot, random_seed=random_seed,inference_step_star=inference_step_star, manipulated_mask_func=manipulated_mask_func,reverse_mask=reverse_mask,one_step_mode=one_step_mode,**mask_kwargs)
    # Save generated images
    # save_dir = '/n/home13/xupan/sompolinsky_lab/object_relation/prompt_ablation/end_imgs'
    #save_end_imgs(prompt,save_dir, image_logs,inference_step_star,manipulated_mask_func,mask_kwargs)
    #save_intermediate_imgs(prompt,save_dir, denoiser_traj,inference_step_star,manipulated_mask_func,mask_kwargs)
    # calculate scores
    avg_scores = calculate_gen_images_scores(image_logs,prompt)
    save_cv2_eval(prompt_dir=prompt_dir,prompt=prompt,inference_step_star=inference_step_star,manipulated_mask_func=manipulated_mask_func,reverse_mask=reverse_mask,one_step_mode=one_step_mode,avg_scores=avg_scores,mask_kwargs=mask_kwargs)
    plot_denoiser_traj(axes, denoiser_traj, num_image_to_plot,inference_step_star)
    # Clear variables to free memory
    del image_logs
    del latents_traj 
    del pred_traj
    del t_traj
    del denoiser_traj
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    #return image_logs

def generate_and_quantify_images_original(prompt_dir, pipeline, prompt, max_length=120, weight_dtype=torch.bfloat16,
                   num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25,  device="cuda", random_seed=0,inference_step_star=None,manipulated_mask_func=None,reverse_mask=False,one_step_mode=False,**mask_kwargs):
    image_logs, latents_traj, pred_traj, t_traj, denoiser_traj = visualize_prompt_with_traj(pipeline=pipeline, prompt=prompt, max_length=max_length, weight_dtype=weight_dtype, num_images_per_prompt=num_images_per_prompt, \
        random_seed=random_seed,inference_step_star=inference_step_star, manipulated_mask_func=manipulated_mask_func,reverse_mask=reverse_mask,one_step_mode=one_step_mode,**mask_kwargs)
    # Save generated images
    # save_dir = '/n/home13/xupan/sompolinsky_lab/object_relation/prompt_ablation/end_imgs'
    #save_end_imgs(prompt,save_dir, image_logs,inference_step_star,manipulated_mask_func,mask_kwargs)
    #save_intermediate_imgs(prompt,save_dir, denoiser_traj,inference_step_star,manipulated_mask_func,mask_kwargs)
    # calculate scores
    avg_scores = calculate_gen_images_scores(image_logs,prompt)
    save_cv2_eval(prompt_dir=prompt_dir,prompt=prompt,inference_step_star=inference_step_star,manipulated_mask_func=manipulated_mask_func,reverse_mask=reverse_mask,one_step_mode=one_step_mode,avg_scores=avg_scores,mask_kwargs=mask_kwargs)
    #plot_denoiser_traj(axes, denoiser_traj, num_image_to_plot,inference_step_star)
    # Clear variables to free memory
    del image_logs
    del latents_traj 
    del pred_traj
    del t_traj
    del denoiser_traj
    
    # Clear CUDA cache
    torch.cuda.empty_cache()

def generate_and_quantify_images_batch(prompt_dir, pipeline, prompt, max_length, weight_dtype, num_image_to_plot, mask_func, reverse_mask, one_step_mode,mask_type):
    fig, axeses = plt.subplots(14,14, figsize=(14,14))
    for j, axes in enumerate(axeses):
        generate_and_quantify_images(prompt_dir=prompt_dir, axes=axes, pipeline=pipeline, prompt=prompt, max_length=max_length, weight_dtype=weight_dtype,
            num_inference_steps=14, guidance_scale=4.5, num_images_per_prompt=25, num_image_to_plot=num_image_to_plot, device="cuda", random_seed=0,
            inference_step_star=j, manipulated_mask_func=mask_func, reverse_mask=reverse_mask,
            one_step_mode=one_step_mode, part_to_mask=mask_type)
    if one_step_mode:
        plt.savefig(os.path.join(prompt_dir, f'sample_{num_image_to_plot}_{mask_type}_one_step.png'))
    elif reverse_mask:
        plt.savefig(os.path.join(prompt_dir, f'sample_{num_image_to_plot}_{mask_type}_ablate_before.png'))
    else:
        plt.savefig(os.path.join(prompt_dir, f'sample_{num_image_to_plot}_{mask_type}_ablate_after.png'))
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot', help='Directory containing config and checkpoints')
    parser.add_argument('--t5_path', type=str, default='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/output/pretrained_models/t5_ckpts/t5-v1_1-xxl', help='Path to T5 model')
    parser.add_argument('--text_feat_dir', type=str, default='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/objectRel_pilot_rndembposemb/caption_feature_wmask', help='Directory containing word embedding dict')
    parser.add_argument('--save_dir', type=str, default='/n/home13/xupan/sompolinsky_lab/object_relation/prompt_ablation/', help='Directory to save results')
    parser.add_argument('--prompt_file', type=str, default='/n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation/jingxuan/prompt_ablation_experiments/spatial_custom.txt', help='Path to text file containing prompts (one per line)')
    parser.add_argument('--pretrained_model', type=str, default='PixArt-alpha/PixArt-XL-2-512x512', help='Path to pretrained model')
    parser.add_argument('--from_pretrained', default=False, help='Whether to load from pretrained model instead of custom trained model')
    parser.add_argument('--mask_func', type=str, default='mask_semantic_parts_attention', help='Mask function to use (use "None" to disable)')
    parser.add_argument('--reverse_mask', default=False, help='Whether to reverse mask')
    parser.add_argument('--one_step_mode', default=True, help='Whether to use one step mode')
    parser.add_argument('--num_image_to_plot', type=int, default=0, help='Number of images to plot')
    args = parser.parse_args()

    # Convert mask_func string to actual function
    if args.mask_func == "None":
        args.mask_func = None
    elif args.mask_func == "mask_semantic_parts_attention":
        args.mask_func = mask_semantic_parts_attention
    else:
        # Handle other mask functions if needed
        raise ValueError(f"Unknown mask function: {args.mask_func}")

    def load_custom_trained_model(model_dir, t5_path, text_feat_dir,weight_dtype=torch.bfloat16):
        # Load model checkpoint and config
        ckpt = torch.load(join(model_dir, "checkpoints", "epoch_4000_step_160000.pth"))
        config = read_config(join(model_dir, 'config.py'))
        config.mixed_precision = "bf16"
        
        # Initialize pipeline
        pipeline = construct_diffuser_pipeline_from_config(config, pipeline_class=PixArtAlphaPipeline_custom)
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
    
    def load_pretrained_model(pretrained_model,weight_dtype=torch.bfloat16):
        # fp16 will have overflow problem
        # bf16 and float32 are good.
        # Load the pretrained PixArt Alpha model from Hugging Face
        pipeline = PixArtAlphaPipeline_custom.from_pretrained(pretrained_model, torch_dtype=weight_dtype)
        max_length = pipeline.tokenizer.model_max_length
        return pipeline, max_length

    
    save_dir = args.save_dir
    # Read prompts from file
    with open(args.prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create figure and axes for all prompts
    num_prompts = len(prompts)
    torch.cuda.empty_cache()
    if args.from_pretrained:
        print("Loading pretrained model")
        pipeline, max_length = load_pretrained_model(args.pretrained_model)
    else:
        print("Loading custom trained model")
        pipeline, max_length = load_custom_trained_model(args.model_dir, args.t5_path, args.text_feat_dir)
    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{num_prompts}: {prompt}")
        prompt_dir = os.path.join(save_dir, prompt.replace(' ', '_'))
        os.makedirs(prompt_dir, exist_ok=True)
        
        if args.mask_func is not None:
            if os.path.exists(os.path.join(prompt_dir, 'sample_0_colors_one_step.png')):
                print(f"Skipping prompt {prompt} as saved_metrics already exists")
                continue
            parts_to_mask = ['colors', 'objects', 'spatial']
            for p in parts_to_mask:
                generate_and_quantify_images_batch(prompt_dir=prompt_dir, pipeline=pipeline, prompt=prompt, max_length=max_length, weight_dtype=torch.bfloat16, num_image_to_plot=args.num_image_to_plot, mask_func=args.mask_func, reverse_mask=args.reverse_mask, one_step_mode=args.one_step_mode, mask_type=p)
        else:
            generate_and_quantify_images_original(prompt_dir=prompt_dir, pipeline=pipeline, prompt=prompt, max_length=max_length, weight_dtype=torch.bfloat16,one_step_mode=args.one_step_mode)

if __name__ == "__main__":
    main()