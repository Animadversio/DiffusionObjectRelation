import os
from os.path import join 
import torch
import sys
sys.path.append("/n/home13/xupan/Projects/DiffusionObjectRelation/DiffusionObjectRelation/PixArt-alpha")
from diffusion import IDDPM
# from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
sys.path.append("/n/home13/xupan/Projects/DiffusionObjectRelation/DiffusionObjectRelation/utils")
from pixart_utils import state_dict_convert
from image_utils import pil_images_to_grid
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler

# Add a new hook to get the embedding based on Binxu's code
# subclass a new pipeline from PixArtAlphaPipeline
from typing import Callable, List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps
from collections import defaultdict
# from diffusers.pipelines.pixart_alpha import EXAMPLE_DOC_STRING, ImagePipelineOutput
import matplotlib.pyplot as plt
from tqdm import tqdm


class PixArtAlphaPipeline_hookembedding(PixArtAlphaPipeline):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.hook_handles = []
    #     self.embedding = defaultdict(list)
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        pipeline = super().from_pretrained(*args, **kwargs)
        pipeline.hook_handles = []
        pipeline.embedding = defaultdict(list)
        return pipeline

    def clear_embedding(self):
        self.embedding = defaultdict(list)
    
    def hook_forger(self, key: str):
        """Create a hook to capture attention patterns"""
        def hook(module, input, output):
            self.embedding[key].append(input[0].chunk(2)[0].detach().cpu().numpy()) # only use the first half of the embedding, the second half is the negative embedding
        return hook
    
    def setup_embedding_hooks(self, embedding_layer: int = None):
        """Set up hooks for all transformer blocks"""
        # print("Setting up hooks for PixArt attention modules:")
        if embedding_layer is None:
            for block_idx, block in enumerate(self.transformer.transformer_blocks):
                self.hook_handles.append(block.register_forward_hook(self.hook_forger(f"block{block_idx:02d}")))
        else:
            for block_idx, block in enumerate(self.transformer.transformer_blocks):
                if block_idx == embedding_layer:
                    self.hook_handles.append(block.register_forward_hook(self.hook_forger(f"block{block_idx:02d}")))
                    break

    def cleanup_embedding_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    @torch.no_grad()
    def __call__(
        self,
        embedding_when: float = 0.6, # relative time step to hook the embedding. 0.6 means 60% of the way through the diffusion process.
        embedding_layer: int = None, # if None, hook all layers
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
        #########################################
        # initialize the embedding hook
        self.clear_embedding()
        self.cleanup_embedding_hooks()
        #########################################
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
        # print(prompt_embeds.shape)
        # print(prompt_attention_mask.shape)
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        ################################################
        # which timestep to hook the embedding
        hook_timestep = timesteps[int(embedding_when * num_inference_steps)]
        ################################################

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

                if t == hook_timestep:
                    self.setup_embedding_hooks(embedding_layer)

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

                # early stop if reached the hook timestep
                if t == hook_timestep:
                    self.cleanup_embedding_hooks()
                    # return self.embedding

        latents_traj.append(latents)
        if not output_type == "latent":
            image = pipeline.vae.decode(latents.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0]
            image = pipeline.image_processor.postprocess(image, output_type="pil")
        else:
            image = latents

        # if not output_type == "latent":
        #     image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (self.embedding, image,)
        if return_sample_pred_traj:
            return ImagePipelineOutput(images=image), pred_traj, latents_traj, t_traj
        return ImagePipelineOutput(images=image)
    

@torch.inference_mode()
def get_embeddings(pipeline, validation_prompts, prompt_cache_dir, embedding_when=0.6, embedding_layer=None, max_length=120, weight_dtype=torch.float16,
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

    uncond_data = torch.load(f'{prompt_cache_dir}/uncond_{max_length}token.pth', map_location='cpu')
    uncond_prompt_embeds = uncond_data['caption_embeds'].to(device)
    uncond_prompt_attention_mask = uncond_data['emb_mask'].to(device)

    embeddings = []
    images = []

    for _, prompt in enumerate(validation_prompts):
        if not os.path.exists(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth'):
            continue
        embed = torch.load(f'{prompt_cache_dir}/{prompt}_{max_length}token.pth', map_location='cpu')
        caption_embs, emb_masks = embed['caption_embeds'].to(device), embed['emb_mask'].to(device)
        output = pipeline(
            embedding_when=embedding_when,
            embedding_layer=embedding_layer,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            # generator=generator,
            guidance_scale=guidance_scale,
            prompt_embeds=caption_embs,
            prompt_attention_mask=emb_masks,
            negative_prompt=None,
            negative_prompt_embeds=uncond_prompt_embeds,
            negative_prompt_attention_mask=uncond_prompt_attention_mask,
            use_resolution_binning=False, # need this for smaller images like ours. 
            return_sample_pred_traj=False,
            return_dict=False,
            output_type="pil",
        )
    
        embeddings.append(output[0])
        images.append(output[1])

    return embeddings, images


##############################
# load model
##############################

savedir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot"


config = read_config(join(savedir, 'config.py'))

weight_dtype = torch.float32
if config.mixed_precision == "fp16": # accelerator.
    weight_dtype = torch.float16
elif config.mixed_precision == "bf16": # accelerator.
    weight_dtype = torch.bfloat16
    
image_size = config.image_size  # @param [256, 512, 1024]
latent_size = int(image_size) // 8
pred_sigma = getattr(config, 'pred_sigma', True)
learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                'model_max_length': config.model_max_length}
# train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
model = build_model(config.model,
                config.grad_checkpointing,
                config.get('fp32_attention', False),
                input_size=latent_size,
                learn_sigma=learn_sigma,
                pred_sigma=pred_sigma,
                **model_kwargs).train()

transformer = Transformer2DModel(
        sample_size=image_size // 8,
        num_layers=len(model.blocks),
        attention_head_dim=model.blocks[0].hidden_size // model.num_heads,
        in_channels=model.in_channels,
        out_channels=model.out_channels,
        patch_size=model.patch_size,
        attention_bias=True,
        num_attention_heads=model.num_heads,
        cross_attention_dim=model.blocks[0].hidden_size,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
)
# state_dict = state_dict_convert(all_state_dict.pop("state_dict"))
transformer.load_state_dict(state_dict_convert(model.state_dict()))
pipeline = PixArtAlphaPipeline_hookembedding.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    transformer=transformer,
    tokenizer=None,
    text_encoder=None,
    torch_dtype=weight_dtype,
)

ckptdir = join(savedir, "checkpoints")
validation_prompts = config.validation_prompts
prompt_cache_dir = config.prompt_cache_dir
ckpt = torch.load(join(ckptdir, "epoch_4000_step_160000.pth"))
model.load_state_dict(ckpt['state_dict_ema']) # 'state_dict'
pipeline.transformer.load_state_dict(state_dict_convert(ckpt['state_dict_ema'])) # model.state_dict()
# visualize the prompts




def optimize_feature_vector(epoch=30, embedding_when=0.6, embedding_layer=7, label = [1,1,1,-8/5,1,1,-8/5,-8/5,-8/5,-8/5,1,0,0,1,1], target='triangle'):
    layer_name = f'block{embedding_layer:02d}'

    # 0 "triangle is to the upper left of square", 
    # 1 "blue triangle is to the upper left of red square", 
    # 2 "triangle is above and to the right of square", 
    # 3 "blue circle is above and to the right of blue square", 
    # 4 "triangle is to the left of square", 
    # 5 "triangle is to the left of triangle", 
    # 6 "circle is below red square",
    # 7 "red circle is to the left of blue square",
    # 8 "blue square is to the right of red circle",
    # 9 "red circle is above square",
    # 10 "triangle is above red circle",
    # 11 "red is above blue",
    # 12 "red is to the left of red",
    # 13 "blue triangle is above red triangle", 
    # 14 "blue circle is above blue square", 

    # Has triangle
    
    label_tensor = torch.tensor(label, dtype=torch.float32, device='cuda')

    embedding_dim = 768
    embedding_vector = torch.randn(embedding_dim, device='cuda', requires_grad=True)  # Ensure requires_grad=True
    embedding_vector = torch.nn.Parameter(embedding_vector / embedding_vector.norm())  # Convert to a leaf tensor

    optimizer = torch.optim.Adam([embedding_vector], lr=0.05)
    loss_history = []
    # Training loop
    for i in tqdm(range(epoch)):
        optimizer.zero_grad()
        # Run diffusion, get embeddings
        embeddings, images = get_embeddings(pipeline, validation_prompts, prompt_cache_dir,
                                embedding_when=0.6,
                                embedding_layer=embedding_layer,
                                max_length=config.model_max_length, 
                                weight_dtype=weight_dtype)
        # Convert the list of embeddings to a single tensor
        # shape: (prompt, batch, sequence, embedding)
        all_embeddings_tensor = torch.stack([torch.tensor(prompt_embedding[layer_name][0], device='cuda') for prompt_embedding in embeddings], dim=0)
        
        # Project the embeddings to the embedding vector
        # shape: (prompt, batch, sequence)
        projection = all_embeddings_tensor @ embedding_vector

        # Find the max value of the projection in the image dimension
        max_projection_values = torch.max(projection, dim=2).values

        # Compute loss
        contrastive_loss = -(label_tensor.unsqueeze(1) * max_projection_values).mean()
        # L2_loss = l2_lambda*embedding_vector.norm()
        total_loss = contrastive_loss #+ L2_loss

        # Backpropagate the loss
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            embedding_vector /= embedding_vector.norm()

        loss_history.append(total_loss.item())

    plt.plot(loss_history)
    # Add label to the plot
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')

    # Get the embeddings to visualize
    embeddings, images = get_embeddings(pipeline, validation_prompts, prompt_cache_dir,
                            embedding_when=0.6,
                            embedding_layer=embedding_layer,
                            max_length=config.model_max_length, 
                            weight_dtype=weight_dtype)
    all_embeddings_tensor = torch.stack([torch.tensor(prompt_embedding[layer_name][0], device='cuda') for prompt_embedding in embeddings], dim=0)

    # Project the embeddings to the embedding vector
    # shape: (prompt, batch, sequence)
    projection = all_embeddings_tensor @ embedding_vector


    # Plot activation map
    batch_idx = 0

    fig, axes = plt.subplots(5, 6, figsize=(15, 15))  # Adjusted figsize to reduce empty space

    # Determine the min and max values for the color scale
    vmin = projection.min().cpu().detach().numpy()
    vmax = projection.max().cpu().detach().numpy()

    for prompt_idx in range(15):
        row = prompt_idx // 3
        col = (prompt_idx % 3) * 2
        
        # Display the image
        axes[row, col].imshow(images[prompt_idx][batch_idx])
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Image {prompt_idx + 1}')
        
        # Display the corresponding projection
        reshaped_projection = projection[prompt_idx][batch_idx].reshape(8, 8)
        axes[row, col + 1].imshow(reshaped_projection.cpu().detach().numpy(), cmap='hot', vmin=vmin, vmax=vmax)
        axes[row, col + 1].axis('off')
        axes[row, col + 1].set_title(f'Projection {prompt_idx + 1}')

    fig.suptitle(f"Triangle,  embedding_when = {embedding_when},  embedding_layer = {embedding_layer}")

    plt.tight_layout()
    plt.show()
    # Save the plot to the specified path with a file name
    save_path = f'/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/feature_map_pilot/feature_map_{target}_layer{embedding_layer}_when{embedding_when}.png'
    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")


for layer in range(12):
    for embedding_when in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        optimize_feature_vector(epoch=30, embedding_when=embedding_when, embedding_layer=layer)
