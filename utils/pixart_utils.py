import os
import torch
from diffusers import Transformer2DModel

config_dict = {
    "DiT_S-2": {"num_layers": 12, "attention_head_dim": 64, "in_channels": 4, "out_channels": 8, "patch_size": 2, "num_attention_heads": 6, "cross_attention_dim": 384},
    "DiT_B-2": {"num_layers": 12, "attention_head_dim": 64, "in_channels": 4, "out_channels": 8, "patch_size": 2, "num_attention_heads": 12, "cross_attention_dim": 768},
    "DiT_L-2": {"num_layers": 24, "attention_head_dim": 64, "in_channels": 4, "out_channels": 8, "patch_size": 2, "num_attention_heads": 12, "cross_attention_dim": 768},
    "DiT_XL-2": {"num_layers": 28, "attention_head_dim": 72, "in_channels": 4, "out_channels": 8, "patch_size": 2, "num_attention_heads": 16, "cross_attention_dim": 1152},
}

def state_dict_convert(state_dict, num_layers=28, image_size=128, multi_scale_train=False):
    converted_state_dict = {}

    # Patch embeddings.
    converted_state_dict["pos_embed.proj.weight"] = state_dict.pop("x_embedder.proj.weight")
    converted_state_dict["pos_embed.proj.bias"] = state_dict.pop("x_embedder.proj.bias")

    # Caption projection.
    converted_state_dict["caption_projection.linear_1.weight"] = state_dict.pop("y_embedder.y_proj.fc1.weight")
    converted_state_dict["caption_projection.linear_1.bias"] = state_dict.pop("y_embedder.y_proj.fc1.bias")
    converted_state_dict["caption_projection.linear_2.weight"] = state_dict.pop("y_embedder.y_proj.fc2.weight")
    converted_state_dict["caption_projection.linear_2.bias"] = state_dict.pop("y_embedder.y_proj.fc2.bias")

    # AdaLN-single LN
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_1.weight"] = state_dict.pop(
        "t_embedder.mlp.0.weight"
    )
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_1.bias"] = state_dict.pop("t_embedder.mlp.0.bias")
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_2.weight"] = state_dict.pop(
        "t_embedder.mlp.2.weight"
    )
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_2.bias"] = state_dict.pop("t_embedder.mlp.2.bias")

    if image_size == 1024 and multi_scale_train:
        # Resolution.
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_1.weight"] = state_dict.pop(
            "csize_embedder.mlp.0.weight"
        )
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_1.bias"] = state_dict.pop(
            "csize_embedder.mlp.0.bias"
        )
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_2.weight"] = state_dict.pop(
            "csize_embedder.mlp.2.weight"
        )
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_2.bias"] = state_dict.pop(
            "csize_embedder.mlp.2.bias"
        )
        # Aspect ratio.
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_1.weight"] = state_dict.pop(
            "ar_embedder.mlp.0.weight"
        )
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_1.bias"] = state_dict.pop(
            "ar_embedder.mlp.0.bias"
        )
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_2.weight"] = state_dict.pop(
            "ar_embedder.mlp.2.weight"
        )
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_2.bias"] = state_dict.pop(
            "ar_embedder.mlp.2.bias"
        )
    # Shared norm.
    converted_state_dict["adaln_single.linear.weight"] = state_dict.pop("t_block.1.weight")
    converted_state_dict["adaln_single.linear.bias"] = state_dict.pop("t_block.1.bias")

    for depth in range(num_layers):
        if f"blocks.{depth}.scale_shift_table" not in state_dict:
            break
        # Transformer blocks.
        converted_state_dict[f"transformer_blocks.{depth}.scale_shift_table"] = state_dict.pop(
            f"blocks.{depth}.scale_shift_table"
        )

        # Attention is all you need 🤘

        # Self attention.
        q, k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.attn.qkv.weight"), 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict.pop(f"blocks.{depth}.attn.qkv.bias"), 3, dim=0)
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_q.bias"] = q_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_k.bias"] = k_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_v.weight"] = v
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_v.bias"] = v_bias
        # Projection.
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.bias"
        )

        # Feed-forward.
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc1.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc1.bias"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.2.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc2.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.2.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc2.bias"
        )

        # Cross-attention.
        q = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.weight")
        q_bias = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.bias")
        k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.weight"), 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.bias"), 2, dim=0)

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.bias"] = q_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.bias"] = k_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.weight"] = v
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.bias"] = v_bias

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.bias"
        )

    # Final block.
    converted_state_dict["proj_out.weight"] = state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = state_dict.pop("final_layer.linear.bias")
    converted_state_dict["scale_shift_table"] = state_dict.pop("final_layer.scale_shift_table")
    return converted_state_dict

def convert_checkpoint_to_diffusers_transformer(orig_ckpt_path, dump_path=None, model_name="DiT_XL-2", image_size=128, multi_scale_train=False, config=None,
                                                state_dict=None):
    if model_name not in config_dict and config is None:
        raise ValueError(f"Model name {model_name} not supported")
    if config is None:
        config = config_dict[model_name]
    else:
        config = config
    if state_dict is None:
        all_state_dict = torch.load(orig_ckpt_path, map_location='cpu')
        state_dict = all_state_dict.pop("state_dict")
    else:
        state_dict = state_dict
    
    converted_state_dict = {}

    # Patch embeddings.
    converted_state_dict["pos_embed.proj.weight"] = state_dict.pop("x_embedder.proj.weight")
    converted_state_dict["pos_embed.proj.bias"] = state_dict.pop("x_embedder.proj.bias")

    # Caption projection.
    converted_state_dict["caption_projection.linear_1.weight"] = state_dict.pop("y_embedder.y_proj.fc1.weight")
    converted_state_dict["caption_projection.linear_1.bias"] = state_dict.pop("y_embedder.y_proj.fc1.bias")
    converted_state_dict["caption_projection.linear_2.weight"] = state_dict.pop("y_embedder.y_proj.fc2.weight")
    converted_state_dict["caption_projection.linear_2.bias"] = state_dict.pop("y_embedder.y_proj.fc2.bias")

    # AdaLN-single LN
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_1.weight"] = state_dict.pop(
        "t_embedder.mlp.0.weight"
    )
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_1.bias"] = state_dict.pop("t_embedder.mlp.0.bias")
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_2.weight"] = state_dict.pop(
        "t_embedder.mlp.2.weight"
    )
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_2.bias"] = state_dict.pop("t_embedder.mlp.2.bias")

    if image_size == 1024 and multi_scale_train:
        # Resolution.
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_1.weight"] = state_dict.pop(
            "csize_embedder.mlp.0.weight"
        )
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_1.bias"] = state_dict.pop(
            "csize_embedder.mlp.0.bias"
        )
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_2.weight"] = state_dict.pop(
            "csize_embedder.mlp.2.weight"
        )
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_2.bias"] = state_dict.pop(
            "csize_embedder.mlp.2.bias"
        )
        # Aspect ratio.
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_1.weight"] = state_dict.pop(
            "ar_embedder.mlp.0.weight"
        )
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_1.bias"] = state_dict.pop(
            "ar_embedder.mlp.0.bias"
        )
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_2.weight"] = state_dict.pop(
            "ar_embedder.mlp.2.weight"
        )
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_2.bias"] = state_dict.pop(
            "ar_embedder.mlp.2.bias"
        )
    # Shared norm.
    converted_state_dict["adaln_single.linear.weight"] = state_dict.pop("t_block.1.weight")
    converted_state_dict["adaln_single.linear.bias"] = state_dict.pop("t_block.1.bias")

    for depth in range(config["num_layers"]):
        # Transformer blocks.
        converted_state_dict[f"transformer_blocks.{depth}.scale_shift_table"] = state_dict.pop(
            f"blocks.{depth}.scale_shift_table"
        )

        # Attention is all you need 🤘

        # Self attention.
        q, k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.attn.qkv.weight"), 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict.pop(f"blocks.{depth}.attn.qkv.bias"), 3, dim=0)
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_q.bias"] = q_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_k.bias"] = k_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_v.weight"] = v
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_v.bias"] = v_bias
        # Projection.
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.bias"
        )

        # Feed-forward.
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc1.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc1.bias"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.2.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc2.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.2.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc2.bias"
        )

        # Cross-attention.
        q = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.weight")
        q_bias = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.bias")
        k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.weight"), 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.bias"), 2, dim=0)

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.bias"] = q_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.bias"] = k_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.weight"] = v
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.bias"] = v_bias

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.bias"
        )

    # Final block.
    converted_state_dict["proj_out.weight"] = state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = state_dict.pop("final_layer.linear.bias")
    converted_state_dict["scale_shift_table"] = state_dict.pop("final_layer.scale_shift_table")

    # DiT XL/2
    transformer = Transformer2DModel(
        sample_size=image_size // 8,
        num_layers=config["num_layers"],
        attention_head_dim=config["attention_head_dim"],
        in_channels=4,
        out_channels=8,
        patch_size=config["patch_size"],
        attention_bias=True,
        num_attention_heads=config["num_attention_heads"],
        cross_attention_dim=config["cross_attention_dim"],
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
    )
    transformer.load_state_dict(converted_state_dict, strict=True)

    assert transformer.pos_embed.pos_embed is not None
    state_dict.pop("pos_embed")
    state_dict.pop("y_embedder.y_embedding")
    assert len(state_dict) == 0, f"State dict is not empty, {state_dict.keys()}"

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")
    if dump_path:
        transformer.save_pretrained(os.path.join(dump_path, "transformer"))
    return transformer


def construct_pixart_transformer_from_config(config, ):
    """
    config: config object
    transformer_params: dict, if None, use PixArt_model_configs[config.model]
        should contain fields: depth, hidden_size, patch_size, num_heads
    return: transformer object
    """
    import sys
    sys.path.append("/n/home12/binxuwang/Github/DiffusionObjectRelation/PixArt-alpha")
    from diffusion.model.builder import build_model
    
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
                    **model_kwargs) #.train()
    return model


PixArt_model_configs = {
    'PixArt_XL_2': {'depth': 28, 'hidden_size': 1152, 'patch_size': 2, 'num_heads': 16},
    'PixArt_L_2': {'depth': 24, 'hidden_size': 768, 'patch_size': 2, 'num_heads': 12}, 
    'PixArt_B_2': {'depth': 12, 'hidden_size': 768, 'patch_size': 2, 'num_heads': 12},
    'PixArt_S_2': {'depth': 12, 'hidden_size': 384, 'patch_size': 2, 'num_heads': 6},
    'PixArt_mini_2':  {'depth': 6, 'hidden_size': 384, 'patch_size': 2, 'num_heads': 6},
    'PixArt_micro_2': {'depth': 6, 'hidden_size': 192, 'patch_size': 2, 'num_heads': 3},
    'PixArt_nano_2':  {'depth': 3, 'hidden_size': 192, 'patch_size': 2, 'num_heads': 3}
}


def construct_diffuser_transformer_from_config(config, transformer_params=None):
    """
    config: config object
    transformer_params: dict, if None, use PixArt_model_configs[config.model]
        should contain fields: depth, hidden_size, patch_size, num_heads
    return: transformer object
    """
    if transformer_params is None:
        transformer_params = PixArt_model_configs[config.model]
    
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

    transformer = Transformer2DModel(
            sample_size=image_size // 8,
            num_layers=transformer_params['depth'],#len(model.blocks),
            attention_head_dim=transformer_params['hidden_size'] // transformer_params['num_heads'],
            in_channels=4,
            out_channels=8 if learn_sigma else 4,
            patch_size=transformer_params['patch_size'],
            attention_bias=True,
            num_attention_heads=transformer_params['num_heads'],
            cross_attention_dim=transformer_params['hidden_size'],
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
            caption_channels=config.caption_channels,
    )
    return transformer



from diffusers.pipelines.pixart_alpha import PixArtAlphaPipeline
def construct_diffuser_pipeline_from_config(config, pipeline_class=PixArtAlphaPipeline):
    """
    config: config object
    return: diffuser pipeline object
    """
    transformer = construct_diffuser_transformer_from_config(config)
    
    weight_dtype = torch.float32
    if config.mixed_precision == "fp16": # accelerator.
        weight_dtype = torch.float16
    elif config.mixed_precision == "bf16": # accelerator.
        weight_dtype = torch.bfloat16
    
    pipeline = pipeline_class.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        transformer=transformer,
        tokenizer=None,
        text_encoder=None,
        torch_dtype=weight_dtype,
    )
    return pipeline