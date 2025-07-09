import torch
from typing import Optional

__all__ = [
    'compute_entropy_last_dim',
    'attention_maps_to_spatial_maps',
    'infer_spatial_shape',
    'token_distance_matrix',
    'average_attention_distance',
    'local_attention_score',
    'top_k_attention_score',
    'attention_spatial_variance',
    'attention_spatial_total_variation',
    'plot_attention_layer_head_heatmaps',
    'plot_single_attn_map',
    "visualize_attn_maps",
    "toggle_fused_attn",
]


# Statistics of the attention maps
def compute_entropy_last_dim(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Shannon entropy H(p) = -sum p*log(p) along `dim`, which should sum to 1 along `dim`. 
    treating 0·log(0) = 0 exactly.
    """
    # compute log(p) but set log(0)→0 so that 0·log(0)=0
    log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
    return -(p * log_p).sum(dim)


def attention_maps_to_spatial_maps(attn_maps: torch.Tensor, shape: Optional[tuple] = None) -> torch.Tensor:
    """
    Convert attention maps to spatial maps.
    If shape is not provided, it will be inferred from the number of tokens.
    The last dimension of the attention maps should be the number of tokens.
    """
    if shape is None:
        shape = infer_spatial_shape(attn_maps.shape[-1])
    attn_maps_shape = attn_maps.shape
    return attn_maps.reshape(*attn_maps_shape[:-1], *shape)


def infer_spatial_shape(token_num: int) -> tuple:
    """
    Infer the spatial shape of the attention maps from the number of tokens.
    """
    return ( int(token_num**0.5), int(token_num**0.5))


def token_distance_matrix(shape=(16, 16), dist_type: str = "L2"):
    """
    Compute a matrix of distances between all tokens in a grid of shape `shape`.
    The distance between two tokens is the Euclidean distance between their coordinates.
    """
    x, y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
    pos_vec = torch.stack([x.flatten(), y.flatten()], dim=1).float()
    if dist_type == "L2":
        return torch.cdist(pos_vec, pos_vec, p=2)
    elif dist_type == "L2_squared":
        return torch.cdist(pos_vec, pos_vec, p=2)**2
    elif dist_type == "L1":
        return torch.cdist(pos_vec, pos_vec, p=1)
    else:
        raise ValueError(f"Invalid distance type: {dist_type}")


def average_attention_distance(attn_maps: torch.Tensor, shape: Optional[tuple] = None, dist_type: str = "L2") -> torch.Tensor:
    """
    Average the attention distance of the attention maps.
    """
    if shape is None:
        shape = infer_spatial_shape(attn_maps.shape[-1]) # (H, W)
    # compute the distance matrix of certain type
    dist_matrix = token_distance_matrix(shape, dist_type) # (H*W, H*W)
    # average the attention distance of the attention maps, note we sum over the last dim since attention weight sums to 1
    return (attn_maps * dist_matrix).sum(dim=-1)


def local_attention_score(attn_maps: torch.Tensor, shape: Optional[tuple] = None, dist_type: str = "L2", threshold: float = 2.0) -> torch.Tensor:
    """
    Compute the local attention score of the attention maps.
    """
    if shape is None:
        shape = infer_spatial_shape(attn_maps.shape[-1]) # (H, W)
    dist_matrix = token_distance_matrix(shape, dist_type) # (H*W, H*W)
    # compute the locality mask
    locality_mask = (dist_matrix < threshold).float() # (H*W, H*W)
    # compute the local attention score
    return (attn_maps * locality_mask).sum(dim=-1)


def top_k_attention_score(attn_maps: torch.Tensor, k: int = 10, dim: int = -1) -> torch.Tensor:
    """
    Compute the sum of top k attention score of the attention maps.
    """
    return attn_maps.topk(k, dim=dim).values.sum(dim=dim)


def attention_spatial_variance(attn_maps: torch.Tensor, shape: Optional[tuple] = None) -> torch.Tensor:
    """
    Compute the spatial variance of the attention maps.
    """
    if shape is None:
        shape = infer_spatial_shape(attn_maps.shape[-1]) # (H, W)
    attn_maps_spatial = attention_maps_to_spatial_maps(attn_maps, shape)
    return weighted_variance_2d(attn_maps_spatial)


def attention_spatial_total_variation(attn_maps: torch.Tensor, shape: Optional[tuple] = None) -> torch.Tensor:
    """
    Compute the spatial total variation of the attention maps.
    """
    if shape is None:
        shape = infer_spatial_shape(attn_maps.shape[-1]) # (H, W)
    attn_maps_spatial = attention_maps_to_spatial_maps(attn_maps, shape)
    return tv2d(attn_maps_spatial)


def weighted_variance_2d(A: torch.Tensor) -> torch.Tensor:
    """
    A: (..., H, W), rows sum to 1
    returns: (...) det(cov)^{-1/2}  (a scalar per map)
    """
    H, W = A.shape[-2:]
    ys = torch.arange(H, device=A.device, dtype=A.dtype).view(-1,1)
    xs = torch.arange(W, device=A.device, dtype=A.dtype).view(1,-1)
    # normalize A so it sums to 1 per map
    P = A / (A.sum(dim=(-2,-1), keepdim=True) + 1e-12)
    mu_y = (P * ys).sum(dim=(-2,-1), keepdim=True)
    mu_x = (P * xs).sum(dim=(-2,-1), keepdim=True)
    dy = ys - mu_y; dx = xs - mu_x
    # compute cov matrix entries
    c_yy = (P * dy * dy).sum(dim=(-2,-1))
    c_xx = (P * dx * dx).sum(dim=(-2,-1))
    c_xy = (P * dy * dx).sum(dim=(-2,-1))
    # det of covariance
    det = c_yy * c_xx - c_xy**2
    return det 
    

def clusteriness_2d(A: torch.Tensor) -> torch.Tensor:
    """
    A: (..., H, W), rows sum to 1
    returns: (...) det(cov)^{-1/2}  (a scalar per map)
    """
    return 1.0 / (weighted_variance_2d(A) + 1e-12)


def tv2d(A: torch.Tensor) -> torch.Tensor:
    """
    Total variation of A along its last two dims (spatial dimensions)
    Lower TV ⇒ smoother.
    Returns shape A.shape[:-2].
    """
    dh = (A[..., 1:, :] - A[..., :-1, :]).abs().sum(dim=(-2, -1))
    dw = (A[..., :, 1:] - A[..., :, :-1]).abs().sum(dim=(-2, -1))
    return dh + dw

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def plot_attention_layer_head_heatmaps(score_tensor, title_str, figsize=(12, 8), sample_idx=-1, num_heads=6, share_clim=False, panel_shape=None):
    """
    Plot attention heatmaps for multiple heads.
    
    Args:
        score_tensor: Tensor of shape (layers, steps, samples, heads) or (layers, steps, heads)
        title_str: Title string for the plot
        figsize: Figure size tuple
        num_heads: Number of attention heads to plot
    """
    if panel_shape is None:
        panel_shape = (num_heads // 3, 3)
    
    figh, axs = plt.subplots(panel_shape[0], panel_shape[1], figsize=figsize, sharex=True, sharey=True)
    axs = axs.flatten()
    num_heads = score_tensor.shape[-1]
    if share_clim:
        vmin = score_tensor.min()
        vmax = score_tensor.max()
    else:
        vmin = None
        vmax = None
    for head_idx in range(num_heads):
        if sample_idx is not None:  # (layers, steps, samples, heads)
            if isinstance(sample_idx, slice) or isinstance(sample_idx, list):
                data = score_tensor.cpu()[:, :, sample_idx, head_idx].mean(2)  # Last sample
            else:
                data = score_tensor.cpu()[:, :, sample_idx, head_idx]  # Last sample
        else:  # (layers, steps, heads)
            data = score_tensor.cpu()[:, :, :, head_idx].mean(dim=2)
            
        sns.heatmap(data, cmap="viridis", ax=axs[head_idx], cbar=True, vmin=vmin, vmax=vmax)
        axs[head_idx].set_title(f"Head {head_idx}")
        axs[head_idx].set_xlabel("Sample step")
        axs[head_idx].set_ylabel("Layer")
    
    plt.suptitle(title_str+(f" | sample_id = {sample_idx}" if sample_idx is not None else " | avg over all samples"))
    plt.tight_layout()
    plt.show()
    return figh


def plot_attention_layer_head_time_heatmaps(score_tensor, title_str, figsize=(12, 8), sample_idx=-1, num_steps=14, share_clim=False, panel_shape=None):
    """
    Plot attention heatmaps for multiple heads. Layer by head per panel, each panel is a step.
    
    Args:
        score_tensor: Tensor of shape (layers, steps, samples, heads) or (layers, steps, heads)
        title_str: Title string for the plot
        figsize: Figure size tuple
        num_heads: Number of attention heads to plot
    """
    if panel_shape is None:
        panel_shape = (num_steps // 3, 3)
    num_steps = score_tensor.shape[1]
    figh, axs = plt.subplots(int(np.ceil(num_steps / 3)), 3, figsize=figsize, sharex=True, sharey=True)
    axs = axs.flatten()
    if share_clim:
        vmin = score_tensor.min()
        vmax = score_tensor.max()
    else:
        vmin = None
        vmax = None
    for step_idx in range(num_steps):
        if sample_idx is not None:  # (layers, steps, samples, heads)
            data = score_tensor.cpu()[:, step_idx, sample_idx, :]  # Last sample
        else:  # (layers, steps, heads)
            data = score_tensor.cpu()[:, step_idx, :, :].mean(dim=1)
            
        sns.heatmap(data, cmap="viridis", ax=axs[step_idx], cbar=True, vmin=vmin, vmax=vmax)
        axs[step_idx].set_title(f"Step {step_idx}")
        axs[step_idx].set_xlabel("Head number")
        axs[step_idx].set_ylabel("Layer")
        axs[step_idx].set_aspect("equal")
        plt.axis("tight")
    
    plt.suptitle(title_str+(f" | sample_id = {sample_idx}" if sample_idx is not None else " | avg over all samples"))
    plt.tight_layout()
    plt.show()
    return figh


def plot_single_attn_map(ax, attn_map, token_idx=None, map_shape=(16, 16), use_heatmap=False, cbar=True):
    """Plot a single attention map on the given axes.
    
    Args:
        ax: matplotlib axes to plot on
        attn_map: attention map tensor of shape (num_tokens)
        token_idx: index of token to highlight
        map_shape: shape of the attention map (H,W)
        use_heatmap: whether to use seaborn heatmap instead of imshow
    """
    if use_heatmap:
        sns.heatmap(attn_map.reshape(map_shape), cmap="viridis", ax=ax, cbar=cbar)
    else:
        im = ax.imshow(attn_map.reshape(map_shape), cmap="viridis")
        if cbar:
            cbar = ax.figure.colorbar(im, ax=ax)
    if token_idx is not None:
        H_idx, W_idx = np.unravel_index(token_idx, map_shape)
        if use_heatmap:
            H_idx = H_idx + 0.5
            W_idx = W_idx + 0.5
        ax.scatter(W_idx, H_idx, color="red", marker="x")
    ax.set_aspect("equal")
    return im


def visualize_attn_maps_old(attn_maps_stacked, layer_idx, step_idx, sample_idx, head_idx, token_idx, map_shape=(16, 16), use_heatmap=False):
    """
    Visualize attention maps with different slicing options.
    
    attn_maps_stacked has shape (num_layers, num_steps, num_samples, num_heads, num_tokens, num_tokens)
    """
    n_layers, n_steps, n_samples, n_heads, n_tokens, _ = attn_maps_stacked.shape
    
    # Single head case
    if head_idx is not None:
        fig, ax = plt.subplots()
        attn_map = attn_maps_stacked[layer_idx,step_idx,sample_idx,head_idx,token_idx,:]
        plot_single_attn_map(ax, attn_map, token_idx, map_shape, use_heatmap)
        return fig
        
    # Multiple heads cases
    if layer_idx is not None and step_idx is not None:
        # Plot all heads for given layer and step
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
        axs = axs.flatten()
        for head_idx in range(n_heads):
            attn_map = attn_maps_stacked[layer_idx,step_idx,sample_idx,head_idx,token_idx,:]
            plot_single_attn_map(axs[head_idx], attn_map, token_idx, map_shape, use_heatmap)
            axs[head_idx].set_title(f"Head {head_idx}")
        plt.suptitle(f"Layer{layer_idx}, Step{step_idx}, Sample{sample_idx}, Token{token_idx} | all heads")
            
    elif layer_idx is None and step_idx is not None:
        # Plot all layers x heads for given step
        fig, axs = plt.subplots(n_layers, n_heads, figsize=(15, 30), sharex=True, sharey=True)
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                attn_map = attn_maps_stacked[layer_idx,step_idx,sample_idx,head_idx,token_idx,:]
                plot_single_attn_map(axs[layer_idx,head_idx], attn_map, token_idx, map_shape, use_heatmap)
                axs[layer_idx,head_idx].set_title(f"Lyr{layer_idx}, Hd{head_idx}")
        plt.suptitle(f"Step{step_idx}, Sample{sample_idx}, Token{token_idx} | all layers x heads")
            
    elif step_idx is None and layer_idx is not None:
        # Plot all steps x heads for given layer
        fig, axs = plt.subplots(n_steps, n_heads, figsize=(15, 40), sharex=True, sharey=True)
        for step_idx in range(n_steps):
            for head_idx in range(n_heads):
                attn_map = attn_maps_stacked[layer_idx,step_idx,sample_idx,head_idx,token_idx,:]
                plot_single_attn_map(axs[step_idx,head_idx], attn_map, token_idx, map_shape, use_heatmap)
                axs[step_idx,head_idx].set_title(f"Step{step_idx}, Hd{head_idx}")
        plt.suptitle(f"Layer{layer_idx}, Sample{sample_idx}, Token{token_idx} | all steps x heads")
    
    plt.tight_layout()
    return fig


def visualize_attn_maps(
    attn_maps, 
    layer_idx, step_idx, sample_idx, head_idx, token_idx, 
    row_dim='layer', 
    col_dim='head',
    map_shape=(16,16),
    use_heatmap=False,
    cbar=True,
):
    """
    attn_maps: shape (L, S, N, H, T, T)
    fixed: dict of the two dims you want to hold constant, e.g. {'step':3, 'head':1}
    row_dim, col_dim: the two dims you want to vary along rows and columns of subplots
    """
    # possible dims
    fixed = {}
    if layer_idx is not None:
        fixed['layer'] = layer_idx
    if step_idx is not None:
        fixed['step'] = step_idx
    if sample_idx is not None:
        fixed['sample'] = sample_idx
    if head_idx is not None:
        fixed['head'] = head_idx
    if token_idx is not None:
        fixed['token'] = token_idx
    dims = {
        'layer': attn_maps.shape[0],
        'step' : attn_maps.shape[1],
        'sample': attn_maps.shape[2],
        'head' : attn_maps.shape[3],
        'token': attn_maps.shape[4],
         None: 1 # when some dim is None, it means it's singleton
    }
    non_fixed_dims = set(dims) - set(fixed) - {None}
    assert row_dim in dims and col_dim in dims
    if len(non_fixed_dims) == 0:
        row_dim = None
        col_dim = None
    elif len(non_fixed_dims) == 1:
        col_dim = non_fixed_dims.pop()
        row_dim = None
    elif len(non_fixed_dims) == 2:
        if row_dim is None or col_dim is None:
            row_dim = non_fixed_dims.pop()
            col_dim = non_fixed_dims.pop()
        else:
            assert row_dim in non_fixed_dims and col_dim in non_fixed_dims
    else:
        raise ValueError(f"non_fixed_dims must have length 0, 1, or 2, got {non_fixed_dims}")
    # sanity check
    row_vals = range(dims[row_dim])
    col_vals = range(dims[col_dim])

    fig, axs = plt.subplots(
        len(row_vals), len(col_vals),
        figsize=(3*len(col_vals), 3*len(row_vals)),
        sharex=True, sharey=True, squeeze=False
    )

    # make sure axs is 2D
    axs = np.atleast_2d(axs)

    for i, r in enumerate(row_vals):
        for j, c in enumerate(col_vals):
            idxs = {
                row_dim: r,
                col_dim: c,
                **fixed
            }
            # pull out the attention map slice
            attn_map = attn_maps[
                idxs['layer'],
                idxs['step'],
                sample_idx,
                idxs['head'],
                token_idx,
                :
            ]
            ax = axs[i, j]
            plot_single_attn_map(
                ax, attn_map, token_idx,
                map_shape, use_heatmap, cbar
            )
            ax.set_title(", ".join([f"{k}={v}" for k,v in [(row_dim, r), (col_dim, c)] if k is not None]))
            # ax.set_title(f"{row_dim}={r}, {col_dim}={c}")
    plt.suptitle(", ".join([f"{k}={v}" for k, v in fixed.items()]))
    # plt.tight_layout()
    return fig


def plot_layer_head_score_summary(template_similarity_scores, template_type, step_sum_type="max", step_id=None):
    n_samples = template_similarity_scores.shape[1] // 2
    cond_slice = slice(n_samples, n_samples * 2)
    uncond_slice = slice(0, n_samples)
    if step_sum_type == "max":
        temporal_summary = template_similarity_scores.max(dim=1).values
    elif step_sum_type == "mean":
        temporal_summary = template_similarity_scores.mean(dim=1)
    elif step_sum_type == "index" and step_id is not None:
        temporal_summary = template_similarity_scores[:, step_id, :, :]
    else:
        raise ValueError(f"Invalid step_sum_type: {step_sum_type}")
    layer_head_summary = temporal_summary[:, cond_slice, :].mean(dim=-2).numpy() # average over samples 
    fig = plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    sns.heatmap(layer_head_summary)
    plt.title("Cond pass")
    plt.axis('image')
    plt.ylabel("Layer")
    plt.xlabel("Head")
    plt.subplot(1, 2, 2)
    layer_head_summary = temporal_summary[:, uncond_slice, :].mean(dim=-2).numpy() # average over samples 
    sns.heatmap(layer_head_summary)
    plt.title("Uncond pass")
    plt.axis('image')
    plt.ylabel("Layer")
    plt.xlabel("Head")
    plt.suptitle(f"Attention template similarity Layer-Head Summary | {step_sum_type} over steps | {template_type} ")
    plt.show()
    return fig


def toggle_fused_attn(model, fused_attn=True):
    for block in model.blocks:
        block.attn.fused_attn = fused_attn
    print(f"Fused attn turned {fused_attn}")
    return model

