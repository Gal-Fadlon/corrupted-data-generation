import os
import logging
import torch
import numpy as np

def get_random_mask(image_shape, survival_probability, mask_full_rgb=False, same_for_all_batch=False, device='cuda', seed=None):
    """
    Generate random corruption mask for Ambient Diffusion sampling.
    
    Args:
        image_shape: (batch_size, num_channels, height, width)
        survival_probability: Probability of a pixel being visible (1 = keep, 0 = corrupt)
        mask_full_rgb: If True, apply same mask to all channels
        same_for_all_batch: If True, apply same mask to all batch samples
        device: Device to create mask on
        seed: Random seed for reproducibility
    
    Returns:
        corruption_mask: Binary tensor (1 = visible, 0 = corrupted)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if same_for_all_batch:
        # Generate one mask and repeat for all batch samples
        corruption_mask = np.random.binomial(1, survival_probability, size=image_shape[1:]).astype(np.float32)
        corruption_mask = torch.tensor(corruption_mask, device=device, dtype=torch.float32).repeat([image_shape[0], 1, 1, 1])
    else:
        # Generate independent mask for each batch sample
        corruption_mask = np.random.binomial(1, survival_probability, size=image_shape).astype(np.float32)
        corruption_mask = torch.tensor(corruption_mask, device=device, dtype=torch.float32)
    
    if mask_full_rgb:
        # Apply same mask to all RGB channels
        corruption_mask = corruption_mask[:, 0]
        corruption_mask = corruption_mask.repeat([3, 1, 1, 1]).transpose(1, 0)
    
    return corruption_mask

def log_config_and_tags(args, logger, name):
    logger.log_name_params('config/hyperparameters', vars(args))
    logger.log_name_params('config/name', name)
    logger.add_tags(args.tags)
    logger.add_tags([args.dataset])


def create_model_name_and_dir(args):
    name = (f'conditional-'
            f'bs={args.batch_size}-'
            f'-lr={args.learning_rate:.4f}-'
            f'ch_mult={args.ch_mult}-'
            f'attn_res={args.attn_resolution}-'
            f'unet_ch={args.unet_channels}'
            )

    assert (args.delay is not None and args.embedding is not None)
    name += f'-delay={args.delay}-{args.embedding}'
    args.log_dir = '%s/%s/%s' % (args.log_dir, args.dataset, name)
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    return name

def restore_state(args, state,ema_model=None):
    logging.info("restoring checkpoint from: {}".format(args.log_dir))
    restore_checkpoint(args.log_dir, state, ema_model = ema_model)
    init_epoch = state['epoch']
    return init_epoch


def print_model_params(logger, model):
    params_num = sum(param.numel() for param in model.parameters())
    logging.info("number of model parameters: {}".format(params_num))
    logger.log_name_params('config/params_num', params_num)


def create_further_corruption(corruption_matrix, delta_probability, device='cuda'):
    """
    Create further corruption matrix (Ã) from original corruption matrix (A).
    
    This implements the Ambient Diffusion strategy where Ã = A ⊙ B,
    where B is an additional corruption applied to A.
    
    Args:
        corruption_matrix: [B, C, H, W] tensor with 1=observed, 0=missing
        delta_probability: Probability of further corrupting a visible pixel (δ)
        device: Device to create tensors on
    
    Returns:
        hat_corruption_matrix: [B, C, H, W] tensor with Ã = A ⊙ B
        
    Example:
        If corruption_matrix has 60% visible pixels and delta_probability=0.1,
        hat_corruption_matrix will have ~54% visible pixels (0.6 * 0.9).
    """
    import torch
    
    # Create extra corruption mask B: each pixel survives with prob (1-δ)
    extra_mask = torch.rand_like(corruption_matrix, device=device)
    extra_mask = (extra_mask > delta_probability).float()
    
    # Ã = A ⊙ B (element-wise product)
    # If A[i] = 0 (already missing) → Ã[i] = 0
    # If A[i] = 1 (visible) → Ã[i] = B[i] (survives with prob 1-δ)
    hat_corruption_matrix = corruption_matrix * extra_mask
    
    return hat_corruption_matrix

