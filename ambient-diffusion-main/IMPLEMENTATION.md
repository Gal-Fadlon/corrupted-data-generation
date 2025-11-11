# Ambient Diffusion - Implementation Documentation

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts from Paper](#core-concepts-from-paper)
3. [Project Structure](#project-structure)
4. [Training Implementation](#training-implementation)
5. [Corruption Operators](#corruption-operators)
6. [Loss Functions](#loss-functions)
7. [Network Architecture](#network-architecture)
8. [Sampling Implementation](#sampling-implementation)
9. [Foundation Model Finetuning](#foundation-model-finetuning)
10. [Inverse Problems](#inverse-problems)
11. [Evaluation](#evaluation)
12. [Paper-to-Code Mapping](#paper-to-code-mapping)

---

## Overview

This document provides a comprehensive implementation analysis of the **Ambient Diffusion** codebase, mapping each theoretical component from the paper to its concrete implementation in code.

**Paper Reference**: "Ambient Diffusion: Learning Clean Distributions from Corrupted Data"

**Core Innovation**: Training diffusion models on corrupted data by introducing additional corruption during training, forcing the model to learn the full clean distribution without ever seeing clean samples.

---

## Core Concepts from Paper

### Mathematical Foundation

From the paper (Section 2.1 and Equation 100):

**Training Objective**:
```
J^{corr}(θ) = (1/2) E_{(x₀, x_t, A, Ã)} ||A(h_θ(Ã, Ã x_t, t) - x₀)||²
```

Where:
- `x₀`: Clean image (never observed during training)
- `A`: Original corruption matrix
- `Ã = BA`: Further corrupted matrix (B is additional corruption)
- `x_t = x₀ + σ_t η`: Noisy clean image at timestep t
- `h_θ`: Neural network predicting clean image

**Key Insight**: The model receives `Ã x_t` (further corrupted noisy image) and must predict `x₀` such that the loss is computed on pixels visible in `A` (but not necessarily in `Ã`).

---

## Project Structure

```
ambient-diffusion-main/
├── train.py                      # Main training script for from-scratch training
├── train_text_to_image.py       # Foundation model (IF) finetuning
├── eval.py                      # Generation and FID evaluation
├── solve_inverse_problems.py    # Solving inverse problems with trained models
├── if_inference.py              # IF model inference script
├── pipeline_if.py               # Custom IF pipeline with Ambient Diffusion
├── scheduling_ddpm.py           # Modified DDPM scheduler
├── find_dataset_neighbors.py    # Memorization analysis using DINO features
├── dataset_tool.py              # Dataset preparation utilities
│
├── torch_utils/
│   ├── ambient_diffusion.py     # Core corruption operators (A and Ã)
│   ├── distributed.py           # Distributed training utilities
│   ├── misc.py                  # Helper functions
│   ├── persistence.py           # Model serialization
│   └── training_stats.py        # Training statistics logging
│
├── training/
│   ├── dataset.py               # Dataset loader with corruption
│   ├── loss.py                  # AmbientLoss implementation
│   ├── networks.py              # Network architectures (SongUNet, DhariwalUNet)
│   ├── training_loop.py         # Main training loop
│   ├── augment.py               # Data augmentation pipeline
│   └── samplers.py              # Sampling algorithms (placeholder)
│
└── dnnlib/
    ├── util.py                  # Utility functions
    └── __init__.py
```

---

## Training Implementation

### 1. Main Training Script (`train.py`)

#### Command Line Interface

```python
@click.command()
@click.option('--corruption_probability', default=0.4)  # p: probability of corrupting pixel
@click.option('--delta_probability', default=0.1)       # δ: probability of further corruption
@click.option('--mask_full_rgb', default=False)         # Whether to mask all RGB channels together
@click.option('--corruption_pattern', default='dust')   # 'dust', 'box', 'fixed_box', 'keep_patch'
@click.option('--precond', default='ambient')           # Use 'ambient' for Ambient Diffusion
```

**Paper Connection**: 
- `corruption_probability` = `p` from paper (Section 5.1)
- `delta_probability` = `δ` from paper (Section 5.1)
- These control the severity of corruption `A` and further corruption `Ã = BA`

#### Configuration Setup (Lines 118-196)

```python
# Dataset configuration with corruption
c.dataset_kwargs = dnnlib.EasyDict(
    class_name='training.dataset.ImageFolderDataset',
    path=opts.data,
    corruption_probability=opts.corruption_probability,
    delta_probability=opts.delta_probability,
    mask_full_rgb=opts.mask_full_rgb,
    corruption_pattern=opts.corruption_pattern
)

# Loss configuration for Ambient Diffusion
if opts.precond == 'ambient':
    c.network_kwargs.class_name = 'training.networks.EDMPrecond'
    c.loss_kwargs.class_name = 'training.loss.AmbientLoss'
    c.loss_kwargs.norm = opts.norm  # L1 or L2 norm
```

**Paper Connection**: Section 5.1 - "We use the EDM codebase to train our models. We replace convolutions with Gated Convolutions which are known to perform better for inpainting-type problems."

### 2. Training Loop (`training/training_loop.py`)

#### Core Training Loop (Lines 131-175)

```python
while True:
    optimizer.zero_grad(set_to_none=True)
    
    for round_idx in range(num_accumulation_rounds):
        # Load batch with corruption matrices
        images, labels, corruption_matrix, hat_corruption_matrix = next(dataset_iterator)
        
        # Compute losses: train_loss, val_loss, test_loss
        train_loss, val_loss, test_loss = loss_fn(
            net=ddp, 
            images=images,          # Clean images x₀
            labels=labels,
            augment_pipe=augment_pipe,
            corruption_matrix=corruption_matrix,      # A
            hat_corruption_matrix=hat_corruption_matrix  # Ã
        )
        
        # Use validation loss for backpropagation
        loss = val_loss
        scalar_loss = loss.sum().mul(loss_scaling / batch_gpu_total)
        scalar_loss.backward()
```

**Implementation Details**:

1. **Three Loss Types**:
   - `train_loss`: Loss on pixels visible in `Ã` (further corrupted)
   - `val_loss`: Loss on pixels visible in `A` (originally corrupted) 
   - `test_loss`: Loss on ALL pixels (for evaluation only)

2. **Why use `val_loss`?**: This corresponds to the objective in Equation 100 of the paper, which evaluates on pixels in `A`.

#### Network Input Format (Lines 77-78)

```python
# Network expects concatenated input [image, mask]
interface_kwargs = dict(
    img_resolution=dataset_obj.resolution,
    label_dim=dataset_obj.label_dim,
    img_channels=2 * dataset_obj.num_channels  # 6 for RGB (3 + 3)
)
```

The network receives:
- 3 channels: `Ã x_t` (further corrupted noisy image)
- 3 channels: `Ã` (the corruption mask itself)

**Paper Connection**: Section 5.1 - "To use the mask Ã as an additional input to the model, we simply concatenate it with the image x."

---

## Corruption Operators

### Core Implementation (`torch_utils/ambient_diffusion.py`)

This file implements the corruption matrices `A` and `Ã` from the paper.

#### 1. Random Inpainting (`get_random_mask`)

**Lines 11-24**:

```python
def get_random_mask(image_shape, survival_probability, 
                    mask_full_rgb=False, same_for_all_batch=False, 
                    device='cuda', seed=None):
    """
    Creates binary mask for random inpainting.
    
    Args:
        survival_probability: 1 - p (probability pixel survives)
        mask_full_rgb: If True, same mask for all RGB channels
    
    Returns:
        Binary mask where 1 = visible, 0 = corrupted
    """
    corruption_mask = np.random.binomial(1, survival_probability, size=image_shape)
    corruption_mask = torch.tensor(corruption_mask, device=device)
    
    if mask_full_rgb:
        # Apply same mask to R, G, B channels
        corruption_mask = corruption_mask[:, 0].repeat([3, 1, 1, 1]).transpose(1, 0)
    
    return corruption_mask
```

**Paper Connection**: Section 5.1 - "The corruption model we use for these experiments is random inpainting: we form our dataset by deleting each pixel with probability p."

#### 2. Box Masking (`get_box_mask`)

**Lines 27-66**:

```python
def get_box_mask(image_shape, survival_probability, 
                 same_for_all_batch=False, device='cuda'):
    """
    Creates mask with random box corruption.
    
    Box size = (1 - survival_probability) * image_dimension
    Box location is random for each image in batch.
    """
    batch_size, num_channels, height, width = image_shape
    mask = torch.zeros(image_shape, device=device)
    
    # Random box location
    box_start_row = torch.randint(0, height, (batch_size, 1, 1), device=device)
    box_start_col = torch.randint(0, width, (batch_size, 1, 1), device=device)
    
    # Box dimensions
    box_height = torch.ceil(torch.tensor((1 - survival_probability) * height)).int()
    box_width = torch.ceil(torch.tensor((1 - survival_probability) * width)).int()
    
    # Create mask (vectorized for efficiency)
    rows = torch.arange(height, device=device).view(1, 1, -1, 1).expand_as(mask)
    cols = torch.arange(width, device=device).view(1, 1, 1, -1).expand_as(mask)
    
    inside_box = ((rows >= box_start_row_expanded) & 
                  (rows < box_start_row_expanded + box_height) &
                  (cols >= box_start_col_expanded) & 
                  (cols < box_start_col_expanded + box_width))
    
    mask[inside_box] = 1.0
    return 1 - mask  # Invert: 1 = visible, 0 = corrupted
```

**Paper Connection**: Used for MRI experiments in the paper - "For all our experiments, we use Deepfloyd's IF model, which is one of the most powerful open-source diffusion generative models available."

#### 3. Forward Operator Classes

**`MaskingForwardOperator` (Lines 171-208)**:

```python
class MaskingForwardOperator(ForwardOperator):
    def __init__(self, corruption_probability, delta_probability, mask_full_rgb=True):
        self.corruption_probability = corruption_probability  # p
        self.delta_probability = delta_probability            # δ
        self.mask_full_rgb = mask_full_rgb
    
    def corrupt(self, x, mask=None):
        """Apply corruption A to clean image x"""
        if mask is None:
            mask = get_random_mask(x.shape, 1 - self.corruption_probability, 
                                  mask_full_rgb=self.mask_full_rgb, device=x.device)
        return x * mask, mask
    
    def hat_corrupt(self, x, mask=None, hat_mask=None):
        """Apply further corruption Ã = BA to already corrupted image"""
        if mask is None:
            _, mask = self.corrupt(x)
        
        if hat_mask is None:
            # Create additional corruption B
            hat_mask = get_random_mask(x.shape, 1 - self.delta_probability,
                                       mask_full_rgb=self.mask_full_rgb, device=x.device)
        
        # Ã = B * A (element-wise multiplication)
        hat_mask = mask * hat_mask
        return x * hat_mask, hat_mask
```

**Paper Connection**: 
- Section 5.1 - "To create the matrix Ã, we further delete each row of A with probability δ -- this removes an additional δ-fraction of the surviving pixels."
- This is the core implementation of the "further corruption" idea!

#### 4. Other Operators

**`BoxMaskingForwardOperator`** (Lines 212-248): Similar to masking but with box corruption

**`CompressedSensingOperator`** (Lines 278-317): Implements random Gaussian measurements
- Creates random measurement matrix `A ∈ ℝ^{m×n}` where m < n
- `hat_corrupt` removes one row and adds a new random row

**`AveragingForwardOperator`** (Lines 250-276): Implements downsampling corruption

---

## Loss Functions

### Implementation (`training/loss.py`)

#### AmbientLoss Class (Lines 88-120)

```python
@persistence.persistent_class
class AmbientLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, norm=2):
        """
        EDM-style loss adapted for Ambient Diffusion.
        
        Args:
            P_mean, P_std: Noise schedule parameters (log-normal)
            sigma_data: Data standard deviation
            norm: 1 for L1, 2 for L2 loss
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.norm = norm
    
    def __call__(self, net, images, corruption_matrix, hat_corruption_matrix, 
                 labels=None, augment_pipe=None):
        # 1. Sample noise level σ from log-normal distribution
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        # 2. Compute loss weighting (from EDM paper)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        # 3. Apply data augmentation (if enabled)
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        # 4. Add noise: x_t = y + σ * η
        n = torch.randn_like(y) * sigma
        
        # 5. Apply FURTHER corruption Ã
        masked_image = hat_corruption_matrix * (y + n)  # Ã * x_t
        noisy_image = masked_image
        
        # 6. Network input: concatenate [Ã * x_t, Ã]
        cat_input = torch.cat([noisy_image, hat_corruption_matrix], axis=1)
        
        # 7. Network prediction
        D_yn = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :3]
        
        # 8. Compute three losses
        if self.norm == 2:
            # Training loss: evaluated on Ã (further corrupted pixels)
            train_loss = weight * ((hat_corruption_matrix * (D_yn - y)) ** 2)
            
            # Validation loss: evaluated on A (originally corrupted pixels)
            val_loss = weight * ((corruption_matrix * (D_yn - y)) ** 2)
            
            # Test loss: evaluated on ALL pixels
            test_loss = weight * ((D_yn - y) ** 2)
        elif self.norm == 1:
            # Same but with L1 norm
            train_loss = weight * (hat_corruption_matrix * torch.abs(D_yn - y))
            val_loss = weight * (corruption_matrix * torch.abs(D_yn - y))
            test_loss = weight * torch.abs(D_yn - y)
        
        return train_loss, val_loss, test_loss
```

**Critical Implementation Details**:

1. **Three-Loss Design**:
   - `train_loss`: Sanity check that model learns on observed pixels
   - `val_loss`: **THIS IS THE ACTUAL TRAINING OBJECTIVE** from Equation 100
   - `test_loss`: Evaluation metric (model never optimized for this)

2. **Why `val_loss` for training?**: Looking at `training_loop.py` line 152:
   ```python
   loss = val_loss  # Use validation loss for backpropagation
   ```
   This is correct! The paper optimizes for pixels in `A`, not `Ã`.

3. **Noise Schedule**: Uses EDM's log-normal schedule for σ, which has been shown to work well for diffusion models.

**Paper Connection**:
- Equation 100: `J^{corr}(θ) = (1/2) E_{(x₀, x_t, A, Ã)} ||A(h_θ(Ã, Ã x_t, t) - x₀)||²`
- Implementation matches exactly: `corruption_matrix * (D_yn - y)` where:
  - `corruption_matrix` = `A`
  - `D_yn` = `h_θ(Ã, Ã x_t, σ)`
  - `y` = `x₀`

---

## Network Architecture

### Implementation (`training/networks.py`)

#### 1. EDM Preconditioning (`EDMPrecond`, Lines 664-705)

```python
@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self, img_resolution, img_channels, label_dim=0, 
                 sigma_min=0, sigma_max=float('inf'), sigma_data=0.5,
                 model_type='DhariwalUNet', gated=True, **model_kwargs):
        super().__init__()
        self.sigma_data = sigma_data
        
        # Underlying U-Net architecture
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,      # 6 for RGB with mask
            out_channels=img_channels//2,  # 3 for RGB
            label_dim=label_dim,
            gated=gated,
            **model_kwargs
        )
    
    def forward(self, x, sigma, class_labels=None, **model_kwargs):
        """
        EDM preconditioning wrapper.
        
        Maps (x_t, σ) to denoised prediction using optimal scaling.
        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        # Preconditioning scales (from EDM paper)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        # Network prediction
        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), 
                         class_labels=class_labels, **model_kwargs)
        
        # Combine skip connection and network output
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x
```

**Paper Connection**: Section 5.1 - "We use the EDM codebase to train our models."

The preconditioning ensures optimal signal-to-noise ratios at all noise levels.

#### 2. Gated Convolutions (`Conv2d`, Lines 48-116)

```python
@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, gated=True, **kwargs):
        super().__init__()
        self.gated = gated
        
        # Standard convolution weights
        self.weight = torch.nn.Parameter(
            weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
        )
        
        if gated:
            # Additional gating weights
            self.gate_weight = torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
            )
    
    def forward(self, x):
        w = self.weight.to(x.dtype)
        
        if self.gated:
            # Gated convolution: output = conv(x) * sigmoid(gate_conv(x))
            gw = self.gate_weight.to(x.dtype)
            gate_output = torch.sigmoid(torch.nn.functional.conv2d(x, gw, padding=w_pad))
            x = torch.nn.functional.conv2d(x, w, padding=w_pad)
            x = torch.mul(x, gate_output)
        else:
            # Standard convolution
            x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        
        return x
```

**Why Gated Convolutions?**
- Better for inpainting tasks
- Learns to suppress information from corrupted regions
- Used in DeepFill and partial convolutions literature

**Paper Connection**: Section 5.1 - "We replace convolutions with Gated Convolutions which are known to perform better for inpainting-type problems."

#### 3. SongUNet Architecture (Lines 256-391)

This is a U-Net with:
- **Encoder**: Downsamples image through multiple resolutions
- **Decoder**: Upsamples with skip connections
- **Self-attention**: At specified resolutions (typically 16×16)
- **Embedding**: Sinusoidal positional encoding for noise level σ

**Key Feature**: Input channels = 6 (image + mask), Output channels = 3 (predicted clean image)

```python
class SongUNet(torch.nn.Module):
    def __init__(self, img_resolution, in_channels, out_channels, ...):
        # in_channels = 6 for [RGB image + RGB mask]
        # out_channels = 3 for RGB prediction
        ...
```

---

## Dataset Implementation

### Dataset with Corruption (`training/dataset.py`)

#### ImageFolderDataset (Lines 219-302)

```python
class ImageFolderDataset(Dataset):
    def __init__(self, path, corruption_probability=0., 
                 delta_probability=0., mask_full_rgb=False,
                 corruption_pattern="dust", **kwargs):
        self.corruption_probability = corruption_probability  # p
        self.delta_probability = delta_probability            # δ
        self.corruption_pattern = corruption_pattern
        ...
    
    def __getitem__(self, idx):
        """
        Returns: (image, label, corruption_mask, hat_corruption_mask)
        
        - image: Clean image (normalized to [-1, 1])
        - corruption_mask: A (original corruption)
        - hat_corruption_mask: Ã (further corruption)
        """
        raw_idx = self._raw_idx[idx]
        image = self._load_raw_image(raw_idx)
        
        # Normalize image
        if self.normalize:
            image = image.astype(np.float32) / 127.5 - 1
        
        # Set seed for reproducibility (same corruption for same image)
        np.random.seed(raw_idx)
        torch.manual_seed(raw_idx)
        
        if self.corruption_pattern == "dust":
            # Random inpainting
            if self.mask_full_rgb:
                # Same mask for all channels
                corruption_mask = np.random.binomial(
                    1, 1 - self.corruption_probability, size=image.shape[1:]
                )
                corruption_mask = corruption_mask[np.newaxis, :, :].repeat(
                    image.shape[0], axis=0
                )
                
                extra_mask = np.random.binomial(
                    1, 1 - self.delta_probability, size=image.shape[1:]
                )
                extra_mask = extra_mask[np.newaxis, :, :].repeat(
                    image.shape[0], axis=0
                )
                
                # Ã = A ⊙ B (element-wise product)
                hat_corruption_mask = np.minimum(corruption_mask, extra_mask)
            else:
                # Independent mask per channel
                corruption_mask = np.random.binomial(
                    1, 1 - self.corruption_probability, size=image.shape
                )
                hat_corruption_mask = np.minimum(
                    corruption_mask,
                    np.random.binomial(1, 1 - self.delta_probability, size=image.shape)
                )
        
        elif self.corruption_pattern == "box":
            # Box masking (two random boxes)
            corruption_mask = get_box_mask(
                (1,) + image.shape, 1 - self.corruption_probability, device='cpu'
            )[0]
            hat_corruption_mask = get_box_mask(
                (1,) + image.shape, 1 - self.corruption_probability, device='cpu'
            )[0]
            hat_corruption_mask = corruption_mask * hat_corruption_mask
        
        elif self.corruption_pattern == "fixed_box":
            # Fixed-size box at random location
            patch_size = int(self.corruption_probability * image.shape[-2])
            corruption_mask = 1 - get_patch_mask(
                (1,) + image.shape, patch_size, device='cpu'
            )[0]
            
            if self.delta_probability > 0:
                hat_corruption_mask = 1 - get_patch_mask(
                    (1,) + image.shape, patch_size, device='cpu'
                )[0]
                hat_corruption_mask = corruption_mask * hat_corruption_mask
            else:
                hat_corruption_mask = corruption_mask
        
        return image.copy(), self.get_label(idx), corruption_mask, hat_corruption_mask
```

**Critical Details**:

1. **Deterministic Corruption**: Using `raw_idx` as seed ensures same corruption for same image across epochs. This is important for:
   - Reproducibility
   - Consistent evaluation
   - Proper EMA updates

2. **Two Masks Returned**:
   - `corruption_mask` (A): Original corruption applied to dataset
   - `hat_corruption_mask` (Ã): Further corruption for training

3. **Multiple Corruption Patterns**:
   - `dust`: Random pixel dropout (main experiments)
   - `box`: Random box regions
   - `fixed_box`: Fixed-size box (used for MRI experiments)
   - `keep_patch`: Keep only a patch, corrupt rest

**Paper Connection**: Section 5.1 describes these exact corruption patterns.

---

## Sampling Implementation

### 1. Fixed Mask Sampler (`eval.py`, Lines 34-151)

```python
def ambient_sampler(
    net, latents, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    survival_probability=0.54, mask_full_rgb=False,
    num_masks=1, guidance_scale=0.0, clipping=True):
    """
    Ambient diffusion sampler with fixed or multiple masks.
    
    Implements Equation 127 from the paper:
    x_{t-Δt} = γ_t x_t + (1-γ_t) E[x₀|Ã x_t, Ã]
    
    Args:
        net: Trained Ambient Diffusion model
        latents: Initial noise z ~ N(0, σ_max² I)
        survival_probability: (1-p)(1-δ) for sampling mask
        num_masks: Number of masks for reconstruction guidance
        guidance_scale: Weight for reconstruction guidance term
    """
    # Adjust noise levels
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # Sample corruption mask(s)
    masks = []
    for _ in range(num_masks):
        masks.append(get_random_mask(
            latents.shape, survival_probability,
            mask_full_rgb=mask_full_rgb, device=latents.device
        ))
    masks = torch.stack(masks)
    
    # Time step discretization (EDM-style)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max**(1/rho) + step_indices/(num_steps-1) * 
               (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    
    # Initialize: x_T = z * σ_max
    x_next = latents.to(torch.float64) * t_steps[0]
    
    # Denoising loop
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Optional: Add stochasticity (S_churn parameter)
        gamma = min(S_churn/num_steps, np.sqrt(2)-1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        
        x_hat.requires_grad = True  # For reconstruction guidance
        
        # Denoise with first mask
        denoised = []
        for mask_index in range(num_masks):
            corruption_mask = masks[mask_index]
            
            # Apply corruption: Ã * x_hat
            masked_image = corruption_mask * x_hat
            
            # Network input: [Ã * x_hat, Ã]
            net_input = torch.cat([masked_image, corruption_mask], dim=1)
            
            # Predict clean image: E[x₀|Ã x_hat, Ã]
            net_output = net(net_input, t_hat, class_labels).to(torch.float64)[:, :3]
            
            # Optional clipping
            if clipping:
                net_output = tensor_clipping(net_output, static=static)
            
            denoised.append(net_output)
        
        # Reconstruction guidance (if multiple masks)
        if guidance_scale > 0 and num_masks > 1:
            stack_denoised = torch.stack(denoised)
            # Encourage consistency across different mask predictions
            flattened = stack_denoised.view(stack_denoised.shape[0], -1)
            l2_norm = cdist_masked(flattened, flattened, None, None).mean()
            rec_grad = torch.autograd.grad(l2_norm, inputs=x_hat)[0]
        else:
            rec_grad = 0
        
        clean_pred = denoised[0]
        
        # Euler step (Equation 127)
        # x_{t-Δt} = γ_t x_t + (1-γ_t) E[x₀|Ã x_t, Ã]
        # Rewritten as: x_{t-Δt} = x_t + (t_next - t_hat)/t_hat * (x_t - clean_pred)
        single_mask_grad = (t_next - t_hat) * (x_hat - clean_pred) / t_hat
        
        # Add reconstruction guidance
        grad_1 = single_mask_grad - guidance_scale * rec_grad
        x_next = x_hat + grad_1
        
        # Heun's second-order correction (optional)
        if i < num_steps - 1:
            # Predict again at x_next
            x_next.requires_grad = True
            # ... (similar prediction + guidance) ...
            grad_2 = ... # computed similarly
            
            # Average of two gradients
            x_next = x_hat + 0.5 * (grad_1 + grad_2)
    
    return x_next
```

**Paper Connection**:
- **Fixed Mask Sampler**: Section 4.1, Equation 127
  ```
  x_{t-Δt} = (σ_{t-Δt}/σ_t) x_t + (σ_t - σ_{t-Δt})/σ_t * E[x₀|Ã x_t, Ã]
  ```

- **Reconstruction Guidance**: Section 4.1 - "We add an additional term that enforces updates that lead to consistency on the reconstructed image."
  ```
  x_{t-Δt} = γ_t x_t + (1-γ_t) E[x₀|Ã x_t, Ã] - w_t ∇_x E_{A'} ||E[x₀|Ã x_t,Ã] - E[x₀|Ã' x_t,Ã']||²
  ```

### 2. Reconstruction Guidance

The implementation includes reconstruction guidance (lines 109-117):

```python
# Compute consistency loss between different mask predictions
stack_denoised = torch.stack(denoised)  # [num_masks, B, C, H, W]
flattened = stack_denoised.view(stack_denoised.shape[0], -1)
l2_norm = cdist_masked(flattened, flattened, None, None).mean()

# Gradient of consistency loss w.r.t. x_hat
rec_grad = torch.autograd.grad(l2_norm, inputs=x_hat)[0]

# Include in update
grad_1 = single_mask_grad - guidance_scale * rec_grad
```

This corresponds to the term `-w_t ∇_x E_{A'} ||...||²` in the paper.

---

## Foundation Model Finetuning

### IF Model Finetuning (`train_text_to_image.py`)

This script finetunes DeepFloyd's IF model on corrupted data for:
1. **Privacy**: Training on small datasets without memorization
2. **Domain adaptation**: Learning new domains with limited corrupted data

#### Key Components

**1. Model Setup (Lines 428-437)**:

```python
# Load T5 text encoder (frozen, 8-bit quantized)
if not args.empty_text:
    text_encoder = T5EncoderModel.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", subfolder="text_encoder",
        device_map="auto", load_in_8bit=True, variant="8bit"
    )
    text_encoder.requires_grad_(False)
else:
    # Use pre-computed empty embeddings for unconditional generation
    empty_embeds = torch.load("empty_embeds.pt").to(accelerator.device)

# Load U-Net (trainable)
unet = UNet2DConditionModel.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="unet")
unet.requires_grad_(True)
```

**2. Corruption in Dataset (Lines 596-614)**:

```python
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    
    image_shape = (len(examples["pixel_values"]), 3, 64, 64)
    
    if args.corruption_pattern == 'dust':
        # Random pixel corruption
        corruption_mask = get_random_mask(
            image_shape, 
            survival_probability=1 - args.corruption_probability,
            mask_full_rgb=True
        )
        extra_mask = get_random_mask(
            image_shape,
            survival_probability=1 - args.delta_probability,
            mask_full_rgb=True
        )
    elif args.corruption_pattern == 'fixed_box':
        # Fixed-size box corruption (for MRI-like data)
        patch_size = int(args.corruption_probability * image_shape[-2])
        corruption_mask = 1 - get_patch_mask(image_shape, patch_size)
        
        if args.delta_probability > 0:
            extra_mask = 1 - get_patch_mask(image_shape, patch_size)
        else:
            extra_mask = corruption_mask
    
    hat_corruption_mask = extra_mask * corruption_mask
    
    examples["corruption_masks"] = [mask for mask in corruption_mask]
    examples["hat_corruption_masks"] = [mask for mask in hat_corruption_mask]
    
    return examples
```

**3. Training Loop (Lines 752-856)**:

```python
for step, batch in enumerate(train_dataloader):
    with accelerator.accumulate(unet):
        latents = batch["pixel_values"]  # Clean images
        
        # Sample noise and timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                   (bsz,), device=latents.device)
        
        # Add noise: x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Apply FURTHER corruption Ã
        noisy_latents = batch["hat_corruption_masks"] * noisy_latents
        
        # Get text embeddings
        if not args.empty_text:
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
        else:
            encoder_hidden_states = empty_embeds.repeat([noisy_latents.shape[0], 1, 1])
        
        # Predict noise
        noise_pred_all = unet(
            noisy_latents, timesteps, encoder_hidden_states
        ).sample
        
        # Split prediction and variance
        noise_pred, predicted_variance = torch.split(
            noise_pred_all, noisy_latents.shape[1], dim=1
        )
        
        # Compute predicted clean image from noise prediction
        alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps.cpu()]
        alpha_prod_t = alpha_prod_t.reshape(-1, 1, 1, 1).to(accelerator.device)
        beta_prod_t = 1 - alpha_prod_t
        
        # Tweedie's formula: E[x₀|x_t] = (x_t - √(1-ᾱ_t) ε_θ(x_t,t)) / √ᾱ_t
        pred_original_sample = (
            batch["hat_corruption_masks"] * 
            (noisy_latents - beta_prod_t**(0.5) * noise_pred) / alpha_prod_t**(0.5) +
            (1 - batch["hat_corruption_masks"]) * noise_pred
        )
        
        # AMBIENT DIFFUSION LOSS: Evaluate on pixels in A (not Ã)
        loss = F.mse_loss(
            (batch["corruption_masks"] * pred_original_sample).float(),
            (batch["corruption_masks"] * batch["pixel_values"]).float(),
            reduction="mean"
        )
        
        # Also compute losses for monitoring
        with torch.no_grad():
            # Loss on ALL pixels (test)
            test_loss = F.mse_loss(
                pred_original_sample.float(),
                batch["pixel_values"].float(),
                reduction="mean"
            )
            
            # Loss on unobserved pixels only
            unobserved_loss = F.mse_loss(
                (1 - batch["corruption_masks"]) * pred_original_sample,
                (1 - batch["corruption_masks"]) * batch["pixel_values"],
                reduction="mean"
            )
            
            # Loss on pixels in A but not Ã (the "hat" pixels)
            loss_on_hat_pixels = F.mse_loss(
                (batch["corruption_masks"] * (1 - batch["hat_corruption_masks"]) * 
                 pred_original_sample).float(),
                (batch["corruption_masks"] * (1 - batch["hat_corruption_masks"]) * 
                 batch["pixel_values"]).float(),
                reduction="mean"
            )
        
        # Backpropagation
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

**Critical Insights**:

1. **DDPM vs EDM**: The IF model uses DDPM formulation (predicting noise ε instead of denoised image directly). The loss is converted to clean image space using Tweedie's formula.

2. **Loss on Predicted Clean Image**: Unlike EDM loss which optimizes `||h_θ(x_t,t) - x₀||²`, DDPM formulation first predicts noise, then computes clean image, then evaluates loss. Mathematically equivalent but numerically different.

3. **Four Loss Types**:
   - `loss`: Training objective (on A)
   - `test_loss`: Full image loss
   - `unobserved_loss`: Loss on missing pixels (should be good if model generalizes)
   - `loss_on_hat_pixels`: Loss on pixels removed by further corruption (tests if model learned on those)

**Paper Connection**: Section 5.3 - "We show that we can also finetune a pre-trained foundation model on a new domain given a limited-sized dataset in a few hours in a single GPU."

---

## Inverse Problems

### Implementation (`solve_inverse_problems.py`)

This script solves inverse problems using Ambient Diffusion models via **Diffusion Posterior Sampling (DPS)**.

#### Core Sampler (Lines 29-116)

```python
def ambient_sampler(
    net, latents, corrupted_images, operator, operator_params,
    class_labels=None, dps_scale=5.0, ...):
    """
    Solve inverse problem: recover x₀ from measurement y = Φ(x₀)
    
    Uses DPS: add gradient of data-fidelity term to denoising update
    
    Args:
        net: Ambient diffusion model
        corrupted_images: Measurements y = Φ(x₀)
        operator: Forward operator Φ (e.g., inpainting, downsampling)
        dps_scale: Weight for data-fidelity term
    """
    # Time step discretization (same as generation)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max**(1/rho) + step_indices/(num_steps-1) * 
               (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    
    # Sample corruption mask for model input
    corruption_mask = get_random_mask(
        latents.shape, survival_probability,
        mask_full_rgb=mask_full_rgb, device=latents.device
    )
    
    # Initialize
    x_next = latents * t_steps[0]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Stochasticity
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        
        x_hat.requires_grad = True  # For DPS gradient
        
        # Denoise with Ambient model
        masked_image = corruption_mask * x_hat
        net_input = torch.cat([masked_image, corruption_mask], dim=1)
        net_output = net(net_input, t_hat, class_labels).to(torch.float32)[:, :3]
        
        if clipping:
            net_output = tensor_clipping(net_output)
        
        # DPS: Compute data-fidelity gradient
        # Φ(x̂₀) where x̂₀ = net_output is predicted clean image
        corrupted_net_output = operator.corrupt(net_output, operator_params)[0]
        
        # Data-fidelity loss: ||Φ(x̂₀) - y||²
        data_fidelity = torch.linalg.norm(corrupted_net_output - corrupted_images)
        
        # Gradient: ∇_x_hat ||Φ(x̂₀) - y||²
        dps_grad_1 = -torch.autograd.grad(
            outputs=data_fidelity, inputs=x_hat
        )[0]
        
        # Denoising gradient (standard Ambient Diffusion update)
        denoising_grad_1 = (t_next - t_hat) * (x_hat - net_output) / t_hat
        
        # Combined update: denoising + data-fidelity
        grad_1 = denoising_grad_1 + dps_scale * dps_grad_1
        x_next = x_hat + grad_1
        
        # Heun's method (second-order correction)
        if i < num_steps - 1:
            x_next.requires_grad = True
            
            # Predict again at x_next
            masked_image = corruption_mask * x_next
            net_input = torch.cat([masked_image, corruption_mask], dim=1)
            net_output = net(net_input, t_next, class_labels).to(torch.float32)[:, :3]
            
            if clipping:
                net_output = tensor_clipping(net_output)
            
            # DPS gradient at x_next
            corrupted_net_output = operator.corrupt(net_output, operator_params)[0]
            data_fidelity = torch.linalg.norm(corrupted_net_output - corrupted_images)
            dps_grad_2 = -torch.autograd.grad(
                outputs=data_fidelity, inputs=x_next
            )[0]
            
            denoising_grad_2 = (t_next - t_hat) * (x_next - net_output) / t_next
            
            grad_2 = denoising_grad_2 + dps_scale * dps_grad_2
            
            # Average gradients
            x_next = x_hat + 0.5 * (grad_1 + grad_2)
    
    return x_next
```

**How DPS Works**:

1. **Standard Denoising**: `x_{t-1} = x_t + (t_{t-1} - t_t)/t_t * (x_t - x̂₀)`

2. **Add Data-Fidelity**: `x_{t-1} = x_t + (denoising_grad) + λ * (data_fidelity_grad)`
   - `data_fidelity_grad = -∇_x ||Φ(x̂₀) - y||²`
   - Pushes x_t towards values that, when denoised, match the measurements

3. **Why This Works**: 
   - Denoising gradient ≈ score function of p(x_t)
   - Data-fidelity gradient ≈ score function of p(y|x_t)
   - Combined: samples from p(x_t|y) ∝ p(x_t) p(y|x_t)

**Paper Connection**: Not explicitly in main paper, but this is a standard technique (DPS, 2022) adapted to work with Ambient Diffusion models.

---

## Evaluation

### 1. FID and Inception Score (`eval.py`, Lines 366-520)

```python
def calculate_inception_stats(image_path, num_expected=None, seed=0, 
                              max_batch_size=64, device=torch.device('cuda')):
    """
    Compute Inception features and FID statistics.
    
    Uses Inception-v3 to extract 2048-d features from generated images,
    then computes mean and covariance for FID calculation.
    """
    # Load Inception-v3
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/...'
    with dnnlib.util.open_url(detector_url) as f:
        detector_net = pickle.load(f).to(device)
    
    # Load dataset
    dataset_obj = dataset.ImageFolderDataset(
        path=image_path, max_size=num_expected, 
        random_seed=seed, normalize=False
    )
    
    # Compute features
    mu = torch.zeros([2048], dtype=torch.float64, device=device)
    sigma = torch.zeros([2048, 2048], dtype=torch.float64, device=device)
    all_features = []
    
    for images, _labels, _, _ in data_loader:
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])  # Grayscale to RGB
        
        # FID features
        features = detector_net(images.to(device), return_features=True)
        features = features.to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features
        
        # Inception score features (with softmax)
        inception_features = torch.clamp(
            detector_net(images.to(device), no_output_bias=True),
            min=1e-6, max=1.0
        )
        all_features.append(inception_features.to(torch.float64))
    
    # Compute FID statistics
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    
    # Compute Inception Score
    all_features = torch.cat(all_features, dim=0)
    gen_probs = all_features.cpu().numpy()
    
    # IS = exp(E[KL(p(y|x) || p(y))])
    kl = gen_probs * (np.log(gen_probs) - 
                      np.log(np.mean(gen_probs, axis=0, keepdims=True)))
    kl = np.mean(np.sum(kl, axis=1))
    inception_score = np.mean(np.exp(kl))
    
    return mu.cpu().numpy(), sigma.cpu().numpy(), inception_score

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    """
    Compute FID between two distributions:
    
    FID = ||μ - μ_ref||² + Tr(Σ + Σ_ref - 2√(ΣΣ_ref))
    """
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))
```

**Paper Connection**: Table 2 reports FID and Inception scores for models trained with different corruption levels.

### 2. Memorization Analysis (`find_dataset_neighbors.py`)

```python
@click.command()
def main(input_dir, output_dir, features_path, data, ...):
    """
    Find nearest neighbors of generated images in training set using DINO features.
    
    Used to detect memorization (Figure 5 in paper).
    """
    # Load DINO-v2 feature extractor
    feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor, ...)
    
    # Extract features from training dataset
    if not is_file(features_path):
        features = []
        for dataset_iter in tqdm(dataset_iterator):
            images = dataset_iter[0].to('cuda').to(torch.float32)
            local_features = feature_extractor((pad_image(images) + 1) / 2.).cpu()
            features.append(local_features)
        features = np.concatenate(features)
        np.save(features_path, features)
    else:
        features = np.load(features_path)
    
    # Normalize features
    if normalize:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # For each generated image, find nearest neighbors
    max_products = []
    softmax_products = []
    
    for iter_index, dataset_iter in enumerate(tqdm(outputs_dataset_obj)):
        images = dataset_iter[0]
        images = torch.tensor(images, device='cuda').to(torch.float32).unsqueeze(0)
        
        # Extract DINO features
        local_features = feature_extractor((pad_image(images) + 1) / 2.).cpu()
        
        if normalize:
            local_features /= np.linalg.norm(local_features, axis=1, keepdims=True)
        
        # Compute cosine similarity with all training images
        products = (local_features @ features.T).squeeze()
        
        # Metrics
        softmax_products.append(torch.nn.functional.softmax(products).max())
        max_products.append(float(products.max()))
        
        # Save top-3 nearest neighbors
        sorted_indices = products.argsort().tolist()[::-1]
        for k in range(3):
            i = sorted_indices[k]
            neighbor_image = torch.tensor(dataset_obj[i][0])
            save_image(neighbor_image, 
                      os.path.join(output_dir, f'{iter_index}_nearest_neighbor_{k}.png'))
    
    # Save similarity distributions
    with open(os.path.join(output_dir, 'max_products.pkl'), 'wb') as f:
        pickle.dump(max_products, f)
```

**Paper Connection**: 
- Section 5.3 (Memorization): "To quantify the memorization, we follow the methodology of Somepalli et al. 2022. Specifically, we generate 10000 images from each model and we use DINO-v2 to compute top-1 similarity to the training images."
- Figure 5 shows DINO similarity distributions for different corruption levels

---

## Paper-to-Code Mapping

### Theoretical Concepts → Implementation

| Paper Concept | Code Location | Implementation |
|--------------|---------------|----------------|
| **Corruption Matrix A** | `torch_utils/ambient_diffusion.py` | `MaskingForwardOperator.corrupt()` |
| **Further Corruption Ã = BA** | `torch_utils/ambient_diffusion.py` | `MaskingForwardOperator.hat_corrupt()` |
| **Training Objective (Eq. 100)** | `training/loss.py` | `AmbientLoss.__call__()` with `val_loss` |
| **E[x₀\|Ãx_t,Ã]** | `training/networks.py` | `EDMPrecond.forward()` output |
| **Fixed Mask Sampler (Eq. 127)** | `eval.py` | `ambient_sampler()` |
| **Reconstruction Guidance** | `eval.py` | Lines 109-117, multi-mask consistency |
| **Gated Convolutions** | `training/networks.py` | `Conv2d` class with `gated=True` |
| **Random Inpainting** | `training/dataset.py` | `corruption_pattern="dust"` |
| **Box Inpainting** | `training/dataset.py` | `corruption_pattern="fixed_box"` |
| **IF Finetuning** | `train_text_to_image.py` | Full script |
| **Inverse Problems (DPS)** | `solve_inverse_problems.py` | `ambient_sampler()` with DPS |
| **DINO Memorization** | `find_dataset_neighbors.py` | Full script |

### Key Equations → Code

**Equation 100** (Training Loss):
```
J^{corr}(θ) = (1/2) E_{(x₀,x_t,A,Ã)} ||A(h_θ(Ã,Ãx_t,t) - x₀)||²
```
→ `training/loss.py:108-110`
```python
val_loss = weight * ((corruption_matrix * (D_yn - y)) ** 2)
```

**Equation 127** (Sampling):
```
x_{t-Δt} = (σ_{t-Δt}/σ_t)x_t + (σ_t - σ_{t-Δt})/σ_t * E[x₀|Ãx_t,Ã]
```
→ `eval.py:115`
```python
single_mask_grad = (t_next - t_hat) * (x_hat - clean_pred) / t_hat
x_next += single_mask_grad
```

**Theorem 1** (Minimizer):
```
h_θ*(Ã,y_t,t) = E[x₀|Ãx_t,Ã]
```
→ Implicit in network training: `training/loss.py:106` optimizes for this by penalizing `||A(h_θ(...) - x₀)||²`

### Corruption Patterns

| Paper Description | Code Parameter | Implementation |
|------------------|----------------|----------------|
| "deleting each pixel with probability p" | `corruption_probability=0.9` | `corruption_pattern="dust"` |
| "further delete δ-fraction of surviving pixels" | `delta_probability=0.1` | `hat_corruption_mask = corruption_mask * extra_mask` |
| "random box of fixed size" | `corruption_probability=0.25` | `corruption_pattern="fixed_box"` |
| "Gaussian measurements" | N/A (not in main experiments) | `CompressedSensingOperator` |

---

## Consistency Verification

### ✅ Verified Implementations

1. **Loss Function**: 
   - ✅ Three losses computed (train, val, test)
   - ✅ Uses `val_loss` for backpropagation (correct!)
   - ✅ Mask concatenation with image

2. **Dataset Corruption**:
   - ✅ Returns two masks (A and Ã)
   - ✅ Deterministic corruption per image
   - ✅ Multiple corruption patterns

3. **Network Architecture**:
   - ✅ 6 input channels (3 image + 3 mask)
   - ✅ 3 output channels (RGB prediction)
   - ✅ Gated convolutions enabled
   - ✅ EDM preconditioning

4. **Sampling**:
   - ✅ Fixed mask sampler matches Equation 127
   - ✅ Reconstruction guidance implemented
   - ✅ Heun's second-order method

5. **IF Finetuning**:
   - ✅ Loss on predicted clean image, not noise
   - ✅ Further corruption applied to noisy latents
   - ✅ Empty text embeddings for unconditional

### ⚠️ Important Notes

1. **Why `val_loss`?**: 
   - Training optimizes for pixels in `A` (original corruption)
   - `train_loss` (on Ã) is just for monitoring
   - `test_loss` (on all pixels) shows generalization

2. **Mask as Input**:
   - Network MUST know which pixels are corrupted
   - Concatenating mask is simple and effective
   - Alternative: use Partial Convolutions (not used here)

3. **Further Corruption `δ`**:
   - Can be very small (paper uses 0.1)
   - As long as δ > 0, Theorem 1 holds
   - Smaller δ means Ã ≈ A (closer to desired distribution)

4. **Sampling Masks**:
   - At test time, sample masks from p(A) or p(Ã)
   - Paper mostly uses p(Ã) (survival_probability = (1-p)(1-δ))
   - Multiple masks enable reconstruction guidance

---

## Advanced Topics

### 1. Gated Convolutions

**Why Important**: Standard convolutions treat all spatial locations equally. For inpainting:
- Corrupted pixels should contribute less
- Gating learns to suppress invalid regions
- Similar to attention mechanism

**Implementation**: Each convolution has:
- Standard weights `w`
- Gate weights `g`
- Output = `conv(x, w) * sigmoid(conv(x, g))`

### 2. EDM Preconditioning

**Why Important**: Ensures optimal training dynamics across all noise levels

**Scaling Factors**:
```python
c_skip = σ_data² / (σ² + σ_data²)
c_out = σ · σ_data / √(σ² + σ_data²)
c_in = 1 / √(σ_data² + σ²)
c_noise = log(σ) / 4
```

These ensure:
- Network operates at unit scale
- Skip connection weight balanced with network output
- Loss scale approximately constant across σ

### 3. Noise Schedule

**Log-Normal Distribution**:
```python
σ ~ LogNormal(μ=-1.2, std=1.2)
```

Samples more at mid-range noise levels where learning is hardest.

**Time Steps** (EDM discretization):
```python
t_i = (σ_max^{1/ρ} + i/(N-1) * (σ_min^{1/ρ} - σ_max^{1/ρ}))^ρ
```
with ρ=7 concentrates steps at low noise levels.

### 4. EMA (Exponential Moving Average)

```python
# training_loop.py:178-183
ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
for p_ema, p_net in zip(ema.parameters(), net.parameters()):
    p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
```

EMA maintains a moving average of network weights. Typically produces better samples than final weights.

---

## Common Issues and Solutions

### Issue 1: Training is Unstable

**Symptom**: Loss explodes or becomes NaN

**Solutions**:
1. Enable gradient clipping: `--max_grad_norm=1.0`
2. Check corruption probabilities aren't too extreme
3. Reduce learning rate
4. Ensure masks are binary (0 or 1), not continuous

### Issue 2: Model Doesn't Learn Unobserved Pixels

**Symptom**: `test_loss >> val_loss`

**Diagnosis**: 
- Model hasn't learned full distribution
- Likely issue: δ = 0 (no further corruption)
- Theorem 1 requires δ > 0

**Solution**: Ensure `delta_probability > 0` (even 0.01 works)

### Issue 3: Sampling Produces Blurry Images

**Symptom**: Generated images lack detail

**Solutions**:
1. Increase number of sampling steps
2. Enable stochasticity (S_churn > 0)
3. Check mask used for sampling matches training distribution
4. Try reconstruction guidance

### Issue 4: IF Model Finetuning Fails

**Symptom**: Loss doesn't decrease

**Solutions**:
1. Verify corruption is applied to `noisy_latents`, not just `latents`
2. Check that loss is computed on predicted clean image
3. Ensure text embeddings have correct shape
4. Reduce learning rate (IF is pretrained, needs gentle finetuning)

---

## Recommended Experiments

### For Understanding:
1. **Vary δ**: Train with δ ∈ {0, 0.05, 0.1, 0.2}
   - δ=0 should fail (model only learns observed pixels)
   - Larger δ might hurt performance (Ã far from A)

2. **Ablate Gated Convolutions**: `--gated=False`
   - Compare FID scores
   - Check if model handles unobserved pixels well

3. **Different Corruption Patterns**:
   - `dust` vs `box` vs `fixed_box`
   - See which generalizes better

### For Research:
1. **Compressed Sensing**: Use `CompressedSensingOperator`
   - Can you learn from random projections?
   - How many measurements needed?

2. **Inverse Problems**: Use trained model for:
   - Super-resolution
   - Inpainting with different patterns
   - Deblurring

3. **Memorization-Privacy Trade-off**:
   - Plot DINO similarity vs FID for different p
   - Find optimal corruption level

---

## Code Quality Observations

### Strengths:
1. ✅ Modular design (operators, loss, networks separate)
2. ✅ Extensive use of type hints and docstrings
3. ✅ Distributed training support
4. ✅ Reproducible (seeds, deterministic corruption)
5. ✅ Multiple corruption patterns supported

### Potential Improvements:
1. ⚠️ `samplers.py` is a placeholder (could consolidate sampling code)
2. ⚠️ Some repeated code between `eval.py` and `solve_inverse_problems.py`
3. ⚠️ Hard-coded paths in some scripts
4. ⚠️ Limited documentation for IF pipeline modifications

---

## Additional Implementation Details

### 1. Custom DDPM Scheduler (`scheduling_ddpm.py`)

The IF pipeline uses a modified DDPM scheduler that accepts a `clean_image` parameter:

```python
def step(self, model_output, timestep, sample, generator=None, 
         return_dict=True, clean_image=None):
    """
    Modified DDPM step function that can override predicted clean image.
    
    Standard DDPM: Predict x₀ from (x_t, ε_θ), then compute x_{t-1}
    
    With clean_image: Use provided x₀ instead of model's prediction
    """
    # Split model output into noise prediction and variance
    model_output, predicted_variance = torch.split(
        model_output, sample.shape[1], dim=1
    )
    
    # Compute predicted original sample from predicted noise
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t**(0.5) * model_output) / 
                               alpha_prod_t**(0.5)
    
    # Clip or threshold predicted x₀
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    
    # KEY MODIFICATION: Override with provided clean image
    if clean_image is not None:
        pred_original_sample = clean_image
    
    # Compute x_{t-1} using standard DDPM formula
    pred_original_sample_coeff = (alpha_prod_t_prev**(0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t**(0.5) * beta_prod_t_prev / beta_prod_t
    
    pred_prev_sample = (pred_original_sample_coeff * pred_original_sample + 
                        current_sample_coeff * sample)
    
    # Add noise variance
    if t > 0:
        variance = self._get_variance(t, predicted_variance)
        pred_prev_sample = pred_prev_sample + variance
    
    return pred_prev_sample, pred_original_sample
```

**Why This Matters**:
- Allows injecting custom denoising predictions into DDPM sampling
- Used in IF pipeline (line 925) to use Ambient Diffusion predictions:
  ```python
  pred_original_sample = mask * (model_input - ...) / ... + (1 - mask) * model_output
  intermediate_images = self.scheduler.step(
      noise_pred, t, intermediate_images, 
      clean_image=pred_original_sample  # Custom prediction
  )[0]
  ```

**Paper Connection**: Not mentioned in paper, but necessary for adapting DDPM-based models (IF) to Ambient Diffusion.

---

### 2. IF Pipeline Modifications (`pipeline_if.py`)

The custom IF pipeline extends the standard DeepFloyd pipeline with Ambient Diffusion support:

#### Key Modifications in `__call__` Method (Lines 900-925):

```python
def __call__(self, ..., corruption_probability=0.0, delta_probability=0.0, 
             corruption_pattern="dust", ...):
    """
    Modified IF pipeline that applies corruption during generation.
    """
    # ... standard setup ...
    
    # Denoising loop
    mask = None
    for i, t in enumerate(timesteps):
        model_input = intermediate_images
        
        # Create corruption masks (only once, then fixed for all steps)
        if mask is None:
            if corruption_pattern == "dust":
                corruption_mask = get_random_mask(
                    model_input.shape, 1 - corruption_probability,
                    device=device, mask_full_rgb=True
                )
                extra_mask = get_random_mask(
                    model_input.shape, 1 - delta_probability,
                    device=device, mask_full_rgb=True
                )
            elif corruption_pattern == "fixed_box":
                patch_size = int(corruption_probability * model_input.shape[-2])
                corruption_mask = 1 - get_patch_mask(
                    model_input.shape, patch_size
                )
                extra_mask = 1 - get_patch_mask(
                    model_input.shape, patch_size
                )
            
            # Ã = A ⊙ B
            mask = corruption_mask * extra_mask
        
        # Apply corruption to input
        corrupted_model_input = (model_input * mask).to(model_input.dtype)
        
        # U-Net prediction (noise)
        noise_pred = self.unet(
            corrupted_model_input, t, 
            encoder_hidden_states=prompt_embeds
        ).sample
        
        # Split noise and variance predictions
        model_output, predicted_variance = torch.split(
            noise_pred, model_input.shape[1], dim=1
        )
        
        # Convert noise prediction to clean image prediction
        # Tweedie's formula with mask awareness:
        # For observed pixels: standard formula
        # For unobserved pixels: use noise prediction directly
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        
        pred_original_sample = (
            mask * (model_input - beta_prod_t**(0.5) * model_output) / 
                   alpha_prod_t**(0.5) +
            (1 - mask) * model_output
        )
        
        # Threshold/clip prediction
        pred_original_sample = self.scheduler._threshold_sample(pred_original_sample)
        
        # DDPM step with custom clean image
        intermediate_images = self.scheduler.step(
            noise_pred, t, intermediate_images,
            clean_image=pred_original_sample  # Use our prediction
        )[0]
```

**Key Differences from Standard IF**:
1. **Fixed Mask**: Mask is created once and reused (not resampled each step)
2. **Masked Input**: U-Net receives corrupted input at each step
3. **Masked Denoising**: For unobserved pixels, use noise prediction directly
4. **Custom Scheduler**: Passes custom `pred_original_sample` to scheduler

**Paper Connection**: Section 5.3 describes IF finetuning. This pipeline enables both training and inference with corruption.

---

### 3. Data Augmentation Pipeline (`training/augment.py`)

The codebase includes a sophisticated augmentation pipeline (from EDM paper):

```python
class AugmentPipe:
    def __init__(self, p=1,
        # Pixel operations
        xflip=0, yflip=0, rotate_int=0, translate_int=0,
        # Geometric transformations
        scale=0, rotate_frac=0, aniso=0, translate_frac=0,
        # Color transformations  
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0):
        """
        Adaptive Discriminator Augmentation (ADA) pipeline.
        
        Each augmentation has a probability multiplier (0 = disabled, 1 = enabled).
        Overall probability 'p' scales all augmentations.
        """
    
    def __call__(self, images):
        """
        Apply augmentations to batch of images.
        
        Returns: (augmented_images, augmentation_labels)
        - augmentation_labels: embedding of applied transformations
        """
        device = images.device
        labels = []
        
        # 1. Pixel blitting (cheap operations)
        if self.xflip > 0:
            # Random horizontal flips
            w = torch.randint(2, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand(...) < self.xflip * self.p, w, 0)
            images = torch.where(w == 1, images.flip(3), images)
            labels += [w]
        
        # 2. Geometric transformations (warping)
        if self.scale > 0 or self.rotate_frac > 0 or ...:
            # Build transformation matrix
            G_inv = I_3  # Identity
            
            if self.scale > 0:
                # Isotropic scaling
                s = torch.randn([N], device=device).mul(self.scale_std).exp2()
                G_inv = G_inv @ scale2d_inv(s, s)
            
            if self.rotate_frac > 0:
                # Fractional rotation
                w = (torch.rand([N], device=device) * 2 - 1) * 
                    (np.pi * self.rotate_frac_max)
                G_inv = G_inv @ rotate2d_inv(-w)
            
            # Apply geometric transformation using grid sampling
            # (includes upsampling, warping, downsampling for antialiasing)
            images = apply_geometric_transform(images, G_inv)
        
        # 3. Color transformations
        if self.brightness > 0 or self.contrast > 0 or ...:
            # Build color transformation matrix (4x4 for RGB + offset)
            M = I_4
            
            if self.brightness > 0:
                b = torch.randn([N], device=device) * self.brightness_std
                M = translate3d(b, b, b) @ M
            
            if self.contrast > 0:
                c = torch.randn([N], device=device).mul(self.contrast_std).exp2()
                M = scale3d(c, c, c) @ M
            
            if self.hue > 0:
                # Rotate around luma axis in RGB space
                w = (torch.rand([N], device=device) * 2 - 1) * 
                    (np.pi * self.hue_max)
                M = rotate3d(luma_axis, w) @ M
            
            # Apply color transform
            images = M[:, :3, :3] @ images + M[:, :3, 3:]
        
        # Concatenate all augmentation labels
        labels = torch.cat([x.flatten() for x in labels], dim=1)
        
        return images, labels
```

**Usage in Training** (`train.py` lines 186-188):
```python
if opts.augment > 0:
    c.augment_kwargs = dnnlib.EasyDict(
        class_name='training.augment.AugmentPipe', 
        p=opts.augment  # Overall augmentation probability
    )
    c.augment_kwargs.update(
        xflip=1e8, yflip=1, scale=1, rotate_frac=1, 
        aniso=1, translate_frac=1
    )
    c.network_kwargs.augment_dim = 9  # Dimension of augmentation labels
```

**Why Augmentation Matters**:
1. **Regularization**: Prevents overfitting, especially with corrupted data
2. **Generalization**: Model sees more varied corrupted patterns
3. **Conditional Training**: Augmentation labels are fed to network

**Paper Connection**: Section 5.1 mentions "augment=0.15" as a hyperparameter. This is the probability `p` in the augmentation pipeline.

---

### 4. IF Inference Script (`if_inference.py`)

Convenience script for generating images from finetuned IF models:

```python
@click.command()
@click.option('--checkpoint_path', type=str, required=True)
@click.option('--output_dir', type=str, required=True)
@click.option('--corruption_probability', type=float, default=0.8)
@click.option('--delta_probability', type=float, default=0.1)
def main(checkpoint_path, output_dir, corruption_probability, ...):
    """
    Generate images from finetuned IF model.
    
    Process:
    1. Load IF stage-I (64×64 generation)
    2. Load finetuned weights
    3. Generate with corruption
    4. Optionally upscale with IF stage-II (256×256)
    """
    # Load text encoder (frozen, 8-bit)
    text_encoder = T5EncoderModel.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", subfolder="text_encoder",
        load_in_8bit=True
    )
    
    # Create pipeline (without U-Net initially)
    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", 
        text_encoder=text_encoder,
        unet=None
    )
    
    # Encode prompts
    prompts = batch_size * [""]  # Empty for unconditional
    prompt_embeds, negative_embeds = pipe.encode_prompt(prompts)
    
    # Delete text encoder (save memory)
    del text_encoder, pipe
    
    # Reload pipeline with U-Net
    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        text_encoder=None,  # Already encoded
        variant="fp16",
        torch_dtype=torch.float16
    )
    
    # Load finetuned weights
    if checkpoint_path is not None:
        loaded_dict = torch.load(checkpoint_path)
        # Fix key names (remove compilation prefixes)
        fixed_dict = OrderedDict({
            key.replace("_orig_mod.module.", ""): value
            for key, value in loaded_dict.items()
        })
        pipe.unet.load_state_dict(fixed_dict)
    
    # Patch scheduler with custom step function
    pipe.scheduler.step = step.__get__(pipe.scheduler, DDPMScheduler)
    
    # Generate images
    for batch_index in range(num_images // batch_size):
        images = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type="pt",
            corruption_probability=corruption_probability,
            delta_probability=delta_probability,
            guidance_scale=0.0,  # Unconditional
            corruption_pattern=corruption_pattern
        ).images
        
        # Save images
        for image_index, image in enumerate(images):
            save_image(image, 
                      os.path.join(output_dir, f"{batch_index}_{image_index}.png"))
    
    # Optional: Upscale with IF Stage-II
    if upscale:
        super_res_pipe = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None
        )
        
        # Load generated images and upscale
        for image in load_images(output_dir):
            super_res_image = super_res_pipe(
                image=image,
                prompt_embeds=prompt_embeds,
                guidance_scale=0.0
            ).images
            save_upscaled(super_res_image)
```

**Key Features**:
1. **Memory Efficient**: Loads text encoder separately, encodes once, then deletes
2. **Checkpoint Loading**: Handles compiled model checkpoints
3. **Custom Scheduler**: Patches scheduler with modified step function
4. **Cascaded Generation**: Supports IF's 64→256 upsampling pipeline

**Paper Connection**: Section 5.3 describes IF finetuning. This script enables using those finetuned models.

---

### 5. Consistency Checks: Paper vs Code

Let me verify all major claims from the paper are implemented:

#### From Paper Abstract:
✅ "introduce additional measurement distortion during the diffusion process" 
   → `hat_corruption_mask` in `AmbientLoss`

✅ "predict the original corrupted image from the further corrupted image"
   → Loss computed on `corruption_matrix`, not `hat_corruption_mask`

✅ "learn the conditional expectation of the full uncorrupted image"
   → `h_θ(Ã, Ãx_t, t)` trained to minimize `||A(h_θ(...) - x₀)||²`

✅ "holds for any corruption process that satisfies some technical conditions"
   → Multiple operators: `MaskingForwardOperator`, `BoxMaskingForwardOperator`, 
     `CompressedSensingOperator`, `AveragingForwardOperator`

✅ "includes inpainting and compressed sensing"
   → `corruption_pattern="dust"` and `CompressedSensingOperator`

✅ "learn the distribution even when all the training samples have 90% of their pixels missing"
   → `corruption_probability=0.9` supported

✅ "finetune foundation models on small corrupted datasets"
   → `train_text_to_image.py` with `max_train_samples=3000`

✅ "learn the clean distribution without memorizing the training set"
   → `find_dataset_neighbors.py` for DINO similarity analysis

#### From Paper Section 3 (Method):
✅ "Concretely, we randomly corrupt A to obtain Ã = BA"
   → Lines 206-208 in `torch_utils/ambient_diffusion.py`:
   ```python
   hat_mask = get_random_mask(..., 1 - self.delta_probability, ...)
   hat_mask = mask * hat_mask  # Ã = A ⊙ B
   ```

✅ "we randomly corrupt A to obtain Ã by randomly erasing an additional fraction δ of the pixels"
   → `delta_probability=δ` parameter throughout

✅ "The key idea: the learner does not know if a missing pixel is missing because we never had it or because it was deliberately erased"
   → Model receives `hat_corruption_mask` but loss is on `corruption_mask`

✅ "the probability δ of further corruption can be arbitrarily small as long as it stays positive"
   → Default `delta_probability=0.1`, but any value > 0 works

#### From Paper Section 4 (Sampling):
✅ "The simplest idea is to sample a mask Ã ~ p(Ã) and approximate E[x₀|x_t] with E[x₀|Ãx_t,Ã]"
   → `ambient_sampler` in `eval.py` lines 61-62:
   ```python
   masks.append(get_random_mask(latents.shape, survival_probability, ...))
   ```

✅ "discretized sampling rule" (Equation 127)
   → Lines 115-117 in `eval.py`:
   ```python
   single_mask_grad = (t_next - t_hat) * (x_hat - clean_pred) / t_hat
   x_next += single_mask_grad
   ```

✅ "Reconstruction Guidance: add an additional term that enforces updates that lead to consistency"
   → Lines 107-117 in `eval.py`:
   ```python
   l2_norm = cdist_masked(flattened, flattened, None, None).mean()
   rec_grad = torch.autograd.grad(l2_norm, inputs=x_hat)[0]
   grad_1 = single_mask_grad - guidance_scale * rec_grad
   ```

#### From Paper Section 5 (Experiments):
✅ "We use the EDM codebase to train our models"
   → `train.py` based on EDM, uses `EDMPrecond`

✅ "We replace convolutions with Gated Convolutions"
   → `Conv2d` class in `training/networks.py` with `gated=True`

✅ "To use the mask Ã as an additional input to the model, we simply concatenate it with the image"
   → Line 105 in `training/loss.py`:
   ```python
   cat_input = torch.cat([noisy_image, hat_corruption_matrix], axis=1)
   ```

✅ "unless mentioned otherwise, we use δ=0.1"
   → Default in `train.py`: `--delta_probability=0.1`

✅ "We use DINO-v2 to compute top-1 similarity to the training images"
   → `find_dataset_neighbors.py` lines 48-49:
   ```python
   feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
   ```

---

## Missing Details Now Covered

The document now includes:

1. ✅ **Custom DDPM Scheduler**: Detailed explanation of `clean_image` parameter
2. ✅ **IF Pipeline Modifications**: How the pipeline handles corruption during generation
3. ✅ **Data Augmentation**: Complete explanation of AugmentPipe and its role
4. ✅ **IF Inference Script**: Practical usage of finetuned models
5. ✅ **Comprehensive Paper-Code Mapping**: Every major claim verified against code

---

## Conclusion

This implementation is a faithful and well-engineered realization of the Ambient Diffusion paper. Key strengths:

1. **Correctness**: Core algorithm (Equation 100) implemented exactly
2. **Flexibility**: Multiple corruption patterns, architectures, training modes
3. **Completeness**: Training, sampling, finetuning, inverse problems all included
4. **Performance**: Uses state-of-art techniques (EDM, gated convs, DPS)

The codebase successfully demonstrates that:
- Diffusion models can learn from highly corrupted data
- Further corruption during training is key (Theorem 1)
- Foundation model finetuning reduces memorization
- Trained models can solve inverse problems

**For users**: Focus on understanding `AmbientLoss` and `MaskingForwardOperator` first, then explore sampling and applications.

**For researchers**: The modular design makes it easy to extend to new corruption types or integrate with other diffusion frameworks.

