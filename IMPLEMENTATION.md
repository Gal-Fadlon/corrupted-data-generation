# Implementation Documentation: Irregular Time Series Generation Using Diffusion Models

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Components](#architecture-components)
3. [Training Pipeline](#training-pipeline)
4. [Model Components](#model-components)
5. [Data Processing](#data-processing)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Utilities and Logging](#utilities-and-logging)
8. [Configuration System](#configuration-system)

---

## 1. Project Overview

This project implements a novel approach for generating regular (evenly-sampled) time series data from irregular (unevenly-sampled) time series with missing values. The key innovation is a two-step framework combining:

1. **Time Series Transformer (TST)** for completing missing values
2. **Vision-based diffusion model** (based on ImagenTime) with masking to avoid over-reliance on completed values

### Key Research Contributions
- Transforms time series into images using delay embedding
- Uses TST to create "natural neighborhoods" (avoiding zeros)
- Applies masking during diffusion training to minimize dependence on imputed values
- Achieves 70% improvement in discriminative score and 85% reduction in computational cost

---

## 2. Architecture Components

### 2.1 Overall System Architecture

```
Input: Irregular Time Series (with NaN values)
    ↓
[TST Encoder + Decoder] → Complete missing values
    ↓
[Delay Embedding] → Convert to Image
    ↓
[Diffusion Model with Masking] → Train on real (non-imputed) pixels
    ↓
[Sampling Process] → Generate new images
    ↓
[Inverse Delay Embedding] → Convert back to Time Series
    ↓
Output: Regular Time Series
```

---

## 3. Training Pipeline

### 3.1 Main Training Script (`run_irregular.py`)

The main training loop consists of two phases:

#### Phase 1: Pre-training TST Encoder-Decoder (First Epoch)
```python
# Lines 200-228
for step in range(1, args.first_epoch + 1):
    for i, data in enumerate(train_loader, 1):
        # 1. Propagate NaN values forward/backward
        x = propagate_values(x)
        
        # 2. Create padding masks
        padding_masks = ~torch.isnan(x).any(dim=-1)
        
        # 3. Encode with TST
        h = embedder(x, padding_masks)
        
        # 4. Decode to reconstruct
        x_tilde = decoder(h)
        
        # 5. Compute loss on non-NaN values only
        loss_e_t0 = MSE_loss(x_tilde_no_nan, x_no_nan)
        loss_e_0 = sqrt(loss_e_t0) * 10
```

**Purpose**: Train the TST autoencoder to reconstruct time series, learning to complete missing values.

#### Phase 2: Joint Training (Main Epochs)
```python
# Lines 230-313
for epoch in range(init_epoch, args.epochs):
    for i, data in enumerate(train_loader, 1):
        # 1. Prepare data
        x_ts = propagate_values(x_ts)
        
        # 2. TST encode
        h = embedder(x_ts, padding_masks)
        x_recon = decoder(h)
        
        # 3. Convert to image
        x_tilde_img = model.ts_to_img(x_recon)
        
        # 4. Create mask (1 for real data, 0 for imputed)
        mask = torch.isnan(x_img).float() * -1 + 1
        
        # 5. Diffusion loss with masking
        loss = model.loss_fn_irregular(x_tilde_img, mask)
        
        # 6. Update diffusion model
        optimizer.step()
        model.on_train_batch_end()  # Update EMA
        
        # 7. Continue refining TST
        loss_e = TST_reconstruction_loss()
        optimizer_er.step()
```

**Purpose**: Train the diffusion model while continuing to refine the TST encoder-decoder.

### 3.2 Key Helper Functions

#### `propagate_values()` (Lines 25-67)
Fills NaN values by forward and backward propagation:
```python
def propagate_values(tensor):
    # Forward propagation: carry last valid value forward
    tensor = propagate_values_forward(tensor)
    
    # Backward propagation: carry next valid value backward
    tensor = propagate_values_backward(tensor)
    
    # Fill any remaining NaNs with 0
    tensor[torch.isnan(tensor)] = 0
    return tensor
```

This ensures the TST has valid inputs to work with while training.

#### `save_checkpoint()` (Lines 69-115)
Saves complete training state including:
- Diffusion model weights
- EMA model weights
- TST encoder/decoder weights
- All optimizer states
- Evaluation scores (discriminative, predictive, FID, correlation)

---

## 4. Model Components

### 4.1 Main Model: `TS2img_Karras` (`models/our.py`)

This is the primary model that orchestrates the entire pipeline.

#### Initialization (Lines 8-41)
```python
class TS2img_Karras(nn.Module):
    def __init__(self, args, device):
        # Karras diffusion parameters
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        
        # U-Net backbone
        self.net = EDMPrecond(...)
        
        # Delay embedding transformer
        self.ts_img = DelayEmbedder(...)
        
        # EMA (Exponential Moving Average)
        self.model_ema = LitEma(self.net, decay=0.9999)
```

#### Key Methods

**`loss_fn_irregular()` (Lines 53-68)**
Computes the masked diffusion loss:
```python
def loss_fn_irregular(self, x, mask=None):
    # 1. Create mask if not provided
    if mask is None:
        mask = torch.isnan(x).float() * -1 + 1
        x = torch.nan_to_num(x, nan=0.0)
    
    # 2. Forward pass with masking
    output, weight = self.forward_irregular(x, mask)
    
    # 3. Apply mask to both prediction and target
    x = self.unpad(x * mask, x.shape)
    output = self.unpad(output * mask, x.shape)
    
    # 4. Weighted MSE loss (only on masked pixels)
    loss = (weight * (output - x).square()).mean()
    
    return loss, to_log
```

**`forward_irregular()` (Lines 70-78)**
Implements the Karras diffusion forward process with masking:
```python
def forward_irregular(self, x, mask, labels=None, augment_pipe=None):
    # 1. Sample noise level from log-normal distribution
    rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
    sigma = (rnd_normal * self.P_std + self.P_mean).exp()
    
    # 2. Compute loss weight
    weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
    
    # 3. Add masked noise
    n = torch.randn_like(y) * sigma
    masked_noise = n * mask  # KEY: Only noise the real pixels!
    
    # 4. Denoise
    D_yn = self.net(y + masked_noise, sigma, labels, augment_labels)
    
    return D_yn, weight
```

**Why This Works**: By masking the noise, the diffusion model only learns to denoise real (observed) pixels, not the imputed ones. This prevents the model from over-relying on potentially incorrect completions.

#### ⚠️ CRITICAL: Understanding the Masking Strategy

This is the most important and subtle aspect of the implementation. Let's be absolutely clear:

**Question 1: Does the diffusion network see all values?**
- **YES** - The U-Net receives the complete image including imputed values
- **BUT** - Noise is ONLY added to real (mask=1) pixels: `masked_noise = n * mask`
- Imputed pixels serve as context but are not corrupted by noise

**Question 2: Is the loss computed on all values?**
- **NO** - Loss is ONLY computed on real pixels
- Both predictions and targets are masked: `x * mask` and `output * mask`
- For imputed pixels: `(0 - 0)² = 0`, contributing nothing to the loss

**Question 3: Does denoising happen on all values?**
- **YES** - The network produces predictions for ALL pixels
- **BUT** - Only real pixels had noise added, so only they need denoising
- Imputed pixels flow through the network but don't affect training

**Key Insight**: The architecture is NOT masked (no masked convolutions), but the TRAINING SIGNAL is masked. This allows:
- ✅ Network can use imputed values as context (sees full neighborhoods)
- ✅ Network is not penalized for errors on imputed values
- ✅ Gradient updates only come from real observations
- ✅ Natural neighborhoods (no zeros) for convolution kernels

This is fundamentally different from:
- ❌ **Simple masking**: Would use zeros, creating unnatural neighborhoods
- ❌ **Architecture masking**: Would require custom masked convolutions
- ✅ **Loss masking**: Clean architecture, masked training signal

**Visual Representation**:
```
Input Image (with TST-completed values):
[Real=5] [Real=3] [Imputed=4] [Real=7]
    ↓        ↓          ↓          ↓
Add Noise (masked):
[5+noise] [3+noise] [4+0] [7+noise]
    ↓        ↓          ↓          ↓
U-Net Denoising (sees all, predicts all):
[pred=5.1] [pred=2.9] [pred=4.2] [pred=6.8]
    ↓        ↓          ↓          ↓
Apply Mask to Loss:
[loss=0.01] [loss=0.01] [loss=0.0] [loss=0.04]
                                     ↑
                              No gradient here!
```

### 4.2 Time Series Transformer (`models/TST.py`)

#### TSTransformerEncoder (Lines 153-206)

This is the core completion model based on the transformer architecture.

**Architecture**:
```python
class TSTransformerEncoder(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, 
                 dim_feedforward, dropout, pos_encoding, activation, norm):
        # 1. Input projection: feat_dim → d_model
        self.project_inp = nn.Linear(feat_dim, d_model)
        
        # 2. Positional encoding (fixed sinusoidal or learnable)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout, max_len)
        
        # 3. Transformer encoder stack
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 4. Output projection
        self.output_layer = nn.Linear(d_model, d_model)
```

**Forward Pass** (Lines 181-206):
```python
def forward(self, X, padding_masks):
    # X: (batch_size, seq_length, feat_dim)
    # padding_masks: (batch_size, seq_length) - True = keep, False = padding
    
    # 1. Replace NaNs with 0
    masked_X = X.clone()
    masked_X[torch.isnan(masked_X)] = 0
    
    # 2. Permute to (seq_length, batch_size, feat_dim)
    inp = masked_X.permute(1, 0, 2)
    
    # 3. Project and scale
    inp = self.project_inp(inp) * math.sqrt(self.d_model)
    
    # 4. Add positional encoding
    inp = self.pos_enc(inp)
    
    # 5. Transformer encoding with padding mask
    output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
    
    # 6. Apply activation and project
    output = self.act(output)
    output = output.permute(1, 0, 2)
    output = self.output_layer(output)
    
    return output  # (batch_size, seq_length, d_model)
```

**Key Features**:
- **Padding masks**: Tell the transformer which positions are valid
- **Self-attention**: Captures long-range dependencies
- **Positional encoding**: Preserves temporal order information

#### TST_Decoder (`models/decoder.py`)

Simple GRU-based decoder:
```python
class TST_Decoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, layers, args):
        # Bidirectional GRU
        self.rnn = nn.GRU(input_size=inp_dim, 
                         hidden_size=hidden_dim,
                         bidirectional=True,
                         num_layers=layers,
                         batch_first=True)
        
        # Linear projection to time series dimension
        self.linear = nn.Linear(hidden_dim * 2, args.input_channels)
    
    def forward(self, z):
        h, _ = self.rnn(z)
        x_hat = F.sigmoid(self.linear(h))
        return x_hat
```

**Why GRU?**: 
- Captures sequential patterns
- Bidirectional → uses both past and future context
- Sigmoid output → ensures values in [0,1]

### 4.3 Delay Embedding Transformer (`models/img_transformations.py`)

This component converts time series to images and back.

#### Delay Embedding Theory

Given a time series `x[1:T]`, delay embedding creates an image by:
1. Taking overlapping windows of size `embedding`
2. Shifting each window by `delay` steps
3. Stacking windows as image columns

**Example**:
```
Time series: [1, 2, 3, 4, 5, 6, 7, 8]
embedding=3, delay=2

Column 0: [1, 2, 3]
Column 1: [3, 4, 5]  (start = 0*2 + 2)
Column 2: [5, 6, 7]  (start = 1*2 + 2)
Column 3: [7, 8, ?]  (partial)

Image (3 x 4):
[[1, 3, 5, 7],
 [2, 4, 6, 8],
 [3, 5, 7, ?]]
```

#### Implementation

**`ts_to_img()` (Lines 79-108)**:
```python
def ts_to_img(self, signal, pad=True, mask=0):
    batch, length, features = signal.shape
    
    # Initialize image
    x_image = torch.zeros((batch, features, self.embedding, self.embedding))
    
    # Fill columns
    i = 0
    while (i * self.delay + self.embedding) <= self.seq_len:
        start = i * self.delay
        end = start + self.embedding
        x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
        i += 1
    
    # Handle partial column at end
    if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
        start = i * self.delay
        end = signal[:, start:].permute(0, 2, 1).shape[-1]
        x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
        i += 1
    
    # Cache shape and crop
    self.img_shape = (batch, features, self.embedding, i)
    x_image = x_image[:, :, :, :i]
    
    # Pad to square
    if pad:
        x_image = self.pad_to_square(x_image, mask)
    
    return x_image
```

**`img_to_ts()` (Lines 140-170)**:
```python
def img_to_ts(self, x_image_square):
    # Unpad to original shape
    x_image_non_square = self.unpad(x_image_square, self.img_shape)
    batch_size, channels, rows, cols = x_image_non_square.shape
    
    # Initialize output
    reconstructed_ts = torch.zeros((batch_size, self.seq_len, channels))
    
    # Use pre-computed mapping
    for ts_idx in range(self.seq_len):
        image_positions = self.mapping.get(ts_idx, [])
        if not image_positions:
            continue
        
        # Collect all image values that map to this time point
        values = []
        for row, col in image_positions:
            value = x_image_non_square[:, :, row, col]
            values.append(value)
        
        # Average them (KEY IMPROVEMENT over original ImagenTime)
        values_tensor = torch.stack(values, dim=0)
        mean_values = values_tensor.mean(dim=0)
        reconstructed_ts[:, ts_idx, :] = mean_values
    
    return reconstructed_ts.permute(0, 1, 2)
```

**Key Innovation**: The paper improves on ImagenTime by averaging all pixel values that correspond to each time point, rather than just taking the first one. This uses more information and reduces reconstruction error.

#### Mapping Creation (`create_mapping()`, Lines 111-138)

Creates a mapping `{time_index: [(row, col), ...]}` showing which image pixels correspond to each time point:

```python
def create_mapping(self):
    # Create helper series with unique values 1, 2, ..., seq_len
    helper_series = torch.arange(1, self.seq_len + 1)
    helper_series = helper_series.unsqueeze(0).unsqueeze(-1)
    helper_series = helper_series.repeat(batch_size, 1, num_features)
    
    # Convert to image
    helper_image = self.ts_to_img(helper_series, pad=True, mask=0)
    helper_image_non_square = self.unpad(helper_image, self.img_shape)
    
    # Build mapping
    self.mapping = {}
    for row in range(rows):
        for col in range(cols):
            val = helper_image_non_square[0, 0, row, col].item()
            if val != 0:  # Ignore padding
                ts_idx = int(val) - 1
                if ts_idx not in self.mapping:
                    self.mapping[ts_idx] = []
                self.mapping[ts_idx].append((row, col))
```

### 4.4 U-Net Diffusion Backbone (`models/networks.py`)

This file contains the diffusion model architecture from NVIDIA's EDM paper.

#### EDMPrecond (Lines 632-672)

Wraps the U-Net with preconditioning for improved training:

```python
class EDMPrecond(torch.nn.Module):
    def __init__(self, img_resolution, img_channels, label_dim, use_fp16,
                 sigma_min, sigma_max, sigma_data, model_type, **model_kwargs):
        
        # Model parameters
        self.sigma_data = sigma_data  # Expected data std (typically 0.5)
        self.sigma_min = sigma_min    # Min noise level
        self.sigma_max = sigma_max    # Max noise level
        
        # The actual U-Net
        self.model = globals()[model_type](...)
    
    def forward(self, x, sigma, class_labels=None, force_fp32=False):
        # Compute preconditioning coefficients
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        # Forward through network
        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), 
                        class_labels=class_labels)
        
        # Preconditioned output
        D_x = c_skip * x + c_out * F_x
        return D_x
```

**Why Preconditioning?**:
- `c_in`: Normalizes input magnitude across noise levels
- `c_out`, `c_skip`: Balances contribution of network output vs. noisy input
- `c_noise`: Conditions network on noise level
- Results in more stable training and better sample quality

#### DhariwalUNet (Lines 372-461)

The actual U-Net architecture (ADM-style):

**Structure**:
```
Input Image (C x H x W)
    ↓
Positional Embedding (time step)
    ↓
Encoder:
  - Conv blocks with downsampling
  - Self-attention at specified resolutions
  - Channel doubling at each level
    ↓
Bottleneck:
  - Self-attention
  - Residual blocks
    ↓
Decoder:
  - Conv blocks with upsampling
  - Skip connections from encoder
  - Self-attention at specified resolutions
    ↓
Output Conv
    ↓
Output Image (C x H x W)
```

**Key Components**:

1. **UNetBlock** (Lines 134-187): Residual block with optional attention
```python
class UNetBlock(torch.nn.Module):
    def forward(self, x, emb):
        # 1. First conv + groupnorm
        x = self.conv0(silu(self.norm0(x)))
        
        # 2. Add time embedding
        params = self.affine(emb).unsqueeze(2).unsqueeze(3)
        scale, shift = params.chunk(chunks=2, dim=1)
        x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        
        # 3. Second conv + dropout
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        
        # 4. Skip connection
        x = x + self.skip(orig)
        x = x * self.skip_scale
        
        # 5. Self-attention (if specified)
        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(...).unbind(2)
            w = AttentionOp.apply(q, k)  # Compute attention weights
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)) + x
            x = x * self.skip_scale
        
        return x
```

2. **GroupNorm** (Lines 96-106): Normalizes features in groups
3. **AttentionOp** (Lines 113-126): Custom attention with FP32 precision

### 4.5 Sampling Process (`models/sampler.py`)

Implements the Karras ODE sampler for high-quality generation.

#### DiffusionProcess (Lines 4-72)

**Key Method: `sample()` (Lines 32-65)**:

```python
def sample(self, latents, class_labels=None):
    sigma_min = max(self.sigma_min, self.net.sigma_min)
    sigma_max = min(self.sigma_max, self.net.sigma_max)
    
    # 1. Create noise schedule (high to low)
    step_indices = torch.arange(self.num_steps)
    t_steps = (sigma_max ** (1/rho) + step_indices/(num_steps-1) * 
              (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    
    # 2. Initialize with pure noise
    x_next = latents * t_steps[0]
    
    # 3. Iterative denoising
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Stochastic noise injection (S_churn)
        gamma = min(S_churn / num_steps, sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        
        # Euler step
        denoised = self.net(x_hat, t_hat, class_labels)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # Heun's 2nd order correction
        if i < num_steps - 1:
            denoised = self.net(x_next, t_next, class_labels)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    
    return x_next
```

**Algorithm**: 
- **Euler method**: First-order ODE solver
- **Heun's method**: Second-order correction for better accuracy
- **Stochastic injection**: Optional noise for stochastic sampling

### 4.6 Exponential Moving Average (`models/ema.py`)

EMA maintains a smoothed version of model weights for better inference.

```python
class LitEma(nn.Module):
    def forward(self, model):
        decay = self.decay
        
        # Decay schedule: starts at 0, approaches self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        one_minus_decay = 1.0 - decay
        
        # Update shadow parameters
        for key in m_param:
            if m_param[key].requires_grad:
                sname = self.m_name2s_name[key]
                if self.num_updates > self.warmup:
                    # EMA update: shadow = decay * shadow + (1-decay) * param
                    shadow_params[sname].sub_(one_minus_decay * 
                                              (shadow_params[sname] - m_param[key]))
                else:
                    # During warmup: just copy
                    shadow_params[sname].copy_(m_param[key])
```

**Benefits**:
- Reduces variance in training
- Often gives better validation/test performance
- Minimal overhead (just extra memory for shadow params)

---

## 5. Data Processing

### 5.1 Data Loading (`utils/utils_data.py`)

#### Dataset Types

1. **Synthetic Sine Data** (`sine_data_generation`, Lines 78-122):
```python
def sine_data_generation(no, seq_len, dim, missing_rate):
    ori_data = []
    irregular_dataset = []
    
    for i in range(no):
        temp = []
        for k in range(dim):
            # Random frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)
            
            # Generate sine wave
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)
        
        # Normalize to [0,1]
        temp = (np.transpose(np.array(temp)) + 1) * 0.5
        ori_data.append(temp.copy())
        
        # Create irregular version
        removed_points = torch.randperm(seq_len)[:int(seq_len * missing_rate)]
        temp[removed_points] = float('nan')
        
        # Add time index column
        idx = np.array(range(seq_len)).reshape(-1, 1)
        temp = np.concatenate((temp, idx), axis=1)
        irregular_dataset.append(temp)
    
    return ori_data, irregular_dataset
```

2. **Real-world Data** (`real_data_loading`, Lines 141-212):
```python
def real_data_loading(data_name, seq_len, missing_rate, gaussian_noise_level=0):
    # Load CSV
    if data_name == 'stock':
        ori_data = np.loadtxt('./data/stock.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('./data/energy.csv', delimiter=",", skiprows=1)
    # ... etc
    
    # Flip chronologically
    ori_data = ori_data[::-1]
    
    # Normalize
    ori_data = MinMaxScaler(ori_data)
    
    # Create irregular version
    irregular_dataset = ori_data.copy()
    
    # Optional: Add Gaussian noise
    if gaussian_noise_level > 0:
        irregular_dataset = add_gaussian_noise(ori_data, gaussian_noise_level)
    
    # Random missing values
    removed_points = torch.randperm(ori_data.shape[0])[
                     :int(ori_data.shape[0] * missing_rate)]
    irregular_dataset[removed_points] = float('nan')
    
    # Cut into sequences
    temp_data = []
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
    
    # Shuffle
    idx = np.random.permutation(len(temp_data))
    data = [temp_data[idx[i]] for i in range(len(temp_data))]
    
    # Same for irregular
    irregular_data = [irregular_temp_data[idx[i]] for i in range(len(irregular_temp_data))]
    
    return data, irregular_data
```

3. **MuJoCo Dataset** (`MujocoDataset`, Lines 48-76):
```python
class MujocoDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, data_name, missing_rate=0.0):
        # Load pre-saved tensors
        tensors = load_data(base_loc)
        self.samples = tensors['mujoco_irregular']
        self.original_sample = tensors['mujoco_regular']
        
        # Apply missing rate
        for i in range(len(self.samples)):
            removed_points = torch.randperm(self.samples[i].shape[0])[
                            :int(self.samples[i].shape[0] * missing_rate)]
            self.samples[i][removed_points, :-1] = float('nan')
```

#### DataLoader Creation (`gen_dataloader`, Lines 215-249)

```python
def gen_dataloader(args):
    if args.dataset == 'sine':
        ori_data, irregular_data = sine_data_generation(...)
        ori_data = torch.Tensor(np.array(ori_data))
        irregular_data = torch.Tensor(np.array(irregular_data))
    
    elif args.dataset in ['stock', 'energy', ...]:
        ori_data, irregular_data = real_data_loading(...)
        ori_data = torch.Tensor(np.array(ori_data))
        irregular_data = torch.Tensor(np.array(irregular_data))
    
    elif args.dataset == 'mujoco':
        train_set = MujocoDataset(...)
        # Extract data from dataset
        ori_data = ...
        irregular_data = ...
    
    # Create DataLoaders
    train_loader = Data.DataLoader(dataset=irregular_train_set, 
                                   batch_size=args.batch_size, 
                                   shuffle=True,
                                   num_workers=args.num_workers)
    
    test_loader = Data.DataLoader(dataset=ori_train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)
    
    return train_loader, test_loader, None
```

### 5.2 Data Preprocessing

#### MinMaxScaler (Lines 10-26)
```python
def MinMaxScaler(data, return_scalers=False):
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    
    if return_scalers:
        return norm_data, min, max
    return norm_data
```

Normalizes each feature independently to [0, 1] range.

---

## 6. Evaluation Metrics

### 6.1 Discriminative Score (`metrics/discriminative_torch.py`)

Measures how easily real and synthetic data can be distinguished.

```python
def discriminative_score_metrics(ori_data, generated_data, args):
    # Define discriminator
    class Discriminator(nn.Module):
        def __init__(self, inp_dim, hidden_dim):
            self.rnn = nn.GRU(input_size=inp_dim, 
                             hidden_size=hidden_dim,
                             bidirectional=False,
                             num_layers=1,
                             batch_first=True)
            self.linear = nn.Linear(hidden_dim, 1)
        
        def forward(self, x):
            _, last_hidden = self.rnn(x)
            y_hat_logit = self.linear(last_hidden)
            y_hat = sigmoid(y_hat_logit)
            return y_hat_logit, y_hat
    
    model = Discriminator(args.input_size, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Split data
    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(ori_data, generated_data)
    
    # Train discriminator
    for itt in range(2000):
        X_mb = batch_generator(train_x, 32)
        X_hat_mb = batch_generator(train_x_hat, 32)
        
        y_logit_real, y_pred_real = model(X_mb)
        y_logit_fake, y_pred_fake = model(X_hat_mb)
        
        d_loss_real = binary_cross_entropy_with_logits(y_logit_real, ones)
        d_loss_fake = binary_cross_entropy_with_logits(y_logit_fake, zeros)
        
        d_loss = d_loss_real + d_loss_fake
        optimizer.step()
    
    # Test
    y_pred_real_curr = model(test_x)
    y_pred_fake_curr = model(test_x_hat)
    
    # Compute accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = abs(0.5 - acc)
    
    return discriminative_score
```

**Interpretation**:
- Score = |0.5 - accuracy|
- Lower is better
- 0.5 = perfect confusion (can't tell real from fake)
- 0.0 = perfect discrimination (very bad synthetic data)

### 6.2 Predictive Score (`metrics/predictive_metrics.py`)

Measures usefulness for downstream tasks using "train on synthetic, test on real" protocol.

```python
def predictive_score_metrics(ori_data, generated_data):
    # Define predictor
    class GRUPredictorIrregular(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            self.gru = nn.GRU(input_size=input_dim,
                             hidden_size=hidden_dim,
                             batch_first=True)
            self.dense = nn.Linear(hidden_dim, input_dim)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, X, T):
            # Pack sequence
            packed_X = pack_padded_sequence(X, lengths=T, 
                                           batch_first=True,
                                           enforce_sorted=False)
            p_outputs, _ = self.gru(packed_X)
            p_outputs, _ = pad_packed_sequence(p_outputs, batch_first=True)
            
            y_hat_logit = self.dense(p_outputs)
            y_hat = self.sigmoid(y_hat_logit)
            return y_hat
    
    predictor_model = GRUPredictorIrregular(dim, hidden_dim)
    p_solver = optim.Adam(predictor_model.parameters())
    
    # Train on generated data
    for itt in range(iterations):
        X_mb = [generated_data[i][:-1] for i in train_idx]
        Y_mb = [generated_data[i][1:] for i in train_idx]
        T_mb = [generated_time[i] - 1 for i in train_idx]
        
        y_hat = predictor_model(X_tensor, T_mb)
        p_loss = F.l1_loss(y_hat, Y_tensor)
        
        p_solver.step()
    
    # Test on original data
    pred_Y_curr = []
    for i in range(no):
        x_i = X_mb_test[i].unsqueeze(0)
        t_i = [T_mb_test[i]]
        y_hat_i = predictor_model(x_i, t_i).squeeze(0)
        pred_Y_curr.append(y_hat_i.numpy())
    
    # Compute MAE
    MAE_temp = sum(mean_absolute_error(Y_mb_test[i], pred_Y_curr[i]) 
                   for i in range(no))
    predictive_score = MAE_temp / no
    
    return predictive_score
```

**Interpretation**:
- Lower MAE is better
- Measures how well a model trained on synthetic data can predict real data
- Tests transfer learning capability

### 6.3 Context-FID Score (`metrics/context_fid.py`)

Measures distributional similarity using learned representations.

```python
def Context_FID(ori_data, generated_data):
    # 1. Train TS2Vec encoder on real data
    model = TS2Vec(input_dims=ori_data.shape[-1], 
                  device=0, 
                  batch_size=8,
                  lr=0.001,
                  output_dims=320,
                  max_train_length=3000)
    model.fit(ori_data, verbose=False)
    
    # 2. Encode both datasets
    ori_representation = model.encode(ori_data, encoding_window='full_series')
    gen_representation = model.encode(generated_data, encoding_window='full_series')
    
    # 3. Shuffle
    idx = np.random.permutation(ori_data.shape[0])
    ori_representation = ori_representation[idx]
    gen_representation = gen_representation[idx]
    
    # 4. Compute FID
    fid_score = calculate_fid_score(ori_representation, gen_representation)
    
    return fid_score

def calculate_fid_score(act1, act2):
    # Calculate statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Calculate sqrt of product between covariances
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid
```

**Interpretation**:
- FID = Fréchet Inception Distance
- Measures distance between two multivariate Gaussians
- Lower is better
- 0 = identical distributions

**TS2Vec Encoder**:
- Self-supervised contrastive learning on time series
- Creates meaningful representations
- Captures temporal and cross-feature dependencies

### 6.4 Correlation Score (`metrics/correlation_score.py`)

Measures cross-correlation similarity between real and synthetic data.

```python
def calculate_pearson_correlation(real_sig, gen_sig):
    x_real = torch.from_numpy(real_sig)
    x_fake = torch.from_numpy(gen_sig)
    
    correlational_score = []
    size = int(x_real.shape[0] / iterations)
    
    for i in range(iterations):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        
        corr = CrossCorrelLoss(x_real[real_idx], name='CrossCorrelLoss')
        loss = corr.compute(x_fake[fake_idx])
        correlational_score.append(loss.item())
    
    mean, std, conf_interval = display_scores(correlational_score)
    return mean, std, conf_interval

class CrossCorrelLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super().__init__(**kwargs)
        # Compute cross-correlation of real data
        self.cross_correl_real = cacf_torch(self.transform(x_real), 1).mean(0)[0]
    
    def compute(self, x_fake):
        # Compute cross-correlation of fake data
        cross_correl_fake = cacf_torch(self.transform(x_fake), 1).mean(0)[0]
        
        # L1 distance
        loss = torch.abs(cross_correl_fake - self.cross_correl_real).sum(0)
        return loss / 10.

def cacf_torch(x, max_lag, dim=(0, 1)):
    # Get lower triangular indices (all feature pairs)
    ind = get_lower_triangular_indices(x.shape[2])
    
    # Standardize
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    
    cacf_list = []
    for i in range(max_lag):
        # Lag-i cross-correlation
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)
    
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))
```

**Interpretation**:
- Measures cross-feature correlations
- Lower is better
- Captures multivariate dependencies

### 6.5 Evaluation Orchestration (`metrics/__init__.py`)

```python
def evaluate_model_irregular(real_sig, gen_sig, args, calc_other_metrics=False):
    metric_iteration = 10
    
    # Always compute discriminative score
    disc_res = []
    for _ in range(metric_iteration):
        dsc = discriminative_score_metrics(real_sig, gen_sig, args)
        disc_res.append(dsc)
    disc_mean, disc_std = np.round(np.mean(disc_res), 4), np.round(np.std(disc_res), 4)
    
    if calc_other_metrics:
        # Compute predictive score
        predictive_score = []
        for _ in range(metric_iteration):
            temp_pred = predictive_score_metrics(real_sig, gen_sig)
            predictive_score.append(temp_pred)
        pred_mean, pred_std = np.round(np.mean(predictive_score), 4), ...
        
        # Compute FID
        fid_mean, fid_std, fid_conf_interval = calculate_fid(real_sig, gen_sig)
        
        # Compute correlation
        correlation_score_mean, correlation_score_std, correlation_score_conf_interval = \
            calculate_pearson_correlation(real_sig, gen_sig)
        
        return {
            'pred_score_mean': pred_mean,
            'pred_score_std': pred_std,
            'fid_score_mean': fid_mean,
            'fid_score_std': fid_std,
            'fid_score_conf_interval': fid_conf_interval,
            'correlation_score_mean': correlation_score_mean,
            'correlation_score_std': correlation_score_std,
            'correlation_score_conf_interval': correlation_score_conf_interval
        }
    else:
        return {'disc_mean': disc_mean, 'disc_std': disc_std}
```

**Strategy**:
- Always compute discriminative score (fast)
- Only compute expensive metrics (predictive, FID, correlation) when there's improvement
- Run multiple iterations for statistical significance

---

## 7. Utilities and Logging

### 7.1 Argument Parsing (`utils/utils_args.py`)

Uses OmegaConf for flexible configuration:

```python
def parse_args_irregular():
    parser = argparse.ArgumentParser()
    
    # General args
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--missing_rate', type=float, default=0.3)
    
    # Diffusion args
    parser.add_argument('--beta1', type=float, default=1e-5)
    parser.add_argument('--betaT', type=float, default=1e-2)
    parser.add_argument('--deterministic', action='store_true', default=False)
    
    # Config file (contains most settings)
    parser.add_argument('--config', type=str, 
                       default='./configs/seq_len_24/stock.yaml')
    
    # Training args
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    
    # Data args
    parser.add_argument('--dataset', 
                       choices=['sine', 'energy', 'mujoco', 'stock', ...])
    parser.add_argument('--seq_len', type=int)
    
    # Model args
    parser.add_argument('--img_resolution', type=int)
    parser.add_argument('--input_channels', type=int)
    parser.add_argument('--unet_channels', type=int)
    parser.add_argument('--ch_mult', type=int, nargs='+')
    parser.add_argument('--attn_resolution', type=int, nargs='+')
    parser.add_argument('--diffusion_steps', type=int)
    
    # TST args
    parser.add_argument('--hidden_dim', type=int, default=40)
    parser.add_argument('--n_heads', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pos_encoding', choices=['fixed', 'learnable'])
    parser.add_argument('--activation', choices=['relu', 'gelu'])
    parser.add_argument('--norm', choices=['BatchNorm', 'LayerNorm'])
    
    parsed_args = parser.parse_args()
    
    # Load config file
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    
    # Override with CLI args
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    
    # Add configs not in parsed args
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)
    
    parsed_args.input_size = parsed_args.input_channels
    return parsed_args
```

**Design Philosophy**:
- Config file contains defaults
- CLI args override config file
- Allows easy experimentation while maintaining reproducibility

### 7.2 Logger Abstraction (`utils/loggers/`)

#### Base Logger (`base_logger.py`)

Abstract interface:
```python
class BaseLogger(ABC):
    @abstractmethod
    def log(self, name: str, data: Any, step=None):
        pass
    
    @abstractmethod
    def log_fig(self, name: str, fig: Any):
        pass
    
    @abstractmethod
    def log_hparams(self, params: Dict[str, Any]):
        pass
    
    @abstractmethod
    def add_tags(self, tags: List[str]):
        pass
    
    @abstractmethod
    def log_name_params(self, name: str, params: Any):
        pass
```

#### Neptune Logger (`neptune_logger.py`)

Cloud-based experiment tracking:
```python
class NeptuneLogger(BaseLogger):
    def __init__(self, project=None):
        # Read API token
        api_token = Path('neptune/token.txt').read_text().strip()
        project = Path('neptune/project.txt').read_text().strip()
        
        # Initialize Neptune run
        self.run = neptune.init_run(project=project, api_token=api_token)
    
    def log(self, name: str, data: Any, step=None):
        self.run[name].append(data)
    
    def log_fig(self, name: str, fig: Any):
        if isinstance(fig, np.ndarray):
            fig = Image.fromarray(fig)
        self.run[name].append(fig)
    
    def log_hparams(self, params: Dict[str, Any]):
        params = convert_no_basic_to_str(params)
        self.run['hyperparameters'] = params
    
    def add_tags(self, tags: List[str]):
        self.run['sys/tags'].add(tags)
```

#### Composite Logger (`composite_logger.py`)

Logs to multiple backends simultaneously:
```python
class CompositeLogger(BaseLogger):
    def __init__(self, loggers):
        self.loggers = loggers
    
    def log(self, name: str, data: Any, step=None):
        for logger in self.loggers:
            logger.log(name, data, step)
    
    # Similar for other methods...
```

**Usage**:
```python
# Log to both Neptune and console
with CompositeLogger([NeptuneLogger(), PrintLogger()]) as logger:
    logger.log('train/loss', loss.item(), epoch)
```

### 7.3 Model Utilities (`utils/utils.py`)

**`create_model_name_and_dir()`** (Lines 11-24):
```python
def create_model_name_and_dir(args):
    name = (f'conditional-'
            f'bs={args.batch_size}-'
            f'-lr={args.learning_rate:.4f}-'
            f'ch_mult={args.ch_mult}-'
            f'attn_res={args.attn_resolution}-'
            f'unet_ch={args.unet_channels}'
            f'-delay={args.delay}-{args.embedding}')
    
    args.log_dir = f'{args.log_dir}/{args.dataset}/{name}'
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    return name
```

Creates unique directory for each hyperparameter configuration.

**`print_model_params()`** (Lines 33-37):
```python
def print_model_params(logger, model):
    params_num = sum(param.numel() for param in model.parameters())
    logging.info(f"number of model parameters: {params_num}")
    logger.log_name_params('config/params_num', params_num)
```

Logs total parameter count for model size tracking.

### 7.4 Persistence (`utils/persistence.py`)

Based on NVIDIA's pickling system for saving model source code alongside weights.

**Key Decorator: `@persistent_class`** (Lines 35-138):
```python
def persistent_class(orig_class):
    """
    Class decorator that extends a given class to save its source code
    when pickled. This ensures previously exported pickles remain usable
    even if the code changes.
    """
    assert orig_class.__module__ in sys.modules
    orig_module = sys.modules[orig_class.__module__]
    orig_module_src = inspect.getsource(orig_module)
    
    class Decorator(orig_class):
        _orig_module_src = orig_module_src
        _orig_class_name = orig_class.__name__
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Record constructor arguments
            self._init_args = copy.deepcopy(args)
            self._init_kwargs = copy.deepcopy(kwargs)
        
        @property
        def init_args(self):
            return copy.deepcopy(self._init_args)
        
        @property
        def init_kwargs(self):
            return dnnlib.EasyDict(copy.deepcopy(self._init_kwargs))
        
        def __reduce__(self):
            # Custom pickling: save source code + state
            fields = list(super().__reduce__())
            meta = dict(
                type='class',
                version=_version,
                module_src=self._orig_module_src,
                class_name=self._orig_class_name,
                state=fields[2]
            )
            fields[0] = _reconstruct_persistent_obj
            fields[1] = (meta,)
            fields[2] = None
            return tuple(fields)
    
    return Decorator
```

**Benefits**:
- Pickles remain usable even after code changes
- Stores constructor arguments for re-instantiation
- Enables reproducibility across code versions

---

## 8. Configuration System

### 8.1 Config File Structure

Example: `configs/seq_len_24/stock.yaml`

```yaml
# Training parameters
epochs: 1000
batch_size: 128
first_epoch: 30  # TST pre-training epochs
learning_rate: 0.0001
weight_decay: 0.00001

# Data
dataset: stock
seq_len: 24

# Image transformation
delay: 3
embedding: 8

# Diffusion model
img_resolution: 8
attn_resolution: [8, 4, 2]
input_channels: 6
unet_channels: 128
ch_mult: [1, 2, 2, 2]
diffusion_steps: 18
ema: true
ema_warmup: 100

# Logging
logging_iter: 10

# TST
input_size: 6
```

### 8.2 Dataset-Specific Configurations

Different sequence lengths require different image resolutions:

#### Short Sequences (seq_len=24)
```yaml
delay: 3
embedding: 8
img_resolution: 8
# Results in 8x8 image
```

#### Medium Sequences (seq_len=96)
```yaml
delay: 6
embedding: 16
img_resolution: 16
# Results in 16x16 image
```

#### Long Sequences (seq_len=768)
```yaml
delay: 12
embedding: 64
img_resolution: 64
# Results in 64x64 image
```

**Design Principle**: Larger sequences → larger images → more U-Net capacity

### 8.3 Key Hyperparameters

#### Image Transformation
- **`delay`**: Stride between consecutive image columns
- **`embedding`**: Height of each column (window size)
- **Trade-off**: 
  - Small delay + small embedding = tall narrow image
  - Large delay + large embedding = short wide image

#### U-Net Architecture
- **`unet_channels`**: Base channel count (typically 128 or 192)
- **`ch_mult`**: Channel multipliers per resolution level
  - Example: `[1, 2, 2, 2]` → 128, 256, 256, 256 channels
- **`attn_resolution`**: Which resolutions get self-attention
  - Example: `[8, 4, 2]` → attention at 8×8, 4×4, 2×2
  - Higher resolutions = more attention but slower

#### Diffusion Process
- **`diffusion_steps`**: Number of denoising steps
  - More steps = higher quality but slower
  - Typical range: 10-50 for training, 18-40 for this work
- **`beta1`, `betaT`**: Noise schedule endpoints
  - Controls signal-to-noise ratio over time

#### TST Architecture
- **`hidden_dim`**: Dimension of transformer embeddings (typically 40-64)
- **`n_heads`**: Number of attention heads (typically 4-8)
- **`num_layers`**: Number of transformer layers (typically 4-8)
- **`dim_feedforward`**: FFN dimension (typically 2048)
- **`pos_encoding`**: 'fixed' (sinusoidal) or 'learnable'
- **`norm`**: 'BatchNorm' or 'LayerNorm'

#### Training
- **`first_epoch`**: TST pre-training epochs (typically 20-50)
- **`epochs`**: Total training epochs (typically 500-2000)
- **`batch_size`**: Batch size (adjust based on GPU memory)
- **`learning_rate`**: Learning rate for diffusion model (typically 1e-4)
- **`ema`**: Use EMA for inference (almost always True)
- **`ema_warmup`**: Steps before EMA starts (typically 100-1000)

---

## 9. Implementation Highlights and Best Practices

### 9.1 Key Innovations

1. **Masked Diffusion Loss**:
   - Only compute loss on real (observed) pixels
   - Prevents model from trusting imputed values too much
   - Noise is also masked: `masked_noise = n * mask`

2. **Averaged Inverse Delay Embedding**:
   - Original ImagenTime used first pixel only
   - This implementation averages all corresponding pixels
   - Reduces reconstruction error

3. **Two-Phase Training**:
   - Phase 1: Pre-train TST (first_epoch iterations)
   - Phase 2: Joint training of diffusion + TST
   - Ensures TST is good enough before diffusion training starts

4. **Natural Neighborhoods via TST**:
   - TST fills missing values with plausible estimates
   - Creates smooth, realistic image neighborhoods
   - Much better than zero-filling for convolution kernels

### 9.2 Design Decisions

#### Why Delay Embedding?
- **Pros**:
  - Geometric transformation (interpretable)
  - Preserves local temporal structure
  - Natural fit for convolutional networks
  - Efficient for long sequences
- **Cons**:
  - Overlapping values create redundancy
  - Requires careful inverse transformation

#### Why TST for Completion?
- **Pros**:
  - Captures long-range dependencies
  - Parallel processing (faster than RNN)
  - Flexible attention patterns
  - No gradient vanishing
- **Cons**:
  - More parameters than simple methods
  - Requires pre-training

#### Why Karras Diffusion (EDM)?
- **Pros**:
  - State-of-the-art image generation
  - Principled noise schedule
  - Stable training
  - High sample quality
- **Cons**:
  - More complex than DDPM
  - Requires careful tuning

#### Why EMA?
- **Pros**:
  - Smooths out training noise
  - Often better validation performance
  - Negligible computational cost
- **Cons**:
  - Extra memory (2x model size)
  - Slightly delayed updates

### 9.3 Common Pitfalls and Solutions

#### 1. NaN Handling
**Problem**: NaNs can propagate through the network
**Solution**: 
```python
# Always fill NaNs before network input
masked_X = X.clone()
masked_X[torch.isnan(masked_X)] = 0
```

#### 2. Padding Mask Confusion
**Problem**: Different libraries use opposite conventions
**Solution**:
```python
# PyTorch: True = keep, False = mask
# Some others: False = keep, True = mask
# Always check documentation!
padding_masks = ~torch.isnan(x).any(dim=-1)  # True = valid position
output = transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # Note the ~
```

#### 3. Memory Issues with Long Sequences
**Problem**: Long sequences create large images → OOM
**Solution**:
- Use gradient checkpointing
- Reduce batch size
- Use mixed precision (FP16)
```python
# In training loop
with torch.cuda.amp.autocast():
    loss = model.loss_fn_irregular(x_tilde_img, mask)
```

#### 4. Slow Evaluation
**Problem**: Computing all metrics is expensive
**Solution**:
- Only compute disc score every epoch
- Only compute other metrics when disc score improves
```python
if curr_disc_score < best_disc_score:
    new_scores = evaluate_model_irregular(real_sig, gen_sig, args, 
                                          calc_other_metrics=True)
```

#### 5. Checkpoint Management
**Problem**: Checkpoints are large and can fill disk
**Solution**:
- Only keep best checkpoint
- Delete old checkpoints before saving new ones
```python
# In save_checkpoint()
for f in glob.glob(os.path.join(full_path, "*")):
    os.remove(f)  # Clear old checkpoints
```

### 9.4 Performance Optimization Tips

1. **Data Loading**:
   ```python
   # Use multiple workers
   train_loader = DataLoader(..., num_workers=4, pin_memory=True)
   ```

2. **GPU Utilization**:
   ```python
   # Use larger batch sizes if memory allows
   # Profile to find optimal batch size
   torch.cuda.synchronize()  # For accurate timing
   ```

3. **Gradient Accumulation** (for large effective batch size):
   ```python
   accumulation_steps = 4
   for i, data in enumerate(train_loader):
       loss = model.loss_fn(...) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **Mixed Precision Training**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       loss = model.loss_fn(...)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

### 9.5 Hyperparameter Tuning Guidelines

#### Critical Parameters (tune first):
1. **learning_rate**: Start with 1e-4, try 5e-5, 2e-4
2. **first_epoch**: Should be enough for TST convergence (20-50)
3. **diffusion_steps**: Balance quality vs. speed (15-25)
4. **batch_size**: As large as GPU memory allows

#### Secondary Parameters:
5. **delay/embedding**: Match to sequence length
6. **unet_channels**: 128 for short, 192 for long sequences
7. **n_heads**: 4-8, ensure d_model % n_heads == 0
8. **num_layers**: 4-8, more for longer sequences

#### Fine-tuning Parameters:
9. **dropout**: 0.0-0.2, higher for overfitting
10. **weight_decay**: 1e-5 to 1e-4
11. **ema_warmup**: 100-1000
12. **logging_iter**: 10-50

### 9.6 Debugging Strategies

#### Check TST Pre-training:
```python
print(f"TST loss: {np.sqrt(loss_e_t0.item()):.4f}")
# Should decrease to < 0.1 during first_epoch
```

#### Check Diffusion Training:
```python
logger.log('train/karras_loss', loss.item(), epoch)
# Should stabilize after ~100 epochs
# Typical range: 0.1 - 1.0
```

#### Visualize Generated Samples:
```python
# After training
x_img_sampled = process.sampling(sampling_number=16)
x_ts = model.img_to_ts(x_img_sampled)
# Plot x_ts vs. real data
```

#### Check for Mode Collapse:
```python
# Generate many samples
gen_samples = [process.sampling(16) for _ in range(100)]
gen_samples = np.vstack(gen_samples)
# Check diversity: std, range, etc.
```

---

## 10. Summary

This implementation combines three key ideas:

1. **Time Series → Image**: Delay embedding enables powerful vision models
2. **TST Completion**: Creates natural neighborhoods for convolution
3. **Masked Diffusion**: Prevents over-reliance on imputed values

The result is a **robust, efficient, and high-quality** generative model for irregular time series that:
- Achieves state-of-the-art performance (70% better discriminative score)
- Reduces computational cost by 85%
- Scales to long sequences (up to 10,000+ time steps)
- Handles multiple missing rates (30%-70%)
- Works with various datasets (synthetic, finance, energy, physics)

The code is modular, well-documented, and follows best practices for deep learning research implementation.

