import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from utils.ambient_net_input import concat_mask_channel, concat_ones_mask
from models.networks import EDMPrecond, EDMPrecondConditional
from models.ema import LitEma
from models.img_transformations import DelayEmbedder, SpectrogramEmbedder, GAFEmbedder, MultiViewEmbedder
from utils.utils_stl import stl_decompose_batch


class TS2img_Karras_Cond(nn.Module):
    """
    Conditional diffusion model for DiffEM-style training.
    
    This model learns q_theta(x|y) - the posterior distribution of clean data
    given corrupted observations. It concatenates conditioning channels 
    (completed sequence + mask) to the network input.
    """
    
    def __init__(self, args, device, cond_channels=None):
        """
        Args:
            args: configuration arguments
            device: target device
            cond_channels: number of conditioning channels (default: 2 * input_channels for cond + mask)
        """
        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        self.device = device
        self.target_channels = args.input_channels
        
        # Conditioning channels: completed sequence + binary mask
        if cond_channels is None:
            self.cond_channels = args.input_channels + 1  # cond_img + mask channel
        else:
            self.cond_channels = cond_channels
        
        # Use conditional EDMPrecond that properly handles target/cond channels
        self.net = EDMPrecondConditional(
            args.img_resolution,
            target_channels=self.target_channels,
            cond_channels=self.cond_channels,
            channel_mult=args.ch_mult,
            model_channels=args.unet_channels, 
            attn_resolutions=args.attn_resolution
        )
        
        self.delay = args.delay
        self.embedding = args.embedding
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.num_features = args.input_channels

        self.ts_img = DelayEmbedder(self.device, args.seq_len, args.delay, args.embedding, 
                                     self.batch_size, self.num_features)

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def ts_to_img(self, signal, pad_val=None):
        """Convert time series to image representation."""
        return self.ts_img.ts_to_img(signal, True, pad_val) if pad_val else self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        """Convert image back to time series."""
        return self.ts_img.img_to_ts(img)

    def forward_conditional(self, x, cond_img, cond_mask, labels=None, augment_pipe=None):
        """
        Forward pass for conditional diffusion training.
        
        Args:
            x: clean target image (batch, target_channels, H, W)
            cond_img: conditioning image - TST completed (batch, target_channels, H, W)
            cond_mask: binary mask indicating observed pixels (batch, 1, H, W)
            labels: optional class labels
            augment_pipe: optional augmentation pipeline
        
        Returns:
            D_yn: denoised output (batch, target_channels, H, W)
            weight: loss weight
        """
        # Sample noise level
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        # Apply augmentation if provided
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        
        # Add noise ONLY to the target (clean) image, not to conditioning
        n = torch.randn_like(y) * sigma
        y_noisy = y + n
        
        # Concatenate: [noisy_target | conditioning_img | conditioning_mask]
        # cond_mask should be (batch, 1, H, W)
        if cond_mask.dim() == 3:
            cond_mask = cond_mask.unsqueeze(1)
        
        x_in = torch.cat([y_noisy, cond_img, cond_mask], dim=1)
        
        # Forward through network - EDMPrecondConditional returns target channels only
        D_yn = self.net(x_in, sigma, labels, augment_labels=augment_labels)
        
        return D_yn, weight

    def loss_fn_conditional(self, x, cond_img, cond_mask, obs_mask=None):
        """
        Compute conditional diffusion loss for DiffEM M-step.
        
        Args:
            x: clean target image (batch, target_channels, H, W)
            cond_img: conditioning image (batch, target_channels, H, W)
            cond_mask: mask for conditioning (batch, 1, H, W) or (batch, H, W)
            obs_mask: optional mask to weight loss only on originally observed pixels
                     If None, uses cond_mask for loss weighting
        
        Returns:
            loss: scalar loss value
            to_log: dict of values to log
        """
        to_log = {}
        
        # Forward pass
        output, weight = self.forward_conditional(x, cond_img, cond_mask)
        
        # Ensure masks have correct shape
        if cond_mask.dim() == 3:
            cond_mask = cond_mask.unsqueeze(1)
        
        # Use observation mask for loss if provided, otherwise use cond_mask
        if obs_mask is not None:
            if obs_mask.dim() == 3:
                obs_mask = obs_mask.unsqueeze(1)
            loss_mask = obs_mask
        else:
            loss_mask = cond_mask
        
        # Compute masked loss (only on observed pixels)
        diff_sq = (output - x).square()
        
        # Apply mask and weight
        masked_diff = diff_sq * loss_mask
        
        # Mean over masked pixels
        num_valid = loss_mask.sum() + 1e-8
        loss = (weight * masked_diff).sum() / num_valid
        
        to_log['cond_karras_loss'] = loss.detach().item()
        
        return loss, to_log

    def loss_fn_conditional_full(self, x, cond_img, cond_mask):
        """
        Compute conditional diffusion loss on ALL pixels (not masked).
        Use this when training on fully reconstructed data.
        """
        to_log = {}
        
        output, weight = self.forward_conditional(x, cond_img, cond_mask)
        
        loss = (weight * (output - x).square()).mean()
        
        to_log['cond_karras_loss_full'] = loss.detach().item()
        
        return loss, to_log

    @contextmanager
    def ema_scope(self, context=None):
        """Context manager to temporarily switch to EMA weights during inference."""
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        """Update EMA model after each training batch."""
        if self.use_ema:
            self.model_ema(self.net)

    def reset_ema(self):
        """Reset EMA shadow weights to current model weights (fresh start per EM lap)."""
        if self.use_ema:
            self.model_ema.reset(self.net)


class TS2img_Karras(nn.Module):
    def __init__(self, args, device):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        self.device = device
        self.ambient_concat_further_mask = bool(
            getattr(args, 'ambient_concat_further_mask', False)
        )
        _in_ch = args.input_channels + (
            1 if self.ambient_concat_further_mask else 0
        )
        self.net = EDMPrecond(
            args.img_resolution, _in_ch, channel_mult=args.ch_mult,
            model_channels=args.unet_channels, attn_resolutions=args.attn_resolution,
        )

        self.delay = args.delay
        self.embedding = args.embedding
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.num_features = args.input_channels

        self.ts_img = DelayEmbedder(self.device, args.seq_len, args.delay, args.embedding, self.batch_size, self.num_features)

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def ts_to_img(self, signal, pad_val=None):
        """
        Args:
            signal: signal to convert to image
        """
        # pad_val is used only for delay embedding, as the value to pad the image with
        # when creating the mask, we need to use 1 as padding value
        # if pad_val is given, it is used to overwrite the default value of 0
        return self.ts_img.ts_to_img(signal, True, pad_val) if pad_val else self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        return self.ts_img.img_to_ts(img)

    def loss_fn_irregular(self, x, mask=None):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}
        if mask is None:
            mask = torch.isnan(x).float() * -1 + 1
            x = torch.nan_to_num(x, nan=0.0)
        output, weight = self.forward_irregular(x, mask)
        x = self.unpad(x * mask, x.shape)
        output = self.unpad(output * mask, x.shape)
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()
        return loss, to_log

    def forward_irregular(self, x, mask, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        n = torch.randn_like(y) * sigma
        masked_noise = n * (mask)
        x_in = concat_ones_mask(
            y + masked_noise, self.ambient_concat_further_mask
        )
        D_yn = self.net(x_in, sigma, labels, augment_labels=augment_labels)
        return D_yn, weight

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    @contextmanager
    def ema_scope(self, context=None):
        """
        Context manager to temporarily switch to EMA weights during inference.
        Args:
            context: some string to print when switching to EMA weights

        Returns:

        """
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        """
        this function updates the EMA model, if it is used
        Args:
            *args:

        Returns:

        """
        if self.use_ema:
            self.model_ema(self.net)

    def reset_ema(self):
        """Reset EMA shadow weights to current model weights (fresh start per EM lap)."""
        if self.use_ema:
            self.model_ema.reset(self.net)

    # =========================================================================
    # Ambient-aware loss for training with masked/corrupted data
    # (Combines Ambient-Omni reweighting + Ambient Diffusion masking)
    # =========================================================================

    def loss_fn_ambient_masked(self, x_img, mask_img, delta=0.3):
        """Ambient loss for masked data in delay-embedded image space.

        Combines two ideas:
          1. Ambient Diffusion (Daras et al., NeurIPS 2023): further corrupt the
             already-masked data with mask B; network input is ``tilde{A} ⊙ (x_0 + σ ε)``
             (zeros on dropped positions — not fresh noise); loss on all ``A``
             positions. Optionally concatenates ``tilde{A}`` as an extra channel
             when ``args.ambient_concat_further_mask`` is set (paper / official code).
          2. Ambient-Omni (Daras et al., 2025): reweight the loss with
             ``ambient_factor = sigma^4 / (sigma^2 - sigma_tn^2)^2`` so that
             corrupted data contributes properly at each noise level.

        For masking corruption the observed pixels have ``sigma_tn = 0`` (clean),
        so the ambient factor simplifies to ``sigma^4 / sigma^4 = 1`` and the
        total weight reduces to the standard EDM weight.  The key benefit comes
        from the *further corruption* + *loss-on-A* mechanism which forces the
        network to predict missing positions, learning the clean distribution.

        Args:
            x_img:    data in image space ``(B, C, H, W)`` — observed positions
                      are clean; missing positions are zero-filled.
            mask_img: binary observation mask ``(B, C, H, W)`` or ``(B, 1, H, W)``.
                      1 = observed, 0 = missing.
            delta:    probability of masking an *observed* position in the further
                      corruption step (controls ``tilde{A} = B * A``).

        Returns:
            loss:   scalar loss value.
            to_log: dict with diagnostic quantities.
        """
        to_log = {}

        # Ensure mask has same spatial dims as data
        if mask_img.shape[1] == 1 and x_img.shape[1] > 1:
            mask_img = mask_img.expand_as(x_img)

        # --- Further corruption: tilde{A} = B * A ---
        B = (torch.rand_like(mask_img) >= delta).float()
        further_mask = mask_img * B  # subset of observed positions

        # --- Sample noise level (log-normal, standard EDM schedule) ---
        rnd_normal = torch.randn([x_img.shape[0], 1, 1, 1], device=x_img.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # EDM denoising weight:  w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        edm_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # --- Forward diffusion: x_t = x_0 + sigma * z ---
        noise = torch.randn_like(x_img) * sigma
        x_noisy = x_img + noise

        # Paper Eq. (100): feed tilde{A} ⊙ (x_0 + ση); zeros on further-erased pixels.
        # (Official ambient-diffusion: hat * (y + n), not independent noise in holes.)
        x_data = x_noisy * further_mask
        x_input = concat_mask_channel(
            x_data, further_mask, self.ambient_concat_further_mask
        )

        # --- Network prediction h_theta(x_input, sigma) -> E[X_0 | x_input] ---
        D_x = self.net(x_input, sigma, None)

        # --- Loss on *original* mask positions (A), normalized by observed count ---
        x_unpad = self.unpad(x_img * mask_img, x_img.shape)
        D_unpad = self.unpad(D_x * mask_img, x_img.shape)

        n_observed = mask_img.sum() + 1e-8
        loss = (edm_weight * (D_unpad - x_unpad).square()).sum() / n_observed

        to_log['ambient_masked_loss'] = loss.detach().item()
        with torch.no_grad():
            to_log['sigma_mean'] = sigma.mean().item()
            to_log['further_mask_ratio'] = further_mask.mean().item()
            to_log['original_mask_ratio'] = mask_img.mean().item()

        return loss, to_log

    def loss_fn_ambient_operator(self, x_img, y_obs, operator,
                                  corruption_type='missing', delta=0.1,
                                  lambda_obs=1.0, sigma_n=0.0):
        """A-aware Ambient loss for unconditional model with general ForwardOperator.

        Generalises ``loss_fn_ambient_masked`` to arbitrary linear operators A.
        The loss is a weighted combination of:
          * observation-space loss:  ||A(D_theta) - y_obs||^2  (robust to bad recons)
          * full-space loss:         ||D_theta   - x_img||^2   (detail recovery)

        For **missing data** the Ambient further-corruption trick is applied
        (randomly drop delta fraction of observed positions from the input) so
        the model is forced to predict beyond what it directly observes.

        For **blur / smoothing / projection** the input is standard noisy data
        and the observation-space loss flows gradients through A, teaching the
        model to produce outputs consistent with y when projected.

        For **noise (A = I)** the caller should set ``lambda_obs = 0`` because
        the obs target *is* the noisy observation and using it directly would
        bias the model toward generating noisy data.

        Args:
            x_img:  (B, C, H, W) starting point for forward diffusion.
                    In EM iterations this is the E-step reconstruction (image space).
                    In Phase-0 pre-training this is the zero-filled / corrupted obs.
            y_obs:  original corrupted observation **in the operator's output space**.
                    Missing/blur/noise: (B, C, H, W).
                    Smoothing (TSComposite): (B, T, C).
                    Projection: (B, d_obs).
            operator: ForwardOperator with ``.forward()`` / ``.adjoint()``.
                      For missing data it must expose a ``.mask`` attribute.
            corruption_type: one of 'missing', 'ts_missing_noise',
                    'gaussian_blur', 'gaussian_noise', 'ts_gaussian_noise',
                    'ts_temporal_smoothing', 'random_projection'.
            delta:  further-corruption probability (missing data only).
            lambda_obs: blend weight for obs-space vs full-space loss.
            sigma_n: observation noise std (reserved for future noise-floor
                     correction; not used in loss computation today).

        Returns:
            loss, to_log dict.
        """
        to_log = {}

        # --- noise level (log-normal, standard EDM schedule) ---
        rnd_normal = torch.randn([x_img.shape[0], 1, 1, 1], device=x_img.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        edm_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # --- forward diffusion ---
        noise = torch.randn_like(x_img) * sigma
        x_noisy = x_img + noise

        # --- input construction ---
        is_mask_type = corruption_type in ('missing', 'ts_missing_noise')
        if is_mask_type and hasattr(operator, 'mask'):
            obs_mask = operator.mask.to(dtype=x_img.dtype, device=x_img.device)
            if obs_mask.shape[1] == 1 and x_img.shape[1] > 1:
                obs_mask = obs_mask.expand_as(x_img)
            B_rand = (torch.rand_like(obs_mask) >= delta).float()
            further_mask = obs_mask * B_rand
            x_data = x_noisy * further_mask
            x_input = concat_mask_channel(
                x_data, further_mask, self.ambient_concat_further_mask
            )
        else:
            x_input = concat_ones_mask(
                x_noisy, self.ambient_concat_further_mask)

        # --- network prediction ---
        D_x = self.net(x_input, sigma, None)

        # =====================  full-space loss  =====================
        x_up = self.unpad(x_img, x_img.shape)
        D_up = self.unpad(D_x, x_img.shape)
        loss_full = (edm_weight * (D_up - x_up).square()).mean()

        # =====================  observation-space loss  ==============
        if lambda_obs > 0:
            pred_obs = operator.forward(D_x.to(y_obs.dtype))
            pred_obs = pred_obs.to(x_img.dtype)
            y_target = y_obs.to(x_img.dtype)

            if is_mask_type and hasattr(operator, 'mask'):
                obs_m = operator.mask.to(dtype=x_img.dtype, device=x_img.device)
                if obs_m.shape[1] == 1 and x_img.shape[1] > 1:
                    obs_m = obs_m.expand_as(x_img)
                pred_up = self.unpad(pred_obs, x_img.shape)
                y_up = self.unpad(y_target, x_img.shape)
                obs_up = self.unpad(obs_m, x_img.shape)
                n_obs = obs_up.sum() + 1e-8
                loss_obs = (edm_weight * (pred_up - y_up).square() * obs_up).sum() / n_obs
            elif pred_obs.shape == x_img.shape:
                pred_up_o = self.unpad(pred_obs, x_img.shape)
                y_up_o = self.unpad(y_target, x_img.shape)
                loss_obs = (edm_weight * (pred_up_o - y_up_o).square()).mean()
            else:
                w = edm_weight.view(edm_weight.shape[0],
                                     *([1] * (pred_obs.ndim - 1)))
                loss_obs = (w * (pred_obs - y_target).square()).mean()
        else:
            loss_obs = torch.tensor(0.0, device=x_img.device)

        # =====================  combined  ============================
        loss = lambda_obs * loss_obs + (1.0 - lambda_obs) * loss_full

        to_log['ambient_op_loss'] = loss.detach().item()
        to_log['ambient_op_loss_obs'] = (loss_obs.detach().item()
                                          if torch.is_tensor(loss_obs) else 0.0)
        to_log['ambient_op_loss_full'] = loss_full.detach().item()
        to_log['lambda_obs'] = lambda_obs
        with torch.no_grad():
            to_log['sigma_mean'] = sigma.mean().item()
            if is_mask_type and hasattr(operator, 'mask'):
                to_log['further_mask_ratio'] = further_mask.mean().item()

        return loss, to_log


# =============================================================================
# Decomposition-Enhanced Model
# =============================================================================

class TS2img_Karras_Decomposed(nn.Module):
    """
    Decomposition-enhanced unconditional diffusion model.

    Key insight from time-series literature (Diffusion-TS ICLR 2024,
    STDiffusion 2025, FFT-DM ICLR 2026): time series have well-known
    decomposition into trend + seasonal + residual.  By presenting each
    component as a separate channel group, the UNet can:
      - Learn component-specific features (smooth trend vs periodic seasonal)
      - Maintain cross-component correlations through its attention layers
      - Operate on statistically cleaner per-component signals

    Image layout:  (B, 3*C, H, W)
      channels [  0 :   C]  → trend  images
      channels [  C : 2*C]  → seasonal images
      channels [2*C : 3*C]  → residual images

    Optional spectral loss encourages the UNet to preserve the frequency
    structure: trend should be smooth (low-freq), seasonal should preserve
    dominant frequencies.
    """

    N_COMPONENTS = 3

    def __init__(self, args, device, stl_period=None):
        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        self.device = device
        self.original_channels = args.input_channels
        total_channels = args.input_channels * self.N_COMPONENTS

        self.net = EDMPrecond(
            args.img_resolution, total_channels,
            channel_mult=args.ch_mult,
            model_channels=args.unet_channels,
            attn_resolutions=args.attn_resolution,
        )

        self.delay = args.delay
        self.embedding = args.embedding
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.num_features = args.input_channels
        self.stl_period = stl_period

        self.ts_img = DelayEmbedder(
            self.device, args.seq_len, args.delay, args.embedding,
            self.batch_size, self.num_features,
        )

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(
                self.net, decay=0.9999, use_num_upates=True,
                warmup=args.ema_warmup,
            )
        else:
            self.use_ema = False

    # ------------------------------------------------------------------
    # Decomposition helpers
    # ------------------------------------------------------------------

    def decompose_ts(self, signal):
        """
        STL-decompose a batch of time series.

        Args:
            signal: (B, T, C) tensor — must NOT contain NaN

        Returns:
            trend, seasonal, residual — each (B, T, C)
        """
        return stl_decompose_batch(signal, period=self.stl_period)

    # ------------------------------------------------------------------
    # TS ↔ Image (decomposed multi-channel)
    # ------------------------------------------------------------------

    def ts_to_img_decomposed(self, signal):
        """
        Convert TS → decomposed 3-channel-group image.

        signal: (B, T, C) clean tensor (no NaN)
        returns: (B, 3*C, H, W)
        """
        trend, seasonal, residual = self.decompose_ts(signal)
        trend_img = self.ts_img.ts_to_img(trend)
        seasonal_img = self.ts_img.ts_to_img(seasonal)
        residual_img = self.ts_img.ts_to_img(residual)
        return torch.cat([trend_img, seasonal_img, residual_img], dim=1)

    def ts_to_img_from_components(self, trend, seasonal, residual):
        """
        Convert pre-decomposed components → stacked image.
        Each input: (B, T, C) tensor
        """
        return torch.cat([
            self.ts_img.ts_to_img(trend),
            self.ts_img.ts_to_img(seasonal),
            self.ts_img.ts_to_img(residual),
        ], dim=1)

    def img_to_ts_decomposed(self, img):
        """
        Convert decomposed image → time series by summing components.

        img:     (B, 3*C, H, W)
        returns: (B, T, C)
        """
        C = self.original_channels
        trend_ts = self.ts_img.img_to_ts(img[:, :C])
        seasonal_ts = self.ts_img.img_to_ts(img[:, C:2 * C])
        residual_ts = self.ts_img.img_to_ts(img[:, 2 * C:3 * C])
        return trend_ts + seasonal_ts + residual_ts

    def ts_to_img(self, signal, pad_val=None):
        """Plain (non-decomposed) conversion — used for masks."""
        if pad_val:
            return self.ts_img.ts_to_img(signal, True, pad_val)
        return self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        """Dispatches to decomposed or plain conversion based on channel count."""
        if img.shape[1] == self.original_channels * self.N_COMPONENTS:
            return self.img_to_ts_decomposed(img)
        return self.ts_img.img_to_ts(img)

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward_irregular(self, x, mask=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = x
        n = torch.randn_like(y) * sigma
        if mask is not None:
            n = n * mask
        D_yn = self.net(y + n, sigma, None)
        return D_yn, weight

    def loss_fn_irregular(self, x, mask=None):
        """
        Standard EDM denoising loss on the decomposed image.
        """
        to_log = {}
        if mask is None:
            mask = torch.isnan(x).float() * -1 + 1
            x = torch.nan_to_num(x, nan=0.0)
        output, weight = self.forward_irregular(x, mask)
        x_m = self.unpad(x * mask, x.shape)
        out_m = self.unpad(output * mask, x.shape)
        loss = (weight * (out_m - x_m).square()).mean()
        to_log['karras loss'] = loss.detach().item()
        return loss, to_log

    def loss_fn_decomposed(self, x_decomposed, lambda_trend=0.1, lambda_spectral=0.05):
        """
        EDM loss with component-aware regularisation.

        x_decomposed: (B, 3*C, H, W) — stacked [trend | seasonal | residual]
        lambda_trend:    weight for trend total-variation penalty
        lambda_spectral: weight for spectral matching penalty on seasonal

        Auxiliary losses are gated by noise level: they only contribute
        when sigma < 1 (high-SNR regime) so they don't destabilise
        training at high noise where the denoised estimate is uncertain.
        TV loss is computed on the valid (unpadded) region only.
        """
        to_log = {}
        C = self.original_channels

        rnd_normal = torch.randn([x_decomposed.shape[0], 1, 1, 1],
                                 device=x_decomposed.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x_decomposed) * sigma
        output = self.net(x_decomposed + n, sigma, None)
        base_loss = (weight * (output - x_decomposed).square()).mean()
        to_log['karras loss'] = base_loss.detach().item()

        # Gate: auxiliary losses only active at low noise (sigma < 1)
        aux_gate = (sigma.view(-1) < 1.0).float()  # (B,)
        n_active = aux_gate.sum().clamp(min=1)

        # Crop to valid (unpadded) region for aux losses
        valid_h, valid_w = self.ts_img.img_shape[2], self.ts_img.img_shape[3]
        output_valid = output[:, :, :valid_h, :valid_w]
        target_valid = x_decomposed[:, :, :valid_h, :valid_w]

        # Trend smoothness: total variation on valid region only
        trend_out = output_valid[:, :C]
        tv_h = (trend_out[:, :, 1:, :] - trend_out[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
        tv_w = (trend_out[:, :, :, 1:] - trend_out[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
        trend_tv_per_sample = tv_h + tv_w  # (B,)
        trend_tv_loss = (aux_gate * trend_tv_per_sample).sum() / n_active
        to_log['trend_tv'] = trend_tv_loss.detach().item()

        # Spectral consistency on valid region, gated by noise level
        seasonal_out = output_valid[:, C:2 * C]
        seasonal_tgt = target_valid[:, C:2 * C]
        fft_out = torch.fft.rfft2(seasonal_out)
        fft_tgt = torch.fft.rfft2(seasonal_tgt)
        spec_per_sample = (fft_out.abs() - fft_tgt.abs()).square().mean(dim=(1, 2, 3))
        spectral_loss = (aux_gate * spec_per_sample).sum() / n_active
        to_log['spectral_loss'] = spectral_loss.detach().item()

        total = base_loss + lambda_trend * trend_tv_loss + lambda_spectral * spectral_loss
        to_log['total_loss'] = total.detach().item()
        return total, to_log

    def unpad(self, x, original_shape):
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        if self.use_ema:
            self.model_ema(self.net)

    def reset_ema(self):
        if self.use_ema:
            self.model_ema.reset(self.net)


# =============================================================================
# Multi-View Model (Delay + optional Spectrogram + optional GAF)
# =============================================================================

class TS2img_Karras_MultiView(nn.Module):
    """
    Unconditional diffusion model using multiple image representations
    stacked along the channel dimension.

    Supported views (controlled by ``view_names``):
      - 'delay'       — delay embedding (always primary, used for img_to_ts)
      - 'spectrogram' — STFT magnitude spectrogram
      - 'gaf'         — Gramian Angular Sum Field (GASF)

    The UNet receives (B, n_views * C, H, W) images.
    Inversion always uses the delay-embedding channels.
    """

    def __init__(self, args, device, view_names=('delay', 'spectrogram')):
        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        self.device = device
        self.view_names = list(view_names)
        self.n_views = len(view_names)
        self.original_channels = args.input_channels

        total_channels = args.input_channels * self.n_views

        self.net = EDMPrecond(
            args.img_resolution, total_channels,
            channel_mult=args.ch_mult,
            model_channels=args.unet_channels,
            attn_resolutions=args.attn_resolution,
        )

        self.delay = args.delay
        self.embedding = args.embedding
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.num_features = args.input_channels

        embedders = []
        for v in view_names:
            if v == 'delay':
                embedders.append(
                    DelayEmbedder(device, args.seq_len, args.delay,
                                  args.embedding, args.batch_size,
                                  args.input_channels)
                )
            elif v == 'spectrogram':
                embedders.append(
                    SpectrogramEmbedder(device, args.seq_len,
                                       n_fft=args.embedding,
                                       hop_length=args.delay,
                                       img_resolution=args.img_resolution)
                )
            elif v == 'gaf':
                embedders.append(
                    GAFEmbedder(device, args.seq_len,
                                img_resolution=args.img_resolution)
                )
            else:
                raise ValueError(f"Unknown view: {v}")

        self.ts_img = MultiViewEmbedder(embedders, device, args.seq_len)

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999,
                                    use_num_upates=True,
                                    warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def ts_to_img(self, signal, pad_val=None):
        if pad_val:
            return self.ts_img.ts_to_img(signal, True, pad_val)
        return self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        return self.ts_img.img_to_ts(img)

    def forward_irregular(self, x, mask=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = x
        n = torch.randn_like(y) * sigma
        if mask is not None:
            n = n * mask
        D_yn = self.net(y + n, sigma, None)
        return D_yn, weight

    def loss_fn_irregular(self, x, mask=None):
        to_log = {}
        if mask is None:
            mask = torch.isnan(x).float() * -1 + 1
            x = torch.nan_to_num(x, nan=0.0)
        output, weight = self.forward_irregular(x, mask)
        x_m = self.unpad(x * mask, x.shape)
        out_m = self.unpad(output * mask, x.shape)
        loss = (weight * (out_m - x_m).square()).mean()
        to_log['karras loss'] = loss.detach().item()
        return loss, to_log

    def unpad(self, x, original_shape):
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        if self.use_ema:
            self.model_ema(self.net)

    def reset_ema(self):
        if self.use_ema:
            self.model_ema.reset(self.net)
