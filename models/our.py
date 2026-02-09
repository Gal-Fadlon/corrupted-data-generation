import torch
import torch.nn as nn
from contextlib import contextmanager
from models.networks import EDMPrecond, EDMPrecondConditional
from models.ema import LitEma
from models.img_transformations import DelayEmbedder


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
        self.net = EDMPrecond(args.img_resolution, args.input_channels, channel_mult=args.ch_mult,
                              model_channels=args.unet_channels, attn_resolutions=args.attn_resolution)

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
        D_yn = self.net(y + masked_noise, sigma, labels, augment_labels=augment_labels)
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
