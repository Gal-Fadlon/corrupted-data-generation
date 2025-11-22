import torch
import torch.nn as nn
from contextlib import contextmanager
from models.networks import EDMPrecond
from models.ema import LitEma
from models.img_transformations import DelayEmbedder

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
        
        # Determine number of input channels based on training style
        self.use_ambient_style = getattr(args, 'use_ambient_style', False)
        if self.use_ambient_style:
            # Ambient style: network receives [image, mask] concatenated = 2*C channels
            network_input_channels = args.input_channels * 2
            print(f"🔧 Ambient Diffusion mode: Network expects {network_input_channels} input channels")
            print(f"   ({args.input_channels} for image + {args.input_channels} for mask)")
        else:
            # Original style: network receives only image = C channels
            network_input_channels = args.input_channels
            print(f"🔧 Original mode: Network expects {network_input_channels} input channels")
        
        self.net = EDMPrecond(args.img_resolution, network_input_channels, channel_mult=args.ch_mult,
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

    def ts_to_img(self, signal):
        """
        Args:
            signal: signal to convert to image
        """
        return self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        return self.ts_img.img_to_ts(img)

    def loss_fn_irregular(self, x, mask=None):
        '''
        ORIGINAL METHOD: Uses TST completion + single mask
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}
        if mask is None:
            mask = torch.isnan(x).float() * -1 + 1
            x = torch.nan_to_num(x, nan=0.0)
        output, weight = self.forward_irregular(x, mask)
        x = self.unpad(x * mask, self.ts_img.img_shape)
        output = self.unpad(output * mask, self.ts_img.img_shape)
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()
        return loss, to_log
    
    def loss_fn_ambient(self, x, corruption_matrix, hat_corruption_matrix, padded_mask):
        '''
        AMBIENT DIFFUSION STYLE LOSS: No TST completion, dual corruption masks
        
        Args:
            x: Clean/completed image [B, C, H, W] (NaN replaced with 0)
            corruption_matrix (A): Original corruption from data [B, C, H, W] (1=observed, 0=missing)
            hat_corruption_matrix (Ã): Further corrupted version [B, C, H, W]
        
        Returns:
            val_loss: Loss on A pixels (USED FOR BACKPROP)
            to_log: Dictionary with all three losses
                - train_loss: Loss on Ã pixels (monitoring)
                - val_loss: Loss on A pixels (training objective)
                - test_loss: Loss on all pixels (evaluation)
        '''
        to_log = {}
        
        # Forward pass with Ambient-style noise and masking
        # Sigma is sampled from log-normal distribution (same as original)
        output, weight = self.forward_ambient(x, hat_corruption_matrix, padded_mask)
        
        # Unpad for loss computation (use cached non-square shape, not square shape!)
        x_unpad = self.unpad(x * padded_mask, self.ts_img.img_shape)
        output_unpad = self.unpad(output * padded_mask, self.ts_img.img_shape)
        corruption_matrix_unpad = self.unpad(corruption_matrix * padded_mask, self.ts_img.img_shape)
        hat_corruption_matrix_unpad = self.unpad(hat_corruption_matrix * padded_mask, self.ts_img.img_shape)
        
        # Compute THREE losses (like Ambient Diffusion)
        # 1. train_loss: on Ã (further corrupted) - for monitoring
        train_loss = (weight * (hat_corruption_matrix_unpad * (output_unpad - x_unpad)).square()).mean()
        
        # 2. val_loss: on A (original corruption) - USED FOR TRAINING
        val_loss = (weight * (corruption_matrix_unpad * (output_unpad - x_unpad)).square()).mean()
        
        # 3. test_loss: on ALL pixels - for evaluation
        test_loss = (weight * (output_unpad - x_unpad).square()).mean()
        
        # Log all three
        to_log['train_loss'] = train_loss.detach().item()
        to_log['val_loss'] = val_loss.detach().item()
        to_log['test_loss'] = test_loss.detach().item()
        
        # Return val_loss for backprop (like Ambient Diffusion paper)
        return val_loss, to_log

    def forward_irregular(self, x, mask, labels=None, augment_pipe=None):
        '''ORIGINAL METHOD: Masks noise before adding to image'''
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        n = torch.randn_like(y) * sigma
        masked_noise = n * (mask)
        D_yn = self.net(y + masked_noise, sigma, labels, augment_labels=augment_labels)
        return D_yn, weight
    
    def forward_ambient(self, x, hat_corruption_matrix, padded_mask, labels=None, augment_pipe=None):
        '''
        AMBIENT DIFFUSION STYLE FORWARD: Applies corruption to noisy image
        
        Args:
            x: Input image [B, C, H, W]
            hat_corruption_matrix: Corruption mask Ã [B, C, H, W] (1=visible, 0=corrupted)
            labels: Class labels (optional)
            augment_pipe: Data augmentation (optional)
        
        Returns:
            D_yn: Network prediction [B, C, H, W] (only first C channels)
            weight: Loss weight scalar
        '''
        # Sample noise level from log-normal distribution (same as original)
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        # Compute loss weight
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # Apply augmentation if provided
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        
        # Add noise to image
        n = torch.randn_like(y) * sigma
        
        # Apply corruption to NOISY image: Ã(x + σₜη)
        masked_image = hat_corruption_matrix * (y + n)
        noisy_image = masked_image
        
        # Network receives: [noisy_image, hat_corruption_matrix] concatenated
        # Input: 2*C channels (image + mask)
        cat_input = torch.cat([noisy_image, hat_corruption_matrix], dim=1)
        
        # Forward through network
        # Network may output 2*C channels, but we only use first C channels (the prediction)
        D_yn = self.net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :self.num_features]
        
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
