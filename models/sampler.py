import numpy as np
import torch


class DiffusionProcess():
    def __init__(self, args, diffusion_fn, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''
        self.args = args
        self.device = args.device
        self.shape = shape
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps), dim=0).to(device=self.device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=self.device), self.alpha_bars[:-1]])
        self.deterministic = args.deterministic
        self.net = diffusion_fn.to(device=self.device)
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

    def sample(self, latents, class_labels=None):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    @torch.no_grad()
    def sampling(self, sampling_number=16, impute=False, xT=None):
        if xT is None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        return self.sample(xT)


class ConditionalDiffusionProcess():
    """
    Conditional diffusion sampler for DiffEM E-step.
    
    Samples X ~ q_theta(x|y) given corrupted observation Y using
    a conditional diffusion model.
    """
    
    def __init__(self, args, diffusion_fn, target_shape, cond_channels):
        """
        Args:
            args: configuration arguments
            diffusion_fn: conditional diffusion network
            target_shape: shape of target output (C, H, W)
            cond_channels: number of conditioning channels
        """
        self.args = args
        self.device = args.device
        self.target_shape = target_shape  # (target_C, H, W)
        self.cond_channels = cond_channels
        self.net = diffusion_fn.to(device=self.device)
        
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

    def sample_conditional(self, cond_img, cond_mask, class_labels=None, xT=None):
        """
        Sample from the conditional distribution q_theta(x|y).
        
        Args:
            cond_img: conditioning image (batch, cond_C, H, W)
            cond_mask: conditioning mask (batch, 1, H, W)
            class_labels: optional class labels
            xT: initial noise (if None, sample from N(0, I))
        
        Returns:
            x_0: sampled clean image (batch, target_C, H, W)
        """
        batch_size = cond_img.shape[0]
        
        # Ensure mask has correct shape
        if cond_mask.dim() == 3:
            cond_mask = cond_mask.unsqueeze(1)
        
        # Initialize with random noise for target channels only
        if xT is None:
            xT = torch.randn([batch_size, *self.target_shape], device=self.device)
        
        # Adjust noise levels based on what's supported by the network
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=self.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop
        x_next = xT.to(torch.float64) * t_steps[0]
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # Increase noise temporarily
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Build conditional input: [x_hat | cond_img | cond_mask]
            x_in = torch.cat([x_hat.to(cond_img.dtype), cond_img, cond_mask], dim=1)
            
            # Euler step - EDMPrecondConditional returns target channels directly
            denoised = self.net(x_in, t_hat, class_labels).to(torch.float64)
            
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction
            if i < self.num_steps - 1:
                x_in = torch.cat([x_next.to(cond_img.dtype), cond_img, cond_mask], dim=1)
                denoised = self.net(x_in, t_next, class_labels).to(torch.float64)
                
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(torch.float32)

    @torch.no_grad()
    def sampling(self, cond_img, cond_mask, num_samples=None):
        """
        Sample from the conditional distribution.
        
        Args:
            cond_img: conditioning image (batch, cond_C, H, W)
            cond_mask: conditioning mask
            num_samples: number of samples per conditioning (default 1)
        
        Returns:
            samples: (batch * num_samples, target_C, H, W)
        """
        if num_samples is None or num_samples == 1:
            return self.sample_conditional(cond_img, cond_mask)
        
        # Generate multiple samples per conditioning
        batch_size = cond_img.shape[0]
        all_samples = []
        
        for _ in range(num_samples):
            samples = self.sample_conditional(cond_img, cond_mask)
            all_samples.append(samples)
        
        return torch.cat(all_samples, dim=0)