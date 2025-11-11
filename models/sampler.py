import numpy as np
import torch
from utils.utils import get_random_mask

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
        
        # Ambient Diffusion parameters
        self.use_ambient_style = getattr(args, 'use_ambient_style', False)
        self.num_features = getattr(args, 'input_channels', shape[0])  # Number of image channels (not including mask)
        self.clipping = getattr(args, 'sampling_clipping', True)  # Output clipping
        
        # Calculate survival probability for sampling mask
        # This should match the training distribution: (1-p)(1-δ)
        # where p = corruption_probability, δ = delta_probability
        if hasattr(args, 'sampling_survival_probability'):
            # Explicit override from args
            self.survival_probability = args.sampling_survival_probability
        else:
            # Calculate from corruption probabilities
            corruption_prob = getattr(args, 'corruption_probability', 0.4)
            delta_prob = getattr(args, 'delta_probability', 0.1)
            self.survival_probability = (1 - corruption_prob) * (1 - delta_prob)
        
        if self.use_ambient_style:
            print(f"🎭 Ambient Diffusion Sampler initialized:")
            print(f"   - Survival probability: {self.survival_probability:.4f}")
            print(f"   - Clipping enabled: {self.clipping}")
            print(f"   - Number of image channels: {self.num_features}")
    
    def tensor_clipping(self, x, static=False):
        """
        Clip network output to valid range.
        
        Args:
            x: Tensor to clip
            static: If True, hard clip to [0,1]. If False, soft clip using tanh.
        
        Returns:
            Clipped tensor
        """
        if static:
            # Hard clipping to [0, 1]
            return torch.clamp(x, 0.0, 1.0)
        else:
            # Soft clipping using tanh: maps (-inf, inf) -> (0, 1)
            return 0.5 * (1.0 + torch.tanh(x))

    def sample(self, latents, class_labels=None):
        """
        Sample from the diffusion model.
        
        Supports two modes:
        1. Original mode: Standard EDM sampling (network input = image only)
        2. Ambient mode: Ambient Diffusion sampling (network input = [masked_image, mask])
        """
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Generate corruption mask for Ambient-style sampling
        if self.use_ambient_style:
            # Generate mask once at the beginning (fixed mask throughout sampling)
            corruption_mask = get_random_mask(
                latents.shape, 
                self.survival_probability, 
                mask_full_rgb=False,
                same_for_all_batch=False, 
                device=latents.device
            )

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily (stochasticity).
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # ===== FIRST NETWORK CALL (Euler step) =====
            if self.use_ambient_style:
                # Ambient Diffusion: mask the noisy image and concatenate with mask
                masked_image = corruption_mask * x_hat
                net_input = torch.cat([masked_image, corruption_mask], dim=1)
                net_output = self.net(net_input, t_hat, class_labels).to(torch.float64)
                # CRITICAL: Slice output to get only image channels
                denoised = net_output[:, :self.num_features]
                # Optional: Apply clipping
                if self.clipping:
                    denoised = self.tensor_clipping(denoised, static=False)
            else:
                # Original: direct network call with full image
                denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            
            # Euler step
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # ===== SECOND NETWORK CALL (Heun's 2nd order correction) =====
            if i < self.num_steps - 1:
                if self.use_ambient_style:
                    # Ambient Diffusion: mask and concatenate for 2nd order correction
                    masked_image = corruption_mask * x_next
                    net_input = torch.cat([masked_image, corruption_mask], dim=1)
                    net_output = self.net(net_input, t_next, class_labels).to(torch.float64)
                    # CRITICAL: Slice output to get only image channels
                    denoised = net_output[:, :self.num_features]
                    # Optional: Apply clipping
                    if self.clipping:
                        denoised = self.tensor_clipping(denoised, static=False)
                else:
                    # Original: direct network call with full image
                    denoised = self.net(x_next, t_next, class_labels).to(torch.float64)
                
                # 2nd order correction
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    @torch.no_grad()
    def sampling(self, sampling_number=16, impute=False, xT=None):
        if xT is None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        return self.sample(xT)
