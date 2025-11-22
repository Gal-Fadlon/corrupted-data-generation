import torch
import numpy as np

#----------------------------------------------------------------------------
# Tensor clipping utility from Ambient Diffusion

def tensor_clipping(x, static=True, p=0.99):
    dtype = x.dtype
    if static:
        return torch.clip(x, -1.0, 1.0)
    else:
        s_val = torch.tensor(np.percentile(torch.abs(x).detach().cpu().numpy(), p, axis=tuple(range(1, x.ndim))), device=x.device, dtype=dtype)
        s_val = torch.max(s_val, torch.tensor(1.0))
        s_val = s_val.reshape((-1, 1, 1, 1))
        return torch.clip(x, -s_val, s_val) / s_val

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------

@torch.no_grad()
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------

def get_random_mask(image_shape, survival_probability, mask_full_rgb=False, same_for_all_batch=False, device='cuda', seed=None):
    if seed is not None:
        np.random.seed(seed)
    if same_for_all_batch:
        corruption_mask = np.random.binomial(1, survival_probability, size=image_shape[1:]).astype(np.float32)
        corruption_mask = torch.tensor(corruption_mask, device=device, dtype=torch.float32).repeat([image_shape[0], 1, 1, 1])
    else:
        corruption_mask = np.random.binomial(1, survival_probability, size=image_shape).astype(np.float32)
        corruption_mask = torch.tensor(corruption_mask, device=device, dtype=torch.float32)

    if mask_full_rgb:
        # Synchronized mask: same corruption pattern for ALL channels
        # Use case: Time series data where entire time steps are missing (all features at once)
        # Original Ambient code was hardcoded for RGB (3 channels), now generalized to any C
        num_channels = image_shape[1]
        corruption_mask = corruption_mask[:, 0]  # Take first channel [B, H, W]
        corruption_mask = corruption_mask.repeat([num_channels, 1, 1, 1]).transpose(1, 0)  # [B, C, H, W]

    return corruption_mask

def cdist_masked(x1, x2, mask1=None, mask2=None):
    if mask1 is None or mask2 is None:
        mask1 = torch.ones_like(x1)
        mask2 = torch.ones_like(x2)
    x1 = x1[0].unsqueeze(0)
    diffs = x1.unsqueeze(1) - x2.unsqueeze(0)
    combined_mask = mask1.unsqueeze(1) * mask2.unsqueeze(0)
    error = 0.5 * torch.linalg.norm(combined_mask * diffs)**2
    return error

#----------------------------------------------------------------------------

def ambient_sampler(
    net, latents, args, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    sampler_seed=42, survival_probability=0.54,
    mask_full_rgb=False,
    same_for_all_batch=False,
    num_masks=1,
    guidance_scale=0.0,
    clipping=True,
    static=False,  # whether to use soft clipping or static clipping
    resample_guidance_masks=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    clean_image = None

    def sample_masks():
        masks = []
        for _ in range(num_masks):
            masks.append(get_random_mask(latents.shape, survival_probability, mask_full_rgb=mask_full_rgb,
                                         same_for_all_batch=same_for_all_batch, device=latents.device))
        masks = torch.stack(masks)
        return masks

    masks = sample_masks()

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        if resample_guidance_masks:
            guidance_masks = sample_masks()
            masks[:, 1:] = guidance_masks[:, 1:]

        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        x_hat = x_hat.detach()
        x_hat.requires_grad = True

        denoised = []
        for mask_index in range(num_masks):
            corruption_mask = masks[mask_index]
            masked_image = corruption_mask * x_hat
            noisy_image = masked_image

            net_input = torch.cat([noisy_image, corruption_mask], dim=1)
            net_output = net(net_input, t_hat, class_labels).to(torch.float64)[:, :args.input_channels]
            # print_tensor_stats(net_output, 'Denoised')
            if clipping:
                net_output = tensor_clipping(net_output, static=static)

            if clean_image is not None:
                net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image

            # Euler step.
            denoised.append(net_output)

        stack_denoised = torch.stack(denoised)
        flattened = stack_denoised.view(stack_denoised.shape[0], -1)
        l2_norm = cdist_masked(flattened, flattened, None, None)
        l2_norm = l2_norm.mean()
        rec_grad = torch.autograd.grad(l2_norm, inputs=x_hat)[0]

        clean_pred = stack_denoised[0]

        single_mask_grad = (t_next - t_hat) * (x_hat - clean_pred) / t_hat
        grad_1 = single_mask_grad - guidance_scale * rec_grad
        x_next += grad_1

        if i < num_steps - 1:
            x_next = x_next.detach()
            x_next.requires_grad = True

            denoised = []
            for mask_index in range(num_masks):
                corruption_mask = masks[mask_index]
                masked_image = corruption_mask * x_next
                noisy_image = masked_image
                net_input = torch.cat([noisy_image, corruption_mask], dim=1)
                net_output = net(net_input, t_next, class_labels).to(torch.float64)[:, :args.input_channels]
                if clipping:
                    net_output = tensor_clipping(net_output, static=static)

                if clean_image is not None:
                    net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image
                denoised.append(net_output)

            stack_denoised = torch.stack(denoised)
            flattened = stack_denoised.view(stack_denoised.shape[0], -1)
            l2_norm = cdist_masked(flattened, flattened, None, None)
            rec_grad = torch.autograd.grad(l2_norm, inputs=x_next)[0]
            clean_pred = stack_denoised[0]
            single_mask_grad = (t_next - t_hat) * (x_next - clean_pred) / t_next
            grad_2 = single_mask_grad - guidance_scale * rec_grad
            x_next = x_hat + 0.5 * (grad_1 + grad_2)
        else:
            if clean_image is not None:
                x_next = masks[0] * x_next + (1 - masks[0]) * clean_image
            else:
                clean_image = x_next
                x_next = x_hat + grad_1
    return x_next

