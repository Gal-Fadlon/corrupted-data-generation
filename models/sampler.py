import numpy as np
import torch


# -----------------------------------------------------------------------------
# Exact copies (1:1) of Ambient Diffusion sampling helpers, vendored locally.
# Source of truth:
# - `ambient-diffusion-main/eval.py`: cdist_masked(), ambient_sampler()
# - `ambient-diffusion-main/torch_utils/ambient_diffusion.py`: get_random_mask()
# - `ambient-diffusion-main/dnnlib/util.py`: tensor_clipping()

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
        corruption_mask = corruption_mask[:, 0]
        corruption_mask = corruption_mask.repeat([3, 1, 1, 1]).transpose(1, 0)
    return corruption_mask


def tensor_clipping(x, static=True, p=0.99):
    dtype = x.dtype
    if static:
        return torch.clip(x, -1.0, 1.0)
    else:
        s_val = torch.tensor(np.percentile(torch.abs(x).detach().cpu().numpy(), p, axis=tuple(range(1, x.ndim))), device=x.device, dtype=dtype)
        s_val = torch.max(s_val, torch.tensor(1.0))
        s_val = s_val.reshape((-1, 1, 1, 1))
        return torch.clip(x, -s_val, s_val) / s_val


def cdist_masked(x1, x2, mask1=None, mask2=None):
    if mask1 is None or mask2 is None:
        mask1 = torch.ones_like(x1)
        mask2 = torch.ones_like(x2)
    x1 = x1[0].unsqueeze(0)
    diffs = x1.unsqueeze(1) - x2.unsqueeze(0)
    combined_mask = mask1.unsqueeze(1) * mask2.unsqueeze(0)
    error = 0.5 * torch.linalg.norm(combined_mask * diffs)**2
    return error


def ambient_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
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
    # Time-series support: if provided, sample masks in TS space and then transform via ts_to_img().
    ts_img=None,
    valid_pixel_mask=None,  # optional extra structural mask (can be None if ts_to_img mask already includes padding)
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    clean_image = None

    def apply_valid_mask(x):
        if valid_pixel_mask is None:
            return x
        m = valid_pixel_mask.to(device=x.device, dtype=x.dtype)
        if m.ndim == 4 and m.shape[0] == 1 and x.ndim == 4 and x.shape[0] != 1:
            m = m.expand(x.shape[0], -1, -1, -1)
        return x * m

    def sample_masks():
        masks = []
        for _ in range(num_masks):
            if ts_img is None:
                masks.append(get_random_mask(latents.shape, survival_probability, mask_full_rgb=mask_full_rgb, 
                                                    same_for_all_batch=same_for_all_batch, device=latents.device))
            else:
                # Sample in TS space (B, T, C) and map to image mask (B, C, H, W) via delay-embedding.
                b = latents.shape[0]
                t = int(ts_img.seq_len)
                c = int(getattr(ts_img, "num_features", latents.shape[1]))
                if same_for_all_batch:
                    base = torch.full((1, t, c), float(survival_probability), device=latents.device, dtype=torch.float32)
                    mask_ts = torch.bernoulli(base).expand(b, -1, -1)
                else:
                    probs = torch.full((b, t, c), float(survival_probability), device=latents.device, dtype=torch.float32)
                    mask_ts = torch.bernoulli(probs)
                mask_img = ts_img.ts_to_img(mask_ts)  # returns padded square with zeros on padding/unused pixels
                masks.append(mask_img.to(device=latents.device, dtype=torch.float32))
        masks = torch.stack(masks)
        if valid_pixel_mask is not None:
            # Never "observe" structurally invalid pixels.
            masks = apply_valid_mask(masks)
        return masks

    masks = sample_masks()

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float32) * t_steps[0]
    x_next = apply_valid_mask(x_next)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        if resample_guidance_masks:
            guidance_masks = sample_masks()
            masks[:, 1:] = guidance_masks[:, 1:]

        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        x_hat = apply_valid_mask(x_hat)
        
        x_hat = x_hat.detach()
        x_hat.requires_grad = True

        denoised = []
        for mask_index in range(num_masks):
            corruption_mask = masks[mask_index]
            masked_image = corruption_mask * x_hat
            noisy_image = masked_image

            net_input = torch.cat([noisy_image, corruption_mask], dim=1)
            net_output = net(net_input, t_hat, class_labels).to(torch.float32)[:, :latents.shape[1]]
            if clipping:
                net_output = tensor_clipping(net_output, static=static)

            if clean_image is not None:
                net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image

            # Euler step.
            denoised.append(net_output)

        stack_denoised = torch.stack(denoised)
        if guidance_scale != 0.0:
            flattened = stack_denoised.view(stack_denoised.shape[0], -1)
            l2_norm = cdist_masked(flattened, flattened, None, None)
            l2_norm = l2_norm.mean()
            rec_grad = torch.autograd.grad(l2_norm, inputs=x_hat)[0]
        else:
            # Avoid autograd under torch.no_grad() when guidance is disabled.
            rec_grad = torch.zeros_like(x_hat)

        clean_pred = stack_denoised[0]
        single_mask_grad = (t_next - t_hat) * (x_hat - clean_pred) / t_hat
        grad_1 = single_mask_grad - guidance_scale * rec_grad

        x_next += grad_1
        x_next = apply_valid_mask(x_next)

        if i < num_steps - 1:
            x_next = x_next.detach()
            x_next.requires_grad = True

            denoised = []
            for mask_index in range(num_masks):
                corruption_mask = masks[mask_index]
                masked_image = corruption_mask * x_next
                noisy_image = masked_image
                net_input = torch.cat([noisy_image, corruption_mask], dim=1)
                net_output = net(net_input, t_next, class_labels).to(torch.float32)[:, :latents.shape[1]]
                if clipping:
                    net_output = tensor_clipping(net_output, static=static)

                if clean_image is not None:
                    net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image
                denoised.append(net_output)

            stack_denoised = torch.stack(denoised)
            if guidance_scale != 0.0:
                flattened = stack_denoised.view(stack_denoised.shape[0], -1)
                l2_norm = cdist_masked(flattened, flattened, None, None)
                rec_grad = torch.autograd.grad(l2_norm, inputs=x_next)[0]
            else:
                rec_grad = torch.zeros_like(x_next)
            clean_pred = stack_denoised[0]
            single_mask_grad = (t_next - t_hat) * (x_next - clean_pred) / t_next
            grad_2 = single_mask_grad - guidance_scale * rec_grad
            x_next = x_hat + 0.5 * (grad_1 + grad_2)
            x_next = apply_valid_mask(x_next)
        else:
            if clean_image is not None:
                x_next = masks[0] * x_next + (1 - masks[0]) * clean_image
            else:
                clean_image = x_next
                x_next = x_hat + grad_1
            x_next = apply_valid_mask(x_next)
    return x_next


class DiffusionProcess:
    """
    Thin wrapper so the rest of the codebase can keep calling `process.sampling(...)`,
    but the actual sampling is the 1:1 Ambient Diffusion sampler.
    """
    def __init__(
        self,
        args,
        diffusion_fn,
        shape,
        survival_probability=0.54,
        mask_full_rgb=False,
        same_for_all_batch=False,
        num_masks=1,
        guidance_scale=0.0,
        clipping=True,
        static=False,
        resample_guidance_masks=False,
        ts_img=None,
        valid_pixel_mask=None,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
    ):
        self.args = args
        self.device = args.device
        self.shape = shape
        self.net = diffusion_fn.to(device=self.device)
        self.survival_probability = survival_probability
        self.mask_full_rgb = mask_full_rgb
        self.same_for_all_batch = same_for_all_batch
        self.num_masks = num_masks
        self.guidance_scale = guidance_scale
        self.clipping = clipping
        self.static = static
        self.resample_guidance_masks = resample_guidance_masks
        self.ts_img = ts_img
        self.valid_pixel_mask = valid_pixel_mask
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

    @torch.no_grad()
    def sampling(self, sampling_number=16, impute=False, xT=None):
        if xT is None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        return ambient_sampler(
            net=self.net,
            latents=xT,
            class_labels=None,
            num_steps=self.args.diffusion_steps,
            survival_probability=self.survival_probability,
            mask_full_rgb=self.mask_full_rgb,
            same_for_all_batch=self.same_for_all_batch,
            num_masks=self.num_masks,
            guidance_scale=self.guidance_scale,
            clipping=self.clipping,
            static=self.static,
            resample_guidance_masks=self.resample_guidance_masks,
            ts_img=self.ts_img,
            valid_pixel_mask=self.valid_pixel_mask,
            S_churn=self.S_churn,
            S_min=self.S_min,
            S_max=self.S_max,
            S_noise=self.S_noise,
        )
