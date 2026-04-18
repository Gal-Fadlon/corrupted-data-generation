"""
Agentic DiffEM v3 — Spectral CG MMPS with closed-loop adaptation.

Extends run_diffem_mmps_spectral_cg.py with:
  - Multi-sample E-step  → K samples as data augmentation (K×N training set)
  - Spectral diagnosis   → per-frequency inter-sample Fourier CV (raw, not normalised)
  - Bidirectional adaptive f_cutoff → increases/decreases based on diagnosis
  - Uniform M-step       → standard loss on augmented samples (no confidence weighting)
  - Progress-based curriculum → anneals extra reveals over full EM range
  - Early stopping       → patience-based on disc_mean

The MMPS sampler is identical to SpectralCGMMPSDiffusionProcess except
`f_cutoff_base` is an externally-set parameter updated by the agentic
planner rather than being hardcoded at 1.0.

With --estep_n_samples 1 this degrades gracefully to the fixed baseline.
"""

import logging

import numpy as np
import torch
import torch.autograd

from utils.agentic_em import agentic_em_main
from utils.diffem_mmps_irregular_em import IrregularMMPSExperimentSpec
from utils.utils_args import parse_args_irregular


def conjugate_gradient(A_fn, b, maxiter=1):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()

    for _ in range(maxiter):
        Ap = A_fn(p)
        rTr = (r * r).sum()
        pTAp = (p * Ap).sum()
        alpha = rTr / (pTAp + 1e-30)
        x = x + alpha * p
        r_new = r - alpha * Ap
        rTr_new = (r_new * r_new).sum()
        beta = rTr_new / (rTr + 1e-30)
        p = r_new + beta * p
        r = r_new

    return x


class AdaptiveSpectralCGMMPSDiffusionProcess:
    """
    MMPS with anisotropic Sigma_y and an externally-adaptive f_cutoff_base.

    Identical to SpectralCGMMPSDiffusionProcess in
    run_diffem_mmps_spectral_cg.py except that f_cutoff_base is a
    constructor parameter set by the agentic planner each EM iteration,
    rather than being implicitly 1.0.

    Within a single diffusion chain the cutoff still varies with
    sigma_ratio; across EM iterations the *baseline* shifts based on
    the spectral diagnosis.
    """

    def __init__(self, args, diffusion_fn, shape, model,
                 sigma_y=0.01, cg_iters=1, filter_order=2,
                 f_cutoff_base=0.5):
        self.args = args
        self.device = args.device
        self.shape = shape
        self.net = diffusion_fn.to(device=self.device)
        self.model = model

        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

        self.sigma_y = sigma_y
        self.cg_iters = cg_iters
        self.filter_order = filter_order
        self.f_cutoff_base = f_cutoff_base

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        sigma_sq = sigma ** 2
        sigma_y_sq = self.sigma_y ** 2
        sigma_ratio = float(sigma) / self.sigma_max

        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True,
            )
            return grad

        r = x_obs - mask * denoised

        if sigma_ratio < 0.05:
            def cg_operator(v):
                return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)
        else:
            f_cutoff = max(0.05, self.f_cutoff_base * (1.0 - sigma_ratio))

            def spectral_sigma_y(v):
                v_ts = self.model.img_to_ts(v.float())
                V_fft = torch.fft.rfft(v_ts, dim=1)
                T_len = v_ts.shape[1]
                freqs = torch.fft.rfftfreq(T_len, device=v.device).to(dtype=v.dtype)
                aniso = 1.0 + (freqs / f_cutoff) ** self.filter_order
                scale = (sigma_y_sq * aniso).to(dtype=V_fft.real.dtype)
                V_fft = V_fft * scale[None, :, None]
                v_ts = torch.fft.irfft(V_fft, n=T_len, dim=1)
                return self.model.ts_to_img(v_ts).to(v.dtype)

            def cg_operator(v):
                return spectral_sigma_y(v) + sigma_sq * mask * vjp_fn(mask * v)

        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)
        score = vjp_fn(mask * v)
        return denoised.detach() + sigma_sq * score.detach()

    def mmps_sample(self, latents, x_obs_img, mask_img):
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_obs = x_obs_img.to(torch.float64)
        mask = mask_img.to(torch.float64)
        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.posterior_denoise(x_hat, t_hat, x_obs, mask)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised_2 = self.posterior_denoise(x_next, t_next, x_obs, mask)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sampling_mmps(self, x_obs_img, mask_img):
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, x_obs_img, mask_img)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        latents = torch.randn([sampling_number, *self.shape], device=self.device)
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.net(x_hat, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


def _build_process(args, uncond_model):
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)
    filter_order = getattr(args, 'spectral_filter_order', 2)
    f_cutoff_base = getattr(args, 'adaptive_f_cutoff', 0.5)
    return AdaptiveSpectralCGMMPSDiffusionProcess(
        args, uncond_model.net, target_shape, uncond_model,
        sigma_y=sigma_y, cg_iters=cg_iters, filter_order=filter_order,
        f_cutoff_base=f_cutoff_base,
    )


SPEC = IrregularMMPSExperimentSpec(
    wandb_name_prefix="agentic_spectral_cg",
    build_process_fn=_build_process,
    e_step_title="Agentic Spectral CG (adaptive anisotropic Sigma_y) MMPS",
    tqdm_desc="E-step Agentic-Spectral-CG",
    phase2_start_line=(
        "Starting Agentic Spectral CG DiffEM-MMPS with {em_iters} EM iterations"
    ),
    completion_message="Agentic Spectral CG DiffEM-MMPS Complete!",
)


if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    agentic_em_main(args, SPEC)
