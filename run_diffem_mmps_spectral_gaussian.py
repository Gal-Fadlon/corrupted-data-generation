"""
DiffEM MMPS — Spectral (FFT) approximate solve for inpainting (A = diag(mask))

Proposal C adapted for fair comparison with other run_diffem_mmps_spectral_*.py:
shared STL init, curriculum, EMA reset, M-step, and Phase 3 via
utils.diffem_mmps_irregular_em.

ONLY ALGORITHMIC DIFFERENCE vs vanilla masked MMPS: the linear solve for v in
    (sigma_y^2 I + sigma_t^2 M J^T M) v = r,  r = y - M x_hat,
is approximated by a closed-form rfft2 diagonal solve using a Hutchinson estimate
of the spectral density of the masked Jacobian channel (same spirit as the A=I
Gaussian-noise variant, extended with M in the probe and residual).

CG fallback uses the exact masked operator (same as spectral_post baseline).
"""

import logging

import numpy as np
import torch
import torch.autograd

from utils.diffem_mmps_irregular_em import (
    IrregularMMPSExperimentSpec,
    diffem_mmps_irregular_main,
)
from utils.utils_args import parse_args_irregular


def _rademacher_like(x):
    u = torch.randint(0, 2, x.shape, device=x.device, dtype=torch.int8)
    return (2 * u.to(dtype=x.dtype) - 1)


def estimate_spectral_density_masked(vjp_fn, mask, shape, sigma_sq, device, n_probes=1):
    psd_accum = None
    for _ in range(n_probes):
        z = _rademacher_like(torch.empty(shape, device=device, dtype=torch.float64))
        mz = mask * z
        jmz = vjp_fn(mz)
        scaled = sigma_sq * mask * jmz
        fft_val = torch.fft.rfft2(scaled.to(torch.float64))
        psd_probe = (fft_val.abs() ** 2).mean(dim=(0, 1))
        if psd_accum is None:
            psd_accum = psd_probe
        else:
            psd_accum = psd_accum + psd_probe
    return psd_accum / n_probes


def spectral_solve(r, sigma_y_sq, spectral_density, reg=1e-8):
    r64 = r.to(torch.float64)
    sy = sigma_y_sq.to(dtype=torch.float64, device=r.device) if isinstance(
        sigma_y_sq, torch.Tensor) else torch.tensor(
            float(sigma_y_sq), dtype=torch.float64, device=r.device)
    r_fft = torch.fft.rfft2(r64)
    denom = sy + spectral_density.to(r_fft.dtype)[None, None, :, :] + reg
    v_fft = r_fft / denom
    v = torch.fft.irfft2(v_fft, s=r.shape[-2:])
    return v.to(r.dtype)


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


class SpectralMMPSDiffusionProcess:
    """Masked MMPS with FFT approximate solve or exact CG fallback."""

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01,
                 n_probes=1, use_cg_fallback=False, cg_iters=1):
        self.args = args
        self.device = args.device
        self.shape = shape
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

        self.sigma_y = sigma_y
        self.n_probes = n_probes
        self.use_cg_fallback = use_cg_fallback
        self.cg_iters = cg_iters

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        sigma_sq = sigma ** 2
        sigma_y_sq = self.sigma_y ** 2

        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return grad

        r = x_obs - mask * denoised

        if self.use_cg_fallback:
            def cg_operator(v):
                return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)
            v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)
        else:
            psd = estimate_spectral_density_masked(
                vjp_fn, mask, denoised.shape, sigma_sq, self.device,
                n_probes=self.n_probes
            )
            v = spectral_solve(r, sigma_y_sq, psd)

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
    n_probes = getattr(args, 'spectral_n_probes', 1)
    use_cg_fallback = getattr(args, 'spectral_cg_fallback', False)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)
    return SpectralMMPSDiffusionProcess(
        args, uncond_model.net, target_shape,
        sigma_y=sigma_y, n_probes=n_probes,
        use_cg_fallback=use_cg_fallback, cg_iters=cg_iters,
    )


SPEC = IrregularMMPSExperimentSpec(
    wandb_name_prefix="diffem_mmps_spectral_gaussian",
    build_process_fn=_build_process,
    e_step_title="Spectral MMPS (FFT inpainting)",
    tqdm_desc="E-step Spectral FFT",
    phase2_start_line="Starting Spectral FFT DiffEM-MMPS with {em_iters} EM iterations",
    completion_message="Spectral FFT DiffEM-MMPS Complete!",
)


if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    diffem_mmps_irregular_main(args, SPEC)
