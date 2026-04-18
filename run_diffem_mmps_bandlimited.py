"""
DiffEM MMPS with Band-Limited Likelihood (Inpainting, A = diag(mask))

Proposal B from the Spectral MMPS framework.

EM scaffolding is shared via utils.diffem_mmps_irregular_em. Only the MMPS
posterior denoiser differs: the observation likelihood is band-limited so both
the residual AND the CG operator are projected.

Key insight: At high noise sigma_t, the denoiser estimate x_hat captures only
low-frequency content reliably.  The existing freq-gated code only filters the
residual r, but leaves the CG operator M unchanged — theoretically inconsistent.

The correct band-limited MMPS defines a band-limited likelihood:
    q_omega(y|x_t) = N(P_omega y | P_omega A x_hat,
                       P_omega (Sigma_y + A V A^T) P_omega^T)

This modifies BOTH the residual AND the CG operator:
    [P_omega Sigma_y P_omega + P_omega A V A^T P_omega] v = P_omega r

The optimal cutoff omega(sigma_t) is determined by SNR at each frequency:
    Include frequency f when:  S_x(f) / sigma_t^2 > tau
where S_x(f) is the PSD of the denoiser output.

Theoretical basis:
  - DiEM paper, Eq. 12 (MMPS likelihood)
  - NFC (ICML 2025): approximation error grows with sigma_t for high-freq
  - Thaker et al. (ICCV 2025), Theorem A.3-A.4: drift from high-freq errors
  - Tian et al. (2025): frequency-guided posterior sampling (FGPS)
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


def compute_snr_frequency_mask(denoised, sigma_sq, snr_threshold=1.0):
    fft_denoised = torch.fft.rfft2(denoised.to(torch.float64))
    psd_signal = (fft_denoised.abs() ** 2).mean(dim=(0, 1))

    snr = psd_signal / (sigma_sq.to(torch.float64) + 1e-10)

    freq_mask = torch.sigmoid(5.0 * (snr - snr_threshold))
    return freq_mask


def _sigma_float(sigma):
    if isinstance(sigma, torch.Tensor):
        return float(sigma.detach().cpu().item())
    return float(sigma)


def compute_adaptive_frequency_mask(denoised, sigma, sigma_max_eff,
                                     snr_threshold=1.0, min_cutoff=0.1):
    sigma_sq = sigma ** 2

    snr_mask = compute_snr_frequency_mask(denoised, sigma_sq, snr_threshold)

    H, W_half = snr_mask.shape
    freq_h = torch.arange(H, device=denoised.device).float()
    freq_h = torch.min(freq_h, H - freq_h) / max(H // 2, 1)
    freq_w = torch.arange(W_half, device=denoised.device).float() / max(W_half - 1, 1)
    freq_grid = torch.sqrt(freq_h[:, None] ** 2 + freq_w[None, :] ** 2)

    ratio = _sigma_float(sigma) / max(float(sigma_max_eff), 1e-12)
    ratio = max(0.0, min(1.0, ratio))
    floor_cutoff = min_cutoff + (1.0 - min_cutoff) * (1.0 - ratio)
    floor_mask = (freq_grid <= floor_cutoff).to(dtype=snr_mask.dtype)

    combined = torch.max(snr_mask, floor_mask)
    return combined


def bandlimited_project(x, freq_mask):
    x64 = x.to(torch.float64)
    x_fft = torch.fft.rfft2(x64)
    projected = x_fft * freq_mask.to(x_fft.dtype)[None, None, :, :]
    return torch.fft.irfft2(projected, s=x.shape[-2:]).to(x.dtype)


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


class BandLimitedMMPSDiffusionProcess:
    """MMPS sampler with band-limited likelihood for inpainting."""

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01, cg_iters=1,
                 snr_threshold=1.0, min_cutoff=0.1):
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
        self.cg_iters = cg_iters
        self.snr_threshold = snr_threshold
        self.min_cutoff = min_cutoff

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        sigma_sq = sigma ** 2
        sigma_y_sq = self.sigma_y ** 2

        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True)
            return grad

        sigma_max_eff = min(self.sigma_max, self.net.sigma_max)
        freq_mask = compute_adaptive_frequency_mask(
            denoised, sigma, sigma_max_eff,
            snr_threshold=self.snr_threshold,
            min_cutoff=self.min_cutoff)

        r = x_obs - mask * denoised
        r_bl = bandlimited_project(r, freq_mask)

        def cg_operator_bandlimited(v):
            pv = bandlimited_project(v, freq_mask)
            raw = sigma_y_sq * pv + sigma_sq * mask * vjp_fn(mask * pv)
            return bandlimited_project(raw, freq_mask)

        v = conjugate_gradient(cg_operator_bandlimited, r_bl, maxiter=self.cg_iters)

        score = vjp_fn(mask * v)
        result = denoised.detach() + sigma_sq * score.detach()
        return result

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
    snr_threshold = getattr(args, 'bandlimited_snr_threshold', 1.0)
    min_cutoff = getattr(args, 'bandlimited_min_cutoff', 0.1)
    return BandLimitedMMPSDiffusionProcess(
        args, uncond_model.net, target_shape,
        sigma_y=sigma_y, cg_iters=cg_iters,
        snr_threshold=snr_threshold, min_cutoff=min_cutoff,
    )


def _phase2_extra(a):
    snr_threshold = getattr(a, 'bandlimited_snr_threshold', 1.0)
    min_cutoff = getattr(a, 'bandlimited_min_cutoff', 0.1)
    sy = getattr(a, 'mmps_sigma_y', 0.01)
    cg = getattr(a, 'mmps_cg_iters', 1)
    print(f"  sigma_y={sy}  cg_iters={cg}")
    print(f"  snr_threshold={snr_threshold}  min_cutoff={min_cutoff}")


SPEC = IrregularMMPSExperimentSpec(
    wandb_name_prefix="diffem_mmps_bandlimited",
    build_process_fn=_build_process,
    e_step_title="Band-Limited MMPS",
    tqdm_desc="E-step Band-Limited",
    phase2_start_line="Starting Band-Limited DiffEM-MMPS with {em_iters} EM iterations",
    completion_message="Band-Limited DiffEM-MMPS Complete!",
    phase2_extra_printer=_phase2_extra,
)


if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    diffem_mmps_irregular_main(args, SPEC)
