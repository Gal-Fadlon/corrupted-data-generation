"""
DiffEM with MMPS E-step — Iterative-STL Init + Spectral Post-Correction

Pipeline shared via utils.diffem_mmps_irregular_em (same STL init, curriculum,
EMA reset, M-step, eval, Phase 3 as other spectral MMPS run scripts).

ONLY DIFFERENCE: the MMPS class.  Instead of decomposition-projection
(_project_denoised), this variant applies a sigma-dependent Butterworth
low-pass filter to the MMPS correction (sigma^2 * score) in the time
series frequency domain.

Theoretical basis:
  - Equivalent to using anisotropic Sigma_y in the MMPS likelihood
    (Wiener filter connection, DiEM Eq. 12-13).
  - At high diffusion noise only low-freq corrections pass through;
    at low noise all corrections pass — matching the reliability of
    the Gaussian approximation across frequency bands.
  - Justified by SAGD (Scimeca et al.) and CPS (Narasimhan et al.).
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


class SpectralPostMMPSDiffusionProcess:
    """
    MMPS sampler with spectral post-correction on the likelihood score.

    After the standard MMPS CG solve (identical to vanilla MMPS), the
    correction term (sigma^2 * score) is converted to time-series space,
    low-pass filtered via a sigma-dependent Butterworth filter in the
    frequency domain, and converted back to image space.

    This suppresses unreliable high-frequency corrections at high noise
    levels while preserving them at low noise.  This is a Wiener-style
    spectral shaping of the *correction* in true time-index space (per
    feature), not the same linear operator as replacing sigma_y^2 I with
    a frequency-dependent Sigma_y inside the CG solve (see spectral_cg
    variant).  The two are related in spirit (trust low frequencies more
    when the Gaussian MMPS approximation is coarse) but not identical.
    """

    def __init__(self, args, diffusion_fn, shape, model,
                 sigma_y=0.01, cg_iters=1, filter_order=2):
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

        def cg_operator(v):
            return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)

        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)
        score = vjp_fn(mask * v)

        denoised_det = denoised.detach()
        correction = sigma_sq * score.detach()

        sigma_ratio = float(sigma) / self.sigma_max
        if sigma_ratio < 0.05:
            return denoised_det + correction

        corr_ts = self.model.img_to_ts(correction.float())
        T = corr_ts.shape[1]
        C_fft = torch.fft.rfft(corr_ts, dim=1)
        freqs = torch.fft.rfftfreq(T, device=corr_ts.device).to(dtype=corr_ts.dtype)

        f_cutoff = max(0.05, 1.0 - sigma_ratio)
        weights = 1.0 / (1.0 + (freqs / f_cutoff) ** self.filter_order)
        weights = weights.to(dtype=C_fft.real.dtype)

        C_fft = C_fft * weights[None, :, None]
        corr_filtered = torch.fft.irfft(C_fft, n=T, dim=1)
        correction_img = self.model.ts_to_img(corr_filtered).to(denoised_det.dtype)

        return denoised_det + correction_img

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
    return SpectralPostMMPSDiffusionProcess(
        args, uncond_model.net, target_shape, uncond_model,
        sigma_y=sigma_y, cg_iters=cg_iters, filter_order=filter_order,
    )


SPEC = IrregularMMPSExperimentSpec(
    wandb_name_prefix="diffem_mmps_spectral_post",
    build_process_fn=_build_process,
    e_step_title="Spectral Post-Correction MMPS",
    tqdm_desc="E-step Spectral-Post",
    phase2_start_line=(
        "Starting Spectral Post-Correction DiffEM-MMPS with {em_iters} EM iterations"
    ),
    completion_message="Spectral Post-Correction DiffEM-MMPS Complete!",
)


if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    diffem_mmps_irregular_main(args, SPEC)
