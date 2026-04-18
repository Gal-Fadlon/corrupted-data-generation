"""
DiffEM MMPS with Spectral Preconditioner for CG (Inpainting, A = diag(mask))

Proposal A from the Spectral MMPS framework.

EM scaffolding is shared via utils.diffem_mmps_irregular_em (same as other
spectral MMPS run scripts). Only the MMPS posterior denoiser differs: standard
CG is replaced by preconditioned CG using a circulant (FFT) preconditioner.

Key insight: The MMPS CG system
    (sigma_y^2 I + sigma_t^2 mask J^T mask) v = r
is poorly conditioned.  For time series, J^T (denoiser Jacobian) is
approximately Toeplitz, so the operator has spectral structure.

A circulant preconditioner P (Chan 1988) clusters eigenvalues of P^{-1}M
around 1, reducing the effective condition number from O(T) to O(1)
(Tyrtyshnikov's clustering theorem).  1 iteration of preconditioned CG
matches ~3 iterations of standard CG.

The preconditioner is built by ignoring the mask (pretending A = I):
    P(f) = sigma_y^2 + S_hat(f)
where S_hat(f) is the estimated PSD of V[x|x_t] via Hutchinson.

Application:  P^{-1}(v) = IFFT( FFT(v) / P(f) )

Theoretical basis:
  - DiEM paper, Algorithm 3 (MMPS) for inpainting
  - Chan (1988), optimal circulant preconditioner
  - Tyrtyshnikov clustering theorem
  - Hutchinson (1990), stochastic trace estimation
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


def estimate_spectral_density(vjp_fn, shape, sigma_sq, device, n_probes=1):
    psd_accum = None
    for _ in range(n_probes):
        z = _rademacher_like(torch.empty(shape, device=device, dtype=torch.float64))
        Jz = vjp_fn(z)
        scaled = sigma_sq * Jz
        fft_val = torch.fft.rfft2(scaled.to(torch.float64))
        psd_probe = (fft_val.abs() ** 2).mean(dim=(0, 1))
        if psd_accum is None:
            psd_accum = psd_probe
        else:
            psd_accum = psd_accum + psd_probe
    return psd_accum / n_probes


def build_spectral_preconditioner(sigma_y_sq, spectral_density, reg=1e-8):
    P_diag = (sigma_y_sq + spectral_density + reg).to(torch.float64)

    def precond_inv(v):
        v64 = v.to(torch.float64)
        v_fft = torch.fft.rfft2(v64)
        v_fft = v_fft / P_diag[None, None, :, :].to(v_fft.dtype)
        return torch.fft.irfft2(v_fft, s=v.shape[-2:]).to(v.dtype)

    return precond_inv


def preconditioned_conjugate_gradient(A_fn, b, precond_inv, maxiter=1):
    x = torch.zeros_like(b)
    r = b.clone()
    z = precond_inv(r)
    p = z.clone()

    for _ in range(maxiter):
        Ap = A_fn(p)
        rTz = (r * z).sum()
        pTAp = (p * Ap).sum()
        alpha = rTz / (pTAp + 1e-30)

        x = x + alpha * p
        r_new = r - alpha * Ap
        z_new = precond_inv(r_new)

        rTz_new = (r_new * z_new).sum()
        beta = rTz_new / (rTz + 1e-30)

        p = z_new + beta * p
        r = r_new
        z = z_new

    return x


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


class SpectralPrecondMMPSDiffusionProcess:
    """MMPS sampler with spectral-preconditioned CG for inpainting."""

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01, cg_iters=1,
                 n_probes=1, use_precond=True):
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
        self.n_probes = n_probes
        self.use_precond = use_precond

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

        r = x_obs - mask * denoised

        def cg_operator(v):
            return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)

        if self.use_precond:
            psd = estimate_spectral_density(
                vjp_fn, denoised.shape, sigma_sq, self.device,
                n_probes=self.n_probes)
            precond_inv = build_spectral_preconditioner(sigma_y_sq, psd)
            v = preconditioned_conjugate_gradient(
                cg_operator, r, precond_inv, maxiter=self.cg_iters)
        else:
            v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)

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
    n_probes = getattr(args, 'spectral_n_probes', 1)
    use_precond = not getattr(args, 'spectral_no_precond', False)
    return SpectralPrecondMMPSDiffusionProcess(
        args, uncond_model.net, target_shape,
        sigma_y=sigma_y, cg_iters=cg_iters,
        n_probes=n_probes, use_precond=use_precond,
    )


def _phase2_extra(a):
    use_precond = not getattr(a, 'spectral_no_precond', False)
    sy = getattr(a, 'mmps_sigma_y', 0.01)
    cg = getattr(a, 'mmps_cg_iters', 1)
    print(f"  sigma_y={sy}  cg_iters={cg}  precond={use_precond}")


SPEC = IrregularMMPSExperimentSpec(
    wandb_name_prefix="diffem_mmps_spectral_precond",
    build_process_fn=_build_process,
    e_step_title="Spectral Preconditioned MMPS",
    tqdm_desc="E-step Spectral Precond",
    phase2_start_line="Starting Spectral Precond DiffEM-MMPS with {em_iters} EM iterations",
    completion_message="Spectral Precond DiffEM-MMPS Complete!",
    phase2_extra_printer=_phase2_extra,
)


if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    diffem_mmps_irregular_main(args, SPEC)
