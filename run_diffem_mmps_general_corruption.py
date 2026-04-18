"""
DiffEM with Unconditional Model + MMPS E-step — General A Matrix

Extends run_diffem_mmps_ema_reset_stl_curriculum.py to support arbitrary linear
forward operators A, not just missing-data masks.

Corruption model:  Y = A(X) + epsilon,  epsilon ~ N(0, sigma_y^2 I)

Supported corruption types:
  - missing:              A = diag(mask),  y = mask * x  (original behavior)
  - gaussian_noise:       A = I,           y = x + N(0, sigma_y^2 I) in image space
  - gaussian_blur:        A = blur kernel, y = blur(x) + epsilon
  - random_projection:    A = random mat,  y = Ax + epsilon
  - ts_gaussian_noise:    noise in TS space, then ts_to_img; A=I in image space
  - ts_temporal_smoothing:moving-average in TS space; TSCompositeOperator bridges domains
  - ts_missing_noise:     missing data + additive noise on observed entries

The MMPS posterior denoiser generalizes from mask-specific to operator-based:
  r  = y - A(D_theta(x_t))
  CG:  (sigma_y^2 I + sigma_t^2 * A(J^T(A^T(v)))) v = r
  score = J^T(A^T(v))
  D_posterior = D_theta + sigma_t^2 * score

References:
  - DiffEM paper (Rozet et al.) for the general formulation
  - DiffEM-main/priors/diffusion.py:481-558 for PosteriorDenoiser
"""

import torch
import torch.autograd
import torch.multiprocessing
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import os
import sys
import math
import numpy as np
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm

from metrics import evaluate_model_irregular
from metrics.memorization import compute_memorization_metric
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import (
    gen_dataloader, save_reconstructions, load_reconstructions
)
from utils.utils_args import parse_args_irregular
from utils.train_unconditional import train_unconditional_regular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.ambient_net_input import concat_mask_channel, concat_ones_mask
from utils.utils_stl import initialize_with_stl, stl_decompose_single, auto_detect_period

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Forward Operator Abstraction
# =============================================================================

class ForwardOperator(ABC):
    """
    Base class for linear forward operators: y = A(x) + epsilon.

    Subclasses must implement forward() and adjoint().  The MMPS posterior
    denoiser uses these to compute:
        r  = y - A(D(x_t))
        CG operator: sigma_y^2 * v + sigma_t^2 * A(J^T(A^T(v)))
        score = J^T(A^T(v))
    """

    def __init__(self, sigma_y: float):
        self._sigma_y = sigma_y

    @property
    def sigma_y(self) -> float:
        return self._sigma_y

    @sigma_y.setter
    def sigma_y(self, value: float):
        self._sigma_y = value

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward operator A(x)."""
        ...

    @abstractmethod
    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        """Apply adjoint operator A^T(v)."""
        ...

    def corrupt(self, x_clean: torch.Tensor) -> torch.Tensor:
        """Generate corrupted observation y = A(x) + epsilon."""
        y = self.forward(x_clean)
        if self._sigma_y > 0:
            y = y + self._sigma_y * torch.randn_like(y)
        return y


class MissingDataOperator(ForwardOperator):
    """
    A = diag(mask): diagonal binary mask selecting observed entries.
    forward: mask * x,  adjoint: mask * v  (self-adjoint).
    """

    def __init__(self, mask: torch.Tensor, sigma_y: float = 0.01):
        super().__init__(sigma_y)
        self.mask = mask  # (B, 1, H, W) or broadcastable

    def forward(self, x):
        return self.mask * x

    def adjoint(self, v):
        return self.mask * v


class GaussianNoiseOperator(ForwardOperator):
    """
    A = I (identity): observation is x + noise.
    forward: x,  adjoint: v  (identity is self-adjoint).
    sigma_y IS the corruption level.
    """

    def __init__(self, sigma_y: float):
        super().__init__(sigma_y)

    def forward(self, x):
        return x

    def adjoint(self, v):
        return v


class GaussianBlurOperator(ForwardOperator):
    """
    A = Gaussian blur convolution in image space.
    Since a Gaussian kernel is symmetric, A = A^T.
    Implemented via FFT for efficiency: A(x) = F^{-1}(H * F(x)).
    """

    def __init__(self, kernel_size: int, blur_sigma: float,
                 num_channels: int, sigma_y: float = 0.01,
                 device: str = 'cpu'):
        super().__init__(sigma_y)
        self.kernel_size = kernel_size
        self.blur_sigma = blur_sigma
        self.device = device

        kernel_1d = self._gaussian_kernel_1d(kernel_size, blur_sigma)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(
            num_channels, 1, 1, 1
        ).to(device)
        self.num_channels = num_channels
        self.padding = kernel_size // 2

    @staticmethod
    def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        return g / g.sum()

    def forward(self, x):
        k = self.kernel.to(dtype=x.dtype)
        return F.conv2d(x, k, padding=self.padding,
                        groups=self.num_channels)

    def adjoint(self, v):
        k = self.kernel.to(dtype=v.dtype)
        return F.conv2d(v, k, padding=self.padding,
                        groups=self.num_channels)


class RandomProjectionOperator(ForwardOperator):
    """
    A = random dense matrix: compressed sensing-style projection.
    forward: A @ flatten(x) -> y in R^{d_obs}
    adjoint: A^T @ y -> unflatten to image shape

    Each sample in the batch shares the same projection matrix.
    """

    def __init__(self, d_obs: int, img_shape: tuple,
                 sigma_y: float = 0.01, device: str = 'cpu',
                 seed: int = 42):
        super().__init__(sigma_y)
        self.d_obs = d_obs
        self.img_shape = img_shape  # (C, H, W)
        d_full = math.prod(img_shape)

        rng = torch.Generator(device='cpu').manual_seed(seed)
        A = torch.randn(d_obs, d_full, generator=rng) / math.sqrt(d_obs)
        self.A_mat = A.to(device)
        self.device = device

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        A = self.A_mat.to(dtype=x_flat.dtype)
        return torch.mm(x_flat, A.T)  # (B, d_obs)

    def adjoint(self, v):
        B = v.shape[0]
        A = self.A_mat.to(dtype=v.dtype)
        x_flat = torch.mm(v, A)  # (B, d_full)
        return x_flat.reshape(B, *self.img_shape)


class TSCompositeOperator(ForwardOperator):
    """
    Wraps a TS-space linear operator for use with the image-space diffusion model.

    forward: x_img -> img_to_ts -> A_ts -> y_ts  (observation in TS space)
    adjoint: v_ts -> A_ts^T -> ts_to_img -> v_img (back to image space)

    CG operates in TS space; the VJP chain maps correctly through image space.

    NOTE: img_to_ts / ts_to_img internally allocate float32 tensors, but the
    MMPS CG solver operates in float64.  We cast outputs back to the caller's
    dtype so that the CG residual, VJP cotangent, and score all stay in float64.
    """

    def __init__(self, ts_forward_fn, ts_adjoint_fn, uncond_model,
                 sigma_y: float = 0.01):
        super().__init__(sigma_y)
        self.ts_forward = ts_forward_fn
        self.ts_adjoint = ts_adjoint_fn
        self.model = uncond_model

    def forward(self, x_img):
        orig_dtype = x_img.dtype
        x_ts = self.model.img_to_ts(x_img)
        result = self.ts_forward(x_ts)
        return result.to(orig_dtype)

    def adjoint(self, v_ts):
        orig_dtype = v_ts.dtype
        a_v = self.ts_adjoint(v_ts)
        result = self.model.ts_to_img(a_v)
        return result.to(orig_dtype)


def temporal_moving_average(x_ts, window: int):
    """
    Apply a symmetric moving average along the time dimension per feature.

    Args:
        x_ts: (B, T, C) tensor
        window: odd integer window size

    Returns:
        smoothed: (B, T, C) tensor
    """
    B, T, C = x_ts.shape
    pad = window // 2
    x = x_ts.permute(0, 2, 1)  # (B, C, T)
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / window
    smoothed = F.conv1d(
        x_padded.reshape(B * C, 1, -1), kernel
    ).reshape(B, C, T)
    return smoothed.permute(0, 2, 1)  # (B, T, C)


# =============================================================================
# PPCA Covariance for Posterior Denoiser
# =============================================================================

class PPCACov:
    """
    Low-rank covariance Sigma_x = U U^T + D_diag * I in image-flattened space.

    Provides efficient matrix-vector products for the MMPS posterior:
        Sigma_{x|t} @ v = alpha * v + beta * U @ M^{-1} @ U^T @ v
    where:
        alpha = sigma_t^2 * D_diag / (D_diag + sigma_t^2)
        beta  = sigma_t^4 / (D_diag + sigma_t^2)^2
        M     = I + U^T U / (D_diag + sigma_t^2)

    This replaces the Jacobian-VJP covariance in the CG operator,
    making CG iterations gradient-free and better regularised.
    """

    def __init__(self, U, D_diag, device='cpu'):
        self.U = U.to(device)           # (D, rank), float64
        self.D_diag = float(D_diag)
        self.rank = U.shape[1]
        self.UtU = self.U.T @ self.U    # (rank, rank), precomputed

    def cov_x_xt_mv(self, z_flat, sigma_t_sq):
        """
        Apply Sigma_{x|t} @ z where z is (B, D) flattened image vectors.

        Sigma_{x|t} = sigma_t^2 I - sigma_t^4 (Sigma_x + sigma_t^2 I)^{-1}
        Expanded via Woodbury into the alpha/beta form above.
        """
        c = self.D_diag + sigma_t_sq
        alpha = sigma_t_sq * self.D_diag / c
        beta = sigma_t_sq ** 2 / c ** 2

        M = torch.eye(self.rank, device=self.U.device, dtype=self.U.dtype) + self.UtU / c
        M_inv = torch.linalg.inv(M)

        Ut_z = z_flat @ self.U             # (B, rank)
        M_inv_Ut_z = Ut_z @ M_inv          # (B, rank)  — M is symmetric
        return alpha * z_flat + beta * (M_inv_Ut_z @ self.U.T)


def ppca_fit_images(reconstructions, uncond_model, device, rank=32,
                    batch_size=64):
    """
    Fit PPCA on image-space representations of reconstructed time series.

    Returns a PPCACov object for use in the posterior denoiser.
    """
    N, T, C = reconstructions.shape
    all_imgs = []

    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_ts = torch.tensor(reconstructions[start:end],
                                dtype=torch.float32, device=device)
            x_img = uncond_model.ts_to_img(x_ts)
            all_imgs.append(x_img.cpu())

    imgs = torch.cat(all_imgs, dim=0)  # (N, C_img, H, W)
    X = imgs.reshape(N, -1).numpy()    # (N, D)
    D = X.shape[1]

    mu = np.mean(X, axis=0)
    X_c = X - mu

    if N < D:
        C_mat = X_c @ X_c.T / N
    else:
        C_mat = X_c.T @ X_c / N

    rank = min(rank, min(N, D) - 1, D)
    eigvals, eigvecs = np.linalg.eigh(C_mat)
    eigvals = eigvals[-rank:][::-1]
    eigvecs = eigvecs[:, -rank:][:, ::-1]

    if N < D:
        eigvecs = X_c.T @ eigvecs
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)

    if rank < D:
        D_diag = max((np.trace(C_mat) - np.sum(eigvals)) / (D - rank), 1e-6)
    else:
        D_diag = 1e-6

    U_np = eigvecs * np.sqrt(np.maximum(eigvals - D_diag, 0.0))[None, :]
    U_t = torch.tensor(U_np, dtype=torch.float64, device=device)

    var_explained = np.sum(np.maximum(eigvals - D_diag, 0.0))
    total_var = var_explained + D_diag * D
    print(f"PPCA posterior fit: rank={rank}, D_diag={D_diag:.4e}, "
          f"var_ratio={var_explained / total_var:.3f}, D={D}")

    return PPCACov(U_t, D_diag, device=device)


# =============================================================================
# Conjugate Gradient Solver
# =============================================================================

def conjugate_gradient(A_fn, b, maxiter=1):
    """
    Conjugate gradient solver for A*x = b where A is a linear operator.

    For maxiter=1, reduces to a single scaled residual step:
        x = (r^T r / r^T A r) * r  where r = b
    """
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


# =============================================================================
# MMPSDiffusionProcess — Generalized for any ForwardOperator
# =============================================================================

class MMPSDiffusionProcess:
    """
    MMPS-style diffusion sampler using a general forward operator A.

    Uses the EDM/Karras sigma schedule. At each reverse step, the unconditional
    denoiser is replaced by a posterior denoiser incorporating the observation
    model y = A(x) + epsilon via MMPS.

    MMPS approximation:
        p(y | x_t) ~ N(y | A * D_theta(x_t), sigma_y^2 I + A * V[x0|xt] * A^T)
    where V[x0|xt] = sigma_t^2 * J_D^T  (Jacobian of denoiser scaled by sigma^2)
    """

    def __init__(self, args, diffusion_fn, shape, cg_iters=1,
                 ppca_cov=None):
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

        self.cg_iters = cg_iters
        self.ppca_cov = ppca_cov
        self._ambient_concat = bool(
            getattr(args, 'ambient_concat_further_mask', False))
        self._data_c = int(shape[0])

    def _net_input_ambient(self, x_data, operator):
        """Match TS2img_Karras when trained with an extra mask channel."""
        if not self._ambient_concat:
            return x_data
        if getattr(operator, 'mask', None) is not None:
            m = operator.mask
            if m.dtype != x_data.dtype:
                m = m.to(dtype=x_data.dtype)
            return concat_mask_channel(x_data, m, True)
        return concat_ones_mask(x_data, True)

    def _trunc_vjp(self, grad):
        if not self._ambient_concat:
            return grad
        return grad[:, : self._data_c, :, :]

    def posterior_denoise(self, x_t, sigma, y_obs, operator):
        """
        Posterior denoiser with general forward operator A.

        Generalizes the mask-based version:
          1. Forward pass: denoised = D_theta(x_t)
          2. VJP via autograd.grad (robust to in-place UNet ops)
          3. Residual: r = y - A(denoised)
          4. CG solve: (sigma_y^2 I + sigma_t^2 * A(J^T(A^T(v)))) v = r
          5. Score: J^T(A^T(v))
          6. Return: denoised + sigma_t^2 * score

        Args:
            x_t: noisy input (B, C, H, W), float64
            sigma: noise level scalar
            y_obs: observation in measurement space, float64
            operator: ForwardOperator instance

        Returns:
            posterior denoised estimate (B, C, H, W), float64
        """
        sigma_sq = sigma ** 2
        sigma_y_sq = operator.sigma_y ** 2

        x_data = x_t.detach().requires_grad_(True)
        x_t_input = self._net_input_ambient(x_data, operator)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return self._trunc_vjp(grad)

        r = y_obs - operator.forward(denoised)

        def cg_operator(v):
            return sigma_y_sq * v + sigma_sq * operator.forward(
                vjp_fn(operator.adjoint(v))
            )

        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)

        score = vjp_fn(operator.adjoint(v))

        result = denoised.detach() + sigma_sq * score.detach()
        return result

    def posterior_denoise_ppca(self, x_t, sigma, y_obs, operator):
        """
        Posterior denoiser using PPCA covariance instead of Jacobian VJP.

        The CG operator uses the pre-fitted data covariance via Woodbury identity,
        so CG iterations are gradient-free.  Only the final score still requires
        one VJP call through the denoiser.

        Sigma_{x|t} replaces the Jacobian-based sigma_t^2 * J^T in the CG operator:
            CG op: sigma_y^2 v + A( Sigma_{x|t} @ A^T(v) )
        Final score: J^T( A^T(v) )   — one VJP through the network
        """
        sigma_sq = sigma ** 2
        sigma_y_sq = operator.sigma_y ** 2
        ppca = self.ppca_cov

        x_data = x_t.detach().requires_grad_(True)
        x_t_input = self._net_input_ambient(x_data, operator)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return self._trunc_vjp(grad)

        r = y_obs - operator.forward(denoised)
        img_shape = denoised.shape  # (B, C, H, W)

        def cg_operator(v):
            At_v = operator.adjoint(v)                               # -> (B,C,H,W)
            At_v_flat = At_v.reshape(At_v.shape[0], -1)              # -> (B, D)
            cov_At_v_flat = ppca.cov_x_xt_mv(At_v_flat, sigma_sq)   # -> (B, D)
            cov_At_v = cov_At_v_flat.reshape(img_shape)              # -> (B,C,H,W)
            return sigma_y_sq * v + operator.forward(cov_At_v)

        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)

        score = vjp_fn(operator.adjoint(v))

        result = denoised.detach() + sigma_sq * score.detach()
        return result

    def mmps_sample(self, latents, y_obs, operator):
        """
        MMPS sampling with general forward operator.

        Args:
            latents: initial noise (B, C, H, W)
            y_obs: observation tensor, float64
            operator: ForwardOperator instance

        Returns:
            reconstructed result (B, C, H, W)
        """
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64,
                                     device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps),
                             torch.zeros_like(t_steps[:1])])

        y = y_obs.to(torch.float64)

        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) \
                if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * \
                    self.S_noise * torch.randn_like(x_cur)

            denoise_fn = (self.posterior_denoise_ppca
                          if self.ppca_cov is not None
                          else self.posterior_denoise)
            denoised = denoise_fn(x_hat, t_hat, y, operator)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised_2 = denoise_fn(x_next, t_next, y, operator)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sampling_mmps(self, y_obs, operator):
        """Convenience wrapper for MMPS sampling (no @torch.no_grad — needs grads)."""
        batch_size = y_obs.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, y_obs, operator)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        """Standard unconditional sampling (for evaluation)."""
        latents = torch.randn([sampling_number, *self.shape], device=self.device)

        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64,
                                     device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps),
                             torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) \
                if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * \
                    self.S_noise * torch.randn_like(x_cur)

            x_in = concat_ones_mask(x_hat, self._ambient_concat)
            denoised = self.net(x_in, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                x_in2 = concat_ones_mask(x_next, self._ambient_concat)
                denoised = self.net(x_in2, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


# =============================================================================
# Data helpers
# =============================================================================

def get_corrupted_data_from_loader(train_loader, device):
    """Extract NaN-corrupted time series + boolean masks from the train loader."""
    all_corrupted = []
    all_masks = []

    for batch_idx, data in enumerate(train_loader):
        x_irregular = data[0]
        x_ts = x_irregular[:, :, :-1]

        mask = ~torch.isnan(x_ts).any(dim=-1)

        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())

    corrupted_data = np.vstack(all_corrupted)
    obs_masks = np.vstack(all_masks)

    return corrupted_data, obs_masks


def get_clean_data_from_loader(loader):
    """Extract clean time series from the test/original loader.

    The test_loader is built from ori_data which does NOT have a time-index
    column (that column is only appended to irregular_data in gen_dataloader).
    """
    all_data = []
    for data in loader:
        all_data.append(data[0].numpy())
    return np.vstack(all_data)


def initialize_with_gaussian(corrupted_data, obs_masks, seed=None):
    """
    Initialize reconstructions using Gaussian prior.
    Observed values kept; missing values sampled from per-feature Gaussian.
    """
    if seed is not None:
        np.random.seed(seed)

    N, seq_len, features = corrupted_data.shape

    mu = np.nanmean(corrupted_data, axis=(0, 1))
    std = np.nanstd(corrupted_data, axis=(0, 1))
    std = np.where(std == 0, 1.0, std)

    initial_recon = corrupted_data.copy()

    for i in range(N):
        for t in range(seq_len):
            if not obs_masks[i, t]:
                initial_recon[i, t, :] = np.random.normal(mu, std)

    initial_recon = np.clip(initial_recon, 0, 1)

    print(f"Gaussian initialization complete. Generated {N} sequences.")
    return initial_recon


# =============================================================================
# Corruption-specific data preparation
# =============================================================================

def prepare_data_missing(args, train_loader, uncond_model, device):
    """
    Prepare data for missing-data corruption (original behavior).

    Returns:
        clean_data_ts: None (not available for missing)
        corrupted_data_ts: (N, T, C) with NaN
        obs_masks: (N, T) boolean
        y_obs_img_list: None (built per-batch in E-step)
        operator_factory: callable(batch_mask_img) -> MissingDataOperator
    """
    corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, device)
    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)

    def operator_factory(mask_img):
        return MissingDataOperator(mask_img, sigma_y=sigma_y)

    return None, corrupted_data, obs_masks, operator_factory


def prepare_data_gaussian_noise(args, train_loader, test_loader, uncond_model, device):
    """
    Prepare data for Gaussian noise corruption: y = x + N(0, sigma_y^2 I).

    Noise is added in IMAGE space so the MMPS likelihood model (i.i.d. Gaussian
    noise with A=I in the diffusion model's operating space) is exactly correct.

    Returns:
        clean_data_ts: (N, T, C) original clean data
        corrupted_imgs: (N, C, H, W) noisy observations in image space
        obs_masks: (N, T) all True
        operator_factory: callable() -> GaussianNoiseOperator
    """
    clean_data = get_clean_data_from_loader(test_loader)
    sigma_y = args.corruption_noise_level

    N = clean_data.shape[0]
    batch_size = args.batch_size
    noisy_img_list = []

    torch.manual_seed(args.seed)
    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_ts = torch.tensor(clean_data[start:end], dtype=torch.float32,
                                device=device)
            x_img = uncond_model.ts_to_img(x_ts)
            noise_img = sigma_y * torch.randn_like(x_img)
            y_img = x_img + noise_img
            noisy_img_list.append(y_img.cpu())

    corrupted_imgs = torch.cat(noisy_img_list, dim=0).numpy()
    obs_masks = np.ones((N, clean_data.shape[1]), dtype=bool)

    def operator_factory(**kwargs):
        return GaussianNoiseOperator(sigma_y=sigma_y)

    print(f"Gaussian noise corruption (image-space): sigma_y={sigma_y}, "
          f"{N} sequences")
    return clean_data, corrupted_imgs, obs_masks, operator_factory


def prepare_data_gaussian_blur(args, train_loader, test_loader, uncond_model, device):
    """
    Prepare data for Gaussian blur corruption: y = blur(x) + epsilon.

    Blur and measurement noise are applied in IMAGE space and stored as images
    to avoid lossy ts_to_img / img_to_ts round-trips.

    Returns:
        clean_data_ts: (N, T, C) original clean data
        corrupted_imgs: (N, C, H, W) blurred observations in image space
        obs_masks: (N, T) all True
        operator_factory: callable() -> GaussianBlurOperator
    """
    clean_data = get_clean_data_from_loader(test_loader)
    sigma_y = getattr(args, 'corruption_noise_level', 0.01)
    blur_sigma = getattr(args, 'blur_sigma', 2.0)
    blur_kernel_size = getattr(args, 'blur_kernel_size', None)

    if blur_kernel_size is None:
        blur_kernel_size = int(4 * blur_sigma + 1)
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1

    num_channels = args.input_channels

    blur_op = GaussianBlurOperator(
        kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        num_channels=num_channels,
        sigma_y=sigma_y,
        device=device
    )

    N = clean_data.shape[0]
    batch_size = args.batch_size
    blurred_img_list = []

    torch.manual_seed(args.seed + 1)
    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_ts = torch.tensor(clean_data[start:end], dtype=torch.float32,
                                device=device)
            x_img = uncond_model.ts_to_img(x_ts)

            y_img = blur_op.forward(x_img)
            if sigma_y > 0:
                y_img = y_img + sigma_y * torch.randn_like(y_img)

            blurred_img_list.append(y_img.cpu())

    corrupted_imgs = torch.cat(blurred_img_list, dim=0).numpy()
    obs_masks = np.ones((N, clean_data.shape[1]), dtype=bool)

    def operator_factory(**kwargs):
        return blur_op

    print(f"Gaussian blur corruption (image-space): kernel_size={blur_kernel_size}, "
          f"blur_sigma={blur_sigma}, sigma_y={sigma_y}, {N} sequences")
    return clean_data, corrupted_imgs, obs_masks, operator_factory


def prepare_data_random_projection(args, train_loader, test_loader,
                                    uncond_model, device):
    """
    Prepare data for random projection corruption: y = A*x + epsilon.

    Returns:
        clean_data_ts: (N, T, C) original
        projected_data: list of (N, d_obs) projected observations
        obs_masks: (N, T) all True
        operator_factory: callable() -> RandomProjectionOperator
    """
    clean_data = get_clean_data_from_loader(test_loader)
    sigma_y = getattr(args, 'corruption_noise_level', 0.01)
    projection_dim = getattr(args, 'projection_dim', None)

    img_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    d_full = math.prod(img_shape)

    if projection_dim is None:
        projection_dim = d_full // 2

    proj_op = RandomProjectionOperator(
        d_obs=projection_dim,
        img_shape=img_shape,
        sigma_y=sigma_y,
        device=device,
        seed=args.seed
    )

    N = clean_data.shape[0]
    batch_size = args.batch_size
    projected_list = []

    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_ts = torch.tensor(clean_data[start:end], dtype=torch.float32,
                                device=device)
            x_img = uncond_model.ts_to_img(x_ts)
            y = proj_op.corrupt(x_img)
            projected_list.append(y.cpu().numpy())

    projected_data = np.vstack(projected_list)
    obs_masks = np.ones((N, clean_data.shape[1]), dtype=bool)

    def operator_factory(**kwargs):
        return proj_op

    print(f"Random projection: d_obs={projection_dim}, d_full={d_full}, "
          f"sigma_y={sigma_y}, {N} sequences")
    return clean_data, projected_data, obs_masks, operator_factory


def prepare_data_ts_gaussian_noise(args, train_loader, test_loader,
                                   uncond_model, device):
    """
    Gaussian noise applied in TS space, then converted to image space.

    y_ts = clip(x_ts + N(0, sigma^2 I), 0, 1)
    y_img = ts_to_img(y_ts)

    MMPS uses A=I in image space with sigma_y = sigma_noise.
    The noise inherits delay-embedding correlation structure, which is more
    natural than i.i.d. image-space noise.

    Returns:
        clean_data_ts: (N, T, C)
        corrupted_imgs: (N, C, H, W) noisy observations in image space
        obs_masks: (N, T) all True
        operator_factory: callable() -> GaussianNoiseOperator
    """
    clean_data = get_clean_data_from_loader(test_loader)
    sigma_noise = args.corruption_noise_level

    N = clean_data.shape[0]
    batch_size = args.batch_size
    noisy_img_list = []

    torch.manual_seed(args.seed)
    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_ts = torch.tensor(clean_data[start:end], dtype=torch.float32,
                                device=device)
            noise_ts = sigma_noise * torch.randn_like(x_ts)
            y_ts = torch.clamp(x_ts + noise_ts, 0, 1)
            y_img = uncond_model.ts_to_img(y_ts)
            noisy_img_list.append(y_img.cpu())

    corrupted_imgs = torch.cat(noisy_img_list, dim=0).numpy()
    obs_masks = np.ones((N, clean_data.shape[1]), dtype=bool)

    sigma_y = sigma_noise

    def operator_factory(**kwargs):
        return GaussianNoiseOperator(sigma_y=sigma_y)

    print(f"TS-space Gaussian noise: sigma_noise={sigma_noise}, "
          f"sigma_y={sigma_y}, {N} sequences")
    return clean_data, corrupted_imgs, obs_masks, operator_factory


def prepare_data_ts_temporal_smoothing(args, train_loader, test_loader,
                                       uncond_model, device):
    """
    Temporal moving-average smoothing applied in TS space.

    y_ts[t] = mean(x_ts[t-w//2 : t+w//2+1]) per feature, reflected edges.

    The smoothing is linear and self-adjoint (symmetric kernel), so the
    TSCompositeOperator bridges TS and image space for the MMPS CG solver.
    Observations are stored in TS space (N, T, C).

    Returns:
        clean_data_ts: (N, T, C)
        corrupted_ts: (N, T, C) smoothed observations in TS space
        obs_masks: (N, T) all True
        operator_factory: callable(uncond_model=) -> TSCompositeOperator
    """
    clean_data = get_clean_data_from_loader(test_loader)
    window = getattr(args, 'smoothing_window', 5)
    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)

    N = clean_data.shape[0]
    batch_size = args.batch_size
    smoothed_list = []

    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_ts = torch.tensor(clean_data[start:end], dtype=torch.float32,
                                device=device)
            y_ts = temporal_moving_average(x_ts, window)
            smoothed_list.append(y_ts.cpu().numpy())

    corrupted_ts = np.vstack(smoothed_list)
    obs_masks = np.ones((N, clean_data.shape[1]), dtype=bool)

    def smooth_fn(x_ts):
        return temporal_moving_average(x_ts, window)

    def operator_factory(uncond_model=uncond_model, **kwargs):
        return TSCompositeOperator(
            ts_forward_fn=smooth_fn,
            ts_adjoint_fn=smooth_fn,   # self-adjoint: symmetric kernel
            uncond_model=uncond_model,
            sigma_y=sigma_y,
        )

    print(f"TS-space temporal smoothing: window={window}, "
          f"sigma_y={sigma_y}, {N} sequences")
    return clean_data, corrupted_ts, obs_masks, operator_factory


def prepare_data_ts_missing_noise(args, train_loader, test_loader,
                                  uncond_model, device):
    """
    Combined missing data + additive Gaussian noise on observed entries.

    y = mask * (x + N(0, sigma^2 I)); missing entries remain NaN.
    Forward operator: A = diag(mask), sigma_y = sqrt(sigma_noise^2 + base^2).

    Returns:
        clean_data_ts: None (clean not available from train_loader)
        corrupted_data_ts: (N, T, C) with NaN for missing + noise on observed
        obs_masks: (N, T) boolean
        operator_factory: callable(mask_img) -> MissingDataOperator
    """
    corrupted_data, obs_masks = get_corrupted_data_from_loader(
        train_loader, device
    )
    sigma_noise = getattr(args, 'corruption_noise_level', 0.05)
    base_sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    sigma_y = math.sqrt(sigma_noise ** 2 + base_sigma_y ** 2)

    np.random.seed(args.seed)
    noise = sigma_noise * np.random.randn(*corrupted_data.shape).astype(
        np.float32
    )
    observed = ~np.isnan(corrupted_data)
    corrupted_data = np.where(observed, corrupted_data + noise, corrupted_data)
    corrupted_data = np.where(observed,
                              np.clip(corrupted_data, 0, 1),
                              corrupted_data)

    def operator_factory(mask_img):
        return MissingDataOperator(mask_img, sigma_y=sigma_y)

    print(f"TS-space missing + noise: sigma_noise={sigma_noise}, "
          f"sigma_y={sigma_y:.4f}, {corrupted_data.shape[0]} sequences, "
          f"observed_rate={obs_masks.mean():.2%}")
    return None, corrupted_data, obs_masks, operator_factory


# =============================================================================
# Difficulty Scoring (STL-based curriculum)
# =============================================================================

def compute_difficulty_scores(initial_reconstructions, obs_masks, period=None):
    """
    Score each sample by how much of its variance STL cannot explain
    (residual/total ratio).  Works for any corruption type that produces
    full time-series estimates.
    """
    N, T, C = initial_reconstructions.shape
    if period is None:
        period = auto_detect_period(T)

    difficulty_scores = np.zeros(N)

    for i in range(N):
        total_var = 0.0
        resid_var = 0.0
        for c in range(C):
            series = initial_reconstructions[i, :, c]
            _, _, resid_c = stl_decompose_single(series, period)

            total_var += np.var(series) + 1e-8
            resid_var += np.var(resid_c)

        difficulty_scores[i] = resid_var / total_var

    sorted_indices = np.argsort(difficulty_scores)

    easy_thresh = np.percentile(difficulty_scores, 33)
    hard_thresh = np.percentile(difficulty_scores, 67)
    n_easy = (difficulty_scores <= easy_thresh).sum()
    n_hard = (difficulty_scores > hard_thresh).sum()
    n_medium = N - n_easy - n_hard

    print(f"Difficulty scoring complete (period={period}):")
    print(f"  Min: {difficulty_scores.min():.4f}, Max: {difficulty_scores.max():.4f}, "
          f"Mean: {difficulty_scores.mean():.4f}")
    print(f"  Easy (bottom 33%): {n_easy}, Medium: {n_medium}, Hard (top 33%): {n_hard}")

    return sorted_indices, difficulty_scores


# =============================================================================
# Initialization per corruption type
# =============================================================================

def initialize_reconstructions(corruption_type, corrupted_data, obs_masks,
                                clean_data_ts, uncond_model, operator_factory,
                                device, seed=None):
    """
    Produce initial reconstructions (in TS space) for the curriculum scoring.

    - missing:            STL-informed imputation (original)
    - gaussian_noise:     Convert noisy images to TS (already close to clean)
    - gaussian_blur:      Convert blurred images to TS (blurred but structured)
    - random_projection:  Compute A^T y as minimum-norm estimate
    - ts_gaussian_noise:  Convert noisy images to TS (noise applied in TS space)
    - ts_temporal_smoothing: Smoothed TS is the initial reconstruction
    - ts_missing_noise:   STL-informed imputation (same as missing)
    """
    if corruption_type in ('missing', 'ts_missing_noise'):
        return initialize_with_stl(corrupted_data, obs_masks, seed=seed)

    elif corruption_type in ('gaussian_noise', 'gaussian_blur', 'ts_gaussian_noise'):
        # corrupted_data is in image space (N, C, H, W) — convert to TS
        print(f"{corruption_type} init: converting image-space observations to TS.")
        N = corrupted_data.shape[0]
        batch_size = 64
        all_ts = []

        uncond_model.eval()
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                y_img = torch.tensor(corrupted_data[start:end],
                                     dtype=torch.float32, device=device)
                y_ts = uncond_model.img_to_ts(y_img)
                all_ts.append(y_ts.cpu().numpy())

        result = np.vstack(all_ts)
        result = np.clip(result, 0, 1)
        print(f"  Initial estimates: {result.shape}")
        return result

    elif corruption_type == 'ts_temporal_smoothing':
        # corrupted_data is in TS space (N, T, C), already a reasonable init
        print("Temporal smoothing init: using smoothed TS as initial reconstruction.")
        result = np.clip(corrupted_data.copy(), 0, 1)
        print(f"  Initial estimates: {result.shape}")
        return result

    elif corruption_type == 'random_projection':
        print("Random projection init: computing A^T y estimates.")
        return _pseudoinverse_init(corrupted_data, operator_factory,
                                    uncond_model, device)

    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


def _pseudoinverse_init(projected_data, operator_factory, uncond_model, device):
    """
    Compute A^T y as a minimum-norm initialization for random projections.
    Converts the back-projected observations from image space to TS.
    """
    print("  Computing A^T y initialization...")
    N = projected_data.shape[0]
    batch_size = 32
    all_ts = []

    operator = operator_factory()

    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            y = torch.tensor(projected_data[start:end],
                             dtype=torch.float32, device=device)
            x_img = operator.adjoint(y)  # A^T y -> (B, C, H, W)
            ts_init = uncond_model.img_to_ts(x_img)
            all_ts.append(ts_init.cpu().numpy())

    result = np.vstack(all_ts)
    result = np.clip(result, 0, 1)
    print(f"  A^T y init complete: {result.shape}")
    return result


# =============================================================================
# Core EM functions
# =============================================================================

def _run_mmps_batch(process, target_shape, y_obs, operator, device):
    """Run a single MMPS sampling pass for one batch."""
    B = y_obs.shape[0]
    latents = torch.randn([B, *target_shape], device=device)
    return process.mmps_sample(latents, y_obs, operator)


def e_step(args, uncond_model, corrupted_obs, obs_masks,
           operator_factory, em_iter, device, logger=None,
           ppca_cov=None, sigma_y_override=None):
    """
    E-step: Reconstruct clean data using MMPS with the unconditional model
    and the specified forward operator.

    Enhancements controlled by args:
        --num_posterior_samples K : draw K posterior samples and average
        --obs_consistency_mmps    : overwrite observed positions with true values
        --use_ppca_posterior      : use PPCA covariance in CG operator

    Args:
        corrupted_obs: corrupted observations. Shape depends on corruption type:
            missing:              (N, T, C) TS space with NaN
            gaussian_noise:       (N, C, H, W) image space
            gaussian_blur:        (N, C, H, W) image space
            random_projection:    (N, d_obs) observation space
            ts_gaussian_noise:    (N, C, H, W) image space (noise in TS, then ts_to_img)
            ts_temporal_smoothing:(N, T, C) smoothed TS space
            ts_missing_noise:     (N, T, C) TS space with NaN + noise on observed
        ppca_cov: optional PPCACov object for posterior denoiser
        sigma_y_override: if set, override operator sigma_y for annealing
    """
    corruption_type = getattr(args, 'corruption_type', 'missing')
    K = getattr(args, 'num_posterior_samples', 1)
    obs_consist = getattr(args, 'obs_consistency_mmps', False)

    print(f"\n=== E-Step (EM iter {em_iter}) — MMPS Reconstruction "
          f"[{corruption_type}] ===")
    if K > 1:
        print(f"  Multi-sample averaging: K={K}")
    if obs_consist:
        print(f"  Observation consistency: ON")
    if ppca_cov is not None:
        print(f"  PPCA posterior covariance: ON")

    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution,
                    args.img_resolution)

    N = len(corrupted_obs)
    batch_size = getattr(args, 'e_step_batch_size', 64)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    with uncond_model.ema_scope():
        process = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            cg_iters=cg_iters,
            ppca_cov=ppca_cov
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step MMPS"):
            end_idx = min(start_idx + batch_size, N)
            B = end_idx - start_idx

            if corruption_type == 'missing':
                corrupted_batch = corrupted_obs[start_idx:end_idx]
                mask_batch = obs_masks[start_idx:end_idx]

                corrupted_ts = torch.tensor(
                    corrupted_batch, dtype=torch.float32, device=device)
                mask_ts = torch.tensor(
                    mask_batch, dtype=torch.float32, device=device)

                obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)
                y_obs_img = uncond_model.ts_to_img(obs_ts)

                mask_ts_expanded = mask_ts.unsqueeze(-1).expand(
                    -1, -1, corrupted_ts.shape[-1])
                mask_img = uncond_model.ts_to_img(mask_ts_expanded)
                mask_img = mask_img[:, :1, :, :]

                operator = operator_factory(mask_img.to(torch.float64))
                if sigma_y_override is not None:
                    operator.sigma_y = sigma_y_override
                y_obs_64 = y_obs_img.to(torch.float64)

            elif corruption_type in ('gaussian_noise', 'gaussian_blur',
                                     'ts_gaussian_noise'):
                batch_imgs = corrupted_obs[start_idx:end_idx]
                y_obs_64 = torch.tensor(
                    batch_imgs, dtype=torch.float64, device=device)
                operator = operator_factory()
                if sigma_y_override is not None:
                    operator.sigma_y = sigma_y_override

            elif corruption_type == 'ts_temporal_smoothing':
                batch_ts = corrupted_obs[start_idx:end_idx]
                y_obs_64 = torch.tensor(
                    batch_ts, dtype=torch.float64, device=device)
                operator = operator_factory(uncond_model=uncond_model)
                if sigma_y_override is not None:
                    operator.sigma_y = sigma_y_override

            elif corruption_type == 'ts_missing_noise':
                corrupted_batch = corrupted_obs[start_idx:end_idx]
                mask_batch = obs_masks[start_idx:end_idx]

                corrupted_ts = torch.tensor(
                    corrupted_batch, dtype=torch.float32, device=device)
                mask_ts = torch.tensor(
                    mask_batch, dtype=torch.float32, device=device)

                obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)
                y_obs_img = uncond_model.ts_to_img(obs_ts)

                mask_ts_expanded = mask_ts.unsqueeze(-1).expand(
                    -1, -1, corrupted_ts.shape[-1])
                mask_img = uncond_model.ts_to_img(mask_ts_expanded)
                mask_img = mask_img[:, :1, :, :]

                operator = operator_factory(mask_img.to(torch.float64))
                if sigma_y_override is not None:
                    operator.sigma_y = sigma_y_override
                y_obs_64 = y_obs_img.to(torch.float64)

            elif corruption_type == 'random_projection':
                batch_proj = corrupted_obs[start_idx:end_idx]
                y_obs_64 = torch.tensor(
                    batch_proj, dtype=torch.float64, device=device)
                operator = operator_factory()
                if sigma_y_override is not None:
                    operator.sigma_y = sigma_y_override

            else:
                raise ValueError(
                    f"Unknown corruption type: {corruption_type}")

            # --- Multi-sample averaging ---
            if K > 1:
                samples = []
                for _ in range(K):
                    s = _run_mmps_batch(
                        process, target_shape, y_obs_64, operator, device)
                    samples.append(s)
                x_img_recon = torch.stack(samples).mean(dim=0)
            else:
                x_img_recon = _run_mmps_batch(
                    process, target_shape, y_obs_64, operator, device)

            # --- Observation consistency (missing data types) ---
            if obs_consist and corruption_type in ('missing', 'ts_missing_noise'):
                x_ts_recon = uncond_model.img_to_ts(x_img_recon)
                x_ts_np = x_ts_recon.cpu().numpy()
                obs_np = np.nan_to_num(
                    corrupted_obs[start_idx:end_idx], nan=0.0)
                mask_np = obs_masks[start_idx:end_idx]
                x_ts_np = np.where(
                    mask_np[..., None] if x_ts_np.ndim > mask_np.ndim
                    else mask_np,
                    obs_np, x_ts_np
                )
                all_reconstructions.append(x_ts_np)
            else:
                x_ts_recon = uncond_model.img_to_ts(x_img_recon)
                all_reconstructions.append(x_ts_recon.cpu().numpy())

            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    print(f"E-step complete. Reconstructed {len(reconstructions)} sequences.")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    cache_dir = os.path.join(
        args.recon_cache_dir, args.dataset,
        _cache_subdir(args)
    )
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def _cache_subdir(args):
    """Build a descriptive cache subdirectory based on corruption type."""
    corruption_type = getattr(args, 'corruption_type', 'missing')
    if corruption_type == 'missing':
        return f'missing_{int(args.missing_rate * 100)}'
    elif corruption_type == 'gaussian_noise':
        level = getattr(args, 'corruption_noise_level', 0.1)
        return f'gaussian_noise_{level}'
    elif corruption_type == 'gaussian_blur':
        sigma = getattr(args, 'blur_sigma', 2.0)
        return f'gaussian_blur_s{sigma}'
    elif corruption_type == 'random_projection':
        dim = getattr(args, 'projection_dim', 0)
        return f'random_proj_d{dim}'
    elif corruption_type == 'ts_gaussian_noise':
        level = getattr(args, 'corruption_noise_level', 0.1)
        return f'ts_noise_{level}'
    elif corruption_type == 'ts_temporal_smoothing':
        window = getattr(args, 'smoothing_window', 5)
        return f'ts_smooth_w{window}'
    elif corruption_type == 'ts_missing_noise':
        level = getattr(args, 'corruption_noise_level', 0.05)
        rate = getattr(args, 'missing_rate', 0.3)
        return f'ts_missing{int(rate*100)}_noise{level}'
    else:
        return corruption_type


def m_step(args, uncond_model, optimizer, reconstructions, em_iter,
           device, logger=None):
    """
    M-step: Train unconditional model on fully reconstructed data.
    Identical to original — the model learns the clean data distribution
    regardless of corruption type.
    """
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training unconditional model for {args.m_step_epochs} epochs "
          f"on {len(reconstructions)} samples...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    uncond_model.train()

    for epoch in range(args.m_step_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (x_clean,) in enumerate(recon_loader):
            x_clean = x_clean.to(device)

            x_img = uncond_model.ts_to_img(x_clean)

            loss, to_log = uncond_model.loss_fn_irregular(x_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss += loss.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  M-step epoch {epoch+1}/{args.m_step_epochs}, "
                  f"loss: {avg_loss:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, em_iter, device,
                    logger=None):
    """Evaluate the unconditional model via standard unconditional sampling."""
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    print("Evaluating unconditional model...")

    uncond_model.eval()
    gen_sig = []
    real_sig = []

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution, args.img_resolution)
            )

            for data in tqdm(test_loader, desc="Evaluating"):
                x_img_sampled = process.sampling(
                    sampling_number=data[0].shape[0]
                )
                x_ts = uncond_model.img_to_ts(x_img_sampled)

                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    print(f"EM iter {em_iter} metrics (unconditional):")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")

    mem_plot_path = f"memorization_hist_em_iter_{em_iter}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig,
        generated_data=gen_sig,
        device=device,
        plot_path=mem_plot_path
    )

    print(f"EM iter {em_iter} memorization metrics:")
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if logger is not None:
        all_metrics = {}
        for key, value in scores.items():
            all_metrics[f'test/{key}'] = value
        for k, v in mem_stats.items():
            all_metrics[f'test/memorization/{k}'] = v
        logger.log_metrics(all_metrics, em_iter)

        upload_successful = False
        try:
            logger.log_file('test/memorization/histogram',
                            mem_plot_path, em_iter)
            upload_successful = True
        except Exception as e:
            print(f"Failed to upload memorization plot: {e}")

        if upload_successful:
            try:
                if os.path.exists(mem_plot_path):
                    os.remove(mem_plot_path)
            except Exception as e:
                print(f"Failed to delete temporary plot file: {e}")

    return scores


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    """Main DiffEM training loop with general corruption support."""

    corruption_type = getattr(args, 'corruption_type', 'missing')
    curriculum_warmup_frac = getattr(args, 'curriculum_warmup_frac', 0.5)
    curriculum_easy_frac = getattr(args, 'curriculum_easy_frac', 0.7)

    name = create_model_name_and_dir(args)
    name = f"diffem_general_{corruption_type}_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        print_model_params(logger, uncond_model)

        # === Prepare data based on corruption type ===
        print(f"\n{'='*60}")
        print(f"Corruption type: {corruption_type}")
        print(f"{'='*60}")

        # corrupted_obs: the corrupted observations. Shape depends on type:
        #   missing:              (N, T, C) TS space with NaN
        #   gaussian_noise:       (N, C, H, W) image space
        #   gaussian_blur:        (N, C, H, W) image space
        #   random_projection:    (N, d_obs) observation space
        #   ts_gaussian_noise:    (N, C, H, W) image space (noise in TS, then ts_to_img)
        #   ts_temporal_smoothing:(N, T, C) smoothed TS space
        #   ts_missing_noise:     (N, T, C) TS space with NaN + noise
        if corruption_type == 'missing':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_missing(args, train_loader, uncond_model, args.device)
            print(f"Extracted {len(corrupted_obs)} sequences with "
                  f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        elif corruption_type == 'gaussian_noise':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_gaussian_noise(
                    args, train_loader, test_loader,
                    uncond_model, args.device
                )

        elif corruption_type == 'gaussian_blur':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_gaussian_blur(
                    args, train_loader, test_loader,
                    uncond_model, args.device
                )

        elif corruption_type == 'random_projection':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_random_projection(
                    args, train_loader, test_loader,
                    uncond_model, args.device
                )

        elif corruption_type == 'ts_gaussian_noise':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_ts_gaussian_noise(
                    args, train_loader, test_loader,
                    uncond_model, args.device
                )

        elif corruption_type == 'ts_temporal_smoothing':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_ts_temporal_smoothing(
                    args, train_loader, test_loader,
                    uncond_model, args.device
                )

        elif corruption_type == 'ts_missing_noise':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_ts_missing_noise(
                    args, train_loader, test_loader,
                    uncond_model, args.device
                )

        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

        # === Phase 1: Initialize reconstructions ===
        print(f"\n{'='*60}")
        print("Phase 1: Initialization")
        print(f"{'='*60}")

        initial_reconstructions = initialize_reconstructions(
            corruption_type, corrupted_obs, obs_masks,
            clean_data_ts, uncond_model, operator_factory,
            args.device, seed=args.seed
        )

        # === Phase 1.2: Compute difficulty scores ===
        print(f"\n{'='*60}")
        print("Phase 1.2: Computing STL difficulty scores for curriculum")
        print(f"{'='*60}")

        sorted_indices, difficulty_scores = compute_difficulty_scores(
            initial_reconstructions, obs_masks
        )

        # === Phase 1.3 (optional): Fit PPCA posterior covariance ===
        use_ppca_post = getattr(args, 'use_ppca_posterior', False)
        ppca_cov = None

        if use_ppca_post:
            ppca_rank = getattr(args, 'ppca_posterior_rank', 32)
            print(f"\n{'='*60}")
            print(f"Phase 1.3: Fitting PPCA posterior covariance "
                  f"(rank={ppca_rank})")
            print(f"{'='*60}")
            ppca_cov = ppca_fit_images(
                initial_reconstructions, uncond_model,
                args.device, rank=ppca_rank
            )

        # === Sigma-y annealing schedule ===
        do_anneal = getattr(args, 'sigma_y_anneal', False)
        sigma_y_start = getattr(args, 'sigma_y_start', 0.1)
        sigma_y_end = getattr(args, 'sigma_y_end', 0.01)
        sigma_y_base = getattr(args, 'mmps_sigma_y', 0.01)

        def get_sigma_y(em_iter_idx, total_iters):
            if not do_anneal:
                return sigma_y_base
            frac = em_iter_idx / max(total_iters - 1, 1)
            return sigma_y_start * (sigma_y_end / sigma_y_start) ** frac

        # === Phase 1.5: Initial model training on easy samples ===
        n_easy_init = int(len(sorted_indices) * curriculum_easy_frac)
        easy_idx_init = sorted_indices[:n_easy_init]

        print(f"\n{'='*60}")
        print(f"Phase 1.5: Initial model training on "
              f"{n_easy_init}/{len(sorted_indices)} easiest samples")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions[easy_idx_init], em_iter=-1,
            device=args.device, logger=logger
        )
        print("Initial unconditional model training complete.")

        # === Phase 2: EM Loop with Curriculum ===
        warmup_end = int(args.em_iters * curriculum_warmup_frac)

        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS [{corruption_type}] with "
              f"{args.em_iters} EM iterations")
        print(f"  MMPS cg_iters: {getattr(args, 'mmps_cg_iters', 1)}")
        print(f"  Curriculum: warmup={warmup_end} iters, "
              f"easy_frac={curriculum_easy_frac}")
        if do_anneal:
            print(f"  sigma_y annealing: {sigma_y_start} → {sigma_y_end}")
        if use_ppca_post:
            print(f"  PPCA posterior: rank={ppca_rank}")
        print(f"{'='*60}")

        best_metrics = None
        metrics = None

        try:
            for em_iter in range(args.em_iters):
                print(f"\n{'='*60}")
                print(f"EM Iteration {em_iter + 1}/{args.em_iters}")

                # --- Sigma-y annealing ---
                current_sigma_y = get_sigma_y(em_iter, args.em_iters)
                sy_override = current_sigma_y if do_anneal else None
                if do_anneal:
                    print(f"  sigma_y = {current_sigma_y:.5f}")

                if em_iter < warmup_end:
                    n_use = int(len(sorted_indices) * curriculum_easy_frac)
                    phase_label = (f"CURRICULUM (using {n_use}/"
                                   f"{len(sorted_indices)} easiest samples)")
                else:
                    n_use = len(sorted_indices)
                    phase_label = f"FULL (using all {n_use} samples)"
                print(f"  {phase_label}")
                print(f"{'='*60}")

                # E-step: reconstruct ALL samples
                reconstructions = e_step(
                    args, uncond_model, corrupted_obs, obs_masks,
                    operator_factory, em_iter, args.device, logger,
                    ppca_cov=ppca_cov, sigma_y_override=sy_override
                )

                uncond_model.reset_ema()

                # Re-fit PPCA on updated reconstructions every few iterations
                if use_ppca_post and (em_iter + 1) % 3 == 0:
                    print("  Re-fitting PPCA posterior covariance ...")
                    ppca_cov = ppca_fit_images(
                        reconstructions, uncond_model,
                        args.device, rank=ppca_rank
                    )

                # M-step: train on curriculum-selected subset
                if em_iter < warmup_end:
                    train_idx = sorted_indices[:n_use]
                    uncond_model = m_step(
                        args, uncond_model, optimizer,
                        reconstructions[train_idx], em_iter,
                        args.device, logger
                    )
                else:
                    uncond_model = m_step(
                        args, uncond_model, optimizer,
                        reconstructions, em_iter,
                        args.device, logger
                    )

                if logger is not None:
                    logger.log('em/curriculum_n_samples', n_use, em_iter)
                    logger.log('em/curriculum_phase',
                               0 if em_iter < warmup_end else 1, em_iter)
                    if do_anneal:
                        logger.log('em/sigma_y', current_sigma_y, em_iter)

                if ((em_iter + 1) % args.em_eval_interval == 0 or
                        em_iter == args.em_iters - 1):
                    metrics = evaluate_uncond(
                        args, uncond_model, test_loader,
                        em_iter, args.device, logger
                    )

                    if (best_metrics is None or
                            metrics.get('disc_mean', float('inf')) <
                            best_metrics.get('disc_mean', float('inf'))):
                        best_metrics = metrics

                if logger is not None:
                    logger.log('em/iteration', em_iter, em_iter)

        except Exception as e:
            logging.error(f"EM loop failed at iteration {em_iter}: "
                          f"{type(e).__name__}: {e}", exc_info=True)
            if logger is not None:
                logger.log('em/failed_at_iter', em_iter, em_iter)
            raise

        # ================================================================
        # Phase 3 — Train unconditional model (same as run_regular.py)
        # ================================================================
        final_metrics = train_unconditional_regular(
            args, reconstructions, test_loader, args.device, logger,
        )
        if final_metrics:
            print("Phase 3 (unconditional) final metrics:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")

        print("\n" + "=" * 60)
        print(f"DiffEM-MMPS [{corruption_type}] + STL Curriculum Complete!")
        print("=" * 60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS general corruption training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    if not hasattr(args, 'corruption_type'):
        args.corruption_type = 'missing'
    if not hasattr(args, 'corruption_noise_level'):
        args.corruption_noise_level = 0.01
    if not hasattr(args, 'blur_sigma'):
        args.blur_sigma = 2.0
    if not hasattr(args, 'blur_kernel_size'):
        args.blur_kernel_size = None
    if not hasattr(args, 'projection_dim'):
        args.projection_dim = None
    if not hasattr(args, 'curriculum_warmup_frac'):
        args.curriculum_warmup_frac = 0.5
    if not hasattr(args, 'curriculum_easy_frac'):
        args.curriculum_easy_frac = 0.7
    if not hasattr(args, 'num_posterior_samples'):
        args.num_posterior_samples = 1
    if not hasattr(args, 'obs_consistency_mmps'):
        args.obs_consistency_mmps = False
    if not hasattr(args, 'use_ppca_posterior'):
        args.use_ppca_posterior = False
    if not hasattr(args, 'ppca_posterior_rank'):
        args.ppca_posterior_rank = 32
    if not hasattr(args, 'sigma_y_anneal'):
        args.sigma_y_anneal = False
    if not hasattr(args, 'sigma_y_start'):
        args.sigma_y_start = 0.1
    if not hasattr(args, 'sigma_y_end'):
        args.sigma_y_end = 0.01
    if not hasattr(args, 'smoothing_window'):
        args.smoothing_window = 5

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
