"""
DiffEM MMPS + STL-Guided Structural Correction

Same EM pipeline as run_diffem_mmps_ema_reset_iter_stl_init.py with ONE addition:
  After each MMPS posterior_denoise step, a lightweight structural correction
  nudges the denoised estimate toward trend/seasonal consistency with
  observations. The correction uses analytical gradients of a structural loss
  (moving-average trend + periodic-average seasonal) and spreads information
  from observed positions into missing gaps through the self-adjoint filter
  operators.

Key idea: MMPS enforces point-wise M*(x-y)=0 at observations.  The structural
correction additionally enforces that the *trend* and *seasonality* of x at
observed positions match those of the previous-iteration reconstruction,
propagating domain-specific structure into the gaps.
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.autograd
import torch.multiprocessing
import torch.nn.functional as F_nn
import torch.utils.data as Data
from tqdm import tqdm

from utils.diffem_mmps_irregular_em import (
    get_corrupted_data_from_loader,
    m_step,
    evaluate_uncond,
)
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader, save_reconstructions
from utils.utils_args import parse_args_irregular
from utils.utils_stl import auto_detect_period, initialize_with_iterative_stl
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.train_unconditional import train_unconditional_regular

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Pure-PyTorch structural decomposition operators
# =============================================================================

def torch_moving_avg(x, window):
    """
    Symmetric moving-average trend filter (pure PyTorch, same as fast_decompose_batch).

    Args:
        x: (N, T, C) tensor
        window: odd integer kernel width

    Returns:
        trend: (N, T, C) tensor
    """
    N, T, C = x.shape
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / window
    pad = window // 2
    # (N, T, C) -> (N*C, 1, T) for grouped conv1d
    x_flat = x.permute(0, 2, 1).reshape(N * C, 1, T)
    x_padded = F_nn.pad(x_flat, (pad, pad), mode='replicate')
    trend_flat = F_nn.conv1d(x_padded, kernel)
    return trend_flat.reshape(N, C, T).permute(0, 2, 1)


def torch_periodic_avg(x, period):
    """
    Periodic-averaging seasonal filter (pure PyTorch, same as fast_decompose_batch).

    For each phase k in [0, period), averages all values at that phase across
    the time axis and broadcasts back.

    Args:
        x: (N, T, C) tensor (typically the detrended signal)
        period: seasonal period

    Returns:
        seasonal: (N, T, C) tensor
    """
    N, T, C = x.shape
    seasonal = torch.zeros_like(x)
    for k in range(period):
        idx = torch.arange(k, T, period, device=x.device)
        phase_mean = x[:, idx, :].mean(dim=1, keepdim=True)  # (N, 1, C)
        seasonal[:, idx, :] = phase_mean
    return seasonal


def decompose_torch(x, period, window):
    """
    Fast structural decomposition: trend (moving avg) + seasonal (periodic avg).

    Args:
        x: (N, T, C) tensor — no NaN
        period: seasonal period
        window: moving-average window (odd integer)

    Returns:
        trend, seasonal, residual — each (N, T, C)
    """
    trend = torch_moving_avg(x, window)
    detrended = x - trend
    seasonal = torch_periodic_avg(detrended, period)
    residual = x - trend - seasonal
    return trend, seasonal, residual


# =============================================================================
# Conjugate Gradient Solver (identical to base MMPS)
# =============================================================================

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


# =============================================================================
# STL-Guided MMPS Diffusion Process
# =============================================================================

class STLGuidedMMPSDiffusionProcess:
    """
    MMPS posterior sampling with additional structural (trend + seasonal) guidance.

    Identical to MMPSDiffusionProcess except that after each posterior_denoise
    call (when sigma is below a threshold), a structural correction is applied
    that nudges the denoised estimate toward trend/seasonal consistency with
    the previous EM iteration's reconstruction at observed positions.

    The correction uses the analytical gradient of
        L = ||M*(T(x) - T_obs)||^2 + ||M*(S(x) - S_obs)||^2
    where T = moving average, S = periodic average of detrended signal,
    M = observation mask.  Since T and PA (periodic average) are self-adjoint
    linear operators, the gradient simplifies to:
        dL/dx = 2 * [ T(M*r_t) + PA(M*r_s) - T(PA(M*r_s)) ]
    which naturally spreads corrections from observed positions into gaps.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01, cg_iters=1,
                 struct_lambda=0.1, struct_sigma_threshold=10.0,
                 period=13, window=7):
        self.args = args
        self.device = args.device
        self.shape = shape
        self.net = diffusion_fn.to(device=self.device)

        # EDM parameters
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

        # MMPS parameters
        self.sigma_y = sigma_y
        self.cg_iters = cg_iters

        # Structural guidance parameters
        self.struct_lambda = struct_lambda
        self.struct_sigma_threshold = struct_sigma_threshold
        self.period = period
        self.window = window

        # Per-batch structural targets (set before each sampling_mmps call)
        self._mask_ts = None
        self._trend_obs = None
        self._season_obs = None
        self._uncond_model = None

    def set_batch_targets(self, mask_ts, trend_obs, season_obs, uncond_model):
        """
        Set structural targets for the current batch.

        Args:
            mask_ts: (batch, seq_len) float tensor, 1=observed
            trend_obs: (batch, seq_len, features) tensor, target trend
            season_obs: (batch, seq_len, features) tensor, target seasonal
            uncond_model: model with ts_to_img / img_to_ts methods
        """
        self._mask_ts = mask_ts
        self._trend_obs = trend_obs
        self._season_obs = season_obs
        self._uncond_model = uncond_model

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        """Standard MMPS posterior denoiser (unchanged from base)."""
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
        return denoised.detach() + sigma_sq * score.detach()

    def structural_correct(self, denoised_img, sigma):
        """
        Apply structural (trend + seasonal) correction to the denoised estimate.

        Converts to TS space, computes the analytical gradient of the structural
        loss, and applies an annealed correction.

        Args:
            denoised_img: (batch, C, H, W) float64 posterior-denoised image
            sigma: current noise level

        Returns:
            corrected image: (batch, C, H, W) float64
        """
        if self._uncond_model is None or self._mask_ts is None:
            return denoised_img

        model = self._uncond_model
        mask_ts = self._mask_ts        # (batch, T)
        trend_obs = self._trend_obs    # (batch, T, C)
        season_obs = self._season_obs  # (batch, T, C)

        # 1. Convert denoised image to time-series space
        x_ts = model.img_to_ts(denoised_img)  # (batch, T, C)

        # 2. Decompose current estimate
        trend_x, season_x, _ = decompose_torch(x_ts, self.period, self.window)

        # 3. Masked structural residuals
        M = mask_ts.unsqueeze(-1)  # (batch, T, 1) — broadcasts over C
        r_t = M * (trend_x - trend_obs.to(x_ts.device, x_ts.dtype))
        r_s = M * (season_x - season_obs.to(x_ts.device, x_ts.dtype))

        # 4. Analytical gradient of structural loss
        #    dL/dx = 2 * [ T(M*r_t) + PA(M*r_s) - T(PA(M*r_s)) ]
        grad_trend = torch_moving_avg(r_t, self.window)
        pa_rs = torch_periodic_avg(r_s, self.period)
        grad_season = pa_rs - torch_moving_avg(pa_rs, self.window)
        grad_ts = 2.0 * (grad_trend + grad_season)

        # 5. Annealing: ramp up correction as sigma decreases
        sigma_f = float(sigma)
        anneal = min(1.0, self.struct_sigma_threshold / sigma_f)
        scale = self.struct_lambda * anneal

        # 6. Apply correction in TS space, clamp to prevent blow-up
        correction_norm = grad_ts.norm()
        x_ts_norm = x_ts.norm().clamp(min=1e-6)
        max_ratio = 0.1
        if correction_norm * scale > max_ratio * x_ts_norm:
            scale = (max_ratio * x_ts_norm / (correction_norm + 1e-30)).item()

        x_ts_corrected = x_ts - scale * grad_ts

        # 7. Convert back to image space (float32 for ts_to_img, then back to float64)
        corrected_img = model.ts_to_img(x_ts_corrected.float()).to(torch.float64)
        return corrected_img

    def mmps_sample(self, latents, x_obs_img, mask_img):
        """MMPS sampling with structural correction after each posterior_denoise."""
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_obs = x_obs_img.to(torch.float64)
        mask = mask_img.to(torch.float64)
        x_next = latents.to(torch.float64) * t_steps[0]

        has_targets = self._mask_ts is not None
        threshold = self.struct_sigma_threshold

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step with POSTERIOR denoiser
            denoised = self.posterior_denoise(x_hat, t_hat, x_obs, mask)
            if has_targets and float(t_hat) < threshold:
                denoised = self.structural_correct(denoised, t_hat)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Heun 2nd-order correction
            if i < self.num_steps - 1:
                denoised_2 = self.posterior_denoise(x_next, t_next, x_obs, mask)
                if has_targets and float(t_next) < threshold:
                    denoised_2 = self.structural_correct(denoised_2, t_next)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sampling_mmps(self, x_obs_img, mask_img):
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, x_obs_img, mask_img)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        """Unconditional sampling (no structural correction, for evaluation)."""
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


# =============================================================================
# E-step with structural targets
# =============================================================================

def precompute_structural_targets(recon_data, period, window):
    """
    Decompose full reconstructions into trend + seasonal targets.

    Args:
        recon_data: numpy (N, T, C) — fully imputed, no NaN
        period: seasonal period
        window: moving-average window

    Returns:
        trend_all, season_all: numpy (N, T, C)
    """
    x = torch.tensor(recon_data, dtype=torch.float32)
    trend, seasonal, _ = decompose_torch(x, period, window)
    return trend.numpy(), seasonal.numpy()


def e_step_stl_guided(args, uncond_model, corrupted_data, obs_masks,
                      trend_targets, season_targets, em_iter, device,
                      logger=None):
    """
    E-step: MMPS imputation with structural guidance.

    Same as standard MMPS E-step but additionally:
      - Passes per-batch structural targets to the sampler
      - The sampler applies structural correction at each reverse step
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — STL-Guided MMPS ===")
    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    N = len(corrupted_data)
    batch_size = args.batch_size

    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)
    struct_lambda = getattr(args, 'struct_lambda', 0.1)
    struct_sigma_threshold = getattr(args, 'struct_sigma_threshold', 10.0)

    period = getattr(args, 'stl_period', None)
    if period is None:
        period = auto_detect_period(corrupted_data.shape[1])
    window = min(period, max(3, corrupted_data.shape[1] // 3))
    if window % 2 == 0:
        window += 1

    with uncond_model.ema_scope():
        process = STLGuidedMMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters,
            struct_lambda=struct_lambda,
            struct_sigma_threshold=struct_sigma_threshold,
            period=period, window=window,
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step STL-Guided MMPS"):
            end_idx = min(start_idx + batch_size, N)

            corrupted_batch = corrupted_data[start_idx:end_idx]
            mask_batch = obs_masks[start_idx:end_idx]

            corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
            mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)
            obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)

            x_obs_img = uncond_model.ts_to_img(obs_ts)
            mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
            mask_img = uncond_model.ts_to_img(mask_ts_expanded)
            mask_img = mask_img[:, :1, :, :]

            # Set structural targets for this batch
            trend_batch = torch.tensor(
                trend_targets[start_idx:end_idx], dtype=torch.float32, device=device
            )
            season_batch = torch.tensor(
                season_targets[start_idx:end_idx], dtype=torch.float32, device=device
            )
            process.set_batch_targets(mask_ts, trend_batch, season_batch, uncond_model)

            x_img_imputed = process.sampling_mmps(x_obs_img, mask_img)
            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)

            all_reconstructions.append(x_ts_recon.cpu().numpy())
            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    print(f"E-step complete. Imputed {len(reconstructions)} sequences.")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    cache_dir = os.path.join(
        args.recon_cache_dir, args.dataset, f'missing_{int(args.missing_rate * 100)}'
    )
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_stl_guided_{name}"

    logging.info(args)

    struct_lambda = getattr(args, 'struct_lambda', 0.1)
    struct_sigma_threshold = getattr(args, 'struct_sigma_threshold', 10.0)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print_model_params(logger, uncond_model)

        # === Extract corrupted data and masks ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences with "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # === Structural decomposition parameters ===
        seq_len = corrupted_data.shape[1]
        period = getattr(args, 'stl_period', None)
        if period is None:
            period = auto_detect_period(seq_len)
        window = min(period, max(3, seq_len // 3))
        if window % 2 == 0:
            window += 1

        # === Phase 1: Iterative STL Warm-Start ===
        print(f"\n{'='*60}")
        print("Phase 1: Iterative STL Warm-Start Initialization")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_iterative_stl(
            corrupted_data, obs_masks, seed=args.seed
        )

        # Pre-compute initial structural targets
        print("Pre-computing structural targets from initial reconstructions...")
        trend_targets, season_targets = precompute_structural_targets(
            initial_reconstructions, period, window
        )

        # === Phase 1.5: Train unconditional model on STL-initialized data ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial unconditional model training")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger,
        )

        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting STL-Guided DiffEM-MMPS with {args.em_iters} EM iterations")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")
        print(f"  struct_lambda: {struct_lambda}")
        print(f"  struct_sigma_threshold: {struct_sigma_threshold}")
        print(f"  STL period: {period}, window: {window}")
        print(f"{'='*60}")

        best_metrics = None
        last_recon = initial_reconstructions

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # --- Curriculum: reveal extra positions early on ---
            progress = min(1.0, (em_iter + 1) / max(args.em_iters * 0.6, 1))
            reveal_prob = (1.0 - progress) * 0.3
            extra_reveal = (np.random.rand(*obs_masks.shape) < reveal_prob) & ~obs_masks
            curriculum_masks = obs_masks | extra_reveal
            curriculum_corrupted = corrupted_data.copy()
            curriculum_corrupted[extra_reveal] = last_recon[extra_reveal]
            n_revealed = extra_reveal.sum()
            if n_revealed > 0:
                print(f"  Curriculum: revealed {n_revealed} extra positions "
                      f"(progress={progress:.2f})")

            # --- E-step with structural guidance ---
            reconstructions = e_step_stl_guided(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                trend_targets, season_targets,
                em_iter, args.device, logger,
            )
            last_recon = reconstructions

            # Refresh structural targets from new reconstructions
            print("  Refreshing structural targets...")
            trend_targets, season_targets = precompute_structural_targets(
                reconstructions, period, window
            )

            # --- EMA reset + M-step ---
            uncond_model.reset_ema()

            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, em_iter, args.device, logger,
            )

            # --- Evaluate periodically ---
            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, uncond_model, test_loader,
                    em_iter, args.device, logger,
                )
                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        # === Phase 3: Final unconditional training ===
        final_metrics = train_unconditional_regular(
            args, reconstructions, test_loader, args.device, logger,
        )
        if final_metrics:
            print("Phase 3 (unconditional) final metrics:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")

        print("\n" + "=" * 60)
        print("STL-Guided DiffEM-MMPS Complete!")
        print("=" * 60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("STL-Guided DiffEM-MMPS training is complete")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
