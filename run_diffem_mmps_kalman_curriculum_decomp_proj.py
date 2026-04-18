"""
DiffEM with MMPS E-step — Kalman Init + Curriculum + Decomposition Projection

Combines three orthogonal improvements:
  1. Kalman smoother initialization — optimal linear state-space imputation
     via UnobservedComponents with Kalman smoothing that handles NaN natively.
  2. Decomposition-projected MMPS E-step — after each posterior denoising step,
     the estimate is projected through a decomposition-recomposition that
     enforces structural constraints. Projection strength is sigma-dependent.
  3. Curriculum missing rate — early EM iterations reveal extra observations
     to make the problem easier, gradually increasing to full difficulty.

Based on run_diffem_mmps_iter_stl_decomp_projection.py with init swapped to
initialize_with_kalman.
"""

import torch
import torch.autograd
import torch.multiprocessing
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import os
import sys
import numpy as np
import logging
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

from metrics import evaluate_model_irregular
from metrics.memorization import compute_memorization_metric
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import (
    gen_dataloader, save_reconstructions, load_reconstructions
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.utils_stl import initialize_with_kalman, auto_detect_period
from utils.train_unconditional import train_unconditional_regular

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Fast decomposition-projection in TS space (fully vectorized)
# =============================================================================

def decompose_and_smooth_ts(ts_batch, period=None, trend_smooth_window=None,
                            residual_keep_fraction=0.8):
    """
    Vectorized decomposition + smoothing for a batch of time series.

    Decomposes into trend + seasonal + residual, then attenuates the residual
    to enforce structural smoothness.

    Args:
        ts_batch: (B, T, C) numpy array — no NaN
        period: decomposition period (auto-detected if None)
        trend_smooth_window: additional smoothing window for trend.
            None = auto-scale with T (max(5, T // 50)).
        residual_keep_fraction: fraction of residual energy to keep (0-1).
            1.0 = identity (no smoothing), 0.0 = fully remove residual.

    Returns:
        projected: (B, T, C) numpy array with smoothed trend + periodic seasonal
                   + attenuated residual
    """
    B, T, C = ts_batch.shape
    if period is None:
        period = auto_detect_period(T)
    if trend_smooth_window is None:
        trend_smooth_window = max(5, T // 50)

    window = min(period, max(3, T // 3))
    if window % 2 == 0:
        window += 1

    trends = uniform_filter1d(ts_batch, size=window, axis=1, mode='nearest')

    if trend_smooth_window > 1:
        sw = trend_smooth_window if trend_smooth_window % 2 == 1 else trend_smooth_window + 1
        trends = uniform_filter1d(trends, size=sw, axis=1, mode='nearest')

    detrended = ts_batch - trends
    seasonals = np.zeros_like(ts_batch)
    for k in range(period):
        idx = np.arange(k, T, period)
        phase_mean = detrended[:, idx, :].mean(axis=1, keepdims=True)
        seasonals[:, idx, :] = phase_mean

    residuals = ts_batch - trends - seasonals
    return trends + seasonals + residual_keep_fraction * residuals


# =============================================================================
# Conjugate Gradient Solver
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
# Decomposition-Projected MMPSDiffusionProcess
# =============================================================================

class DecompProjectedMMPSDiffusionProcess:
    """
    MMPS sampler with decomposition consistency projection.

    After each MMPS posterior correction, the denoised estimate is converted
    to TS, decomposed, trend-smoothed, recomposed, and blended back.
    The blending weight alpha = sigma / sigma_max ensures the projection
    is strongest at high noise (where the model is uncertain) and vanishes
    at low noise (preserving fine details).
    """

    def __init__(self, args, diffusion_fn, shape, model,
                 sigma_y=0.01, cg_iters=1,
                 projection_strength=0.5, sigma_threshold_frac=0.5):
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
        self.projection_strength = projection_strength
        self.sigma_threshold_frac = sigma_threshold_frac

    def _project_denoised(self, denoised_img, sigma):
        """Apply decomposition projection to denoised image."""
        sigma_ratio = float(sigma) / self.sigma_max
        if sigma_ratio < 0.05:
            return denoised_img

        alpha = min(1.0, sigma_ratio / self.sigma_threshold_frac) * self.projection_strength

        ts = self.model.img_to_ts(denoised_img.float())
        ts_np = ts.detach().cpu().numpy()

        projected_np = decompose_and_smooth_ts(ts_np)
        projected_ts = torch.tensor(projected_np, dtype=torch.float32, device=self.device)
        projected_img = self.model.ts_to_img(projected_ts).to(denoised_img.dtype)

        return (1.0 - alpha) * denoised_img + alpha * projected_img

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
        result = denoised.detach() + sigma_sq * score.detach()

        result = self._project_denoised(result, sigma)

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


# =============================================================================
# Helper functions
# =============================================================================

def get_corrupted_data_from_loader(train_loader, device):
    all_corrupted, all_masks = [], []
    for batch_idx, data in enumerate(train_loader):
        x_irregular = data[0]
        x_ts = x_irregular[:, :, :-1]
        mask = ~torch.isnan(x_ts).any(dim=-1)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())
    return np.vstack(all_corrupted), np.vstack(all_masks)


# =============================================================================
# Core EM functions
# =============================================================================

def e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    print(f"\n=== E-Step (EM iter {em_iter}) — Decomp-Projected MMPS ===")
    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    N = len(corrupted_data)
    batch_size = getattr(args, 'e_step_batch_size', args.batch_size)
    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    with uncond_model.ema_scope():
        process = DecompProjectedMMPSDiffusionProcess(
            args, uncond_model.net, target_shape, uncond_model,
            sigma_y=sigma_y, cg_iters=cg_iters,
            projection_strength=0.5, sigma_threshold_frac=0.5
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step Decomp-Projected"):
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

            x_img_imputed = process.sampling_mmps(x_obs_img, mask_img)
            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)

            all_reconstructions.append(x_ts_recon.cpu().numpy())
            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    print(f"E-step complete. Imputed {len(reconstructions)} sequences.")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def m_step(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training unconditional model for {args.m_step_epochs} epochs...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    uncond_model.train()
    for epoch in range(args.m_step_epochs):
        epoch_loss, num_batches = 0, 0
        for (x_clean,) in recon_loader:
            x_clean = x_clean.to(device)
            x_img = uncond_model.ts_to_img(x_clean)
            loss, _ = uncond_model.loss_fn_irregular(x_img)
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
            print(f"  M-step epoch {epoch+1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")
        if logger is not None:
            logger.log('em/m_step_loss', avg_loss, em_iter * args.m_step_epochs + epoch)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, em_iter, device, logger=None):
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    uncond_model.eval()
    gen_sig, real_sig = [], []

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution, args.img_resolution)
            )
            for data in tqdm(test_loader, desc="Evaluating"):
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = uncond_model.img_to_ts(x_img_sampled)
                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)
    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    print(f"EM iter {em_iter} metrics:")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

    mem_plot_path = f"memorization_hist_em_iter_{em_iter}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig,
        device=device, plot_path=mem_plot_path
    )
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)
    if logger is not None:
        try:
            logger.log_file('test/memorization/histogram', mem_plot_path, em_iter)
            if os.path.exists(mem_plot_path):
                os.remove(mem_plot_path)
        except Exception:
            pass

    return scores


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_kalman_curriculum_decomp_proj_{name}"
    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate, weight_decay=args.weight_decay
        )
        print_model_params(logger, uncond_model)

        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences with "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        print(f"\n{'='*60}")
        print("Phase 1: Kalman Smoother Warm-Start Initialization")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_kalman(
            corrupted_data, obs_masks, seed=args.seed,
            per_fit_timeout=getattr(args, 'kalman_fit_timeout', 5),
            max_seconds=getattr(args, 'kalman_global_timeout', 1800),
        )

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )

        print(f"\n{'='*60}")
        print(f"Starting Kalman + Curriculum + Decomp-Projected DiffEM-MMPS with {args.em_iters} EM iterations")
        print(f"{'='*60}")

        best_metrics = None
        last_recon = initial_reconstructions

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # Curriculum
            progress = min(1.0, (em_iter + 1) / max(args.em_iters * 0.6, 1))
            reveal_prob = (1.0 - progress) * getattr(args, 'curriculum_reveal_max', 0.3)
            extra_reveal = (np.random.rand(*obs_masks.shape) < reveal_prob) & ~obs_masks
            curriculum_masks = obs_masks | extra_reveal
            curriculum_corrupted = corrupted_data.copy()
            curriculum_corrupted[extra_reveal] = last_recon[extra_reveal]
            n_revealed = extra_reveal.sum()
            if n_revealed > 0:
                print(f"  Curriculum: revealed {n_revealed} extra positions (progress={progress:.2f})")

            reconstructions = e_step(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                em_iter, args.device, logger
            )
            last_recon = reconstructions

            uncond_model.reset_ema()

            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, em_iter, args.device, logger
            )

            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, uncond_model, test_loader,
                    em_iter, args.device, logger
                )
                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)


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
        print("\n" + "="*60)
        print("Kalman + Curriculum + Decomp-Projected DiffEM-MMPS Complete!")
        print("="*60)
        if best_metrics:
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
