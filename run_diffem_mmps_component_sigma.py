"""
DiffEM with MMPS E-step — Component-Wise Observation Noise (Anisotropic sigma_y)

Fork of run_diffem_mmps_stl_curriculum.py with one key change:
  Instead of a single sigma_y for all observations, uses different
  observation noise levels for trend, seasonal, and residual components.

  - sigma_y_trend:    small (0.005) — trend observations are reliable
  - sigma_y_seasonal: medium (0.01) — seasonal observations are moderately reliable
  - sigma_y_residual: large (0.05)  — residual is noisy by nature

  This prevents MMPS from overfitting to residual noise while still
  tightly constraining the trend, which is the most important component
  for distribution learning.

Implementation:
  The residual r = x_obs - mask * denoised is decomposed into three
  frequency bands via a fast moving-average decomposition.  Each band
  is scaled by 1/sigma_y_component before the CG solve, effectively
  giving different confidence levels to different components.
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
from utils.utils_stl import initialize_with_stl, auto_detect_period

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Image-space decomposition (fast, differentiable-compatible)
# =============================================================================

def decompose_image_lowpass(x, kernel_size=5):
    """
    Fast low-pass filter in image space using average pooling + upsample.

    Args:
        x: (B, C, H, W) tensor
        kernel_size: averaging kernel size

    Returns:
        lowpass: (B, C, H, W) smoothed tensor
    """
    pad = kernel_size // 2
    return F.avg_pool2d(
        F.pad(x.float(), (pad, pad, pad, pad), mode='reflect'),
        kernel_size, stride=1
    ).to(x.dtype)


def decompose_residual_components(r, trend_kernel=5, seasonal_kernel=3):
    """
    Decompose a residual image into trend / seasonal / high-freq components
    using cascaded low-pass filtering.

    Args:
        r: (B, C, H, W) residual tensor
        trend_kernel: kernel size for trend extraction
        seasonal_kernel: kernel size for seasonal extraction

    Returns:
        r_trend: low-frequency part of residual
        r_seasonal: mid-frequency part
        r_residual: high-frequency part
    """
    r_trend = decompose_image_lowpass(r, trend_kernel)
    r_detrended = r - r_trend
    r_seasonal = decompose_image_lowpass(r_detrended, seasonal_kernel)
    r_residual = r_detrended - r_seasonal
    return r_trend, r_seasonal, r_residual


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
# Component-Sigma MMPSDiffusionProcess
# =============================================================================

class ComponentSigmaMMPSDiffusionProcess:
    """
    MMPS sampler with component-wise observation noise.

    The residual r = x_obs - mask * denoised is decomposed into
    trend/seasonal/residual bands, each weighted by its own sigma_y.
    The CG solve uses an effective sigma_y that varies per pixel
    based on which frequency band dominates there.
    """

    def __init__(self, args, diffusion_fn, shape,
                 sigma_y_trend=0.005, sigma_y_seasonal=0.01,
                 sigma_y_residual=0.05, cg_iters=1):
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

        self.sigma_y_trend = sigma_y_trend
        self.sigma_y_seasonal = sigma_y_seasonal
        self.sigma_y_residual = sigma_y_residual
        self.cg_iters = cg_iters

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        """Component-wise sigma_y posterior denoiser."""
        sigma_sq = sigma ** 2

        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return grad

        r_full = x_obs - mask * denoised

        # Decompose residual into frequency bands
        r_trend, r_seasonal, r_resid = decompose_residual_components(r_full)

        # Weight each component by 1/sigma_y^2 (higher confidence = lower sigma_y = larger weight)
        # Then normalize to construct an effective weighted residual
        w_t = 1.0 / (self.sigma_y_trend ** 2)
        w_s = 1.0 / (self.sigma_y_seasonal ** 2)
        w_r = 1.0 / (self.sigma_y_residual ** 2)

        r_weighted = (w_t * r_trend + w_s * r_seasonal + w_r * r_resid) / (w_t + w_s + w_r)

        # Effective sigma_y for CG: harmonic mean weighted by component energies
        sigma_y_eff_sq = 1.0 / (w_t + w_s + w_r)

        def cg_operator(v):
            return sigma_y_eff_sq * v + sigma_sq * mask * vjp_fn(mask * v)

        v = conjugate_gradient(cg_operator, r_weighted, maxiter=self.cg_iters)
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


def e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    print(f"\n=== E-Step (EM iter {em_iter}) — Component-Sigma MMPS ===")
    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    N = len(corrupted_data)
    batch_size = args.batch_size

    sigma_y_trend = getattr(args, 'sigma_y_trend', 0.005)
    sigma_y_seasonal = getattr(args, 'sigma_y_seasonal', 0.01)
    sigma_y_residual = getattr(args, 'sigma_y_residual', 0.05)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    with uncond_model.ema_scope():
        process = ComponentSigmaMMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y_trend=sigma_y_trend,
            sigma_y_seasonal=sigma_y_seasonal,
            sigma_y_residual=sigma_y_residual,
            cg_iters=cg_iters
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step Component-Sigma"):
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
        print(f"  WARNING: {nan_count} NaN values!")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def m_step(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_loader = Data.DataLoader(
        Data.TensorDataset(recon_tensor),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
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
                x_img = process.sampling(sampling_number=data[0].shape[0])
                gen_sig.append(uncond_model.img_to_ts(x_img).cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)
    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig, device=device,
        plot_path=f"memorization_hist_em_iter_{em_iter}.png"
    )
    for k, v in mem_stats.items():
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)
    if logger is not None:
        p = f"memorization_hist_em_iter_{em_iter}.png"
        try:
            logger.log_file('test/memorization/histogram', p, em_iter)
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    return scores


# =============================================================================
# Main
# =============================================================================

def main(args):
    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_component_sigma_{name}"
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
        print(f"Extracted {len(corrupted_data)} sequences, "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        initial_reconstructions = initialize_with_stl(
            corrupted_data, obs_masks, seed=args.seed
        )

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )

        print(f"\n{'='*60}")
        print(f"Starting Component-Sigma DiffEM-MMPS ({args.em_iters} EM iters)")
        print(f"  sigma_y_trend:    {getattr(args, 'sigma_y_trend', 0.005)}")
        print(f"  sigma_y_seasonal: {getattr(args, 'sigma_y_seasonal', 0.01)}")
        print(f"  sigma_y_residual: {getattr(args, 'sigma_y_residual', 0.05)}")
        print(f"{'='*60}")

        best_metrics = None
        last_recon = initial_reconstructions

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # Curriculum
            progress = min(1.0, (em_iter + 1) / max(args.em_iters * 0.6, 1))
            reveal_prob = (1.0 - progress) * 0.3
            extra_reveal = (np.random.rand(*obs_masks.shape) < reveal_prob) & ~obs_masks
            curriculum_masks = obs_masks | extra_reveal
            curriculum_corrupted = corrupted_data.copy()
            curriculum_corrupted[extra_reveal] = last_recon[extra_reveal]

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

        print("\n" + "="*60)
        print("Component-Sigma DiffEM-MMPS Complete!")
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
