"""
DiffEM with Trajectory-Corrected Posterior Sampling — Variant B (Denoiser-Level)

Same core insight as TCPS: we know the exact forward process at observed
positions, so we can compute trajectory errors and propagate them.

Difference from TCPS (variant A / run_diffem_trajectory_correction.py):
  - TCPS-A corrects x_{t-1} AFTER the reverse step (post-step correction)
  - TCPS-B corrects the denoised estimate x̂ BEFORE the reverse step
    (denoiser-level correction, like MMPS but without CG)

At each reverse step:
  1. Denoise: x̂ = D_θ(x_t) — this estimates E[x_0 | x_t]
  2. At observed positions, we know x_0 = y_obs, so error = x̂[obs] - y_obs
  3. Propagate error to missing via Jacobian: correction = Jᵀ(mask · error)
  4. Corrected denoiser: x̂_corrected = x̂ - α · correction
  5. Hard replace observed: x̂_corrected[obs] = y_obs
  6. Feed x̂_corrected into the Heun solver as the denoised estimate

This is closer to the MMPS framework (modifying the denoiser output) but:
  - No CG solver — single VJP
  - No σ_y parameter — hard observation constraint
  - The correction target is the clean y_obs (not noise-matched), because
    the denoiser output estimates E[x_0|x_t], not E[x_t|x_t]

Additionally, like TCPS-A, observed positions in the trajectory state are
hard-replaced with noise-matched targets: x_t[obs] = y_obs + σ_t · z.
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
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import (
    gen_dataloader, save_reconstructions, load_reconstructions
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.utils_stl import initialize_with_iterative_stl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Trajectory-Corrected Posterior Sampling — Variant B (Denoiser-Level)
# =============================================================================

class TCPSv2DiffusionProcess:
    """
    TCPS Variant B: correct the denoised estimate before the reverse step.

    At each reverse step t → t-1:
      1. Forward pass with grad: x̂ = D_θ(x_t)
      2. Error at observed: error = mask * (x̂ - y_obs)
      3. Propagate via Jacobian: correction = Jᵀ(mask * error)
      4. Correct denoised: x̂_corrected = x̂ - α * correction
      5. Hard replace observed: x̂_corrected[obs] = y_obs
      6. Heun step uses x̂_corrected as the denoised estimate
      7. After Heun step, hard replace trajectory: x_{t-1}[obs] = y_obs + σ_{t-1} * z
    """

    def __init__(self, args, diffusion_fn, shape, correction_strength=1.0):
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

        # TCPS parameter
        self.correction_strength = correction_strength

    def _corrected_denoise(self, x_t, sigma, x_obs, mask):
        """
        Denoiser-level trajectory correction via input-space correction.

        1. Denoise: x̂ = D_θ(x_t)
        2. Error at observed: error = x̂[obs] - y_obs
        3. Jacobian maps error from output to input space: Jᵀ(mask · error)
        4. Correct x_t in INPUT space: x_t_corrected = x_t - α · Jᵀ(error)
        5. Re-denoise with corrected input: x̂_corrected = D_θ(x_t_corrected)
        6. Hard replace observed: x̂_corrected[obs] = y_obs

        This is principled: Jᵀ maps output-space errors to input-space
        corrections, so we correct x_t (the input) and re-denoise to get
        a clean output that respects the observations.

        Args:
            x_t: noisy input (batch, C, H, W), float64
            sigma: noise level scalar (float64)
            x_obs: clean observed data (batch, C, H, W), float64
            mask: binary mask (batch, 1, H, W), float64, 1=observed

        Returns:
            denoised_corrected: corrected denoised estimate (batch, C, H, W), float64
            diagnostics: dict with logging info
        """
        # Forward pass with gradient tracking
        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        # Error: denoised output at observed positions vs clean observations
        # The denoiser estimates E[x_0 | x_t], and at observed positions x_0 = y_obs
        error = mask * (denoised - x_obs)

        # --- Diagnostics ---
        n_obs = mask.sum().clamp(min=1)
        n_miss = (1 - mask).sum().clamp(min=1)
        error_norm_obs = error.abs().sum().item() / n_obs.item()

        # Jᵀ maps output-space error to input-space correction
        input_correction, = torch.autograd.grad(
            denoised, x_t_input, grad_outputs=error,
            retain_graph=False
        )
        input_correction = input_correction.detach()

        # --- Diagnostics ---
        correction_at_missing = ((1 - mask) * input_correction).abs()
        correction_norm_miss = correction_at_missing.sum().item() / n_miss.item()

        # Correct x_t in input space, then re-denoise
        x_t_corrected = x_t.detach() - self.correction_strength * input_correction

        with torch.no_grad():
            denoised_corrected = self.net(x_t_corrected, sigma, None).to(torch.float64)

        # --- Diagnostics: error AFTER re-denoising (before hard replace) ---
        error_after = (mask * (denoised_corrected - x_obs)).abs().sum().item() / n_obs.item()

        # Hard replace observed positions with exact clean values
        denoised_corrected = mask * x_obs + (1 - mask) * denoised_corrected

        diag = {
            'sigma': float(sigma),
            'denoiser_error_obs': error_norm_obs,
            'denoiser_error_after': error_after,
            'correction_norm_miss': correction_norm_miss,
        }

        return denoised_corrected, diag

    def tcps_sample(self, latents, x_obs_img, mask_img):
        """
        TCPS-B sampling: Heun solver with denoiser-level correction.

        Both the Euler step and Heun correction use the corrected denoiser.
        After each complete step, observed positions are hard-replaced with
        the noise-matched target.

        Args:
            latents: initial noise (batch, C, H, W)
            x_obs_img: observed data in image space (batch, C, H, W), clean values
            mask_img: binary mask in image space (batch, 1, H, W), 1=observed, 0=missing

        Returns:
            x_next: imputed result (batch, C, H, W)
        """
        # Adjust noise levels
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Cast observations to float64
        x_obs = x_obs_img.to(torch.float64)
        mask = mask_img.to(torch.float64)

        # Sample z ONCE and fix for the entire reverse process
        z_fixed = latents.to(torch.float64)

        # Initialize: noised observations at observed, pure noise at missing
        sigma_init = t_steps[0]
        x_next = mask * (x_obs + sigma_init * z_fixed) + (1 - mask) * (sigma_init * z_fixed)

        step_diagnostics = []

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # Stochastic churn (typically 0)
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step with CORRECTED denoiser
            denoised, diag = self._corrected_denoise(x_hat, t_hat, x_obs, mask)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Heun correction with CORRECTED denoiser
            if i < self.num_steps - 1:
                denoised_2, _ = self._corrected_denoise(x_next, t_next, x_obs, mask)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # Hard replace observed positions with noise-matched target
            if t_next > 0:
                target = x_obs + t_next * z_fixed
            else:
                target = x_obs
            x_next = mask * target + (1 - mask) * x_next

            step_diagnostics.append(diag)

        # Log diagnostics for first batch
        if not hasattr(self, '_logged_sample') or not self._logged_sample:
            self._logged_sample = True
            print("\n  TCPS-B Sampling Diagnostics (first sample):")
            print(f"  {'Step':>4s}  {'sigma':>10s}  {'err_before':>12s}  {'err_after':>12s}  {'reduction':>10s}  {'correction':>12s}")
            print(f"  {'----':>4s}  {'-----':>10s}  {'----------':>12s}  {'---------':>12s}  {'---------':>10s}  {'----------':>12s}")
            for step_i, d in enumerate(step_diagnostics):
                err_b = d['denoiser_error_obs']
                err_a = d['denoiser_error_after']
                reduction = (1 - err_a / (err_b + 1e-10)) * 100
                print(f"  {step_i:4d}  {d['sigma']:10.4f}  {err_b:12.6f}  {err_a:12.6f}  {reduction:9.1f}%  {d['correction_norm_miss']:12.6f}")
            print()

        # Store diagnostics for wandb logging
        self._last_diagnostics = step_diagnostics

        return x_next

    def sampling_tcps(self, x_obs_img, mask_img):
        """Convenience wrapper for TCPS-B sampling."""
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.tcps_sample(latents, x_obs_img, mask_img)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        """Standard unconditional sampling (for evaluation)."""
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
    """
    Extract corrupted time series (with NaNs) from the train loader.

    Args:
        train_loader: dataloader with irregular observations
        device: target device

    Returns:
        corrupted_data: numpy array (N, seq_len, features) with NaN for missing
        obs_masks: numpy array (N, seq_len) with True for observed
    """
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


def initialize_with_gaussian(corrupted_data, obs_masks, seed=None):
    """
    Initialize reconstructions using Gaussian prior.
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
    print(f"  Mean per feature: {mu}")
    print(f"  Std per feature: {std}")

    return initial_recon


# =============================================================================
# Core EM functions
# =============================================================================

def e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    """
    E-step: Impute missing values using TCPS-B with the unconditional model.
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — TCPS-B Imputation ===")
    print("Imputing via Trajectory-Corrected Posterior Sampling (denoiser-level)...")

    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = args.batch_size

    correction_strength = getattr(args, 'tcps_correction_strength', 1.0)

    with uncond_model.ema_scope():
        process = TCPSv2DiffusionProcess(
            args, uncond_model.net, target_shape,
            correction_strength=correction_strength
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step TCPS-B"):
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

            x_img_imputed = process.sampling_tcps(x_obs_img, mask_img)

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

        # Log TCPS-B diagnostics from first batch
        if hasattr(process, '_last_diagnostics') and process._last_diagnostics:
            diags = process._last_diagnostics
            for step_i, d in enumerate(diags):
                logger.log(f'tcps/step_{step_i}/sigma', d['sigma'], em_iter)
                logger.log(f'tcps/step_{step_i}/denoiser_error_obs', d['denoiser_error_obs'], em_iter)
                logger.log(f'tcps/step_{step_i}/denoiser_error_after', d['denoiser_error_after'], em_iter)
                logger.log(f'tcps/step_{step_i}/correction_norm_miss', d['correction_norm_miss'], em_iter)
            # Log summary stats
            avg_error_before = np.mean([d['denoiser_error_obs'] for d in diags])
            avg_error_after = np.mean([d['denoiser_error_after'] for d in diags])
            avg_reduction = (1 - avg_error_after / (avg_error_before + 1e-10)) * 100
            avg_correction = np.mean([d['correction_norm_miss'] for d in diags])
            logger.log('tcps/avg_error_before', avg_error_before, em_iter)
            logger.log('tcps/avg_error_after', avg_error_after, em_iter)
            logger.log('tcps/avg_error_reduction_pct', avg_reduction, em_iter)
            logger.log('tcps/avg_correction_norm', avg_correction, em_iter)

    return reconstructions


def m_step(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """
    M-step: Train unconditional model on fully imputed data.
    """
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training unconditional model for {args.m_step_epochs} epochs...")

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
            print(f"  M-step epoch {epoch+1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)

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
            eval_step = (em_iter + 2) * args.m_step_epochs - 1
            logger.log(f'test/{key}', value, eval_step)

    return scores


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    """Main DiffEM training loop with TCPS-B E-step."""

    name = create_model_name_and_dir(args)
    name = f"diffem_tcps_v2_{name}"

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

        # === Extract corrupted data and masks ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences with {obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # === Phase 1: Initialize with Iterative STL prior ===
        print(f"\n{'='*60}")
        print("Phase 1: Iterative STL Warm-Start Initialization")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_iterative_stl(
            corrupted_data, obs_masks, seed=args.seed
        )

        # === Phase 1.5: Train unconditional model on Iterative-STL-initialized data ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial unconditional model training on Iterative-STL-initialized data")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )
        print("Initial unconditional model training complete.")

        # === Phase 2: EM Loop with Curriculum ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-TCPS-B with {args.em_iters} EM iterations")
        print(f"  TCPS correction strength: {args.tcps_correction_strength}")
        print(f"{'='*60}")

        best_metrics = None
        metrics = None
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
            n_revealed = extra_reveal.sum()
            if n_revealed > 0:
                print(f"  Curriculum: revealed {n_revealed} extra positions (progress={progress:.2f})")

            # E-step
            reconstructions = e_step(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                em_iter, args.device, logger
            )
            last_recon = reconstructions

            # Reset EMA
            uncond_model.reset_ema()

            # M-step
            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, em_iter, args.device, logger
            )

            # Evaluate
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
        print("DiffEM-TCPS-B Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-TCPS-B training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    if not hasattr(args, 'tcps_correction_strength') or args.tcps_correction_strength is None:
        args.tcps_correction_strength = 1.0

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
