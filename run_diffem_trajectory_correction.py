"""
DiffEM with Trajectory-Corrected Posterior Sampling (TCPS)

A novel posterior sampling method for diffusion-based EM that exploits the
known forward process at observed positions.

Key insight: At each reverse diffusion step t, we know EXACTLY what the
correctly-noised observation should look like: target_t = y_obs + σ_t · z
(where z is fixed throughout). After the unconditional denoiser takes a
reverse step, we can compute the exact trajectory error at observed positions
and propagate it to missing positions via the denoiser's Jacobian.

This is fundamentally different from MMPS:
  - MMPS compares denoised output against clean y_obs, uses CG to solve a
    soft optimization balancing Σ_y and V[x|x_t]
  - TCPS compares the trajectory state against the noise-matched target,
    uses a single VJP to propagate the error. No CG, no σ_y parameter.

Properties:
  - Built-in annealing: at high σ, target is noisy → error is small → gentle
    corrections. At low σ, target ≈ y_obs → tight corrections.
  - Exact observation consistency: observed positions are hard-replaced with
    the correct noised value at every step (and clean y_obs at t=0).
  - Cheaper than MMPS: single VJP per step vs CG with multiple VJPs.
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
# Trajectory-Corrected Posterior Sampling (TCPS)
# =============================================================================

class TCPSDiffusionProcess:
    """
    Trajectory-Corrected Posterior Sampling (TCPS) for inpainting with an
    unconditional diffusion model.

    At each reverse step t → t-1:
      1. Unconditional denoiser + Heun solver produce x_{t-1}
      2. Compute exact target at observed positions: target = y_obs + σ_{t-1} · z
      3. Compute trajectory error: error = x_{t-1}[obs] - target[obs]
      4. Propagate error to missing positions via Jacobian: correction = Jᵀ(mask · error)
      5. Correct missing positions: x_{t-1}[missing] -= α · correction[missing]
      6. Hard replace observed positions: x_{t-1}[obs] = target[obs]

    A fixed noise vector z is sampled once and reused throughout, ensuring
    the target trajectory is deterministic and consistent across steps.
    """

    def __init__(self, args, diffusion_fn, shape, correction_strength=1.0):
        self.args = args
        self.device = args.device
        self.shape = shape
        self.net = diffusion_fn.to(device=self.device)

        # EDM parameters (same as DiffusionProcess)
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

    def _trajectory_correct(self, x, sigma_next, x_obs, mask, z_fixed):
        """
        Apply trajectory correction after a reverse step.

        Uses the denoiser's Jacobian to propagate the trajectory error from
        observed positions to missing positions.

        Args:
            x: current state after reverse step (batch, C, H, W), float64
            sigma_next: noise level at this state (scalar, float64)
            x_obs: clean observed data (batch, C, H, W), float64
            mask: binary mask (batch, 1, H, W), float64, 1=observed
            z_fixed: fixed noise vector (batch, C, H, W), float64

        Returns:
            x_corrected: trajectory-corrected state (batch, C, H, W), float64
            diagnostics: dict with logging info (or None at final step)
        """
        # Compute exact target at observed positions for this noise level
        if sigma_next > 0:
            target = x_obs + sigma_next * z_fixed
        else:
            target = x_obs

        # At the final step (sigma=0), skip Jacobian correction —
        # just hard replace observed and return
        if sigma_next <= 0:
            x_corrected = x.detach().clone()
            # Log final reconstruction error at observed positions
            final_obs_error = (mask * (x - target)).abs()
            n_obs = mask.sum().clamp(min=1)
            diag = {
                'sigma': 0.0,
                'obs_error_before': final_obs_error.sum().item() / n_obs.item(),
                'obs_error_after': 0.0,
                'correction_norm': 0.0,
                'error_norm': final_obs_error.sum().item() / n_obs.item(),
            }
            x_corrected = mask * target + (1 - mask) * x_corrected
            return x_corrected, diag

        # Compute trajectory error at observed positions
        error = mask * (x - target)

        # --- Diagnostics: error before correction ---
        n_obs = mask.sum().clamp(min=1)
        n_miss = (1 - mask).sum().clamp(min=1)
        error_norm_obs = error.abs().sum().item() / n_obs.item()

        # Propagate error to missing positions via Jacobian
        # Forward pass with gradient tracking
        x_input = x.detach().requires_grad_(True)
        denoised = self.net(x_input, sigma_next, None).to(torch.float64)

        # Single VJP: Jᵀ(mask · error)
        correction, = torch.autograd.grad(
            denoised, x_input, grad_outputs=error,
            retain_graph=False  # No need to retain — single VJP call
        )
        correction = correction.detach()

        # --- Diagnostics: correction magnitude ---
        correction_at_missing = ((1 - mask) * correction).abs()
        correction_norm_miss = correction_at_missing.sum().item() / n_miss.item()

        # Apply correction to missing positions only
        x_corrected = x.detach().clone()
        x_corrected = x_corrected - self.correction_strength * (1 - mask) * correction

        # --- Diagnostics: what would error be at observed after correction (before hard replace) ---
        obs_error_after_correction = (mask * (x_corrected - target)).abs().sum().item() / n_obs.item()

        # Hard replace observed positions with exact target
        x_corrected = mask * target + (1 - mask) * x_corrected

        diag = {
            'sigma': float(sigma_next),
            'obs_error_before': error_norm_obs,
            'obs_error_after': obs_error_after_correction,
            'correction_norm': correction_norm_miss,
            'error_norm': error_norm_obs,
        }

        return x_corrected, diag

    def tcps_sample(self, latents, x_obs_img, mask_img):
        """
        TCPS sampling: impute missing regions using trajectory correction.

        Uses the standard Heun ODE solver for denoising, with trajectory
        correction applied after each complete step (Euler + Heun correction).

        Args:
            latents: initial noise (batch, C, H, W)
            x_obs_img: observed data in image space (batch, C, H, W), clean values
            mask_img: binary mask in image space (batch, 1, H, W), 1=observed, 0=missing

        Returns:
            x_next: imputed result (batch, C, H, W)
        """
        # Adjust noise levels based on what's supported by the network
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization (same as DiffusionProcess)
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Cast observations to float64 for numerical precision
        x_obs = x_obs_img.to(torch.float64)
        mask = mask_img.to(torch.float64)

        # Sample z ONCE and fix it for the entire reverse process
        z_fixed = latents.to(torch.float64)

        # Initialize: noised observations at observed, pure noise at missing
        # Use t_steps[0] (the actual first sigma) for consistency with the schedule
        sigma_init = t_steps[0]
        x_next = mask * (x_obs + sigma_init * z_fixed) + (1 - mask) * (sigma_init * z_fixed)

        step_diagnostics = []

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # Increase noise temporarily (stochastic churn, typically 0)
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step with unconditional denoiser (no gradients needed here)
            with torch.no_grad():
                denoised = self.net(x_hat, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction (Heun) with unconditional denoiser
            if i < self.num_steps - 1:
                with torch.no_grad():
                    denoised_2 = self.net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # Trajectory correction: compute error at observed positions,
            # propagate to missing via Jacobian, hard replace observed
            x_next, diag = self._trajectory_correct(
                x_next, t_next, x_obs, mask, z_fixed
            )
            step_diagnostics.append(diag)

        # Log diagnostics for first sample in batch (only on first batch call)
        if not hasattr(self, '_logged_sample') or not self._logged_sample:
            self._logged_sample = True
            print("\n  TCPS Sampling Diagnostics (first sample):")
            print(f"  {'Step':>4s}  {'sigma':>10s}  {'obs_err_before':>14s}  {'obs_err_after':>14s}  {'correction':>12s}")
            print(f"  {'----':>4s}  {'-----':>10s}  {'--------------':>14s}  {'-------------':>14s}  {'----------':>12s}")
            for step_i, d in enumerate(step_diagnostics):
                print(f"  {step_i:4d}  {d['sigma']:10.4f}  {d['obs_error_before']:14.6f}  {d['obs_error_after']:14.6f}  {d['correction_norm']:12.6f}")
            print()

        # Store diagnostics for wandb logging
        self._last_diagnostics = step_diagnostics

        return x_next

    def sampling_tcps(self, x_obs_img, mask_img):
        """
        Convenience wrapper for TCPS sampling.

        Args:
            x_obs_img: observed data in image space (batch, C, H, W)
            mask_img: binary mask in image space (batch, 1, H, W), 1=observed, 0=missing

        Returns:
            imputed result (batch, C, H, W)
        """
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.tcps_sample(latents, x_obs_img, mask_img)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        """
        Standard unconditional sampling (for evaluation).
        Same as DiffusionProcess.sampling.
        """
        latents = torch.randn([sampling_number, *self.shape], device=self.device)

        # Adjust noise levels
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization
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
        x_irregular = data[0]  # (batch, seq_len, features+1)
        x_ts = x_irregular[:, :, :-1]  # Remove time index column

        # Create mask: True where observed
        mask = ~torch.isnan(x_ts).any(dim=-1)  # (batch, seq_len)

        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())

    corrupted_data = np.vstack(all_corrupted)
    obs_masks = np.vstack(all_masks)

    return corrupted_data, obs_masks


def initialize_with_gaussian(corrupted_data, obs_masks, seed=None):
    """
    Initialize reconstructions using Gaussian prior.

    For observed values: use the actual observed data
    For missing values: sample from Gaussian fitted on observed data

    Args:
        corrupted_data: numpy array (N, seq_len, features) with NaN for missing
        obs_masks: numpy array (N, seq_len) with True for observed
        seed: random seed

    Returns:
        initial_recon: numpy array (N, seq_len, features) with all values filled
    """
    if seed is not None:
        np.random.seed(seed)

    N, seq_len, features = corrupted_data.shape

    # Compute mean and std from observed data
    mu = np.nanmean(corrupted_data, axis=(0, 1))  # Per-feature mean
    std = np.nanstd(corrupted_data, axis=(0, 1))  # Per-feature std

    # Handle edge case where std is 0
    std = np.where(std == 0, 1.0, std)

    # Create initial reconstructions
    initial_recon = corrupted_data.copy()

    # Fill missing values with Gaussian samples
    for i in range(N):
        for t in range(seq_len):
            if not obs_masks[i, t]:
                initial_recon[i, t, :] = np.random.normal(mu, std)

    # Clip to [0, 1] since data is normalized
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
    E-step: Impute missing values using TCPS with the unconditional model.

    Uses TCPSDiffusionProcess: at each reverse step, the unconditional denoiser
    is followed by trajectory correction that propagates observation errors to
    missing positions via the Jacobian.

    Args:
        args: configuration arguments
        uncond_model: unconditional diffusion model
        corrupted_data: numpy array (N, seq_len, features) with NaN for missing
        obs_masks: numpy array (N, seq_len) with True for observed
        em_iter: current EM iteration number
        device: target device
        logger: optional logger

    Returns:
        reconstructions: numpy array of imputed sequences (N, seq_len, features)
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — TCPS Imputation ===")
    print("Imputing missing values via Trajectory-Corrected Posterior Sampling...")

    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = args.batch_size

    correction_strength = getattr(args, 'tcps_correction_strength', 1.0)

    with uncond_model.ema_scope():
        process = TCPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            correction_strength=correction_strength
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step TCPS"):
            end_idx = min(start_idx + batch_size, N)

            # Get batch
            corrupted_batch = corrupted_data[start_idx:end_idx]
            mask_batch = obs_masks[start_idx:end_idx]

            # Convert to tensors
            corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
            mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)

            # Zero-fill NaN values for observed data image
            obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)

            # Convert observed data to image space
            x_obs_img = uncond_model.ts_to_img(obs_ts)

            # Create mask image: expand (N, seq_len) -> (N, seq_len, features),
            # then ts_to_img(), take first channel [:, :1, :, :]
            mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
            mask_img = uncond_model.ts_to_img(mask_ts_expanded)
            mask_img = mask_img[:, :1, :, :]  # Take first channel for mask

            # TCPS sampling
            x_img_imputed = process.sampling_tcps(x_obs_img, mask_img)

            # Convert back to time series
            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)

            all_reconstructions.append(x_ts_recon.cpu().numpy())

            # Free memory
            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    print(f"E-step complete. Imputed {len(reconstructions)} sequences.")

    # Verify no NaN in output
    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    # Cache reconstructions to disk
    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

        # Log TCPS diagnostics from first batch
        if hasattr(process, '_last_diagnostics') and process._last_diagnostics:
            diags = process._last_diagnostics
            for step_i, d in enumerate(diags):
                logger.log(f'tcps/step_{step_i}/sigma', d['sigma'], em_iter)
                logger.log(f'tcps/step_{step_i}/obs_error_before', d['obs_error_before'], em_iter)
                logger.log(f'tcps/step_{step_i}/obs_error_after', d['obs_error_after'], em_iter)
                logger.log(f'tcps/step_{step_i}/correction_norm', d['correction_norm'], em_iter)
            # Log summary stats across all steps
            avg_error = np.mean([d['obs_error_before'] for d in diags])
            avg_correction = np.mean([d['correction_norm'] for d in diags])
            final_error = diags[-1]['obs_error_before']
            logger.log('tcps/avg_obs_error', avg_error, em_iter)
            logger.log('tcps/avg_correction_norm', avg_correction, em_iter)
            logger.log('tcps/final_obs_error', final_error, em_iter)

    return reconstructions


def m_step(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """
    M-step: Train unconditional model on fully imputed data.

    Since the data has no NaN (all values were imputed in E-step), the internal
    mask in loss_fn_irregular is all-ones automatically.

    Args:
        args: configuration arguments
        uncond_model: unconditional diffusion model
        optimizer: model optimizer
        reconstructions: numpy array of fully imputed sequences from E-step
        em_iter: current EM iteration number
        device: target device
        logger: optional logger

    Returns:
        uncond_model: updated model
    """
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training unconditional model for {args.m_step_epochs} epochs...")

    # Create dataset from reconstructions (fully imputed, no NaN)
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

            # Convert to image
            x_img = uncond_model.ts_to_img(x_clean)

            # Standard unconditional diffusion loss
            # Since data has no NaN, the internal mask is all-ones
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
    """Main DiffEM training loop with TCPS E-step."""

    # Model name and directory
    name = create_model_name_and_dir(args)
    name = f"diffem_tcps_{name}"

    logging.info(args)

    # Set up logger
    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        # Set up device and data
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        # === Create single unconditional model ===
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
        print(f"Starting DiffEM-TCPS with {args.em_iters} EM iterations")
        print(f"  TCPS correction strength: {args.tcps_correction_strength}")
        print(f"{'='*60}")

        best_metrics = None
        metrics = None
        last_recon = initial_reconstructions

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # Curriculum: reveal extra observations early, gradually reach full difficulty
            progress = min(1.0, (em_iter + 1) / max(args.em_iters * 0.6, 1))
            reveal_prob = (1.0 - progress) * 0.3
            extra_reveal = (np.random.rand(*obs_masks.shape) < reveal_prob) & ~obs_masks
            curriculum_masks = obs_masks | extra_reveal
            curriculum_corrupted = corrupted_data.copy()
            curriculum_corrupted[extra_reveal] = last_recon[extra_reveal]
            n_revealed = extra_reveal.sum()
            if n_revealed > 0:
                print(f"  Curriculum: revealed {n_revealed} extra positions (progress={progress:.2f})")

            # E-step: TCPS imputation with curriculum masks
            reconstructions = e_step(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                em_iter, args.device, logger
            )
            last_recon = reconstructions

            # Reset EMA so it tracks only this iteration's training dynamics
            uncond_model.reset_ema()

            # M-step: Train unconditional model on fully imputed data
            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, em_iter, args.device, logger
            )

            # Evaluate periodically
            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, uncond_model, test_loader,
                    em_iter, args.device, logger
                )

                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics

            # Log EM iteration
            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        print("\n" + "="*60)
        print("DiffEM-TCPS Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-TCPS training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    # Force DiffEM mode
    args.use_diffem = True

    # Set TCPS defaults
    if not hasattr(args, 'tcps_correction_strength') or args.tcps_correction_strength is None:
        args.tcps_correction_strength = 1.0

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
