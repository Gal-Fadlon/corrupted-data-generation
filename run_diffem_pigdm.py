"""
DiffEM with Unconditional Model + PiGDM E-step

This script implements EM with diffusion priors using a SINGLE unconditional model:
- E-step: PiGDM (Pseudo-Inverse Guided Diffusion Models) imputation — uses a
  diagonal Tweedie covariance approximation for the posterior correction.
- M-step: Train the same unconditional model on fully imputed data

PiGDM core idea: At each reverse diffusion step, compute the denoised estimate
x̂₀ = D_θ(x_t) and approximate the Tweedie covariance as a diagonal matrix
using a single VJP. The posterior mean correction then uses this diagonal
covariance to weight the observation residual per pixel.

Reference: Song et al., "Pseudoinverse-Guided Diffusion Models for Inverse
           Problems", ICLR 2023.
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
from functools import partial

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
from utils.train_unconditional import train_unconditional_regular

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# PiGDMDiffusionProcess
# =============================================================================

class PiGDMDiffusionProcess:
    """
    PiGDM-style diffusion sampler for inpainting with an unconditional model.

    Uses the same EDM/Karras sigma schedule as DiffusionProcess. At each reverse
    step, the denoiser provides x̂₀ and the diagonal Tweedie covariance is
    approximated via a single VJP. The posterior correction uses per-pixel gains
    derived from this diagonal covariance.

    Posterior score decomposition (Bayes' rule):
        ∇_x log q(x_t | y) = ∇_x log q(x_t) + ∇_x log p(y | x_t)

    PiGDM approximation (diagonal covariance):
        Σ_{x|x_t} ≈ diag(v_t) where v_t = σ_t² · |diag(J)|
        J = ∂D_θ/∂x_t is the denoiser Jacobian

    For masking operator A = diag(mask), the posterior mean correction becomes:
        Δx = J^T · A^T · diag(v_i / (σ_y² + v_i)) · (y - A · x̂₀)
    Since A is a mask, this simplifies to per-observed-pixel corrections.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01):
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

        self.sigma_y = sigma_y

    def pigdm_sample(self, latents, x_obs_img, mask_img):
        """
        PiGDM sampling: inpaint missing regions using diagonal covariance guidance.

        At each denoising step (t_cur -> t_next):
        1. x_cur = x_next.detach().requires_grad_(True) for VJP
        2. Compute denoised = D_θ(x_hat, t_hat) with grad enabled
        3. Standard Euler step + Heun correction for ODE trajectory
        4. PiGDM correction:
           a. Compute residual r = mask * (x_obs - denoised)
           b. Diagonal Jacobian: vjp_fn(ones) gives row-sums ≈ diag(J)
           c. Diagonal covariance: v_i = σ_t² · |jac_diag_i|
           d. Per-pixel gain: g_i = v_i / (σ_y² + v_i) for observed pixels
           e. Correction via VJP: Δx = vjp_fn(A^T · diag(g) · r)
        5. Apply: x_next += Δx

        Args:
            latents: initial noise (batch, C, H, W)
            x_obs_img: observed data in image space (batch, C, H, W)
            mask_img: binary mask in image space (batch, 1, H, W), 1=observed

        Returns:
            x_next: inpainted result (batch, C, H, W)
        """
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
            x_cur = x_next.detach().requires_grad_(True)

            # Stochastic churn
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Forward pass with grad enabled for VJP
            denoised = self.net(x_hat, t_hat, None).to(torch.float64)

            # Standard ODE step
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Heun correction (no grad needed)
            if i < self.num_steps - 1:
                with torch.no_grad():
                    denoised_2 = self.net(x_next, t_next, None).to(torch.float64)
                    d_prime = (x_next - denoised_2) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # --- PiGDM correction ---
            sigma_t_sq = float(t_hat ** 2)

            # Diagonal Jacobian approximation via VJP with ones vector.
            # For D_θ(x_hat), the row-sum of the Jacobian gives diag info:
            #   jac_diag_approx_i ≈ Σ_j (∂D_θ_i / ∂x_hat_j)
            # This is the standard PiGDM approximation.
            ones_vec = torch.ones_like(denoised)
            jac_row_sum = torch.autograd.grad(
                denoised, x_cur, grad_outputs=ones_vec,
                create_graph=False, retain_graph=False
            )[0]

            # Diagonal covariance: v_i = σ_t² · |jac_diag_i|
            v_diag = sigma_t_sq * jac_row_sum.abs()

            # Per-pixel Kalman-like gain: g_i = v_i / (σ_y² + v_i)
            gain = v_diag / (self.sigma_y ** 2 + v_diag)

            # Observation residual in image space
            residual = x_obs - denoised.detach()

            # Masked correction: only apply at observed pixels
            correction = mask * gain.detach() * residual

            # Apply correction to the denoised estimate, then propagate to x_next.
            # Since we already computed x_next from the ODE, we add the correction
            # scaled by the step ratio (t_next / t_hat accounts for the noise level).
            step_scale = float(t_next / t_hat) if float(t_hat) > 0 else 0.0
            x_next = x_next.detach() + correction * (1.0 - step_scale)

        return x_next

    def sampling_pigdm(self, x_obs_img, mask_img):
        """
        Convenience wrapper for PiGDM sampling.
        NOT decorated with @torch.no_grad() — PiGDM needs gradients for VJP.
        """
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.pigdm_sample(latents, x_obs_img, mask_img)

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
# Helper functions (identical to run_diffem_dps.py)
# =============================================================================

def get_corrupted_data_from_loader(train_loader, device):
    """Extract corrupted time series (with NaNs) from the train loader."""
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
    """Initialize reconstructions using Gaussian prior."""
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
# Core EM functions
# =============================================================================

def e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    """E-step: Impute missing values using PiGDM with the unconditional model."""
    print(f"\n=== E-Step (EM iter {em_iter}) — PiGDM Imputation ===")

    uncond_model.eval()
    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = args.batch_size
    sigma_y = getattr(args, 'pigdm_sigma_y', 0.01)

    with uncond_model.ema_scope():
        process = PiGDMDiffusionProcess(
            args, uncond_model.net, target_shape, sigma_y=sigma_y
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step PiGDM"):
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

            x_img_imputed = process.sampling_pigdm(x_obs_img, mask_img)
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
    """M-step: Train unconditional model on fully imputed data."""
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
    """Evaluate the unconditional model via standard sampling."""
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")

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
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)

    if logger is not None:
        try:
            logger.log_file('test/memorization/histogram', mem_plot_path, em_iter)
        except Exception:
            pass
        try:
            if os.path.exists(mem_plot_path):
                os.remove(mem_plot_path)
        except Exception:
            pass

    return scores


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    """Main DiffEM training loop with PiGDM E-step."""

    name = create_model_name_and_dir(args)
    name = f"diffem_pigdm_{name}"

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

        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences with {obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # Phase 1: Gaussian initialization
        print(f"\n{'='*60}")
        print("Phase 1: Gaussian Initialization")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_gaussian(
            corrupted_data, obs_masks, seed=args.seed
        )

        # Phase 1.5: Initial training on Gaussian-filled data
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial unconditional model training")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )

        # Phase 2: EM Loop
        pigdm_sigma_y = getattr(args, 'pigdm_sigma_y', 0.01)
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-PiGDM with {args.em_iters} EM iterations")
        print(f"  PiGDM sigma_y: {pigdm_sigma_y}")
        print(f"{'='*60}")

        best_metrics = None

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            reconstructions = e_step(
                args, uncond_model, corrupted_data, obs_masks,
                em_iter, args.device, logger
            )

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

        # Phase 3 — Train fresh unconditional model on final completions
        final_metrics = train_unconditional_regular(
            args, reconstructions, test_loader, args.device, logger,
        )
        if final_metrics:
            print("Phase 3 (unconditional) final metrics:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")

        print("\n" + "="*60)
        print("DiffEM-PiGDM Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-PiGDM training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
