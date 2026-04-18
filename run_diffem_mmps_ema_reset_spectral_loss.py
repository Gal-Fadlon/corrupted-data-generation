"""
DiffEM with Unconditional Model + MMPS E-step — EMA Reset + Spectral Loss

Same as run_diffem_mmps_ema_reset.py with ONE enhancement:
  An FFT-based spectral matching loss is added during M-step training.
  The spectral loss penalises mismatches in the frequency-magnitude spectrum
  between the model's denoised prediction and the target image, encouraging
  the UNet to preserve the time-series frequency structure (an implicit
  form of frequency-domain decomposition).

  total_loss = standard_denoising_loss + lambda_spec * spectral_loss

  E-step, evaluation, model architecture, and EMA reset are all unchanged.
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
from utils.train_unconditional import train_unconditional_regular

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Spectral Loss Helper
# =============================================================================

def compute_spectral_loss(output, target):
    """
    L2 loss between FFT magnitudes of predicted and target images.
    Encourages the model to preserve frequency content.

    Args:
        output: model prediction (B, C, H, W)
        target: clean target (B, C, H, W)
    Returns:
        scalar spectral loss
    """
    fft_out = torch.fft.rfft2(output.float())
    fft_tgt = torch.fft.rfft2(target.float())
    return (fft_out.abs() - fft_tgt.abs()).square().mean()


# =============================================================================
# Conjugate Gradient Solver
# =============================================================================

def conjugate_gradient(A_fn, b, maxiter=1):
    """
    Conjugate gradient solver for linear system A·x = b, where A is given
    as a function (linear operator).

    For maxiter=1, reduces to a single scaled residual step:
        x = (rᵀr / rᵀAr) · r  where r = b (initial residual with x₀=0)

    Args:
        A_fn: callable, linear operator v -> A·v
        b: right-hand side tensor
        maxiter: number of CG iterations (default=1)

    Returns:
        x: approximate solution tensor (same shape as b)
    """
    x = torch.zeros_like(b)
    r = b.clone()  # r = b - A·x₀ = b (since x₀ = 0)
    p = r.clone()

    for _ in range(maxiter):
        Ap = A_fn(p)
        rTr = (r * r).sum()
        pTAp = (p * Ap).sum()

        # Avoid division by zero
        alpha = rTr / (pTAp + 1e-30)

        x = x + alpha * p
        r_new = r - alpha * Ap

        rTr_new = (r_new * r_new).sum()
        beta = rTr_new / (rTr + 1e-30)

        p = r_new + beta * p
        r = r_new

    return x


# =============================================================================
# MMPSDiffusionProcess
# =============================================================================

class MMPSDiffusionProcess:
    """
    MMPS-style diffusion sampler for inpainting with an unconditional model.

    Uses the same EDM/Karras sigma schedule as DiffusionProcess. At each reverse
    step, the unconditional denoiser is replaced by a posterior denoiser that
    incorporates observation constraints with covariance correction.

    MMPS approximation (full Gaussian with covariance):
        p(y | x_t) ≈ N(y | A·D_θ(x_t), σ_y²I + A·V[x₀|x_t]·Aᵀ)
    where V[x₀|x_t] = σ²·Jᵀ_D (Jacobian-transpose of denoiser, scaled by σ²)

    The covariance term A·V·Aᵀ regularizes the guidance, preventing over-correction
    at high noise levels where the denoiser is uncertain.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01, cg_iters=1):
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

        # MMPS parameters
        self.sigma_y = sigma_y
        self.cg_iters = cg_iters

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        """
        Posterior denoiser: replaces unconditional D_θ(x_t) with posterior estimate.

        Following Rozet's PosteriorDenoiser (DiffEM-main/priors/diffusion.py:481-558):

        1. Forward pass with gradient: denoised = D_θ(x_t)
        2. Define VJP function using torch.autograd.grad (robust to in-place ops)
        3. Compute residual: r = x_obs - mask * denoised
        4. Solve linear system via CG: (σ_y²I + σ² · mask · Jᵀ · mask) · v = r
        5. Compute posterior correction: score = vjp_fn(mask · v)
        6. Return: denoised + σ² · score

        Why torch.autograd.grad instead of torch.func.vjp:
            The UNet has in-place .add_() operations (Conv2d, ResBlock, attention
            in models/networks.py) which break torch.func.vjp.

        Args:
            x_t: noisy input (batch, C, H, W), float64
            sigma: noise level scalar (float64)
            x_obs: observed data (batch, C, H, W), float64
            mask: binary mask (batch, 1, H, W), float64, 1=observed

        Returns:
            posterior_denoised: corrected denoised estimate (batch, C, H, W), float64
        """
        sigma_sq = sigma ** 2
        sigma_y_sq = self.sigma_y ** 2

        # Forward pass with gradient tracking
        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        # Define VJP function: cotangent -> Jᵀ · cotangent
        # Uses autograd.grad with retain_graph=True to allow multiple VJP calls
        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return grad

        # Compute residual: r = x_obs - mask * denoised
        r = x_obs - mask * denoised

        # CG linear operator: M(v) = σ_y² · v + σ² · mask · Jᵀ(mask · v)
        # This is the covariance matrix (σ_y²I + A·V·Aᵀ) applied to v,
        # where A = mask (diagonal), V = σ²·Jᵀ
        def cg_operator(v):
            return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)

        # Solve (σ_y²I + σ² · mask · Jᵀ · mask) · v = r
        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)

        # Compute posterior correction: score = Jᵀ(mask · v)
        score = vjp_fn(mask * v)

        # Detach from computation graph
        result = denoised.detach() + sigma_sq * score.detach()

        return result

    def mmps_sample(self, latents, x_obs_img, mask_img):
        """
        MMPS sampling: inpaint missing regions using posterior denoiser.

        Same structure as DiffusionProcess.sample(), but BOTH the Euler step
        and Heun correction call posterior_denoise() instead of self.net().

        This is the key structural difference from DPS: MMPS modifies the
        denoised estimate itself, so both the Euler predictor and Heun corrector
        naturally use the posterior estimate.

        Args:
            latents: initial noise (batch, C, H, W)
            x_obs_img: observed data in image space (batch, C, H, W), clean values
            mask_img: binary mask in image space (batch, 1, H, W), 1=observed, 0=missing

        Returns:
            x_next: inpainted result (batch, C, H, W)
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

        # Main sampling loop
        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # Increase noise temporarily (stochastic churn, typically 0)
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step with POSTERIOR denoiser
            denoised = self.posterior_denoise(x_hat, t_hat, x_obs, mask)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction with POSTERIOR denoiser
            if i < self.num_steps - 1:
                denoised_2 = self.posterior_denoise(x_next, t_next, x_obs, mask)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sampling_mmps(self, x_obs_img, mask_img):
        """
        Convenience wrapper for MMPS sampling.
        NOT decorated with @torch.no_grad() — MMPS needs gradients for VJP.

        Args:
            x_obs_img: observed data in image space (batch, C, H, W)
            mask_img: binary mask in image space (batch, 1, H, W), 1=observed, 0=missing

        Returns:
            inpainted result (batch, C, H, W)
        """
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, x_obs_img, mask_img)

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
# Helper functions (from run_diffem_uncond.py)
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
    E-step: Impute missing values using MMPS with the unconditional model.

    Uses MMPSDiffusionProcess: at each reverse step, the unconditional denoiser
    is replaced by a posterior denoiser with covariance correction via the
    denoiser Jacobian.

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
    print(f"\n=== E-Step (EM iter {em_iter}) — MMPS Imputation ===")
    print("Imputing missing values via MMPS with unconditional model...")

    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = args.batch_size

    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    # No torch.no_grad() outer context — MMPS requires gradient computation for VJP
    with uncond_model.ema_scope():
        process = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step MMPS"):
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

            # MMPS sampling
            x_img_imputed = process.sampling_mmps(x_obs_img, mask_img)

            # Convert back to time series
            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)

            all_reconstructions.append(x_ts_recon.cpu().numpy())

            # Free autograd memory
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

    return reconstructions


def m_step(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """
    M-step with spectral loss: standard denoising loss + FFT magnitude loss.
    """
    lambda_spec = getattr(args, 'lambda_spec', 0.01)
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training for {args.m_step_epochs} epochs (spectral loss lambda={lambda_spec})...")

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
        epoch_spec = 0
        num_batches = 0

        for batch_idx, (x_clean,) in enumerate(recon_loader):
            x_clean = x_clean.to(device)

            x_img = uncond_model.ts_to_img(x_clean)

            mask = torch.ones_like(x_img)
            output, weight = uncond_model.forward_irregular(x_img, mask)

            x_unpad = uncond_model.unpad(x_img, x_img.shape)
            out_unpad = uncond_model.unpad(output, x_img.shape)
            denoising_loss = (weight * (out_unpad - x_unpad).square()).mean()

            spec_loss = compute_spectral_loss(out_unpad, x_unpad)
            loss = denoising_loss + lambda_spec * spec_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss += denoising_loss.item()
            epoch_spec += spec_loss.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches
        avg_spec = epoch_spec / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  M-step epoch {epoch+1}/{args.m_step_epochs}, "
                  f"denoise: {avg_loss:.4f}, spectral: {avg_spec:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)
            logger.log('em/m_step_spectral_loss', avg_spec, global_step)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, em_iter, device, logger=None):
    """
    Evaluate the unconditional model via standard unconditional sampling.

    Same evaluation as run_irregular.py for fair comparison:
    generates completely new sequences and compares to real data.
    Includes memorization metrics.

    Args:
        args: configuration arguments
        uncond_model: unconditional diffusion model
        test_loader: test data loader
        em_iter: current EM iteration
        device: target device
        logger: optional logger

    Returns:
        scores: dict of evaluation metrics
    """
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    print("Evaluating unconditional model (same as run_irregular.py)...")

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
                # Unconditional sampling — generates completely new sequences
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = uncond_model.img_to_ts(x_img_sampled)

                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    # Compute metrics
    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    print(f"EM iter {em_iter} metrics (unconditional):")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

    # --- Memorization Check ---
    mem_plot_path = f"memorization_hist_em_iter_{em_iter}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig,
        generated_data=gen_sig,
        device=device,
        plot_path=mem_plot_path
    )

    # Log memorization stats
    print(f"EM iter {em_iter} memorization metrics:")
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)

    if logger is not None:
        upload_successful = False
        try:
            logger.log_file('test/memorization/histogram', mem_plot_path, em_iter)
            upload_successful = True
        except Exception as e:
            print(f"Failed to upload memorization plot: {e}")

        if upload_successful:
            try:
                if os.path.exists(mem_plot_path):
                    os.remove(mem_plot_path)
            except Exception as e:
                print(f"Failed to delete temporary plot file {mem_plot_path}: {e}")

    return scores


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    """Main DiffEM-Uncond training loop with MMPS E-step."""

    # Model name and directory
    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_ema_reset_spectral_loss_{name}"

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

        # === Phase 1: Initialize with Gaussian prior ===
        print(f"\n{'='*60}")
        print("Phase 1: Gaussian Initialization")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_gaussian(
            corrupted_data, obs_masks, seed=args.seed
        )

        # === Phase 1.5: Train unconditional model on Gaussian-filled data ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial unconditional model training on Gaussian-filled data")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )
        print("Initial unconditional model training complete.")

        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS + Spectral Loss with {args.em_iters} EM iterations")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")
        print(f"{'='*60}")

        best_metrics = None
        metrics = None

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # E-step: MMPS imputation with current model
            reconstructions = e_step(
                args, uncond_model, corrupted_data, obs_masks,
                em_iter, args.device, logger
            )

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
        print("DiffEM-MMPS + Spectral Loss Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS + Spectral Loss training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    # Force DiffEM mode
    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
