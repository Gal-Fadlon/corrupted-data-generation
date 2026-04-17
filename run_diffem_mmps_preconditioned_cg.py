"""
DiffEM with MMPS E-step — EMA Reset + Iterative-STL Init + Curriculum Missing Rate
+ Preconditioned CG

Apply a frequency-domain diagonal preconditioner to the CG system.  A good
preconditioner M⁻¹ ≈ (Σ_y + A·V·Aᵀ)⁻¹ speeds up convergence.  For time
series, the covariance is approximately diagonal in Fourier space, so a
frequency-domain diagonal preconditioner is natural and effective.
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
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.utils_stl import initialize_with_iterative_stl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Conjugate Gradient Solver (original — used by unconditional sampling path)
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
# Preconditioned Conjugate Gradient Solver
# =============================================================================

def preconditioned_conjugate_gradient(A_fn, b, precond_fn, maxiter=3):
    """
    Preconditioned conjugate gradient solver for A·x = b.

    Uses a preconditioner M⁻¹ (given as precond_fn) to improve conditioning
    of the linear system.  For time series, a frequency-domain diagonal
    preconditioner is effective because the covariance is approximately
    diagonal in Fourier space.

    Args:
        A_fn: callable, linear operator v -> A·v
        b: right-hand side tensor
        precond_fn: callable, preconditioner v -> M⁻¹·v
        maxiter: number of PCG iterations (default=3)

    Returns:
        x: approximate solution tensor (same shape as b)
    """
    x = torch.zeros_like(b)
    r = b.clone()
    z = precond_fn(r)
    p = z.clone()

    for _ in range(maxiter):
        Ap = A_fn(p)
        rTz = (r * z).sum()
        pTAp = (p * Ap).sum()
        alpha = rTz / (pTAp + 1e-30)
        x = x + alpha * p
        r_new = r - alpha * Ap
        z_new = precond_fn(r_new)
        rTz_new = (r_new * z_new).sum()
        beta = rTz_new / (rTz + 1e-30)
        p = z_new + beta * p
        r, z = r_new, z_new

    return x


# =============================================================================
# MMPSDiffusionProcess — Preconditioned CG variant
# =============================================================================

class MMPSDiffusionProcess:
    """
    MMPS-style diffusion sampler for inpainting with an unconditional model.

    Uses the same EDM/Karras sigma schedule as DiffusionProcess. At each reverse
    step, the unconditional denoiser is replaced by a posterior denoiser that
    incorporates observation constraints with covariance correction.

    This variant uses a preconditioned CG solver with a frequency-domain
    diagonal preconditioner.  The preconditioner approximates the inverse of
    (sigma_y^2 I + sigma^2 * power_spectrum) in the Fourier basis, which is
    natural for time series data where the covariance is approximately
    stationary.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01, cg_iters=1,
                 precond_power_decay=2.0):
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
        self.precond_power_decay = precond_power_decay

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        """
        Posterior denoiser with preconditioned CG.

        The preconditioner is a diagonal approximation in frequency domain:
            M⁻¹(v) = IFFT( FFT(v) / (sigma_y^2 + sigma^2 * power_spectrum) )
        where power_spectrum decays as 1/f^alpha for typical time series
        (alpha = precond_power_decay, default 2.0).

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
        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return grad

        # Compute residual: r = x_obs - mask * denoised
        r = x_obs - mask * denoised

        # CG linear operator: M(v) = σ_y² · v + σ² · mask · Jᵀ(mask · v)
        def cg_operator(v):
            return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)

        # Frequency-domain diagonal preconditioner
        W = r.shape[-1]  # last dimension (time axis in image space)
        power_decay = self.precond_power_decay

        def preconditioner(v):
            v_freq = torch.fft.rfft(v, dim=-1)  # FFT along W (time axis)
            # Diagonal approx: 1 / (sigma_y^2 + sigma^2 * power_spectrum)
            # where power_spectrum decays as 1/f^alpha for typical time series
            freqs = torch.arange(v_freq.shape[-1], device=v.device, dtype=v.dtype)
            power_est = 1.0 / (1.0 + (freqs / max(1, v_freq.shape[-1] // 4)) ** power_decay)
            diag_inv = 1.0 / (sigma_y_sq + sigma_sq * power_est + 1e-10)
            v_freq = v_freq * diag_inv
            return torch.fft.irfft(v_freq, n=v.shape[-1], dim=-1)

        # Solve with preconditioned CG
        v = preconditioned_conjugate_gradient(
            cg_operator, r, preconditioner, maxiter=self.cg_iters
        )

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
    precond_power_decay = getattr(args, 'precond_power_decay', 2.0)

    # No torch.no_grad() outer context — MMPS requires gradient computation for VJP
    with uncond_model.ema_scope():
        process = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters,
            precond_power_decay=precond_power_decay
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

    if logger is not None:
        e_step_step = (em_iter + 1) * args.m_step_epochs
        logger.log('em/e_step_samples', len(reconstructions), e_step_step)

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
            global_step = (em_iter + 1) * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, em_iter, device, logger=None):
    """
    Evaluate the unconditional model via standard unconditional sampling.

    Same evaluation as run_irregular.py for fair comparison:
    generates completely new sequences and compares to real data.

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

    eval_step = (em_iter + 2) * args.m_step_epochs - 1

    print(f"EM iter {em_iter} metrics (unconditional):")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, eval_step)

    return scores


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    """Main DiffEM-Uncond training loop with MMPS E-step."""

    # Model name and directory
    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_preconditioned_cg_{name}"

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

        # === Baseline evaluation (before EM) ===
        baseline_metrics = evaluate_uncond(
            args, uncond_model, test_loader,
            em_iter=-1, device=args.device, logger=logger
        )

        # === Phase 2: EM Loop with Curriculum ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS + Curriculum with {args.em_iters} EM iterations")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")
        print(f"  Precond power decay: {args.precond_power_decay}")
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

            # E-step: MMPS imputation with curriculum masks
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
                iter_step = (em_iter + 2) * args.m_step_epochs
                logger.log('em/iteration', em_iter, iter_step)

        print("\n" + "="*60)
        print("DiffEM-MMPS + Preconditioned CG Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS + Preconditioned CG training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    # Force DiffEM mode
    args.use_diffem = True

    # Preconditioned CG specific args
    if not hasattr(args, 'precond_power_decay'):
        args.precond_power_decay = 2.0

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
