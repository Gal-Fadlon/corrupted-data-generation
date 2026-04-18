"""
DiffEM with Unconditional Model + MMPS E-step

This script implements EM with diffusion priors using a SINGLE unconditional model:
- E-step: MMPS (Moment-Matching Posterior Sampling) imputation — approximates the
  posterior denoiser using the full Gaussian likelihood with covariance correction
  via the denoiser Jacobian.
- M-step: Train the same unconditional model on fully imputed data

Paper: "Learning Diffusion Priors from Observations by Expectation Maximization"
       (Rozet et al., NeurIPS 2024)

MMPS core idea: Replace the unconditional denoiser D_θ(x_t) with a posterior
denoiser that accounts for the observation likelihood. The posterior denoiser
solves a linear system involving the denoiser Jacobian (via VJP) to compute
how much to correct the unconditional estimate toward the observations.

Key difference from DPS: MMPS includes the covariance term A·V[x₀|x_t]·Aᵀ
(from 2nd-order Tweedie) in the likelihood, preventing over-correction at high
noise levels where the denoiser is uncertain.
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
from itertools import chain

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
from models.decoder import TST_Decoder
from models.TST import TSTransformerEncoder
from utils.train_unconditional import train_unconditional_regular

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


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


def propagate_values_forward(tensor):
    """Replace fully-missing sequences with a neighboring sequence."""
    for batch_idx in range(tensor.size(0)):
        sequence = tensor[batch_idx]
        if torch.isnan(sequence).all():
            if batch_idx + 1 < tensor.size(0):
                tensor[batch_idx] = tensor[batch_idx + 1]
            elif batch_idx > 0:
                tensor[batch_idx] = tensor[batch_idx - 1]
    return tensor


def propagate_values(tensor):
    """Match run_irregular.py preprocessing before the TST warm-start."""
    return propagate_values_forward(tensor)


def _loss_e_t0(x_tilde, x):
    return F.mse_loss(x_tilde, x)


def _loss_e_0(loss_e_t0):
    return torch.sqrt(loss_e_t0) * 10


def build_irregular_tst(args, device):
    """Build the TST encoder/decoder pair used in the irregular SOTA bootstrap."""
    tst_config = {
        'feat_dim': args.input_size,
        'max_len': args.seq_len,
        'd_model': args.hidden_dim,
        'n_heads': args.n_heads,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'pos_encoding': args.pos_encoding,
        'activation': args.activation,
        'norm': args.norm,
        'freeze': args.freeze
    }

    embedder = TSTransformerEncoder(
        feat_dim=tst_config['feat_dim'],
        max_len=tst_config['max_len'],
        d_model=tst_config['d_model'],
        n_heads=tst_config['n_heads'],
        num_layers=tst_config['num_layers'],
        dim_feedforward=tst_config['dim_feedforward'],
        dropout=tst_config['dropout'],
        pos_encoding=tst_config['pos_encoding'],
        activation=tst_config['activation'],
        norm=tst_config['norm'],
        freeze=tst_config['freeze']
    ).to(device)

    decoder = TST_Decoder(
        inp_dim=args.hidden_dim,
        hidden_dim=int(args.hidden_dim + (args.input_size - args.hidden_dim) / 2),
        layers=3,
        args=args
    ).to(device)

    optimizer_er = optim.Adam(chain(embedder.parameters(), decoder.parameters()))
    return embedder, decoder, optimizer_er


def train_irregular_sota_bootstrap(
    args, uncond_model, optimizer, train_loader, device, logger=None
):
    """
    Warm-start the unconditional diffusion model using the latest irregular SOTA.

    This replaces the Gaussian fill + initial M-step with:
    1. observed-only TST pretraining
    2. masked diffusion warm-start driven by TST reconstructions
    """
    pretrain_epochs = getattr(args, 'bootstrap_pretrain_epochs', None)
    if pretrain_epochs is None:
        pretrain_epochs = args.first_epoch

    diffusion_epochs = getattr(args, 'bootstrap_diffusion_epochs', None)
    if diffusion_epochs is None:
        diffusion_epochs = args.m_step_epochs

    print(f"\n{'=' * 60}")
    print("Phase 1: Irregular SOTA Bootstrap")
    print(f"{'=' * 60}")
    print(f"  TST pretrain epochs: {pretrain_epochs}")
    print(f"  Masked diffusion bootstrap epochs: {diffusion_epochs}")

    embedder, decoder, optimizer_er = build_irregular_tst(args, device)
    embedder.train()
    decoder.train()
    uncond_model.train()

    if pretrain_epochs > 0:
        print("\n--- Bootstrap Step 1/2: TST pretraining on observed values ---")
        for step in range(1, pretrain_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            for i, data in enumerate(train_loader, 1):
                x = data[0].to(device)
                x_ts = x[:, :, :-1]
                x_ts = propagate_values(x_ts)
                padding_masks = ~torch.isnan(x_ts).any(dim=-1)
                h = embedder(x_ts, padding_masks)
                x_tilde = decoder(h)

                x_no_nan = x_ts[~torch.isnan(x_ts)]
                x_tilde_no_nan = x_tilde[~torch.isnan(x_ts)]
                loss_e_t0 = _loss_e_t0(x_tilde_no_nan, x_no_nan)
                loss_e = _loss_e_0(loss_e_t0)

                optimizer_er.zero_grad()
                loss_e.backward()
                optimizer_er.step()

                epoch_loss += loss_e_t0.item()
                num_batches += 1
                torch.cuda.empty_cache()

            rmse = np.sqrt(epoch_loss / max(num_batches, 1))
            print(f"  TST pretrain epoch {step}/{pretrain_epochs}, rmse: {rmse:.4f}")
            if logger is not None:
                logger.log('bootstrap/tst_pretrain_rmse', rmse, step - 1)

    if diffusion_epochs > 0:
        print("\n--- Bootstrap Step 2/2: masked diffusion warm-start ---")
        for epoch in range(diffusion_epochs):
            epoch_loss = 0.0
            epoch_recovery_loss = 0.0
            num_batches = 0

            for i, data in enumerate(train_loader, 1):
                x = data[0].to(device)
                x_ts = x[:, :, :-1]
                x_ts = propagate_values(x_ts)

                padding_masks = ~torch.isnan(x_ts).any(dim=-1)
                x_img = uncond_model.ts_to_img(x_ts)
                mask = torch.isnan(x_img).float() * -1 + 1

                h = embedder(x_ts, padding_masks)
                x_recon = decoder(h)
                x_tilde_img = uncond_model.ts_to_img(x_recon)

                loss, to_log = uncond_model.loss_fn_irregular(x_tilde_img, mask)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
                optimizer.step()
                uncond_model.on_train_batch_end()

                h = embedder(x_ts, padding_masks)
                x_tilde = decoder(h)
                x_no_nan = x_ts[~torch.isnan(x_ts)]
                x_tilde_no_nan = x_tilde[~torch.isnan(x_ts)]
                loss_e_t0 = _loss_e_t0(x_tilde_no_nan, x_no_nan)
                loss_e = _loss_e_0(loss_e_t0)
                optimizer_er.zero_grad()
                loss_e.backward()
                optimizer_er.step()

                epoch_loss += loss.item()
                epoch_recovery_loss += loss_e_t0.item()
                num_batches += 1
                torch.cuda.empty_cache()

                if logger is not None:
                    for key, value in to_log.items():
                        logger.log(f'bootstrap/{key}', value, epoch)

            avg_loss = epoch_loss / max(num_batches, 1)
            recovery_rmse = np.sqrt(epoch_recovery_loss / max(num_batches, 1))
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == diffusion_epochs - 1:
                print(
                    f"  Bootstrap diffusion epoch {epoch + 1}/{diffusion_epochs}, "
                    f"loss: {avg_loss:.4f}, recovery_rmse: {recovery_rmse:.4f}"
                )
            if logger is not None:
                logger.log('bootstrap/diffusion_loss', avg_loss, epoch)
                logger.log('bootstrap/recovery_rmse', recovery_rmse, epoch)

    # Keep the trained diffusion model only; EM refinement remains purely MMPS-based.
    del embedder
    del decoder
    torch.cuda.empty_cache()

    return uncond_model


def bootstrap_model(args, uncond_model, optimizer, train_loader, corrupted_data, obs_masks, device, logger=None):
    """Select the bootstrap strategy before the MMPS EM loop begins."""
    bootstrap_method = getattr(args, 'bootstrap_method', 'irregular')

    if bootstrap_method == 'gaussian':
        print(f"\n{'=' * 60}")
        print("Phase 1: Gaussian Initialization")
        print(f"{'=' * 60}")
        initial_reconstructions = initialize_with_gaussian(
            corrupted_data, obs_masks, seed=args.seed
        )

        print(f"\n{'=' * 60}")
        print("Phase 1.5: Initial unconditional model training on Gaussian-filled data")
        print(f"{'=' * 60}")
        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=device, logger=logger
        )
        print("Initial unconditional model training complete.")
        return uncond_model

    if bootstrap_method == 'irregular':
        return train_irregular_sota_bootstrap(
            args, uncond_model, optimizer, train_loader, device, logger=logger
        )

    raise ValueError(f"Unsupported bootstrap_method: {bootstrap_method}")


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
    batch_size = getattr(args, 'e_step_batch_size', args.batch_size)

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
    name = f"diffem_mmps_sota_bootstrap_{name}"

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

        # === Bootstrap: warm-start before EM ===
        uncond_model = bootstrap_model(
            args,
            uncond_model,
            optimizer,
            train_loader,
            corrupted_data,
            obs_masks,
            args.device,
            logger=logger
        )

        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS with {args.em_iters} EM iterations")
        print(f"  Bootstrap method: {args.bootstrap_method}")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")
        print(f"{'='*60}")

        best_metrics = None
        metrics = None
        disc_mean_trajectory = []

        if getattr(args, 'eval_bootstrap', True):
            bootstrap_metrics = evaluate_uncond(
                args, uncond_model, test_loader,
                em_iter=-1, device=args.device, logger=logger
            )
            bootstrap_disc_mean = bootstrap_metrics.get('disc_mean')
            if bootstrap_disc_mean is not None:
                disc_mean_trajectory.append(('bootstrap', bootstrap_disc_mean))
            best_metrics = bootstrap_metrics

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # E-step: MMPS imputation with current model
            reconstructions = e_step(
                args, uncond_model, corrupted_data, obs_masks,
                em_iter, args.device, logger
            )

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

                disc_mean = metrics.get('disc_mean')
                if disc_mean is not None:
                    disc_mean_trajectory.append((f'em_{em_iter}', disc_mean))
                    if logger is not None:
                        logger.log('em/disc_mean_best_so_far', min(v for _, v in disc_mean_trajectory), em_iter)

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
        print("DiffEM-MMPS Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")
        if disc_mean_trajectory:
            print("disc_mean trajectory:")
            for stage_name, value in disc_mean_trajectory:
                print(f"  {stage_name}: {value:.4f}")

        logging.info("DiffEM-MMPS training is complete")


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
