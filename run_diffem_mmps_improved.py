"""
Improved DiffEM with Unconditional Model + MMPS E-step

Safe improvements over run_diffem_mmps.py (preserving the core training dynamics):
  1. PPCA initialization (covariance-aware, replaces naive Gaussian)
  2. Observation consistency: observed values replaced after E-step sampling
  3. More CG iterations (3) for better posterior denoiser approximation
  4. Better diagnostics and logging throughout
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
# Logging helpers
# =============================================================================

def _log(logger, name, value, step):
    if logger is not None:
        logger.log(name, value, step)


def _log_stats(tag, arr, step, logger):
    arr = np.asarray(arr).flatten()
    _log(logger, f'{tag}/mean', float(np.nanmean(arr)), step)
    _log(logger, f'{tag}/std', float(np.nanstd(arr)), step)
    _log(logger, f'{tag}/min', float(np.nanmin(arr)), step)
    _log(logger, f'{tag}/max', float(np.nanmax(arr)), step)


def _upload_and_cleanup(logger, plot_path, em_iter):
    if logger is None:
        return
    uploaded = False
    try:
        logger.log_file('test/memorization/histogram', plot_path, em_iter)
        uploaded = True
    except Exception as exc:
        print(f"Failed to upload memorization plot: {exc}")
    if uploaded:
        try:
            os.remove(plot_path)
        except OSError:
            pass


# =============================================================================
# Conjugate Gradient Solver (3 iterations default for better approximation)
# =============================================================================

def conjugate_gradient(A_fn, b, maxiter=3):
    """
    Conjugate gradient solver for A·x = b where A is a linear operator.

    Using 3 iterations (vs 1 in original) for a tighter solve of the posterior
    linear system, since the Jacobian-based covariance is well-conditioned
    enough to benefit from extra iterations without overfitting to noise.
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
# MMPSDiffusionProcess
# =============================================================================

class MMPSDiffusionProcess:
    """
    MMPS-style diffusion sampler for inpainting with an unconditional model.

    Uses the same EDM/Karras sigma schedule as DiffusionProcess. At each reverse
    step, the unconditional denoiser is replaced by a posterior denoiser that
    incorporates observation constraints with covariance correction.

    Improvement over original: 3 CG iterations (vs 1) for better posterior solve.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01, cg_iters=3):
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

        self.sigma_y = sigma_y
        self.cg_iters = cg_iters

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        """
        Posterior denoiser: replaces unconditional D_θ(x_t) with posterior estimate.

        1. Forward pass with gradient: denoised = D_θ(x_t)
        2. Define VJP function using torch.autograd.grad
        3. Compute residual: r = x_obs - mask * denoised
        4. Solve linear system via CG: (σ_y²I + σ² · mask · Jᵀ · mask) · v = r
        5. Compute posterior correction: score = vjp_fn(mask · v)
        6. Return: denoised + σ² · score
        """
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

        return result

    def mmps_sample(self, latents, x_obs_img, mask_img):
        """
        MMPS sampling: same Heun integrator as original, both predictor and
        corrector use the posterior denoiser. S_churn=0 (deterministic).
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
        """Single posterior sample (not averaged — avoids over-smoothing)."""
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, x_obs_img, mask_img)

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
# Data extraction
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


# =============================================================================
# PPCA Initialization (replaces naive Gaussian — covariance-aware)
# =============================================================================

def ppca_fit(X, rank=16):
    N, D = X.shape
    mu = np.nanmean(X, axis=0)

    X_filled = X.copy()
    nan_mask = np.isnan(X_filled)
    for d in range(D):
        X_filled[nan_mask[:, d], d] = mu[d]

    X_c = X_filled - mu

    if N < D:
        C = X_c @ X_c.T / N
    else:
        C = X_c.T @ X_c / N

    rank = min(rank, min(N, D) - 1, D)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = eigvals[-rank:][::-1]
    eigvecs = eigvecs[:, -rank:][:, ::-1]

    if N < D:
        eigvecs = X_c.T @ eigvecs
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)

    if rank < D:
        D_diag = max((np.trace(C) - np.sum(eigvals)) / (D - rank), 1e-6)
    else:
        D_diag = 1e-6

    U = eigvecs * np.sqrt(np.maximum(eigvals - D_diag, 0.0))[None, :]
    return mu, U, D_diag


def gaussian_posterior_impute(X_obs, obs_mask, mu, U, D_diag, seed=None):
    rng = np.random.default_rng(seed)
    N, D = X_obs.shape
    rank = U.shape[1]
    X_out = X_obs.copy()

    for i in range(N):
        o_idx = np.where(obs_mask[i])[0]
        m_idx = np.where(~obs_mask[i])[0]
        if len(m_idx) == 0:
            continue
        if len(o_idx) == 0:
            X_out[i, m_idx] = mu[m_idx] + rng.normal(size=len(m_idx)) * np.sqrt(D_diag)
            continue

        x_o = X_obs[i, o_idx]
        mu_o = mu[o_idx]
        mu_m = mu[m_idx]
        U_o = U[o_idx, :]
        U_m = U[m_idx, :]

        M = np.eye(rank) + (U_o.T @ U_o) / D_diag
        M_inv = np.linalg.inv(M)
        diff = x_o - mu_o
        alpha = diff / D_diag - U_o @ (M_inv @ (U_o.T @ diff)) / (D_diag ** 2)

        cond_mean = mu_m + U_m @ (U_o.T @ alpha)
        cond_std = np.sqrt(D_diag)
        X_out[i, m_idx] = cond_mean + rng.normal(size=len(m_idx)) * cond_std

    return X_out


def initialize_with_ppca(corrupted_data, obs_masks, rank=32, em_iters=8, seed=0):
    N, seq_len, features = corrupted_data.shape
    D = seq_len * features

    X = corrupted_data.reshape(N, D).copy()
    obs_flat = np.repeat(obs_masks, features, axis=1)

    X_work = np.where(np.isnan(X), 0.0, X)
    X_work[~obs_flat] = np.nan

    print(f"PPCA initialization: rank={rank}, em_iters={em_iters}, data shape=({N}, {D})")

    for it in range(em_iters):
        mu, U, D_diag = ppca_fit(X_work, rank=min(rank, D - 1))
        X_work = gaussian_posterior_impute(
            X_obs=np.where(obs_flat, corrupted_data.reshape(N, D), np.nan),
            obs_mask=obs_flat,
            mu=mu, U=U, D_diag=D_diag,
            seed=seed + it
        )
        var_explained = np.sum(np.maximum(np.linalg.norm(U, axis=0) ** 2, 0))
        total_var = var_explained + D_diag * D
        print(f"  PPCA iter {it+1}/{em_iters}: D_diag={D_diag:.4e}, "
              f"var_explained_ratio={var_explained/total_var:.3f}, "
              f"recon range=[{X_work.min():.3f}, {X_work.max():.3f}]")

    recon = X_work.reshape(N, seq_len, features)

    for i in range(N):
        for t in range(seq_len):
            if obs_masks[i, t]:
                orig = corrupted_data[i, t]
                if not np.any(np.isnan(orig)):
                    recon[i, t] = orig

    print(f"PPCA initialization complete. {N} sequences, "
          f"range=[{recon.min():.3f}, {recon.max():.3f}]")
    return recon


# =============================================================================
# Core EM functions
# =============================================================================

def e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    """
    E-step: Impute missing values using MMPS with the unconditional model.

    Improvements over original:
    - 3 CG iterations for better posterior solve
    - Observation consistency: observed values explicitly restored after sampling
    - Diagnostics logged for debugging
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — MMPS Imputation ===")
    print("Imputing missing values via MMPS with unconditional model...")

    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = args.batch_size

    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 3)

    print(f"  sigma_y={sigma_y}, cg_iters={cg_iters}, "
          f"diffusion_steps={args.diffusion_steps}")

    with uncond_model.ema_scope():
        process = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step MMPS"):
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
            x_np = x_ts_recon.cpu().numpy()

            # Observation consistency: restore observed values exactly
            for b in range(len(corrupted_batch)):
                obs = mask_batch[b].astype(bool)
                x_np[b, obs, :] = corrupted_batch[b][obs, :]

            all_reconstructions.append(x_np)
            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)

    # --- Diagnostics ---
    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    print(f"  Reconstructions: shape={reconstructions.shape}, "
          f"range=[{reconstructions.min():.4f}, {reconstructions.max():.4f}], "
          f"mean={reconstructions.mean():.4f}, std={reconstructions.std():.4f}")

    obs_recon_vals = []
    for i in range(min(N, 500)):
        for t in range(corrupted_data.shape[1]):
            if obs_masks[i, t]:
                orig = corrupted_data[i, t]
                if not np.any(np.isnan(orig)):
                    obs_recon_vals.append(np.abs(reconstructions[i, t] - orig).mean())
    if obs_recon_vals:
        obs_mae = np.mean(obs_recon_vals)
        print(f"  Observed-value MAE (should be ~0 with obs consistency): {obs_mae:.6f}")
        _log(logger, 'em/e_step_obs_mae', obs_mae, em_iter)

    _log_stats('em/e_step_recon', reconstructions, em_iter, logger)

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def m_step(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """
    M-step: Train unconditional model on fully imputed data.

    Keeps the PERSISTENT optimizer (no reset) so Adam momentum accumulates
    across EM iterations, and EMA accumulates continuously.
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
    """Evaluate via unconditional sampling + memorization check."""
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
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = uncond_model.img_to_ts(x_img_sampled)

                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    print(f"EM iter {em_iter} metrics (unconditional):")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

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
            logger.log(f'test/memorization/{k}', v, em_iter)

    _upload_and_cleanup(logger, mem_plot_path, em_iter)

    return scores


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    """Main DiffEM-Uncond training loop with MMPS E-step (safe improvements)."""

    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_improved_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

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

        # Log config
        _log(logger, 'config/mmps_sigma_y', getattr(args, 'mmps_sigma_y', 0.01), 0)
        _log(logger, 'config/mmps_cg_iters', getattr(args, 'mmps_cg_iters', 3), 0)
        _log(logger, 'config/m_step_epochs', args.m_step_epochs, 0)
        _log(logger, 'config/em_iters', args.em_iters, 0)
        _log(logger, 'config/missing_rate', args.missing_rate, 0)
        _log(logger, 'config/diffusion_steps', args.diffusion_steps, 0)

        # === Extract corrupted data and masks ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        pct = obs_masks.sum() / obs_masks.size * 100
        print(f"Extracted {len(corrupted_data)} sequences with {pct:.1f}% observed")

        # === Phase 1: PPCA Initialization ===
        print(f"\n{'='*60}")
        print("Phase 1: PPCA-based Initialization")
        print(f"{'='*60}")

        ppca_rank = min(32, corrupted_data.shape[1] * corrupted_data.shape[2] - 1)
        initial_reconstructions = initialize_with_ppca(
            corrupted_data, obs_masks, rank=ppca_rank, em_iters=8, seed=args.seed
        )
        _log_stats('em/init_recon', initial_reconstructions, -1, logger)

        # === Phase 1.5: Train unconditional model on PPCA-filled data ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial unconditional model training on PPCA-filled data")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )
        print("Initial unconditional model training complete.")

        # === Phase 2: EM Loop ===
        sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
        cg_iters = getattr(args, 'mmps_cg_iters', 3)

        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS (safe improvements) with {args.em_iters} EM iterations")
        print(f"  MMPS sigma_y: {sigma_y} (constant)")
        print(f"  MMPS cg_iters: {cg_iters}")
        print(f"  Diffusion steps: {args.diffusion_steps}")
        print(f"  Persistent optimizer + continuous EMA")
        print(f"  Observation consistency: ON")
        print(f"  PPCA initialization: ON")
        print(f"{'='*60}")

        best_metrics = None
        best_em_iter = -1
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

                disc = metrics.get('disc_mean', float('inf'))
                if best_metrics is None or disc < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics
                    best_em_iter = em_iter
                    print(f"  *** New best disc_mean={disc:.4f} at EM iter {em_iter} ***")

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
        print("DiffEM-MMPS-Improved Training Complete!")
        print("="*60)
        if best_metrics:
            print(f"Best metrics at EM iter {best_em_iter}:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS-Improved training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    # Default CG iters to 3 if not explicitly set via CLI
    if args.mmps_cg_iters == 1:
        args.mmps_cg_iters = 3

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
