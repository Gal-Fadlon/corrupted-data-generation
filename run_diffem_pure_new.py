"""
Improved Pure DiffEM Training Script — Aligned with the Original Paper

Key improvements over run_diffem_pure.py:
  1. PPCA-based Gaussian initialization with covariance-aware imputation
     (matches the paper's fit_moments / PPCA strategy).
  2. Many more M-step training epochs (default 200) with cosine LR schedule.
  3. Fresh random corruption every epoch in the M-step (not fixed).
  4. More diffusion sampling steps in E-step (default 64, overridable).
  5. Extensive diagnostic logging for tracking training health.
  6. Proper model warm-starting between EM laps (fine-tuning, not from scratch).
"""

import torch
import torch.multiprocessing
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import os
import sys
import numpy as np
import logging
import math
from tqdm import tqdm
from contextlib import contextmanager

from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.train_unconditional import train_unconditional_regular
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import (
    gen_dataloader, apply_corruption,
    save_reconstructions, load_reconstructions
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras_Cond
from models.sampler import ConditionalDiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

# ---------------------------------------------------------------------------
# Improved defaults for this script (paper-aligned)
# These override the old argparse defaults from parse_args_irregular.
# ---------------------------------------------------------------------------
SCRIPT_DEFAULTS = dict(
    m_step_epochs=100,       # paper uses 256 per lap; old default was 50
    e_step_sample_steps=64,  # paper uses 256 DDPM steps; 64 EDM Heun ≈ 256 DDPM
    em_iters=10,
    em_eval_interval=1,
    uncond_epochs_per_iter=200,  # match m_step_epochs for fair uncond training
)

# ============================= helpers =====================================

def _log(logger, name, value, step):
    if logger is not None:
        logger.log(name, value, step)


def _log_stats(tag, arr, step, logger):
    """Log basic statistics of a numpy array."""
    arr = np.asarray(arr).flatten()
    _log(logger, f'{tag}/mean', float(np.nanmean(arr)), step)
    _log(logger, f'{tag}/std', float(np.nanstd(arr)), step)
    _log(logger, f'{tag}/min', float(np.nanmin(arr)), step)
    _log(logger, f'{tag}/max', float(np.nanmax(arr)), step)


# ============================= data ========================================

def get_corrupted_data_from_loader(train_loader):
    """Extract corrupted time series (with NaNs) and masks from the train loader."""
    all_corrupted, all_masks = [], []
    for data in train_loader:
        x_irregular = data[0]                      # (batch, seq_len, features+1)
        x_ts = x_irregular[:, :, :-1]              # drop time-index column
        mask = ~torch.isnan(x_ts).any(dim=-1)      # (batch, seq_len)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())
    return np.vstack(all_corrupted), np.vstack(all_masks)


# ============================= initialization ==============================

def ppca_fit(X, rank=16):
    """Probabilistic PCA: fit (mu, U, D) such that cov ≈ U U^T + D*I.

    Args:
        X: (N, D) array — may contain NaN for missing entries.
        rank: number of principal components.

    Returns:
        mu: (D,) mean vector
        U:  (D, rank) principal component loadings
        D_diag: scalar — isotropic residual variance
    """
    N, D = X.shape

    # Robust mean — per feature, ignoring NaN
    mu = np.nanmean(X, axis=0)

    # For PPCA we need a complete matrix; impute NaN with the mean first.
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
    """Impute missing values using Gaussian posterior under the PPCA model.

    For each sample, observed entries x_o give a Gaussian posterior for
    missing entries x_m:
        x_m | x_o ~ N(mu_m|o, Sigma_m|o)

    We draw one sample from this posterior for each missing entry.
    """
    rng = np.random.default_rng(seed)
    N, D = X_obs.shape
    rank = U.shape[1]

    # Full covariance: Sigma = U U^T + D_diag * I
    # For efficiency we work with the Woodbury identity.
    X_out = X_obs.copy()

    for i in range(N):
        o_idx = np.where(obs_mask[i])[0]   # observed feature indices
        m_idx = np.where(~obs_mask[i])[0]  # missing feature indices
        if len(m_idx) == 0:
            continue
        if len(o_idx) == 0:
            X_out[i, m_idx] = mu[m_idx] + rng.normal(size=len(m_idx)) * np.sqrt(D_diag)
            continue

        x_o = X_obs[i, o_idx]
        mu_o = mu[o_idx]
        mu_m = mu[m_idx]
        U_o = U[o_idx, :]          # (|o|, rank)
        U_m = U[m_idx, :]          # (|m|, rank)

        # Sigma_oo = U_o U_o^T + D_diag I
        # inv via Woodbury: inv(Sigma_oo) = (1/D) I - (1/D) U_o (I + U_o^T U_o/D)^{-1} U_o^T / D
        M = np.eye(rank) + (U_o.T @ U_o) / D_diag   # (rank, rank)
        M_inv = np.linalg.inv(M)
        diff = x_o - mu_o
        # alpha = inv(Sigma_oo) @ diff
        alpha = diff / D_diag - U_o @ (M_inv @ (U_o.T @ diff)) / (D_diag ** 2)

        # Conditional mean: mu_m + Sigma_mo @ inv(Sigma_oo) @ (x_o - mu_o)
        # Sigma_mo = U_m U_o^T
        cond_mean = mu_m + U_m @ (U_o.T @ alpha)

        # For sampling we need the conditional covariance — but drawing from
        # the mean alone already beats naive Gaussian filling.  To keep things
        # fast, we add isotropic noise at the residual variance level.
        cond_std = np.sqrt(D_diag)
        X_out[i, m_idx] = cond_mean + rng.normal(size=len(m_idx)) * cond_std

    return X_out


def initialize_with_ppca(corrupted_data, obs_masks, rank=32, em_iters=8, seed=0):
    """Initialize reconstructions via iterative PPCA — aligned with the paper's
    fit_moments + PPCA strategy.

    1. Fit PPCA on the observed data (with mean-imputation for NaN).
    2. Impute missing values via the Gaussian posterior.
    3. Repeat for `em_iters` rounds to refine the covariance estimate.
    """
    N, seq_len, features = corrupted_data.shape
    D = seq_len * features

    # Flatten to (N, D) — each sample is a vector
    X = corrupted_data.reshape(N, D).copy()

    # Convert obs_masks (N, seq_len) bool → per-feature (N, D) bool
    obs_flat = np.repeat(obs_masks, features, axis=1)

    # Replace NaN with 0 temporarily (they'll be handled inside ppca_fit)
    X_work = np.where(np.isnan(X), 0.0, X)
    # But mark them properly for posterior imputation
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

    # Ensure observed values are exact
    for i in range(N):
        for t in range(seq_len):
            if obs_masks[i, t]:
                orig = corrupted_data[i, t]
                if not np.any(np.isnan(orig)):
                    recon[i, t] = orig

    print(f"PPCA initialization complete. {N} sequences, "
          f"range=[{recon.min():.3f}, {recon.max():.3f}]")
    return recon


# ============================= M-step dataset ==============================

class FreshCorruptionDataset(Data.Dataset):
    """M-step dataset that generates FRESH random corruption each __getitem__.

    Aligned with the paper: at each M-step, Y ~ Q(X) is a *new* corruption
    of the reconstruction X.  Using a different random corruption every time
    the sample is accessed provides more diversity than a fixed corruption.
    """

    def __init__(self, clean_data, missing_rate, noise_level=0.0, epoch_seed=0):
        if isinstance(clean_data, torch.Tensor):
            self.clean = clean_data.numpy()
        else:
            self.clean = np.asarray(clean_data, dtype=np.float32)
        self.missing_rate = missing_rate
        self.noise_level = noise_level
        self.epoch_seed = epoch_seed   # bump this every epoch for fresh corruption

    def set_epoch(self, epoch):
        self.epoch_seed = epoch

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        x = self.clean[idx].copy()

        rng = np.random.RandomState((self.epoch_seed * len(self.clean) + idx) % (2**32 - 1))
        seq_len = x.shape[0]

        # Additive noise
        if self.noise_level > 0:
            x_noisy = x + rng.normal(0, self.noise_level, x.shape).astype(np.float32)
        else:
            x_noisy = x.copy()

        # Random masking
        num_remove = int(seq_len * self.missing_rate)
        mask = np.ones(seq_len, dtype=np.float32)
        if num_remove > 0:
            remove_idx = rng.permutation(seq_len)[:num_remove]
            x_noisy[remove_idx] = np.nan
            mask[remove_idx] = 0.0

        clean_t = torch.from_numpy(x.astype(np.float32))
        corrupted_t = torch.from_numpy(x_noisy.astype(np.float32))
        mask_t = torch.from_numpy(mask)
        return clean_t, corrupted_t, mask_t


# ============================= E-step ======================================

def e_step(args, cond_model, corrupted_data, obs_masks, em_iter, device,
           sample_steps=None, logger=None):
    """E-step: sample X ~ q_theta(x | Y) for all observations.

    Uses more sampling steps than the default diffusion_steps for higher
    quality posterior samples (critical for EM convergence).
    """
    print(f"\n{'='*50}")
    print(f"E-Step  (EM iter {em_iter})")
    print(f"{'='*50}")

    if sample_steps is None:
        sample_steps = getattr(args, 'e_step_sample_steps', SCRIPT_DEFAULTS['e_step_sample_steps'])
    print(f"  Sampling steps: {sample_steps}")

    cond_model.eval()
    all_recon = []

    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    cond_channels = args.input_channels + 1

    # Temporarily override diffusion_steps for higher-quality sampling
    orig_steps = args.diffusion_steps
    args.diffusion_steps = sample_steps

    N = len(corrupted_data)
    batch_size = args.batch_size

    with torch.no_grad():
        with cond_model.ema_scope():
            process = ConditionalDiffusionProcess(
                args, cond_model.net, target_shape, cond_channels
            )
            for start in tqdm(range(0, N, batch_size), desc="E-step"):
                end = min(start + batch_size, N)
                corr_batch = corrupted_data[start:end]
                mask_batch = obs_masks[start:end]

                corr_ts = torch.tensor(corr_batch, dtype=torch.float32, device=device)
                mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)
                cond_ts = torch.nan_to_num(corr_ts, nan=0.0)

                cond_img = cond_model.ts_to_img(cond_ts)
                mask_exp = mask_ts.unsqueeze(-1).expand(-1, -1, corr_ts.shape[-1])
                mask_img = cond_model.ts_to_img(mask_exp)
                mask_img = mask_img[:, :1, :, :]

                x_img = process.sampling(cond_img, mask_img)
                x_ts = cond_model.img_to_ts(x_img)
                all_recon.append(x_ts.cpu().numpy())

    args.diffusion_steps = orig_steps  # restore

    recon = np.vstack(all_recon)

    # ---- diagnostics ----
    print(f"  Reconstructions: shape={recon.shape}, "
          f"range=[{recon.min():.4f}, {recon.max():.4f}], "
          f"mean={recon.mean():.4f}, std={recon.std():.4f}")
    _log_stats('em/e_step_recon', recon, em_iter, logger)
    _log(logger, 'em/e_step_samples', len(recon), em_iter)

    # Check how much observed values changed (should be small if model is good)
    obs_orig = corrupted_data[~np.isnan(corrupted_data)]
    obs_recon_vals = []
    for i in range(min(N, 1000)):
        for t in range(corrupted_data.shape[1]):
            if obs_masks[i, t]:
                orig = corrupted_data[i, t]
                if not np.any(np.isnan(orig)):
                    obs_recon_vals.append(np.abs(recon[i, t] - orig).mean())
    if obs_recon_vals:
        obs_mae = np.mean(obs_recon_vals)
        print(f"  Observed-value MAE (recon vs original): {obs_mae:.6f}")
        _log(logger, 'em/e_step_obs_mae', obs_mae, em_iter)

    # Cache
    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                             f'missing_{int(args.missing_rate * 100)}')
    save_reconstructions(recon, cache_dir, em_iter)

    return recon


# ============================= M-step ======================================

def m_step(args, cond_model, optimizer, scheduler, reconstructions, em_iter,
           device, logger=None):
    """M-step: train conditional model on (X, Y ~ Q(X)) pairs.

    Aligned with the paper:
      - Fresh random corruption each epoch.
      - Cosine LR schedule within each M-step.
      - More training epochs (default 200).
    """
    n_epochs = getattr(args, 'm_step_epochs', SCRIPT_DEFAULTS['m_step_epochs'])
    corruption_rate = args.em_corruption_rate if args.em_corruption_rate else args.missing_rate
    noise_level = args.em_noise_level if args.em_noise_level else args.gaussian_noise_level

    print(f"\n{'='*50}")
    print(f"M-Step  (EM iter {em_iter}) — {n_epochs} epochs, "
          f"corruption={corruption_rate:.0%}, noise={noise_level}")
    print(f"{'='*50}")

    dataset = FreshCorruptionDataset(
        reconstructions,
        missing_rate=corruption_rate,
        noise_level=noise_level,
        epoch_seed=args.seed + em_iter * 10000,
    )

    loader = Data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )

    cond_model.train()

    loss_history = []

    patience = getattr(args, 'early_stop_patience', 20)
    min_delta = getattr(args, 'early_stop_min_delta', 1e-4)
    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(n_epochs):
        dataset.set_epoch(args.seed + em_iter * 10000 + epoch)

        epoch_loss = 0.0
        n_batches = 0

        for clean, corrupted, mask in loader:
            clean = clean.to(device)
            corrupted = corrupted.to(device)
            mask = mask.to(device)

            cond_ts = torch.nan_to_num(corrupted, nan=0.0)

            x_img = cond_model.ts_to_img(clean)
            cond_img = cond_model.ts_to_img(cond_ts)

            mask_exp = mask.unsqueeze(-1).expand(-1, -1, clean.shape[-1])
            mask_img = cond_model.ts_to_img(mask_exp)
            mask_img = mask_img[:, :1, :, :]

            loss, _ = cond_model.loss_fn_conditional_full(x_img, cond_img, mask_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cond_model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            cond_model.on_train_batch_end()

            epoch_loss += loss.item()
            n_batches += 1
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == n_epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  epoch {epoch+1:4d}/{n_epochs}  loss={avg_loss:.5f}  lr={lr_now:.2e}")

        global_step = em_iter * n_epochs + epoch
        _log(logger, 'em/m_step_loss', avg_loss, global_step)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience and epoch >= 30:
            print(f"  Early stopping at epoch {epoch+1}/{n_epochs} "
                  f"(no improvement for {patience} epochs, best={best_loss:.5f})")
            break

    # ---- diagnostics ----
    first5 = np.mean(loss_history[:5]) if len(loss_history) >= 5 else loss_history[0]
    last5 = np.mean(loss_history[-5:])
    print(f"  M-step loss: first-5-avg={first5:.5f} → last-5-avg={last5:.5f} "
          f"(delta={last5 - first5:+.5f}, ran {len(loss_history)}/{n_epochs} epochs)")
    _log(logger, 'em/m_step_loss_start', first5, em_iter)
    _log(logger, 'em/m_step_loss_end', last5, em_iter)

    return cond_model


# ============================= main loop ===================================

def main(args):
    name = create_model_name_and_dir(args)
    name = f"diffem_pure_new_{name}"
    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        args.device = device

        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset ready.')

        # ============================
        # Apply improved defaults for this script
        # ============================
        # Unconditionally set our paper-aligned defaults.
        # CLI args can still override via --m_step_epochs etc.
        # We detect "user explicitly set" by checking against old argparse defaults.
        OLD_ARGPARSE_DEFAULTS = dict(m_step_epochs=50, em_iters=10,
                                     em_eval_interval=1, uncond_epochs_per_iter=None)
        for k, v in SCRIPT_DEFAULTS.items():
            old_default = OLD_ARGPARSE_DEFAULTS.get(k, '__MISSING__')
            current = getattr(args, k, '__MISSING__')
            if current == '__MISSING__' or current is None or current == old_default:
                setattr(args, k, v)

        sample_steps = getattr(args, 'e_step_sample_steps', SCRIPT_DEFAULTS['e_step_sample_steps'])
        m_epochs = args.m_step_epochs
        em_iters = args.em_iters

        print(f"\n{'#'*60}")
        print(f"  DiffEM-pure-new  |  EM iters={em_iters}  M-epochs={m_epochs}  "
              f"E-steps={sample_steps}")
        print(f"  dataset={args.dataset}  missing_rate={args.missing_rate}  "
              f"diffusion_steps(default)={args.diffusion_steps}")
        print(f"{'#'*60}\n")

        # Log all key hyperparams
        _log(logger, 'config/m_step_epochs', m_epochs, 0)
        _log(logger, 'config/e_step_sample_steps', sample_steps, 0)
        _log(logger, 'config/em_iters', em_iters, 0)
        _log(logger, 'config/missing_rate', args.missing_rate, 0)

        # ============================
        # Conditional model
        # ============================
        cond_model = TS2img_Karras_Cond(args=args, device=device).to(device)
        cond_optimizer = torch.optim.AdamW(
            cond_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        print_model_params(logger, cond_model)

        # Total steps for cosine schedule across all EM iterations
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader)
        n_train = len(corrupted_data)
        steps_per_epoch = max(n_train // args.batch_size, 1)

        total_cond_steps = (m_epochs + m_epochs) * steps_per_epoch  # init + first lap
        total_cond_steps += em_iters * m_epochs * steps_per_epoch
        cond_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            cond_optimizer, T_max=total_cond_steps, eta_min=args.learning_rate * 0.01
        )

        print(f"Extracted {n_train} sequences, "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # ============================
        # Phase 1 — PPCA Initialization
        # ============================
        print(f"\n{'='*60}")
        print("Phase 1: PPCA-based Gaussian Initialization")
        print(f"{'='*60}")

        ppca_rank = min(32, corrupted_data.shape[1] * corrupted_data.shape[2] - 1)
        initial_recon = initialize_with_ppca(
            corrupted_data, obs_masks, rank=ppca_rank, em_iters=8, seed=args.seed
        )
        _log_stats('em/init_recon', initial_recon, -1, logger)

        # ============================
        # Phase 1.5 — Initial M-step
        # ============================
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial conditional model training on PPCA recon")
        print(f"{'='*60}")

        cond_model = m_step(
            args, cond_model, cond_optimizer, cond_scheduler,
            initial_recon, em_iter=-1, device=device, logger=logger
        )

        # ============================
        # Phase 2 — EM Loop
        # ============================
        print(f"\n{'='*60}")
        print(f"Phase 2: EM Loop — {em_iters} iterations")
        print(f"{'='*60}")

        for em_iter in range(em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{em_iters}")
            print(f"{'='*60}")

            # ---------- E-step ----------
            reconstructions = e_step(
                args, cond_model, corrupted_data, obs_masks,
                em_iter, device, sample_steps=sample_steps, logger=logger
            )

            # ---------- M-step ----------
            cond_model = m_step(
                args, cond_model, cond_optimizer, cond_scheduler,
                reconstructions, em_iter, device, logger=logger
            )

            _log(logger, 'em/iteration', em_iter, em_iter)

        # ============================
        # Phase 3 — Train unconditional model (same as run_regular.py)
        # ============================
        final_metrics = train_unconditional_regular(
            args, reconstructions, test_loader, device, logger,
        )

        # ============================
        # Summary
        # ============================
        print(f"\n{'='*60}")
        print("DiffEM-pure-new Training Complete!")
        print(f"{'='*60}")
        if final_metrics:
            print("Final metrics:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")
        logging.info("DiffEM-pure-new training complete")


if __name__ == '__main__':
    # Inject defaults so the script runs without any CLI args.
    # These match: energy, 50% missing, wandb, seed=0.
    sys.argv = [
        sys.argv[0],
        '--config', './configs/seq_len_24/energy.yaml',
        '--missing_rate', '0.5',
        '--wandb', 'True',
        '--tags', 'diffem_pure_new', 'energy', '50_missing',
        '--seed', '0',
    ] + sys.argv[1:]  # any explicit CLI args still take precedence

    args = parse_args_irregular()
    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
