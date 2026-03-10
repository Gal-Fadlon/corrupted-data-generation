"""
Combined Pure DiffEM Training Script for Irregular Time Series

Implements the DiffEM paper's conditional diffusion model approach:
  - E-step: sample X ~ q_theta(x|y) using a conditional diffusion model
  - M-step: train d_theta(x_t, t | y) on (clean, freshly-corrupted) pairs

Supported corruption types (--corruption_type):
  - missing:                random masking of time steps
  - gaussian_noise:         additive Gaussian noise on all time steps
  - temporal_smoothing:     moving-average smoothing along time axis
  - combined_missing_noise: random masking + additive noise on observed entries

Key features:
  1. Observation consistency (missing/combined only): after E-step sampling,
     observed positions are replaced with true observed values.
  2. Fresh optimizer + EMA reset at every M-step.
  3. PPCA-based Gaussian initialization (missing/combined) or identity init.
  4. FreshCorruptionDataset — generates a new random corruption every
     __getitem__ call for maximal diversity during M-step training.
  5. Configurable E-step sampling steps for better posterior samples.
"""

import torch
import torch.multiprocessing
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
from utils.utils import (
    restore_state, create_model_name_and_dir,
    print_model_params, log_config_and_tags,
)
from utils.utils_data import (
    gen_dataloader, apply_corruption,
    save_reconstructions, load_reconstructions,
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras, TS2img_Karras_Cond
from models.sampler import DiffusionProcess, ConditionalDiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

DEFAULT_M_STEP_EPOCHS = 100
DEFAULT_E_STEP_SAMPLE_STEPS = 64


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log(logger, name, value, step):
    if logger is not None:
        logger.log(name, value, step)


def _log_stats(tag, arr, step, logger):
    arr = np.asarray(arr).flatten()
    _log(logger, f'{tag}/mean', float(np.nanmean(arr)), step)
    _log(logger, f'{tag}/std',  float(np.nanstd(arr)),  step)
    _log(logger, f'{tag}/min',  float(np.nanmin(arr)),  step)
    _log(logger, f'{tag}/max',  float(np.nanmax(arr)),  step)


def _upload_and_cleanup(logger, plot_path, em_iter):
    if logger is None:
        return
    uploaded = False
    try:
        logger.log_file('test/memorization/histogram', plot_path, em_iter)
        uploaded = True
    except Exception as exc:
        print(f"  [warn] Plot upload failed: {exc}")
    if uploaded:
        try:
            os.remove(plot_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def get_corrupted_data_from_loader(train_loader):
    """Extract corrupted time series and observation masks from the train loader."""
    all_corrupted, all_masks = [], []
    for data in train_loader:
        x_irregular = data[0]                        # (B, seq_len, features+1)
        x_ts = x_irregular[:, :, :-1]                # drop time-index column
        mask = ~torch.isnan(x_ts).any(dim=-1)        # (B, seq_len)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())
    return np.vstack(all_corrupted), np.vstack(all_masks)


# ---------------------------------------------------------------------------
# PPCA initialisation  (from run_diffem_pure_new.py)
# ---------------------------------------------------------------------------

def ppca_fit(X, rank=16):
    """Probabilistic PCA on (N, D) data with possible NaN entries."""
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
    """Impute missing values via the PPCA Gaussian posterior."""
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

        x_o  = X_obs[i, o_idx]
        mu_o, mu_m = mu[o_idx], mu[m_idx]
        U_o, U_m   = U[o_idx, :], U[m_idx, :]

        M     = np.eye(rank) + (U_o.T @ U_o) / D_diag
        M_inv = np.linalg.inv(M)
        diff  = x_o - mu_o
        alpha = diff / D_diag - U_o @ (M_inv @ (U_o.T @ diff)) / (D_diag ** 2)

        cond_mean = mu_m + U_m @ (U_o.T @ alpha)
        cond_std  = np.sqrt(D_diag)
        X_out[i, m_idx] = cond_mean + rng.normal(size=len(m_idx)) * cond_std

    return X_out


def initialize_with_ppca(corrupted_data, obs_masks, rank=32, ppca_iters=8,
                         seed=0):
    """Iterative PPCA initialisation — paper-aligned fit_moments strategy."""
    N, seq_len, features = corrupted_data.shape
    D = seq_len * features

    X = corrupted_data.reshape(N, D).copy()
    obs_flat = np.repeat(obs_masks, features, axis=1)

    X_work = np.where(np.isnan(X), 0.0, X)
    X_work[~obs_flat] = np.nan

    print(f"PPCA init: rank={rank}, iters={ppca_iters}, shape=({N}, {D})")

    for it in range(ppca_iters):
        mu, U, D_diag = ppca_fit(X_work, rank=min(rank, D - 1))
        X_work = gaussian_posterior_impute(
            X_obs=np.where(obs_flat, corrupted_data.reshape(N, D), np.nan),
            obs_mask=obs_flat,
            mu=mu, U=U, D_diag=D_diag,
            seed=seed + it,
        )
        var_expl = np.sum(np.maximum(np.linalg.norm(U, axis=0) ** 2, 0))
        total_var = var_expl + D_diag * D
        print(f"  PPCA iter {it+1}/{ppca_iters}: D_diag={D_diag:.4e}, "
              f"var_ratio={var_expl / total_var:.3f}, "
              f"range=[{X_work.min():.3f}, {X_work.max():.3f}]")

    recon = X_work.reshape(N, seq_len, features)

    for i in range(N):
        for t in range(seq_len):
            if obs_masks[i, t]:
                orig = corrupted_data[i, t]
                if not np.any(np.isnan(orig)):
                    recon[i, t] = orig

    print(f"PPCA init done: {N} sequences, "
          f"range=[{recon.min():.3f}, {recon.max():.3f}]")
    return recon


# ---------------------------------------------------------------------------
# Corruption helpers
# ---------------------------------------------------------------------------

def temporal_moving_average_np(x, window):
    """Apply symmetric moving average along time axis with reflect padding.

    Args:
        x: (seq_len, features) numpy array
        window: odd integer window size

    Returns:
        smoothed: (seq_len, features) numpy array
    """
    pad = window // 2
    seq_len, features = x.shape
    out = np.empty_like(x)
    for c in range(features):
        padded = np.pad(x[:, c], pad, mode='reflect')
        kernel = np.ones(window, dtype=np.float32) / window
        out[:, c] = np.convolve(padded, kernel, mode='valid')
    return out


def get_clean_data_from_loader(loader):
    """Extract clean (non-corrupted) time series from a data loader.

    The loader is expected to yield batches where data[0] has shape
    (B, seq_len, features+1) with the last column being a time index.
    This function strips the time column and returns the raw signal.
    """
    all_data = []
    for data in loader:
        x = data[0][:, :, :-1]
        all_data.append(x.numpy())
    return np.vstack(all_data)


# ---------------------------------------------------------------------------
# FreshCorruptionDataset — general corruption types
# ---------------------------------------------------------------------------

class FreshCorruptionDataset(Data.Dataset):
    """M-step dataset that generates a new random corruption on every access.

    Supports multiple corruption types:
      - missing:                random mask + optional noise, NaN for missing
      - gaussian_noise:         additive N(0, sigma^2) noise, mask = all 1s
      - temporal_smoothing:     moving-average smoothing, mask = all 1s
      - combined_missing_noise: missing mask + noise on observed entries
    """

    def __init__(self, clean_data, corruption_type='missing',
                 missing_rate=0.0, noise_level=0.0, smoothing_window=5,
                 epoch_seed=0):
        self.clean = (clean_data.numpy() if isinstance(clean_data, torch.Tensor)
                      else np.asarray(clean_data, dtype=np.float32))
        self.corruption_type = corruption_type
        self.missing_rate = missing_rate
        self.noise_level = noise_level
        self.smoothing_window = smoothing_window
        self.epoch_seed = epoch_seed

    def set_epoch(self, epoch):
        self.epoch_seed = epoch

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        x = self.clean[idx].copy()
        rng = np.random.RandomState(
            (self.epoch_seed * len(self.clean) + idx) % (2**32 - 1))
        seq_len = x.shape[0]

        if self.corruption_type == 'gaussian_noise':
            sigma = self.noise_level if self.noise_level > 0 else 0.1
            corrupted = x + rng.normal(0, sigma, x.shape).astype(np.float32)
            mask = np.ones(seq_len, dtype=np.float32)

        elif self.corruption_type == 'temporal_smoothing':
            corrupted = temporal_moving_average_np(x, self.smoothing_window)
            mask = np.ones(seq_len, dtype=np.float32)

        elif self.corruption_type == 'combined_missing_noise':
            sigma = self.noise_level if self.noise_level > 0 else 0.1
            corrupted = x + rng.normal(0, sigma, x.shape).astype(np.float32)
            num_remove = int(seq_len * self.missing_rate)
            mask = np.ones(seq_len, dtype=np.float32)
            if num_remove > 0:
                remove_idx = rng.permutation(seq_len)[:num_remove]
                corrupted[remove_idx] = np.nan
                mask[remove_idx] = 0.0

        else:  # 'missing' (default)
            if self.noise_level > 0:
                corrupted = x + rng.normal(0, self.noise_level, x.shape).astype(np.float32)
            else:
                corrupted = x.copy()
            num_remove = int(seq_len * self.missing_rate)
            mask = np.ones(seq_len, dtype=np.float32)
            if num_remove > 0:
                remove_idx = rng.permutation(seq_len)[:num_remove]
                corrupted[remove_idx] = np.nan
                mask[remove_idx] = 0.0

        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(corrupted.astype(np.float32)),
                torch.from_numpy(mask))


# ---------------------------------------------------------------------------
# E-step  (observation consistency + configurable sampling steps + diagnostics)
# ---------------------------------------------------------------------------

def e_step(args, cond_model, corrupted_data, obs_masks, em_iter, device,
           sample_steps=None, logger=None):
    """Sample reconstructions from the conditional model.

    For corruption types with clean anchors (missing, combined_missing_noise),
    observation consistency is enforced: observed positions are replaced with
    true observed values after sampling.

    For fully-corrupted types (gaussian_noise, temporal_smoothing), no
    observation consistency is applied since all entries are corrupted.
    """
    corruption_type = getattr(args, 'corruption_type', 'missing')
    if sample_steps is None:
        sample_steps = getattr(args, 'e_step_sample_steps',
                               DEFAULT_E_STEP_SAMPLE_STEPS)

    use_obs_consistency = corruption_type == 'missing'

    print(f"\n{'='*50}")
    print(f"E-Step  (EM iter {em_iter})  sampling_steps={sample_steps}  "
          f"type={corruption_type}")
    if use_obs_consistency:
        print(f"  Observation consistency: ON")
    else:
        print(f"  Observation consistency: OFF (all entries corrupted)")
    print(f"{'='*50}")

    cond_model.eval()
    all_recon = []

    target_shape = (args.input_channels, args.img_resolution,
                    args.img_resolution)
    cond_channels = args.input_channels + 1

    orig_steps = args.diffusion_steps
    args.diffusion_steps = sample_steps

    N = len(corrupted_data)
    batch_size = args.batch_size

    with torch.no_grad():
        with cond_model.ema_scope():
            process = ConditionalDiffusionProcess(
                args, cond_model.net, target_shape, cond_channels)

            for start in tqdm(range(0, N, batch_size), desc="E-step"):
                end = min(start + batch_size, N)
                corr_batch = corrupted_data[start:end]
                mask_batch = obs_masks[start:end]

                corr_ts = torch.tensor(corr_batch, dtype=torch.float32,
                                       device=device)
                mask_ts = torch.tensor(mask_batch, dtype=torch.float32,
                                       device=device)

                cond_ts  = torch.nan_to_num(corr_ts, nan=0.0)
                cond_img = cond_model.ts_to_img(cond_ts)

                mask_exp = mask_ts.unsqueeze(-1).expand_as(corr_ts)
                mask_img = cond_model.ts_to_img(mask_exp)
                mask_img = mask_img[:, :1, :, :]

                x_img = process.sampling(cond_img, mask_img)
                x_ts  = cond_model.img_to_ts(x_img)

                x_np = x_ts.cpu().numpy()

                if use_obs_consistency:
                    for b in range(len(corr_batch)):
                        obs = mask_batch[b].astype(bool)
                        orig = corr_batch[b][obs, :]
                        if not np.any(np.isnan(orig)):
                            x_np[b, obs, :] = orig

                all_recon.append(x_np)

    args.diffusion_steps = orig_steps

    recon = np.vstack(all_recon)

    # --- Diagnostics ---
    print(f"  Reconstructions: shape={recon.shape}, "
          f"range=[{recon.min():.4f}, {recon.max():.4f}], "
          f"mean={recon.mean():.4f}, std={recon.std():.4f}")
    _log_stats('em/e_step_recon', recon, em_iter, logger)
    _log(logger, 'em/e_step_samples', len(recon), em_iter)

    if use_obs_consistency:
        obs_recon_vals = []
        for i in range(min(N, 1000)):
            for t in range(corrupted_data.shape[1]):
                if obs_masks[i, t]:
                    orig = corrupted_data[i, t]
                    if not np.any(np.isnan(orig)):
                        obs_recon_vals.append(np.abs(recon[i, t] - orig).mean())
        if obs_recon_vals:
            obs_mae = np.mean(obs_recon_vals)
            print(f"  Observed-value MAE (should be ~0): {obs_mae:.6f}")
            _log(logger, 'em/e_step_obs_mae', obs_mae, em_iter)

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                             _cache_subdir(args))
    save_reconstructions(recon, cache_dir, em_iter)

    return recon


def _cache_subdir(args):
    """Build a descriptive cache subdirectory name from corruption config."""
    ct = getattr(args, 'corruption_type', 'missing')
    if ct == 'missing':
        return f'missing_{int(args.missing_rate * 100)}'
    elif ct == 'gaussian_noise':
        level = getattr(args, 'corruption_noise_level', 0.1)
        return f'gaussian_noise_{level}'
    elif ct == 'temporal_smoothing':
        w = getattr(args, 'smoothing_window', 5)
        return f'temporal_smooth_w{w}'
    elif ct == 'combined_missing_noise':
        rate = args.missing_rate
        level = getattr(args, 'corruption_noise_level', 0.1)
        return f'combined_m{int(rate*100)}_n{level}'
    return ct


# ---------------------------------------------------------------------------
# M-step  (fresh optimizer + EMA reset + FreshCorruptionDataset)
# ---------------------------------------------------------------------------

def m_step(args, cond_model, reconstructions, em_iter, device, logger=None):
    """Train conditional model with fresh optimizer, EMA reset, and fresh
    corruption each epoch.

    Per the paper's reference code: optimizer and EMA state are reset each
    lap while model weights continue from the previous lap.
    """
    m_epochs = getattr(args, 'm_step_epochs', DEFAULT_M_STEP_EPOCHS)
    corruption_type = getattr(args, 'corruption_type', 'missing')

    corruption_rate = (args.em_corruption_rate
                       if args.em_corruption_rate else args.missing_rate)
    noise_level = getattr(args, 'corruption_noise_level', 0.01)
    if args.em_noise_level:
        noise_level = args.em_noise_level
    elif corruption_type == 'missing':
        noise_level = (args.em_noise_level
                       if args.em_noise_level else args.gaussian_noise_level)
    smoothing_window = getattr(args, 'smoothing_window', 5)

    print(f"\n{'='*50}")
    print(f"M-Step  (EM iter {em_iter}) — {m_epochs} epochs, "
          f"type={corruption_type}")
    print(f"  missing_rate={corruption_rate}, noise_level={noise_level}, "
          f"smooth_window={smoothing_window}")
    print(f"{'='*50}")

    optimizer = torch.optim.AdamW(
        cond_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    cond_model.reset_ema()

    dataset = FreshCorruptionDataset(
        reconstructions,
        corruption_type=corruption_type,
        missing_rate=corruption_rate,
        noise_level=noise_level,
        smoothing_window=smoothing_window,
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

    for epoch in range(m_epochs):
        dataset.set_epoch(args.seed + em_iter * 10000 + epoch)

        epoch_loss, n_batches = 0.0, 0
        for clean, corrupted, mask in loader:
            clean     = clean.to(device)
            corrupted = corrupted.to(device)
            mask      = mask.to(device)

            cond_ts      = torch.nan_to_num(corrupted, nan=0.0)
            x_img        = cond_model.ts_to_img(clean)
            cond_img     = cond_model.ts_to_img(cond_ts)
            mask_exp     = mask.unsqueeze(-1).expand_as(clean)
            obs_mask_img = cond_model.ts_to_img(mask_exp)
            obs_mask_img = obs_mask_img[:, :1, :, :]

            loss, _ = cond_model.loss_fn_conditional_full(
                x_img, cond_img, obs_mask_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cond_model.parameters(), 1.0)
            optimizer.step()
            cond_model.on_train_batch_end()

            epoch_loss += loss.item()
            n_batches  += 1
            torch.cuda.empty_cache()

        avg = epoch_loss / max(n_batches, 1)
        loss_history.append(avg)

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == m_epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  epoch {epoch+1:4d}/{m_epochs}  "
                  f"loss={avg:.5f}  lr={lr_now:.2e}")

        _log(logger, 'em/m_step_loss', avg, em_iter * m_epochs + epoch)

        if avg < best_loss - min_delta:
            best_loss = avg
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience and epoch >= 30:
            print(f"  Early stopping at epoch {epoch+1}/{m_epochs} "
                  f"(no improvement for {patience} epochs, best={best_loss:.5f})")
            break

    first5 = np.mean(loss_history[:5]) if len(loss_history) >= 5 else loss_history[0]
    last5  = np.mean(loss_history[-5:])
    print(f"  M-step loss: first-5={first5:.5f} → last-5={last5:.5f} "
          f"(Δ={last5 - first5:+.5f}, ran {len(loss_history)}/{m_epochs} epochs)")
    _log(logger, 'em/m_step_loss_start', first5, em_iter)
    _log(logger, 'em/m_step_loss_end',   last5,  em_iter)

    return cond_model


# ---------------------------------------------------------------------------
# Unconditional model evaluation  (fresh optimizer + EMA reset)
# ---------------------------------------------------------------------------

def train_and_evaluate_unconditional(args, uncond_model, reconstructions,
                                     test_loader, em_iter, device,
                                     logger=None):
    """Train unconditional model on current reconstructions and evaluate.

    Fresh optimizer + EMA reset each call so the unconditional model trains
    cleanly on the latest reconstructions.
    """
    uncond_epochs = getattr(args, 'uncond_epochs_per_iter', None)
    if uncond_epochs is None:
        uncond_epochs = getattr(args, 'm_step_epochs', DEFAULT_M_STEP_EPOCHS)

    print(f"\n--- Uncond Train+Eval (EM iter {em_iter}), "
          f"{uncond_epochs} epochs ---")

    uncond_opt = torch.optim.AdamW(
        uncond_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    uncond_model.reset_ema()

    recon_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(reconstructions, dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )

    # --- Training ---
    uncond_model.train()
    uncond_loss_history = []

    patience = getattr(args, 'early_stop_patience', 20)
    min_delta = getattr(args, 'early_stop_min_delta', 1e-4)
    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(uncond_epochs):
        epoch_loss, nb = 0.0, 0
        for (x_clean,) in recon_loader:
            x_clean = x_clean.to(device)
            x_img   = uncond_model.ts_to_img(x_clean)
            loss, _ = uncond_model.loss_fn_irregular(x_img)

            uncond_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            uncond_opt.step()
            uncond_model.on_train_batch_end()

            epoch_loss += loss.item()
            nb += 1
            torch.cuda.empty_cache()

        avg = epoch_loss / max(nb, 1)
        uncond_loss_history.append(avg)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    uncond epoch {epoch+1}/{uncond_epochs}  loss={avg:.5f}")
        _log(logger, 'train/uncond_loss', avg, em_iter * uncond_epochs + epoch)

        if avg < best_loss - min_delta:
            best_loss = avg
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience and epoch >= 30:
            print(f"    Early stopping at epoch {epoch+1}/{uncond_epochs} "
                  f"(no improvement for {patience} epochs, best={best_loss:.5f})")
            break

    # --- Evaluation (unconditional sampling, same protocol as run_irregular.py) ---
    uncond_model.eval()
    gen_sig, real_sig = [], []

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution,
                 args.img_resolution),
            )
            for data in tqdm(test_loader, desc="Uncond eval"):
                x_img = process.sampling(sampling_number=data[0].shape[0])
                x_ts  = uncond_model.img_to_ts(x_img)
                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig  = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)
    print(f"  Unconditional metrics (EM iter {em_iter}):")
    for k, v in scores.items():
        print(f"    {k}: {v:.4f}")
        _log(logger, f'test/{k}', v, em_iter)

    mem_path = f"mem_hist_em{em_iter}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig,
        device=device, plot_path=mem_path,
    )
    for k, v in mem_stats.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    mem/{k}: {val_str}")
        _log(logger, f'test/memorization/{k}', v, em_iter)
    _upload_and_cleanup(logger, mem_path, em_iter)

    return scores


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _prepare_corrupted_data(args, train_loader, test_loader, device):
    """Prepare corrupted observations and masks based on corruption_type.

    For missing/combined_missing_noise the dataloader already provides
    corrupted data with NaN.  For gaussian_noise/temporal_smoothing we
    load clean data and apply corruption ourselves.

    Returns:
        corrupted_data: (N, seq_len, features) — the corrupted observations
        obs_masks:      (N, seq_len)           — 1 where observed, 0 where missing
        clean_data:     (N, seq_len, features) or None — clean data if available
    """
    corruption_type = getattr(args, 'corruption_type', 'missing')

    if corruption_type in ('missing', 'combined_missing_noise'):
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader)
        clean_data = None

        if corruption_type == 'combined_missing_noise':
            noise_level = getattr(args, 'corruption_noise_level', 0.1)
            rng = np.random.RandomState(args.seed)
            noise = noise_level * rng.randn(*corrupted_data.shape).astype(np.float32)
            observed = np.repeat(
                obs_masks[:, :, None].astype(bool),
                corrupted_data.shape[2], axis=2)
            corrupted_data = np.where(
                observed & ~np.isnan(corrupted_data),
                corrupted_data + noise,
                corrupted_data
            )

        pct = obs_masks.sum() / obs_masks.size * 100
        print(f"Extracted {len(corrupted_data)} sequences, "
              f"{pct:.1f}% observed [{corruption_type}]")
        return corrupted_data, obs_masks, clean_data

    clean_data = get_clean_data_from_loader(train_loader)
    N, seq_len, features = clean_data.shape
    obs_masks = np.ones((N, seq_len), dtype=np.float32)

    if corruption_type == 'gaussian_noise':
        noise_level = getattr(args, 'corruption_noise_level', 0.1)
        rng = np.random.RandomState(args.seed)
        corrupted_data = clean_data + noise_level * rng.randn(
            *clean_data.shape).astype(np.float32)
        print(f"Applied Gaussian noise (sigma={noise_level}) to "
              f"{N} sequences [{corruption_type}]")

    elif corruption_type == 'temporal_smoothing':
        window = getattr(args, 'smoothing_window', 5)
        corrupted_data = np.empty_like(clean_data)
        for i in range(N):
            corrupted_data[i] = temporal_moving_average_np(
                clean_data[i], window)
        print(f"Applied temporal smoothing (window={window}) to "
              f"{N} sequences [{corruption_type}]")

    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")

    return corrupted_data, obs_masks, clean_data


def _initialize_reconstructions(args, corrupted_data, obs_masks, clean_data):
    """Generate initial reconstructions for the first M-step.

    For missing data: PPCA-based imputation.
    For fully-corrupted data: use the corrupted observations directly
    (the conditional model will learn to invert the corruption).
    """
    corruption_type = getattr(args, 'corruption_type', 'missing')

    if corruption_type in ('missing', 'combined_missing_noise'):
        ppca_rank = min(
            getattr(args, 'ppca_rank', 32),
            corrupted_data.shape[1] * corrupted_data.shape[2] - 1)
        ppca_iters = getattr(args, 'ppca_iters', 8)
        return initialize_with_ppca(
            corrupted_data, obs_masks,
            rank=ppca_rank, ppca_iters=ppca_iters, seed=args.seed,
        )

    print(f"  Using corrupted data as initial reconstruction "
          f"({corruption_type}: no missing entries to impute)")
    return corrupted_data.copy()


def main(args):
    name = create_model_name_and_dir(args)
    name = f"diffem_pure_combined_{name}"

    logging.info(args)

    with (CompositeLogger([WandbLogger()]) if args.wandb
          else PrintLogger()) as logger:
        log_config_and_tags(args, logger, name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        args.device = device

        corruption_type = getattr(args, 'corruption_type', 'missing')
        m_epochs = getattr(args, 'm_step_epochs', DEFAULT_M_STEP_EPOCHS)
        sample_steps = getattr(args, 'e_step_sample_steps',
                               DEFAULT_E_STEP_SAMPLE_STEPS)

        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset ready.')

        print(f"\n{'#'*60}")
        print(f"  DiffEM-pure-combined  [{corruption_type}]")
        print(f"  EM iters={args.em_iters}  M-epochs={m_epochs}  "
              f"E-steps={sample_steps}")
        print(f"  dataset={args.dataset}  missing={args.missing_rate}  "
              f"noise={getattr(args, 'corruption_noise_level', 0)}")
        print(f"  diffusion_steps={args.diffusion_steps}")
        print(f"{'#'*60}\n")

        _log(logger, 'config/m_step_epochs', m_epochs, 0)
        _log(logger, 'config/e_step_sample_steps', sample_steps, 0)
        _log(logger, 'config/em_iters', args.em_iters, 0)
        _log(logger, 'config/missing_rate', args.missing_rate, 0)
        _log(logger, 'config/corruption_noise_level',
             getattr(args, 'corruption_noise_level', 0), 0)

        # ---- Conditional model ----
        cond_model = TS2img_Karras_Cond(
            args=args, device=device).to(device)
        print_model_params(logger, cond_model)

        # ---- Prepare corrupted observations & masks ----
        corrupted_data, obs_masks, clean_data = _prepare_corrupted_data(
            args, train_loader, test_loader, device)

        # ================================================================
        # Phase 1 — Initialisation
        # ================================================================
        print(f"\n{'='*60}\nPhase 1: Initialisation [{corruption_type}]\n"
              f"{'='*60}")
        initial_recon = _initialize_reconstructions(
            args, corrupted_data, obs_masks, clean_data)
        _log_stats('em/init_recon', initial_recon, -1, logger)

        # ================================================================
        # Phase 1.5 — Initial M-step
        # ================================================================
        print(f"\n{'='*60}\n"
              f"Phase 1.5: Initial conditional model training\n{'='*60}")
        cond_model = m_step(
            args, cond_model, initial_recon,
            em_iter=-1, device=device, logger=logger,
        )

        # ---- Unconditional model (evaluation only) ----
        uncond_model = TS2img_Karras(
            args=args, device=device).to(device)
        print("Unconditional evaluation model created.")

        # ================================================================
        # Phase 2 — EM loop
        # ================================================================
        print(f"\n{'='*60}\n"
              f"Pure DiffEM [{corruption_type}] — "
              f"{args.em_iters} EM iterations\n{'='*60}")

        best_metrics = None
        best_em_iter = -1

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}\n"
                  f"EM Iteration {em_iter + 1}/{args.em_iters}\n{'='*60}")

            reconstructions = e_step(
                args, cond_model, corrupted_data, obs_masks,
                em_iter, device,
                sample_steps=sample_steps, logger=logger,
            )

            cond_model = m_step(
                args, cond_model, reconstructions,
                em_iter, device, logger=logger,
            )

            if ((em_iter + 1) % args.em_eval_interval == 0
                    or em_iter == args.em_iters - 1):
                metrics = train_and_evaluate_unconditional(
                    args, uncond_model, reconstructions,
                    test_loader, em_iter, device, logger,
                )
                disc = metrics.get('disc_mean', float('inf'))
                _log(logger, 'test/disc_mean_track', disc, em_iter)

                if (best_metrics is None
                        or disc < best_metrics.get('disc_mean', float('inf'))):
                    best_metrics = metrics
                    best_em_iter = em_iter
                    print(f"  *** New best disc_mean={disc:.4f} "
                          f"at EM iter {em_iter} ***")

            _log(logger, 'em/iteration', em_iter, em_iter)

        # ================================================================
        # Done
        # ================================================================
        print(f"\n{'='*60}\n"
              f"Pure DiffEM (combined) — Complete\n{'='*60}")
        if best_metrics:
            print(f"Best metrics at EM iter {best_em_iter}:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    if not any('m_step_epochs' in a for a in sys.argv):
        args.m_step_epochs = DEFAULT_M_STEP_EPOCHS
        print(f"[combined] m_step_epochs = {DEFAULT_M_STEP_EPOCHS} "
              f"(override with --m_step_epochs)")

    if not any('e_step_sample_steps' in a for a in sys.argv):
        args.e_step_sample_steps = DEFAULT_E_STEP_SAMPLE_STEPS
        print(f"[combined] e_step_sample_steps = {DEFAULT_E_STEP_SAMPLE_STEPS} "
              f"(override with --e_step_sample_steps)")

    if not hasattr(args, 'smoothing_window'):
        args.smoothing_window = 5

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    main(args)
