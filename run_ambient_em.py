"""
AmbientEM: State-of-the-Art Irregular-to-Regular Time Series Generation

Combines DiffEM's EM framework (arXiv:2510.12691) with Ambient-Omni's
noise-level-aware training (arXiv:2506.10038) for learning clean time
series distributions from 50 % missing data.

Key contributions over existing DiffEM variants:
  1. PPCA initialisation (from DiffEM paper) for covariance-aware imputation.
  2. Conditional EM loop with observation consistency and fresh optim/EMA reset.
  3. **Ambient-aware unconditional training**: the final unconditional model
     trains on a *mixture* of EM reconstructions (standard EDM loss) and the
     original corrupted data (Ambient masked loss from Ambient Diffusion /
     Ambient-Omni).  This injects the original data diversity back into the
     model and reduces mode collapse from imperfect reconstructions.

Evaluation is identical to run_irregular.py: unconditional sampling via the
EDM Heun sampler, then disc_score via discriminative_score_metrics.
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

DEFAULT_M_STEP_EPOCHS = 80
DEFAULT_E_STEP_SAMPLE_STEPS = 64
DEFAULT_UNCOND_EPOCHS = 1000
DEFAULT_UNCOND_EVAL_EPOCHS = 80
DEFAULT_EARLY_STOP_PATIENCE = 20
DEFAULT_EARLY_STOP_MIN_DELTA = 1e-4


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


def _upload_and_cleanup(logger, plot_path, step_id):
    if logger is None:
        return
    uploaded = False
    try:
        logger.log_file('test/memorization/histogram', plot_path, step_id)
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
    """Extract corrupted time series and per-timestep observation masks."""
    all_corrupted, all_masks = [], []
    for data in train_loader:
        x_irregular = data[0]                        # (B, seq_len, features+1)
        x_ts = x_irregular[:, :, :-1]                # drop time-index column
        mask = ~torch.isnan(x_ts).any(dim=-1)        # (B, seq_len)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())
    return np.vstack(all_corrupted), np.vstack(all_masks)


# ---------------------------------------------------------------------------
# PPCA initialisation
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
# FreshCorruptionDataset
# ---------------------------------------------------------------------------

class FreshCorruptionDataset(Data.Dataset):
    """M-step dataset that generates a new random corruption on every access."""

    def __init__(self, clean_data, missing_rate, noise_level=0.0,
                 epoch_seed=0):
        self.clean = (clean_data.numpy() if isinstance(clean_data, torch.Tensor)
                      else np.asarray(clean_data, dtype=np.float32))
        self.missing_rate = missing_rate
        self.noise_level = noise_level
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

        if self.noise_level > 0:
            x_noisy = x + rng.normal(0, self.noise_level, x.shape).astype(np.float32)
        else:
            x_noisy = x.copy()

        num_remove = int(seq_len * self.missing_rate)
        mask = np.ones(seq_len, dtype=np.float32)
        if num_remove > 0:
            remove_idx = rng.permutation(seq_len)[:num_remove]
            x_noisy[remove_idx] = np.nan
            mask[remove_idx] = 0.0

        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(x_noisy.astype(np.float32)),
                torch.from_numpy(mask))


# ---------------------------------------------------------------------------
# E-step
# ---------------------------------------------------------------------------

def e_step(args, cond_model, corrupted_data, obs_masks, em_iter, device,
           sample_steps=None, logger=None):
    """Sample reconstructions and enforce observation consistency."""
    if sample_steps is None:
        sample_steps = getattr(args, 'e_step_sample_steps',
                               DEFAULT_E_STEP_SAMPLE_STEPS)

    print(f"\n{'='*50}")
    print(f"E-Step  (EM iter {em_iter})  sampling_steps={sample_steps}")
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

                # --- Observation consistency ---
                x_np = x_ts.cpu().numpy()
                for b in range(len(corr_batch)):
                    obs = mask_batch[b].astype(bool)
                    x_np[b, obs, :] = corr_batch[b][obs, :]

                all_recon.append(x_np)

    args.diffusion_steps = orig_steps
    recon = np.vstack(all_recon)

    print(f"  Reconstructions: shape={recon.shape}, "
          f"range=[{recon.min():.4f}, {recon.max():.4f}], "
          f"mean={recon.mean():.4f}, std={recon.std():.4f}")
    _log_stats('em/e_step_recon', recon, em_iter, logger)
    _log(logger, 'em/e_step_samples', len(recon), em_iter)

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
                             f'missing_{int(args.missing_rate * 100)}')
    save_reconstructions(recon, cache_dir, em_iter)

    return recon


# ---------------------------------------------------------------------------
# M-step
# ---------------------------------------------------------------------------

def m_step(args, cond_model, reconstructions, em_iter, device, logger=None):
    """Train conditional model with fresh optimizer, EMA reset, and fresh
    corruption each epoch."""
    m_epochs = getattr(args, 'm_step_epochs', DEFAULT_M_STEP_EPOCHS)
    corruption_rate = (args.em_corruption_rate
                       if args.em_corruption_rate else args.missing_rate)
    noise_level = (args.em_noise_level
                   if args.em_noise_level else args.gaussian_noise_level)

    print(f"\n{'='*50}")
    print(f"M-Step  (EM iter {em_iter}) — {m_epochs} epochs, "
          f"corruption={corruption_rate:.0%}, noise={noise_level}")
    print(f"{'='*50}")

    optimizer = torch.optim.AdamW(
        cond_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    cond_model.reset_ema()

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

    patience = getattr(args, 'early_stop_patience', DEFAULT_EARLY_STOP_PATIENCE)
    min_delta = getattr(args, 'early_stop_min_delta', DEFAULT_EARLY_STOP_MIN_DELTA)

    cond_model.train()
    loss_history = []
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
            _log(logger, 'em/m_step_early_stop_epoch', epoch + 1, em_iter)
            break

    first5 = np.mean(loss_history[:5]) if len(loss_history) >= 5 else loss_history[0]
    last5  = np.mean(loss_history[-5:])
    print(f"  M-step loss: first-5={first5:.5f} -> last-5={last5:.5f} "
          f"(delta={last5 - first5:+.5f}), ran {len(loss_history)}/{m_epochs} epochs")
    _log(logger, 'em/m_step_loss_start', first5, em_iter)
    _log(logger, 'em/m_step_loss_end',   last5,  em_iter)
    _log(logger, 'em/m_step_actual_epochs', len(loss_history), em_iter)

    return cond_model


# ---------------------------------------------------------------------------
# Ambient-aware unconditional training  (novel contribution)
# ---------------------------------------------------------------------------

class AmbientMixedDataset(Data.Dataset):
    """Dataset that returns *either* a clean EM reconstruction or an original
    corrupted observation (with its mask) for ambient-aware unconditional
    training.

    Each sample is drawn probabilistically: with probability ``alpha`` we
    return a clean EM reconstruction, otherwise an original corrupted sample.
    The ``is_clean`` flag tells the training loop which loss to apply:
      - is_clean=True  -> standard EDM denoising loss (loss_fn_irregular)
      - is_clean=False -> ambient masked loss (loss_fn_ambient_masked)
    """

    def __init__(self, reconstructions, corrupted_data, obs_masks, alpha=0.7):
        """
        Args:
            reconstructions: (N, T, C) clean EM reconstructions.
            corrupted_data:  (N, T, C) original corrupted data (zero-filled).
            obs_masks:       (N, T) boolean masks (True=observed).
            alpha:           probability of drawing a clean reconstruction.
        """
        self.recon = np.asarray(reconstructions, dtype=np.float32)
        self.corrupted = np.asarray(corrupted_data, dtype=np.float32)
        self.obs_masks = np.asarray(obs_masks)
        self.alpha = alpha
        self.n_clean = len(self.recon)
        self.n_corr = len(self.corrupted)

    def __len__(self):
        return max(self.n_clean, self.n_corr)

    def __getitem__(self, idx):
        if np.random.random() < self.alpha:
            cidx = idx % self.n_clean
            x = self.recon[cidx]
            mask = np.ones(x.shape[0], dtype=np.float32)
            is_clean = 1.0
        else:
            cidx = idx % self.n_corr
            x = self.corrupted[cidx].copy()
            mask = self.obs_masks[cidx].astype(np.float32)
            x = np.nan_to_num(x, nan=0.0)
            is_clean = 0.0

        return (torch.from_numpy(x),
                torch.from_numpy(mask),
                torch.tensor(is_clean))


def train_unconditional_ambient(args, uncond_model, reconstructions,
                                corrupted_data, obs_masks,
                                test_loader, device, logger=None):
    """Ambient-aware unconditional training: mixed EM reconstructions + original
    corrupted data.

    For clean reconstructions: standard EDM loss (loss_fn_irregular).
    For corrupted data: ambient masked loss (loss_fn_ambient_masked) — forces
    the model to learn from positions it didn't see, injecting original data
    diversity.
    """
    uncond_epochs = getattr(args, 'uncond_epochs', DEFAULT_UNCOND_EPOCHS)
    ambient_alpha = getattr(args, 'ambient_alpha', 0.7)
    ambient_delta = getattr(args, 'ambient_delta', 0.3)

    print(f"\n{'#'*60}")
    print(f"  Ambient-Aware Unconditional Training")
    print(f"  epochs={uncond_epochs}  alpha={ambient_alpha}  delta={ambient_delta}")
    print(f"  {len(reconstructions)} clean + {len(corrupted_data)} corrupted samples")
    print(f"{'#'*60}")

    optimizer = torch.optim.AdamW(
        uncond_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    uncond_model.reset_ema()

    dataset = AmbientMixedDataset(
        reconstructions, corrupted_data, obs_masks,
        alpha=ambient_alpha,
    )
    loader = Data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=uncond_epochs, eta_min=args.learning_rate * 0.01)

    uncond_model.train()
    best_disc = float('inf')
    best_metrics = None

    for epoch in range(uncond_epochs):
        epoch_loss_clean, epoch_loss_ambient = 0.0, 0.0
        n_clean, n_ambient = 0, 0

        for x_batch, mask_batch, is_clean_batch in loader:
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            is_clean_batch = is_clean_batch.to(device)

            clean_idx = is_clean_batch > 0.5
            ambient_idx = ~clean_idx

            total_loss = torch.tensor(0.0, device=device)

            # --- Standard EDM loss on clean reconstructions ---
            if clean_idx.any():
                x_clean = x_batch[clean_idx]
                x_img_clean = uncond_model.ts_to_img(x_clean)
                loss_clean, _ = uncond_model.loss_fn_irregular(x_img_clean)
                total_loss = total_loss + loss_clean * clean_idx.float().sum() / len(x_batch)
                epoch_loss_clean += loss_clean.item()
                n_clean += 1

            # --- Ambient masked loss on original corrupted data ---
            if ambient_idx.any():
                x_corr = x_batch[ambient_idx]
                m_corr = mask_batch[ambient_idx]
                x_img_corr = uncond_model.ts_to_img(x_corr)
                m_exp = m_corr.unsqueeze(-1).expand_as(x_corr)
                mask_img = uncond_model.ts_to_img(m_exp)
                loss_amb, _ = uncond_model.loss_fn_ambient_masked(
                    x_img_corr, mask_img, delta=ambient_delta)
                total_loss = total_loss + loss_amb * ambient_idx.float().sum() / len(x_batch)
                epoch_loss_ambient += loss_amb.item()
                n_ambient += 1

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
                optimizer.step()
                uncond_model.on_train_batch_end()

            torch.cuda.empty_cache()

        scheduler.step()

        avg_clean = epoch_loss_clean / max(n_clean, 1)
        avg_amb = epoch_loss_ambient / max(n_ambient, 1)

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == uncond_epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  epoch {epoch+1:4d}/{uncond_epochs}  "
                  f"clean={avg_clean:.5f}  ambient={avg_amb:.5f}  lr={lr_now:.2e}")

        _log(logger, 'uncond/loss_clean', avg_clean, epoch)
        _log(logger, 'uncond/loss_ambient', avg_amb, epoch)

        # Evaluate periodically
        if ((epoch + 1) % args.logging_iter == 0
                or epoch == uncond_epochs - 1):
            metrics = evaluate_unconditional(
                args, uncond_model, test_loader, device, epoch, logger)
            disc = metrics.get('disc_mean', float('inf'))
            if disc < best_disc:
                best_disc = disc
                best_metrics = metrics
                print(f"  *** New best disc_mean={disc:.4f} at epoch {epoch+1} ***")

    return best_metrics


# ---------------------------------------------------------------------------
# Unconditional evaluation (identical to run_irregular.py)
# ---------------------------------------------------------------------------

def evaluate_unconditional(args, uncond_model, test_loader, device,
                           step_id, logger=None):
    """Evaluate unconditional model using the same protocol as run_irregular.py."""
    uncond_model.eval()
    gen_sig, real_sig = [], []

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution,
                 args.img_resolution),
            )
            for data in tqdm(test_loader, desc="Eval"):
                x_img = process.sampling(sampling_number=data[0].shape[0])
                x_ts  = uncond_model.img_to_ts(x_img)
                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig  = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)
    print(f"  Metrics (step {step_id}):")
    for k, v in scores.items():
        print(f"    {k}: {v:.4f}")
        _log(logger, f'test/{k}', v, step_id)

    mem_path = f"mem_hist_step{step_id}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig,
        device=device, plot_path=mem_path,
    )
    for k, v in mem_stats.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    mem/{k}: {val_str}")
        _log(logger, f'test/memorization/{k}', v, step_id)
    _upload_and_cleanup(logger, mem_path, step_id)

    uncond_model.train()
    return scores


# ---------------------------------------------------------------------------
# Quick unconditional eval per EM iteration (disc_mean progress tracking)
# ---------------------------------------------------------------------------

def quick_unconditional_eval(args, reconstructions, test_loader, em_iter,
                             device, logger=None):
    """Train a lightweight unconditional model on current EM reconstructions
    and evaluate disc_mean.  Gives progress signal during the EM loop without
    waiting for the full Phase 3 ambient training.

    Uses a fresh model + optimizer each call so metrics are comparable across
    EM iterations.
    """
    eval_epochs = getattr(args, 'uncond_eval_epochs', DEFAULT_UNCOND_EVAL_EPOCHS)

    print(f"\n--- Quick Uncond Eval (EM iter {em_iter}), "
          f"{eval_epochs} epochs ---")

    model = TS2img_Karras(args=args, device=device).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    recon_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(reconstructions, dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )

    patience = getattr(args, 'early_stop_patience', DEFAULT_EARLY_STOP_PATIENCE)
    min_delta = getattr(args, 'early_stop_min_delta', DEFAULT_EARLY_STOP_MIN_DELTA)
    best_loss = float('inf')
    no_improve_count = 0

    model.train()
    for epoch in range(eval_epochs):
        epoch_loss, nb = 0.0, 0
        for (x_clean,) in recon_loader:
            x_clean = x_clean.to(device)
            x_img = model.ts_to_img(x_clean)
            loss, _ = model.loss_fn_irregular(x_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.on_train_batch_end()

            epoch_loss += loss.item()
            nb += 1
            torch.cuda.empty_cache()

        avg = epoch_loss / max(nb, 1)
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == eval_epochs - 1:
            print(f"    uncond epoch {epoch+1}/{eval_epochs}  loss={avg:.5f}")
        _log(logger, 'em_eval/uncond_loss', avg,
             em_iter * eval_epochs + epoch)

        if avg < best_loss - min_delta:
            best_loss = avg
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience and epoch >= 20:
            print(f"    Early stopping uncond at epoch {epoch+1}/{eval_epochs}")
            break

    scores = evaluate_unconditional(
        args, model, test_loader, device,
        step_id=em_iter, logger=logger,
    )

    disc = scores.get('disc_mean', float('inf'))
    print(f"  EM iter {em_iter} -> disc_mean = {disc:.4f}")
    _log(logger, 'em_eval/disc_mean', disc, em_iter)

    del model, optimizer
    torch.cuda.empty_cache()
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    name = create_model_name_and_dir(args)
    name = f"ambient_em_{name}"

    logging.info(args)

    with (CompositeLogger([WandbLogger()]) if args.wandb
          else PrintLogger()) as logger:
        log_config_and_tags(args, logger, name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        args.device = device

        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset ready.')

        m_epochs = getattr(args, 'm_step_epochs', DEFAULT_M_STEP_EPOCHS)
        sample_steps = getattr(args, 'e_step_sample_steps',
                               DEFAULT_E_STEP_SAMPLE_STEPS)
        uncond_epochs = getattr(args, 'uncond_epochs', DEFAULT_UNCOND_EPOCHS)
        uncond_eval_epochs = getattr(args, 'uncond_eval_epochs',
                                     DEFAULT_UNCOND_EVAL_EPOCHS)
        ambient_alpha = getattr(args, 'ambient_alpha', 0.7)
        ambient_delta = getattr(args, 'ambient_delta', 0.3)

        print(f"\n{'#'*60}")
        print(f"  AmbientEM — DiffEM + Ambient-Omni Unconditional Training")
        print(f"  EM iters={args.em_iters}  M-epochs={m_epochs}  "
              f"E-steps={sample_steps}")
        print(f"  Uncond epochs={uncond_epochs}  eval_epochs={uncond_eval_epochs}  "
              f"alpha={ambient_alpha}  delta={ambient_delta}")
        print(f"  dataset={args.dataset}  missing={args.missing_rate}  "
              f"diffusion_steps(default)={args.diffusion_steps}")
        print(f"{'#'*60}\n")

        _log(logger, 'config/m_step_epochs', m_epochs, 0)
        _log(logger, 'config/e_step_sample_steps', sample_steps, 0)
        _log(logger, 'config/em_iters', args.em_iters, 0)
        _log(logger, 'config/missing_rate', args.missing_rate, 0)
        _log(logger, 'config/uncond_epochs', uncond_epochs, 0)
        _log(logger, 'config/uncond_eval_epochs', uncond_eval_epochs, 0)
        _log(logger, 'config/ambient_alpha', ambient_alpha, 0)
        _log(logger, 'config/ambient_delta', ambient_delta, 0)

        # ---- Conditional model ----
        cond_model = TS2img_Karras_Cond(
            args=args, device=device).to(device)
        print_model_params(logger, cond_model)

        # ---- Extract corrupted observations & masks ----
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader)
        pct = obs_masks.sum() / obs_masks.size * 100
        print(f"Extracted {len(corrupted_data)} sequences, "
              f"{pct:.1f}% observed")

        # ================================================================
        # Phase 1 — PPCA initialisation
        # ================================================================
        print(f"\n{'='*60}\nPhase 1: PPCA Initialisation\n{'='*60}")
        ppca_rank = min(
            getattr(args, 'ppca_rank', 32),
            corrupted_data.shape[1] * corrupted_data.shape[2] - 1)
        ppca_iters = getattr(args, 'ppca_iters', 8)
        initial_recon = initialize_with_ppca(
            corrupted_data, obs_masks,
            rank=ppca_rank, ppca_iters=ppca_iters, seed=args.seed,
        )
        _log_stats('em/init_recon', initial_recon, -1, logger)

        # ---- Quick eval on PPCA-init data (baseline disc_mean) ----
        quick_unconditional_eval(
            args, initial_recon, test_loader, em_iter=-1,
            device=device, logger=logger,
        )

        # ================================================================
        # Phase 1.5 — Initial M-step on PPCA-filled data
        # ================================================================
        print(f"\n{'='*60}\n"
              f"Phase 1.5: Initial conditional model training\n{'='*60}")
        cond_model = m_step(
            args, cond_model, initial_recon,
            em_iter=-1, device=device, logger=logger,
        )

        # ================================================================
        # Phase 2 — EM loop
        # ================================================================
        print(f"\n{'='*60}\n"
              f"AmbientEM — {args.em_iters} EM iterations\n{'='*60}")

        reconstructions = initial_recon

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}\n"
                  f"EM Iteration {em_iter + 1}/{args.em_iters}\n{'='*60}")

            # ---- E-step ----
            reconstructions = e_step(
                args, cond_model, corrupted_data, obs_masks,
                em_iter, device,
                sample_steps=sample_steps, logger=logger,
            )

            # ---- M-step ----
            cond_model = m_step(
                args, cond_model, reconstructions,
                em_iter, device, logger=logger,
            )

            _log(logger, 'em/iteration', em_iter, em_iter)

            # ---- Quick unconditional eval (disc_mean progress) ----
            quick_unconditional_eval(
                args, reconstructions, test_loader, em_iter,
                device, logger=logger,
            )

        # ================================================================
        # Phase 3 — Ambient-aware unconditional training
        # ================================================================
        print(f"\n{'='*60}\n"
              f"Phase 3: Ambient-Aware Unconditional Training\n{'='*60}")

        uncond_model = TS2img_Karras(
            args=args, device=device).to(device)
        print("Unconditional model created.")
        print_model_params(logger, uncond_model)

        # Zero-fill corrupted data for the ambient dataset
        corrupted_for_ambient = corrupted_data.copy()
        corrupted_for_ambient = np.nan_to_num(corrupted_for_ambient, nan=0.0)

        best_metrics = train_unconditional_ambient(
            args, uncond_model, reconstructions,
            corrupted_for_ambient, obs_masks,
            test_loader, device, logger,
        )

        # ================================================================
        # Done
        # ================================================================
        print(f"\n{'='*60}\n"
              f"AmbientEM — Complete\n{'='*60}")
        if best_metrics:
            print(f"Best metrics:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True
    args.ambient_em = True

    if not any('m_step_epochs' in a for a in sys.argv):
        args.m_step_epochs = DEFAULT_M_STEP_EPOCHS
        print(f"[AmbientEM] m_step_epochs = {DEFAULT_M_STEP_EPOCHS}")

    if not any('e_step_sample_steps' in a for a in sys.argv):
        args.e_step_sample_steps = DEFAULT_E_STEP_SAMPLE_STEPS
        print(f"[AmbientEM] e_step_sample_steps = {DEFAULT_E_STEP_SAMPLE_STEPS}")

    if not any('uncond_epochs' in a for a in sys.argv):
        args.uncond_epochs = DEFAULT_UNCOND_EPOCHS
        print(f"[AmbientEM] uncond_epochs = {DEFAULT_UNCOND_EPOCHS}")

    if not any('uncond_eval_epochs' in a for a in sys.argv):
        args.uncond_eval_epochs = DEFAULT_UNCOND_EVAL_EPOCHS
        print(f"[AmbientEM] uncond_eval_epochs = {DEFAULT_UNCOND_EVAL_EPOCHS}")

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    main(args)
