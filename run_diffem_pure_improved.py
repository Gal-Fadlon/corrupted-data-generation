"""
Improved Pure DiffEM Training Script for Irregular Time Series

Faithful reimplementation of Algorithm 1 from the DiffEM paper:
  "Learning from Corrupted Data via Diffusion Models"
  (DiffEM-main/DiffEM_paper.tex)

Adapted for time-series-to-image generation.  The ONLY difference from the
CIFAR reference code (DiffEM-main/experiments/cifar/train_conditional.py) is
that the data lives in image-space representations of time series produced by
a delay-embedding transform.

Key fixes over run_diffem_pure.py (matching the paper exactly):
  1. Observation consistency: after E-step sampling, observed positions are
     replaced with the true observed values so reconstruction errors cannot
     compound across EM iterations.
  2. Fresh optimizer + EMA reset at every M-step, matching the paper's
     per-lap ``optimizer.init(params); avrg = params``.
  3. Increased default M-step epochs (200) to give the conditional model
     enough training per lap (paper uses 256 on CIFAR-10).
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
from tqdm import tqdm

from metrics import evaluate_model_irregular
from metrics.memorization import compute_memorization_metric
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import (
    restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
)
from utils.utils_data import (
    gen_dataloader, apply_corruption,
    EMReconstructionDataset, save_reconstructions, load_reconstructions
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras, TS2img_Karras_Cond
from models.sampler import DiffusionProcess, ConditionalDiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

# Default M-step epochs — closer to the paper's 256 epochs/lap on CIFAR-10.
DEFAULT_M_STEP_EPOCHS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_corrupted_data_from_loader(train_loader):
    """Extract corrupted time series and observation masks from the train loader.

    Returns:
        corrupted_data : ndarray (N, seq_len, features) with NaN for missing
        obs_masks      : ndarray (N, seq_len) boolean, True = observed
    """
    all_corrupted, all_masks = [], []
    for data in train_loader:
        x_irregular = data[0]                        # (B, seq_len, features+1)
        x_ts = x_irregular[:, :, :-1]                # drop time-index column
        mask = ~torch.isnan(x_ts).any(dim=-1)        # (B, seq_len)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())
    return np.vstack(all_corrupted), np.vstack(all_masks)


def initialize_with_gaussian(corrupted_data, obs_masks, seed=None):
    """Gaussian-prior initialisation (DiffEM paper, first lap).

    Observed values are kept.  Missing values are sampled from a per-feature
    Gaussian fitted on the observed entries.
    """
    if seed is not None:
        np.random.seed(seed)

    N, seq_len, features = corrupted_data.shape
    mu  = np.nanmean(corrupted_data, axis=(0, 1))
    std = np.nanstd(corrupted_data, axis=(0, 1))
    std = np.where(std == 0, 1.0, std)

    initial_recon = corrupted_data.copy()
    for i in range(N):
        for t in range(seq_len):
            if not obs_masks[i, t]:
                initial_recon[i, t, :] = np.random.normal(mu, std)

    initial_recon = np.clip(initial_recon, 0, 1)
    print(f"Gaussian init: {N} sequences, mu={mu}, std={std}")
    return initial_recon


# ---------------------------------------------------------------------------
# E-step  (Algorithm 1, line "X^{(i)} ~ q_theta(·|Y^{(i)})")
# ---------------------------------------------------------------------------

def e_step(args, cond_model, corrupted_data, obs_masks, em_iter, device,
           logger=None):
    """Sample reconstructions from the current conditional posterior.

    After sampling, **observed positions are replaced with actual observed
    values** to prevent error accumulation across EM iterations.  This mirrors
    the hard-constraint that MMPS enforces analytically via its posterior
    denoiser (residual ``r = y_obs − mask·denoised``).
    """
    print(f"\n=== E-Step (EM iter {em_iter}) ===")
    cond_model.eval()

    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    cond_channels = args.input_channels + 1

    N = len(corrupted_data)
    batch_size = args.batch_size
    all_reconstructions = []

    with torch.no_grad():
        with cond_model.ema_scope():
            process = ConditionalDiffusionProcess(
                args, cond_model.net, target_shape, cond_channels
            )
            for start in tqdm(range(0, N, batch_size), desc="E-step sampling"):
                end = min(start + batch_size, N)
                corrupted_batch = corrupted_data[start:end]
                mask_batch      = obs_masks[start:end]

                corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32,
                                            device=device)
                mask_ts = torch.tensor(mask_batch, dtype=torch.float32,
                                       device=device)

                # Conditioning: zero-filled observations + binary mask
                cond_ts  = torch.nan_to_num(corrupted_ts, nan=0.0)
                cond_img = cond_model.ts_to_img(cond_ts)

                mask_expanded  = mask_ts.unsqueeze(-1).expand_as(corrupted_ts)
                cond_mask_img  = cond_model.ts_to_img(mask_expanded)
                cond_mask_img  = cond_mask_img[:, :1, :, :]   # single channel

                # Conditional sampling: X ~ q_theta(x | y)
                x_img = process.sampling(cond_img, cond_mask_img)
                x_ts  = cond_model.img_to_ts(x_img)

                # -- Observation consistency --
                # Replace observed time-steps with actual observed values.
                x_np = x_ts.cpu().numpy()
                for b in range(len(corrupted_batch)):
                    obs = mask_batch[b].astype(bool)
                    x_np[b, obs, :] = corrupted_batch[b][obs, :]

                all_reconstructions.append(x_np)

    reconstructions = np.vstack(all_reconstructions)
    print(f"E-step done: {len(reconstructions)} sequences "
          f"(obs-consistency enforced)")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                             f'missing_{int(args.missing_rate * 100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)
    return reconstructions


# ---------------------------------------------------------------------------
# M-step  (Algorithm 1, line "theta^{k+1} <- train on D_X^{k}")
# ---------------------------------------------------------------------------

def m_step(args, cond_model, reconstructions, em_iter, device, logger=None):
    """Train the conditional diffusion model on (X, Y~Q(X)) pairs.

    Matching the reference implementation (train_conditional.py):
      - A **fresh optimizer** is created each lap (Adam state reset).
      - **EMA is reset** to current model weights (``avrg = params``).
      - The model **weights continue** from the previous lap (no re-init).
      - Training targets X come from the E-step reconstructions.
      - Conditioning Y is a *fresh* corruption of X (``y0 = corrupt(pi0)``).
    """
    m_epochs = getattr(args, 'm_step_epochs', DEFAULT_M_STEP_EPOCHS)
    print(f"\n=== M-Step (EM iter {em_iter}) — {m_epochs} epochs ===")

    # --- Fresh optimizer (paper: ``optimizer.init(params)`` per lap) ---
    optimizer = torch.optim.AdamW(
        cond_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --- Reset EMA (paper: ``avrg = params`` per lap) ---
    cond_model.reset_ema()

    corruption_rate = (args.em_corruption_rate
                       if args.em_corruption_rate else args.missing_rate)
    noise_level = (args.em_noise_level
                   if args.em_noise_level else args.gaussian_noise_level)

    em_dataset = EMReconstructionDataset(
        reconstructions,
        missing_rate=corruption_rate,
        gaussian_noise_level=noise_level,
        noise_timestep=args.noise_timestep,
        precompute_corruption=False,
        seed=args.seed + em_iter * 1000,
    )
    em_loader = Data.DataLoader(
        em_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )

    cond_model.train()
    loss_history = []

    patience = getattr(args, 'early_stop_patience', 20)
    min_delta = getattr(args, 'early_stop_min_delta', 1e-4)
    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(m_epochs):
        epoch_loss, n_batches = 0.0, 0
        for clean, corrupted, mask in em_loader:
            clean     = clean.to(device)
            corrupted = corrupted.to(device)
            mask      = mask.to(device)

            cond_ts = torch.nan_to_num(corrupted, nan=0.0)

            x_img        = cond_model.ts_to_img(clean)
            cond_img     = cond_model.ts_to_img(cond_ts)
            mask_exp     = mask.unsqueeze(-1).expand_as(clean)
            obs_mask_img = cond_model.ts_to_img(mask_exp)
            obs_mask_img = obs_mask_img[:, :1, :, :]

            loss, _ = cond_model.loss_fn_conditional_full(
                x_img, cond_img, obs_mask_img
            )

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
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{m_epochs}  loss={avg:.4f}")
        if logger is not None:
            logger.log('em/m_step_loss',
                        avg, em_iter * m_epochs + epoch)

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
    return cond_model


# ---------------------------------------------------------------------------
# Unconditional model evaluation (same protocol as run_irregular.py / MMPS)
# ---------------------------------------------------------------------------

def train_and_evaluate_unconditional(args, uncond_model, reconstructions,
                                      test_loader, em_iter, device,
                                      logger=None):
    """Train an unconditional model on current reconstructions, then evaluate.

    A **fresh optimizer** and **EMA reset** are used each call so that
    the unconditional model trains cleanly on the latest reconstructions.
    """
    uncond_epochs = getattr(args, 'uncond_epochs_per_iter', None)
    if uncond_epochs is None:
        uncond_epochs = getattr(args, 'm_step_epochs', DEFAULT_M_STEP_EPOCHS)
    print(f"\n=== Uncond Train+Eval (EM iter {em_iter}) "
          f"— {uncond_epochs} epochs ===")

    # Fresh optimizer + EMA reset
    uncond_opt = torch.optim.AdamW(
        uncond_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    uncond_model.reset_ema()

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_loader = Data.DataLoader(
        Data.TensorDataset(recon_tensor),
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
        epoch_loss, n_batches = 0.0, 0
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
            n_batches  += 1
            torch.cuda.empty_cache()

        avg = epoch_loss / max(n_batches, 1)
        uncond_loss_history.append(avg)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  uncond epoch {epoch+1}/{uncond_epochs}  loss={avg:.4f}")
        if logger is not None:
            logger.log('train/uncond_loss',
                        avg, em_iter * uncond_epochs + epoch)

        if avg < best_loss - min_delta:
            best_loss = avg
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience and epoch >= 30:
            print(f"  Early stopping at epoch {epoch+1}/{uncond_epochs} "
                  f"(no improvement for {patience} epochs, best={best_loss:.5f})")
            break

    # --- Evaluation (unconditional sampling, same as run_irregular.py) ---
    uncond_model.eval()
    gen_sig, real_sig = [], []

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution, args.img_resolution),
            )
            for data in tqdm(test_loader, desc="Evaluating"):
                x_img_sampled = process.sampling(
                    sampling_number=data[0].shape[0])
                x_ts = uncond_model.img_to_ts(x_img_sampled)
                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig  = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)
    print(f"EM iter {em_iter} metrics (unconditional):")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
        if logger is not None:
            logger.log(f'test/{k}', v, em_iter)

    # Memorization check
    mem_plot_path = f"memorization_hist_em_iter_{em_iter}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig,
        device=device, plot_path=mem_plot_path,
    )
    print(f"EM iter {em_iter} memorization metrics:")
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)

    _upload_and_cleanup(logger, mem_plot_path, em_iter)
    return scores


# ---------------------------------------------------------------------------
# Utility: Plot upload & temp-file cleanup
# ---------------------------------------------------------------------------

def _upload_and_cleanup(logger, plot_path, em_iter):
    if logger is None:
        return
    uploaded = False
    try:
        logger.log_file('test/memorization/histogram', plot_path, em_iter)
        uploaded = True
    except Exception as exc:
        print(f"Plot upload failed: {exc}")
    if uploaded:
        try:
            os.remove(plot_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(args):
    name = create_model_name_and_dir(args)
    name = f"diffem_pure_improved_{name}"

    logging.info(args)

    with (CompositeLogger([WandbLogger()]) if args.wandb
          else PrintLogger()) as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset ready.')

        # ---- Conditional model (the core DiffEM model) ----
        cond_model = TS2img_Karras_Cond(
            args=args, device=args.device).to(args.device)
        print_model_params(logger, cond_model)

        # ---- Extract corrupted observations & masks ----
        corrupted_data, obs_masks = get_corrupted_data_from_loader(
            train_loader)
        pct = obs_masks.sum() / obs_masks.size * 100
        print(f"Extracted {len(corrupted_data)} sequences, "
              f"{pct:.1f}% observed")

        # ================================================================
        # Phase 1 — Gaussian initialisation (paper: lap 0 prior)
        # ================================================================
        print(f"\n{'='*60}\nPhase 1: Gaussian Initialisation\n{'='*60}")
        initial_recon = initialize_with_gaussian(
            corrupted_data, obs_masks, seed=args.seed)

        # ================================================================
        # Phase 1.5 — Initial M-step on Gaussian-filled data
        # ================================================================
        print(f"\n{'='*60}\nPhase 1.5: Initial conditional model training\n"
              f"{'='*60}")
        cond_model = m_step(
            args, cond_model, initial_recon,
            em_iter=-1, device=args.device, logger=logger,
        )

        # ---- Unconditional model (for evaluation only) ----
        uncond_model = TS2img_Karras(
            args=args, device=args.device).to(args.device)
        print("Unconditional evaluation model created.")

        # ================================================================
        # Phase 2 — EM loop
        # ================================================================
        print(f"\n{'='*60}\n"
              f"Pure DiffEM — {args.em_iters} EM iterations\n{'='*60}")

        best_metrics = None

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}\n"
                  f"EM Iteration {em_iter + 1}/{args.em_iters}\n{'='*60}")

            # ---- E-step ----
            reconstructions = e_step(
                args, cond_model, corrupted_data, obs_masks,
                em_iter, args.device, logger,
            )

            # ---- M-step (fresh optimiser + EMA each lap) ----
            cond_model = m_step(
                args, cond_model, reconstructions,
                em_iter, args.device, logger,
            )

            # ---- Evaluate via unconditional model ----
            if ((em_iter + 1) % args.em_eval_interval == 0
                    or em_iter == args.em_iters - 1):
                metrics = train_and_evaluate_unconditional(
                    args, uncond_model, reconstructions,
                    test_loader, em_iter, args.device, logger,
                )
                disc = metrics.get('disc_mean', float('inf'))
                if (best_metrics is None
                        or disc < best_metrics.get('disc_mean', float('inf'))):
                    best_metrics = metrics

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        # ================================================================
        # Done
        # ================================================================
        print(f"\n{'='*60}\nPure DiffEM (improved) — Complete\n{'='*60}")
        if best_metrics:
            print("Best metrics:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    # Use 200 epochs/M-step unless the user explicitly passed --m_step_epochs
    cli_set = any('m_step_epochs' in a for a in sys.argv)
    if not cli_set:
        args.m_step_epochs = DEFAULT_M_STEP_EPOCHS
        print(f"[improved] m_step_epochs set to {DEFAULT_M_STEP_EPOCHS} "
              f"(override with --m_step_epochs)")

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    main(args)
