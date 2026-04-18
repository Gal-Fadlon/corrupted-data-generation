"""
Agentic EM pipeline v3 for irregular time series.

Key design principles:
  1. M-step uses uniform loss on K*N augmented samples
  2. Spectral diagnosis: one bin per rfft frequency, raw (un-normalised) CV,
     bidirectional f_cutoff adaptation with slow blending (0.2)
  3. Curriculum anneals over the full EM range (not cut off at 60%)
  4. Early stopping on disc_mean with configurable patience
  5. No Phase 3 — best EM checkpoint is the final output

Changes from v2:
  - Fixed spectral diagnosis: 32 bins → n_freqs bins (no empty bins),
    removed normalisation, fixed threshold comparison
  - f_cutoff adaptation is bidirectional and slower (blend 0.2 vs 0.5)
  - Added early stopping with patience
  - Added E-step quality metrics (observed residual MSE, spectral energy ratio)
  - Removed Phase 3 (fresh unconditional model that destroyed EM gains)

With estep_n_samples=1 the system degrades to the fixed spectral-CG baseline.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
import numpy as np
import torch
import torch.multiprocessing
import torch.utils.data as Data
from tqdm import tqdm

from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader, save_reconstructions
from models.our import TS2img_Karras
from utils.utils_stl import initialize_with_iterative_stl
from utils.diffem_mmps_irregular_em import (
    IrregularMMPSExperimentSpec,
    get_corrupted_data_from_loader,
    evaluate_uncond,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgenticConfig:
    n_samples: int = 3
    spectral_diag_threshold: float = 0.3
    adaptive_f_cutoff_init: float = 0.5
    f_cutoff_min: float = 0.1
    f_cutoff_max: float = 1.0
    f_cutoff_blend: float = 0.2
    em_patience: int = 2

    @classmethod
    def from_args(cls, args) -> "AgenticConfig":
        return cls(
            n_samples=getattr(args, "estep_n_samples", 3),
            spectral_diag_threshold=getattr(args, "spectral_diag_threshold", 0.3),
            adaptive_f_cutoff_init=getattr(args, "adaptive_f_cutoff_init", 0.5),
            em_patience=getattr(args, "em_patience", 2),
        )


# ---------------------------------------------------------------------------
# E-step: K-sample posterior sampling → data augmentation
# ---------------------------------------------------------------------------

def e_step_augmented(
    args,
    uncond_model,
    corrupted_data,
    obs_masks,
    em_iter,
    device,
    logger,
    build_process_fn,
    e_step_title: str,
    tqdm_desc: str,
    ac: AgenticConfig,
):
    """Draw K posterior samples and return them ALL as augmented training data.

    Returns:
        augmented_data: (K*N, seq_len, n_features) — all K samples concatenated
        mean_recon:     (N, seq_len, n_features)   — mean for curriculum & diagnostics
        all_k_samples:  list of K arrays           — for spectral diagnosis
    """
    K = ac.n_samples
    print(f"\n=== E-Step (EM iter {em_iter}) — {K}-sample {e_step_title} ===")
    uncond_model.eval()

    n_seq = len(corrupted_data)
    batch_size = args.batch_size
    all_k_samples = []

    for k in range(K):
        print(f"  Drawing sample {k + 1}/{K}...")
        sample_parts = []

        with uncond_model.ema_scope():
            process = build_process_fn(args, uncond_model)

            for start_idx in tqdm(range(0, n_seq, batch_size),
                                  desc=f"{tqdm_desc} sample {k + 1}/{K}"):
                end_idx = min(start_idx + batch_size, n_seq)
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
                sample_parts.append(x_ts_recon.cpu().numpy())
                torch.cuda.empty_cache()

        all_k_samples.append(np.vstack(sample_parts).astype(np.float32))

    mean_recon = np.mean(all_k_samples, axis=0)
    augmented_data = np.concatenate(all_k_samples, axis=0)

    inter_sample_std = float(np.std(all_k_samples, axis=0).mean()) if K > 1 else 0.0
    nan_count = int(np.isnan(mean_recon).sum())

    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")
    print(f"E-step complete. {K} samples. Inter-sample std: {inter_sample_std:.4f}")
    print(f"  Augmented M-step data: {augmented_data.shape[0]} samples ({K}x dataset)")

    cache_dir = os.path.join(
        args.recon_cache_dir, args.dataset, f"missing_{int(args.missing_rate * 100)}"
    )
    save_reconstructions(mean_recon, cache_dir, em_iter)

    if logger is not None:
        logger.log("em/e_step_samples", len(mean_recon), em_iter)
        logger.log("em/inter_sample_std", inter_sample_std, em_iter)
        logger.log("em/nan_count", nan_count, em_iter)
        logger.log("em/recon_range_min", float(np.nanmin(mean_recon)), em_iter)
        logger.log("em/recon_range_max", float(np.nanmax(mean_recon)), em_iter)

    return augmented_data, mean_recon, all_k_samples


# ---------------------------------------------------------------------------
# OBSERVE: spectral diagnosis via inter-sample Fourier variance
# ---------------------------------------------------------------------------

def spectral_diagnosis_intersample(all_k_samples, n_bins=None):
    """Per-frequency coefficient of variation across K posterior samples.

    Uses one bin per rfft frequency (n_bins = seq_len//2 + 1) so there are no
    empty bins.  Returns **raw** (un-normalised) CV — the caller decides
    the threshold.

    Returns:
        per_freq_cv: array of shape (n_freqs,) with raw CV per frequency.
                     With K<2 returns zeros.
    """
    K = len(all_k_samples)
    seq_len = all_k_samples[0].shape[1]
    n_freqs = seq_len // 2 + 1

    if K < 2:
        return np.zeros(n_freqs, dtype=np.float32)

    power_spectra = []
    for s in all_k_samples:
        F = np.fft.rfft(s, axis=1)
        P = np.abs(F) ** 2
        if P.ndim == 3:
            P = P.mean(axis=(0, 2))
        else:
            P = P.mean(axis=0)
        power_spectra.append(P)

    P_stack = np.stack(power_spectra, axis=0)   # (K, n_freqs)
    P_mean = P_stack.mean(axis=0)
    P_std = P_stack.std(axis=0)
    cv = P_std / (P_mean + 1e-30)

    return cv.astype(np.float32)


# ---------------------------------------------------------------------------
# PLAN: adapt spectral filter cutoff from diagnosis
# ---------------------------------------------------------------------------

def adapt_spectral_filter(per_freq_cv, ac: AgenticConfig, current_cutoff: float,
                          seq_len: int):
    """Adapt f_cutoff bidirectionally based on per-frequency raw CV.

    Maps the highest reliable frequency index to a normalised cutoff in
    [f_cutoff_min, f_cutoff_max] and blends toward it.  Because the target
    can be lower than current_cutoff, this naturally allows the cutoff to
    *decrease* when high-frequency bands become unreliable.
    """
    n_freqs = len(per_freq_cv)
    freqs = np.fft.rfftfreq(seq_len)[:n_freqs]

    reliable = per_freq_cv < ac.spectral_diag_threshold

    if reliable.any():
        highest_reliable_idx = int(np.where(reliable)[0].max())
        target = float(freqs[highest_reliable_idx] / 0.5)
    else:
        target = ac.f_cutoff_min

    target = float(np.clip(target, ac.f_cutoff_min, ac.f_cutoff_max))
    blended = ac.f_cutoff_blend * target + (1 - ac.f_cutoff_blend) * current_cutoff
    return float(np.clip(blended, ac.f_cutoff_min, ac.f_cutoff_max))


# ---------------------------------------------------------------------------
# M-step: standard uniform loss on augmented data
# ---------------------------------------------------------------------------

def m_step_augmented(
    args, uncond_model, optimizer, augmented_data, em_iter, device, logger=None,
):
    """Baseline-style M-step on all K*N augmented samples (uniform loss)."""
    print(f"\n=== M-Step (EM iter {em_iter}) — {len(augmented_data)} augmented samples ===")
    print(f"Training for {args.m_step_epochs} epochs...")

    recon_tensor = torch.tensor(augmented_data, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )

    uncond_model.train()
    for epoch in range(args.m_step_epochs):
        epoch_loss, num_batches = 0, 0
        for (x_clean,) in recon_loader:
            x_clean = x_clean.to(device)
            x_img = uncond_model.ts_to_img(x_clean)
            loss, _ = uncond_model.loss_fn_irregular(x_img)
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
            print(f"  M-step epoch {epoch + 1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")
        if logger is not None:
            logger.log("em/m_step_loss", avg_loss, em_iter * args.m_step_epochs + epoch)

    if logger is not None:
        logger.log("em/m_step_final_loss", avg_loss, em_iter)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


# ---------------------------------------------------------------------------
# Plain M-step (for Phase 1 warm-start — identical to baseline)
# ---------------------------------------------------------------------------

def m_step_plain(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training unconditional model for {args.m_step_epochs} epochs...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )

    uncond_model.train()
    for epoch in range(args.m_step_epochs):
        epoch_loss, num_batches = 0, 0
        for (x_clean,) in recon_loader:
            x_clean = x_clean.to(device)
            x_img = uncond_model.ts_to_img(x_clean)
            loss, _ = uncond_model.loss_fn_irregular(x_img)
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
            print(f"  M-step epoch {epoch + 1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")
        if logger is not None:
            logger.log("em/m_step_loss", avg_loss, em_iter * args.m_step_epochs + epoch)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


# ---------------------------------------------------------------------------
# Main agentic EM loop (v3)
# ---------------------------------------------------------------------------

def agentic_em_main(args, spec: IrregularMMPSExperimentSpec) -> None:
    ac = AgenticConfig.from_args(args)

    name = create_model_name_and_dir(args)
    name = f"{spec.wandb_name_prefix}_{name}"
    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)
        logger.log_name_params("agentic_config", {
            "n_samples": ac.n_samples,
            "spectral_diag_threshold": ac.spectral_diag_threshold,
            "adaptive_f_cutoff_init": ac.adaptive_f_cutoff_init,
            "f_cutoff_min": ac.f_cutoff_min,
            "f_cutoff_max": ac.f_cutoff_max,
            "f_cutoff_blend": ac.f_cutoff_blend,
            "em_patience": ac.em_patience,
            "version": "v3_fixed_spectral",
        })

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        print_model_params(logger, uncond_model)

        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(
            f"Extracted {len(corrupted_data)} sequences with "
            f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed"
        )

        # ── Phase 1: STL warm-start ──────────────────────────────────
        print(f"\n{'=' * 60}")
        print("Phase 1: Iterative STL Warm-Start Initialization")
        print(f"{'=' * 60}")

        initial_reconstructions = initialize_with_iterative_stl(
            corrupted_data, obs_masks, seed=args.seed,
        )
        uncond_model = m_step_plain(
            args, uncond_model, optimizer, initial_reconstructions,
            em_iter=-1, device=args.device, logger=logger,
        )

        # ── Phase 2: Agentic EM ──────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(spec.phase2_start_line.format(em_iters=args.em_iters))
        print(f"  K={ac.n_samples} samples as data augmentation, "
              f"uniform loss, adaptive f_cutoff")
        if spec.phase2_extra_printer is not None:
            spec.phase2_extra_printer(args)
        print(f"{'=' * 60}")

        best_metrics = None
        best_disc_mean = float("inf")
        best_recon = initial_reconstructions.copy()
        best_model_state = None
        last_recon = initial_reconstructions
        f_cutoff = ac.adaptive_f_cutoff_init
        prev_disc_mean = None
        patience_counter = 0

        for em_iter in range(args.em_iters):
            print(f"\n{'=' * 60}")
            print(f"Agentic EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'=' * 60}")

            # ── Curriculum: anneal over full EM range ──────────────────
            progress = em_iter / max(args.em_iters - 1, 1)
            reveal_prob = (1.0 - progress) * 0.3
            extra_reveal = (
                (np.random.rand(*obs_masks.shape) < reveal_prob)
                & ~obs_masks.astype(bool)
            )
            curriculum_masks = obs_masks.astype(bool) | extra_reveal
            curriculum_corrupted = corrupted_data.copy()
            curriculum_corrupted[extra_reveal] = last_recon[extra_reveal]
            n_revealed = int(extra_reveal.sum())
            if n_revealed > 0:
                print(f"  Curriculum: revealed {n_revealed} extra positions "
                      f"(progress={progress:.2f})")

            # ── Set f_cutoff for this E-step ──────────────────────────
            args.adaptive_f_cutoff = f_cutoff
            print(f"  Plan: f_cutoff_base = {f_cutoff:.3f}")

            if logger is not None:
                logger.log("em/curriculum_n_revealed", n_revealed, em_iter)
                logger.log("em/curriculum_progress", progress, em_iter)
                logger.log("em/curriculum_reveal_prob", reveal_prob, em_iter)
                logger.log("em/f_cutoff_used", f_cutoff, em_iter)

            # ── E-step: K samples → augmented data ────────────────────
            augmented_data, mean_recon, all_k_samples = e_step_augmented(
                args, uncond_model,
                curriculum_corrupted, curriculum_masks.astype(np.float32),
                em_iter, args.device, logger,
                spec.build_process_fn, spec.e_step_title, spec.tqdm_desc,
                ac,
            )
            last_recon = mean_recon

            # ── E-step quality: observed-position residual ─────────────
            obs_bool = obs_masks.astype(bool)[..., np.newaxis]  # (N, seq_len, 1)
            obs_vals = np.nan_to_num(corrupted_data, nan=0.0)
            residual = (mean_recon - obs_vals) * obs_bool
            n_obs_elements = obs_bool.sum() * mean_recon.shape[-1]
            obs_residual_mse = float((residual ** 2).sum() / max(n_obs_elements, 1))

            recon_fft = np.fft.rfft(mean_recon, axis=1)
            recon_power = np.abs(recon_fft) ** 2
            if recon_power.ndim == 3:
                recon_power = recon_power.mean(axis=(0, 2))
            else:
                recon_power = recon_power.mean(axis=0)
            n_f = len(recon_power)
            low_energy = float(recon_power[:n_f // 2].sum())
            high_energy = float(recon_power[n_f // 2:].sum())
            spectral_energy_ratio = high_energy / (low_energy + 1e-30)

            print(f"  E-step quality: obs_residual_mse={obs_residual_mse:.6f}, "
                  f"spectral_hi/lo={spectral_energy_ratio:.4f}")

            if logger is not None:
                logger.log("em/observed_residual_mse", obs_residual_mse, em_iter)
                logger.log("em/spectral_energy_ratio", spectral_energy_ratio, em_iter)

            # ── Spectral diagnosis + adapt f_cutoff ───────────────────
            if ac.n_samples > 1:
                per_freq_cv = spectral_diagnosis_intersample(all_k_samples)
                n_freqs = len(per_freq_cv)
                n_reliable = int((per_freq_cv < ac.spectral_diag_threshold).sum())
                n_unreliable = n_freqs - n_reliable
                old_cutoff = f_cutoff
                f_cutoff = adapt_spectral_filter(
                    per_freq_cv, ac, f_cutoff,
                    seq_len=all_k_samples[0].shape[1],
                )
                delta = f_cutoff - old_cutoff
                print(f"  Spectral diagnosis (inter-sample CV): "
                      f"reliable {n_reliable}/{n_freqs} freqs, "
                      f"unreliable {n_unreliable}/{n_freqs}")
                print(f"  Adapted f_cutoff: {old_cutoff:.3f} → {f_cutoff:.3f} "
                      f"(Δ={delta:+.3f})")

                if logger is not None:
                    logger.log("em/f_cutoff_base", f_cutoff, em_iter)
                    logger.log("em/f_cutoff_delta", delta, em_iter)
                    logger.log("em/spectral_n_reliable_bands", n_reliable, em_iter)
                    logger.log("em/spectral_n_unreliable_bands", n_unreliable, em_iter)
                    logger.log("em/spectral_raw_cv_mean", float(per_freq_cv.mean()), em_iter)
                    logger.log("em/spectral_n_freqs", n_freqs, em_iter)
                    for f_idx in range(n_freqs):
                        logger.log(f"em/spectral_freq_cv_{f_idx}", float(per_freq_cv[f_idx]), em_iter)

            del all_k_samples

            # ── EMA reset + M-step (uniform loss) ─────────────────────
            uncond_model.reset_ema()
            uncond_model = m_step_augmented(
                args, uncond_model, optimizer,
                augmented_data, em_iter, args.device, logger,
            )
            del augmented_data

            # ── EVALUATE ──────────────────────────────────────────────
            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, uncond_model, test_loader, em_iter, args.device, logger,
                )

                current_disc = metrics.get("disc_mean", float("inf"))

                disc_delta = current_disc - prev_disc_mean if prev_disc_mean is not None else 0.0
                prev_disc_mean = current_disc

                if logger is not None:
                    logger.log("em/disc_mean_delta", disc_delta, em_iter)
                    logger.log("em/best_disc_mean", min(best_disc_mean, current_disc), em_iter)

                if current_disc < best_disc_mean:
                    best_disc_mean = current_disc
                    best_metrics = metrics
                    best_recon = mean_recon.copy()
                    best_model_state = {
                        k: v.cpu().clone() for k, v in uncond_model.state_dict().items()
                    }
                    patience_counter = 0
                    print(f"  *** New best disc_mean={current_disc:.4f} at EM iter {em_iter} ***")
                else:
                    patience_counter += 1
                    print(f"  disc_mean={current_disc:.4f} (best={best_disc_mean:.4f}, "
                          f"patience {patience_counter}/{ac.em_patience})")

            if logger is not None:
                logger.log("em/iteration", em_iter, em_iter)
                logger.log("em/patience_counter", patience_counter, em_iter)

            # ── Early stopping check ──────────────────────────────────
            if patience_counter >= ac.em_patience:
                print(f"\n  Early stopping triggered at EM iter {em_iter} "
                      f"(no improvement for {ac.em_patience} iterations)")
                print(f"  Best disc_mean={best_disc_mean:.4f}")
                break

        if best_model_state is not None:
            uncond_model.load_state_dict(best_model_state)
            print("Restored model weights from best EM iteration.")

        print("\n" + "=" * 60)
        print(spec.completion_message)
        print("=" * 60)
        if best_metrics:
            print("Best EM metrics:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")
