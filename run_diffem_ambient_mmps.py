"""
DiffEM with A-Aware Ambient M-step + MMPS E-step

Builds on run_diffem_mmps_general_corruption.py (imported, not modified) by
making the M-step corruption-aware:

  1. Phase 0 (Ambient Pre-training): train the unconditional model directly on
     corrupted observations using an observation-space loss that flows gradients
     through the forward operator A.  For missing data this is the Ambient
     Diffusion further-corruption trick; for blur it is ||blur(D_θ) - y||².

  2. Modified M-step: weighted combination of
       λ_obs  * ||A(D_θ(x_t)) - y_obs||²      (observation-space, reliable)
       (1-λ_obs) * ||D_θ(x_t) - x_recon||²    (full-space, detail recovery)
     with λ_obs annealing from lambda_obs_start (1.0) to lambda_obs_end (0.3)
     over EM iterations.

  3. E-step: unchanged MMPS posterior sampling with general ForwardOperator.

Key ideas:
  - Ambient Diffusion (Daras et al. 2023): further corruption prevents
    memorisation for missing data.
  - Ambient Diffusion Omni (Daras et al. 2025): obs-space loss is valid for
    any linear operator A.
  - DiffEM (Rozet et al.): EM refinement with MMPS posterior.
"""

import torch
import torch.utils.data as Data
import os
import sys
import math
import numpy as np
import logging
from tqdm import tqdm

from run_diffem_mmps_general_corruption import (
    ForwardOperator, MissingDataOperator, GaussianNoiseOperator,
    GaussianBlurOperator, RandomProjectionOperator, TSCompositeOperator,
    temporal_moving_average,
    PPCACov, ppca_fit_images, conjugate_gradient,
    MMPSDiffusionProcess, _run_mmps_batch,
    get_corrupted_data_from_loader, get_clean_data_from_loader,
    prepare_data_missing, prepare_data_gaussian_noise,
    prepare_data_gaussian_blur, prepare_data_random_projection,
    prepare_data_ts_gaussian_noise, prepare_data_ts_temporal_smoothing,
    prepare_data_ts_missing_noise,
    compute_difficulty_scores, initialize_reconstructions,
    e_step, evaluate_uncond,
)

from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Phase 0: Ambient Pre-training (no E-step, trains directly on observations)
# =============================================================================

def ambient_pretrain(args, uncond_model, optimizer, corrupted_obs, obs_masks,
                     operator_factory, corruption_type, device, logger=None):
    """Train the model directly on corrupted observations with obs-space loss.

    For **missing data** this applies the Ambient Diffusion further-corruption
    trick: the model sees a subset of observed positions and must predict all
    observed positions.  This is far superior to training on garbage PPCA
    imputations, especially at 90 % missing.

    For **blur / smoothing** the obs-space loss ``||A(D_θ) - y||²`` flows
    gradients through the operator, teaching the model to produce sharp data
    whose projection matches the actual blurred observation.

    For **noise (A = I)** the obs-space loss equals the standard loss, so we
    fall back to ``loss_fn_irregular`` (the obs target *is* the noisy signal).
    """
    pretrain_epochs = getattr(args, 'ambient_pretrain_epochs', 100)
    delta = getattr(args, 'further_corrupt_delta', 0.1)

    print(f"\n{'=' * 60}")
    print(f"Phase 0: Ambient Pre-training ({pretrain_epochs} epochs)")
    print(f"  Corruption type: {corruption_type}, delta={delta}")
    print(f"{'=' * 60}")

    if corruption_type in ('missing', 'ts_missing_noise'):
        obs_zf = torch.tensor(np.nan_to_num(corrupted_obs, nan=0.0),
                              dtype=torch.float32)
        mask_t = torch.tensor(obs_masks, dtype=torch.float32)
        dataset = Data.TensorDataset(obs_zf, mask_t)
    elif corruption_type in ('gaussian_noise', 'gaussian_blur',
                             'ts_gaussian_noise'):
        dataset = Data.TensorDataset(
            torch.tensor(corrupted_obs, dtype=torch.float32))
    elif corruption_type == 'ts_temporal_smoothing':
        dataset = Data.TensorDataset(
            torch.tensor(corrupted_obs, dtype=torch.float32))
    else:
        print("  Ambient pre-training not applicable; skipping Phase 0.")
        return uncond_model

    loader = Data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    uncond_model.train()

    for epoch in range(pretrain_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if corruption_type in ('missing', 'ts_missing_noise'):
                obs_b, mask_b = batch[0].to(device), batch[1].to(device)
                x_img = uncond_model.ts_to_img(obs_b)
                y_obs = x_img.clone()
                mask_exp = mask_b.unsqueeze(-1).expand(-1, -1, obs_b.shape[-1])
                mask_img = uncond_model.ts_to_img(mask_exp)[:, :1, :, :]
                operator = operator_factory(mask_img)
                loss, _ = uncond_model.loss_fn_ambient_operator(
                    x_img, y_obs, operator,
                    corruption_type=corruption_type,
                    delta=delta, lambda_obs=1.0)

            elif corruption_type == 'gaussian_blur':
                y_img = batch[0].to(device)
                operator = operator_factory()
                loss, _ = uncond_model.loss_fn_ambient_operator(
                    y_img, y_img, operator,
                    corruption_type=corruption_type,
                    delta=0.0, lambda_obs=1.0)

            elif corruption_type in ('gaussian_noise', 'ts_gaussian_noise'):
                y_img = batch[0].to(device)
                loss, _ = uncond_model.loss_fn_irregular(y_img)

            elif corruption_type == 'ts_temporal_smoothing':
                y_ts = batch[0].to(device)
                x_img = uncond_model.ts_to_img(y_ts)
                operator = operator_factory(uncond_model=uncond_model)
                loss, _ = uncond_model.loss_fn_ambient_operator(
                    x_img, y_ts, operator,
                    corruption_type=corruption_type,
                    delta=0.0, lambda_obs=1.0)
            else:
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss += loss.item()
            n_batches += 1
            torch.cuda.empty_cache()

        avg = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Phase 0 epoch {epoch + 1}/{pretrain_epochs}, "
                  f"loss: {avg:.4f}")
        if logger is not None:
            logger.log('em/phase0_loss', avg, epoch)

    print("Phase 0 ambient pre-training complete.")
    return uncond_model


# =============================================================================
# Modified M-step: A-Aware Observation-Space Loss
# =============================================================================

def m_step_ambient(args, uncond_model, optimizer, reconstructions,
                   corrupted_obs, obs_masks, operator_factory,
                   corruption_type, em_iter, lambda_obs, device,
                   logger=None):
    """M-step that blends observation-space and full-space losses.

    loss = λ_obs · ||A(D_θ(x_t)) - y_obs||²
         + (1 - λ_obs) · ||D_θ(x_t) - x_recon||²

    For **missing data** the Ambient further-corruption trick is applied to
    the model input (randomly drop δ fraction of observed positions).
    """
    delta = getattr(args, 'further_corrupt_delta', 0.1)
    sigma_n = (getattr(args, 'corruption_noise_level', 0.0)
               if corruption_type in ('gaussian_noise', 'ts_gaussian_noise')
               else 0.0)

    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"  Training for {args.m_step_epochs} epochs on "
          f"{len(reconstructions)} samples (λ_obs={lambda_obs:.3f})")

    recon_t = torch.tensor(reconstructions, dtype=torch.float32)

    if corruption_type in ('missing', 'ts_missing_noise'):
        corr_t = torch.tensor(np.nan_to_num(corrupted_obs, nan=0.0),
                              dtype=torch.float32)
        mask_t = torch.tensor(obs_masks, dtype=torch.float32)
        dataset = Data.TensorDataset(recon_t, corr_t, mask_t)
    elif corruption_type in ('gaussian_noise', 'gaussian_blur',
                             'ts_gaussian_noise'):
        corr_t = torch.tensor(corrupted_obs, dtype=torch.float32)
        dataset = Data.TensorDataset(recon_t, corr_t)
    elif corruption_type == 'ts_temporal_smoothing':
        corr_t = torch.tensor(corrupted_obs, dtype=torch.float32)
        dataset = Data.TensorDataset(recon_t, corr_t)
    else:
        dataset = Data.TensorDataset(recon_t)

    loader = Data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    uncond_model.train()

    for epoch in range(args.m_step_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if corruption_type in ('missing', 'ts_missing_noise'):
                x_r, obs_b, mask_b = [t.to(device) for t in batch]
                x_img = uncond_model.ts_to_img(x_r)
                y_obs = uncond_model.ts_to_img(obs_b)
                mask_exp = mask_b.unsqueeze(-1).expand(-1, -1, x_r.shape[-1])
                mask_img = uncond_model.ts_to_img(mask_exp)[:, :1, :, :]
                operator = operator_factory(mask_img)

            elif corruption_type in ('gaussian_noise', 'gaussian_blur',
                                     'ts_gaussian_noise'):
                x_r, y_obs = batch[0].to(device), batch[1].to(device)
                x_img = uncond_model.ts_to_img(x_r)
                operator = operator_factory()

            elif corruption_type == 'ts_temporal_smoothing':
                x_r, y_obs = batch[0].to(device), batch[1].to(device)
                x_img = uncond_model.ts_to_img(x_r)
                operator = operator_factory(uncond_model=uncond_model)

            else:
                x_r = batch[0].to(device)
                x_img = uncond_model.ts_to_img(x_r)
                loss, _ = uncond_model.loss_fn_irregular(x_img)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
                optimizer.step()
                uncond_model.on_train_batch_end()
                epoch_loss += loss.item()
                n_batches += 1
                torch.cuda.empty_cache()
                continue

            loss, _ = uncond_model.loss_fn_ambient_operator(
                x_img, y_obs, operator,
                corruption_type=corruption_type,
                delta=delta, lambda_obs=lambda_obs, sigma_n=sigma_n)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss += loss.item()
            n_batches += 1
            torch.cuda.empty_cache()

        avg = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  M-step epoch {epoch + 1}/{args.m_step_epochs}, "
                  f"loss: {avg:.4f} (λ_obs={lambda_obs:.3f})")
        if logger is not None:
            step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg, step)
            logger.log('em/lambda_obs', lambda_obs, step)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


# =============================================================================
# λ_obs annealing
# =============================================================================

def _lambda_obs_schedule(em_iter, total_iters, start, end):
    """Linear interpolation from *start* to *end* over EM iterations."""
    if total_iters <= 1:
        return start
    frac = em_iter / (total_iters - 1)
    return start + (end - start) * frac


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    corruption_type = getattr(args, 'corruption_type', 'missing')
    lambda_obs_start = getattr(args, 'lambda_obs_start', 1.0)
    lambda_obs_end = getattr(args, 'lambda_obs_end', 0.3)
    curriculum_warmup_frac = getattr(args, 'curriculum_warmup_frac', 0.5)
    curriculum_easy_frac = getattr(args, 'curriculum_easy_frac', 0.7)

    name = create_model_name_and_dir(args)
    name = f"ambient_mmps_{corruption_type}_{name}"

    logging.info(args)
    args.ambient_concat_further_mask = True

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        print_model_params(logger, uncond_model)

        # === Prepare data ===
        print(f"\n{'=' * 60}")
        print(f"Corruption type: {corruption_type}")
        print(f"{'=' * 60}")

        if corruption_type == 'missing':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_missing(args, train_loader, uncond_model, args.device)
            print(f"Extracted {len(corrupted_obs)} sequences with "
                  f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")
        elif corruption_type == 'gaussian_noise':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_gaussian_noise(args, train_loader, test_loader,
                                            uncond_model, args.device)
        elif corruption_type == 'gaussian_blur':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_gaussian_blur(args, train_loader, test_loader,
                                           uncond_model, args.device)
        elif corruption_type == 'random_projection':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_random_projection(args, train_loader, test_loader,
                                               uncond_model, args.device)
        elif corruption_type == 'ts_gaussian_noise':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_ts_gaussian_noise(args, train_loader, test_loader,
                                               uncond_model, args.device)
        elif corruption_type == 'ts_temporal_smoothing':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_ts_temporal_smoothing(args, train_loader, test_loader,
                                                   uncond_model, args.device)
        elif corruption_type == 'ts_missing_noise':
            clean_data_ts, corrupted_obs, obs_masks, operator_factory = \
                prepare_data_ts_missing_noise(args, train_loader, test_loader,
                                              uncond_model, args.device)
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

        # === Initialize reconstructions (for difficulty scoring) ===
        print(f"\n{'=' * 60}")
        print("Phase 1: Initialization")
        print(f"{'=' * 60}")

        initial_reconstructions = initialize_reconstructions(
            corruption_type, corrupted_obs, obs_masks,
            clean_data_ts, uncond_model, operator_factory,
            args.device, seed=args.seed,
        )

        sorted_indices, difficulty_scores = compute_difficulty_scores(
            initial_reconstructions, obs_masks)

        # === Optional PPCA posterior ===
        use_ppca_post = getattr(args, 'use_ppca_posterior', False)
        ppca_cov = None
        if use_ppca_post:
            ppca_rank = getattr(args, 'ppca_posterior_rank', 32)
            print(f"\nFitting PPCA posterior covariance (rank={ppca_rank})")
            ppca_cov = ppca_fit_images(initial_reconstructions, uncond_model,
                                       args.device, rank=ppca_rank)

        # === Sigma-y annealing ===
        do_anneal = getattr(args, 'sigma_y_anneal', False)
        sigma_y_start = getattr(args, 'sigma_y_start', 0.1)
        sigma_y_end = getattr(args, 'sigma_y_end', 0.01)
        sigma_y_base = getattr(args, 'mmps_sigma_y', 0.01)

        def get_sigma_y(ei, total):
            if not do_anneal:
                return sigma_y_base
            frac = ei / max(total - 1, 1)
            return sigma_y_start * (sigma_y_end / sigma_y_start) ** frac

        # ==============================================================
        # Phase 0: Ambient Pre-training (replaces old Phase 1.5)
        # ==============================================================
        uncond_model = ambient_pretrain(
            args, uncond_model, optimizer,
            corrupted_obs, obs_masks, operator_factory,
            corruption_type, args.device, logger,
        )
        print("Phase 0 complete — model has a reasonable initialisation.\n")

        # ==============================================================
        # Phase 2: EM Loop with A-Aware M-step
        # ==============================================================
        warmup_end = int(args.em_iters * curriculum_warmup_frac)

        print(f"{'=' * 60}")
        print(f"Starting Ambient-MMPS [{corruption_type}] with "
              f"{args.em_iters} EM iterations")
        print(f"  MMPS cg_iters: {getattr(args, 'mmps_cg_iters', 1)}")
        print(f"  λ_obs: {lambda_obs_start:.2f} → {lambda_obs_end:.2f}")
        print(f"  Curriculum: warmup={warmup_end} iters, "
              f"easy_frac={curriculum_easy_frac}")
        if do_anneal:
            print(f"  σ_y annealing: {sigma_y_start} → {sigma_y_end}")
        print(f"{'=' * 60}")

        best_metrics = None
        metrics = None

        try:
            for em_iter in range(args.em_iters):
                print(f"\n{'=' * 60}")
                print(f"EM Iteration {em_iter + 1}/{args.em_iters}")

                current_sigma_y = get_sigma_y(em_iter, args.em_iters)
                sy_override = current_sigma_y if do_anneal else None
                current_lambda = _lambda_obs_schedule(
                    em_iter, args.em_iters,
                    lambda_obs_start, lambda_obs_end)

                if em_iter < warmup_end:
                    n_use = int(len(sorted_indices) * curriculum_easy_frac)
                    phase_label = (f"CURRICULUM ({n_use}/"
                                   f"{len(sorted_indices)} easiest)")
                else:
                    n_use = len(sorted_indices)
                    phase_label = f"FULL (all {n_use} samples)"

                print(f"  {phase_label}  |  λ_obs={current_lambda:.3f}")
                if do_anneal:
                    print(f"  σ_y = {current_sigma_y:.5f}")
                print(f"{'=' * 60}")

                # --- E-step (unchanged MMPS) ---
                reconstructions = e_step(
                    args, uncond_model, corrupted_obs, obs_masks,
                    operator_factory, em_iter, args.device, logger,
                    ppca_cov=ppca_cov, sigma_y_override=sy_override,
                )

                uncond_model.reset_ema()

                if use_ppca_post and (em_iter + 1) % 3 == 0:
                    print("  Re-fitting PPCA posterior covariance ...")
                    ppca_cov = ppca_fit_images(
                        reconstructions, uncond_model,
                        args.device, rank=getattr(args, 'ppca_posterior_rank', 32))

                # --- M-step (A-aware) ---
                if em_iter < warmup_end:
                    train_idx = sorted_indices[:n_use]
                    m_recon = reconstructions[train_idx]
                    m_corrupted = corrupted_obs[train_idx]
                    m_masks = obs_masks[train_idx]
                else:
                    m_recon = reconstructions
                    m_corrupted = corrupted_obs
                    m_masks = obs_masks

                uncond_model = m_step_ambient(
                    args, uncond_model, optimizer, m_recon,
                    m_corrupted, m_masks, operator_factory,
                    corruption_type, em_iter, current_lambda,
                    args.device, logger,
                )

                if logger is not None:
                    logger.log('em/curriculum_n_samples', n_use, em_iter)
                    logger.log('em/lambda_obs_current', current_lambda, em_iter)
                    if do_anneal:
                        logger.log('em/sigma_y', current_sigma_y, em_iter)

                if ((em_iter + 1) % args.em_eval_interval == 0 or
                        em_iter == args.em_iters - 1):
                    metrics = evaluate_uncond(
                        args, uncond_model, test_loader,
                        em_iter, args.device, logger)
                    if (best_metrics is None or
                            metrics.get('disc_mean', float('inf')) <
                            best_metrics.get('disc_mean', float('inf'))):
                        best_metrics = metrics

                if logger is not None:
                    logger.log('em/iteration', em_iter, em_iter)

        except Exception as e:
            logging.error(f"EM loop failed at iteration {em_iter}: "
                          f"{type(e).__name__}: {e}", exc_info=True)
            if logger is not None:
                logger.log('em/failed_at_iter', em_iter, em_iter)
            raise

        print("\n" + "=" * 60)
        print(f"Ambient-MMPS [{corruption_type}] Complete!")
        print("=" * 60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("Ambient-MMPS training is complete")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    for attr, default in [
        ('corruption_type', 'missing'),
        ('corruption_noise_level', 0.01),
        ('blur_sigma', 2.0),
        ('blur_kernel_size', None),
        ('projection_dim', None),
        ('curriculum_warmup_frac', 0.5),
        ('curriculum_easy_frac', 0.7),
        ('num_posterior_samples', 1),
        ('obs_consistency_mmps', False),
        ('use_ppca_posterior', False),
        ('ppca_posterior_rank', 32),
        ('sigma_y_anneal', False),
        ('sigma_y_start', 0.1),
        ('sigma_y_end', 0.01),
        ('smoothing_window', 5),
        ('ambient_pretrain_epochs', 100),
        ('lambda_obs_start', 1.0),
        ('lambda_obs_end', 0.3),
        ('further_corrupt_delta', 0.1),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
