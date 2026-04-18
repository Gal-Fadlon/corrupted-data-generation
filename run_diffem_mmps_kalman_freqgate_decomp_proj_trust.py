"""
DiffEM with MMPS E-step — Kalman Init + Frequency-Gated + Decomposition Projection
+ Curriculum + Trust-aware M-step

This runner directly experiments with how much the model should trust imputed
neighbors during M-step training on EM reconstructions. It supports three modes:

  - soft_loss:     keep full context, add a small loss weight on imputed pixels
  - input_dropout: hide a subset of imputed pixels from the UNet input
  - two_branch:    mix conservative observed-only batches with full-context ones
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.multiprocessing
import torch.utils.data as Data
from tqdm import tqdm

from metrics import evaluate_model_irregular
from metrics.memorization import compute_memorization_metric
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader, save_reconstructions
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.utils_stl import initialize_with_kalman
from utils.train_unconditional import train_unconditional_regular
from run_diffem_mmps_kalman_freqgate_decomp_proj_ambient import (
    FreqGatedDecompProjectedMMPSDiffusionProcess,
    get_corrupted_data_from_loader,
    m_step_vanilla,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def interpolate_schedule(start, end, progress, mode='linear'):
    progress = float(max(0.0, min(1.0, progress)))
    if mode == 'cosine':
        progress = 0.5 - 0.5 * np.cos(np.pi * progress)
    return start + (end - start) * progress


def get_sigma_y(args, em_iter, total_iters, base_sigma_y):
    if not getattr(args, 'sigma_y_anneal', False):
        return base_sigma_y

    start = getattr(args, 'sigma_y_start', 0.1)
    end = getattr(args, 'sigma_y_end', 0.01)
    frac = em_iter / max(total_iters - 1, 1)
    return start * (end / start) ** frac


def e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    print(f"\n=== E-Step (EM iter {em_iter}) — Freq-Gated + Decomp-Projected MMPS ===")
    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    n_samples = len(corrupted_data)
    batch_size = getattr(args, 'e_step_batch_size', args.batch_size)
    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    with uncond_model.ema_scope():
        process = FreqGatedDecompProjectedMMPSDiffusionProcess(
            args, uncond_model.net, target_shape, uncond_model,
            sigma_y=sigma_y, cg_iters=cg_iters,
            projection_strength=0.5, sigma_threshold_frac=0.5,
            gate_schedule='linear',
        )

        for start_idx in tqdm(range(0, n_samples, batch_size), desc="E-step FreqGate+DecompProj"):
            end_idx = min(start_idx + batch_size, n_samples)

            corrupted_batch = corrupted_data[start_idx:end_idx]
            mask_batch = obs_masks[start_idx:end_idx]

            corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
            mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)
            obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)

            x_obs_img = uncond_model.ts_to_img(obs_ts)
            mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
            mask_img = uncond_model.ts_to_img(mask_ts_expanded)[:, :1, :, :]

            x_img_imputed = process.sampling_mmps(x_obs_img, mask_img)
            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)

            all_reconstructions.append(x_ts_recon.cpu().numpy())
            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    print(f"E-step complete. Imputed {len(reconstructions)} sequences.")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                             f'missing_{int(args.missing_rate * 100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def get_trust_values(args, em_iter):
    progress = em_iter / max(args.em_iters - 1, 1)
    schedule = getattr(args, 'trust_schedule', 'linear')
    return {
        'imputed_weight': interpolate_schedule(
            getattr(args, 'imputed_loss_weight_start', 0.0),
            getattr(args, 'imputed_loss_weight_end', 0.1),
            progress, mode=schedule,
        ),
        'imputed_keep': interpolate_schedule(
            getattr(args, 'imputed_keep_ratio_start', 1.0),
            getattr(args, 'imputed_keep_ratio_end', 1.0),
            progress, mode=schedule,
        ),
        'observed_keep': interpolate_schedule(
            getattr(args, 'observed_keep_ratio_start', 1.0),
            getattr(args, 'observed_keep_ratio_end', 1.0),
            progress, mode=schedule,
        ),
        'full_context_prob': interpolate_schedule(
            getattr(args, 'full_context_prob_start', 0.2),
            getattr(args, 'full_context_prob_end', 0.8),
            progress, mode=schedule,
        ),
    }


def m_step_trust(args, uncond_model, optimizer, reconstructions, obs_masks,
                 em_iter, device, logger=None):
    """Trust-aware M-step on EM reconstructions with the original obs mask."""
    mode = getattr(args, 'trust_mstep_mode', 'soft_loss')
    trust_values = get_trust_values(args, em_iter)

    print(f"\n=== Trust-aware M-Step (EM iter {em_iter}) ===")
    print(f"  mode={mode}")
    print(f"  imputed_weight={trust_values['imputed_weight']:.3f}")
    print(f"  imputed_keep={trust_values['imputed_keep']:.3f}")
    print(f"  observed_keep={trust_values['observed_keep']:.3f}")
    if mode == 'two_branch':
        print(f"  full_context_prob={trust_values['full_context_prob']:.3f}")
    print(f"Training unconditional model for {args.m_step_epochs} epochs...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    mask_tensor = torch.tensor(obs_masks, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor, mask_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )

    uncond_model.train()
    for epoch in range(args.m_step_epochs):
        epoch_loss, num_batches = 0.0, 0
        last_log = None

        for x_clean, mask_batch in recon_loader:
            x_clean = x_clean.to(device)
            mask_batch = mask_batch.to(device)

            x_img = uncond_model.ts_to_img(x_clean)
            mask_expanded = mask_batch.unsqueeze(-1).expand(-1, -1, x_clean.shape[-1])
            mask_img = uncond_model.ts_to_img(mask_expanded)[:, :1, :, :]

            if mode == 'soft_loss':
                loss, last_log = uncond_model.loss_fn_trust_guided(
                    x_img, mask_img,
                    imputed_weight=trust_values['imputed_weight'],
                    imputed_keep_prob=1.0,
                    observed_keep_prob=1.0,
                )
            elif mode == 'input_dropout':
                loss, last_log = uncond_model.loss_fn_trust_guided(
                    x_img, mask_img,
                    imputed_weight=trust_values['imputed_weight'],
                    imputed_keep_prob=trust_values['imputed_keep'],
                    observed_keep_prob=trust_values['observed_keep'],
                )
            elif mode == 'two_branch':
                loss, last_log = uncond_model.loss_fn_trust_mixture(
                    x_img, mask_img,
                    imputed_weight=trust_values['imputed_weight'],
                    imputed_keep_prob=trust_values['imputed_keep'],
                    observed_keep_prob=trust_values['observed_keep'],
                    full_context_prob=trust_values['full_context_prob'],
                )
            else:
                raise ValueError(f"Unsupported trust_mstep_mode: {mode}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss += loss.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(num_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  M-step epoch {epoch + 1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")
        if logger is not None:
            step = (em_iter + 1) * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, step)
            logger.log('em/imputed_loss_weight', trust_values['imputed_weight'], step)
            logger.log('em/imputed_keep_ratio', trust_values['imputed_keep'], step)
            logger.log('em/observed_keep_ratio', trust_values['observed_keep'], step)
            if mode == 'two_branch':
                logger.log('em/full_context_prob', trust_values['full_context_prob'], step)
            if last_log is not None:
                for key, value in last_log.items():
                    logger.log(f'em/{key}', value, step)

    print(f"Trust-aware M-step complete for EM iteration {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, em_iter, device, logger=None):
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    uncond_model.eval()
    gen_sig, real_sig = [], []

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution, args.img_resolution),
            )
            for data in tqdm(test_loader, desc="Evaluating"):
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = uncond_model.img_to_ts(x_img_sampled)
                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)
    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    print(f"EM iter {em_iter} metrics:")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

    mem_plot_path = f"memorization_hist_em_iter_{em_iter}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig,
        device=device, plot_path=mem_plot_path,
    )
    for key, value in mem_stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        if logger is not None:
            logger.log(f'test/memorization/{key}', value, em_iter)
    if logger is not None:
        try:
            logger.log_file('test/memorization/histogram', mem_plot_path, em_iter)
            if os.path.exists(mem_plot_path):
                os.remove(mem_plot_path)
        except Exception:
            pass

    return scores


def main(args):
    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_kalman_freqgate_decomp_proj_trust_{name}"
    logging.info(args)

    base_sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    args.ambient_concat_further_mask = bool(
        getattr(args, 'ambient_concat_further_mask', False)
    )

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)
        print(f"  ambient_concat_further_mask: {args.ambient_concat_further_mask}")
        if logger is not None:
            logger.log(
                'config/ambient_concat_further_mask',
                int(args.ambient_concat_further_mask),
                0,
            )

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        print_model_params(logger, uncond_model)

        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"\n{'=' * 60}")
        print("Phase 1: Kalman Smoother Warm-Start Initialization")
        print(f"{'=' * 60}")

        initial_reconstructions = initialize_with_kalman(
            corrupted_data, obs_masks, seed=args.seed,
            per_fit_timeout=getattr(args, 'kalman_fit_timeout', 5),
            max_seconds=getattr(args, 'kalman_global_timeout', 1800),
        )

        uncond_model = m_step_vanilla(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger,
        )
        print("\n=== Phase-1 evaluation checkpoint (pre-EM) ===")
        best_metrics = evaluate_uncond(
            args, uncond_model, test_loader,
            em_iter=-1, device=args.device, logger=logger,
        )

        print(f"\n{'=' * 60}")
        print("Starting Kalman + FreqGate + DecompProj + Trust-aware DiffEM-MMPS")
        print(f"  EM iterations: {args.em_iters}")
        print(f"  trust mode: {args.trust_mstep_mode}")
        print(f"  trust schedule: {args.trust_schedule}")
        if getattr(args, 'sigma_y_anneal', False):
            print(f"  sigma_y annealing: {args.sigma_y_start} -> {args.sigma_y_end}")
        print(f"{'=' * 60}")

        last_recon = initial_reconstructions

        for em_iter in range(args.em_iters):
            print(f"\n{'=' * 60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'=' * 60}")

            progress = min(1.0, (em_iter + 1) / max(args.em_iters * 0.6, 1))
            reveal_prob = (1.0 - progress) * getattr(args, 'curriculum_reveal_max', 0.3)
            extra_reveal = (np.random.rand(*obs_masks.shape) < reveal_prob) & ~obs_masks
            curriculum_masks = obs_masks | extra_reveal
            curriculum_corrupted = corrupted_data.copy()
            curriculum_corrupted[extra_reveal] = last_recon[extra_reveal]
            n_revealed = extra_reveal.sum()
            if n_revealed > 0:
                print(f"  Curriculum: revealed {n_revealed} extra positions (progress={progress:.2f})")

            current_sigma_y = get_sigma_y(args, em_iter, args.em_iters, base_sigma_y)
            args.mmps_sigma_y = current_sigma_y
            print(f"  sigma_y={current_sigma_y:.5f}")

            reconstructions = e_step(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                em_iter, args.device, logger,
            )
            last_recon = reconstructions

            uncond_model.reset_ema()

            uncond_model = m_step_trust(
                args, uncond_model, optimizer,
                reconstructions, obs_masks,
                em_iter, args.device, logger=logger,
            )

            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, uncond_model, test_loader,
                    em_iter, args.device, logger,
                )
                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)
                logger.log('em/sigma_y', current_sigma_y, em_iter)


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
        print("\n" + "=" * 60)
        print("Kalman + FreqGate + DecompProj + Trust-aware DiffEM-MMPS Complete!")
        print("=" * 60)
        if best_metrics:
            for key, value in best_metrics.items():
                print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
