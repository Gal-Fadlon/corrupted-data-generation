"""
DiffEM with MMPS E-step — Kalman Init + Frequency-Gated + Decomposition Projection
+ Curriculum + A-aware M-step

This variant keeps the strongest current E-step stack and replaces the M-step
with an observation-vs-reconstruction trust schedule:

    loss = lambda_obs * observation_loss + (1 - lambda_obs) * full_loss

For missing data, the observation loss is computed through the binary masking
operator so the model starts by trusting only real observations, then gradually
leans more on the reconstructed hidden distribution as EM improves.
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


class MissingImageOperator:
    """Masking operator in image space for A-aware missing-data training."""

    def __init__(self, mask):
        self.mask = mask

    def forward(self, x):
        mask = self.mask
        if mask.shape[1] == 1 and x.shape[1] > 1:
            mask = mask.expand_as(x)
        return x * mask


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


def m_step_aaware(args, uncond_model, optimizer, reconstructions, corrupted_obs,
                  obs_masks, em_iter, lambda_obs, device, logger=None):
    """Trust-scheduled M-step for missing data using the A-aware loss."""
    delta = getattr(args, 'further_corrupt_delta', 0.1)

    print(f"\n=== A-aware M-Step (EM iter {em_iter}) ===")
    print(f"  Training for {args.m_step_epochs} epochs on {len(reconstructions)} samples")
    print(f"  lambda_obs={lambda_obs:.3f}, delta={delta:.3f}")

    recon_t = torch.tensor(reconstructions, dtype=torch.float32)
    corr_t = torch.tensor(np.nan_to_num(corrupted_obs, nan=0.0), dtype=torch.float32)
    mask_t = torch.tensor(obs_masks, dtype=torch.float32)
    dataset = Data.TensorDataset(recon_t, corr_t, mask_t)
    loader = Data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    uncond_model.train()
    for epoch in range(args.m_step_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x_r, obs_b, mask_b in loader:
            x_r = x_r.to(device)
            obs_b = obs_b.to(device)
            mask_b = mask_b.to(device)

            x_img = uncond_model.ts_to_img(x_r)
            y_obs = uncond_model.ts_to_img(obs_b)
            mask_exp = mask_b.unsqueeze(-1).expand(-1, -1, x_r.shape[-1])
            mask_img = uncond_model.ts_to_img(mask_exp)[:, :1, :, :]
            operator = MissingImageOperator(mask_img)

            loss, to_log = uncond_model.loss_fn_ambient_operator(
                x_img, y_obs, operator,
                corruption_type='missing',
                delta=delta,
                lambda_obs=lambda_obs,
            )

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
            print(f"  M-step epoch {epoch + 1}/{args.m_step_epochs}, loss: {avg:.4f} (lambda_obs={lambda_obs:.3f})")
        if logger is not None:
            step = (em_iter + 1) * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg, step)
            logger.log('em/lambda_obs', lambda_obs, step)
            logger.log('em/ambient_op_loss', to_log['ambient_op_loss'], step)
            logger.log('em/ambient_op_loss_obs', to_log['ambient_op_loss_obs'], step)
            logger.log('em/ambient_op_loss_full', to_log['ambient_op_loss_full'], step)

    print(f"A-aware M-step complete for EM iteration {em_iter}.")
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
    name = f"diffem_mmps_kalman_freqgate_decomp_proj_aaware_{name}"
    logging.info(args)

    lambda_obs_start = getattr(args, 'lambda_obs_start', 1.0)
    lambda_obs_end = getattr(args, 'lambda_obs_end', 0.3)
    trust_schedule = getattr(args, 'trust_schedule', 'linear')
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
        print("Starting Kalman + FreqGate + DecompProj + A-aware DiffEM-MMPS")
        print(f"  EM iterations: {args.em_iters}")
        print(f"  lambda_obs: {lambda_obs_start} -> {lambda_obs_end} ({trust_schedule})")
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

            em_progress = em_iter / max(args.em_iters - 1, 1)
            current_lambda = interpolate_schedule(
                lambda_obs_start, lambda_obs_end, em_progress, mode=trust_schedule,
            )
            current_sigma_y = get_sigma_y(args, em_iter, args.em_iters, base_sigma_y)
            args.mmps_sigma_y = current_sigma_y
            print(f"  lambda_obs={current_lambda:.3f}, sigma_y={current_sigma_y:.5f}")

            reconstructions = e_step(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                em_iter, args.device, logger,
            )
            last_recon = reconstructions

            uncond_model.reset_ema()

            uncond_model = m_step_aaware(
                args, uncond_model, optimizer,
                reconstructions, corrupted_data, obs_masks,
                em_iter, current_lambda, args.device, logger=logger,
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
                logger.log('em/lambda_obs_current', current_lambda, em_iter)
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
        print("Kalman + FreqGate + DecompProj + A-aware DiffEM-MMPS Complete!")
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
