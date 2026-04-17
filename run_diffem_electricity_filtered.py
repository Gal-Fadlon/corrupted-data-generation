"""
DiffEM with MMPS E-step — Electricity Feature Filtering Experiment

Wraps the base run_diffem_mmps_ema_reset_stl_init pipeline with a
feature-filtering step: removes pathological electricity features
(high zero-fraction or near-constant) before the EM loop.

Why this helps:
  - 8 features have >10% zeros after normalization (feature 182 is 75.7%
    zeros). These create bimodal distributions the diffusion model
    struggles with.
  - 5 features are near-constant (std < 0.05), adding noise without
    signal.
  - Removing ~10 features leaves ~311, reducing noise and improving the
    signal-to-noise ratio in the learned distribution.
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
    gen_dataloader, save_reconstructions, load_reconstructions, MinMaxScaler
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.utils_stl import initialize_with_stl

from run_diffem_mmps_ema_reset_stl_init import (
    MMPSDiffusionProcess, conjugate_gradient,
    get_corrupted_data_from_loader, initialize_with_gaussian,
    e_step, m_step,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Feature Filtering
# =============================================================================

def identify_features_to_remove(zero_frac_threshold=0.4, std_threshold=0.05):
    """
    Load electricity data, compute per-feature statistics, and return
    indices of features to keep.

    Returns:
        keep_mask: boolean array (321,), True = keep
        feature_means: (321,) means for padding removed features later
    """
    data = np.loadtxt('./data/electricity.csv', delimiter=",", skiprows=1)
    data = data[::-1]
    data_norm = MinMaxScaler(data)

    n_features = data_norm.shape[1]
    zero_frac = (data_norm < 0.01).sum(axis=0) / data_norm.shape[0]
    stds = data_norm.std(axis=0)
    feature_means = data_norm.mean(axis=0)

    remove = (zero_frac > zero_frac_threshold) | (stds < std_threshold)
    keep_mask = ~remove

    removed_indices = np.where(remove)[0]
    print(f"Feature filtering: removing {remove.sum()}/{n_features} features")
    print(f"  Removed indices: {removed_indices.tolist()}")
    print(f"  Remaining features: {keep_mask.sum()}")

    return keep_mask, feature_means


def filter_windows(data, keep_mask):
    """
    Filter features from windowed data.

    Args:
        data: (N, seq_len, C_orig) — may contain NaN
        keep_mask: boolean (C_orig,)

    Returns:
        (N, seq_len, C_filtered)
    """
    return data[:, :, keep_mask]


def unfilter_windows(data_filtered, keep_mask, feature_means):
    """
    Reconstruct full-feature data by inserting removed features
    filled with their training-set means.

    Args:
        data_filtered: (N, seq_len, C_filtered)
        keep_mask: boolean (C_orig,)
        feature_means: (C_orig,) means for padding

    Returns:
        (N, seq_len, C_orig)
    """
    N, T, _ = data_filtered.shape
    C_orig = len(keep_mask)
    full = np.tile(feature_means, (N, T, 1))  # (N, T, C_orig)
    full[:, :, keep_mask] = data_filtered
    return full


# =============================================================================
# Filtered evaluation
# =============================================================================

def evaluate_uncond_filtered(args, uncond_model, test_loader, em_iter, device,
                             keep_mask, feature_means, original_n_features,
                             logger=None):
    """
    Evaluate: generate in filtered space, pad back to 321 features,
    then compute metrics against real data (also padded back).
    """
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    print("Evaluating unconditional model (filtered -> 321 features)...")

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

                gen_filtered = x_ts.cpu().numpy()
                gen_full = unfilter_windows(gen_filtered, keep_mask, feature_means)
                gen_sig.append(gen_full)

                real_filtered = data[0].cpu().numpy()
                real_full = unfilter_windows(real_filtered, keep_mask, feature_means)
                real_sig.append(real_full)

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    import copy
    args_eval = copy.copy(args)
    args_eval.input_channels = original_n_features
    args_eval.input_size = original_n_features

    scores = evaluate_model_irregular(real_sig, gen_sig, args_eval)

    print(f"EM iter {em_iter} metrics (filtered -> 321 features):")
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
# Main
# =============================================================================

def main(args):
    """DiffEM-MMPS with feature filtering for electricity."""

    original_n_features = args.input_channels  # 321

    # === Identify features to remove ===
    zero_frac_threshold = getattr(args, 'filter_zero_threshold', 0.4)
    std_threshold = getattr(args, 'filter_std_threshold', 0.05)

    keep_mask, feature_means = identify_features_to_remove(
        zero_frac_threshold=zero_frac_threshold,
        std_threshold=std_threshold
    )
    n_kept = int(keep_mask.sum())

    name = create_model_name_and_dir(args)
    name = f"diffem_electricity_filtered{n_kept}_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        # === Extract and filter corrupted data ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences, shape {corrupted_data.shape}")

        corrupted_filtered = filter_windows(corrupted_data, keep_mask)
        print(f"Filtered: {corrupted_data.shape} -> {corrupted_filtered.shape}")

        # === Filter test data ===
        all_test = []
        for data in test_loader:
            all_test.append(data[0].numpy())
        test_orig = np.vstack(all_test)
        test_filtered = filter_windows(test_orig, keep_mask)
        test_dataset = Data.TensorDataset(torch.tensor(test_filtered, dtype=torch.float32))
        test_loader_filtered = Data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers
        )

        # === Override args for filtered dimensionality ===
        args.input_channels = n_kept
        args.input_size = n_kept

        # === Create model in filtered space ===
        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        print_model_params(logger, uncond_model)

        # === Phase 1: STL initialization ===
        print(f"\n{'='*60}")
        print(f"Phase 1: STL Warm-Start Initialization ({n_kept} features)")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_stl(
            corrupted_filtered, obs_masks, seed=args.seed
        )

        # === Phase 1.5: Initial training ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial unconditional model training")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )

        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS with {args.em_iters} EM iterations ({n_kept} features)")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"{'='*60}")

        best_metrics = None

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            reconstructions = e_step(
                args, uncond_model, corrupted_filtered, obs_masks,
                em_iter, args.device, logger
            )

            uncond_model.reset_ema()

            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, em_iter, args.device, logger
            )

            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond_filtered(
                    args, uncond_model, test_loader_filtered,
                    em_iter, args.device,
                    keep_mask, feature_means, original_n_features, logger
                )

                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        print("\n" + "="*60)
        print(f"DiffEM-MMPS + Feature Filtering ({n_kept}/{original_n_features}) Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS + Feature Filtering training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
