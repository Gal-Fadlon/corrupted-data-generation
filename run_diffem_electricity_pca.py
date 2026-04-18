"""
DiffEM with MMPS E-step — Electricity PCA Experiment

Wraps the base run_diffem_mmps_ema_reset_stl_init pipeline with PCA
dimensionality reduction: 321 electricity features are projected to
`pca_components` (default 32) principal components before the EM loop,
and generated samples are inverse-transformed back to 321 features for
evaluation.

Why this helps:
  - 25 PCA components capture 90% of variance across the 321 consumers.
  - Reduces the UNet input from (321, 8, 8) to (32, 8, 8), turning a
    2.5:1 compression into a 4:1 expansion at the first conv layer.
  - Makes the problem structurally similar to the energy dataset (28
    features) where the method already works well.
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
from sklearn.decomposition import PCA

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
from utils.utils_stl import initialize_with_stl

from run_diffem_mmps_ema_reset_stl_init import (
    MMPSDiffusionProcess, conjugate_gradient,
    get_corrupted_data_from_loader, initialize_with_gaussian,
    e_step, m_step,
)
from utils.train_unconditional import train_unconditional_regular

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# PCA Utilities
# =============================================================================

def fit_pca_on_observed(corrupted_data, obs_masks, n_components=32):
    """
    Fit PCA on observed (non-NaN) values across all windows.

    Args:
        corrupted_data: (N, seq_len, 321) with NaN for missing
        obs_masks: (N, seq_len) boolean
        n_components: number of PCA components

    Returns:
        pca: fitted sklearn PCA object
    """
    N, T, C = corrupted_data.shape
    observed_rows = []
    for i in range(N):
        for t in range(T):
            if obs_masks[i, t]:
                observed_rows.append(corrupted_data[i, t, :])

    observed_matrix = np.array(observed_rows)
    nan_mask = np.isnan(observed_matrix).any(axis=1)
    observed_matrix = observed_matrix[~nan_mask]

    print(f"Fitting PCA on {len(observed_matrix)} observed rows, {C} -> {n_components} components")
    pca = PCA(n_components=n_components)
    pca.fit(observed_matrix)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA explains {explained*100:.1f}% of variance with {n_components} components")
    return pca


def transform_windows_pca(data, pca):
    """
    Project windowed data (N, T, C_orig) -> (N, T, n_components).
    NaN values are zero-filled before projection (PCA is linear so
    zero-fill + project is equivalent to projecting the observed part
    when NaN fraction is small).
    """
    N, T, C = data.shape
    flat = np.nan_to_num(data, nan=0.0).reshape(N * T, C)
    projected = pca.transform(flat)
    return projected.reshape(N, T, pca.n_components)


def inverse_transform_pca(data_pca, pca):
    """
    Inverse-project (N, T, n_components) -> (N, T, C_orig).
    """
    N, T, K = data_pca.shape
    flat = data_pca.reshape(N * T, K)
    reconstructed = pca.inverse_transform(flat)
    return reconstructed.reshape(N, T, -1)


# =============================================================================
# PCA-aware evaluation
# =============================================================================

def evaluate_uncond_pca(args, uncond_model, test_loader, em_iter, device,
                        pca, original_n_features, logger=None):
    """
    Evaluate: generate in PCA space, inverse-transform to original 321
    features, then compute metrics against real data.
    """
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    print("Evaluating unconditional model (PCA inverse -> 321 features)...")

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

                gen_pca = x_ts.cpu().numpy()
                gen_orig = inverse_transform_pca(gen_pca, pca)
                gen_sig.append(gen_orig)

                real_pca = data[0].cpu().numpy()
                real_orig = inverse_transform_pca(real_pca, pca)
                real_sig.append(real_orig)

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    args_eval = argparse_copy_with_channels(args, original_n_features)
    scores = evaluate_model_irregular(real_sig, gen_sig, args_eval)

    print(f"EM iter {em_iter} metrics (PCA -> 321 features):")
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


def argparse_copy_with_channels(args, n_channels):
    """Return a shallow copy of args with input_channels overridden."""
    import copy
    args_copy = copy.copy(args)
    args_copy.input_channels = n_channels
    args_copy.input_size = n_channels
    return args_copy


# =============================================================================
# Main
# =============================================================================

def main(args):
    """DiffEM-MMPS with PCA dimensionality reduction for electricity."""

    pca_components = getattr(args, 'pca_components', 32)
    original_n_features = args.input_channels  # 321

    name = create_model_name_and_dir(args)
    name = f"diffem_electricity_pca{pca_components}_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        # === Extract corrupted data (321 features) ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences, shape {corrupted_data.shape}")

        # === Fit PCA on observed data ===
        print(f"\n{'='*60}")
        print(f"Fitting PCA: {original_n_features} -> {pca_components} components")
        print(f"{'='*60}")
        pca = fit_pca_on_observed(corrupted_data, obs_masks, n_components=pca_components)

        # === Project corrupted data to PCA space ===
        corrupted_pca = transform_windows_pca(corrupted_data, pca)
        nan_positions = np.isnan(corrupted_data).any(axis=-1)
        for i in range(corrupted_pca.shape[0]):
            for t in range(corrupted_pca.shape[1]):
                if nan_positions[i, t]:
                    corrupted_pca[i, t, :] = np.nan

        print(f"Projected corrupted data: {corrupted_data.shape} -> {corrupted_pca.shape}")

        # === Project test data to PCA space (replace test_loader) ===
        all_test = []
        for data in test_loader:
            all_test.append(data[0].numpy())
        test_orig = np.vstack(all_test)
        test_pca = transform_windows_pca(test_orig, pca)
        test_dataset = Data.TensorDataset(torch.tensor(test_pca, dtype=torch.float32))
        test_loader_pca = Data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers
        )

        # === Override args for PCA dimensionality ===
        args.input_channels = pca_components
        args.input_size = pca_components

        # === Create model in PCA space ===
        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        print_model_params(logger, uncond_model)

        # === Phase 1: STL initialization in PCA space ===
        print(f"\n{'='*60}")
        print("Phase 1: STL Warm-Start Initialization (PCA space)")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_stl(
            corrupted_pca, obs_masks, seed=args.seed
        )

        # === Phase 1.5: Initial training ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial unconditional model training (PCA space)")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )

        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS with {args.em_iters} EM iterations (PCA space)")
        print(f"  PCA components: {pca_components}")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")
        print(f"{'='*60}")

        best_metrics = None

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            reconstructions = e_step(
                args, uncond_model, corrupted_pca, obs_masks,
                em_iter, args.device, logger
            )

            uncond_model.reset_ema()

            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, em_iter, args.device, logger
            )

            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond_pca(
                    args, uncond_model, test_loader_pca,
                    em_iter, args.device,
                    pca, original_n_features, logger
                )

                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics

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
        print(f"DiffEM-MMPS + PCA({pca_components}) Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS + PCA training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    if not hasattr(args, 'pca_components') or args.pca_components is None:
        args.pca_components = 32

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
