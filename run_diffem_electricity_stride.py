"""
DiffEM with MMPS E-step — Electricity Strided Windows Experiment

Wraps the base run_diffem_mmps_ema_reset_stl_init pipeline with strided
sliding windows for the electricity dataset.

Why this helps:
  - With stride=1, the electricity dataset produces 26,280 windows from
    26,304 timesteps — adjacent windows differ by only ~4%. This 24x
    redundancy wastes compute and may cause overfitting.
  - With stride=6, we get ~4,380 windows with meaningful diversity,
    each sharing at most 75% of its content with the nearest neighbour.
  - Reduces training time per M-step epoch proportionally.
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
    save_reconstructions, load_reconstructions,
    MinMaxScaler, compute_sigma_from_schedule, add_gaussian_noise
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.utils_stl import initialize_with_stl

from run_diffem_mmps_ema_reset_stl_init import (
    MMPSDiffusionProcess, conjugate_gradient,
    get_corrupted_data_from_loader, initialize_with_gaussian,
    e_step, m_step, evaluate_uncond,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Strided data loading
# =============================================================================

def real_data_loading_strided(data_name, seq_len, missing_rate, stride=6,
                              gaussian_noise_level=0, noise_timestep=None):
    """
    Same as real_data_loading but with configurable stride for the
    sliding window. Reduces redundancy for large datasets like
    electricity.
    """
    assert data_name in ['stock', 'energy', 'ETTh1', 'ETTh2', 'ETTm1',
                          'ETTm2', 'weather', 'electricity']

    ori_data = np.loadtxt(f'./data/{data_name}.csv', delimiter=",", skiprows=1)

    ori_data = ori_data[::-1]
    ori_data = MinMaxScaler(ori_data)

    irregular_dataset = ori_data.copy()

    if noise_timestep is not None:
        gaussian_noise_level = compute_sigma_from_schedule(noise_timestep)
        print(f"Using diffusion schedule noise: timestep={noise_timestep}, "
              f"sigma={gaussian_noise_level:.6f}")

    if gaussian_noise_level > 0:
        irregular_dataset = add_gaussian_noise(ori_data, gaussian_noise_level)

    generator = torch.Generator().manual_seed(56789)
    removed_points = torch.randperm(ori_data.shape[0], generator=generator)[
                     :int(ori_data.shape[0] * missing_rate)].sort().values
    irregular_dataset[removed_points] = float('nan')

    total_length = len(ori_data)
    index = np.array(range(total_length)).reshape(-1, 1)

    temp_data = []
    for i in range(0, len(ori_data) - seq_len, stride):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    irregular_dataset_with_idx = np.concatenate((irregular_dataset, index), axis=1)
    irregular_temp_data = []
    for i in range(0, len(irregular_dataset_with_idx) - seq_len, stride):
        _x = irregular_dataset_with_idx[i:i + seq_len]
        irregular_temp_data.append(_x)

    irregular_data = []
    for i in range(len(irregular_temp_data)):
        irregular_data.append(irregular_temp_data[idx[i]])

    print(f"Strided loading: stride={stride}, windows={len(data)} "
          f"(vs {len(ori_data) - seq_len} with stride=1)")

    return data, irregular_data


def gen_dataloader_strided(args, stride=6):
    """
    gen_dataloader replacement that uses strided sliding windows.
    """
    noise_timestep = getattr(args, 'noise_timestep', None)
    ori_data, irregular_data_np = real_data_loading_strided(
        args.dataset, args.seq_len, missing_rate=args.missing_rate,
        stride=stride,
        gaussian_noise_level=args.gaussian_noise_level,
        noise_timestep=noise_timestep
    )

    ori_data = torch.Tensor(np.array(ori_data))
    ori_train_set = Data.TensorDataset(ori_data)
    irregular_data = torch.Tensor(np.array(irregular_data_np))
    irregular_train_set = Data.TensorDataset(irregular_data)

    train_loader = Data.DataLoader(
        dataset=irregular_train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    test_loader = Data.DataLoader(
        dataset=ori_train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    return train_loader, test_loader, None


# =============================================================================
# Main
# =============================================================================

def main(args):
    """DiffEM-MMPS with strided sliding windows for electricity."""

    stride = getattr(args, 'window_stride', 6)

    name = create_model_name_and_dir(args)
    name = f"diffem_electricity_stride{stride}_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use strided data loading instead of gen_dataloader
        train_loader, test_loader, _ = gen_dataloader_strided(args, stride=stride)
        logging.info(f'{args.dataset} dataset is ready (stride={stride}).')

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        print_model_params(logger, uncond_model)

        # === Extract corrupted data ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences with "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # === Phase 1: STL initialization ===
        print(f"\n{'='*60}")
        print("Phase 1: STL Warm-Start Initialization")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_stl(
            corrupted_data, obs_masks, seed=args.seed
        )

        # === Phase 1.5: Initial training ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial unconditional model training")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )
        print("Initial unconditional model training complete.")

        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS with {args.em_iters} EM iterations (stride={stride})")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")
        print(f"{'='*60}")

        best_metrics = None

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            reconstructions = e_step(
                args, uncond_model, corrupted_data, obs_masks,
                em_iter, args.device, logger
            )

            uncond_model.reset_ema()

            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, em_iter, args.device, logger
            )

            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, uncond_model, test_loader,
                    em_iter, args.device, logger
                )

                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        print("\n" + "="*60)
        print(f"DiffEM-MMPS + Stride({stride}) Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS + Strided Windows training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    if not hasattr(args, 'window_stride') or args.window_stride is None:
        args.window_stride = 6

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
