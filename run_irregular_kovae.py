"""
KoVAE (Variational Koopman) baseline — irregular time-series generation.

Refactored to share the same logging, save, timing, and TS2Vec-cache infrastructure
as run_co_evolving_em.py (our method) and run_irregular.py (ImageI2R).

Differences from the original:
- Uses parse_args_irregular() (shared) instead of a custom argparse.
- Uses WandbLogger instead of Neptune.
- Reads --missing_rate (was --missing_value).
- Computes natural cubic spline coefficients on-the-fly every run; no disk cache
  at /home/galfad/data_for_koopman/.
- Block patterns supported via --missing_type; loaded through
  utils_data.real_data_loading / sine_data_generation / MujocoDataset.
- Saves with maybe_save_if_improved → tree placement under
  saved_models/{method}/missing_observation/{missing_type}/{dataset}/seq_len_N/missing_P/{run_name}/.
- Logs time/init_minutes, time/epoch_minutes, time/cumulative_training_minutes,
  time/time_to_best_disc_inc_init_minutes / _exc_init_minutes, time/epochs_to_best_disc,
  time/best_disc_value.
"""

import os
import sys
import time
import logging

import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim

import controldiffeq

from metrics import evaluate_model_irregular
from utils.loggers import CompositeLogger, WandbLogger, PrintLogger
from utils.utils import create_model_name_and_dir, log_config_and_tags
from utils.utils_args import parse_args_irregular
from utils.utils_data import (
    real_data_loading,
    sine_data_generation,
    MujocoDataset,
)
from utils.utils_save import maybe_save_if_improved
from utils.gpu_heartbeat import start_gpu_heartbeat, stop_gpu_heartbeat

from models.kovae_irregular import VK


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Dataset adapter — recomputes spline coefficients in memory each call.
# =============================================================================

class KoVAEAdapter(torch.utils.data.Dataset):
    """Loads (clean, irregular) sequences via shared utils_data, computes natural cubic
    spline coefficients in memory, and exposes the {'data', 'inter', 'original_data'}
    dict format that the VK model expects.

    No disk I/O. Splines are recomputed every run.
    """

    def __init__(self, args):
        self.device = args.device
        seq_len = args.seq_len
        corruption_type = getattr(args, 'corruption_type', 'missing_observation')
        missing_type = getattr(args, 'missing_type', 'fix_missing_rates')

        # ---- Get clean + irregular sequences using shared loaders ----
        if args.dataset == 'sine':
            args.dataset_size = 10000
            ori_list, irregular_list = sine_data_generation(
                args.dataset_size, seq_len,
                getattr(args, 'input_channels', 5),
                args.missing_rate,
                corruption_type=corruption_type,
                missing_type=missing_type,
                gaussian_noise_level=getattr(args, 'gaussian_noise_level', 0.0),
            )
            ori_arr = np.asarray(ori_list)
            irr_arr = np.asarray(irregular_list)

        elif args.dataset == 'mujoco':
            mset = MujocoDataset(seq_len, args.dataset, missing_rate=args.missing_rate)
            ori_list, irr_list = [], []
            for ori_b, irr_b in mset:
                ori_list.append(np.asarray(ori_b))
                irr_list.append(np.asarray(irr_b))
            ori_arr = np.asarray(ori_list)
            irr_arr = np.asarray(irr_list)

        else:
            ori_list, irregular_list = real_data_loading(
                args.dataset, seq_len,
                missing_rate=args.missing_rate,
                gaussian_noise_level=getattr(args, 'gaussian_noise_level', 0.0),
                noise_timestep=getattr(args, 'noise_timestep', None),
                corruption_type=corruption_type,
                missing_type=missing_type,
            )
            ori_arr = np.asarray(ori_list)
            irr_arr = np.asarray(irregular_list)

        self.original_sample = ori_arr                 # shape: (N, seq_len, F)
        self.samples = irr_arr                          # shape: (N, seq_len, F+1) with trailing time-index col

        # ---- Compute cubic spline coefficients in memory ----
        # Strip the trailing time-index column for the actual signal.
        norm_data_tensor = torch.tensor(self.samples[:, :, :-1], dtype=torch.float32, device=self.device)
        time_axis = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time_axis, norm_data_tensor)

        self.size = len(self.samples)

    def __getitem__(self, index):
        batch_coeff = (
            self.train_coeffs[0][index].float(),
            self.train_coeffs[1][index].float(),
            self.train_coeffs[2][index].float(),
            self.train_coeffs[3][index].float(),
        )
        return {
            'data': self.samples[index],
            'inter': batch_coeff,
            'original_data': self.original_sample[index],
        }

    def __len__(self):
        return len(self.samples)


# =============================================================================
# Helpers
# =============================================================================

def _set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def _agg_losses(running, losses):
    if not running:
        running = [[] for _ in range(len(losses))]
    for j, l in enumerate(losses):
        running[j].append(l.item())
    return running


# =============================================================================
# Main
# =============================================================================

def main(args):
    name = create_model_name_and_dir(args)
    name = f"kovae_{name}"

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    _set_seed(args.seed)

    # Start GPU heartbeat so the cluster's idle-GPU monitor doesn't kill us during
    # long CPU-bound init (spline computation can take minutes on large datasets).
    start_gpu_heartbeat()

    # Kovae-specific hyperparams (defaulted if not on CLI/config).
    args.z_dim = getattr(args, 'z_dim', 16)
    args.w_rec = getattr(args, 'w_rec', 1.0)
    args.w_kl = getattr(args, 'w_kl', 7e-4)
    args.w_pred_prior = getattr(args, 'w_pred_prior', 0.1)
    args.num_steps = getattr(args, 'num_steps', 1)
    args.budget = getattr(args, 'budget', 9.5e-2 * 0.5)
    args.pinv_solver = getattr(args, 'pinv_solver', False)
    args.batch_norm = getattr(args, 'batch_norm', True)
    args.logging_iter = getattr(args, 'logging_iter', 10)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        # ---- Init: load data + compute splines (this is "init_minutes") ----
        init_start = time.time()
        dataset = KoVAEAdapter(args)
        init_seconds = time.time() - init_start
        init_minutes = init_seconds / 60.0
        if logger is not None:
            logger.log('time/init_minutes', init_minutes, 0)
        print(f"  Init (data + spline) took {init_minutes:.2f} min")

        # input dims now known from data
        input_dim = dataset.original_sample.shape[-1]
        args.input_size = input_dim
        args.img_channels = input_dim
        args.inp_dim = input_dim

        # num_workers=0: the KoVAEAdapter stores spline coefficients as CUDA tensors,
        # and DataLoader workers are forked subprocesses that can't access CUDA memory
        # (would raise "CUDA error: initialization error" in __getitem__).
        train_loader = Data.DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0,
        )
        logging.info(f"{args.dataset} dataset is ready ({len(dataset)} sequences).")

        # ---- Model ----
        model = VK(args).to(device=args.device)
        learning_rate = getattr(args, 'learning_rate', 1e-3)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=getattr(args, 'weight_decay', 0.))

        params_num = sum(p.numel() for p in model.parameters())
        logging.info(f"number of model parameters: {params_num}")
        if logger is not None:
            logger.log('config/params_num', params_num, 0)

        # ---- Convergence + timing tracking ----
        metrics_history = []
        cumulative_training_seconds = 0.0
        best_disc_value = None
        best_disc_epoch = None
        cum_at_best_seconds = None

        stop_gpu_heartbeat()
        for epoch in range(0, args.epochs):
            print(f"Starting epoch {epoch}.")
            model.train()
            losses_running = []

            epoch_train_start = time.time()
            for i, data in enumerate(train_loader, 1):
                x = data['data'].to(args.device).float()
                train_coeffs = data['inter']
                t_axis = torch.arange(args.seq_len, dtype=torch.float32, device=args.device)
                final_index = (torch.ones(x.shape[0], device=args.device) * (args.seq_len - 1)).float()
                x = x[:, :, :-1]

                optimizer.zero_grad()
                x_rec, Z_enc, Z_enc_prior = model(t_axis, train_coeffs, final_index)

                x_no_nan = x[~torch.isnan(x)]
                x_rec_no_nan = x_rec[~torch.isnan(x)]

                losses = model.loss(x_no_nan, x_rec_no_nan, Z_enc, Z_enc_prior)
                losses[0].backward()
                optimizer.step()
                losses_running = _agg_losses(losses_running, losses)

            epoch_train_seconds = time.time() - epoch_train_start
            cumulative_training_seconds += epoch_train_seconds

            if losses_running:
                avg_total = float(np.mean(losses_running[0]))
                if logger is not None:
                    logger.log('train/loss_total', avg_total, epoch)
                print(f"  epoch {epoch}: loss_total={avg_total:.4e}, "
                      f"epoch_time={epoch_train_seconds:.1f}s")

            if logger is not None:
                logger.log('time/epoch_minutes', epoch_train_seconds / 60.0, epoch)
                logger.log('time/cumulative_training_minutes', cumulative_training_seconds / 60.0, epoch)

            # ---- Eval ----
            if epoch % args.logging_iter == 0:
                model.eval()
                gen_chunks = []
                real_chunks = []
                with torch.no_grad():
                    for data in train_loader:
                        n_sample = data['original_data'].shape[0]
                        gen_chunks.append(model.sample_data(n_sample).detach().cpu().numpy())
                        real_chunks.append(data['original_data'].detach().cpu().numpy())
                gen_sig = np.vstack(gen_chunks)
                real_sig = np.vstack(real_chunks)

                scores = evaluate_model_irregular(real_sig, gen_sig, args)
                for k, v in scores.items():
                    print(f"  test/{k} = {v:.4f}")
                    if logger is not None:
                        logger.log(f'test/{k}', v, epoch)

                metrics_history.append({'epoch': epoch, **scores})

                current_disc = scores.get('disc_mean', float('inf'))
                if best_disc_value is None or current_disc < best_disc_value:
                    best_disc_value = current_disc
                    best_disc_epoch = epoch
                    cum_at_best_seconds = cumulative_training_seconds

                # Save-if-improved. Kovae checkpoint also carries kovae optimizer.
                maybe_save_if_improved(
                    args, model, optimizer,
                    real_sig, gen_sig, last_recon=None,
                    metrics_history=metrics_history, em_iter=epoch, logger=logger,
                    current_disc=current_disc, log_step=epoch,
                    extra_state={'kovae_optimizer': optimizer.state_dict()},
                )

        # ---- Convergence summary ----
        if cum_at_best_seconds is not None:
            time_to_best_exc_init_minutes = cum_at_best_seconds / 60.0
            time_to_best_inc_init_minutes = (init_seconds + cum_at_best_seconds) / 60.0
            print("\nConvergence summary:")
            print(f"  Best disc: {best_disc_value:.4f} at epoch={best_disc_epoch}")
            print(f"  Time to best disc (inc init): {time_to_best_inc_init_minutes:.2f} min")
            print(f"  Time to best disc (exc init): {time_to_best_exc_init_minutes:.2f} min")
            if logger is not None:
                logger.log('time/time_to_best_disc_inc_init_minutes', time_to_best_inc_init_minutes, 0)
                logger.log('time/time_to_best_disc_exc_init_minutes', time_to_best_exc_init_minutes, 0)
                logger.log('time/epochs_to_best_disc', best_disc_epoch, 0)
                logger.log('time/best_disc_value', best_disc_value, 0)

        logging.info("KoVAE training is complete")


if __name__ == '__main__':
    args = parse_args_irregular()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
