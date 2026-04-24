"""
CSDI-Impute + EDM Baseline: Stochastic Commit-and-Forget

Pipeline:
  1. Load corrupted time series dataset (same as our method)
  2. Train CSDI on corrupted data (CSDI uses observed positions only)
  3. Generate stochastic imputations for all missing values
  4. Train standard EDM (unconditional diffusion) on the completed data
  5. Evaluate generation quality

This is the strongest possible "stochastic commit-and-forget" baseline:
CSDI produces diverse posterior samples (avoiding variance collapse), but the
imputer is frozen — it never receives feedback from the generator.

Usage:
  python run_csdi_impute_baseline.py \
      --config ./configs/seq_len_24/energy.yaml \
      --missing_rate 0.5 \
      --wandb true \
      --csdi_epochs 200 \
      --csdi_nsample 1
"""

import torch
import torch.utils.data as Data
import numpy as np
import os
import sys
import logging
import yaml
from tqdm import tqdm

# Load CSDI modules without polluting sys.modules (avoids shadowing our own 'utils' package)
CSDI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CSDI')
sys.path.insert(0, CSDI_DIR)

import importlib.util

def _load_csdi_module(name, filepath):
    spec = importlib.util.spec_from_file_location(f"csdi_internal.{name}", filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"csdi_internal.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

_csdi_diff_models = _load_csdi_module('diff_models', os.path.join(CSDI_DIR, 'diff_models.py'))
sys.modules['diff_models'] = _csdi_diff_models  # main_model.py does: from diff_models import ...
_csdi_main = _load_csdi_module('main_model', os.path.join(CSDI_DIR, 'main_model.py'))
_csdi_utils = _load_csdi_module('csdi_utils', os.path.join(CSDI_DIR, 'utils.py'))

CSDI_base = _csdi_main.CSDI_base
csdi_train = _csdi_utils.train

sys.path.remove(CSDI_DIR)

from metrics import evaluate_model_irregular
from metrics.memorization import compute_memorization_metric
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.train_unconditional import train_unconditional_regular

torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# CSDI adapter for our datasets
# =============================================================================

class CSDI_OurDataset(CSDI_base):
    """CSDI model adapted for our time series datasets."""

    def __init__(self, config, device, target_dim):
        super().__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # (B, K, L)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDIDataset(torch.utils.data.Dataset):
    """Wraps our corrupted data into the format CSDI expects."""

    def __init__(self, corrupted_data, obs_masks, seq_len):
        """
        Args:
            corrupted_data: (N, seq_len, features) with NaN for missing
            obs_masks: (N, seq_len) with True for observed time steps
        """
        self.N = len(corrupted_data)
        self.seq_len = seq_len
        self.features = corrupted_data.shape[-1]

        clean_data = np.nan_to_num(corrupted_data, nan=0.0)
        self.observed_data = torch.tensor(clean_data, dtype=torch.float32)

        mask_expanded = np.repeat(obs_masks[:, :, np.newaxis], self.features, axis=2)
        self.observed_mask = torch.tensor(mask_expanded, dtype=torch.float32)
        self.gt_mask = self.observed_mask.clone()
        self.timepoints = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).expand(self.N, -1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            "observed_data": self.observed_data[idx],
            "observed_mask": self.observed_mask[idx],
            "gt_mask": self.gt_mask[idx],
            "timepoints": self.timepoints[idx],
        }


def get_csdi_config(features, csdi_epochs=200):
    """Generate CSDI config dict matching the base config."""
    return {
        "train": {
            "epochs": csdi_epochs,
            "batch_size": 16,
            "lr": 1e-3,
            "itr_per_epoch": int(1e8),
        },
        "diffusion": {
            "layers": 4,
            "channels": 64,
            "nheads": 8,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": 50,
            "schedule": "quad",
            "is_linear": False,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 128,
            "featureemb": 16,
            "target_strategy": "random",
        },
    }


def get_corrupted_data_from_loader(train_loader):
    """Extract corrupted TS (with NaNs) from the train loader."""
    all_corrupted = []
    all_masks = []

    for data in train_loader:
        x_irregular = data[0]
        x_ts = x_irregular[:, :, :-1]
        mask = ~torch.isnan(x_ts).any(dim=-1)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())

    return np.vstack(all_corrupted), np.vstack(all_masks)


def train_csdi_on_corrupted(corrupted_data, obs_masks, seq_len, features,
                             device, csdi_epochs=200, save_dir="csdi_model"):
    """Train CSDI on corrupted data (uses only observed positions)."""
    print(f"\n{'='*60}")
    print(f"Training CSDI imputer ({csdi_epochs} epochs)")
    print(f"  Data: {len(corrupted_data)} samples, {features} features, seq_len={seq_len}")
    print(f"  Observed: {obs_masks.sum() / obs_masks.size * 100:.1f}%")
    print(f"{'='*60}")

    config = get_csdi_config(features, csdi_epochs)

    dataset = CSDIDataset(corrupted_data, obs_masks, seq_len)
    train_loader = Data.DataLoader(
        dataset, batch_size=config["train"]["batch_size"],
        shuffle=True, num_workers=0
    )

    model = CSDI_OurDataset(config, device, target_dim=features).to(device)

    os.makedirs(save_dir, exist_ok=True)
    csdi_train(model, config["train"], train_loader, foldername=save_dir)

    return model


def csdi_impute(model, corrupted_data, obs_masks, seq_len, features,
                device, nsample=1, batch_size=16):
    """Use trained CSDI to stochastically impute all missing values."""
    print(f"\n{'='*60}")
    print(f"CSDI Imputation (nsample={nsample})")
    print(f"{'='*60}")

    dataset = CSDIDataset(corrupted_data, obs_masks, seq_len)
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_imputed = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="CSDI imputing"):
            output = model.evaluate(batch, nsample)
            samples, c_target, target_mask, observed_mask, _ = output

            # samples: (B, nsample, K, L) — take median or first sample
            if nsample > 1:
                imputed = samples.median(dim=1).values  # (B, K, L)
            else:
                imputed = samples[:, 0]  # (B, K, L)

            imputed = imputed.permute(0, 2, 1).cpu()  # (B, L, K), move to CPU

            observed_data = batch["observed_data"].float()  # already CPU from DataLoader
            obs_mask_batch = batch["observed_mask"].float()

            # Merge: keep observed values, use CSDI for missing
            merged = observed_data * obs_mask_batch + imputed * (1 - obs_mask_batch)
            all_imputed.append(merged.numpy())

    imputed_data = np.vstack(all_imputed)
    imputed_data = np.clip(imputed_data, 0, 1)

    nan_count = np.isnan(imputed_data).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values remain after CSDI imputation!")
        imputed_data = np.nan_to_num(imputed_data, nan=0.5)

    print(f"Imputation complete: {len(imputed_data)} fully completed sequences.")
    return imputed_data


# =============================================================================
# Main
# =============================================================================

def main(args):
    name = create_model_name_and_dir(args)
    name = f"csdi_baseline_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)

        # Step 1: Extract corrupted data
        print("Extracting corrupted data...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader)
        seq_len = corrupted_data.shape[1]
        features = corrupted_data.shape[2]
        print(f"Extracted {len(corrupted_data)} sequences, {features} features, "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # Step 2: Train CSDI on corrupted data
        csdi_epochs = getattr(args, 'csdi_epochs', 200)
        csdi_nsample = getattr(args, 'csdi_nsample', 1)
        csdi_save = os.path.join("csdi_checkpoints", args.dataset,
                                  f"missing_{int(args.missing_rate * 100)}")

        csdi_model = train_csdi_on_corrupted(
            corrupted_data, obs_masks, seq_len, features,
            args.device, csdi_epochs=csdi_epochs, save_dir=csdi_save
        )

        # Step 3: Stochastic imputation
        completed_data = csdi_impute(
            csdi_model, corrupted_data, obs_masks, seq_len, features,
            args.device, nsample=csdi_nsample, batch_size=args.batch_size
        )

        if logger is not None:
            logger.log('csdi/imputation_complete', 1, 0)

        # Step 4: Train EDM on completed data (Phase 3 style)
        print(f"\n{'='*60}")
        print("Training EDM on CSDI-completed data")
        print(f"{'='*60}")

        final_metrics = train_unconditional_regular(
            args, completed_data, test_loader, args.device, logger,
        )

        if final_metrics:
            print("\nFinal metrics (CSDI-impute + EDM):")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")

        print("\n" + "="*60)
        print("CSDI Baseline Complete!")
        print("="*60)

        logging.info("CSDI baseline training is complete")


if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
