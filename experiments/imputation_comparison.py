"""
Imputation-quality comparison experiment.

Runs a SINGLE imputation (one method × one dataset × one missing rate) and
reports MSE and MAE over the originally-missing positions.

Methods:
    tst  — TSTransformerEncoder + TST_Decoder, trained from scratch on
           self-supervised MSE-over-observed-positions. (Same structure as
           run_irregular.py's Phase 1 TST pretraining.)
    ncde — NeuralCDE + cubic spline coefficients (cached under
           ncde_spline_cache_dir), trained from scratch on MSE-over-observed.
    our  — Loads the best saved diffusion checkpoint from
           saved_models/our/missing_observation/fix_missing_rates/{dataset}/
           seq_len_{N}/missing_{P}/{best_run_name}/checkpoint.pt and runs the
           E-step (MMPS posterior sampling) on the corrupted data.

Corruption is seeded (56789 in real_data_loading), so all three methods see
identical NaN positions for the same (dataset, missing_rate).

Metrics: MSE and MAE are computed only over positions where M==1 (originally
missing). The known ground truth is the clean data returned by
real_data_loading.

Usage:
    python experiments/imputation_comparison.py \\
        --config ./configs/seq_len_24/stock.yaml \\
        --dataset stock --missing_rate 0.3 --impute_method tst \\
        --impute_epochs 200 --wandb true \\
        --tags "imputation_comparison" "tst" "stock" "30_missing_rate" "fix_missing_rates"
"""

import json
import logging
import os
import sys
import time
import traceback
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch import optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loggers import CompositeLogger, WandbLogger, PrintLogger
from utils.utils import create_model_name_and_dir, log_config_and_tags
from utils.utils_args import parse_args_irregular
from utils.utils_data import real_data_loading
from utils.gpu_heartbeat import start_gpu_heartbeat


# =============================================================================
# Shared helpers
# =============================================================================

def _chmod_777(path):
    try:
        os.chmod(path, 0o777)
    except Exception:
        pass


def _makedirs_777(path):
    os.makedirs(path, exist_ok=True)
    parts = path.split(os.sep)
    for i in range(1, len(parts) + 1):
        sub = os.sep.join(parts[:i]) or os.sep
        if os.path.isdir(sub):
            _chmod_777(sub)


def _propagate_fully_nan_samples(tensor):
    """If an entire sample is NaN, replace it with its neighbor (used by TST preprocess)."""
    for b in range(tensor.size(0)):
        if torch.isnan(tensor[b]).all():
            if b + 1 < tensor.size(0):
                tensor[b] = tensor[b + 1]
            else:
                tensor[b] = tensor[b - 1]
    return tensor


# =============================================================================
# Metric
# =============================================================================

def compute_imputation_metrics(x_hat, x_true, missing_mask):
    """MSE and MAE, averaged only over originally-missing positions.

    Args:
        x_hat:          (N, T, F) — imputer's full output.
        x_true:         (N, T, F) — ground-truth clean data.
        missing_mask:   (N, T) binary float — 1.0 at originally missing timesteps.
    """
    F_dim = x_true.shape[-1]
    # Broadcast missing_mask to (N, T, F): every feature at a missing timestep is missing.
    M = missing_mask[:, :, None] * np.ones((1, 1, F_dim), dtype=np.float32)
    diff = x_hat - x_true
    denom = float(M.sum())
    mse = float(((diff ** 2) * M).sum() / denom)
    mae = float((np.abs(diff) * M).sum() / denom)
    return mse, mae, int(denom)


def compute_kl_divergence(x_hat, x_true, missing_mask, n_bins=50, eps=1e-10):
    """Histogram-based KL(P_real || P_imputed) at originally-missing positions.

    Returns
    -------
    kl_total : float
        KL on the flattened set of missing values (all features together).
    kl_mean_per_feature : float
        Mean over per-feature KLs (one KL per feature dim).

    Notes
    -----
    - Only values at positions where missing_mask == 1 are used — same subset as
      MSE/MAE. Gives an honest "does imputed distribution match the real distribution
      over the missing positions?" measure.
    - Histogram bins are built over the union range [min, max] of real + imputed
      for each computation, using `n_bins` equal-width bins.
    - `eps` is added to both distributions to avoid log(0) / div-by-zero blowups.
    - KL is asymmetric; we compute KL(real || imputed) — penalises imputed for
      missing a mode that exists in real.
    """
    mask_bool = missing_mask.astype(bool)            # (N, T)
    if not mask_bool.any():
        return float('nan'), float('nan')

    # (n_missing_timesteps, F)
    real_at_missing = x_true[mask_bool]
    imp_at_missing = x_hat[mask_bool]
    F_dim = real_at_missing.shape[-1]

    def _hist_kl(real_vals, imp_vals):
        real_vals = np.asarray(real_vals).ravel()
        imp_vals = np.asarray(imp_vals).ravel()
        if real_vals.size == 0 or imp_vals.size == 0:
            return float('nan')
        lo = float(min(real_vals.min(), imp_vals.min()))
        hi = float(max(real_vals.max(), imp_vals.max()))
        if hi <= lo:
            return 0.0
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(real_vals, bins=bins, density=False)
        q, _ = np.histogram(imp_vals, bins=bins, density=False)
        p = p.astype(np.float64)
        q = q.astype(np.float64)
        # Normalise to probability mass functions
        p = (p / max(p.sum(), 1.0)) + eps
        q = (q / max(q.sum(), 1.0)) + eps
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    # Per-feature KLs
    per_feat = []
    for f in range(F_dim):
        kl_f = _hist_kl(real_at_missing[:, f], imp_at_missing[:, f])
        if not np.isnan(kl_f):
            per_feat.append(kl_f)
    kl_mean_per_feature = float(np.mean(per_feat)) if per_feat else float('nan')

    # Overall KL on flattened values
    kl_total = _hist_kl(real_at_missing.ravel(), imp_at_missing.ravel())

    return kl_total, kl_mean_per_feature


# =============================================================================
# TST imputer
# =============================================================================

def impute_with_tst(X_true, X_corrupted, args, logger):
    """Train a TST encoder+decoder from scratch on MSE-over-observed; return full imputed output.

    Mirrors the Phase 1 TST pretraining in run_irregular.py.
    """
    from models.TST import TSTransformerEncoder
    from models.decoder import TST_Decoder

    device = args.device
    N, T, Fdim = X_corrupted.shape

    embedder = TSTransformerEncoder(
        feat_dim=Fdim,
        max_len=T,
        d_model=args.hidden_dim,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pos_encoding=args.pos_encoding,
        activation=args.activation,
        norm=args.norm,
        freeze=args.freeze,
    ).to(device)
    decoder = TST_Decoder(
        inp_dim=args.hidden_dim,
        hidden_dim=int(args.hidden_dim + (Fdim - args.hidden_dim) / 2),
        layers=3,
        args=args,
    ).to(device)
    opt = optim.Adam(chain(embedder.parameters(), decoder.parameters()))
    embedder.train()
    decoder.train()

    # Prep data: keep NaN, handle fully-NaN samples by propagation (rare)
    x_all = torch.tensor(X_corrupted, dtype=torch.float32, device=device)
    x_all = _propagate_fully_nan_samples(x_all)

    dataset = Data.TensorDataset(x_all)
    loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    last_loss_value = None
    for epoch in range(args.impute_epochs):
        epoch_losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            padding_masks = ~torch.isnan(xb).any(dim=-1)
            h = embedder(xb, padding_masks)
            x_tilde = decoder(h)
            nan_mask = torch.isnan(xb)
            if (~nan_mask).any():
                loss = F.mse_loss(x_tilde[~nan_mask], xb[~nan_mask])
            else:
                continue
            loss_train = torch.sqrt(loss) * 10  # same rooted-scaled loss as the original TST pretraining
            opt.zero_grad()
            loss_train.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        if epoch_losses:
            avg = float(np.mean(epoch_losses))
            last_loss_value = avg
            if logger is not None:
                logger.log('imputation/tst_loss', avg, epoch)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  [tst] epoch {epoch + 1}/{args.impute_epochs}: mse_observed={avg:.6f}")

    # Inference: produce full imputation (use filled-with-zero input so forward doesn't see NaN)
    embedder.eval()
    decoder.eval()
    x_hat_chunks = []
    with torch.no_grad():
        eval_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for (xb,) in eval_loader:
            xb = xb.to(device)
            padding_masks = ~torch.isnan(xb).any(dim=-1)
            h = embedder(xb, padding_masks)
            x_tilde = decoder(h)
            x_hat_chunks.append(x_tilde.detach().cpu().numpy())
    x_hat = np.vstack(x_hat_chunks)
    return x_hat, last_loss_value


# =============================================================================
# NCDE imputer
# =============================================================================

def _ncde_spline_cache_path(cache_dir, dataset, seq_len, missing_rate, missing_type, corruption_type):
    sub = os.path.join(cache_dir, str(dataset), f"seq_len_{seq_len}",
                       f"{corruption_type}__{missing_type}__missing_{int(missing_rate * 100)}")
    return sub


def _load_or_compute_splines(X_corrupted, args):
    """Compute natural cubic spline coefficients on X_corrupted. Cache under args.ncde_spline_cache_dir if set."""
    import controldiffeq

    cache_dir = getattr(args, 'ncde_spline_cache_dir', None)
    device = args.device
    T = X_corrupted.shape[1]
    time_axis = torch.arange(T, dtype=torch.float32, device=device)

    if cache_dir:
        sub = _ncde_spline_cache_path(cache_dir, args.dataset, args.seq_len, args.missing_rate,
                                      getattr(args, 'missing_type', 'fix_missing_rates'),
                                      getattr(args, 'corruption_type', 'missing_observation'))
        paths = [os.path.join(sub, f"coeff_{i}.pt") for i in range(4)]
        if all(os.path.isfile(p) for p in paths):
            print(f"[ncde] Loading cached splines from {sub}")
            coeffs = tuple(torch.load(p, map_location=device) for p in paths)
            return coeffs, time_axis

        print(f"[ncde] Computing splines (cache miss at {sub})")
        X_t = torch.tensor(X_corrupted, dtype=torch.float32, device=device)
        coeffs = controldiffeq.natural_cubic_spline_coeffs(time_axis, X_t)

        _makedirs_777(sub)
        for p, c in zip(paths, coeffs):
            torch.save(c.detach().cpu(), p)
            _chmod_777(p)
        print(f"[ncde] Saved 4 spline tensors to {sub}")
        return coeffs, time_axis

    # No cache
    X_t = torch.tensor(X_corrupted, dtype=torch.float32, device=device)
    coeffs = controldiffeq.natural_cubic_spline_coeffs(time_axis, X_t)
    return coeffs, time_axis


def impute_with_ncde(X_true, X_corrupted, args, logger):
    """Train a NeuralCDE from scratch on MSE-over-observed with NCDE's built-in linear head
    producing feature-dim outputs directly."""
    from models.kovae_irregular import FinalTanh
    from models.neuralCDE import NeuralCDE

    device = args.device
    N, T, Fdim = X_corrupted.shape

    hidden_dim_ncde = getattr(args, 'hidden_dim_ncde', 20)
    num_layers = getattr(args, 'num_layers', 3)

    # ---- Splines (cached) ----
    coeffs_full, time_axis = _load_or_compute_splines(X_corrupted, args)

    # Per-sample slice of coeffs
    def slice_coeffs(indices):
        return tuple(c[indices].to(device) for c in coeffs_full)

    # ---- Model: NCDE maps (time, coeffs) → per-time embedding, and its internal linear
    # head projects to output_channels = Fdim, which gives us the imputed sequence directly.
    ode_func = FinalTanh(Fdim, hidden_dim_ncde, hidden_dim_ncde, num_layers).to(device)
    encoder = NeuralCDE(
        func=ode_func, input_channels=Fdim,
        hidden_channels=hidden_dim_ncde, output_channels=Fdim,
    ).to(device)
    opt = optim.Adam(encoder.parameters())
    encoder.train()

    X_t = torch.tensor(X_corrupted, dtype=torch.float32, device=device)
    indices_all = torch.arange(N, device=device)

    last_loss_value = None
    batch_size = args.batch_size
    for epoch in range(args.impute_epochs):
        perm = torch.randperm(N, device=device)
        epoch_losses = []
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            coeffs_b = slice_coeffs(idx)
            final_idx = (torch.ones(len(idx), device=device) * (T - 1)).float()
            x_hat_b = encoder(time_axis, coeffs_b, final_idx, stream=True)
            x_true_b = X_t[idx]
            nan_mask = torch.isnan(x_true_b)
            if (~nan_mask).all():
                loss = F.mse_loss(x_hat_b, x_true_b)
            elif (~nan_mask).any():
                loss = F.mse_loss(x_hat_b[~nan_mask], x_true_b[~nan_mask])
            else:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        if epoch_losses:
            avg = float(np.mean(epoch_losses))
            last_loss_value = avg
            if logger is not None:
                logger.log('imputation/ncde_loss', avg, epoch)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  [ncde] epoch {epoch + 1}/{args.impute_epochs}: mse_observed={avg:.6f}")

    # ---- Inference: produce full imputation ----
    encoder.eval()
    x_hat_chunks = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            idx = indices_all[start:start + batch_size]
            coeffs_b = slice_coeffs(idx)
            final_idx = (torch.ones(len(idx), device=device) * (T - 1)).float()
            x_hat_b = encoder(time_axis, coeffs_b, final_idx, stream=True)
            x_hat_chunks.append(x_hat_b.detach().cpu().numpy())
    x_hat = np.vstack(x_hat_chunks)
    return x_hat, last_loss_value


# =============================================================================
# CSDI imputer
# =============================================================================

def _load_csdi_modules():
    """Load CSDI modules via importlib so our local `utils/` package isn't shadowed
    by CSDI's internal `utils.py`. Returns (CSDI_base class,)."""
    import importlib.util
    csdi_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CSDI')
    if not os.path.isdir(csdi_dir):
        raise RuntimeError(
            f"CSDI directory not found at {csdi_dir}. "
            "Make sure the CSDI repo is cloned into the project root as 'CSDI/'.")

    sys.path.insert(0, csdi_dir)

    def _load(name, filepath):
        spec = importlib.util.spec_from_file_location(f"csdi_internal.{name}", filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"csdi_internal.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        diff_models = _load('diff_models', os.path.join(csdi_dir, 'diff_models.py'))
        sys.modules['diff_models'] = diff_models  # CSDI main_model.py does: from diff_models import ...
        main_mod = _load('main_model', os.path.join(csdi_dir, 'main_model.py'))
    finally:
        if csdi_dir in sys.path:
            sys.path.remove(csdi_dir)

    return main_mod.CSDI_base


def _csdi_config(csdi_epochs):
    return {
        "train": {"epochs": csdi_epochs, "batch_size": 16, "lr": 1e-3, "itr_per_epoch": int(1e8)},
        "diffusion": {
            "layers": 4, "channels": 64, "nheads": 8,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001, "beta_end": 0.5, "num_steps": 50,
            "schedule": "quad", "is_linear": False,
        },
        "model": {
            "is_unconditional": 0, "timeemb": 128, "featureemb": 16,
            "target_strategy": "random",
        },
    }


class _CSDIOurDataset:
    """CSDI subclass adapter — needs to be created lazily because CSDI_base is loaded at runtime."""
    pass


def _build_csdi_model(CSDI_base, config, device, target_dim):
    """Create a subclass of CSDI_base with the correct process_data method for our dataset layout."""

    class CSDI_OurDataset(CSDI_base):
        def __init__(self, config_, device_, target_dim_):
            super().__init__(target_dim_, config_, device_)

        def process_data(self, batch):
            observed_data = batch["observed_data"].to(self.device).float()
            observed_mask = batch["observed_mask"].to(self.device).float()
            observed_tp = batch["timepoints"].to(self.device).float()
            gt_mask = batch["gt_mask"].to(self.device).float()

            observed_data = observed_data.permute(0, 2, 1)   # (B, K, L)
            observed_mask = observed_mask.permute(0, 2, 1)
            gt_mask = gt_mask.permute(0, 2, 1)

            cut_length = torch.zeros(len(observed_data)).long().to(self.device)
            for_pattern_mask = observed_mask

            return (
                observed_data, observed_mask, observed_tp,
                gt_mask, for_pattern_mask, cut_length,
            )

    return CSDI_OurDataset(config, device, target_dim).to(device)


class _CSDIDataset(torch.utils.data.Dataset):
    """Wraps our corrupted data into CSDI's expected dict format."""
    def __init__(self, corrupted_data, obs_masks, seq_len):
        self.N = len(corrupted_data)
        self.seq_len = seq_len
        self.features = corrupted_data.shape[-1]

        clean = np.nan_to_num(corrupted_data, nan=0.0)
        self.observed_data = torch.tensor(clean, dtype=torch.float32)

        # Broadcast timestep mask (N, T) to feature-level (N, T, F)
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


def impute_with_csdi(X_true, X_corrupted, args, logger):
    """Train CSDI from scratch on the corrupted data for args.impute_epochs,
    then stochastically impute missing values. Returns (x_hat, final_loss).

    Training uses CSDI's own loss: the model's forward(batch) returns a scalar
    diffusion loss (mean over the batch) — standard CSDI training convention.
    """
    device = args.device
    N, T, Fdim = X_corrupted.shape

    # ---- Load CSDI modules and build model ----
    CSDI_base = _load_csdi_modules()
    config = _csdi_config(args.impute_epochs)

    # Build data loader
    dataset = _CSDIDataset(X_corrupted, ~np.isnan(X_corrupted).any(axis=-1), T)
    train_loader = Data.DataLoader(dataset, batch_size=config["train"]["batch_size"],
                                    shuffle=True, num_workers=0)

    model = _build_csdi_model(CSDI_base, config, device, target_dim=Fdim)
    opt = optim.Adam(model.parameters(), lr=config["train"]["lr"])

    # ---- Custom training loop (for per-epoch loss logging) ----
    last_loss_value = None
    for epoch in range(args.impute_epochs):
        model.train()
        batch_losses = []
        for batch in train_loader:
            opt.zero_grad()
            loss = model(batch)
            if torch.is_tensor(loss) and loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.item()))
        if batch_losses:
            avg = float(np.mean(batch_losses))
            last_loss_value = avg
            if logger is not None:
                logger.log('imputation/csdi_loss', avg, epoch)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  [csdi] epoch {epoch + 1}/{args.impute_epochs}: mean_loss={avg:.6f}")

    # ---- Stochastic imputation on the same corrupted data ----
    model.eval()
    nsample = getattr(args, 'csdi_nsample', 1)
    eval_loader = Data.DataLoader(dataset, batch_size=config["train"]["batch_size"],
                                   shuffle=False, num_workers=0)

    all_imputed = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="[csdi] imputing"):
            samples, c_target, target_mask, observed_mask, _ = model.evaluate(batch, nsample)
            # samples: (B, nsample, K, L)
            imputed = samples.median(dim=1).values if nsample > 1 else samples[:, 0]
            imputed = imputed.permute(0, 2, 1).cpu()  # (B, L, K)

            observed = batch["observed_data"].float()
            obs_mask = batch["observed_mask"].float()

            # Keep observed values, fill only missing with CSDI output
            merged = observed * obs_mask + imputed * (1.0 - obs_mask)
            all_imputed.append(merged.numpy())

    x_hat = np.vstack(all_imputed).astype(np.float32)
    return x_hat, last_loss_value


# =============================================================================
# 'our' E-step imputer (load best checkpoint, run MMPS)
# =============================================================================

def _find_our_checkpoint(args):
    """Locate the single saved-best checkpoint under
    saved_models/our/missing_observation/fix_missing_rates/{dataset}/seq_len_{N}/missing_{P}/
    There should be exactly one run dir (best-so-far save rule). Returns the checkpoint path.
    """
    root = getattr(args, 'save_models_root', None)
    if not root:
        raise RuntimeError("--save_models_root not set; cannot locate 'our' checkpoint.")

    sub = os.path.join(
        root, 'our',
        getattr(args, 'corruption_type', 'missing_observation'),
        getattr(args, 'missing_type', 'fix_missing_rates'),
        args.dataset,
        f"seq_len_{args.seq_len}",
        f"missing_{int(args.missing_rate * 100)}",
    )
    if not os.path.isdir(sub):
        raise FileNotFoundError(f"'our' checkpoint parent dir does not exist: {sub}")

    candidates = []
    for name in os.listdir(sub):
        if not name.startswith("disc_"):
            continue
        ckpt_path = os.path.join(sub, name, "checkpoint.pt")
        if os.path.isfile(ckpt_path):
            # Parse the disc value from the dir name to pick the best if multiple exist.
            try:
                disc = float(name[len("disc_"):].split("_", 1)[0])
            except Exception:
                disc = float("inf")
            candidates.append((disc, ckpt_path))
    if not candidates:
        raise FileNotFoundError(f"No 'disc_*' run dir with checkpoint.pt under {sub}")
    candidates.sort()
    return candidates[0][1]


def impute_with_our_e_step(X_true, X_corrupted, args, logger):
    """Load the best 'our' checkpoint and run its E-step (MMPS posterior sampling) on the
    same corrupted data that was handed to TST/NCDE.
    """
    import argparse

    # Import model + e_step from the co-evolving EM runner.
    from models.our import TS2img_Karras
    from run_co_evolving_em import e_step as coeom_e_step

    device = args.device

    # ---- Locate + load checkpoint ----
    ckpt_path = _find_our_checkpoint(args)
    print(f"[our] Loading checkpoint: {ckpt_path}")
    # weights_only=False: the checkpoint we saved contains args dict + numpy RNG state,
    # which PyTorch 2.6+ refuses to load under the default weights_only=True. We trust
    # the source (we wrote the file), so bypass the restriction.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Rebuild args from checkpoint (so the model + E-step config match what was trained)
    model_args_dict = dict(ckpt.get('args', {}))
    # Preserve runtime-only overrides we need from the current invocation
    model_args_dict['device'] = device
    # Redirect recon_cache_dir to a temp location — e_step side-effect writes go here.
    model_args_dict['recon_cache_dir'] = os.path.join(
        getattr(args, 'impute_results_dir', '/cs/azencot_fsas/gal_and_idan/corrupted-data-generation/experiments/imputation_results'),
        '_tmp_recon_our', args.dataset, f"missing_{int(args.missing_rate * 100)}",
    )
    _makedirs_777(model_args_dict['recon_cache_dir'])
    model_args = argparse.Namespace(**model_args_dict)

    # ---- Rebuild diffusion model and load weights ----
    model = TS2img_Karras(args=model_args, device=device).to(device)

    # e_step does `with uncond_model.ema_scope():`, which copies model.model_ema's
    # shadow into model.net. So we need model.model_ema to hold the correct EMA
    # weights — NOT just load them into model.net (that would get clobbered).
    # Strategy:
    #   1. Load net_raw into model.net (resume position).
    #   2. Load net_ema into model.net temporarily, call model_ema.reset(model.net)
    #      so the shadow buffers now hold the EMA weights.
    #   3. Restore net_raw into model.net so post-scope behavior is correct.
    # When e_step enters ema_scope(): it stores current net (=raw), copies shadow
    # (=EMA) into net, and E-step runs on EMA — which is what we want.
    if not ckpt.get('net_raw'):
        raise RuntimeError("Checkpoint missing net_raw; cannot load 'our' model.")
    model.net.load_state_dict(ckpt['net_raw'])

    if getattr(model, 'use_ema', False) and ckpt.get('net_ema'):
        # Push EMA weights through net → reset → back to raw.
        model.net.load_state_dict(ckpt['net_ema'])
        model.model_ema.reset(model.net)                       # shadow := EMA weights
        model.net.load_state_dict(ckpt['net_raw'])             # net   := raw weights
        print("[our] Loaded raw into net and EMA into model_ema.shadow")
    elif ckpt.get('net_ema') and not getattr(model, 'use_ema', False):
        # Model was rebuilt without EMA but ckpt has ema weights → use them directly.
        model.net.load_state_dict(ckpt['net_ema'])
        print("[our] No use_ema; loaded EMA weights directly into model.net")
    else:
        print("[our] No EMA weights available; using raw weights in model.net")

    model.eval()

    # ---- Build obs_masks (True where observed, False where originally missing) ----
    obs_masks = ~np.isnan(X_corrupted).any(axis=-1)  # (N, T) bool

    # ---- Run E-step (MMPS) on the shared corrupted data ----
    reconstructions = coeom_e_step(
        model_args, model, X_corrupted, obs_masks,
        em_iter=0, device=device, logger=None,   # keep wandb logs from coeom e_step out of this run
    )
    return reconstructions, None  # no training loss for this method


# =============================================================================
# Main
# =============================================================================

def _persist_results(args, mse, mae, n_missing, final_train_loss,
                     kl_total=None, kl_mean_per_feature=None):
    out_dir = getattr(args, 'impute_results_dir', '/cs/azencot_fsas/gal_and_idan/corrupted-data-generation/experiments/imputation_results')
    _makedirs_777(out_dir)
    fname = f"{args.impute_method}_{args.dataset}_missing_{int(args.missing_rate * 100)}.json"
    out_path = os.path.join(out_dir, fname)
    payload = {
        'impute_method': args.impute_method,
        'dataset': args.dataset,
        'missing_rate': args.missing_rate,
        'missing_type': getattr(args, 'missing_type', 'fix_missing_rates'),
        'seq_len': args.seq_len,
        'impute_epochs': getattr(args, 'impute_epochs', None),
        'seed': args.seed,
        'mse': mse,
        'mae': mae,
        'kl_total': kl_total,
        'kl_mean_per_feature': kl_mean_per_feature,
        'n_missing_points': n_missing,
        'final_train_loss': final_train_loss,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    _chmod_777(out_path)
    print(f"[impute] results written to {out_path}")


def _run_impute_dispatch(impute_method, X_true, X_corrupted, args, logger):
    """Call the selected imputer; catch + print any exception verbatim to stdout
    (so wandb's captured stdout has the full traceback), then re-raise."""
    try:
        if impute_method == 'tst':
            return impute_with_tst(X_true, X_corrupted, args, logger)
        elif impute_method == 'ncde':
            return impute_with_ncde(X_true, X_corrupted, args, logger)
        elif impute_method == 'our':
            return impute_with_our_e_step(X_true, X_corrupted, args, logger)
        elif impute_method == 'csdi':
            return impute_with_csdi(X_true, X_corrupted, args, logger)
        else:
            raise ValueError(f"Unknown --impute_method: {impute_method}")
    except Exception:
        print(f"\n[impute][FATAL] Imputer '{impute_method}' raised an exception:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # GPU heartbeat — prevents cluster idle-GPU monitors from killing the job
    # during long CPU-bound phases (spline computation, data prep, etc.).
    start_gpu_heartbeat()

    name = create_model_name_and_dir(args)
    name = f"impute_{args.impute_method}_{name}"

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        # ---- Load shared corrupted data (same seed → identical NaN positions across methods) ----
        try:
            print(f"[impute] Loading data: {args.dataset} seq_len={args.seq_len} missing={args.missing_rate}")
            ori_list, irregular_list = real_data_loading(
                args.dataset, args.seq_len,
                missing_rate=args.missing_rate,
                gaussian_noise_level=getattr(args, 'gaussian_noise_level', 0.0),
                noise_timestep=getattr(args, 'noise_timestep', None),
                corruption_type=getattr(args, 'corruption_type', 'missing_observation'),
                missing_type=getattr(args, 'missing_type', 'fix_missing_rates'),
            )
            X_true = np.asarray(ori_list, dtype=np.float32)                       # (N, T, F)
            X_corrupted = np.asarray(irregular_list, dtype=np.float32)[:, :, :-1]  # (N, T, F) with NaN; drop idx col

            assert X_true.shape == X_corrupted.shape, \
                f"shape mismatch: clean {X_true.shape} vs corrupted {X_corrupted.shape}"

            # Mask of originally-missing timesteps (1 at a timestep where any feature is NaN).
            missing_mask = np.isnan(X_corrupted).any(axis=-1).astype(np.float32)  # (N, T)
            n_missing = int(missing_mask.sum())
            n_total_timesteps = int(missing_mask.size)
            print(f"[impute] N={X_true.shape[0]}, T={X_true.shape[1]}, F={X_true.shape[2]}; "
                  f"missing_timesteps={n_missing}/{n_total_timesteps} "
                  f"({100 * n_missing / n_total_timesteps:.1f}%)")

            args.input_size = X_true.shape[-1]
            args.num_features = X_true.shape[-1]
            # TST_Decoder's output layer uses args.input_channels; keep in sync with data.
            args.input_channels = X_true.shape[-1]
        except Exception:
            print("\n[impute][FATAL] Data loading failed:", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise

        # ---- Run the requested imputer (wrapped in try/except inside _run_impute_dispatch) ----
        t0 = time.time()
        x_hat, final_loss = _run_impute_dispatch(args.impute_method, X_true, X_corrupted, args, logger)
        wall_clock_minutes = (time.time() - t0) / 60.0
        print(f"[impute] {args.impute_method} done in {wall_clock_minutes:.2f} min")

        # ---- Validate + score ----
        try:
            assert x_hat.shape == X_true.shape, \
                f"imputer returned shape {x_hat.shape}, expected {X_true.shape}"

            # Sanitize (rare NaN/Inf in imputed output → fail loudly)
            n_bad = int(np.isnan(x_hat).sum()) + int(np.isinf(x_hat).sum())
            if n_bad > 0:
                raise RuntimeError(
                    f"imputer output contains {n_bad} NaN/Inf values — MSE/MAE would be invalid.")

            # Compute MSE / MAE at missing positions only
            mse, mae, n_eval_pts = compute_imputation_metrics(x_hat, X_true, missing_mask)
            # KL divergence between real and imputed distributions at missing positions.
            kl_total, kl_mean_per_feature = compute_kl_divergence(x_hat, X_true, missing_mask)
            print(f"[impute] RESULTS for {args.impute_method} / {args.dataset} / "
                  f"missing={args.missing_rate:.2f}: mse={mse:.6f}, mae={mae:.6f}, "
                  f"kl_total={kl_total:.6f}, kl_mean_per_feature={kl_mean_per_feature:.6f}, "
                  f"n_missing_points={n_eval_pts}")

            # Log to wandb + persist
            if logger is not None:
                logger.log('imputation/mse', mse, 0)
                logger.log('imputation/mae', mae, 0)
                logger.log('imputation/kl_total', kl_total, 0)
                logger.log('imputation/kl_mean_per_feature', kl_mean_per_feature, 0)
                logger.log('imputation/n_missing_points', n_eval_pts, 0)
                logger.log('imputation/wall_clock_minutes', wall_clock_minutes, 0)
                if final_loss is not None:
                    logger.log('imputation/final_train_loss', final_loss, 0)

            _persist_results(args, mse, mae, n_eval_pts, final_loss,
                             kl_total=kl_total, kl_mean_per_feature=kl_mean_per_feature)
        except Exception:
            print("\n[impute][FATAL] Post-imputation scoring/logging failed:", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise


if __name__ == '__main__':
    # Top-level guard: print full traceback to stdout so the wandb-captured stdout
    # always shows the failure. Re-raises so the process exit code stays non-zero.
    try:
        args = parse_args_irregular()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        main(args)
    except SystemExit:
        raise
    except BaseException:
        print("\n[impute][FATAL] Top-level uncaught exception:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
