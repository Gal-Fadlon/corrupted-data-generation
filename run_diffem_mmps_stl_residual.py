"""
STL Residual-Only Diffusion with MMPS E-step

Decomposes time series into trend + seasonal + residual via STL.
Trend and seasonal are modelled by a lightweight statistical model
(PCA + multivariate Gaussian).  Only the residual is learned by
the diffusion model (same 28-channel TS2img_Karras as the baseline).

Key principles (learned from run_diffem_mmps_decomposed.py failure):
  - Keep 28 channels — same TS2img_Karras, no architecture change
  - Keep the standard MMPS E-step observation model
  - EMA reset every EM iteration
  - Diffusion model sees only residuals (simpler distribution)
  - Re-estimate trend+seasonal after each EM iteration

References:
    Rozet et al., "Learning Diffusion Priors from Observations by
        Expectation Maximization" (NeurIPS 2024)
    Cleveland et al., "STL: A Seasonal-Trend Decomposition Procedure
        Based on Loess" (1990)
"""

import torch
import torch.autograd
import torch.multiprocessing
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
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader, save_reconstructions
from utils.utils_args import parse_args_irregular
from utils.utils_stl import (
    auto_detect_period, fast_decompose_batch,
)
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# TrendSeasonalModel — lightweight generative model for trend + seasonal
# =============================================================================

class TrendSeasonalModel:
    """
    PCA + multivariate-Gaussian model over (trend, seasonal) vectors.

    Fit:
      1. Concatenate trend and seasonal into a single vector per sample.
      2. PCA to reduce dimensionality (retains >=95 % variance).
      3. Fit a Gaussian (mean, cov) in PCA space.

    Sample:
      1. Draw from the Gaussian in PCA space.
      2. Inverse-PCA to get (trend, seasonal) vectors.
      3. Reshape back to (n, T, C).
    """

    def __init__(self, n_components=None, variance_ratio=0.95):
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.pca = None
        self.mu_pca = None
        self.cov_pca = None
        self.seq_len = None
        self.n_features = None

    def fit(self, trends, seasonals):
        """
        Args:
            trends:    (N, T, C) numpy array
            seasonals: (N, T, C) numpy array
        """
        N, T, C = trends.shape
        self.seq_len = T
        self.n_features = C

        # Flatten: (N, 2*T*C)
        flat = np.concatenate([
            trends.reshape(N, -1),
            seasonals.reshape(N, -1),
        ], axis=1)

        max_components = self.n_components or min(N - 1, flat.shape[1])
        if self.n_components is None:
            max_components = min(max_components, 50)

        self.pca = PCA(n_components=max_components)
        scores = self.pca.fit_transform(flat)

        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, self.variance_ratio)) + 1
        k = max(1, min(k, scores.shape[1]))

        scores = scores[:, :k]
        self.pca.components_ = self.pca.components_[:k]
        self.pca.explained_variance_ = self.pca.explained_variance_[:k]
        self.pca.explained_variance_ratio_ = self.pca.explained_variance_ratio_[:k]
        self.pca.n_components_ = k
        self.pca.n_components = k

        self.mu_pca = scores.mean(axis=0)
        self.cov_pca = np.cov(scores, rowvar=False)
        if self.cov_pca.ndim == 0:
            self.cov_pca = np.array([[float(self.cov_pca)]])
        # Regularise diagonal for numerical stability with multivariate_normal
        self.cov_pca += np.eye(k) * 1e-6

        print(f"  TrendSeasonalModel: {N} samples -> PCA {k} components "
              f"({cumvar[k-1]*100:.1f}% variance)")

    def sample(self, n, rng=None):
        """
        Returns:
            trends:    (n, T, C) numpy array
            seasonals: (n, T, C) numpy array
        """
        if rng is None:
            rng = np.random.default_rng()

        scores = rng.multivariate_normal(self.mu_pca, self.cov_pca, size=n)
        flat = self.pca.inverse_transform(scores)

        tc = self.seq_len * self.n_features
        trends = flat[:, :tc].reshape(n, self.seq_len, self.n_features)
        seasonals = flat[:, tc:].reshape(n, self.seq_len, self.n_features)
        return trends, seasonals


# =============================================================================
# Conjugate Gradient Solver
# =============================================================================

def conjugate_gradient(A_fn, b, maxiter=1):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()

    for _ in range(maxiter):
        Ap = A_fn(p)
        rTr = (r * r).sum()
        pTAp = (p * Ap).sum()
        alpha = rTr / (pTAp + 1e-30)
        x = x + alpha * p
        r_new = r - alpha * Ap
        rTr_new = (r_new * r_new).sum()
        beta = rTr_new / (rTr + 1e-30)
        p = r_new + beta * p
        r = r_new

    return x


# =============================================================================
# MMPSDiffusionProcess (identical to run_diffem_mmps_ema_reset.py)
# =============================================================================

class MMPSDiffusionProcess:
    """
    MMPS-style diffusion sampler for inpainting with an unconditional model.

    Operates on residual images with the same 28-channel layout as the
    baseline.  The mask and observation model are unchanged.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01, cg_iters=1):
        self.args = args
        self.device = args.device
        self.shape = shape
        self.net = diffusion_fn.to(device=self.device)

        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

        self.sigma_y = sigma_y
        self.cg_iters = cg_iters

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        sigma_sq = sigma ** 2
        sigma_y_sq = self.sigma_y ** 2

        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return grad

        r = x_obs - mask * denoised

        def cg_operator(v):
            return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)

        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)
        score = vjp_fn(mask * v)
        return denoised.detach() + sigma_sq * score.detach()

    def mmps_sample(self, latents, x_obs_img, mask_img):
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_obs = x_obs_img.to(torch.float64)
        mask = mask_img.to(torch.float64)
        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.posterior_denoise(x_hat, t_hat, x_obs, mask)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised_2 = self.posterior_denoise(x_next, t_next, x_obs, mask)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sampling_mmps(self, x_obs_img, mask_img):
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, x_obs_img, mask_img)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        latents = torch.randn([sampling_number, *self.shape], device=self.device)

        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.net(x_hat, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


# =============================================================================
# Helper functions
# =============================================================================

def get_corrupted_data_from_loader(train_loader, device):
    all_corrupted = []
    all_masks = []

    for batch_idx, data in enumerate(train_loader):
        x_irregular = data[0]
        x_ts = x_irregular[:, :, :-1]
        mask = ~torch.isnan(x_ts).any(dim=-1)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())

    corrupted_data = np.vstack(all_corrupted)
    obs_masks = np.vstack(all_masks)
    return corrupted_data, obs_masks


def decompose_dataset(data_np, period=None):
    """
    Decompose a fully-imputed dataset into trend, seasonal, residual.

    Uses the vectorized fast path (moving-avg + periodic-avg) which
    processes the entire (N, T, C) array in one shot via scipy.

    Args:
        data_np: (N, T, C) numpy array — no NaN
        period:  STL period (auto-detected if None)

    Returns:
        trends, seasonals, residuals — each (N, T, C) numpy
    """
    return fast_decompose_batch(data_np, period=period)


# =============================================================================
# Initialization
# =============================================================================

def initialize_residuals(corrupted_data, obs_masks, period=None, seed=None):
    """
    STL-informed initialization that returns both the initial residuals
    and the trend+seasonal estimates.

    Strategy:
      1. Linear-interpolate missing values per sample (loop, but fast)
      2. Batch-decompose ALL interpolated series at once (vectorized)
      3. At observed positions: residual = observation - trend - seasonal
      4. At missing positions: sample residual from per-feature Gaussian
      5. Full reconstruction = trend + seasonal + residual

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing
        obs_masks:      (N, T) boolean, True = observed
        period:         STL period (auto-detected if None)
        seed:           random seed

    Returns:
        initial_residuals: (N, T, C) filled residuals
        trends:            (N, T, C) trend estimates
        seasonals:         (N, T, C) seasonal estimates
    """
    if seed is not None:
        np.random.seed(seed)

    N, T, C = corrupted_data.shape
    if period is None:
        period = auto_detect_period(T)

    # Step 1: Linear-interpolate missing values (loop over samples only)
    interpolated = corrupted_data.copy()
    for i in range(N):
        obs_idx = np.where(obs_masks[i])[0]
        mis_idx = np.where(~obs_masks[i])[0]
        if len(obs_idx) < 2:
            # Too few points for interp — fill with per-feature mean
            for c in range(C):
                mu = np.nanmean(corrupted_data[i, :, c])
                if np.isnan(mu):
                    mu = 0.5
                interpolated[i, mis_idx, c] = mu
        elif len(mis_idx) > 0:
            for c in range(C):
                interpolated[i, mis_idx, c] = np.interp(
                    mis_idx, obs_idx, corrupted_data[i, obs_idx, c]
                )
    interpolated = np.nan_to_num(interpolated, nan=0.5)

    # Step 2: Batch-decompose all interpolated series (vectorized, ~seconds)
    trends, seasonals, _ = fast_decompose_batch(interpolated, period=period)

    # Step 3: Compute residuals — vectorized across all samples
    obs_masks_3d = obs_masks[:, :, np.newaxis]  # (N, T, 1)
    raw_data = np.nan_to_num(corrupted_data, nan=0.0)
    ground_truth_resid = raw_data - trends - seasonals  # (N, T, C)

    # Per-feature Gaussian for missing positions
    obs_per_feature = ground_truth_resid * obs_masks_3d
    counts_per_feature = obs_masks_3d.sum(axis=(0, 1))  # (1, C)
    mu_r = obs_per_feature.sum(axis=(0, 1)) / np.maximum(counts_per_feature.ravel(), 1)
    sq_per_feature = (obs_per_feature ** 2).sum(axis=(0, 1)) / np.maximum(counts_per_feature.ravel(), 1)
    std_r = np.sqrt(np.maximum(sq_per_feature - mu_r ** 2, 1e-8))

    # Fill: observed positions get ground truth, missing get Gaussian samples
    noise = np.random.normal(0, 1, size=(N, T, C)) * std_r[np.newaxis, np.newaxis, :] + mu_r[np.newaxis, np.newaxis, :]
    residuals = np.where(obs_masks_3d, ground_truth_resid, noise)

    obs_pct = obs_masks.sum() / obs_masks.size * 100
    print(f"Residual initialization complete. {N} sequences, {obs_pct:.1f}% observed, period={period}")
    resid_range = (residuals.min(), residuals.max())
    print(f"  Residual range: [{resid_range[0]:.4f}, {resid_range[1]:.4f}]")

    return residuals, trends, seasonals


# =============================================================================
# Core EM functions
# =============================================================================

def e_step(args, uncond_model, corrupted_data, obs_masks,
           trends, seasonals, em_iter, device, logger=None):
    """
    E-step: Impute missing *residuals* using MMPS.

    At observed positions:
        residual_obs = observation - trend - seasonal
    These values anchor the MMPS posterior.  Missing positions are
    imputed by the diffusion model that has learned the residual
    distribution.

    Returns:
        residuals_imputed: (N, T, C) imputed residual time series
        reconstructions:   (N, T, C) = trend + seasonal + residual
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — MMPS on Residuals ===")
    uncond_model.eval()

    all_residuals = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = args.batch_size

    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    # Pre-stage all data on GPU once (avoids per-batch CPU→GPU transfers)
    corrupted_gpu = torch.tensor(
        np.nan_to_num(corrupted_data, nan=0.0), dtype=torch.float32, device=device
    )
    masks_gpu = torch.tensor(obs_masks, dtype=torch.float32, device=device)
    trends_gpu = torch.tensor(trends, dtype=torch.float32, device=device)
    seasons_gpu = torch.tensor(seasonals, dtype=torch.float32, device=device)

    with uncond_model.ema_scope():
        process = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step MMPS (residual)"):
            end_idx = min(start_idx + batch_size, N)

            obs_ts = corrupted_gpu[start_idx:end_idx]
            mask_ts = masks_gpu[start_idx:end_idx]
            trend_t = trends_gpu[start_idx:end_idx]
            season_t = seasons_gpu[start_idx:end_idx]

            mask_3d = mask_ts.unsqueeze(-1).expand_as(obs_ts)
            residual_obs = (obs_ts - trend_t - season_t) * mask_3d

            r_obs_img = uncond_model.ts_to_img(residual_obs)

            mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, obs_ts.shape[-1])
            mask_img = uncond_model.ts_to_img(mask_ts_expanded)
            mask_img = mask_img[:, :1, :, :]

            r_img_imputed = process.sampling_mmps(r_obs_img, mask_img)

            r_ts_imputed = uncond_model.img_to_ts(r_img_imputed)

            all_residuals.append(r_ts_imputed.cpu().numpy())

    del corrupted_gpu, masks_gpu, trends_gpu, seasons_gpu

    residuals_imputed = np.vstack(all_residuals)
    reconstructions = trends + seasonals + residuals_imputed

    print(f"  E-step complete. Imputed {len(residuals_imputed)} residual sequences.")
    nan_count = np.isnan(residuals_imputed).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in residuals!")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                             f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(residuals_imputed), em_iter)
        logger.log('em/residual_std', float(np.std(residuals_imputed)), em_iter)

    return residuals_imputed, reconstructions


def m_step(args, uncond_model, optimizer, residuals, em_iter, device, logger=None):
    """
    M-step: Train diffusion model on imputed residuals only.

    The model learns the distribution of residuals (after trend+seasonal
    removal), which is a simpler, approximately stationary distribution.
    """
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training on residuals for {args.m_step_epochs} epochs...")

    # Pre-compute ts_to_img ONCE (deterministic transform, no learned params)
    recon_tensor = torch.tensor(residuals, dtype=torch.float32)
    all_imgs = []
    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, len(recon_tensor), args.batch_size):
            batch = recon_tensor[start:start + args.batch_size].to(device)
            all_imgs.append(uncond_model.ts_to_img(batch).cpu())
    img_tensor = torch.cat(all_imgs, dim=0)

    img_dataset = Data.TensorDataset(img_tensor)
    use_workers = args.num_workers > 0
    img_loader = Data.DataLoader(
        img_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_workers,
    )

    uncond_model.train()

    for epoch in range(args.m_step_epochs):
        epoch_loss = 0
        num_batches = 0

        for (x_img,) in img_loader:
            x_img = x_img.to(device, non_blocking=True)

            loss, to_log = uncond_model.loss_fn_irregular(x_img)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  M-step epoch {epoch+1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, ts_model, test_loader,
                    em_iter, device, logger=None):
    """
    Evaluate by composing generated residuals with sampled trend+seasonal.

    1. Generate residuals unconditionally from the diffusion model.
    2. Sample trend+seasonal from the TrendSeasonalModel.
    3. Sum and clip to [0, 1].
    4. Compare against real test data.
    """
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    print("Generating: diffusion residual + PCA-Gaussian trend/seasonal...")

    uncond_model.eval()
    gen_sig = []
    real_sig = []

    rng = np.random.default_rng(seed=em_iter)

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution, args.img_resolution)
            )

            for data in tqdm(test_loader, desc="Evaluating"):
                n_samples = data[0].shape[0]

                # Generate residuals from diffusion
                r_img_sampled = process.sampling(sampling_number=n_samples)
                r_ts = uncond_model.img_to_ts(r_img_sampled).cpu().numpy()

                # Sample trend + seasonal from statistical model
                t_sampled, s_sampled = ts_model.sample(n_samples, rng=rng)

                # Compose and clip
                gen_ts = np.clip(t_sampled + s_sampled + r_ts, 0, 1)

                gen_sig.append(gen_ts)
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    print(f"EM iter {em_iter} metrics:")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

    # --- Memorization Check ---
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
# Main EM loop
# =============================================================================

def main(args):
    """Main STL Residual-Only DiffEM training loop."""

    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_stl_residual_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        # === Create unconditional model (same 28ch TS2img_Karras) ===
        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        print_model_params(logger, uncond_model)

        # === Extract corrupted data and masks ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        obs_pct = obs_masks.sum() / obs_masks.size * 100
        print(f"Extracted {len(corrupted_data)} sequences with {obs_pct:.1f}% observed")

        # Auto-detect period
        stl_period = getattr(args, 'stl_period', None)
        if stl_period is None:
            stl_period = auto_detect_period(args.seq_len)
        print(f"STL period: {stl_period}")

        # === Phase 1: STL-informed residual initialization ===
        print(f"\n{'='*60}")
        print("Phase 1: STL Decomposition + Residual Initialization")
        print(f"{'='*60}")

        initial_residuals, trends, seasonals = initialize_residuals(
            corrupted_data, obs_masks, period=stl_period, seed=args.seed
        )

        # Fit initial TrendSeasonalModel
        print("\nFitting initial TrendSeasonalModel...")
        ts_model = TrendSeasonalModel()
        ts_model.fit(trends, seasonals)

        # === Phase 1.5: Train on initial residuals ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial training on STL-extracted residuals")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_residuals, em_iter=-1, device=args.device, logger=logger
        )
        print("Initial residual model training complete.")

        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting STL-Residual DiffEM-MMPS")
        print(f"  EM iters:      {args.em_iters}")
        print(f"  MMPS sigma_y:  {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")
        print(f"  STL period:    {stl_period}")
        print(f"{'='*60}")

        best_metrics = None

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # E-step: MMPS imputation on residuals
            residuals_imputed, reconstructions = e_step(
                args, uncond_model, corrupted_data, obs_masks,
                trends, seasonals, em_iter, args.device, logger
            )

            # Re-decompose full reconstructions to get improved trend+seasonal
            # and self-consistent residuals at all positions (vectorized, ~seconds)
            print("  Re-decomposing reconstructions for updated trend+seasonal...")
            trends, seasonals, residuals_imputed = decompose_dataset(
                reconstructions, period=stl_period
            )

            # At observed positions, override with ground-truth residual
            obs_masks_3d = obs_masks[:, :, np.newaxis]  # (N, T, 1)
            gt_resid = (
                np.nan_to_num(corrupted_data, nan=0.0) - trends - seasonals
            )
            residuals_imputed = np.where(obs_masks_3d, gt_resid, residuals_imputed)

            # Update TrendSeasonalModel
            ts_model.fit(trends, seasonals)

            # Reset EMA for fresh M-step
            uncond_model.reset_ema()

            # M-step: Train on residuals
            uncond_model = m_step(
                args, uncond_model, optimizer,
                residuals_imputed, em_iter, args.device, logger
            )

            # Evaluate periodically
            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, uncond_model, ts_model, test_loader,
                    em_iter, args.device, logger
                )

                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        print("\n" + "=" * 60)
        print("STL-Residual DiffEM-MMPS Complete!")
        print("=" * 60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("STL-Residual DiffEM-MMPS training is complete")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
