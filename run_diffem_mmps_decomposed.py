"""
Decomposition-Enhanced DiffEM with MMPS E-step

Extends run_diffem_mmps.py by leveraging the time-series-specific
decomposition x = trend + seasonal + residual.

Three key enhancements over the base MMPS variant:

1. **Multi-channel decomposition representation**
   The UNet receives (B, 3*C, H, W) images where three channel groups
   correspond to trend, seasonal, and residual.  This lets the UNet learn
   component-specific features: smooth trends, periodic seasonals, and
   stochastic residuals.

2. **STL-informed initialisation**
   Instead of filling missing values with a marginal Gaussian, we linearly
   interpolate each feature, STL-decompose the result, keep the interpolated
   trend+seasonal, and only re-sample the residual at missing positions.

3. **Component-aware loss**
   An optional total-variation penalty on the trend channels and a spectral
   matching penalty on the seasonal channels guide the UNet to preserve the
   frequency structure of the data.

References:
    Rozet et al., "Learning Diffusion Priors from Observations by
        Expectation Maximization" (NeurIPS 2024)
    Yuan & Qiao, "Diffusion-TS: Interpretable Diffusion for General
        Time Series Generation" (ICLR 2024)
    Ma & Huang, "STDiffusion: Effective Series Decomposition and
        Components Learning for Time Series Generation" (2025)
    Caldas et al., "FFT-DM: A Decomposable Forward Process in Diffusion
        Models for Time-Series Forecasting" (ICLR 2026)
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

from metrics import evaluate_model_irregular
from metrics.memorization import compute_memorization_metric
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader, save_reconstructions
from utils.utils_args import parse_args_irregular
from utils.utils_stl import stl_decompose_batch, initialize_with_stl
from models.our import TS2img_Karras_Decomposed
from utils.train_unconditional import train_unconditional_regular

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


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
# MMPS Diffusion Process (works with decomposed channels)
# =============================================================================

class MMPSDiffusionProcess:
    """
    MMPS-style sampler identical to run_diffem_mmps.py but operating on
    the decomposed (B, 3*C, H, W) representation.  The mask is replicated
    across the three channel groups so that the posterior correction is
    applied consistently to trend, seasonal, and residual.
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
    all_corrupted, all_masks = [], []
    for data in train_loader:
        x_irregular = data[0]
        x_ts = x_irregular[:, :, :-1]
        mask = ~torch.isnan(x_ts).any(dim=-1)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())
    return np.vstack(all_corrupted), np.vstack(all_masks)


# =============================================================================
# Decomposed E-step
# =============================================================================

def e_step(args, model, corrupted_data, obs_masks, prev_recons,
           em_iter, device, logger=None):
    """
    E-step: MMPS imputation on the decomposed representation.

    prev_recons are the complete (NaN-free) reconstructions from the
    previous M-step (or from STL initialisation).  We decompose them,
    adjust the residual at observed positions to be consistent with the
    actual observations, and run MMPS on the stacked image.
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — Decomposed MMPS ===")
    model.eval()

    C = model.original_channels
    N = len(corrupted_data)
    batch_size = args.batch_size
    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    target_shape = (C * model.N_COMPONENTS, args.img_resolution, args.img_resolution)

    # Decompose previous reconstructions (full dataset, CPU)
    print("  Decomposing previous reconstructions via STL ...")
    trends_np, seasonals_np, residuals_np = stl_decompose_batch(
        prev_recons, period=model.stl_period
    )

    all_reconstructions = []

    with model.ema_scope():
        process = MMPSDiffusionProcess(
            args, model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters,
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step MMPS"):
            end_idx = min(start_idx + batch_size, N)

            corrupted_batch = corrupted_data[start_idx:end_idx]    # (B,T,C) w/ NaN
            mask_batch = obs_masks[start_idx:end_idx]              # (B,T) bool

            trend_batch = trends_np[start_idx:end_idx]
            season_batch = seasonals_np[start_idx:end_idx]
            resid_batch = residuals_np[start_idx:end_idx]

            corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
            mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)

            obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)

            trend_t = torch.tensor(trend_batch, dtype=torch.float32, device=device)
            season_t = torch.tensor(season_batch, dtype=torch.float32, device=device)

            # At observed positions the residual must satisfy:
            #   raw_obs = trend + seasonal + residual  →  residual = raw - trend - seasonal
            # At missing positions we keep the previous residual.
            adjusted_residual = obs_ts - trend_t - season_t
            resid_prev = torch.tensor(resid_batch, dtype=torch.float32, device=device)
            mask_3d = mask_ts.unsqueeze(-1).expand_as(corrupted_ts)
            residual_t = torch.where(mask_3d.bool(), adjusted_residual, resid_prev)

            # Zero out missing positions for the "observed" signal
            trend_obs = trend_t * mask_3d
            season_obs = season_t * mask_3d
            resid_obs = residual_t * mask_3d

            # Convert each component to image and stack
            trend_img = model.ts_to_img(trend_obs)
            season_img = model.ts_to_img(season_obs)
            resid_img = model.ts_to_img(resid_obs)
            x_obs_img = torch.cat([trend_img, season_img, resid_img], dim=1)

            # Build mask image: single channel (B, 1, H, W) that broadcasts
            # across all 3*C channels — same pattern as original run_diffem_mmps.py
            mask_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
            mask_img = model.ts_to_img(mask_expanded)[:, :1, :, :]

            # MMPS sampling on the decomposed representation
            x_img_imputed = process.sampling_mmps(x_obs_img, mask_img)

            # Convert back to TS by summing the three component groups
            x_ts_recon = model.img_to_ts(x_img_imputed)

            all_reconstructions.append(x_ts_recon.cpu().numpy())
            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    print(f"  E-step complete. Imputed {len(reconstructions)} sequences.")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                             f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


# =============================================================================
# Decomposed M-step
# =============================================================================

def m_step(args, model, optimizer, reconstructions, em_iter, device, logger=None):
    """
    M-step: train the decomposed model on fully imputed data.

    STL decomposition + image conversion is done ONCE before training
    (reconstructions don't change within an M-step).  Training then
    iterates over pre-computed decomposed images.
    """
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"  Training decomposed model for {args.m_step_epochs} epochs ...")

    # --- Pre-decompose ALL reconstructions once (avoids per-batch STL) ---
    print("  Pre-computing STL decomposition for all reconstructions ...")
    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    trends, seasonals, residuals = stl_decompose_batch(
        recon_tensor, period=model.stl_period
    )
    # Convert each component to image and stack → (N, 3*C, H, W)
    all_imgs = []
    pre_bs = args.batch_size * 2
    for start in range(0, len(recon_tensor), pre_bs):
        end = min(start + pre_bs, len(recon_tensor))
        t_batch = trends[start:end].to(device)
        s_batch = seasonals[start:end].to(device)
        r_batch = residuals[start:end].to(device)
        img_batch = model.ts_to_img_from_components(t_batch, s_batch, r_batch)
        all_imgs.append(img_batch.cpu())
    decomposed_imgs = torch.cat(all_imgs, dim=0)
    print(f"  Pre-computed {len(decomposed_imgs)} decomposed images "
          f"with shape {tuple(decomposed_imgs.shape[1:])}")

    img_loader = Data.DataLoader(
        Data.TensorDataset(decomposed_imgs),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )

    lambda_trend = getattr(args, 'lambda_trend', 0.1)
    lambda_spectral = getattr(args, 'lambda_spectral', 0.05)
    use_component_loss = False

    model.train()

    for epoch in range(args.m_step_epochs):
        epoch_loss = 0
        num_batches = 0

        for (x_img,) in img_loader:
            x_img = x_img.to(device)

            if use_component_loss:
                loss, to_log = model.loss_fn_decomposed(
                    x_img, lambda_trend=lambda_trend,
                    lambda_spectral=lambda_spectral,
                )
            else:
                loss, to_log = model.loss_fn_irregular(x_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.on_train_batch_end()

            epoch_loss += loss.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")
        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)

    print(f"  M-step complete for EM iteration {em_iter}.")
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_uncond(args, model, test_loader, em_iter, device, logger=None):
    """
    Generate unconditional samples from the decomposed model and evaluate.

    The model outputs (B, 3*C, H, W) images.  We split into three component
    groups, convert each to TS, and sum to obtain the final time series.
    """
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    model.eval()

    C = model.original_channels
    target_shape = (C * model.N_COMPONENTS, args.img_resolution, args.img_resolution)
    gen_sig, real_sig = [], []

    with torch.no_grad():
        with model.ema_scope():
            process = MMPSDiffusionProcess(
                args, model.net, target_shape,
                sigma_y=getattr(args, 'mmps_sigma_y', 0.01),
            )

            for data in tqdm(test_loader, desc="Evaluating"):
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = model.img_to_ts(x_img_sampled)

                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    print(f"  EM iter {em_iter} metrics:")
    for key, value in scores.items():
        print(f"    {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

    # --- Memorization check ---
    mem_plot_path = f"memorization_hist_em_iter_{em_iter}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig,
        device=device, plot_path=mem_plot_path,
    )

    print(f"  Memorization metrics:")
    for k, v in mem_stats.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
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
            except Exception:
                pass

    return scores


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_decomposed_{name}"
    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        stl_period = getattr(args, 'stl_period', None)

        model = TS2img_Karras_Decomposed(
            args=args, device=args.device, stl_period=stl_period,
        ).to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        print_model_params(logger, model)

        # === Extract corrupted data ===
        print("Extracting corrupted data from train loader ...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(
            train_loader, args.device,
        )
        obs_pct = obs_masks.sum() / obs_masks.size * 100
        print(f"  {len(corrupted_data)} sequences, {obs_pct:.1f}% observed")

        # === Phase 1: STL-informed initialisation ===
        print(f"\n{'='*60}")
        print("Phase 1: STL-Informed Initialisation")
        print(f"{'='*60}")

        initial_recon = initialize_with_stl(
            corrupted_data, obs_masks, period=stl_period, seed=args.seed,
        )

        # === Phase 1.5: initial M-step on STL-filled data ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial training on STL-initialised data")
        print(f"{'='*60}")

        model = m_step(
            args, model, optimizer, initial_recon,
            em_iter=-1, device=args.device, logger=logger,
        )

        # === Phase 2: EM loop ===
        print(f"\n{'='*60}")
        print(f"Starting Decomposition-Enhanced DiffEM-MMPS")
        print(f"  EM iters:       {args.em_iters}")
        print(f"  MMPS sigma_y:   {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters:  {args.mmps_cg_iters}")
        print(f"  STL period:     {stl_period or 'auto'}")
        print(f"  lambda_trend:   {getattr(args, 'lambda_trend', 0.1)}")
        print(f"  lambda_spectral:{getattr(args, 'lambda_spectral', 0.05)}")
        print(f"{'='*60}")

        best_metrics = None
        prev_recons = initial_recon

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            reconstructions = e_step(
                args, model, corrupted_data, obs_masks, prev_recons,
                em_iter, args.device, logger,
            )
            prev_recons = reconstructions

            model = m_step(
                args, model, optimizer, reconstructions,
                em_iter, args.device, logger,
            )

            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, model, test_loader, em_iter,
                    args.device, logger,
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
        print("\n" + "=" * 60)
        print("Decomposition-Enhanced DiffEM-MMPS Complete!")
        print("=" * 60)
        if best_metrics:
            print("Best metrics:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("Training complete.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    main(args)
