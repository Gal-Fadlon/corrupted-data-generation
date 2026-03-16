"""
DiffEM with Unconditional Model + MMPS E-step — Codebook-Regularized E-step

Same as run_diffem_mmps_ema_reset_stl_init.py with ONE addition:
  After MMPS imputation, imputed sub-series patches are soft-blended toward
  their nearest entry in a codebook learned from observed data. This constrains
  imputed regions to resemble real local patterns, improving downstream
  generation quality (disc_mean).

Inspired by TimeMAE (Cheng et al., WSDM 2026) — Masked Codeword Classification
(Sec 3.4.1): a discrete codebook of sub-series prototypes provides a categorical
"snap to reality" signal for masked/imputed positions.
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
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import (
    gen_dataloader, save_reconstructions, load_reconstructions
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.utils_stl import initialize_with_stl
from utils.utils_codebook import build_patch_codebook, codebook_refine_reconstructions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Conjugate Gradient Solver
# =============================================================================

def conjugate_gradient(A_fn, b, maxiter=1):
    """
    Conjugate gradient solver for linear system A·x = b, where A is given
    as a function (linear operator).

    For maxiter=1, reduces to a single scaled residual step:
        x = (rᵀr / rᵀAr) · r  where r = b (initial residual with x₀=0)

    Args:
        A_fn: callable, linear operator v -> A·v
        b: right-hand side tensor
        maxiter: number of CG iterations (default=1)

    Returns:
        x: approximate solution tensor (same shape as b)
    """
    x = torch.zeros_like(b)
    r = b.clone()  # r = b - A·x₀ = b (since x₀ = 0)
    p = r.clone()

    for _ in range(maxiter):
        Ap = A_fn(p)
        rTr = (r * r).sum()
        pTAp = (p * Ap).sum()

        # Avoid division by zero
        alpha = rTr / (pTAp + 1e-30)

        x = x + alpha * p
        r_new = r - alpha * Ap

        rTr_new = (r_new * r_new).sum()
        beta = rTr_new / (rTr + 1e-30)

        p = r_new + beta * p
        r = r_new

    return x


# =============================================================================
# MMPSDiffusionProcess
# =============================================================================

class MMPSDiffusionProcess:
    """
    MMPS-style diffusion sampler for inpainting with an unconditional model.

    Uses the same EDM/Karras sigma schedule as DiffusionProcess. At each reverse
    step, the unconditional denoiser is replaced by a posterior denoiser that
    incorporates observation constraints with covariance correction.

    MMPS approximation (full Gaussian with covariance):
        p(y | x_t) ≈ N(y | A·D_θ(x_t), σ_y²I + A·V[x₀|x_t]·Aᵀ)
    where V[x₀|x_t] = σ²·Jᵀ_D (Jacobian-transpose of denoiser, scaled by σ²)

    The covariance term A·V·Aᵀ regularizes the guidance, preventing over-correction
    at high noise levels where the denoiser is uncertain.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.01, cg_iters=1):
        self.args = args
        self.device = args.device
        self.shape = shape
        self.net = diffusion_fn.to(device=self.device)

        # EDM parameters (same as DiffusionProcess)
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

        # MMPS parameters
        self.sigma_y = sigma_y
        self.cg_iters = cg_iters

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        """
        Posterior denoiser: replaces unconditional D_θ(x_t) with posterior estimate.

        Following Rozet's PosteriorDenoiser (DiffEM-main/priors/diffusion.py:481-558):

        1. Forward pass with gradient: denoised = D_θ(x_t)
        2. Define VJP function using torch.autograd.grad (robust to in-place ops)
        3. Compute residual: r = x_obs - mask * denoised
        4. Solve linear system via CG: (σ_y²I + σ² · mask · Jᵀ · mask) · v = r
        5. Compute posterior correction: score = vjp_fn(mask · v)
        6. Return: denoised + σ² · score

        Args:
            x_t: noisy input (batch, C, H, W), float64
            sigma: noise level scalar (float64)
            x_obs: observed data (batch, C, H, W), float64
            mask: binary mask (batch, 1, H, W), float64, 1=observed

        Returns:
            posterior_denoised: corrected denoised estimate (batch, C, H, W), float64
        """
        sigma_sq = sigma ** 2
        sigma_y_sq = self.sigma_y ** 2

        # Forward pass with gradient tracking
        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return grad

        # Compute residual: r = x_obs - mask * denoised
        r = x_obs - mask * denoised

        def cg_operator(v):
            return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)

        # Solve (σ_y²I + σ² · mask · Jᵀ · mask) · v = r
        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)

        # Compute posterior correction: score = Jᵀ(mask · v)
        score = vjp_fn(mask * v)

        # Detach from computation graph
        result = denoised.detach() + sigma_sq * score.detach()

        return result

    def mmps_sample(self, latents, x_obs_img, mask_img):
        """
        MMPS sampling: inpaint missing regions using posterior denoiser.

        Same structure as DiffusionProcess.sample(), but BOTH the Euler step
        and Heun correction call posterior_denoise() instead of self.net().

        Args:
            latents: initial noise (batch, C, H, W)
            x_obs_img: observed data in image space (batch, C, H, W), clean values
            mask_img: binary mask in image space (batch, 1, H, W), 1=observed, 0=missing

        Returns:
            x_next: inpainted result (batch, C, H, W)
        """
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

            # Euler step with POSTERIOR denoiser
            denoised = self.posterior_denoise(x_hat, t_hat, x_obs, mask)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction with POSTERIOR denoiser
            if i < self.num_steps - 1:
                denoised_2 = self.posterior_denoise(x_next, t_next, x_obs, mask)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sampling_mmps(self, x_obs_img, mask_img):
        """
        Convenience wrapper for MMPS sampling.
        NOT decorated with @torch.no_grad() — MMPS needs gradients for VJP.
        """
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, x_obs_img, mask_img)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        """Standard unconditional sampling (for evaluation)."""
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
    """
    Extract corrupted time series (with NaNs) from the train loader.

    Returns:
        corrupted_data: numpy array (N, seq_len, features) with NaN for missing
        obs_masks: numpy array (N, seq_len) with True for observed
    """
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


def initialize_with_gaussian(corrupted_data, obs_masks, seed=None):
    """Initialize reconstructions using Gaussian prior."""
    if seed is not None:
        np.random.seed(seed)

    N, seq_len, features = corrupted_data.shape

    mu = np.nanmean(corrupted_data, axis=(0, 1))
    std = np.nanstd(corrupted_data, axis=(0, 1))
    std = np.where(std == 0, 1.0, std)

    initial_recon = corrupted_data.copy()

    for i in range(N):
        for t in range(seq_len):
            if not obs_masks[i, t]:
                initial_recon[i, t, :] = np.random.normal(mu, std)

    initial_recon = np.clip(initial_recon, 0, 1)

    print(f"Gaussian initialization complete. Generated {N} sequences.")
    print(f"  Mean per feature: {mu}")
    print(f"  Std per feature: {std}")

    return initial_recon


# =============================================================================
# Core EM functions
# =============================================================================

def e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device,
           codebook=None, logger=None):
    """
    E-step: Impute missing values using MMPS, then refine with codebook.

    1. Standard MMPS imputation (posterior denoiser with covariance correction)
    2. Codebook refinement: blend imputed patches toward nearest codebook entry
       with adaptive, decaying blend strength.
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — MMPS + Codebook Refinement ===")
    print("Imputing missing values via MMPS with unconditional model...")

    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = getattr(args, 'e_step_batch_size', 64)

    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    with uncond_model.ema_scope():
        process = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step MMPS"):
            end_idx = min(start_idx + batch_size, N)

            corrupted_batch = corrupted_data[start_idx:end_idx]
            mask_batch = obs_masks[start_idx:end_idx]

            corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
            mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)

            obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)

            x_obs_img = uncond_model.ts_to_img(obs_ts)

            mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
            mask_img = uncond_model.ts_to_img(mask_ts_expanded)
            mask_img = mask_img[:, :1, :, :]

            x_img_imputed = process.sampling_mmps(x_obs_img, mask_img)

            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)

            all_reconstructions.append(x_ts_recon.cpu().numpy())

            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    print(f"MMPS imputation complete. {len(reconstructions)} sequences.")

    # --- Codebook refinement (TimeMAE MCC-inspired) ---
    if codebook is not None:
        progress = em_iter / max(1, args.em_iters - 1)
        blend = args.codebook_blend_start + progress * (
            args.codebook_blend_end - args.codebook_blend_start
        )
        print(f"  Applying codebook refinement (blend_strength={blend:.3f}, "
              f"patch_size={args.codebook_patch_size})")
        reconstructions = codebook_refine_reconstructions(
            reconstructions, obs_masks, codebook,
            patch_size=args.codebook_patch_size,
            blend_strength=blend
        )
        if logger is not None:
            logger.log('em/codebook_blend_strength', blend, em_iter)

    # Verify no NaN in output
    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    # Cache reconstructions to disk
    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def m_step(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """M-step: Train unconditional model on fully imputed data."""
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training unconditional model for {args.m_step_epochs} epochs...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    uncond_model.train()

    for epoch in range(args.m_step_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (x_clean,) in enumerate(recon_loader):
            x_clean = x_clean.to(device)

            x_img = uncond_model.ts_to_img(x_clean)

            loss, to_log = uncond_model.loss_fn_irregular(x_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss += loss.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  M-step epoch {epoch+1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, em_iter, device, logger=None):
    """Evaluate the unconditional model via standard unconditional sampling."""
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    print("Evaluating unconditional model (same as run_irregular.py)...")

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

                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    print(f"EM iter {em_iter} metrics (unconditional):")
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
    """Main DiffEM-Uncond training loop with MMPS E-step + Codebook refinement."""

    # Model name and directory
    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_codebook_estep_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        # === Create single unconditional model ===
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
        print(f"Extracted {len(corrupted_data)} sequences with "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # === Build patch codebook from observed data ===
        print(f"\n{'='*60}")
        print("Building patch codebook from observed data (TimeMAE MCC)")
        print(f"{'='*60}")
        codebook = build_patch_codebook(
            corrupted_data, obs_masks,
            patch_size=args.codebook_patch_size,
            n_codes=args.codebook_n_codes,
            seed=args.seed
        )

        # === Phase 1: Initialize with STL-informed prior ===
        print(f"\n{'='*60}")
        print("Phase 1: STL Warm-Start Initialization")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_stl(
            corrupted_data, obs_masks, seed=args.seed
        )

        # === Phase 1.5: Train unconditional model on STL-initialized data ===
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial unconditional model training on STL-initialized data")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )
        print("Initial unconditional model training complete.")

        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS + Codebook with {args.em_iters} EM iterations")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")
        print(f"  Codebook: {args.codebook_n_codes} codes, patch_size={args.codebook_patch_size}")
        print(f"  Blend: {args.codebook_blend_start} -> {args.codebook_blend_end}")
        print(f"{'='*60}")

        best_metrics = None
        metrics = None

        try:
            for em_iter in range(args.em_iters):
                print(f"\n{'='*60}")
                print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
                print(f"{'='*60}")

                # E-step: MMPS imputation + codebook refinement
                reconstructions = e_step(
                    args, uncond_model, corrupted_data, obs_masks,
                    em_iter, args.device, codebook=codebook, logger=logger
                )

                # Reset EMA so it tracks only this iteration's training dynamics
                uncond_model.reset_ema()

                # M-step: Train unconditional model on fully imputed data
                uncond_model = m_step(
                    args, uncond_model, optimizer,
                    reconstructions, em_iter, args.device, logger
                )

                # Evaluate periodically
                if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                    metrics = evaluate_uncond(
                        args, uncond_model, test_loader,
                        em_iter, args.device, logger
                    )

                    if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                        best_metrics = metrics

                if logger is not None:
                    logger.log('em/iteration', em_iter, em_iter)

        except Exception as e:
            logging.error(f"EM loop failed at iteration {em_iter}: "
                          f"{type(e).__name__}: {e}", exc_info=True)
            if logger is not None:
                logger.log('em/failed_at_iter', em_iter, em_iter)
            raise

        print("\n" + "="*60)
        print("DiffEM-MMPS + Codebook E-step Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS + Codebook E-step training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    # Force DiffEM mode
    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
