"""
DiffEM with Unconditional Model + MMPS E-step — Momentum-Teacher Stabilized

Same as run_diffem_mmps_ema_reset_stl_init.py with ONE addition:
  A momentum-averaged teacher model provides a stable imputation reference
  via cheap replacement-based sampling (no VJP). The teacher's output is
  blended with the online model's MMPS output, preventing EM oscillation.

  The teacher is updated after each M-step:
      theta_teacher <- eta * theta_teacher + (1 - eta) * theta_online
  and is NEVER reset — it accumulates knowledge across all EM iterations.

Inspired by TimeMAE (Cheng et al., WSDM 2026) — Masked Representation
Regression (Sec 3.4.2, Eq. 5-6): a momentum-updated target encoder provides
stable regression targets, preventing collapse in the siamese architecture.
"""

import copy
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
        """
        Posterior denoiser: replaces unconditional D_θ(x_t) with posterior estimate.
        """
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

        result = denoised.detach() + sigma_sq * score.detach()

        return result

    def mmps_sample(self, latents, x_obs_img, mask_img):
        """MMPS sampling: inpaint missing regions using posterior denoiser."""
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
        """Convenience wrapper for MMPS sampling (needs gradients for VJP)."""
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
# Teacher Replacement-Based Imputation (Repaint-style, no VJP)
# =============================================================================

@torch.no_grad()
def teacher_repaint_impute(teacher_net, uncond_model, corrupted_data, obs_masks,
                           device, args):
    """
    Cheap replacement-based imputation using the teacher model.

    At each reverse diffusion step:
      1. Unconditional denoise: x0_hat = D_teacher(x_t, sigma)
      2. Replace observed pixels in TS space with true values
      3. Convert back and continue reverse process

    This is gradient-free (no VJP), costing ~1/3 of a MMPS pass, while still
    providing a stable reference signal from the momentum teacher.

    Args:
        teacher_net: teacher's network (momentum-averaged weights)
        uncond_model: model instance (for ts_to_img / img_to_ts conversions)
        corrupted_data: (N, T, C) numpy with NaN for missing
        obs_masks: (N, T) boolean, True = observed
        device: target device
        args: configuration arguments

    Returns:
        teacher_recon: (N, T, C) numpy array of teacher imputations
    """
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    sigma_min = max(0.002, teacher_net.sigma_min)
    sigma_max = min(80, teacher_net.sigma_max)
    rho = 7
    num_steps = args.diffusion_steps

    N = len(corrupted_data)
    batch_size = args.batch_size
    all_recon = []

    for start_idx in tqdm(range(0, N, batch_size), desc="Teacher impute"):
        end_idx = min(start_idx + batch_size, N)
        bs = end_idx - start_idx

        corrupted_batch = corrupted_data[start_idx:end_idx]
        mask_batch = obs_masks[start_idx:end_idx]

        corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
        mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)
        obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)

        # Sigma schedule (same Karras discretization as MMPS)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([teacher_net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Start from noise
        latents = torch.randn([bs, *target_shape], device=device)
        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            t_hat = teacher_net.round_sigma(t_cur)
            x_hat = x_cur

            # Unconditional denoise
            denoised = teacher_net(x_hat, t_hat, None).to(torch.float64)

            # Replace observed values in TS space
            x_ts = uncond_model.img_to_ts(denoised)
            mask_expanded = mask_ts.unsqueeze(-1).expand_as(x_ts)
            x_ts = torch.where(mask_expanded.bool(), obs_ts.to(torch.float64), x_ts)
            denoised = uncond_model.ts_to_img(x_ts.float()).to(torch.float64)

            # Euler step
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Heun correction
            if i < num_steps - 1:
                denoised_2 = teacher_net(x_next, t_next, None).to(torch.float64)

                x_ts_2 = uncond_model.img_to_ts(denoised_2)
                x_ts_2 = torch.where(mask_expanded.bool(), obs_ts.to(torch.float64), x_ts_2)
                denoised_2 = uncond_model.ts_to_img(x_ts_2.float()).to(torch.float64)

                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        x_ts_final = uncond_model.img_to_ts(x_next)
        all_recon.append(x_ts_final.cpu().numpy())

        torch.cuda.empty_cache()

    return np.vstack(all_recon)


# =============================================================================
# Helper functions
# =============================================================================

def get_corrupted_data_from_loader(train_loader, device):
    """Extract corrupted time series (with NaNs) from the train loader."""
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

def e_step(args, uncond_model, teacher_model, corrupted_data, obs_masks,
           em_iter, device, logger=None):
    """
    E-step: Impute with MMPS (online) + repaint (teacher), blend results.

    1. Online model: full MMPS imputation (posterior denoiser with VJP)
    2. Teacher model: replacement-based imputation (no VJP, cheap)
    3. Blend: recon = (1 - tw) * online + tw * teacher
       where tw decays linearly from teacher_blend_start to teacher_blend_end
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — MMPS + Momentum Teacher ===")

    uncond_model.eval()

    # --- Pass 1: Online MMPS imputation ---
    print("Pass 1: MMPS imputation with online model...")
    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = args.batch_size

    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    with uncond_model.ema_scope():
        process = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step MMPS (online)"):
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

    online_recon = np.vstack(all_reconstructions)
    print(f"Online MMPS complete. {len(online_recon)} sequences.")

    # --- Pass 2: Teacher replacement-based imputation ---
    print("Pass 2: Replacement-based imputation with teacher model...")
    teacher_recon = teacher_repaint_impute(
        teacher_model.net, uncond_model, corrupted_data, obs_masks,
        device, args
    )
    print(f"Teacher imputation complete. {len(teacher_recon)} sequences.")

    # --- Blend online and teacher ---
    progress = em_iter / max(1, args.em_iters - 1)
    tw = args.teacher_blend_start + progress * (
        args.teacher_blend_end - args.teacher_blend_start
    )
    reconstructions = (1.0 - tw) * online_recon + tw * teacher_recon
    reconstructions = np.clip(reconstructions, 0, 1)

    print(f"Blended: teacher_weight={tw:.3f}")

    if logger is not None:
        logger.log('em/teacher_blend_weight', tw, em_iter)

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


def update_teacher_momentum(teacher_model, online_model, momentum):
    """
    Momentum update for teacher model (TimeMAE Eq. 6):
        theta_teacher <- eta * theta_teacher + (1 - eta) * theta_online

    Only updates the network parameters (not EMA buffers).
    """
    with torch.no_grad():
        for p_teacher, p_online in zip(
            teacher_model.net.parameters(), online_model.net.parameters()
        ):
            p_teacher.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)


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
    """Main DiffEM-Uncond training loop with MMPS + Momentum Teacher."""

    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_momentum_teacher_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        # === Create online unconditional model ===
        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        print_model_params(logger, uncond_model)

        # === Create momentum teacher (deep copy, frozen gradients) ===
        print("Creating momentum teacher model (TimeMAE MRR)...")
        teacher_model = copy.deepcopy(uncond_model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print(f"  Teacher momentum: {args.teacher_momentum}")
        print(f"  Teacher blend: {args.teacher_blend_start} -> {args.teacher_blend_end}")

        # === Extract corrupted data and masks ===
        print("Extracting corrupted data from train loader...")
        try:
            corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        except Exception as e:
            logging.error(f"Failed to extract corrupted data: {type(e).__name__}: {e}",
                          exc_info=True)
            raise
        print(f"Extracted {len(corrupted_data)} sequences with "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

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

        # Sync teacher to trained online model before EM starts
        teacher_model.load_state_dict(uncond_model.state_dict())
        # Strip EMA from teacher — it never uses ema_scope(), so shadow
        # weights just waste GPU memory
        if hasattr(teacher_model, 'model_ema'):
            del teacher_model.model_ema
            teacher_model.use_ema = False
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print("Teacher synced to initial online model weights (EMA stripped).")

        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS + Momentum Teacher with {args.em_iters} EM iterations")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")
        print(f"  Teacher momentum: {args.teacher_momentum}")
        print(f"  Teacher blend: {args.teacher_blend_start} -> {args.teacher_blend_end}")
        print(f"{'='*60}")

        best_metrics = None
        metrics = None

        try:
            for em_iter in range(args.em_iters):
                print(f"\n{'='*60}")
                print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
                print(f"{'='*60}")

                # E-step: MMPS (online) + repaint (teacher) -> blended reconstructions
                reconstructions = e_step(
                    args, uncond_model, teacher_model, corrupted_data, obs_masks,
                    em_iter, args.device, logger=logger
                )

                # Reset EMA so it tracks only this iteration's training dynamics
                uncond_model.reset_ema()

                # M-step: Train unconditional model on blended imputed data
                uncond_model = m_step(
                    args, uncond_model, optimizer,
                    reconstructions, em_iter, args.device, logger
                )

                # Momentum update teacher (NO reset — accumulates across all EM iters)
                update_teacher_momentum(teacher_model, uncond_model, args.teacher_momentum)
                print(f"Teacher updated: eta={args.teacher_momentum}")

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
        print("DiffEM-MMPS + Momentum Teacher Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS + Momentum Teacher training is complete")


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
