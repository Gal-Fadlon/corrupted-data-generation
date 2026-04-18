"""
DiffEM Warm-Start: MMPS Phase + Conditional DiffEM Refinement

Two-phase approach from DiffEM paper (Section 3.1.1, "DiffEM with warm-start"):
  Phase 1: Run MMPS exactly as in run_diffem_mmps.py to reach a strong baseline
  Phase 2: Use conditional DiffEM iterations to refine beyond MMPS

The paper reports that warm-started DiffEM outperforms both standalone DiffEM and
standalone MMPS on CIFAR-10. The key insight: after MMPS converges (its Gaussian
posterior approximation becomes the bottleneck), a conditional model q(x|y) can
continue improving by learning the exact posterior without Gaussian assumptions.

Safety: Phase 1 is an exact copy of run_diffem_mmps.py. Phase 2 adds conservative
refinement with automatic early stopping if disc_mean degrades.
"""

import torch
import torch.autograd
import torch.multiprocessing
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import os
import sys
import copy
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
from models.our import TS2img_Karras, TS2img_Karras_Cond
from models.sampler import DiffusionProcess, ConditionalDiffusionProcess
from utils.train_unconditional import train_unconditional_regular

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Phase 1: MMPS components (identical to run_diffem_mmps.py)
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


class MMPSDiffusionProcess:
    """MMPS posterior sampler — identical to run_diffem_mmps.py."""

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
        result = denoised.detach() + sigma_sq * score.detach()
        return result

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
# Phase 1 helper functions (identical to run_diffem_mmps.py)
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


def initialize_with_gaussian(corrupted_data, obs_masks, seed=None):
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
# Phase 1: MMPS E-step and M-step (identical to run_diffem_mmps.py)
# =============================================================================

def mmps_e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    """E-step: MMPS imputation — identical to run_diffem_mmps.py."""
    print(f"\n=== MMPS E-Step (EM iter {em_iter}) ===")
    uncond_model.eval()

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
    print(f"MMPS E-step complete. Imputed {len(reconstructions)} sequences.")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def mmps_m_step(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """M-step: Train unconditional model on imputed data — identical to run_diffem_mmps.py."""
    print(f"\n=== MMPS M-Step (EM iter {em_iter}) ===")
    print(f"Training unconditional model for {args.m_step_epochs} epochs...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
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
            print(f"  MMPS M-step epoch {epoch+1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)

    print(f"MMPS M-step complete for EM iteration {em_iter}.")
    return uncond_model


# =============================================================================
# Phase 2: Conditional DiffEM E-step and M-step
# =============================================================================

def conditional_e_step(args, cond_model, corrupted_data, obs_masks, ws_iter, device, logger=None):
    """
    Conditional E-step: sample x ~ q_theta(x|y) using the conditional model,
    then enforce observation consistency (replace observed values with true data).
    """
    print(f"\n=== Conditional E-Step (warm-start iter {ws_iter}) ===")
    cond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    cond_channels = args.input_channels + 1

    N = len(corrupted_data)
    batch_size = args.batch_size

    with torch.no_grad():
        with cond_model.ema_scope():
            process = ConditionalDiffusionProcess(
                args, cond_model.net, target_shape, cond_channels
            )

            for start_idx in tqdm(range(0, N, batch_size), desc="E-step Conditional"):
                end_idx = min(start_idx + batch_size, N)

                corrupted_batch = corrupted_data[start_idx:end_idx]
                mask_batch = obs_masks[start_idx:end_idx]

                corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
                mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)

                obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)
                cond_img = cond_model.ts_to_img(obs_ts)

                mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
                mask_img = cond_model.ts_to_img(mask_ts_expanded)
                mask_img = mask_img[:, :1, :, :]

                x_img_sampled = process.sampling(cond_img, mask_img)
                x_ts_recon = cond_model.img_to_ts(x_img_sampled)

                # Observation consistency: replace observed values with true observations
                x_ts_recon_np = x_ts_recon.cpu().numpy()
                for i in range(len(corrupted_batch)):
                    for t in range(corrupted_batch.shape[1]):
                        if mask_batch[i, t]:
                            obs_vals = corrupted_batch[i, t, :]
                            valid = ~np.isnan(obs_vals)
                            x_ts_recon_np[i, t, valid] = obs_vals[valid]

                all_reconstructions.append(x_ts_recon_np)
                torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    print(f"Conditional E-step complete. Sampled {len(reconstructions)} sequences.")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'missing_{int(args.missing_rate*100)}',
                              'warmstart')
    save_reconstructions(reconstructions, cache_dir, ws_iter)

    if logger is not None:
        logger.log('warmstart/e_step_samples', len(reconstructions), ws_iter)

    return reconstructions


def conditional_m_step(args, cond_model, cond_optimizer, reconstructions,
                       corrupted_data, obs_masks, ws_iter, device, logger=None):
    """
    Conditional M-step: train q_theta(x|y) on (corrupted_y, x_reconstruction) pairs.

    Following the DiffEM paper:
    - Fresh optimizer + EMA reset per iteration (to avoid stale momentum)
    - Loss computed on ALL pixels (full denoising loss, not just observed)
    """
    cond_epochs = getattr(args, 'warmstart_cond_epochs', 100)
    print(f"\n=== Conditional M-Step (warm-start iter {ws_iter}) ===")
    print(f"Training conditional model for {cond_epochs} epochs...")

    N = len(reconstructions)
    features = corrupted_data.shape[-1]

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    corrupted_tensor = torch.tensor(np.nan_to_num(corrupted_data, nan=0.0), dtype=torch.float32)
    mask_tensor = torch.tensor(obs_masks, dtype=torch.float32)

    dataset = Data.TensorDataset(recon_tensor, corrupted_tensor, mask_tensor)
    loader = Data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    cond_model.train()

    for epoch in range(cond_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (x_clean, x_corrupt, mask) in enumerate(loader):
            x_clean = x_clean.to(device)
            x_corrupt = x_corrupt.to(device)
            mask = mask.to(device)

            x_target_img = cond_model.ts_to_img(x_clean)
            cond_img = cond_model.ts_to_img(x_corrupt)

            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, features)
            mask_img = cond_model.ts_to_img(mask_expanded)
            mask_img = mask_img[:, :1, :, :]

            loss, to_log = cond_model.loss_fn_conditional_full(x_target_img, cond_img, mask_img)

            cond_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cond_model.parameters(), 1.0)
            cond_optimizer.step()
            cond_model.on_train_batch_end()

            epoch_loss += loss.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Cond M-step epoch {epoch+1}/{cond_epochs}, loss: {avg_loss:.4f}")

        if logger is not None:
            global_step = ws_iter * cond_epochs + epoch
            logger.log('warmstart/cond_m_step_loss', avg_loss, global_step)

    print(f"Conditional M-step complete for warm-start iteration {ws_iter}.")
    return cond_model


def uncond_m_step_finetune(args, uncond_model, optimizer, reconstructions,
                           ws_iter, device, logger=None):
    """
    Continue fine-tuning the unconditional model on the new DiffEM reconstructions.
    Uses the same logic as MMPS M-step.
    """
    print(f"\n=== Uncond Fine-tune (warm-start iter {ws_iter}) ===")
    print(f"Fine-tuning unconditional model for {args.m_step_epochs} epochs...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
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
            print(f"  Uncond fine-tune epoch {epoch+1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")

        if logger is not None:
            global_step = ws_iter * args.m_step_epochs + epoch
            logger.log('warmstart/uncond_m_step_loss', avg_loss, global_step)

    print(f"Uncond fine-tune complete for warm-start iteration {ws_iter}.")
    return uncond_model


# =============================================================================
# Evaluation (identical to run_diffem_mmps.py)
# =============================================================================

def evaluate_uncond(args, uncond_model, test_loader, step_label, device, logger=None, log_prefix='test'):
    """Evaluate the unconditional model. step_label is used for logging."""
    print(f"\n=== Evaluation ({step_label}) ===")

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

    print(f"{step_label} metrics (unconditional):")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'{log_prefix}/{key}', value, step_label if isinstance(step_label, int) else 0)

    mem_plot_path = f"memorization_hist_{step_label}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig,
        device=device, plot_path=mem_plot_path
    )

    print(f"{step_label} memorization metrics:")
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if logger is not None:
            logger.log(f'{log_prefix}/memorization/{k}', v,
                       step_label if isinstance(step_label, int) else 0)

    if logger is not None:
        upload_successful = False
        try:
            logger.log_file(f'{log_prefix}/memorization/histogram',
                           mem_plot_path,
                           step_label if isinstance(step_label, int) else 0)
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
    name = create_model_name_and_dir(args)
    name = f"diffem_warmstart_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        # =====================================================================
        # Phase 1: MMPS (identical to run_diffem_mmps.py)
        # =====================================================================

        print("\n" + "=" * 70)
        print("PHASE 1: MMPS (identical to run_diffem_mmps.py)")
        print("=" * 70)

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        uncond_optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        print_model_params(logger, uncond_model)

        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences with "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # Gaussian initialization
        initial_reconstructions = initialize_with_gaussian(
            corrupted_data, obs_masks, seed=args.seed
        )

        # Initial M-step on Gaussian-filled data
        uncond_model = mmps_m_step(
            args, uncond_model, uncond_optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )

        # MMPS EM loop
        print(f"\nStarting MMPS phase with {args.em_iters} EM iterations")
        print(f"  MMPS sigma_y: {args.mmps_sigma_y}")
        print(f"  MMPS cg_iters: {args.mmps_cg_iters}")

        best_mmps_metrics = None
        last_mmps_reconstructions = initial_reconstructions

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"MMPS EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            reconstructions = mmps_e_step(
                args, uncond_model, corrupted_data, obs_masks,
                em_iter, args.device, logger
            )
            last_mmps_reconstructions = reconstructions

            uncond_model = mmps_m_step(
                args, uncond_model, uncond_optimizer,
                reconstructions, em_iter, args.device, logger
            )

            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, uncond_model, test_loader,
                    em_iter, args.device, logger, log_prefix='test'
                )

                if best_mmps_metrics is None or metrics.get('disc_mean', float('inf')) < best_mmps_metrics.get('disc_mean', float('inf')):
                    best_mmps_metrics = metrics

                if logger is not None:
                    logger.log('em/iteration', em_iter, em_iter)


        print("\n" + "=" * 60)
        print("Phase 1 (MMPS) Complete!")
        print("=" * 60)
        if best_mmps_metrics:
            print("Best MMPS metrics:")
            for k, v in best_mmps_metrics.items():
                print(f"  {k}: {v:.4f}")

        best_mmps_disc = best_mmps_metrics.get('disc_mean', float('inf')) if best_mmps_metrics else float('inf')

        # Save MMPS model state as safety checkpoint
        mmps_uncond_state = copy.deepcopy(uncond_model.state_dict())
        mmps_optimizer_state = copy.deepcopy(uncond_optimizer.state_dict())

        # =====================================================================
        # Phase 2: Conditional DiffEM Refinement
        # =====================================================================

        warmstart_iters = getattr(args, 'warmstart_iters', 5)
        warmstart_cond_epochs = getattr(args, 'warmstart_cond_epochs', 100)
        warmstart_cond_lr = getattr(args, 'warmstart_cond_lr', None) or args.learning_rate
        stop_on_degradation = getattr(args, 'warmstart_stop_on_degradation', True)

        print("\n" + "=" * 70)
        print("PHASE 2: Conditional DiffEM Refinement (warm-start)")
        print("=" * 70)
        print(f"  Warm-start iterations: {warmstart_iters}")
        print(f"  Conditional model epochs/iter: {warmstart_cond_epochs}")
        print(f"  Conditional model LR: {warmstart_cond_lr}")
        print(f"  Stop on degradation: {stop_on_degradation}")
        print(f"  Best MMPS disc_mean (baseline): {best_mmps_disc:.4f}")

        # Create conditional model
        cond_model = TS2img_Karras_Cond(args=args, device=args.device).to(args.device)
        cond_optimizer = torch.optim.AdamW(
            cond_model.parameters(),
            lr=warmstart_cond_lr,
            weight_decay=args.weight_decay
        )

        cond_params = sum(p.numel() for p in cond_model.parameters())
        print(f"Conditional model parameters: {cond_params:,}")

        # Train initial conditional model on MMPS reconstructions
        print("\n--- Training initial conditional model on MMPS reconstructions ---")
        cond_model = conditional_m_step(
            args, cond_model, cond_optimizer,
            last_mmps_reconstructions, corrupted_data, obs_masks,
            ws_iter=-1, device=args.device, logger=logger
        )

        best_overall_metrics = best_mmps_metrics
        best_overall_disc = best_mmps_disc
        degradation_count = 0

        for ws_iter in range(warmstart_iters):
            print(f"\n{'='*60}")
            print(f"Warm-Start DiffEM Iteration {ws_iter + 1}/{warmstart_iters}")
            print(f"{'='*60}")

            # E-step: conditional sampling
            reconstructions = conditional_e_step(
                args, cond_model, corrupted_data, obs_masks,
                ws_iter, args.device, logger
            )

            # Conditional M-step: train q(x|y) with fresh optimizer + EMA reset
            cond_optimizer = torch.optim.AdamW(
                cond_model.parameters(),
                lr=warmstart_cond_lr,
                weight_decay=args.weight_decay
            )
            cond_model.reset_ema()

            cond_model = conditional_m_step(
                args, cond_model, cond_optimizer,
                reconstructions, corrupted_data, obs_masks,
                ws_iter, args.device, logger
            )

            # Unconditional M-step: fine-tune uncond model on new reconstructions
            uncond_model = uncond_m_step_finetune(
                args, uncond_model, uncond_optimizer,
                reconstructions, ws_iter, args.device, logger
            )

            # Evaluate
            metrics = evaluate_uncond(
                args, uncond_model, test_loader,
                args.em_iters + ws_iter, args.device, logger,
                log_prefix='warmstart'
            )

            current_disc = metrics.get('disc_mean', float('inf'))

            if logger is not None:
                logger.log('warmstart/iteration', ws_iter, ws_iter)
                logger.log('warmstart/disc_mean', current_disc, ws_iter)
                logger.log('warmstart/best_disc_mean', best_overall_disc, ws_iter)

            if current_disc < best_overall_disc:
                print(f"  >>> IMPROVED: disc_mean {current_disc:.4f} < {best_overall_disc:.4f}")
                best_overall_disc = current_disc
                best_overall_metrics = metrics
                degradation_count = 0
            else:
                degradation_count += 1
                print(f"  >>> No improvement: disc_mean {current_disc:.4f} >= {best_overall_disc:.4f} "
                      f"(degradation count: {degradation_count})")

                if stop_on_degradation and degradation_count >= 2:
                    print(f"\n  EARLY STOP: disc_mean degraded for {degradation_count} consecutive iterations.")
                    print(f"  Restoring best MMPS model checkpoint.")
                    uncond_model.load_state_dict(mmps_uncond_state)
                    uncond_optimizer.load_state_dict(mmps_optimizer_state)
                    break

        # =====================================================================
        # Final Summary
        # =====================================================================

        print("\n" + "=" * 70)
        print("DiffEM Warm-Start Training Complete!")
        print("=" * 70)
        print(f"Best MMPS disc_mean:    {best_mmps_disc:.4f}")
        print(f"Best overall disc_mean: {best_overall_disc:.4f}")
        if best_overall_disc < best_mmps_disc:
            improvement = (best_mmps_disc - best_overall_disc) / best_mmps_disc * 100
            print(f"  >>> Phase 2 improved by {improvement:.1f}%")
        else:
            print(f"  >>> Phase 2 did not improve over MMPS")

        if best_overall_metrics:
            print("\nBest overall metrics:")
            for k, v in best_overall_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM Warm-Start training is complete")

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


if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
