"""
DiffEM with MMPS E-step — Gibbs-Decomposition (Two-Pass Alternating Imputation)

Fork of run_diffem_mmps_stl_curriculum.py with one key change:
  The E-step runs TWO MMPS passes per iteration:
    Pass 1: Standard MMPS imputation (same as baseline)
    Pass 2: Decompose Pass-1 output, use trend+seasonal as synthetic
            observations at missing positions, and re-impute with these
            "augmented" observations (using higher sigma_y for synthetic obs)

  This is a simplified Gibbs-style approach inspired by:
    Zhang et al. "Bayesian Signal Component Decomposition via
    Diffusion-within-Gibbs Sampling" (2026)

  The second pass benefits from "seeing" the structural components
  (trend+seasonal) everywhere, even if only approximately.  This makes
  the residual imputation much easier.
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
from scipy.ndimage import uniform_filter1d

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
from utils.utils_stl import initialize_with_stl, auto_detect_period, fast_decompose_batch
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
# MMPSDiffusionProcess (supports per-pixel sigma_y via mask weighting)
# =============================================================================

class MMPSDiffusionProcess:
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

    def posterior_denoise(self, x_t, sigma, x_obs, mask, confidence_mask=None):
        """
        Posterior denoiser with optional per-pixel confidence weighting.

        Args:
            confidence_mask: (B, 1, H, W) or None.  Values in (0, 1] indicating
                observation confidence.  1.0 = real observation, < 1.0 = synthetic.
                When provided, effectively scales sigma_y per pixel:
                sigma_y_eff = sigma_y / sqrt(confidence)
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

        # Confidence weighting: scale residual by confidence
        if confidence_mask is not None:
            conf = confidence_mask.to(torch.float64)
            r = r * conf

        def cg_operator(v):
            if confidence_mask is not None:
                conf = confidence_mask.to(torch.float64)
                return sigma_y_sq * v + sigma_sq * mask * conf * vjp_fn(mask * conf * v)
            return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)

        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)

        if confidence_mask is not None:
            conf = confidence_mask.to(torch.float64)
            score = vjp_fn(mask * conf * v)
        else:
            score = vjp_fn(mask * v)

        result = denoised.detach() + sigma_sq * score.detach()
        return result

    def mmps_sample(self, latents, x_obs_img, mask_img, confidence_mask=None):
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

            denoised = self.posterior_denoise(x_hat, t_hat, x_obs, mask, confidence_mask)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised_2 = self.posterior_denoise(x_next, t_next, x_obs, mask, confidence_mask)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sampling_mmps(self, x_obs_img, mask_img, confidence_mask=None):
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, x_obs_img, mask_img, confidence_mask)

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
    for batch_idx, data in enumerate(train_loader):
        x_irregular = data[0]
        x_ts = x_irregular[:, :, :-1]
        mask = ~torch.isnan(x_ts).any(dim=-1)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())
    return np.vstack(all_corrupted), np.vstack(all_masks)


# =============================================================================
# Two-pass Gibbs E-step
# =============================================================================

def e_step_gibbs(args, uncond_model, corrupted_data, obs_masks,
                 em_iter, device, logger=None,
                 synthetic_confidence=0.3):
    """
    Gibbs-style two-pass E-step.

    Pass 1: Standard MMPS imputation using real observations only.
    Pass 2: Decompose Pass-1 output, create augmented observations
            (real obs + synthetic trend+seasonal at missing positions),
            re-run MMPS with confidence weighting.

    Args:
        synthetic_confidence: confidence weight for synthetic observations (0-1).
            Lower values = MMPS trusts synthetic obs less.
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — Gibbs Two-Pass MMPS ===")
    uncond_model.eval()

    all_recon_pass1 = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    N = len(corrupted_data)
    batch_size = args.batch_size
    sigma_y = getattr(args, 'mmps_sigma_y', 0.01)
    cg_iters = getattr(args, 'mmps_cg_iters', 1)

    # === Pass 1: Standard MMPS ===
    print("  Pass 1: Standard MMPS imputation...")
    with uncond_model.ema_scope():
        process = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="Pass 1"):
            end_idx = min(start_idx + batch_size, N)
            corrupted_batch = corrupted_data[start_idx:end_idx]
            mask_batch = obs_masks[start_idx:end_idx]

            corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
            mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)
            obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)

            x_obs_img = uncond_model.ts_to_img(obs_ts)
            mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
            mask_img = uncond_model.ts_to_img(mask_ts_expanded)[:, :1, :, :]

            x_img_imputed = process.sampling_mmps(x_obs_img, mask_img)
            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)
            all_recon_pass1.append(x_ts_recon.cpu().numpy())
            torch.cuda.empty_cache()

    recon_pass1 = np.vstack(all_recon_pass1)
    recon_pass1 = np.clip(recon_pass1, 0, 1)
    print(f"  Pass 1 complete. {len(recon_pass1)} sequences.")

    # === Decompose Pass-1 output ===
    print("  Decomposing Pass-1 output...")
    trends, seasonals, residuals = fast_decompose_batch(recon_pass1)
    structural_estimate = trends + seasonals  # trend + seasonal

    # === Pass 2: Augmented MMPS with synthetic observations ===
    print("  Pass 2: Augmented MMPS with synthetic trend+seasonal...")
    all_recon_pass2 = []

    with uncond_model.ema_scope():
        process2 = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="Pass 2"):
            end_idx = min(start_idx + batch_size, N)
            corrupted_batch = corrupted_data[start_idx:end_idx]
            mask_batch = obs_masks[start_idx:end_idx]
            struct_batch = structural_estimate[start_idx:end_idx]

            corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
            mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)
            struct_ts = torch.tensor(struct_batch, dtype=torch.float32, device=device)

            # Create augmented observations:
            # - Real observations at observed positions
            # - Synthetic trend+seasonal at missing positions
            obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)
            augmented_ts = obs_ts.clone()
            miss_mask_3d = (~mask_ts.bool()).unsqueeze(-1).expand_as(corrupted_ts)
            augmented_ts[miss_mask_3d] = struct_ts[miss_mask_3d]

            # All positions now "observed" (augmented mask = all ones)
            augmented_mask = torch.ones_like(mask_ts)

            # Confidence mask: 1.0 for real obs, synthetic_confidence for synthetic
            conf_ts = torch.where(
                mask_ts.bool(),
                torch.ones_like(mask_ts),
                torch.full_like(mask_ts, synthetic_confidence)
            )

            x_aug_img = uncond_model.ts_to_img(augmented_ts)
            aug_mask_expanded = augmented_mask.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
            aug_mask_img = uncond_model.ts_to_img(aug_mask_expanded)[:, :1, :, :]

            conf_expanded = conf_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
            conf_img = uncond_model.ts_to_img(conf_expanded)[:, :1, :, :]

            x_img_imputed = process2.sampling_mmps(x_aug_img, aug_mask_img, conf_img)
            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)
            all_recon_pass2.append(x_ts_recon.cpu().numpy())
            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_recon_pass2)
    reconstructions = np.clip(reconstructions, 0, 1)
    print(f"  Pass 2 complete. {len(reconstructions)} sequences.")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values!")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def m_step(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_loader = Data.DataLoader(
        Data.TensorDataset(recon_tensor),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    uncond_model.train()
    for epoch in range(args.m_step_epochs):
        epoch_loss, num_batches = 0, 0
        for (x_clean,) in recon_loader:
            x_clean = x_clean.to(device)
            x_img = uncond_model.ts_to_img(x_clean)
            loss, _ = uncond_model.loss_fn_irregular(x_img)
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
            logger.log('em/m_step_loss', avg_loss, em_iter * args.m_step_epochs + epoch)

    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, em_iter, device, logger=None):
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    uncond_model.eval()
    gen_sig, real_sig = [], []

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution, args.img_resolution)
            )
            for data in tqdm(test_loader, desc="Evaluating"):
                x_img = process.sampling(sampling_number=data[0].shape[0])
                gen_sig.append(uncond_model.img_to_ts(x_img).cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)
    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig, device=device,
        plot_path=f"memorization_hist_em_iter_{em_iter}.png"
    )
    for k, v in mem_stats.items():
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)
    if logger is not None:
        p = f"memorization_hist_em_iter_{em_iter}.png"
        try:
            logger.log_file('test/memorization/histogram', p, em_iter)
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    return scores


# =============================================================================
# Main
# =============================================================================

def main(args):
    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_gibbs_decomp_{name}"
    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate, weight_decay=args.weight_decay
        )
        print_model_params(logger, uncond_model)

        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences, "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        initial_reconstructions = initialize_with_stl(
            corrupted_data, obs_masks, seed=args.seed
        )

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )

        synthetic_confidence = getattr(args, 'gibbs_synthetic_confidence', 0.3)

        print(f"\n{'='*60}")
        print(f"Starting Gibbs-Decomposition DiffEM-MMPS ({args.em_iters} EM iters)")
        print(f"  Synthetic confidence: {synthetic_confidence}")
        print(f"{'='*60}")

        best_metrics = None
        last_recon = initial_reconstructions

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # Curriculum
            progress = min(1.0, (em_iter + 1) / max(args.em_iters * 0.6, 1))
            reveal_prob = (1.0 - progress) * 0.3
            extra_reveal = (np.random.rand(*obs_masks.shape) < reveal_prob) & ~obs_masks
            curriculum_masks = obs_masks | extra_reveal
            curriculum_corrupted = corrupted_data.copy()
            curriculum_corrupted[extra_reveal] = last_recon[extra_reveal]

            reconstructions = e_step_gibbs(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                em_iter, args.device, logger,
                synthetic_confidence=synthetic_confidence
            )
            last_recon = reconstructions

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
        print("Gibbs-Decomposition DiffEM-MMPS Complete!")
        if best_metrics:
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
