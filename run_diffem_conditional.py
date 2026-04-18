"""
DiffEM Conditional EM — Evaluate Conditional Model Directly (Approach B)

Same Phase 0-2 as run_diffem_cond_mmps_init.py but eliminates Phase 3 entirely.
Instead, evaluates the conditional model directly on the test set: given test
observations (y, mask), runs ConditionalDiffusionProcess.sampling() to produce
reconstructions, then computes disc_mean and other metrics vs ground truth.

This tests whether the conditional model q_theta(x|y) alone achieves competitive
generation quality without the Phase 3 unconditional model bottleneck.
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
from utils.utils import (
    create_model_name_and_dir,
    print_model_params, log_config_and_tags,
)
from utils.utils_data import (
    gen_dataloader, save_reconstructions,
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras, TS2img_Karras_Cond
from models.sampler import DiffusionProcess, ConditionalDiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

# ---------------------------------------------------------------------------
# Script-level defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_MMPS_BOOTSTRAP_ITERS = 3
DEFAULT_MMPS_BOOTSTRAP_EPOCHS = 50
DEFAULT_COND_M_STEP_EPOCHS = 100
DEFAULT_E_STEP_SAMPLE_STEPS = 64
DEFAULT_COND_EM_ITERS = 5


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log(logger, name, value, step):
    if logger is not None:
        logger.log(name, value, step)


def _log_stats(tag, arr, step, logger):
    arr = np.asarray(arr).flatten()
    _log(logger, f'{tag}/mean', float(np.nanmean(arr)), step)
    _log(logger, f'{tag}/std',  float(np.nanstd(arr)),  step)
    _log(logger, f'{tag}/min',  float(np.nanmin(arr)),  step)
    _log(logger, f'{tag}/max',  float(np.nanmax(arr)),  step)


# ===========================================================================
# Data extraction
# ===========================================================================

def get_corrupted_data_from_loader(train_loader, device):
    all_corrupted = []
    all_masks = []
    for batch_idx, data in enumerate(train_loader):
        x_irregular = data[0]
        x_ts = x_irregular[:, :, :-1]
        mask = ~torch.isnan(x_ts).any(dim=-1)
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())
    return np.vstack(all_corrupted), np.vstack(all_masks)


# ===========================================================================
# Phase 0: MMPS components (from run_diffem_warmstart.py / run_diffem_mmps.py)
# ===========================================================================

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
    return initial_recon


# ===========================================================================
# Phase 0: MMPS E-step and M-step
# ===========================================================================

def mmps_e_step(args, uncond_model, corrupted_data, obs_masks, em_iter,
                device, logger=None):
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
                             f'missing_{int(args.missing_rate * 100)}',
                             'cond_mmps_init')
    save_reconstructions(reconstructions, cache_dir, em_iter)
    _log(logger, 'mmps/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def mmps_m_step(args, uncond_model, optimizer, reconstructions, em_iter,
                device, m_step_epochs=None, logger=None):
    epochs = m_step_epochs or args.m_step_epochs
    print(f"\n=== MMPS M-Step (EM iter {em_iter}) — {epochs} epochs ===")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_loader = Data.DataLoader(
        Data.TensorDataset(recon_tensor),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )

    uncond_model.train()

    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
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
            n_batches += 1
            torch.cuda.empty_cache()

        avg = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  MMPS M-step epoch {epoch + 1}/{epochs}, loss: {avg:.4f}")
        _log(logger, 'mmps/m_step_loss', avg, em_iter * epochs + epoch)

    print(f"MMPS M-step complete for EM iter {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, step, device,
                    logger=None, log_prefix='test'):
    gen_sig, real_sig = [], []
    uncond_model.eval()

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

    print(f"  {log_prefix} (step {step}) metrics:")
    for k, v in scores.items():
        print(f"    {k}: {v:.4f}")
        _log(logger, f'{log_prefix}/{k}', v, step)

    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig,
        device=device, plot_path=None,
    )
    for k, v in mem_stats.items():
        if isinstance(v, float):
            _log(logger, f'{log_prefix}/memorization/{k}', v, step)

    return scores


# ===========================================================================
# Conditional evaluation (Approach B)
# ===========================================================================

def evaluate_conditional(args, cond_model, test_loader, step, device,
                         sample_steps=None, logger=None, log_prefix='test_cond'):
    """Evaluate the conditional model: given corrupted test data, reconstruct and score."""
    if sample_steps is None:
        sample_steps = getattr(args, 'e_step_sample_steps', DEFAULT_E_STEP_SAMPLE_STEPS)

    print(f"\n=== Conditional Evaluation (step {step}, {sample_steps} steps) ===")

    cond_model.eval()
    gen_sig, real_sig = [], []

    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    cond_channels = args.input_channels + 1

    orig_steps = args.diffusion_steps
    args.diffusion_steps = sample_steps

    rng = np.random.RandomState(args.seed + step)

    with torch.no_grad():
        with cond_model.ema_scope():
            process = ConditionalDiffusionProcess(
                args, cond_model.net, target_shape, cond_channels)

            for data in tqdm(test_loader, desc="Cond eval"):
                x_full = data[0]
                batch_size_cur = x_full.shape[0]
                x_ts = x_full[:, :, :-1] if x_full.shape[-1] > args.input_channels else x_full

                seq_len = x_ts.shape[1]
                num_remove = int(seq_len * args.missing_rate)

                all_recon = []
                for b in range(batch_size_cur):
                    x_single = x_ts[b].numpy().copy()
                    mask_single = np.ones(seq_len, dtype=np.float32)
                    if num_remove > 0:
                        remove_idx = rng.permutation(seq_len)[:num_remove]
                        x_single[remove_idx] = np.nan
                        mask_single[remove_idx] = 0.0
                    all_recon.append((x_single, mask_single))

                corr_np = np.stack([r[0] for r in all_recon])
                mask_np = np.stack([r[1] for r in all_recon])

                corr_ts = torch.tensor(corr_np, dtype=torch.float32, device=device)
                mask_ts = torch.tensor(mask_np, dtype=torch.float32, device=device)

                cond_ts = torch.nan_to_num(corr_ts, nan=0.0)
                cond_img = cond_model.ts_to_img(cond_ts)
                mask_exp = mask_ts.unsqueeze(-1).expand_as(corr_ts)
                mask_img = cond_model.ts_to_img(mask_exp)[:, :1, :, :]

                x_img = process.sampling(
                    cond_img, mask_img,
                    x_obs_img=cond_img,
                    obs_mask_img=mask_img,
                )
                x_recon_ts = cond_model.img_to_ts(x_img)
                gen_sig.append(x_recon_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    args.diffusion_steps = orig_steps

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    print(f"  {log_prefix} (step {step}) metrics:")
    for k, v in scores.items():
        print(f"    {k}: {v:.4f}")
        if logger is not None:
            logger.log(f'{log_prefix}/{k}', v, step)

    mem_stats = compute_memorization_metric(
        real_data=real_sig, generated_data=gen_sig,
        device=device, plot_path=None,
    )
    for k, v in mem_stats.items():
        if isinstance(v, float) and logger is not None:
            logger.log(f'{log_prefix}/memorization/{k}', v, step)

    return scores


# ===========================================================================
# FreshCorruptionDataset (from run_diffem_pure_combined.py)
# ===========================================================================

class FreshCorruptionDataset(Data.Dataset):
    """M-step dataset that generates a new random corruption on every access."""

    def __init__(self, clean_data, missing_rate=0.0, noise_level=0.0,
                 epoch_seed=0):
        self.clean = (clean_data.numpy() if isinstance(clean_data, torch.Tensor)
                      else np.asarray(clean_data, dtype=np.float32))
        self.missing_rate = missing_rate
        self.noise_level = noise_level
        self.epoch_seed = epoch_seed

    def set_epoch(self, epoch):
        self.epoch_seed = epoch

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        x = self.clean[idx].copy()
        rng = np.random.RandomState(
            (self.epoch_seed * len(self.clean) + idx) % (2**32 - 1))
        seq_len = x.shape[0]

        if self.noise_level > 0:
            corrupted = x + rng.normal(0, self.noise_level, x.shape).astype(np.float32)
        else:
            corrupted = x.copy()
        num_remove = int(seq_len * self.missing_rate)
        mask = np.ones(seq_len, dtype=np.float32)
        if num_remove > 0:
            remove_idx = rng.permutation(seq_len)[:num_remove]
            corrupted[remove_idx] = np.nan
            mask[remove_idx] = 0.0

        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(corrupted.astype(np.float32)),
                torch.from_numpy(mask))


# ===========================================================================
# Phase 2: Conditional EM components
# ===========================================================================

def cond_e_step(args, cond_model, corrupted_data, obs_masks, em_iter,
                device, sample_steps=None, logger=None):
    """Conditional E-step with repaint guidance and observation consistency."""
    if sample_steps is None:
        sample_steps = DEFAULT_E_STEP_SAMPLE_STEPS

    print(f"\n{'=' * 50}")
    print(f"Conditional E-Step (iter {em_iter})  steps={sample_steps}  repaint=ON")
    print(f"{'=' * 50}")

    cond_model.eval()
    all_recon = []

    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    cond_channels = args.input_channels + 1

    orig_steps = args.diffusion_steps
    args.diffusion_steps = sample_steps

    N = len(corrupted_data)
    batch_size = args.batch_size

    with torch.no_grad():
        with cond_model.ema_scope():
            process = ConditionalDiffusionProcess(
                args, cond_model.net, target_shape, cond_channels)

            for start in tqdm(range(0, N, batch_size), desc="E-step cond+repaint"):
                end = min(start + batch_size, N)
                corr_batch = corrupted_data[start:end]
                mask_batch = obs_masks[start:end]

                corr_ts = torch.tensor(corr_batch, dtype=torch.float32, device=device)
                mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)

                cond_ts = torch.nan_to_num(corr_ts, nan=0.0)
                cond_img = cond_model.ts_to_img(cond_ts)

                mask_exp = mask_ts.unsqueeze(-1).expand_as(corr_ts)
                mask_img = cond_model.ts_to_img(mask_exp)
                mask_img = mask_img[:, :1, :, :]

                x_img = process.sampling(
                    cond_img, mask_img,
                    x_obs_img=cond_img,
                    obs_mask_img=mask_img,
                )
                x_ts = cond_model.img_to_ts(x_img)
                x_np = x_ts.cpu().numpy()

                for b in range(len(corr_batch)):
                    obs = mask_batch[b].astype(bool)
                    orig = corr_batch[b][obs, :]
                    if not np.any(np.isnan(orig)):
                        x_np[b, obs, :] = orig

                all_recon.append(x_np)
                torch.cuda.empty_cache()

    args.diffusion_steps = orig_steps

    recon = np.vstack(all_recon)
    print(f"  Reconstructions: shape={recon.shape}, "
          f"range=[{recon.min():.4f}, {recon.max():.4f}]")
    _log_stats('cond_em/e_step_recon', recon, em_iter, logger)
    _log(logger, 'cond_em/e_step_samples', len(recon), em_iter)

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                             f'missing_{int(args.missing_rate * 100)}',
                             'cond_mmps_init')
    save_reconstructions(recon, cache_dir, 100 + em_iter)

    return recon


def cond_m_step(args, cond_model, optimizer, reconstructions, em_iter,
                device, m_step_epochs=None, logger=None):
    """Conditional M-step with FreshCorruptionDataset. No EMA/optimizer reset."""
    epochs = m_step_epochs or DEFAULT_COND_M_STEP_EPOCHS

    corruption_rate = (args.em_corruption_rate
                       if getattr(args, 'em_corruption_rate', None)
                       else args.missing_rate)
    noise_level = getattr(args, 'em_noise_level', None) or \
                  getattr(args, 'gaussian_noise_level', 0.0)

    print(f"\n{'=' * 50}")
    print(f"Conditional M-Step (iter {em_iter}) — {epochs} epochs, "
          f"missing_rate={corruption_rate}, noise={noise_level}")
    print(f"{'=' * 50}")

    dataset = FreshCorruptionDataset(
        reconstructions,
        missing_rate=corruption_rate,
        noise_level=noise_level,
        epoch_seed=args.seed + em_iter * 10000,
    )
    loader = Data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )

    cond_model.train()
    loss_history = []

    patience = getattr(args, 'early_stop_patience', 20)
    min_delta = getattr(args, 'early_stop_min_delta', 1e-4)
    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(epochs):
        dataset.set_epoch(args.seed + em_iter * 10000 + epoch)

        epoch_loss, n_batches = 0.0, 0
        for clean, corrupted, mask in loader:
            clean     = clean.to(device)
            corrupted = corrupted.to(device)
            mask      = mask.to(device)

            cond_ts      = torch.nan_to_num(corrupted, nan=0.0)
            x_img        = cond_model.ts_to_img(clean)
            cond_img     = cond_model.ts_to_img(cond_ts)
            mask_exp     = mask.unsqueeze(-1).expand_as(clean)
            obs_mask_img = cond_model.ts_to_img(mask_exp)
            obs_mask_img = obs_mask_img[:, :1, :, :]

            loss, _ = cond_model.loss_fn_conditional_full(
                x_img, cond_img, obs_mask_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cond_model.parameters(), 1.0)
            optimizer.step()
            cond_model.on_train_batch_end()

            epoch_loss += loss.item()
            n_batches  += 1
            torch.cuda.empty_cache()

        avg = epoch_loss / max(n_batches, 1)
        loss_history.append(avg)

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  epoch {epoch + 1:4d}/{epochs}  loss={avg:.5f}  lr={lr_now:.2e}")

        _log(logger, 'cond_em/m_step_loss', avg, em_iter * epochs + epoch)

        if avg < best_loss - min_delta:
            best_loss = avg
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience and epoch >= 30:
            print(f"  Early stopping at epoch {epoch + 1}/{epochs} "
                  f"(no improvement for {patience} epochs)")
            break

    first5 = np.mean(loss_history[:5]) if len(loss_history) >= 5 else loss_history[0]
    last5  = np.mean(loss_history[-5:])
    print(f"  M-step loss: first-5={first5:.5f} -> last-5={last5:.5f}")
    _log(logger, 'cond_em/m_step_loss_start', first5, em_iter)
    _log(logger, 'cond_em/m_step_loss_end',   last5,  em_iter)

    return cond_model


# ===========================================================================
# Main
# ===========================================================================

def main(args):
    if getattr(args, 'allow_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for CUDA matmul and cuDNN.")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    name = create_model_name_and_dir(args)
    name = f"diffem_conditional_{name}"

    logging.info(args)

    mmps_bootstrap_iters = getattr(args, 'mmps_bootstrap_iters', DEFAULT_MMPS_BOOTSTRAP_ITERS)
    mmps_bootstrap_epochs = getattr(args, 'mmps_bootstrap_epochs', DEFAULT_MMPS_BOOTSTRAP_EPOCHS)
    cond_em_iters = getattr(args, 'cond_em_iters', DEFAULT_COND_EM_ITERS)
    cond_m_step_epochs = getattr(args, 'cond_m_step_epochs', DEFAULT_COND_M_STEP_EPOCHS)
    e_step_sample_steps = getattr(args, 'e_step_sample_steps', DEFAULT_E_STEP_SAMPLE_STEPS)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(
            train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences with "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # ==================================================================
        # Phase 0 — MMPS Bootstrap
        # ==================================================================
        print(f"\n{'=' * 70}")
        print(f"PHASE 0: MMPS Bootstrap ({mmps_bootstrap_iters} EM iters, "
              f"{mmps_bootstrap_epochs} epochs/iter)")
        print(f"{'=' * 70}")

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        uncond_optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print_model_params(logger, uncond_model)

        initial_recon = initialize_with_gaussian(
            corrupted_data, obs_masks, seed=args.seed)

        uncond_model = mmps_m_step(
            args, uncond_model, uncond_optimizer, initial_recon,
            em_iter=-1, device=args.device,
            m_step_epochs=mmps_bootstrap_epochs, logger=logger,
        )

        reconstructions = initial_recon
        best_mmps_metrics = None

        for em_iter in range(mmps_bootstrap_iters):
            print(f"\n{'=' * 60}")
            print(f"MMPS EM Iteration {em_iter + 1}/{mmps_bootstrap_iters}")
            print(f"{'=' * 60}")

            reconstructions = mmps_e_step(
                args, uncond_model, corrupted_data, obs_masks,
                em_iter, args.device, logger,
            )

            uncond_model = mmps_m_step(
                args, uncond_model, uncond_optimizer, reconstructions,
                em_iter, args.device,
                m_step_epochs=mmps_bootstrap_epochs, logger=logger,
            )

            metrics = evaluate_uncond(
                args, uncond_model, test_loader,
                em_iter, args.device, logger, log_prefix='test',
            )
            if best_mmps_metrics is None or \
               metrics.get('disc_mean', float('inf')) < \
               best_mmps_metrics.get('disc_mean', float('inf')):
                best_mmps_metrics = metrics

            _log(logger, 'em/iteration', em_iter, em_iter)

        print(f"\n{'=' * 60}")
        print("Phase 0 (MMPS Bootstrap) Complete!")
        print(f"{'=' * 60}")
        if best_mmps_metrics:
            print("Best MMPS metrics:")
            for k, v in best_mmps_metrics.items():
                print(f"  {k}: {v:.4f}")

        # ==================================================================
        # Phase 1 — Initial Conditional Training
        # ==================================================================
        print(f"\n{'=' * 70}")
        print(f"PHASE 1: Initial Conditional Model Training "
              f"({cond_m_step_epochs} epochs on MMPS reconstructions)")
        print(f"{'=' * 70}")

        cond_model = TS2img_Karras_Cond(
            args=args, device=args.device).to(args.device)
        cond_optimizer = torch.optim.AdamW(
            cond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        cond_params = sum(p.numel() for p in cond_model.parameters())
        print(f"Conditional model parameters: {cond_params:,}")

        cond_model = cond_m_step(
            args, cond_model, cond_optimizer, reconstructions,
            em_iter=-1, device=args.device,
            m_step_epochs=cond_m_step_epochs, logger=logger,
        )

        # ==================================================================
        # Phase 2 — Conditional EM with Repaint Guidance
        # ==================================================================
        print(f"\n{'=' * 70}")
        print(f"PHASE 2: Conditional EM ({cond_em_iters} iters, "
              f"{e_step_sample_steps} E-step steps, "
              f"{cond_m_step_epochs} M-step epochs, repaint=ON, no EMA reset)")
        print(f"{'=' * 70}")

        for cond_iter in range(cond_em_iters):
            print(f"\n{'=' * 60}")
            print(f"Conditional EM Iteration {cond_iter + 1}/{cond_em_iters}")
            print(f"{'=' * 60}")

            reconstructions = cond_e_step(
                args, cond_model, corrupted_data, obs_masks,
                cond_iter, args.device,
                sample_steps=e_step_sample_steps, logger=logger,
            )

            cond_model = cond_m_step(
                args, cond_model, cond_optimizer, reconstructions,
                cond_iter, args.device,
                m_step_epochs=cond_m_step_epochs, logger=logger,
            )

            _log(logger, 'cond_em/iteration', cond_iter, cond_iter)

        # ==================================================================
        # Evaluate conditional model directly (no Phase 3)
        # ==================================================================
        print(f"\n{'=' * 70}")
        print("EVALUATION: Conditional model on test set (no Phase 3)")
        print(f"{'=' * 70}")

        cond_metrics = evaluate_conditional(
            args, cond_model, test_loader,
            step=cond_em_iters, device=args.device,
            sample_steps=e_step_sample_steps, logger=logger,
            log_prefix='test_cond',
        )

        uncond_final_metrics = evaluate_uncond(
            args, uncond_model, test_loader,
            step=mmps_bootstrap_iters + cond_em_iters,
            device=args.device, logger=logger, log_prefix='test',
        )

        # ==================================================================
        # Summary
        # ==================================================================
        print(f"\n{'=' * 70}")
        print("DiffEM Conditional EM — No Phase 3 — Complete")
        print(f"{'=' * 70}")
        if best_mmps_metrics:
            print("Best MMPS bootstrap metrics:")
            for k, v in best_mmps_metrics.items():
                print(f"  {k}: {v:.4f}")
        if cond_metrics:
            print("Conditional model evaluation metrics:")
            for k, v in cond_metrics.items():
                print(f"  {k}: {v:.4f}")
        if uncond_final_metrics:
            print("MMPS uncond model (no further training) metrics:")
            for k, v in uncond_final_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("Training complete.")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    args = parse_args_irregular()
    args.use_diffem = True

    if getattr(args, 'mmps_bootstrap_iters', None) is None:
        args.mmps_bootstrap_iters = DEFAULT_MMPS_BOOTSTRAP_ITERS
    if getattr(args, 'mmps_bootstrap_epochs', None) is None:
        args.mmps_bootstrap_epochs = DEFAULT_MMPS_BOOTSTRAP_EPOCHS
    if getattr(args, 'cond_em_iters', None) is None:
        args.cond_em_iters = DEFAULT_COND_EM_ITERS
    if getattr(args, 'cond_m_step_epochs', None) is None:
        args.cond_m_step_epochs = DEFAULT_COND_M_STEP_EPOCHS
    if getattr(args, 'e_step_sample_steps', None) is None:
        args.e_step_sample_steps = DEFAULT_E_STEP_SAMPLE_STEPS

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    main(args)
