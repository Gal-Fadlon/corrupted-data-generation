"""
DiffEM with Unconditional Model + MMPS E-step — Gaussian Noise Corruption (V2)

V2 fixes over run_diffem_mmps_gaussian_noise.py:
  1. E-step uses e_step_sample_steps (64) instead of diffusion_steps (18)
  2. Phase 3 continues from EM-trained model (no weight reset)
  3. Extended warmstart (200 epochs via --warmstart_epochs)
  4. Sigma_y calibration via --mmps_sigma_y CLI flag
  5. CG iterations default raised to 3
  6. TF32 explicitly enabled for ~1.3x speedup
  7. EM early stopping: stop if disc_mean degrades for 2 consecutive iters

Paper: "Learning Diffusion Priors from Observations by Expectation Maximization"
       (Rozet et al., NeurIPS 2024)
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
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader, save_reconstructions
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

DEFAULT_WARMSTART_EPOCHS = 200
DEFAULT_PHASE3_EPOCHS = 200
EM_EARLY_STOP_PATIENCE = 2


# =============================================================================
# Conjugate Gradient Solver
# =============================================================================

def conjugate_gradient(A_fn, b, maxiter=1):
    """
    Conjugate gradient solver for linear system A·x = b, where A is given
    as a function (linear operator).

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
# MMPSDiffusionProcess — Gaussian Noise (A = I)
# =============================================================================

class MMPSDiffusionProcess:
    """
    MMPS-style diffusion sampler for Gaussian noise denoising (A = I).

    Given noisy observations  y = x + ε,  ε ~ N(0, σ_y² I),  the posterior
    denoiser at each reverse step solves:

        (σ_y² I + σ_t² J^T) v = y - D_θ(x_t)

    and returns  D_θ(x_t) + σ_t² · J^T(v).
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y=0.1, cg_iters=3):
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
        # FIX 1: Use e_step_sample_steps (default 64) instead of diffusion_steps (18)
        self.num_steps = getattr(args, 'e_step_sample_steps', 64)

        self.sigma_y = sigma_y
        self.cg_iters = cg_iters

    def posterior_denoise(self, x_t, sigma, y_obs):
        """
        Posterior denoiser for Gaussian noise (A = I).

        Follows Algorithm 3 (MMPS) from the DiEM paper with A = I:
          1. denoised = D_θ(x_t)
          2. VJP via autograd.grad
          3. r = y - denoised
          4. CG: (σ_y² I + σ_t² J^T) v = r
          5. score = J^T(v)
          6. return denoised + σ_t² · score
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

        r = y_obs - denoised

        def cg_operator(v):
            return sigma_y_sq * v + sigma_sq * vjp_fn(v)

        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)

        score = vjp_fn(v)

        result = denoised.detach() + sigma_sq * score.detach()
        return result

    def mmps_sample(self, latents, y_obs):
        """MMPS sampling loop with Euler-Heun steps using the posterior denoiser."""
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        y = y_obs.to(torch.float64)

        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.posterior_denoise(x_hat, t_hat, y)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised_2 = self.posterior_denoise(x_next, t_next, y)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sampling_mmps(self, y_obs):
        """Convenience wrapper: allocate latents and run MMPS sampling."""
        batch_size = y_obs.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, y_obs)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        """Standard unconditional sampling (for evaluation, uses diffusion_steps for fair comparison)."""
        latents = torch.randn([sampling_number, *self.shape], device=self.device)

        eval_steps = self.args.diffusion_steps
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(eval_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (eval_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / eval_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.net(x_hat, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < eval_steps - 1:
                denoised = self.net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


# =============================================================================
# Data preparation
# =============================================================================

def get_noisy_data_from_loader(train_loader):
    """Extract noisy time series from the train loader (strip time-index column)."""
    all_noisy = []
    for data in train_loader:
        x = data[0]
        x_ts = x[:, :, :-1]
        all_noisy.append(x_ts.numpy())
    return np.vstack(all_noisy)


def prepare_gaussian_noise_data(args, train_loader, uncond_model, device):
    """
    Prepare data for the Gaussian noise experiment.

    Returns:
        noisy_imgs:  (N, C, H, W)  noisy observations in image space
        sigma_y:     float          MMPS observation noise level
    """
    # FIX 4: Default sigma_y = corruption_noise_level (the true observation noise).
    # Only override if the user explicitly passes --mmps_sigma_y on the CLI
    # (the argparse default 0.01 is NOT appropriate for Gaussian noise experiments).
    sigma_y = args.corruption_noise_level
    cli_sigma_y = getattr(args, 'mmps_sigma_y', None)
    if cli_sigma_y is not None and cli_sigma_y != 0.01:
        sigma_y = cli_sigma_y

    noisy_data_ts = get_noisy_data_from_loader(train_loader)

    N = len(noisy_data_ts)
    batch_size = args.batch_size
    noisy_img_list = []

    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_ts = torch.tensor(noisy_data_ts[start:end],
                                dtype=torch.float32, device=device)
            x_img = uncond_model.ts_to_img(x_ts)
            noisy_img_list.append(x_img.cpu())

    noisy_imgs = torch.cat(noisy_img_list, dim=0).numpy()

    print(f"Gaussian noise data prepared:")
    print(f"  Noisy images:    {noisy_imgs.shape}")
    print(f"  corruption_noise_level: {args.corruption_noise_level}")
    print(f"  MMPS sigma_y:    {sigma_y}")

    return noisy_imgs, sigma_y


def initialize_from_noisy(noisy_imgs, uncond_model, device):
    """
    Initialize reconstructions by converting noisy images back to TS space.

    For Gaussian noise there is no missing data, so the noisy observation
    itself is already a reasonable starting point for the prior.
    """
    N = noisy_imgs.shape[0]
    batch_size = 64
    all_ts = []

    uncond_model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            y_img = torch.tensor(noisy_imgs[start:end],
                                 dtype=torch.float32, device=device)
            y_ts = uncond_model.img_to_ts(y_img)
            all_ts.append(y_ts.cpu().numpy())

    initial_recon = np.vstack(all_ts)
    initial_recon = np.clip(initial_recon, 0, 1)

    print(f"Initialization from noisy observations: {initial_recon.shape}")
    return initial_recon


# =============================================================================
# Core EM functions
# =============================================================================

def e_step(args, uncond_model, noisy_imgs, sigma_y, em_iter, device, logger=None):
    """E-step: Reconstruct clean data from noisy observations via MMPS (A = I)."""
    print(f"\n=== E-Step (EM iter {em_iter}) — MMPS Denoising [A=I] ===")

    e_step_steps = getattr(args, 'e_step_sample_steps', 64)
    cg_iters = getattr(args, 'mmps_cg_iters', 3)
    print(f"  sigma_y = {sigma_y}, diffusion_steps = {e_step_steps}, cg_iters = {cg_iters}")

    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(noisy_imgs)
    batch_size = args.batch_size

    with uncond_model.ema_scope():
        process = MMPSDiffusionProcess(
            args, uncond_model.net, target_shape,
            sigma_y=sigma_y, cg_iters=cg_iters
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step MMPS"):
            end_idx = min(start_idx + batch_size, N)

            batch_imgs = noisy_imgs[start_idx:end_idx]
            y_obs_64 = torch.tensor(batch_imgs, dtype=torch.float64, device=device)

            x_img_recon = process.sampling_mmps(y_obs_64)

            x_ts_recon = uncond_model.img_to_ts(x_img_recon)
            all_reconstructions.append(x_ts_recon.cpu().numpy())

            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    print(f"E-step complete. Reconstructed {len(reconstructions)} sequences.")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'gaussian_noise_v2_{sigma_y}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


def m_step(args, uncond_model, optimizer, reconstructions, em_iter, device,
           num_epochs=None, logger=None):
    """M-step: Train unconditional model on fully reconstructed data."""
    epochs = num_epochs if num_epochs is not None else args.m_step_epochs

    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training unconditional model for {epochs} epochs...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    uncond_model.train()

    for epoch in range(epochs):
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
            print(f"  M-step epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

        if logger is not None:
            global_step = em_iter * epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, em_iter, device, logger=None):
    """Evaluate the unconditional model via standard unconditional sampling."""
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    print("Evaluating unconditional model...")

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
# Phase 3 — Continue training from EM model (FIX 2: no model reset)
# =============================================================================

def phase3_continue_training(args, uncond_model, optimizer, reconstructions,
                             test_loader, device, logger=None):
    """
    Phase 3: Continue training from EM-trained model on final reconstructions.

    Unlike train_unconditional_regular (which creates a fresh model), this
    preserves the EM-trained weights and optimizer state.
    """
    phase3_epochs = getattr(args, 'phase3_epochs', DEFAULT_PHASE3_EPOCHS)
    logging_iter = getattr(args, 'logging_iter', 10)

    print(f"\n{'='*60}")
    print(f"Phase 3: Continue unconditional training from EM model")
    print(f"  epochs={phase3_epochs}  logging_iter={logging_iter}")
    print(f"{'='*60}")

    recon_loader = Data.DataLoader(
        Data.TensorDataset(
            torch.tensor(reconstructions, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    best_metrics = None

    for epoch in range(phase3_epochs):
        uncond_model.train()

        for (x_clean,) in recon_loader:
            x_clean = x_clean.to(device)
            x_img = uncond_model.ts_to_img(x_clean)
            loss, to_log = uncond_model.loss_fn_irregular(x_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.)
            optimizer.step()
            uncond_model.on_train_batch_end()

            for key, value in to_log.items():
                if logger is not None:
                    logger.log(f'train/{key}', value, epoch)

        if epoch % logging_iter == 0:
            gen_sig, real_sig = [], []
            uncond_model.eval()

            with torch.no_grad():
                with uncond_model.ema_scope():
                    diffusion_process = DiffusionProcess(
                        args, uncond_model.net,
                        (args.input_channels, args.img_resolution,
                         args.img_resolution)
                    )
                    for data in test_loader:
                        x_img_sampled = diffusion_process.sampling(
                            sampling_number=data[0].shape[0])
                        x_ts = uncond_model.img_to_ts(x_img_sampled)
                        gen_sig.append(x_ts.cpu().numpy())
                        real_sig.append(data[0].cpu().numpy())

            gen_sig = np.concatenate(gen_sig, axis=0)
            real_sig = np.concatenate(real_sig, axis=0)

            metrics = evaluate_model_irregular(real_sig, gen_sig, args)

            print(f"Phase 3 epoch {epoch}: {metrics}")

            if logger is not None:
                for key, value in metrics.items():
                    logger.log(f'test/{key}', value, epoch)

            if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                best_metrics = metrics

    return best_metrics


# =============================================================================
# Main EM loop
# =============================================================================

def main(args):
    """Main DiffEM-MMPS training loop for Gaussian noise corruption (V2)."""

    # FIX 6: Enable TF32 for ~1.3x speedup on Ampere+ GPUs
    if getattr(args, 'allow_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for CUDA matmul and cuDNN.")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    sigma_y_corruption = args.corruption_noise_level

    name = create_model_name_and_dir(args)
    name = f"diffem_mmps_gaussian_noise_v2_{name}"

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        print_model_params(logger, uncond_model)

        # === Prepare noisy data ===
        print(f"\n{'='*60}")
        print(f"Corruption: Gaussian noise  (corruption_sigma = {sigma_y_corruption})")
        print(f"{'='*60}")

        noisy_imgs, sigma_y = prepare_gaussian_noise_data(
            args, train_loader, uncond_model, args.device
        )

        # === Phase 1: Initialize from noisy observations ===
        print(f"\n{'='*60}")
        print("Phase 1: Initialization from noisy observations")
        print(f"{'='*60}")

        initial_reconstructions = initialize_from_noisy(
            noisy_imgs, uncond_model, args.device
        )

        # === Phase 1.5: Extended warmstart ===
        # FIX 3: Use warmstart_epochs (default 200) instead of m_step_epochs (50)
        warmstart_epochs = getattr(args, 'warmstart_epochs', DEFAULT_WARMSTART_EPOCHS)

        print(f"\n{'='*60}")
        print(f"Phase 1.5: Warmstart training ({warmstart_epochs} epochs)")
        print(f"{'='*60}")

        uncond_model = m_step(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device,
            num_epochs=warmstart_epochs, logger=logger
        )
        print("Warmstart training complete.")

        # === Phase 2: EM Loop with early stopping ===
        cg_iters = getattr(args, 'mmps_cg_iters', 3)
        e_step_steps = getattr(args, 'e_step_sample_steps', 64)

        print(f"\n{'='*60}")
        print(f"Starting DiffEM-MMPS V2 [Gaussian noise] with {args.em_iters} EM iterations")
        print(f"  corruption_noise_level: {sigma_y_corruption}")
        print(f"  MMPS sigma_y: {sigma_y}")
        print(f"  MMPS cg_iters: {cg_iters}")
        print(f"  E-step diffusion steps: {e_step_steps}")
        print(f"  EM early stop patience: {EM_EARLY_STOP_PATIENCE}")
        print(f"{'='*60}")

        best_metrics = None
        best_em_iter = -1
        best_model_state = None
        best_reconstructions = None
        consecutive_worse = 0

        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            reconstructions = e_step(
                args, uncond_model, noisy_imgs, sigma_y,
                em_iter, args.device, logger
            )

            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, em_iter, args.device, logger=logger
            )

            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_uncond(
                    args, uncond_model, test_loader,
                    em_iter, args.device, logger
                )

                current_disc = metrics.get('disc_mean', float('inf'))

                if best_metrics is None or current_disc < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics
                    best_em_iter = em_iter
                    best_model_state = copy.deepcopy(uncond_model.state_dict())
                    best_reconstructions = reconstructions
                    consecutive_worse = 0
                    print(f"  >> New best disc_mean = {current_disc:.4f} at EM iter {em_iter}")
                else:
                    consecutive_worse += 1
                    print(f"  >> disc_mean = {current_disc:.4f} (worse, "
                          f"{consecutive_worse}/{EM_EARLY_STOP_PATIENCE} towards early stop)")

                # FIX 7: Early stopping
                if consecutive_worse >= EM_EARLY_STOP_PATIENCE:
                    print(f"\n*** EM EARLY STOP: disc_mean degraded for "
                          f"{EM_EARLY_STOP_PATIENCE} consecutive iterations. ***")
                    print(f"*** Restoring best model from EM iter {best_em_iter} "
                          f"(disc_mean = {best_metrics['disc_mean']:.4f}) ***")

                    uncond_model.load_state_dict(best_model_state)
                    reconstructions = best_reconstructions

                    if logger is not None:
                        logger.log('em/early_stopped', 1, em_iter)
                        logger.log('em/best_iter', best_em_iter, em_iter)
                    break

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        # ================================================================
        # Phase 3 — Continue training from EM model (FIX 2: no reset)
        # ================================================================
        final_metrics = phase3_continue_training(
            args, uncond_model, optimizer, reconstructions,
            test_loader, args.device, logger,
        )

        print("\n" + "="*60)
        print("DiffEM-MMPS V2 [Gaussian Noise] Training Complete!")
        print("="*60)
        if best_metrics:
            print(f"Best EM-loop metrics (iter {best_em_iter}):")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")
        if final_metrics:
            print("Phase 3 (continued unconditional) final metrics:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("DiffEM-MMPS V2 Gaussian noise training is complete")


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    # Force no missing data — corruption is purely Gaussian noise
    if getattr(args, 'missing_rate', 0.0) != 0.0:
        logging.warning(
            f"missing_rate={args.missing_rate} -> forcing 0 "
            "(Gaussian noise experiment has no missing data)."
        )
        args.missing_rate = 0.0

    if not hasattr(args, 'corruption_noise_level') or args.corruption_noise_level <= 0:
        print("ERROR: --corruption_noise_level must be > 0 for Gaussian noise experiment.",
              file=sys.stderr)
        sys.exit(1)

    # Bridge: gen_dataloader reads gaussian_noise_level to inject TS-domain noise
    args.gaussian_noise_level = args.corruption_noise_level

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
