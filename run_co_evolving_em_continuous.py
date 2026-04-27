"""
Co-Evolving EM for Time Series Generation from Continuous-Time Irregular Sampling.

Specialization of `run_co_evolving_em.py` for the `continuous` corruption type.
The observation operator is `ContinuousResampleOperator` (see obs_operators.py):
per-sample 2-tap linear interpolation at N_obs continuous timestamps drawn in
[0, T-1]. Degenerates exactly to `PointMaskOperator` when t_k is integer.

This script wires the operator-generalized E/M steps into the same
infrastructure we use for block/missing runs:
  * start_gpu_heartbeat() to defeat the cluster idle-GPU monitor
  * init / bootstrap / per-E / per-M wall-clock logging
  * maybe_save_if_improved on strict disc improvement (atomic save + old-best prune)
  * mid-M-step eval via the run_eval_and_save closure and m_step's eval_callback
  * cum_em_at_best_seconds -> time_to_best_disc (inc/exc init)

Save tree (via utils.utils_save.target_parent_dir with corruption='continuous'):
  {save_models_root}/{method}/continuous/{t_distribution}/{dataset}/
  seq_len_{N}/rho_{int(n_obs_ratio*100)}/{run_name}/

Sampling distribution: --t_distribution {uniform|poisson|clustered}
Observation density:    --n_obs_ratio rho in (0, 1]
Interpolation kernel:   --interp_kernel linear   (v1 only)

Paper failure modes -> fixes map (unchanged from the missing/aggregate runners):
  5.1  Cold start             -> Kalman / STL / Kalman-snap warm start
  5.2  CG in the wrong space  -> Observation-Space CG (Prop 4) in R^{N_obs x F}
  5.3  CG instability         -> Adaptive sigma_y = c*sigma_t (Prop 2)
  5.4  Off-manifold drift (E) -> Manifold projection + op.enforce()
  5.5  Off-manifold drift (M) -> L_rep: SNR-gated manifold penalty
  5.6  Observation forgetting -> L_obs: operator-exact ||S D_theta^ts - y||^2
  5.7  CG across steps        -> Warm-started CG from previous reverse step
"""

import torch
import torch.autograd
import torch.multiprocessing
import torch.utils.data as Data
import os
import sys
import time
import numpy as np
import logging
from tqdm import tqdm

from metrics import evaluate_model_irregular
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.train_unconditional import train_unconditional_regular
from utils.utils_stl import (
    initialize_with_iterative_stl,
    initialize_with_kalman,
    initialize_with_kalman_continuous,
)
from utils.utils_save import maybe_save_if_improved
from utils.gpu_heartbeat import start_gpu_heartbeat, stop_gpu_heartbeat

from obs_operators import (
    ContinuousResampleOperator,
    ObservationOperator,
    PointMaskOperator,
    build_operator,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Conjugate Gradient Solver (Section 5.7: warm-start support)
# =============================================================================

def conjugate_gradient(A_fn, b, x0=None, maxiter=1):
    if x0 is not None:
        x = x0
        r = b - A_fn(x)
    else:
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
# Dual-Space MMPS Sampler (operator-aware; Sections 5.2-5.4, 5.7)
# =============================================================================

class DualSpaceMMPS:
    """Operator-aware MMPS posterior sampler. Uses obs-space CG over R^{N_obs x F}
    via S = ContinuousResampleOperator, with warm-start between reverse steps.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y_ratio=0.1, cg_iters=5,
                 ts_to_img_fn=None, img_to_ts_fn=None,
                 use_adaptive_sigma_y=True, use_consistency_projection=True,
                 use_obs_space_cg=True, use_warm_start_cg=True,
                 sigma_y_floor=0.0,
                 op: ObservationOperator = None):
        self.op = op
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

        self.sigma_y_ratio = sigma_y_ratio
        self.cg_iters = cg_iters
        self.sigma_y_floor = sigma_y_floor

        self.ts_to_img_fn = ts_to_img_fn
        self.img_to_ts_fn = img_to_ts_fn

        self.use_obs_space_cg = use_obs_space_cg
        self.use_warm_start_cg = use_warm_start_cg
        self.v_prev = None

        self.use_adaptive_sigma_y = use_adaptive_sigma_y
        self.use_consistency_projection = use_consistency_projection

    def _get_sigma_y_sq(self, sigma):
        if self.use_adaptive_sigma_y:
            sigma_y = self.sigma_y_ratio * sigma
            if self.sigma_y_floor > 0:
                sigma_y = torch.clamp(sigma_y, min=self.sigma_y_floor)
            return sigma_y ** 2
        return self.sigma_y_ratio ** 2

    def posterior_denoise_obs_space(self, x_t, sigma, obs_ts):
        """Obs-space CG (Prop 4) over R^{N_obs x F}. Single VJP per CG iter."""
        sigma_sq = sigma ** 2
        sigma_y_sq = self._get_sigma_y_sq(sigma)

        x_t_input = x_t.detach().requires_grad_(True)
        denoised_img = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent_img):
            grad, = torch.autograd.grad(
                denoised_img, x_t_input, grad_outputs=cotangent_img,
                retain_graph=True,
            )
            return grad

        denoised_ts = self.img_to_ts_fn(denoised_img.float()).to(torch.float64)
        y_obs = obs_ts.to(torch.float64)
        r_obs = y_obs - self.op.apply(denoised_ts)

        def cg_operator_obs(v_ts):
            a_t_v = self.op.adjoint(v_ts).to(torch.float64)
            v_img = self.ts_to_img_fn(a_t_v.float()).to(torch.float64)
            Jv_img = vjp_fn(v_img)
            Jv_ts = self.img_to_ts_fn(Jv_img.float()).to(torch.float64)
            Jv_obs = self.op.apply(Jv_ts)
            return sigma_y_sq * v_ts + sigma_sq * Jv_obs

        x0 = None
        if self.use_warm_start_cg and self.v_prev is not None:
            try:
                if self.v_prev.shape == r_obs.shape:
                    x0 = self.v_prev
            except Exception:
                pass

        v_ts = conjugate_gradient(cg_operator_obs, r_obs, x0=x0, maxiter=self.cg_iters)

        if self.use_warm_start_cg:
            self.v_prev = v_ts.detach()

        a_t_v = self.op.adjoint(v_ts).to(torch.float64)
        v_img = self.ts_to_img_fn(a_t_v.float()).to(torch.float64)
        score_img = vjp_fn(v_img)

        return denoised_img.detach() + sigma_sq * score_img.detach()

    def _consistency_project(self, x_img, obs_ts):
        """Manifold projection + op.enforce Moore-Penrose update."""
        x_ts = self.img_to_ts_fn(x_img.float())
        x_ts_enforced = self.op.enforce(x_ts, obs_ts.float()).float()
        return self.ts_to_img_fn(x_ts_enforced).to(torch.float64)

    def mmps_sample(self, latents, obs_ts):
        self.v_prev = None

        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        obs_ts_d = obs_ts.to(torch.float64)

        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.posterior_denoise_obs_space(x_hat, t_hat, obs_ts_d)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised_2 = self.posterior_denoise_obs_space(x_next, t_next, obs_ts_d)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        if self.use_consistency_projection:
            x_next = self._consistency_project(x_next, obs_ts_d)

        return x_next

    def sampling_mmps(self, obs_ts):
        batch_size = obs_ts.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, obs_ts)


# =============================================================================
# Continuous observation extraction helpers
# =============================================================================

def build_continuous_corruption(seed, rho, dist, nts):
    """Draw N_obs = round(rho * nts) sample times per sample.

    Returns sampler(N) -> [N, N_obs] float in [0, T-1], sorted per row.
    """
    n_obs = max(1, int(round(rho * nts)))
    t_max = float(nts - 1)
    rng = np.random.RandomState(seed)

    def _sample(N: int) -> np.ndarray:
        if dist == 'uniform':
            ts = rng.uniform(0.0, t_max, size=(N, n_obs))
        elif dist == 'poisson':
            overs = int(np.ceil(n_obs * 1.5))
            rate = n_obs / max(t_max, 1.0)
            ts = rng.exponential(1.0 / rate, size=(N, overs)).cumsum(axis=-1)
            ts = np.clip(ts, 0.0, t_max)
            ts = ts[:, :n_obs]
            if ts.shape[1] < n_obs:
                pad = rng.uniform(0.0, t_max, size=(N, n_obs - ts.shape[1]))
                ts = np.concatenate([ts, pad], axis=-1)
        elif dist == 'clustered':
            half = n_obs // 2
            rest = n_obs - half
            c1 = rng.normal(t_max * 0.25, t_max / 8.0, size=(N, half))
            c2 = rng.normal(t_max * 0.75, t_max / 8.0, size=(N, rest))
            ts = np.concatenate([c1, c2], axis=-1)
            ts = np.clip(ts, 0.0, t_max)
        else:
            raise ValueError(
                f"build_continuous_corruption: unknown --t_distribution "
                f"{dist!r}. Supported: uniform, poisson, clustered."
            )
        ts.sort(axis=-1)
        return ts.astype(np.float32)

    _sample.n_obs = n_obs
    return _sample


def get_continuous_data_from_loader(train_loader, rho, dist, seed, device):
    """Extract clean TS, draw per-sample timestamps, apply S.

    Returns (y_obs [N, N_obs, F], timestamps [N, N_obs], x_clean [N, T, F]).
    """
    all_clean = []
    for data in train_loader:
        x_irregular = data[0]
        x_ts = x_irregular[:, :, :-1]
        if torch.isnan(x_ts).any():
            raise ValueError(
                "get_continuous_data_from_loader: training loader contains "
                "NaNs; --corruption_type continuous expects a clean source. "
                "Set --missing_rate 0.0 and verify dataset config."
            )
        all_clean.append(x_ts.numpy())

    x_clean = np.vstack(all_clean).astype(np.float32)
    N, T, F = x_clean.shape

    sampler = build_continuous_corruption(seed=seed, rho=rho, dist=dist, nts=T)
    timestamps = sampler(N)
    N_obs = timestamps.shape[1]
    print(f"  [continuous] rho={rho}, dist={dist}, N_obs={N_obs}/{T}")

    t = torch.as_tensor(timestamps, dtype=torch.float32)
    x = torch.as_tensor(x_clean, dtype=torch.float32)
    op = ContinuousResampleOperator(seq_len=T, n_features=F, n_obs=N_obs)
    op.set_grid(t)
    with torch.no_grad():
        y_obs = op.apply(x).cpu().numpy()

    return y_obs, timestamps, x_clean


def off_manifold_energy_batch(x_img, img_to_ts_fn, ts_to_img_fn):
    with torch.no_grad():
        x_ts_rt = img_to_ts_fn(x_img.float())
        x_img_rt = ts_to_img_fn(x_ts_rt)
        return ((x_img.float() - x_img_rt) ** 2).sum(dim=(1, 2, 3)).mean().item()


# =============================================================================
# E-step (Sections 5.2-5.4, 5.7)
# =============================================================================

def e_step(args, uncond_model, y_obs_all, timestamps_all, em_iter, device,
           logger=None, op: ContinuousResampleOperator = None):
    """E-step: impute full-grid reconstructions from (y_obs, timestamps) via MMPS."""
    assert op is not None, "e_step requires a ContinuousResampleOperator"
    print(f"\n=== E-Step (EM iter {em_iter}) ===")

    uncond_model.eval()

    target_shape = (uncond_model.num_features, args.img_resolution, args.img_resolution)
    N = len(y_obs_all)
    batch_size = args.batch_size

    sigma_y_ratio = getattr(args, 'sigma_y_ratio', 0.1)
    cg_iters = getattr(args, 'mmps_cg_iters', 5)
    use_adaptive = getattr(args, 'adaptive_sigma_y', True)
    use_proj = getattr(args, 'consistency_projection', True)
    use_obs_cg = getattr(args, 'obs_space_cg', True)
    use_warm_cg = getattr(args, 'warm_start_cg', True)
    sigma_y_floor = getattr(args, 'sigma_y_floor', 0.0)

    all_reconstructions = []
    total_off_energy = 0.0
    n_off_samples = 0

    with uncond_model.ema_scope():
        process = DualSpaceMMPS(
            args, uncond_model.net, target_shape,
            sigma_y_ratio=sigma_y_ratio,
            cg_iters=cg_iters,
            ts_to_img_fn=uncond_model.ts_to_img,
            img_to_ts_fn=uncond_model.img_to_ts,
            use_adaptive_sigma_y=use_adaptive,
            use_consistency_projection=use_proj,
            use_obs_space_cg=use_obs_cg,
            use_warm_start_cg=use_warm_cg,
            sigma_y_floor=sigma_y_floor,
            op=op,
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step"):
            end_idx = min(start_idx + batch_size, N)

            y_batch = torch.tensor(
                y_obs_all[start_idx:end_idx], dtype=torch.float32, device=device,
            )
            ts_batch = torch.tensor(
                timestamps_all[start_idx:end_idx], dtype=torch.float32, device=device,
            )
            op.set_grid(ts_batch)

            x_img_imputed = process.sampling_mmps(obs_ts=y_batch)

            cur_bs = end_idx - start_idx
            batch_off = off_manifold_energy_batch(
                x_img_imputed, uncond_model.img_to_ts, uncond_model.ts_to_img,
            )
            total_off_energy += batch_off * cur_bs
            n_off_samples += cur_bs

            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)
            all_reconstructions.append(x_ts_recon.cpu().numpy())

            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    off_energy = total_off_energy / max(n_off_samples, 1)
    print(f"  Completed {len(reconstructions)} sequences. E_off = {off_energy:.6f}")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    if logger is not None:
        logger.log('em/off_manifold_energy', off_energy, em_iter)
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


# =============================================================================
# M-step (Sections 5.5, 5.6) with optional mid-step eval callback
# =============================================================================

def m_step(args, uncond_model, optimizer, reconstructions,
           y_obs_all, timestamps_all, em_iter, device, logger=None,
           op: ContinuousResampleOperator = None,
           eval_callback=None, eval_every_epochs=None):
    """M-step: L = L_SM + lambda_obs * L_obs + lambda_rep * L_rep.

    L_obs uses the operator-exact form ||S * D_theta^ts - y||^2; S is set per
    batch via op.set_grid(timestamps_batch).

    If eval_callback and eval_every_epochs > 0, eval_callback(epoch) is invoked
    after every `eval_every_epochs` training epochs (and always at the final
    epoch if it lands on the cadence). Returns (model, total_eval_time_in_mstep).
    """
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"  Training for {args.m_step_epochs} epochs...")

    lambda_obs = getattr(args, 'lambda_obs', 0.5)
    lambda_rep = getattr(args, 'lambda_rep', 0.0)
    use_snr_gate = not getattr(args, 'no_snr_gate', False)
    use_sm_weight_for_lobs = getattr(args, 'lobs_use_sm_weight', False)
    snr_sigma_d = getattr(args, 'snr_gate_sigma_d', None)
    sigma_data = uncond_model.sigma_data
    sigma_d_for_gate = snr_sigma_d if snr_sigma_d is not None else sigma_data

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    y_tensor = torch.tensor(y_obs_all, dtype=torch.float32)     # [N, N_obs, F]
    ts_tensor = torch.tensor(timestamps_all, dtype=torch.float32)  # [N, N_obs]

    recon_dataset = Data.TensorDataset(recon_tensor, y_tensor, ts_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    uncond_model.train()

    total_eval_time_in_mstep = 0.0

    for epoch in range(args.m_step_epochs):
        epoch_loss_sm = 0.0
        epoch_loss_obs = 0.0
        epoch_loss_rep = 0.0
        num_batches = 0

        for (x_clean, y_ts_batch, ts_batch) in recon_loader:
            x_clean = x_clean.to(device)
            y_ts_batch = y_ts_batch.to(device)
            ts_batch = ts_batch.to(device)

            if lambda_obs > 0:
                op.set_grid(ts_batch)

            x_img = uncond_model.ts_to_img(x_clean)

            rnd_normal = torch.randn([x_img.shape[0], 1, 1, 1], device=device)
            sigma = (rnd_normal * uncond_model.P_std + uncond_model.P_mean).exp()
            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
            noise = torch.randn_like(x_img) * sigma
            denoised = uncond_model.net(x_img + noise, sigma, None)

            x_unpad = uncond_model.unpad(x_img, x_img.shape)
            denoised_unpad = uncond_model.unpad(denoised, x_img.shape)
            loss_sm = (weight * (denoised_unpad - x_unpad).square()).mean()

            if use_snr_gate:
                snr_gate = sigma_d_for_gate ** 2 / (sigma.squeeze() ** 2 + sigma_d_for_gate ** 2)
                if use_sm_weight_for_lobs:
                    snr_gate_obs = snr_gate * weight.squeeze()
                else:
                    snr_gate_obs = snr_gate
            else:
                snr_gate = torch.ones(x_img.shape[0], device=device)
                snr_gate_obs = snr_gate

            if lambda_obs > 0:
                denoised_ts = uncond_model.img_to_ts(denoised)
                sd_ts = op.apply(denoised_ts)                         # [B, N_obs, F]
                obs_residual = sd_ts - y_ts_batch
                loss_obs_per_sample = (obs_residual ** 2).mean(dim=(-1, -2))
                loss_obs = (snr_gate_obs * loss_obs_per_sample).mean()
            else:
                loss_obs = torch.tensor(0.0, device=device)

            if lambda_rep > 0:
                denoised_proj = uncond_model.ts_to_img(uncond_model.img_to_ts(denoised))
                off_manifold = denoised - denoised_proj
                loss_rep_per_sample = (off_manifold ** 2).mean(dim=(1, 2, 3))
                loss_rep = (snr_gate * loss_rep_per_sample).mean()
            else:
                loss_rep = torch.tensor(0.0, device=device)

            loss = loss_sm + lambda_obs * loss_obs + lambda_rep * loss_rep

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss_sm += loss_sm.item()
            epoch_loss_obs += loss_obs.item()
            epoch_loss_rep += loss_rep.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_sm = epoch_loss_sm / num_batches
        avg_obs = epoch_loss_obs / num_batches
        avg_rep = epoch_loss_rep / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.m_step_epochs}  "
                  f"L_SM={avg_sm:.4f}  L_obs={avg_obs:.4f}  L_rep={avg_rep:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/loss_sm', avg_sm, global_step)
            logger.log('em/loss_obs', avg_obs, global_step)
            logger.log('em/loss_rep', avg_rep, global_step)
            logger.log('em/loss_total',
                       avg_sm + lambda_obs * avg_obs + lambda_rep * avg_rep,
                       global_step)

        if eval_callback is not None and eval_every_epochs and (epoch + 1) % eval_every_epochs == 0:
            _eval_start = time.time()
            eval_callback(epoch)
            total_eval_time_in_mstep += time.time() - _eval_start

    print(f"  M-step complete for EM iteration {em_iter}.")
    return uncond_model, total_eval_time_in_mstep


def m_step_bootstrap(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """L_SM-only M-step for the warm-start bootstrap (Section 5.1)."""
    stop_gpu_heartbeat()
    print(f"\n=== Bootstrap M-Step (EM iter {em_iter}) -- L_SM only ===")
    print(f"  Training for {args.m_step_epochs} epochs...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    uncond_model.train()

    for epoch in range(args.m_step_epochs):
        epoch_loss = 0.0
        num_batches = 0

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
            print(f"  Epoch {epoch+1}/{args.m_step_epochs}, L_SM = {avg_loss:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/bootstrap_loss', avg_loss, global_step)

    print(f"  Bootstrap M-step complete for EM iteration {em_iter}.")
    return uncond_model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_uncond(args, uncond_model, test_loader, em_iter, device, logger=None,
                    m_epoch=None):
    """Unconditional sampling + metrics. Returns (scores, real_sig, gen_sig)."""
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")

    uncond_model.eval()
    gen_sig = []
    real_sig = []

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (uncond_model.num_features, args.img_resolution, args.img_resolution),
            )

            for data in tqdm(test_loader, desc="Evaluating"):
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = uncond_model.img_to_ts(x_img_sampled)
                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    if m_epoch is None:
        m_epoch_for_step = args.m_step_epochs - 1
    else:
        m_epoch_for_step = m_epoch
    eval_step = em_iter * args.m_step_epochs + m_epoch_for_step

    print(f"  EM iter {em_iter}, m_epoch {m_epoch_for_step} metrics:")
    for key, value in scores.items():
        print(f"    {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, eval_step)

    return scores, real_sig, gen_sig


# =============================================================================
# CLI shim for continuous-only flags (keeps utils_args.py untouched for these)
# =============================================================================

def _parse_continuous_args():
    """Strip continuous-only flags out of sys.argv and return them."""
    import argparse as _argparse
    p = _argparse.ArgumentParser(add_help=False)
    p.add_argument(
        '--n_obs_ratio', type=float, default=0.5,
        help='Density rho in (0, 1]. N_obs = round(rho * seq_len) per sample.',
    )
    p.add_argument(
        '--t_distribution', type=str, default='uniform',
        choices=['uniform', 'poisson', 'clustered'],
        help='Distribution of per-sample observation times {t_k}.',
    )
    p.add_argument(
        '--interp_kernel', type=str, default='linear',
        choices=['linear', 'cubic', 'sinc'],
        help='Interpolation kernel of S. v1: linear only.',
    )
    p.add_argument(
        '--continuous_curriculum', action='store_true',
        help='Enable timestamp-curriculum: draw extra pi(k)*T continuous '
             'timestamps per EM iter, evaluate prev reconstruction at them, '
             'augment (y_obs, timestamps).',
    )
    extra, remainder = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remainder
    return extra


# =============================================================================
# Main
# =============================================================================

def main(args):
    name = create_model_name_and_dir(args)
    name = f"co_evolving_em_continuous_{name}"

    logging.info(args)

    # Start GPU heartbeat BEFORE long CPU-bound init (Kalman warm-start, spline
    # computation, data loading) so the cluster idle-GPU monitor doesn't kill us.
    start_gpu_heartbeat()

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        print_model_params(logger, uncond_model)

        # --- Build the operator (continuous) ---
        n_obs_ratio = float(getattr(args, 'n_obs_ratio', 0.5))
        t_distribution = str(getattr(args, 't_distribution', 'uniform'))
        interp_kernel = str(getattr(args, 'interp_kernel', 'linear'))
        seq_len = args.seq_len
        n_features = args.input_channels
        n_obs = max(1, int(round(n_obs_ratio * seq_len)))

        op = build_operator(
            corruption_type='continuous',
            seq_len=seq_len,
            n_features=n_features,
            n_obs=n_obs,
            interp_kernel=interp_kernel,
        )
        print(f"[operator] continuous name={op.name} n_obs={op.n_obs} "
              f"rho={n_obs_ratio} dist={t_distribution} kernel={interp_kernel}")

        # --- Extract continuous observations ---
        print(f"Extracting clean TS and applying continuous-time resampling "
              f"(rho={n_obs_ratio}, dist={t_distribution})...")
        y_obs, timestamps_all, _x_clean_dbg = get_continuous_data_from_loader(
            train_loader, rho=n_obs_ratio, dist=t_distribution,
            seed=args.seed, device=args.device,
        )
        print(f"  {len(y_obs)} sequences, y shape={y_obs.shape}, "
              f"timestamps shape={timestamps_all.shape}")

        # --- Oracle disc sanity check on S^+(S real) ---
        try:
            oracle_sampler = build_continuous_corruption(
                seed=args.seed + 9999, rho=n_obs_ratio, dist=t_distribution, nts=seq_len,
            )
            op_oracle = ContinuousResampleOperator(
                seq_len=seq_len, n_features=n_features, n_obs=n_obs,
            )
            oracle_reals, oracle_rt = [], []
            with torch.no_grad():
                for data in test_loader:
                    xb = data[0][:, :, :-1]
                    B_ = xb.shape[0]
                    ts_b = torch.as_tensor(oracle_sampler(B_), dtype=torch.float32)
                    op_oracle.set_grid(ts_b)
                    xb32 = xb.float()
                    y_b = op_oracle.apply(xb32)
                    x_rt = op_oracle.enforce(torch.zeros_like(xb32), y_b)
                    oracle_reals.append(xb32.detach().cpu().numpy())
                    oracle_rt.append(x_rt.detach().cpu().numpy())
                    if sum(r.shape[0] for r in oracle_reals) >= 512:
                        break
            oracle_reals = np.vstack(oracle_reals)
            oracle_rt = np.vstack(oracle_rt)
            oracle_scores = evaluate_model_irregular(
                oracle_reals, oracle_rt, args, calc_other_metrics=False,
            )
            oracle_disc = float(oracle_scores.get('disc_mean', float('nan')))
            print(f"[oracle] disc_mean(real, S^+ S real) = {oracle_disc:.4f} "
                  f"(rho={n_obs_ratio}, dist={t_distribution})")
            if logger is not None:
                logger.log('test/disc_oracle_continuous', oracle_disc, 0)
        except Exception as e:
            print(f"[disc_oracle_continuous] skipped: {e}")

        # --- Section 5.1: Warm start ---
        init_method = getattr(args, 'init_method', 'stl')
        print(f"\n{'='*60}")
        print(f"Section 5.1: Warm Start ({init_method})")
        print(f"{'='*60}")

        init_start_time = time.time()

        if init_method == 'kalman':
            initial_reconstructions = initialize_with_kalman_continuous(
                y_obs, timestamps_all, seq_len=seq_len, seed=args.seed,
                per_fit_timeout=getattr(args, 'kalman_fit_timeout', 5),
                max_seconds=getattr(args, 'kalman_global_timeout', 1800),
            )
        elif init_method == 'kalman_snap':
            # Snap continuous timestamps to nearest integer grid cell, build a
            # NaN-masked [N, T, F] array, route through the PointMask-style
            # Kalman init that handles missing values natively.
            print("Continuous warm start: Kalman with snap-to-grid")
            N_total = y_obs.shape[0]
            F_chan = y_obs.shape[-1]
            grid_series = np.full((N_total, seq_len, F_chan), np.nan, dtype=np.float32)
            grid_mask = np.zeros((N_total, seq_len), dtype=bool)
            for i in range(N_total):
                t_i = timestamps_all[i]
                y_i = y_obs[i]
                idx = np.clip(np.rint(t_i).astype(np.int64), 0, seq_len - 1)
                sum_buf = np.zeros((seq_len, F_chan), dtype=np.float64)
                cnt_buf = np.zeros(seq_len, dtype=np.int64)
                for k, g in enumerate(idx):
                    sum_buf[g] += y_i[k]
                    cnt_buf[g] += 1
                hit = cnt_buf > 0
                grid_mask[i] = hit
                if hit.any():
                    grid_series[i, hit] = (sum_buf[hit] / cnt_buf[hit, None]).astype(np.float32)
            initial_reconstructions = initialize_with_kalman(
                grid_series, grid_mask, seed=args.seed,
                per_fit_timeout=getattr(args, 'kalman_fit_timeout', 5),
                max_seconds=getattr(args, 'kalman_global_timeout', 1800),
            )
        else:
            # STL / replicate: use op.warm_start (piecewise-linear interp + optional STL).
            use_stl_warm = init_method != 'replicate'
            print(f"Continuous warm start: linear interp (use_stl={use_stl_warm})")
            initial_reconstructions = op.warm_start(
                y_obs, timestamps_all, seed=args.seed,
                use_stl=use_stl_warm,
                stl_period=getattr(args, 'stl_period', None),
            )

        init_seconds = time.time() - init_start_time
        init_minutes = init_seconds / 60.0
        if logger is not None:
            logger.log('time/init_minutes', init_minutes, 0)
        print(f"  Init ({init_method}) took {init_minutes:.2f} min")

        # --- Bootstrap M-step (L_SM only; stops heartbeat) ---
        print(f"\n{'='*60}")
        print("Section 5.1: Bootstrap M-Step on warm-start completions")
        print(f"{'='*60}")

        bootstrap_start_time = time.time()
        uncond_model = m_step_bootstrap(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger,
        )
        bootstrap_seconds = time.time() - bootstrap_start_time
        bootstrap_minutes = bootstrap_seconds / 60.0
        if logger is not None:
            logger.log('time/bootstrap_minutes', bootstrap_minutes, 0)
        print(f"  Bootstrap M-step took {bootstrap_minutes:.2f} min")

        # --- Configuration summary ---
        sigma_y_ratio = getattr(args, 'sigma_y_ratio', 0.1)
        lambda_obs = getattr(args, 'lambda_obs', 0.5)
        lambda_rep = getattr(args, 'lambda_rep', 0.0)
        use_adaptive = getattr(args, 'adaptive_sigma_y', True)
        use_proj = getattr(args, 'consistency_projection', True)
        use_obs_cg = getattr(args, 'obs_space_cg', True)
        use_warm_cg = getattr(args, 'warm_start_cg', True)
        sigma_y_floor = getattr(args, 'sigma_y_floor', 0.0)
        use_snr_gate = not getattr(args, 'no_snr_gate', False)
        do_phase3 = getattr(args, 'phase3', True)

        print(f"\n{'='*60}")
        print(f"Co-Evolving EM (continuous) -- {args.em_iters} iterations")
        print(f"  Operator:              {op.name}  N_obs={op.n_obs}  rho={n_obs_ratio}  dist={t_distribution}")
        print(f"  5.1  Warm start:       {init_method}")
        print(f"  5.2  Obs-space CG:     {use_obs_cg}")
        print(f"  5.3  Adaptive sigma_y: {use_adaptive} (c={sigma_y_ratio}, floor={sigma_y_floor})")
        print(f"  5.4  Manifold proj:    {use_proj}")
        print(f"  5.5  L_rep:            lambda_rep={lambda_rep}")
        print(f"  5.6  L_obs:            lambda_obs={lambda_obs}, snr_gate={use_snr_gate}")
        print(f"  5.7  Warm-start CG:    {use_warm_cg}")
        print(f"  Phase 3 (fresh):       {do_phase3}")
        print(f"  CG iters:              {args.mmps_cg_iters}")
        print(f"{'='*60}")

        best_metrics = None
        metrics_history = []
        last_recon = initial_reconstructions
        em_start_time = time.time()

        # Convergence tracking: snapshots whenever disc strictly improves.
        cumulative_em_seconds = 0.0
        best_disc_value = None
        best_disc_em_iter = None
        best_disc_m_epoch = None
        cum_em_at_best_seconds = None

        current_em_iter = [0]

        def run_eval_and_save(m_epoch_in_step):
            em = current_em_iter[0]
            metrics, real_sig, gen_sig = evaluate_uncond(
                args, uncond_model, test_loader, em, args.device, logger,
                m_epoch=m_epoch_in_step,
            )

            metrics_history.append({
                'em_iter': em,
                'm_epoch': m_epoch_in_step,
                **metrics,
            })

            current_disc = metrics.get('disc_mean', float('inf'))
            nonlocal best_disc_value, best_disc_em_iter, best_disc_m_epoch, cum_em_at_best_seconds
            if best_disc_value is None or current_disc < best_disc_value:
                best_disc_value = current_disc
                best_disc_em_iter = em
                best_disc_m_epoch = m_epoch_in_step
                cum_em_at_best_seconds = cumulative_em_seconds

            nonlocal best_metrics
            if best_metrics is None or current_disc < best_metrics.get('disc_mean', float('inf')):
                best_metrics = metrics

            if logger is not None:
                global_epoch = em * args.m_step_epochs + m_epoch_in_step
                logger.log('time/em_iter_at_eval', em, global_epoch)
                logger.log('time/m_epoch_at_eval', m_epoch_in_step, global_epoch)

            maybe_save_if_improved(
                args, uncond_model, optimizer,
                real_sig, gen_sig, last_recon,
                metrics_history, em, logger,
                current_disc=current_disc, m_epoch=m_epoch_in_step,
            )

        m_eval_every = getattr(args, 'm_eval_every_epochs', 0) or 0

        # --- EM loop ---
        for em_iter in range(args.em_iters):
            current_em_iter[0] = em_iter
            iter_start_time = time.time()

            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # Continuous curriculum: draw pi(k)*T extra timestamps, evaluate
            # last_recon at them via S, augment (y, ts) for this EM iter.
            op_iter = op
            y_iter = y_obs
            ts_iter = timestamps_all
            if getattr(args, 'continuous_curriculum', False):
                progress = min(1.0, (em_iter + 1) / max(args.em_iters * 0.6, 1))
                reveal_frac = (1.0 - progress) * 0.3
                n_extra = int(round(reveal_frac * seq_len))
                if n_extra > 0:
                    extra_sampler = build_continuous_corruption(
                        seed=args.seed + 12345 + em_iter,
                        rho=n_extra / float(seq_len),
                        dist=t_distribution,
                        nts=seq_len,
                    )
                    N_total = y_obs.shape[0]
                    ts_extra = extra_sampler(N_total).astype(np.float32)
                    if ts_extra.shape[1] >= n_extra:
                        ts_extra = ts_extra[:, :n_extra]
                        op_extra = ContinuousResampleOperator(
                            seq_len=seq_len, n_features=n_features, n_obs=n_extra,
                        )
                        op_extra.set_grid(torch.as_tensor(ts_extra, dtype=torch.float32))
                        with torch.no_grad():
                            y_extra = op_extra.apply(
                                torch.as_tensor(last_recon, dtype=torch.float32)
                            ).cpu().numpy().astype(np.float32)
                        y_iter = np.concatenate([y_obs, y_extra], axis=1)
                        ts_iter = np.concatenate([timestamps_all, ts_extra], axis=1)
                        op_iter = ContinuousResampleOperator(
                            seq_len=seq_len, n_features=n_features,
                            n_obs=y_obs.shape[1] + n_extra,
                        )
                        print(f"  Continuous curriculum: +{n_extra} timestamps "
                              f"(progress={progress:.2f}, N_obs {y_obs.shape[1]} "
                              f"-> {y_iter.shape[1]})")

            # --- E-step ---
            e_step_start = time.time()
            reconstructions = e_step(
                args, uncond_model, y_iter, ts_iter,
                em_iter, args.device, logger, op=op_iter,
            )
            e_step_seconds = time.time() - e_step_start
            last_recon = reconstructions

            uncond_model.reset_ema()

            # --- M-step with optional mid-step eval ---
            m_step_start = time.time()
            uncond_model, eval_time_in_mstep = m_step(
                args, uncond_model, optimizer,
                reconstructions, y_iter, ts_iter,
                em_iter, args.device, logger, op=op_iter,
                eval_callback=run_eval_and_save if m_eval_every > 0 else None,
                eval_every_epochs=m_eval_every if m_eval_every > 0 else None,
            )
            m_step_seconds_incl_eval = time.time() - m_step_start
            m_step_seconds = m_step_seconds_incl_eval - eval_time_in_mstep

            iter_elapsed = time.time() - iter_start_time
            total_elapsed = time.time() - em_start_time
            cumulative_em_seconds += e_step_seconds + m_step_seconds
            print(f"  Wall-clock: {iter_elapsed:.1f}s (total: {total_elapsed:.1f}s) "
                  f"[E={e_step_seconds:.1f}s, M={m_step_seconds:.1f}s, "
                  f"eval-in-M={eval_time_in_mstep:.1f}s]")

            if logger is not None:
                logger.log('em/wall_clock_seconds', iter_elapsed, em_iter)
                logger.log('em/total_wall_clock', total_elapsed, em_iter)
                logger.log('time/e_step_minutes', e_step_seconds / 60.0, em_iter)
                logger.log('time/m_step_minutes', m_step_seconds / 60.0, em_iter)

            is_last = (em_iter == args.em_iters - 1)
            if m_eval_every <= 0:
                if (em_iter + 1) % args.em_eval_interval == 0 or is_last:
                    run_eval_and_save(args.m_step_epochs - 1)

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        total_em_time = time.time() - em_start_time
        print(f"\nTotal EM wall-clock: {total_em_time:.1f}s ({total_em_time/3600:.2f}h)")
        if logger is not None:
            logger.log('em/total_em_wall_clock_hours', total_em_time / 3600, 0)

        # --- Convergence timing ---
        if cum_em_at_best_seconds is not None:
            time_to_best_exc_init_seconds = bootstrap_seconds + cum_em_at_best_seconds
            time_to_best_inc_init_seconds = init_seconds + time_to_best_exc_init_seconds
            global_epoch_at_best = (
                best_disc_em_iter * args.m_step_epochs
                + (best_disc_m_epoch if best_disc_m_epoch is not None else 0)
            )
            print(f"\nConvergence summary (EM-only):")
            print(f"  Best disc: {best_disc_value:.4f} at em_iter={best_disc_em_iter}, "
                  f"m_epoch={best_disc_m_epoch}")
            print(f"  Time to best disc (inc init): {time_to_best_inc_init_seconds/60.0:.2f} min")
            print(f"  Time to best disc (exc init): {time_to_best_exc_init_seconds/60.0:.2f} min")
            if logger is not None:
                logger.log('time/time_to_best_disc_inc_init_minutes',
                           time_to_best_inc_init_seconds / 60.0, 0)
                logger.log('time/time_to_best_disc_exc_init_minutes',
                           time_to_best_exc_init_seconds / 60.0, 0)
                logger.log('time/em_iters_to_best_disc', best_disc_em_iter, 0)
                logger.log('time/m_epoch_at_best_disc',
                           best_disc_m_epoch if best_disc_m_epoch is not None else 0, 0)
                logger.log('time/global_epoch_at_best_disc', global_epoch_at_best, 0)
                logger.log('time/best_disc_value', best_disc_value, 0)

        # --- Phase 3: fresh model on final completions ---
        if do_phase3:
            print(f"\n{'='*60}")
            print("Phase 3: Fresh model on final completions")
            print(f"{'='*60}")
            phase3_metrics = train_unconditional_regular(
                args, last_recon, test_loader, args.device, logger,
            )
            if phase3_metrics:
                phase3_disc = phase3_metrics.get('disc_mean', float('inf'))
                em_disc = best_metrics.get('disc_mean', float('inf')) if best_metrics else float('inf')
                print(f"  Phase 3 disc_mean: {phase3_disc:.4f} (EM best: {em_disc:.4f})")
                if phase3_disc < em_disc:
                    best_metrics = phase3_metrics

        print("\n" + "=" * 60)
        print("Co-Evolving EM (continuous) complete!")
        print("=" * 60)
        if best_metrics:
            print("Best metrics:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("Co-evolving EM training (continuous) is complete")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    cont_args = _parse_continuous_args()
    args = parse_args_irregular()
    for _k, _v in vars(cont_args).items():
        setattr(args, _k, _v)

    # This script specializes in continuous corruption; force the save-tree
    # routing and argparse invariants consistent with that.
    args.corruption_type = 'continuous'
    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
