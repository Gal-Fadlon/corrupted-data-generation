"""
Best-disc-so-far model persistence.

Saves a full checkpoint (weights + optimizer + RNG + args) only when the
current run's disc_mean is strictly better than the best existing save in the
target directory. On improvement, pred/fid/correlation are computed, the run
name is derived from the 4 metrics, and the old save is replaced atomically.

Layout:
    {save_models_root}/{method}/{corruption_type}/[{missing_type}/]
    {dataset}/seq_len_{N}/{config_subdir}/{run_name}/
"""

import json
import os
import shutil

import numpy as np
import torch
import yaml

from metrics import evaluate_model_irregular


# =============================================================================
# Path / filesystem helpers
# =============================================================================

def _chmod_777(path):
    try:
        os.chmod(path, 0o777)
    except Exception:
        pass


def _makedirs_777(path):
    """Create directory (and parents) and force 0o777 on every new segment."""
    os.makedirs(path, exist_ok=True)
    parts = path.split(os.sep)
    for i in range(1, len(parts) + 1):
        sub = os.sep.join(parts[:i]) or os.sep
        if os.path.isdir(sub):
            _chmod_777(sub)


def _chmod_777_recursive(root):
    _chmod_777(root)
    # Walk without following dir symlinks; explicitly skip file symlinks too,
    # because os.chmod follows symlinks on Linux and we don't want to touch
    # targets outside this tree (e.g., the live slurm log file).
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for d in dirnames:
            full = os.path.join(dirpath, d)
            if not os.path.islink(full):
                _chmod_777(full)
        for f in filenames:
            full = os.path.join(dirpath, f)
            if not os.path.islink(full):
                _chmod_777(full)


# =============================================================================
# Target dir + run-name parsing
# =============================================================================

def target_parent_dir(args):
    """Compute the parent dir that holds (at most) one run dir at a time.

    Returns None if saving is disabled (empty save_models_root).
    """
    root = getattr(args, 'save_models_root', None)
    if not root:
        return None

    method = getattr(args, 'method', 'our')
    corruption = getattr(args, 'corruption_type', 'missing_observation')
    missing_type = getattr(args, 'missing_type', 'fix_missing_rates')
    dataset = args.dataset
    seq_len = args.seq_len
    missing_rate = getattr(args, 'missing_rate', 0.0)
    noise_level = getattr(args, 'gaussian_noise_level', 0.0)

    parts = [root, method, corruption]

    if corruption == 'missing_observation':
        parts.append(missing_type)
        parts.append(dataset)
        parts.append(f"seq_len_{seq_len}")
        parts.append(f"missing_{int(missing_rate * 100)}")
    elif corruption == 'noisy_observations':
        parts.append(dataset)
        parts.append(f"seq_len_{seq_len}")
        parts.append(f"noise_{noise_level}")
    elif corruption == 'combined_missing_noise':
        parts.append(missing_type)
        parts.append(dataset)
        parts.append(f"seq_len_{seq_len}")
        parts.append(f"missing_{int(missing_rate * 100)}_noise_{noise_level}")
    elif corruption == 'continuous':
        # Time-continuous irregular sampling (ContinuousResampleOperator).
        # Route by the sampling distribution and observation density rho.
        t_distribution = getattr(args, 't_distribution', 'uniform')
        n_obs_ratio = getattr(args, 'n_obs_ratio', 0.5)
        parts.append(t_distribution)
        parts.append(dataset)
        parts.append(f"seq_len_{seq_len}")
        parts.append(f"rho_{int(float(n_obs_ratio) * 100)}")
    else:
        parts.append(dataset)
        parts.append(f"seq_len_{seq_len}")

    return os.path.join(*parts)


def _format_run_name(disc, pred, fid, corr):
    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.4f}"

    return f"disc_{fmt(disc)}_pred_{fmt(pred)}_fid_{fmt(fid)}_corr_{fmt(corr)}"


def _parse_disc_from_dirname(name):
    """Extract the disc value from a dir name like 'disc_0.0123_pred_...'."""
    prefix = "disc_"
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix):]
    token = tail.split("_", 1)[0]
    try:
        return float(token)
    except ValueError:
        return None


def find_best_existing(parent_dir):
    """Return (best_disc, best_dir_abspath) among sibling run dirs, or (None, None)."""
    if not os.path.isdir(parent_dir):
        return None, None

    best_disc = None
    best_path = None
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        d = _parse_disc_from_dirname(name)
        if d is None:
            continue
        if best_disc is None or d < best_disc:
            best_disc = d
            best_path = full
    return best_disc, best_path


# =============================================================================
# Checkpoint construction
# =============================================================================

def _args_to_dict(args):
    try:
        return dict(vars(args))
    except TypeError:
        return {k: getattr(args, k) for k in dir(args) if not k.startswith('_')}


def _build_checkpoint(args, uncond_model, optimizer, em_iter, best_metrics, extra_state=None):
    ckpt = {
        'net_ema': None,
        'net_raw': None,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'ema_step': getattr(uncond_model, 'ema_step', None),
        'args': _args_to_dict(args),
        'embedder_stats': {},
        'em_iter': em_iter,
        'best_metrics': best_metrics if best_metrics is not None else {},
        'rng_torch': torch.get_rng_state(),
        'rng_numpy': np.random.get_state(),
        'rng_torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    # EMA weights (inference) and raw weights (resume).
    # Use uncond_model.net.state_dict() when present (TS2img_Karras-style models);
    # otherwise fall back to uncond_model.state_dict() so plain nn.Module baselines
    # (e.g., kovae's VK) also work.
    state_target = uncond_model.net if hasattr(uncond_model, 'net') else uncond_model
    ema_net = None
    if hasattr(uncond_model, 'ema_scope'):
        with uncond_model.ema_scope():
            ema_net = {k: v.detach().cpu().clone() for k, v in state_target.state_dict().items()}
    if ema_net is not None:
        ckpt['net_ema'] = ema_net
    ckpt['net_raw'] = {k: v.detach().cpu().clone() for k, v in state_target.state_dict().items()}

    # Embedder stats (best-effort STFT min/max cache capture)
    for attr in ('stft_min', 'stft_max'):
        if hasattr(uncond_model, attr):
            val = getattr(uncond_model, attr)
            if val is not None:
                ckpt['embedder_stats'][attr] = val

    # Baseline-specific or extra modules (e.g., ImageI2R's TST encoder/decoder/optimizer).
    if extra_state:
        ckpt.update(extra_state)

    return ckpt


# =============================================================================
# Atomic save
# =============================================================================

_SLURM_LOG_DIR_CANDIDATES = (
    '/home/galfad/corrupted-data-generation/slurm_logs',
)


def _symlink_slurm_logs(final_dir):
    """Create symlinks in `final_dir` pointing to the live slurm stdout/stderr for
    the current SLURM job. Symlinks track appends, so reading them later always
    shows the final log contents. No-op outside slurm or if the log files don't
    exist yet.
    """
    job_id = os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID')
    if not job_id:
        return

    for log_dir in _SLURM_LOG_DIR_CANDIDATES:
        out_src = os.path.join(log_dir, f"{job_id}.out")
        err_src = os.path.join(log_dir, f"{job_id}.err")
        if not os.path.isfile(out_src):
            continue
        for src, link_name in ((out_src, 'slurm_stdout.log'),
                                (err_src, 'slurm_stderr.log')):
            link_path = os.path.join(final_dir, link_name)
            if os.path.lexists(link_path):
                try:
                    os.remove(link_path)
                except Exception:
                    pass
            if os.path.isfile(src):
                try:
                    os.symlink(src, link_path)
                except Exception as e:
                    print(f"[save] WARNING: slurm log symlink failed ({link_name}): {e}")
        return  # stop at first working candidate


def _atomic_save(parent_dir, run_name, checkpoint, config_dict, reconstructions, metrics_history):
    """Write all files to a temp sibling dir, then rename to the final run dir."""
    _makedirs_777(parent_dir)

    tmp_name = f".tmp_{os.getpid()}_{run_name}"
    tmp_dir = os.path.join(parent_dir, tmp_name)
    final_dir = os.path.join(parent_dir, run_name)

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    _makedirs_777(tmp_dir)

    torch.save(checkpoint, os.path.join(tmp_dir, 'checkpoint.pt'))

    with open(os.path.join(tmp_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=True)

    if reconstructions is not None:
        np.savez_compressed(os.path.join(tmp_dir, 'final_reconstructions.npz'),
                            reconstructions=reconstructions)

    with open(os.path.join(tmp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2, default=str)

    _chmod_777_recursive(tmp_dir)

    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)

    os.rename(tmp_dir, final_dir)
    _symlink_slurm_logs(final_dir)
    _chmod_777_recursive(final_dir)
    return final_dir


# =============================================================================
# Top-level entry point
# =============================================================================

def maybe_save_if_improved(args, uncond_model, optimizer,
                           real_sig, gen_sig, last_recon,
                           metrics_history, em_iter, logger,
                           current_disc, m_epoch=None,
                           extra_state=None, log_step=None):
    """Called after each evaluation. If current_disc is strictly better than the
    best existing save in the target parent dir, computes pred+fid+correlation
    on the provided (real_sig, gen_sig) samples, atomically saves a new
    checkpoint, and deletes the prior save.

    Pred/fid/correlation are ALSO logged under the `test/` prefix (same as
    disc in evaluate_uncond) so wandb graphs are consistent.

    Returns the final saved dir path if a save happened, else None.
    """
    parent_dir = target_parent_dir(args)
    if parent_dir is None:
        return None

    best_disc, best_path = find_best_existing(parent_dir)
    if best_disc is not None and current_disc >= best_disc:
        print(f"[save] disc {current_disc:.4f} >= best {best_disc:.4f} at {best_path}; skipping save.")
        return None

    print(f"[save] disc improved (current={current_disc:.4f}, "
          f"best_on_disk={best_disc if best_disc is not None else 'none'}). "
          f"Computing pred/fid/correlation on the shared eval sample and saving...")

    scores = evaluate_model_irregular(real_sig, gen_sig, args, calc_other_metrics=True)

    # Log pred/fid/correlation under test/ prefix (same namespace as test/disc_mean).
    if logger is not None:
        if log_step is not None:
            eval_step = log_step
        else:
            m_epoch_for_step = m_epoch if m_epoch is not None else (args.m_step_epochs - 1)
            eval_step = em_iter * args.m_step_epochs + m_epoch_for_step
        for key, value in scores.items():
            logger.log(f'test/{key}', value, eval_step)
    # calc_other_metrics=True returns pred/fid/correlation (not disc). Disc we have already.
    pred_mean = scores.get('pred_score_mean')
    fid_mean = scores.get('fid_score_mean')
    corr_mean = scores.get('correlation_score_mean')

    run_name = _format_run_name(current_disc, pred_mean, fid_mean, corr_mean)
    print(f"[save] run_name = {run_name}")

    full_metrics = dict(scores)
    full_metrics['disc_mean'] = current_disc
    full_metrics['em_iter'] = em_iter
    if m_epoch is not None:
        full_metrics['m_epoch'] = m_epoch

    checkpoint = _build_checkpoint(args, uncond_model, optimizer, em_iter,
                                   best_metrics=full_metrics, extra_state=extra_state)
    config_dict = _args_to_dict(args)
    history_payload = {
        'history': metrics_history,
        'final': full_metrics,
    }

    final_dir = _atomic_save(parent_dir, run_name, checkpoint,
                             config_dict, last_recon, history_payload)

    if best_path is not None and os.path.isdir(best_path) and best_path != final_dir:
        try:
            shutil.rmtree(best_path)
            print(f"[save] removed prior best: {best_path}")
        except Exception as e:
            print(f"[save] WARNING: could not remove prior best {best_path}: {e}")

    print(f"[save] wrote {final_dir}")

    return final_dir
