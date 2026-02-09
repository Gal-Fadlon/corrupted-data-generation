"""
Pure DiffEM Training Script for Irregular Time Series

This script implements the exact DiffEM algorithm from the paper:
"Learning from Corrupted Data with Diffusion Models via Expectation Maximization"

Key differences from run_diffem.py:
- NO TST (Time Series Transformer) - pure DiffEM
- Conditioning is directly on (zero-filled corrupted data, binary mask)
- First iteration initialized with Gaussian prior

Algorithm:
1. Initialize with Gaussian prior
2. E-step: Sample X ~ q_theta(x|Y) where Y = (zero_filled_corrupted, mask)
3. M-step: Train conditional model on (X, Y~Q(X)) pairs
4. Repeat for K iterations
5. (Optional) Train unconditional model on final reconstructions
"""

import torch
import torch.multiprocessing
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import os
import sys
import numpy as np
import logging
from tqdm import tqdm

from metrics import evaluate_model_irregular
from metrics.memorization import compute_memorization_metric
from neptune.types import File
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import (
    gen_dataloader, apply_corruption, 
    EMReconstructionDataset, save_reconstructions, load_reconstructions
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras, TS2img_Karras_Cond
from models.sampler import DiffusionProcess, ConditionalDiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def get_corrupted_data_from_loader(train_loader, device):
    """
    Extract corrupted time series (with NaNs) from the train loader.
    
    Args:
        train_loader: dataloader with irregular observations
        device: target device
    
    Returns:
        corrupted_data: numpy array (N, seq_len, features) with NaN for missing
        obs_masks: numpy array (N, seq_len) with True for observed
    """
    all_corrupted = []
    all_masks = []
    
    for batch_idx, data in enumerate(train_loader):
        x_irregular = data[0]  # (batch, seq_len, features+1)
        x_ts = x_irregular[:, :, :-1]  # Remove time index column
        
        # Create mask: True where observed
        mask = ~torch.isnan(x_ts).any(dim=-1)  # (batch, seq_len)
        
        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())
    
    corrupted_data = np.vstack(all_corrupted)
    obs_masks = np.vstack(all_masks)
    
    return corrupted_data, obs_masks


def initialize_with_gaussian(corrupted_data, obs_masks, seed=None):
    """
    Initialize reconstructions using Gaussian prior.
    
    For observed values: use the actual observed data
    For missing values: sample from Gaussian fitted on observed data
    
    This follows the DiffEM paper's Gaussian initialization strategy.
    
    Args:
        corrupted_data: numpy array (N, seq_len, features) with NaN for missing
        obs_masks: numpy array (N, seq_len) with True for observed
        seed: random seed
    
    Returns:
        initial_recon: numpy array (N, seq_len, features) with all values filled
    """
    if seed is not None:
        np.random.seed(seed)
    
    N, seq_len, features = corrupted_data.shape
    
    # Compute mean and std from observed data
    observed_values = corrupted_data[~np.isnan(corrupted_data)]
    mu = np.nanmean(corrupted_data, axis=(0, 1))  # Per-feature mean
    std = np.nanstd(corrupted_data, axis=(0, 1))  # Per-feature std
    
    # Handle edge case where std is 0
    std = np.where(std == 0, 1.0, std)
    
    # Create initial reconstructions
    initial_recon = corrupted_data.copy()
    
    # Fill missing values with Gaussian samples
    for i in range(N):
        for t in range(seq_len):
            if not obs_masks[i, t]:
                # Sample from N(mu, std) for each feature
                initial_recon[i, t, :] = np.random.normal(mu, std)
    
    # Clip to [0, 1] since data is normalized
    initial_recon = np.clip(initial_recon, 0, 1)
    
    print(f"Gaussian initialization complete. Generated {N} sequences.")
    print(f"  Mean per feature: {mu}")
    print(f"  Std per feature: {std}")
    
    return initial_recon


def e_step(args, cond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    """
    E-step: Sample reconstructions X ~ q_theta(x|y) for all observations.
    
    Pure DiffEM: Condition on (zero-filled corrupted data, binary mask).
    NO TST completion.
    
    Args:
        args: configuration arguments
        cond_model: conditional diffusion model
        corrupted_data: numpy array (N, seq_len, features) with NaN for missing
        obs_masks: numpy array (N, seq_len) with True for observed
        em_iter: current EM iteration number
        device: target device
        logger: optional logger
    
    Returns:
        reconstructions: numpy array of reconstructed sequences (N, seq_len, features)
    """
    print(f"\n=== E-Step (EM iter {em_iter}) ===")
    print("Sampling reconstructions from posterior q_theta(x|y)...")
    
    cond_model.eval()
    
    all_reconstructions = []
    
    # Set up conditional sampler
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    cond_channels = args.input_channels + 1  # zero-filled img + mask
    
    # Create batched dataset
    N = len(corrupted_data)
    batch_size = args.batch_size
    
    with torch.no_grad():
        with cond_model.ema_scope():
            process = ConditionalDiffusionProcess(
                args, cond_model.net, target_shape, cond_channels
            )
            
            for start_idx in tqdm(range(0, N, batch_size), desc="E-step sampling"):
                end_idx = min(start_idx + batch_size, N)
                
                # Get batch
                corrupted_batch = corrupted_data[start_idx:end_idx]
                mask_batch = obs_masks[start_idx:end_idx]
                
                # Convert to tensors
                corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
                mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)
                
                # Zero-fill NaN values for conditioning
                cond_ts = torch.nan_to_num(corrupted_ts, nan=0.0)
                
                # Convert to images
                cond_img = cond_model.ts_to_img(cond_ts)
                
                # Create mask image - expand mask to match features, then convert
                mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
                cond_mask_img = cond_model.ts_to_img(mask_ts_expanded)
                cond_mask_img = cond_mask_img[:, :1, :, :]  # Take first channel for mask
                
                # Sample from posterior q_theta(x | zero_filled_y, mask)
                x_img_sampled = process.sampling(cond_img, cond_mask_img)
                
                # Convert back to time series
                x_ts_recon = cond_model.img_to_ts(x_img_sampled)
                
                all_reconstructions.append(x_ts_recon.cpu().numpy())
    
    reconstructions = np.vstack(all_reconstructions)
    print(f"E-step complete. Reconstructed {len(reconstructions)} sequences.")
    
    # Cache reconstructions to disk
    cache_dir = os.path.join(args.recon_cache_dir, args.dataset, 
                              f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)
    
    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)
    
    return reconstructions


def m_step(args, cond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """
    M-step: Train conditional diffusion model on (X, Y~Q(X)) pairs.
    
    Pure DiffEM: 
    - X is the reconstruction from E-step
    - Y ~ Q(X) is a NEW corruption of X (zero-filled + mask)
    - Train model to denoise X given Y
    
    Args:
        args: configuration arguments
        cond_model: conditional diffusion model
        optimizer: model optimizer
        reconstructions: numpy array of reconstructed sequences from E-step
        em_iter: current EM iteration number
        device: target device
        logger: optional logger
    
    Returns:
        cond_model: updated conditional model
    """
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"Training conditional model for {args.m_step_epochs} epochs...")
    
    # Determine corruption parameters for M-step
    corruption_rate = args.em_corruption_rate if args.em_corruption_rate else args.missing_rate
    noise_level = args.em_noise_level if args.em_noise_level else args.gaussian_noise_level
    
    # Create dataset from reconstructions
    em_dataset = EMReconstructionDataset(
        reconstructions,
        missing_rate=corruption_rate,
        gaussian_noise_level=noise_level,
        noise_timestep=args.noise_timestep,
        precompute_corruption=False,
        seed=args.seed + em_iter * 1000  # Different seed per EM iter
    )
    
    em_loader = Data.DataLoader(
        em_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    cond_model.train()
    
    for epoch in range(args.m_step_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (clean, corrupted, mask) in enumerate(em_loader):
            clean = clean.to(device)
            corrupted = corrupted.to(device)  # Has NaN for missing
            mask = mask.to(device)  # True where observed
            
            # Zero-fill corrupted data for conditioning
            cond_ts = torch.nan_to_num(corrupted, nan=0.0)
            
            # Convert to images
            x_img = cond_model.ts_to_img(clean)
            cond_img = cond_model.ts_to_img(cond_ts)
            
            # Create mask image
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, clean.shape[-1])
            obs_mask_img = cond_model.ts_to_img(mask_expanded)
            obs_mask_img = obs_mask_img[:, :1, :, :]  # Single channel mask
            
            # Compute conditional loss on ALL pixels (full reconstruction)
            # The paper trains on full loss, not masked loss
            loss, to_log = cond_model.loss_fn_conditional_full(
                x_img, cond_img, obs_mask_img
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cond_model.parameters(), 1.0)
            optimizer.step()
            cond_model.on_train_batch_end()
            
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
    return cond_model


def evaluate_em_iteration(args, cond_model, corrupted_data, obs_masks, test_loader,
                          em_iter, device, logger=None):
    """
    Evaluate reconstruction quality at current EM iteration.
    
    Generates samples conditioned on test observations and computes metrics.
    Pure DiffEM: uses zero-filled + mask conditioning.
    """
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    
    cond_model.eval()
    
    gen_sig = []
    real_sig = []
    
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    cond_channels = args.input_channels + 1
    
    with torch.no_grad():
        with cond_model.ema_scope():
            process = ConditionalDiffusionProcess(
                args, cond_model.net, target_shape, cond_channels
            )
            
            for data in tqdm(test_loader, desc="Evaluating"):
                real_data = data[0].to(device)
                
                # Get time series (remove time index if present)
                x_ts = real_data[:, :, :-1] if real_data.shape[-1] > args.input_channels else real_data
                
                # Apply corruption for evaluation
                corrupted_np, mask_np = apply_corruption(
                    x_ts.cpu().numpy(),
                    missing_rate=args.missing_rate,
                    gaussian_noise_level=args.gaussian_noise_level,
                    seed=42  # Fixed seed for reproducible evaluation
                )
                corrupted = torch.tensor(corrupted_np, dtype=torch.float32, device=device)
                mask_ts = torch.tensor(mask_np, dtype=torch.float32, device=device)
                
                # Zero-fill for conditioning
                cond_ts = torch.nan_to_num(corrupted, nan=0.0)
                
                # Convert to images
                cond_img = cond_model.ts_to_img(cond_ts)
                
                # Create mask image
                mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, x_ts.shape[-1])
                cond_mask_img = cond_model.ts_to_img(mask_ts_expanded)
                cond_mask_img = cond_mask_img[:, :1, :, :]
                
                # Sample from posterior
                x_img_sampled = process.sampling(cond_img, cond_mask_img)
                x_ts_recon = cond_model.img_to_ts(x_img_sampled)
                
                gen_sig.append(x_ts_recon.cpu().numpy())
                real_sig.append(x_ts.cpu().numpy())
    
    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)
    
    # Compute metrics
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
    
    # Log memorization stats
    print(f"EM iter {em_iter} memorization metrics:")
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)
    
    # Upload plot to Neptune if available
    if logger is not None:
        upload_successful = False
        if hasattr(logger, 'loggers'):
            for sub_logger in logger.loggers:
                if isinstance(sub_logger, NeptuneLogger):
                    try:
                        sub_logger.log('test/memorization/histogram', File(mem_plot_path), em_iter)
                        sub_logger.run.sync()
                        upload_successful = True
                    except Exception as e:
                        print(f"Failed to upload memorization plot to Neptune: {e}")
        elif isinstance(logger, NeptuneLogger):
            try:
                logger.log('test/memorization/histogram', File(mem_plot_path), em_iter)
                logger.run.sync()
                upload_successful = True
            except Exception as e:
                print(f"Failed to upload memorization plot to Neptune: {e}")
        
        if upload_successful:
            try:
                if os.path.exists(mem_plot_path):
                    os.remove(mem_plot_path)
            except Exception as e:
                print(f"Failed to delete temporary plot file {mem_plot_path}: {e}")
    
    return scores


def train_and_evaluate_unconditional(args, uncond_model, uncond_optimizer, reconstructions, 
                                      test_loader, em_iter, device, logger=None):
    """
    Train unconditional model on current reconstructions and evaluate.
    
    This function trains the unconditional model for a number of epochs on the
    current reconstructions, then evaluates using unconditional sampling.
    Same evaluation as run_irregular.py for fair comparison.
    
    Args:
        args: configuration arguments
        uncond_model: unconditional diffusion model (persistent across EM iterations)
        uncond_optimizer: optimizer for unconditional model
        reconstructions: current reconstructions from E-step
        test_loader: test data loader
        em_iter: current EM iteration
        device: target device
        logger: optional logger
    
    Returns:
        scores: dict of evaluation metrics
    """
    print(f"\n=== Unconditional Training & Evaluation (EM iter {em_iter}) ===")
    
    # Create dataset from current reconstructions
    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Train unconditional model for some epochs
    uncond_epochs_per_iter = getattr(args, 'uncond_epochs_per_iter', None)
    if uncond_epochs_per_iter is None:
        uncond_epochs_per_iter = args.m_step_epochs
    print(f"Training unconditional model for {uncond_epochs_per_iter} epochs...")
    
    uncond_model.train()
    
    for epoch in range(uncond_epochs_per_iter):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (x_clean,) in enumerate(recon_loader):
            x_clean = x_clean.to(device)
            
            # Convert to image
            x_img = uncond_model.ts_to_img(x_clean)
            
            # Standard unconditional diffusion loss
            loss, to_log = uncond_model.loss_fn_irregular(x_img)
            
            uncond_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            uncond_optimizer.step()
            uncond_model.on_train_batch_end()
            
            epoch_loss += loss.item()
            num_batches += 1
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Uncond epoch {epoch+1}/{uncond_epochs_per_iter}, loss: {avg_loss:.4f}")
        
        if logger is not None:
            global_step = em_iter * uncond_epochs_per_iter + epoch
            logger.log('train/uncond_loss', avg_loss, global_step)
    
    # Evaluate unconditional model - SAME AS run_irregular.py
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
                # UNCONDITIONAL sampling - generates completely new sequences
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = uncond_model.img_to_ts(x_img_sampled)
                
                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())
    
    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)
    
    # Compute metrics
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
    
    # Log memorization stats
    print(f"EM iter {em_iter} memorization metrics:")
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)
    
    # Upload plot to Neptune if available
    if logger is not None:
        upload_successful = False
        if hasattr(logger, 'loggers'):
            for sub_logger in logger.loggers:
                if isinstance(sub_logger, NeptuneLogger):
                    try:
                        sub_logger.log('test/memorization/histogram', File(mem_plot_path), em_iter)
                        sub_logger.run.sync()
                        upload_successful = True
                    except Exception as e:
                        print(f"Failed to upload memorization plot to Neptune: {e}")
        elif isinstance(logger, NeptuneLogger):
            try:
                logger.log('test/memorization/histogram', File(mem_plot_path), em_iter)
                logger.run.sync()
                upload_successful = True
            except Exception as e:
                print(f"Failed to upload memorization plot to Neptune: {e}")
        
        if upload_successful:
            try:
                if os.path.exists(mem_plot_path):
                    os.remove(mem_plot_path)
            except Exception as e:
                print(f"Failed to delete temporary plot file {mem_plot_path}: {e}")
    
    return scores


def train_unconditional_final(args, reconstructions, test_loader, device, logger=None):
    """
    Train unconditional diffusion model on final reconstructions.
    
    This mirrors DiffEM's final step of training an unconditional prior
    on the last reconstruction dataset.
    
    DEPRECATED: Use train_and_evaluate_unconditional instead for evaluation during EM loop.
    """
    print("\n=== Training Unconditional Model on Final Reconstructions ===")
    
    # Create unconditional model
    uncond_model = TS2img_Karras(args=args, device=device).to(device)
    uncond_optimizer = torch.optim.AdamW(
        uncond_model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Create dataset from reconstructions
    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    uncond_model.train()
    
    for epoch in range(args.uncond_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (x_clean,) in enumerate(recon_loader):
            x_clean = x_clean.to(device)
            
            # Convert to image
            x_img = uncond_model.ts_to_img(x_clean)
            
            # Standard unconditional diffusion loss
            loss, to_log = uncond_model.loss_fn_irregular(x_img)
            
            uncond_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            uncond_optimizer.step()
            uncond_model.on_train_batch_end()
            
            epoch_loss += loss.item()
            num_batches += 1
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Uncond epoch {epoch+1}/{args.uncond_epochs}, loss: {avg_loss:.4f}")
        
        if logger is not None:
            logger.log('uncond/loss', avg_loss, epoch)
    
    # Evaluate unconditional model
    print("\nEvaluating unconditional model...")
    uncond_model.eval()
    gen_sig = []
    real_sig = []
    
    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution, args.img_resolution)
            )
            
            for data in tqdm(test_loader, desc="Evaluating uncond"):
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = uncond_model.img_to_ts(x_img_sampled)
                
                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())
    
    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)
    
    scores = evaluate_model_irregular(real_sig, gen_sig, args)
    
    print("Unconditional model metrics:")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'uncond/eval/{key}', value, 0)
    
    return uncond_model, scores


def main(args):
    """Main Pure DiffEM training loop."""
    
    # Model name and directory
    name = create_model_name_and_dir(args)
    name = f"diffem_pure_{name}"
    
    logging.info(args)
    
    # Set up logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)
        
        # Set up device and data
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')
        
        # Initialize conditional diffusion model
        cond_model = TS2img_Karras_Cond(args=args, device=args.device).to(args.device)
        cond_optimizer = torch.optim.AdamW(
            cond_model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        print_model_params(logger, cond_model)
        
        # === Extract corrupted data and masks ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences with {obs_masks.sum() / obs_masks.size * 100:.1f}% observed")
        
        # === Phase 1: Initialize with Gaussian prior ===
        print(f"\n{'='*60}")
        print("Phase 1: Gaussian Initialization")
        print(f"{'='*60}")
        
        initial_reconstructions = initialize_with_gaussian(
            corrupted_data, obs_masks, seed=args.seed
        )
        
        # === Phase 1.5: Initial training of conditional model ===
        # Train conditional model on (Gaussian reconstructions, corrupted observations)
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial conditional model training")
        print(f"{'='*60}")
        
        cond_model = m_step(
            args, cond_model, cond_optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )
        print("Initial conditional model training complete.")
        
        # === Create Unconditional Model for Evaluation ===
        # This model is trained progressively and used for FAIR evaluation (same as run_irregular.py)
        print(f"\n{'='*60}")
        print("Creating unconditional model for evaluation")
        print(f"{'='*60}")
        
        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        uncond_optimizer = torch.optim.AdamW(
            uncond_model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        # Note: print_model_params doesn't support prefix, just print a note
        print("Unconditional model created (same architecture as conditional model)")
        
        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting Pure DiffEM with {args.em_iters} EM iterations")
        print(f"{'='*60}")
        
        best_metrics = None
        final_reconstructions = None
        metrics = None
        
        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")
            
            # E-step: Sample reconstructions from posterior
            # q_theta(x | zero_filled_Y, mask)
            reconstructions = e_step(
                args, cond_model, corrupted_data, obs_masks,
                em_iter, args.device, logger
            )
            
            # M-step: Train conditional model on (X, Y~Q(X)) pairs
            cond_model = m_step(
                args, cond_model, cond_optimizer,
                reconstructions, em_iter, args.device, logger
            )
            
            # Evaluate using UNCONDITIONAL model (same as run_irregular.py)
            # This is the FAIR evaluation - generates completely new sequences
            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = train_and_evaluate_unconditional(
                    args, uncond_model, uncond_optimizer, reconstructions,
                    test_loader, em_iter, args.device, logger
                )
                
                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics
            
            # === COMMENTED OUT: Conditional evaluation (imputation-based, NOT comparable to baseline) ===
            # if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
            #     metrics = evaluate_em_iteration(
            #         args, cond_model, corrupted_data, obs_masks, test_loader,
            #         em_iter, args.device, logger
            #     )
            #     
            #     if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
            #         best_metrics = metrics
            
            # Keep track of final reconstructions
            final_reconstructions = reconstructions
            
            # Log EM iteration
            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)
        
        # === Phase 3 is now integrated into the EM loop ===
        # Unconditional model is trained progressively during EM iterations
        # No need for separate final training
        
        print("\n" + "="*60)
        print("Pure DiffEM Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")
        
        logging.info("Pure DiffEM training is complete")


if __name__ == '__main__':
    args = parse_args_irregular()
    
    # Force DiffEM mode
    args.use_diffem = True
    
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
