"""
DiffEM Training Script for Irregular Time Series

This script implements the DiffEM-style EM training loop:
1. E-step: Sample X ~ q_theta(x|y) using conditional diffusion
2. M-step: Train conditional diffusion model on (X, Y~Q(X)) pairs
3. Final: Train unconditional model on final reconstructions

Based on the DiffEM paper: "Learning from Corrupted Data with Diffusion Models 
via Expectation Maximization"
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
from itertools import chain

from metrics import evaluate_model_irregular
from metrics.memorization import compute_memorization_metric
from neptune.types import File
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import (
    gen_dataloader, apply_corruption, build_conditioning, 
    EMReconstructionDataset, save_reconstructions, load_reconstructions
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras, TS2img_Karras_Cond
from models.sampler import DiffusionProcess, ConditionalDiffusionProcess
from models.decoder import TST_Decoder
from models.TST import TSTransformerEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def propagate_values_forward(tensor):
    """Handle sequences that are entirely NaN by copying from adjacent sequences."""
    for b in range(tensor.size(0)):
        sequence = tensor[b]
        if torch.isnan(sequence).all():
            if b + 1 < tensor.size(0):
                tensor[b] = tensor[b + 1]
            else:
                tensor[b] = tensor[b - 1]
    return tensor


def propagate_values(tensor):
    return propagate_values_forward(tensor)


def _loss_e_t0(x_tilde, x):
    return F.mse_loss(x_tilde, x)


def _loss_e_0(loss_e_t0):
    return torch.sqrt(loss_e_t0) * 10


def get_initial_reconstructions(args, embedder, decoder, train_loader, device):
    """
    Get initial reconstructions using TST completions.
    
    This serves as the initial "prior" for the DiffEM loop, similar to how
    DiffEM uses a Gaussian prior or warm-start model for initialization.
    
    Args:
        args: configuration arguments
        embedder: TST encoder
        decoder: TST decoder
        train_loader: dataloader with irregular observations
        device: target device
    
    Returns:
        reconstructions: numpy array of TST-completed sequences (N, seq_len, features)
    """
    print("Generating initial reconstructions using TST completions...")
    
    embedder.eval()
    decoder.eval()
    
    all_reconstructions = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(train_loader, desc="Initial reconstructions")):
            x_irregular = data[0].to(device)
            x_ts = x_irregular[:, :, :-1]  # Remove time index
            x_ts = propagate_values(x_ts)
            
            # Get TST completion
            padding_masks = ~torch.isnan(x_ts).any(dim=-1)
            h = embedder(x_ts, padding_masks)
            completed_ts = decoder(h)
            
            all_reconstructions.append(completed_ts.cpu().numpy())
    
    reconstructions = np.vstack(all_reconstructions)
    print(f"Initial reconstructions complete. Generated {len(reconstructions)} sequences.")
    
    return reconstructions


def pretrain_tst(args, embedder, decoder, train_loader, optimizer, epochs, device, logger=None):
    """
    Pretrain TST encoder/decoder for completion.
    
    This is the initial phase where TST learns to complete irregular sequences.
    """
    print(f"Pretraining TST for {epochs} epochs...")
    embedder.train()
    decoder.train()
    
    for step in range(1, epochs + 1):
        epoch_loss = 0
        num_batches = 0
        
        for i, data in enumerate(train_loader, 1):
            x = data[0].to(device)
            x = x[:, :, :-1]  # Remove time index column
            x = propagate_values(x)
            padding_masks = ~torch.isnan(x).any(dim=-1)
            
            h = embedder(x, padding_masks)
            x_tilde = decoder(h)

            x_no_nan = x[~torch.isnan(x)]
            x_tilde_no_nan = x_tilde[~torch.isnan(x)]
            loss_e_t0 = _loss_e_t0(x_tilde_no_nan, x_no_nan)
            loss_e_0 = _loss_e_0(loss_e_t0)
            
            optimizer.zero_grad()
            loss_e_0.backward()
            optimizer.step()
            
            epoch_loss += loss_e_t0.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_loss = np.sqrt(epoch_loss / num_batches)
        print(f"TST Pretrain step: {step}/{epochs}, loss: {avg_loss:.4f}")
        
        if logger is not None:
            logger.log('pretrain/tst_loss', avg_loss, step)
    
    print("TST pretraining complete.")
    return embedder, decoder


def e_step(args, cond_model, embedder, decoder, train_loader, em_iter, device, logger=None):
    """
    E-step: Sample reconstructions X ~ q_theta(x|y) for all observations.
    
    Args:
        args: configuration arguments
        cond_model: conditional diffusion model
        embedder: TST encoder
        decoder: TST decoder
        train_loader: dataloader with irregular observations
        em_iter: current EM iteration number
        device: target device
        logger: optional logger
    
    Returns:
        reconstructions: numpy array of reconstructed sequences (N, seq_len, features)
    """
    print(f"\n=== E-Step (EM iter {em_iter}) ===")
    print("Sampling reconstructions from posterior q_theta(x|y)...")
    
    cond_model.eval()
    embedder.eval()
    decoder.eval()
    
    all_reconstructions = []
    
    # Set up conditional sampler
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)
    cond_channels = args.input_channels + 1  # cond_img + mask
    
    with torch.no_grad():
        with cond_model.ema_scope():
            process = ConditionalDiffusionProcess(
                args, cond_model.net, target_shape, cond_channels
            )
            
            for batch_idx, data in enumerate(tqdm(train_loader, desc="E-step sampling")):
                x_irregular = data[0].to(device)
                x_ts = x_irregular[:, :, :-1]  # Remove time index
                x_ts = propagate_values(x_ts)
                
                # Get TST completion for conditioning
                padding_masks = ~torch.isnan(x_ts).any(dim=-1)
                h = embedder(x_ts, padding_masks)
                completed_ts = decoder(h)
                
                # Convert to images
                cond_img = cond_model.ts_to_img(completed_ts)
                
                # Create mask image
                mask_ts = torch.isnan(x_ts).any(dim=-1, keepdim=True).float()
                mask_ts = 1.0 - mask_ts  # 1 = observed, 0 = missing
                # Expand mask to match features then convert to image
                mask_ts_expanded = mask_ts.expand(-1, -1, x_ts.shape[-1])
                cond_mask_img = cond_model.ts_to_img(mask_ts_expanded)
                # Take only first channel for mask
                cond_mask_img = cond_mask_img[:, :1, :, :]
                
                # Sample from posterior
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


def m_step(args, cond_model, optimizer, embedder, decoder, reconstructions, 
           em_iter, device, logger=None):
    """
    M-step: Train conditional diffusion model on (X, Y~Q(X)) pairs.
    
    Args:
        args: configuration arguments
        cond_model: conditional diffusion model
        optimizer: model optimizer
        embedder: TST encoder (frozen or trainable)
        decoder: TST decoder (frozen or trainable)
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
    embedder.eval()  # TST should be frozen during M-step
    decoder.eval()
    
    for epoch in range(args.m_step_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (clean, corrupted, mask) in enumerate(em_loader):
            clean = clean.to(device)
            corrupted = corrupted.to(device)
            mask = mask.to(device)
            
            # Get TST completion for conditioning
            with torch.no_grad():
                padding_masks = ~torch.isnan(corrupted).any(dim=-1)
                # Replace NaN with 0 for TST input
                corrupted_filled = torch.nan_to_num(corrupted, nan=0.0)
                h = embedder(corrupted_filled, padding_masks)
                completed_ts = decoder(h)
            
            # Convert to images
            x_img = cond_model.ts_to_img(clean)
            cond_img = cond_model.ts_to_img(completed_ts)
            
            # Create mask image
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, clean.shape[-1])
            obs_mask_img = cond_model.ts_to_img(mask_expanded)
            obs_mask_img = obs_mask_img[:, :1, :, :]  # Single channel mask
            
            # Compute conditional loss
            loss, to_log = cond_model.loss_fn_conditional(
                x_img, cond_img, obs_mask_img, obs_mask=obs_mask_img
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


def evaluate_em_iteration(args, cond_model, embedder, decoder, test_loader, 
                          em_iter, device, logger=None):
    """
    Evaluate reconstruction quality at current EM iteration.
    
    Generates samples conditioned on test observations and computes metrics.
    """
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")
    
    cond_model.eval()
    embedder.eval()
    decoder.eval()
    
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
                
                # For evaluation, we create corrupted versions of test data
                # and try to reconstruct them
                x_ts = real_data[:, :, :-1] if real_data.shape[-1] > args.input_channels else real_data
                
                # Apply corruption for evaluation
                corrupted_np, mask_np = apply_corruption(
                    x_ts.cpu().numpy(),
                    missing_rate=args.missing_rate,
                    gaussian_noise_level=args.gaussian_noise_level,
                    seed=42  # Fixed seed for reproducible evaluation
                )
                corrupted = torch.tensor(corrupted_np, dtype=torch.float32, device=device)
                
                # Get TST completion
                padding_masks = ~torch.isnan(corrupted).any(dim=-1)
                corrupted_filled = torch.nan_to_num(corrupted, nan=0.0)
                h = embedder(corrupted_filled, padding_masks)
                completed_ts = decoder(h)
                
                # Convert to images
                cond_img = cond_model.ts_to_img(completed_ts)
                
                # Create mask image
                mask_ts = torch.tensor(mask_np, dtype=torch.float32, device=device)
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
            logger.log(f'em/memorization/{k}', v, em_iter)
    
    # Upload plot to Neptune if available
    if logger is not None:
        upload_successful = False
        if hasattr(logger, 'loggers'):
            # CompositeLogger case
            for sub_logger in logger.loggers:
                if isinstance(sub_logger, NeptuneLogger):
                    try:
                        sub_logger.log('em/memorization/histogram', File(mem_plot_path), em_iter)
                        sub_logger.run.sync()
                        upload_successful = True
                    except Exception as e:
                        print(f"Failed to upload memorization plot to Neptune: {e}")
        elif isinstance(logger, NeptuneLogger):
            try:
                logger.log('em/memorization/histogram', File(mem_plot_path), em_iter)
                logger.run.sync()
                upload_successful = True
            except Exception as e:
                print(f"Failed to upload memorization plot to Neptune: {e}")
        
        # Clean up plot file after upload
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
    """Main DiffEM training loop."""
    
    # Model name and directory
    name = create_model_name_and_dir(args)
    name = f"diffem_{name}"
    
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
        
        # Initialize TST encoder/decoder
        tst_config = {
            'feat_dim': args.input_size,
            'max_len': args.seq_len,
            'd_model': args.hidden_dim,
            'n_heads': args.n_heads,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'pos_encoding': args.pos_encoding,
            'activation': args.activation,
            'norm': args.norm,
            'freeze': args.freeze
        }
        
        embedder = TSTransformerEncoder(
            feat_dim=tst_config['feat_dim'],
            max_len=tst_config['max_len'],
            d_model=tst_config['d_model'],
            n_heads=tst_config['n_heads'],
            num_layers=tst_config['num_layers'],
            dim_feedforward=tst_config['dim_feedforward'],
            dropout=tst_config['dropout'],
            pos_encoding=tst_config['pos_encoding'],
            activation=tst_config['activation'],
            norm=tst_config['norm'],
            freeze=tst_config['freeze']
        ).to(args.device)
        
        decoder = TST_Decoder(
            inp_dim=args.hidden_dim,
            hidden_dim=int(args.hidden_dim + (args.input_size - args.hidden_dim) / 2),
            layers=3,
            args=args
        ).to(args.device)
        
        tst_optimizer = optim.Adam(chain(embedder.parameters(), decoder.parameters()))
        
        # === Phase 1: Pretrain TST ===
        embedder, decoder = pretrain_tst(
            args, embedder, decoder, train_loader, tst_optimizer,
            epochs=args.first_epoch, device=args.device, logger=logger
        )
        
        # Freeze TST after pretraining if specified
        # Check for the negative flag first, then the positive flag
        should_freeze = getattr(args, 'freeze_tst_after_pretrain', True)
        if getattr(args, 'no_freeze_tst', False):
            should_freeze = False
        
        if should_freeze:
            print("Freezing TST encoder/decoder for EM iterations...")
            for param in embedder.parameters():
                param.requires_grad = False
            for param in decoder.parameters():
                param.requires_grad = False
        
        # === Phase 1.5: Initial training of conditional model ===
        # Before EM loop, we need to train the conditional model initially
        # We use TST completions as the initial "reconstructions" (similar to Gaussian prior in DiffEM)
        print(f"\n{'='*60}")
        print("Phase 1.5: Initial conditional model training")
        print(f"{'='*60}")
        
        initial_reconstructions = get_initial_reconstructions(
            args, embedder, decoder, train_loader, args.device
        )
        
        # Train conditional model on initial reconstructions
        cond_model = m_step(
            args, cond_model, cond_optimizer, embedder, decoder,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )
        print("Initial conditional model training complete.")
        
        # === Phase 2: EM Loop ===
        print(f"\n{'='*60}")
        print(f"Starting DiffEM with {args.em_iters} EM iterations")
        print(f"{'='*60}")
        
        best_metrics = None
        final_reconstructions = None
        metrics = None  # Initialize to avoid undefined variable
        
        for em_iter in range(args.em_iters):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")
            
            # E-step: Sample reconstructions from posterior
            reconstructions = e_step(
                args, cond_model, embedder, decoder, train_loader,
                em_iter, args.device, logger
            )
            
            # M-step: Train conditional model on (X, Y~Q(X)) pairs
            cond_model = m_step(
                args, cond_model, cond_optimizer, embedder, decoder,
                reconstructions, em_iter, args.device, logger
            )
            
            # Evaluate if at interval
            if (em_iter + 1) % args.em_eval_interval == 0 or em_iter == args.em_iters - 1:
                metrics = evaluate_em_iteration(
                    args, cond_model, embedder, decoder, test_loader,
                    em_iter, args.device, logger
                )
                
                if best_metrics is None or metrics.get('discriminative', float('inf')) < best_metrics.get('discriminative', float('inf')):
                    best_metrics = metrics
            
            # Keep track of final reconstructions
            final_reconstructions = reconstructions
            
            # Log EM iteration
            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)
        
        # === Phase 3: Train Unconditional Model ===
        # Check for the negative flag first, then the positive flag
        should_train_uncond = getattr(args, 'train_uncond_after_em', True)
        if getattr(args, 'no_train_uncond', False):
            should_train_uncond = False
        
        if should_train_uncond and final_reconstructions is not None:
            uncond_model, uncond_metrics = train_unconditional_final(
                args, final_reconstructions, test_loader, args.device, logger
            )
        
        print("\n" + "="*60)
        print("DiffEM Training Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics across EM iterations:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")
        
        logging.info("DiffEM training is complete")


if __name__ == '__main__':
    args = parse_args_irregular()
    
    # Force DiffEM mode
    args.use_diffem = True
    
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
