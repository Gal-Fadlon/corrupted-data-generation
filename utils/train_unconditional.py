"""
Shared Phase 3: train an unconditional diffusion model on cleaned data.

Uses the same architecture, hyperparameters, and training loop as
run_regular.py so that downstream metrics are directly comparable to the
fully-observed baseline.
"""

import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm

from metrics import evaluate_model_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess


def _log(logger, name, value, step):
    if logger is not None:
        logger.log(name, value, step)


def train_unconditional_regular(args, reconstructions, test_loader, device,
                                logger=None):
    """Train an unconditional diffusion model on EM-cleaned reconstructions.

    Mirrors run_regular.py exactly: fresh TS2img_Karras, AdamW with config
    lr/wd, full args.epochs, evaluation every args.logging_iter epochs, EMA
    via on_train_batch_end(), grad-clip 1.0.

    Args:
        args:            parsed config namespace (must have .epochs,
                         .learning_rate, .weight_decay, .batch_size, etc.)
        reconstructions: numpy array (N, seq_len, features) — cleaned data
        test_loader:     DataLoader with original clean data for evaluation
        device:          'cuda' or 'cpu'
        logger:          optional wandb / composite logger

    Returns:
        best_metrics: dict of best evaluation metrics (keyed on disc_mean)
    """
    uncond_epochs = args.epochs
    logging_iter = getattr(args, 'logging_iter', 10)

    print(f"\n{'='*60}")
    print(f"Phase 3: Unconditional model (run_regular.py setup)")
    print(f"  epochs={uncond_epochs}  logging_iter={logging_iter}")
    print(f"  lr={args.learning_rate}  wd={args.weight_decay}")
    print(f"{'='*60}")

    model = TS2img_Karras(args=args, device=device).to(device)

    if getattr(args, 'embedder', 'delay') == 'stft':
        with torch.no_grad():
            stats_tensor = torch.as_tensor(
                reconstructions, dtype=torch.float32
            )
            model.cache_embedder_stats(stats_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    recon_loader = Data.DataLoader(
        Data.TensorDataset(
            torch.tensor(reconstructions, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    best_metrics = None

    for epoch in range(uncond_epochs):
        print(f"Starting epoch {epoch}.")

        model.train()

        for (x_clean,) in recon_loader:
            x_clean = x_clean.to(device)
            x_img = model.ts_to_img(x_clean)
            loss, to_log = model.loss_fn_irregular(x_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            model.on_train_batch_end()

            for key, value in to_log.items():
                _log(logger, f'train/{key}', value, epoch)

        if epoch % logging_iter == 0:
            gen_sig, real_sig = [], []
            model.eval()
            with torch.no_grad():
                with model.ema_scope():
                    _is_stft = (getattr(args, 'embedder', 'delay') == 'stft')
                    _pad_mask = model.pad_mask if _is_stft else None
                    process = DiffusionProcess(
                        args, model.net,
                        (model.num_features, args.img_resolution,
                         args.img_resolution),
                        pad_mask=_pad_mask,
                    )
                    for data in tqdm(test_loader,
                                     desc=f"Eval epoch {epoch}"):
                        x_img_sampled = process.sampling(
                            sampling_number=data[0].shape[0])
                        x_ts = model.img_to_ts(x_img_sampled)

                        gen_sig.append(x_ts.detach().cpu().numpy())
                        real_sig.append(data[0].detach().cpu().numpy())

            gen_sig = np.vstack(gen_sig)
            real_sig = np.vstack(real_sig)

            scores = evaluate_model_irregular(real_sig, gen_sig, args)
            print(f"  Epoch {epoch} metrics:")
            for k, v in scores.items():
                print(f"    {k}: {v:.4f}")
                _log(logger, f'test/{k}', v, epoch)

            disc = scores.get('disc_mean', float('inf'))
            if best_metrics is None or disc < best_metrics.get(
                    'disc_mean', float('inf')):
                best_metrics = scores
                print(f"  *** New best disc_mean={disc:.4f} "
                      f"at epoch {epoch} ***")

    return best_metrics or {}
