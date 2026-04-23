"""
ImagenI2R baseline run on block-mean-aggregated observations.

This is a clone of ``run_irregular.py`` that adds a single CLI flag
``--agg_window w`` and wraps the *training* loader so every batch's
features are replaced with the **representative-point NaN encoding** of the
block-mean aggregation:

    y_k = mean(x[w*k : w*(k+1)])    for k = 0 .. T/w - 1
    x_agg[:, w*k,  :] = y_k
    x_agg[:, w*k+j,:] = NaN         for 1 <= j < w

The ImagenI2R method code (TSTransformerEncoder, TST_Decoder, loss_fn_irregular,
propagate_values) is untouched — the baseline simply receives aggregated
data in its native NaN-marked irregular format.  The *test* loader is **not**
wrapped: ``real_sig`` used for the final disc/pred/FID metrics must remain the
clean ground-truth time series so the comparison is well-defined.

See the plan note ("Fairness") for paper framing.
"""

import torch
import torch.multiprocessing
from torch import optim
import torch.nn.functional as F
import os, sys
import glob
import numpy as np
import logging
from tqdm import tqdm
from itertools import chain

from metrics import evaluate_model_irregular
from metrics.memorization import compute_memorization_metric
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from models.decoder import TST_Decoder
from models.TST import TSTransformerEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Aggregated loader wrapper: representative-point NaN encoding
# =============================================================================


class AggregatedLoaderWrapper:
    """Wraps a DataLoader so every batch's feature columns are replaced with
    the representative-point NaN encoding of the block-mean aggregation.

    For each window k in 0 .. T/w - 1:
      - position w*k        carries y_k = mean(x[w*k : w*k+w]).
      - positions w*k+1..w-1 are set to NaN.

    The last column of ``data[0]`` (time-index column appended by
    ``gen_dataloader``) is preserved unchanged.  All non-feature tensors in
    the batch (``data[1:]``) pass through untouched.
    """

    def __init__(self, loader, window: int):
        self.loader = loader
        self.window = int(window)
        if self.window < 1:
            raise ValueError(f"agg_window must be >= 1, got {window}")

    def __iter__(self):
        for data in self.loader:
            yield self._transform(data)

    def __len__(self):
        return len(self.loader)

    # Pass-through a few common DataLoader attributes so callers can still
    # introspect the underlying dataset (e.g. for num_samples).
    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def batch_size(self):
        return self.loader.batch_size

    def _transform(self, data):
        batch = data[0]  # [B, T, F+1]
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch)

        B, T, C = batch.shape
        w = self.window
        if C < 2:
            raise ValueError(
                f"AggregatedLoaderWrapper: expected feat_dim + 1 time-idx "
                f"column (C >= 2), got C={C}."
            )
        if T % w != 0:
            T_trunc = (T // w) * w
            batch = batch[:, :T_trunc, :]
            T = T_trunc

        feats = batch[:, :, :-1]         # [B, T, F]
        time_idx = batch[:, :, -1:]       # [B, T, 1]
        F_dim = feats.shape[-1]

        # Block mean over non-overlapping windows.  Use nanmean defensively
        # so a caller that already has NaNs won't explode; clean input paths
        # will fall back to a regular mean.
        chunks = feats.reshape(B, T // w, w, F_dim)
        if torch.isnan(chunks).any():
            y = torch.nanmean(chunks, dim=2)
        else:
            y = chunks.mean(dim=2)       # [B, T/w, F]

        # Representative-point encoding.
        new_feats = torch.full_like(feats, float('nan'))
        # For each window k, drop y_k at position w*k.
        for k in range(T // w):
            new_feats[:, w * k, :] = y[:, k, :]

        new_batch = torch.cat([new_feats, time_idx], dim=-1)

        if isinstance(data, list):
            return [new_batch] + list(data[1:])
        if isinstance(data, tuple):
            return (new_batch,) + tuple(data[1:])
        # Fallback: assume caller just wants data[0] replaced.
        try:
            data[0] = new_batch  # type: ignore[index]
            return data
        except TypeError:
            return (new_batch,) + tuple(data[1:])


def _parse_aggregate_args():
    """Extract --agg_window from sys.argv before forwarding the remainder to
    utils.utils_args.parse_args_irregular (which does not know this flag).
    """
    import argparse as _argparse
    p = _argparse.ArgumentParser(add_help=False)
    p.add_argument(
        '--agg_window', type=int, default=1,
        help='Window size w for block-mean aggregation (>=2 activates '
             'representative-point NaN encoding on the training loader).',
    )
    extra, remainder = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remainder
    return extra

def propagate_values_forward(tensor):
    # Iterate over the batch and channels
    for b in range(tensor.size(0)):
            # Extract the sequence for the current batch and channel
            sequence = tensor[b]
            if torch.isnan(sequence).all():
                if b + 1 < tensor.size(0):
                    tensor[b] = tensor[b + 1]
                else:
                    tensor[b] = tensor[b - 1]
    return tensor

def propagate_values(tensor):
    tensor = propagate_values_forward(tensor)
    return tensor

def save_checkpoint(args, our_model, our_optimizer, ema_model, encoder, decoder, tst_optimizer, disc_score, pred_score=None, fid_score=None, correlation_score=None):
    """
    Saves the model checkpoint to the specified directory based on args and disc_score.
    """
    try:
        main_path = args.model_save_path
        seq_len = args.seq_len
        data_set_name = args.dataset
        missing_rate = int(args.missing_rate * 100)

        # Build the directory structure
        full_path = os.path.join(main_path, f'seq_len_{seq_len}', data_set_name, f'missing_rate_{missing_rate}')
        os.makedirs(full_path, exist_ok=True)

        # ---- Remove old files ----
        for f in glob.glob(os.path.join(full_path, "*")):
            try:
                os.remove(f)
            except IsADirectoryError:
                # if subdirectories might exist, handle recursively
                import shutil
                shutil.rmtree(f)

        # Generate the file name
        filename = f"disc_score_{disc_score}_pred_score_{pred_score}_fid_score_{fid_score}_correlation_score_{correlation_score}.pth"

        filepath = os.path.join(full_path, filename)

        # Save the checkpoint
        torch.save({
            'our_model_state_dict': our_model.state_dict(),
            'our_optimizer_state_dict': our_optimizer.state_dict(),
            'ema_model': ema_model.state_dict(),
            'tst_encoder': encoder.state_dict(),
            'tst_decoder': decoder.state_dict(),
            'tst_optimizer': tst_optimizer.state_dict(),
            'disc_score': disc_score,
            'pred_score': pred_score,
            'fid_score': fid_score,
            'correlation_score': correlation_score,
            'args': vars(args)
        }, filepath)

        print(f"Checkpoint saved at: {filepath}")

    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

def _loss_e_t0(x_tilde, x):
    return F.mse_loss(x_tilde, x)

def _loss_e_0(loss_e_t0):
    return torch.sqrt(loss_e_t0) * 10


def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    # set-up logger
    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # set-up data and device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        # --- Block-mean aggregation on the *train* loader only ---
        # test_loader is left untouched so that `real_sig` (the ground-truth
        # reference for disc/pred/FID metrics) remains the clean TS and the
        # evaluation is well-defined.
        agg_window = int(getattr(args, 'agg_window', 1))
        if agg_window >= 2:
            if args.missing_rate and args.missing_rate > 0:
                raise ValueError(
                    "run_irregular_aggregate.py expects --missing_rate 0 "
                    "when --agg_window >= 2 (block-mean aggregation is the "
                    "sole corruption under test)."
                )
            print(f"[aggregate] Wrapping train_loader with "
                  f"AggregatedLoaderWrapper(window={agg_window}); "
                  f"test_loader is kept clean for metrics.")
            train_loader = AggregatedLoaderWrapper(train_loader, window=agg_window)

        model = TS2img_Karras(args=args, device=args.device).to(args.device)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        state = dict(model=model, epoch=0)
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None
            init_epoch = restore_state(args, state, ema_model=ema_model)

        # print model parameters
        print_model_params(logger, model)

        tst_config = {
            'feat_dim': args.input_size,
            'max_len': args.seq_len,
            'd_model': args.hidden_dim,
            'n_heads': args.n_heads,  # Number of attention heads
            'num_layers': args.num_layers,  # Number of transformer layers
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'pos_encoding': args.pos_encoding,  # or 'learnable'
            'activation': args.activation,
            'norm': args.norm,
            'freeze': args.freeze
        }
        # Initialize the TST model
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
        optimizer_er = optim.Adam(chain(embedder.parameters(), decoder.parameters()))
        embedder.train()
        decoder.train()

        # --- train model ---
        logging.info(f"Continuing training loop from epoch {init_epoch}.")
        best_disc_score = float('inf')

        print('logging_iter', args.logging_iter)
        for step in range(1, args.first_epoch + 1):
            for i, data in enumerate(train_loader, 1):
                x = data[0].to(args.device)
                x = x[:, :, :-1]
                x = propagate_values(x)
                padding_masks = ~torch.isnan(x).any(dim=-1)
                h = embedder(x, padding_masks)

                # Decoder forward pass with time information
                x_tilde = decoder(h)

                x_no_nan = x[~torch.isnan(x)]
                x_tilde_no_nan = x_tilde[~torch.isnan(x)]
                loss_e_t0 = _loss_e_t0(x_tilde_no_nan, x_no_nan)
                loss_e_0 = _loss_e_0(loss_e_t0)
                optimizer_er.zero_grad()
                loss_e_0.backward()
                optimizer_er.step()
                torch.cuda.empty_cache()

            print(
                "step: "
                + str(step)
                + "/"
                + str(args.first_epoch)
                + ", loss_e: "
                + str(np.round(np.sqrt(loss_e_t0.item()), 4))
            )


        for epoch in range(init_epoch, args.epochs):
            print("Starting epoch %d." % (epoch,))

            model.train()
            model.epoch = epoch

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                x = data[0].to(args.device)
                x_ts = x[:, :, :-1]

                x_ts = propagate_values(x_ts)
                padding_masks = ~torch.isnan(x_ts).any(dim=-1)
                x_img = model.ts_to_img(x_ts)
                mask = torch.isnan(x_img).float() * -1 + 1
                h = embedder(x_ts, padding_masks)
                x_recon = decoder(h)
                x_tilde_img = model.ts_to_img(x_recon)
                loss = model.loss_fn_irregular(x_tilde_img, mask)
                optimizer.zero_grad()
                if len(loss) == 2:
                    loss, to_log = loss
                    for key, value in to_log.items():
                        logger.log(f'train/{key}', value, epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

                # #############Recovery######################
                h = embedder(x_ts, padding_masks)
                x_tilde = decoder(h)
                x_no_nan = x_ts[~torch.isnan(x_ts)]
                x_tilde_no_nan = x_tilde[~torch.isnan(x_ts)]
                loss_e_t0 = _loss_e_t0(x_tilde_no_nan, x_no_nan)

                loss_e_0 = _loss_e_0(loss_e_t0)
                loss_e = loss_e_0
                optimizer_er.zero_grad()
                loss_e.backward()
                optimizer_er.step()
                torch.cuda.empty_cache()

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        for data in tqdm(test_loader):
                            # sample from the model
                            x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                            # --- convert to time series --
                            x_ts = model.img_to_ts(x_img_sampled)

                            gen_sig.append(x_ts.detach().cpu().numpy())
                            real_sig.append(data[0].detach().cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)

                scores = evaluate_model_irregular(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

                # --- Memorization Check ---
                # We use a subset of real data to match the generated size if needed, or full real data
                # real_sig is already available here as a numpy array
                mem_plot_path = f"memorization_hist_epoch_{epoch}.png"
                mem_stats = compute_memorization_metric(
                    real_data=real_sig,
                    generated_data=gen_sig,
                    device=args.device,
                    plot_path=mem_plot_path
                )

                # Log memorization stats
                for k, v in mem_stats.items():
                    logger.log(f'test/memorization/{k}', v, epoch)

                upload_successful = False
                try:
                    logger.log_file('test/memorization/histogram', mem_plot_path, epoch)
                    upload_successful = True
                except Exception as e:
                    print(f"Failed to upload memorization plot: {e}")

                if upload_successful:
                    try:
                        if os.path.exists(mem_plot_path):
                            os.remove(mem_plot_path)
                    except Exception as e:
                        print(f"Failed to delete temporary plot file {mem_plot_path}: {e}")

        logging.info("Training is complete")


if __name__ == '__main__':
    agg_args = _parse_aggregate_args()
    args = parse_args_irregular()
    # Inject aggregate-only flags into the main args namespace.
    for _k, _v in vars(agg_args).items():
        setattr(args, _k, _v)

    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # This is the ImagenI2R baseline on aggregated data — never route to
    # DiffEM variants (the whole point is to run the unmodified baseline).
    if getattr(args, 'pure_diffem', False) or getattr(args, 'use_diffem', False):
        raise ValueError(
            "run_irregular_aggregate.py is the ImagenI2R baseline; "
            "--pure_diffem / --use_diffem are not supported here.  Use "
            "run_co_evolving_em_aggregate.py for DiffEM variants."
        )

    main(args)
