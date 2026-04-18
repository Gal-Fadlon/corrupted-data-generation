"""
Unconditional diffusion model training on fully-observed (regular) time series.

Adapted from the original run_regular setup to work with the current codebase:
  - WandbLogger replaces the removed NeptuneLogger
  - loss_fn_irregular serves as the regular loss (mask is all-ones for clean data)
  - DiffusionProcess constructor updated to current signature
"""

import torch
import torch.multiprocessing
import torch.nn.functional as F
import os, sys
import numpy as np
import logging
from tqdm import tqdm

from metrics import evaluate_model_irregular
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_regular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    name = create_model_name_and_dir(args)

    logging.info(args)

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:

        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        model = TS2img_Karras(args=args, device=args.device).to(args.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        state = dict(model=model, epoch=0)
        init_epoch = 0

        if args.resume:
            ema_model = model.model_ema if args.ema else None
            init_epoch = restore_state(args, state, ema_model=ema_model)

        print_model_params(logger, model)

        for epoch in range(init_epoch, args.epochs):
            print("Starting epoch %d." % (epoch,))

            model.train()
            model.epoch = epoch

            for i, data in enumerate(train_loader, 1):
                x = data[0].to(args.device)
                x_ts = x[:, :, :-1]

                x_tilde_img = model.ts_to_img(x_ts)
                loss, to_log = model.loss_fn_irregular(x_tilde_img)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

                for key, value in to_log.items():
                    logger.log(f'train/{key}', value, epoch)

            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(
                            args, model.net,
                            (args.input_channels, args.img_resolution, args.img_resolution)
                        )
                        for data in tqdm(test_loader):
                            x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                            x_ts = model.img_to_ts(x_img_sampled)

                            gen_sig.append(x_ts.detach().cpu().numpy())
                            real_sig.append(data[0].detach().cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)

                scores = evaluate_model_irregular(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_regular()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
