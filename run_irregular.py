import torch
import torch.multiprocessing
import os, sys
import numpy as np
import logging
from tqdm import tqdm

from metrics import evaluate_model_irregular
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    # Enable cuDNN auto-tuning for faster training
    torch.backends.cudnn.benchmark = True
    
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # set-up data and device (cuda > mps > cpu)
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        model = TS2img_Karras(args=args, device=args.device).to(args.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08
        )
        state = dict(model=model, epoch=0)
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None
            init_epoch = restore_state(args, state, ema_model=ema_model)

        # print model parameters
        print_model_params(logger, model)

        # --- train model ---
        logging.info(f"Continuing training loop from epoch {init_epoch}.")
        best_disc_score = float('inf')

        for epoch in range(init_epoch, args.epochs):
            print("Starting epoch %d." % (epoch,))

            model.train()
            model.epoch = epoch

            # --- train loop ---
            for i, data in enumerate(test_loader, 1):
                indices = data[0]
                x_ts = data[1].to(args.device)

                delta_prob = 0.1

                corruption_matrices = []
                hat_corruption_matrices = []

                # Generate deterministic masks based on sample index
                for idx_in_batch, dataset_idx in enumerate(indices):
                    # Use index as seed for reproducibility per sample
                    seed = int(dataset_idx.item())
                    rng = torch.Generator(device=x_ts.device).manual_seed(seed)
                    
                    # Generate corruption_matrix (A)
                    mask_a = torch.bernoulli(torch.ones_like(x_ts[idx_in_batch]) * (1 - args.missing_rate), generator=rng)
                    
                    # Generate hat_corruption_matrix (A_tilde) with extra corruption
                    extra_mask = torch.bernoulli(torch.ones_like(x_ts[idx_in_batch]) * (1 - delta_prob), generator=rng)
                    mask_a_tilde = torch.minimum(mask_a, extra_mask)
                    
                    corruption_matrices.append(mask_a)
                    hat_corruption_matrices.append(mask_a_tilde)

                corruption_matrix = torch.stack(corruption_matrices)
                hat_corruption_matrix = torch.stack(hat_corruption_matrices)

                original_images = model.ts_to_img(x_ts)
                # IMPORTANT: must be mapping-based (not value-based) because valid data may contain zeros,
                # and delay-embedding can also leave unused pixels inside the unpadded rectangle.
                padding_mask = model.ts_img.get_valid_pixel_mask(original_images)

                corruption_matrix = model.ts_to_img(corruption_matrix)
                hat_corruption_matrix = model.ts_to_img(hat_corruption_matrix)

                optimizer.zero_grad()
                loss, to_log = model.loss_fn_irregular(
                    original_images,
                    corruption_matrix,
                    hat_corruption_matrix,
                    padding_mask,
                )

                for key, value in to_log.items():
                    logger.log(f'train/{key}', value, epoch)

                scalar_loss = loss.sum() / x_ts.shape[0]
                scalar_loss.backward()

                # Handle NaN/Inf gradients (safety net, matching ambient diffusion)
                for param in model.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        # Ambient sampler (1:1) expects survival_probability = (1-p)*(1-delta)
                        delta_prob = 0.1
                        survival_probability = (1 - args.missing_rate) * (1 - delta_prob)
                        process = DiffusionProcess(
                            args,
                            model.net,
                            (args.input_channels, args.img_resolution, args.img_resolution),
                            survival_probability=survival_probability,
                            ts_img=model.ts_img,
                        )
                        for data in tqdm(test_loader):
                            # sample from the model
                            x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                            # --- convert to time series --
                            x_ts = model.img_to_ts(x_img_sampled)

                            gen_sig.append(x_ts.detach().cpu().numpy())
                            real_sig.append(data[1].detach().cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)

                scores = evaluate_model_irregular(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_irregular()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
