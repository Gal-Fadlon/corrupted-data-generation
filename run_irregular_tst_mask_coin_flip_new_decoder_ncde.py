import os, sys
import torch
import numpy as np
import torch.multiprocessing
import logging
from tqdm import tqdm
from torch import optim
from itertools import chain
from metrics import evaluate_model_irregular
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.our_no_mask_on_noise import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags
from models.decoder import TST_Decoder
from utils.utils_data import gen_dataloader, Sine_irregular, MujocoDataset_Ilan, TimeDataset_irregular, \
    TimeDataset_irregular_removed_points
from utils.utils_args import parse_args_irregular
from models.TST import TSTransformerEncoder, TSTransformerDecoder
from models.neuralODE import Multi_Layer_ODENetwork
import torch.nn.functional as F
import random
from pathlib import Path
import shutil
import math


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def create_random_mask(x_img, zero_ratio=0.3):
    """
    Create a mask with the same shape as x_img with all values as 1,
    and randomly set a specified ratio of values to 0.

    Args:
    - x_img (torch.Tensor): The input tensor to base the mask on.
    - zero_ratio (float): The fraction of values to randomly set to 0 (default is 0.3).

    Returns:
    - torch.Tensor: A mask with the same shape as x_img with 30% zeros.
    """
    # Create a mask of all ones
    mask = torch.ones_like(x_img)

    # Calculate the number of elements to set to zero
    num_elements = mask.numel()
    num_zero_elements = int(num_elements * zero_ratio)

    # Get random indices to set to zero
    indices = torch.randperm(num_elements)[:num_zero_elements]

    # Flatten, modify, and reshape the mask
    mask_flat = mask.view(-1)
    mask_flat[indices] = 0
    return mask_flat.view_as(mask)


def create_contained_mask(mask: torch.Tensor, zero_rate: float) -> torch.Tensor:
    # Save the original mask to check for unintended changes later
    original_mask = mask.clone()

    # Calculate the current zero rate
    current_zero_rate = (mask == 0).float().mean().item()

    # If the current zero rate is less than the requested zero rate, return the original mask
    if current_zero_rate < zero_rate:
        return mask

    # Calculate the target zero count to achieve the desired zero rate
    target_zero_count = int(zero_rate * mask.numel())
    current_zero_count = int(current_zero_rate * mask.numel())
    excess_zero_count = current_zero_count - target_zero_count

    # If no adjustment is needed, return the original mask
    if excess_zero_count <= 0:
        return mask

    # Identify the indices of current 0s
    zero_indices = (mask == 0).nonzero(as_tuple=True)

    # Randomly select indices to turn from '0' to '1' to meet the target zero rate
    selected_indices = torch.randperm(len(zero_indices[0]))[:excess_zero_count]
    mask[zero_indices[0][selected_indices], zero_indices[1][selected_indices]] = 1

    # Verify that no '1' values were changed to '0'
    if not torch.all((original_mask == 1) <= (mask == 1)):
        raise ValueError("Error: Some '1' values were incorrectly changed to '0'.")

    return mask


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
            # Loop through each element in the sequence from the second element onward
            # i = 1  # Start from the second element
            # while i < sequence.size(0):
            #     # If the current element is NaN and the previous one is not NaN, replace it
            #     if torch.isnan(sequence[i]).any() and not torch.isnan(sequence[i - 1]).any():
            #         sequence[i] = sequence[i - 1]
            #         i += 1
            #     else:
            #         i += 1
    return tensor

def propagate_values_backward(tensor):
    # Iterate over the batch
    for b in range(tensor.size(0)):
        # Extract the sequence for the current batch and channel
        sequence = tensor[b]

        # Loop through each element in the sequence from the second element onward
        i = 1  # Start from the second element
        while i < sequence.size(0):
            # If the current element is NaN and the previous one is not NaN, replace it
            if torch.isnan(sequence[0]).any() and not torch.isnan(sequence[i]).any():
                sequence[:i] = sequence[i]
            i += 1

    return tensor

def propagate_values(tensor):
    tensor = propagate_values_forward(tensor)
    return tensor
    # return propagate_values_backward(tensor)


def nan_percentage(tensor):
    # Count the number of NaN values in the tensor
    nan_count = torch.isnan(tensor).sum().item()

    # Calculate the total number of elements in the tensor
    total_elements = tensor.numel()

    # Calculate the percentage of NaN values
    nan_percentage = nan_count / total_elements
    return nan_percentage

def update_nan_values(original_tensor, blackbox_output):
    """
    Replace the NaN values in the original tensor with the corresponding values
    from the blackbox output, keeping the non-NaN values unchanged.

    Args:
        original_tensor (torch.Tensor): The input tensor with potential NaN values.
        blackbox_output (torch.Tensor): The output tensor from the blackbox, which may modify all values.

    Returns:
        torch.Tensor: The updated tensor where NaN values have been replaced with blackbox output.
    """
    # Create a copy of the original tensor to avoid modifying the input tensor
    updated_tensor = original_tensor.clone()


    # Create a mask to identify NaN points in the original tensor
    nan_mask = torch.isnan(updated_tensor)

    # Replace NaN values with corresponding values from blackbox output
    updated_tensor[nan_mask] = blackbox_output[nan_mask]

    return updated_tensor

def create_model_directories(args):
    """
    Creates a main directory based on the given arguments and two subdirectories: TST_Model and Our_Model.

    Args:
        args: An object with attributes `dataset` and `seq_length`.
    """
    # Main path
    main_path = '/cs/cs_groups/azencot_group/Irregular_TS_Generation_Gal_Idan'

    # Main directory name
    directory_name = f"Trained_Model_data_set={args.dataset}_sequence_length={args.seq_len}_model=Ours"
    directory_path = Path(f'{main_path}/{directory_name}')

    # Check if the main directory exists
    if directory_path.exists() and directory_path.is_dir():
        # Remove the existing directory
        shutil.rmtree(directory_path)
        print(f"Existing directory {directory_name}' removed.")

    # Create the main directory
    directory_path.mkdir()
    print(f"Directory '{directory_name}' created.")

    # Create subdirectories
    tst_model_path = directory_path / "TST_Model"
    our_model_path = directory_path / "Our_Model"

    tst_model_path.mkdir()
    print(f"Subdirectory '{tst_model_path}' created.")

    our_model_path.mkdir()
    print(f"Subdirectory '{our_model_path}' created.")

    return tst_model_path, our_model_path


def save_checkpoint(args, our_model, our_optimizer, ema_model, encoder, decoder, tst_optimizer, disc_score, pred_score=None, fid_score=None, correlation_score=None):
    """
    Saves the model checkpoint to the specified directory based on args and disc_score.
    """
    import os
    import glob

    try:
        main_path = '/cs/cs_groups/azencot_group/ts2imgIrregular/models/ours'
        seq_len = args.seq_len
        data_set_name = args.dataset
        missing_rate = int(args.missing_rate * 100)

        # Build the directory structure
        full_path = os.path.join(main_path, f'seq_len_{seq_len}', data_set_name, f'missing_rate_{missing_rate}')
        os.makedirs(full_path, exist_ok=True)  # Ensure the directories exist

        # # # Remove any existing checkpoint in this directory
        # for old_file in glob.glob(os.path.join(full_path, "disc_score_*.pth")):
        #     old_file_disc_score = old_file.split('_')[0]
        #     should_delete = float(old_file_disc_score) > disc_score
        #     if should_delete:
        #         os.remove(old_file)

        # Generate the file name
        if pred_score is not None:
            filename = f"disc_score_{disc_score}_pred_score_{pred_score}_fid_score_{fid_score}_correlation_score_{correlation_score}.pth"
        else:
            filename = f"disc_score_{disc_score}.pth"
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
        # raise



def _loss_e_t0(x_tilde, x):
    return F.mse_loss(x_tilde, x)

def _loss_e_0(loss_e_t0):
    return torch.sqrt(loss_e_t0) * 10


def main(args):
    import torch.utils.data as Data
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # set-up data and device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.dataset == 'sine':
            dataset = Sine_irregular(args.seq_len, args.dataset, args.missing_rate)
        elif args.dataset == 'mujoco':
            dataset = MujocoDataset_Ilan(args.seq_len, args.dataset, args.path, missing_rate=args.missing_rate)
        else:
            dataset = TimeDataset_irregular_removed_points(args.seq_len, args.dataset, args.missing_rate, args.gaussian_noise_level)

        train_loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # store original data
        ori_data = list()
        for data in train_loader:
            ori_data.append(data["original_data"].detach().cpu().numpy())
        ori_data = np.vstack(ori_data)

        args.input_size = ori_data.shape[-1]
        args.img_channels = ori_data.shape[-1]
        args.inp_dim = ori_data.shape[-1]

        model = TS2img_Karras(args=args, device=args.device).to(args.device)

        # ncde
        args.num_layers = 3
        args.hidden_dim_ncde = 20
        x_hidden = 48
        # create model
        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        init_epoch = 0


        logging.info(args.dataset + ' dataset is ready.')

        if args.use_stft:
            model.init_stft_embedder(train_loader)

        from run_timegan_irregular import FinalTanh
        from models.neuralCDE import NeuralCDE

        ode_func = FinalTanh(args.input_size, args.hidden_dim_ncde, args.hidden_dim_ncde, args.num_layers)
        embedder = NeuralCDE(func=ode_func, input_channels=args.input_size,
                             hidden_channels=args.hidden_dim_ncde, output_channels=args.hidden_dim_ncde).to(args.device)
        recovery = Multi_Layer_ODENetwork(input_size=args.hidden_dim_ncde, hidden_size=args.hidden_dim_ncde, output_size=args.input_size,
                                          gru_input_size=args.hidden_dim_ncde, x_hidden=x_hidden, num_layer=args.r_layer,
                                          last_activation=args.last_activation_r, delta_t=0.5).to(args.device)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        state = dict(model=model, epoch=0)
        init_epoch = 0

        optimizer_er = optim.Adam(chain(embedder.parameters(), recovery.parameters()))
        embedder.train()
        recovery.train()

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None # load ema model if available
            init_epoch = restore_state(args, state, ema_model=ema_model)

        # print model parameters
        print_model_params(logger, model)


        # Create model directory's to save the models
        # tst_model_path, our_model_path = create_model_directories(args)

        # --- train model ---
        logging.info(f"Continuing training loop from epoch {init_epoch}.")
        best_score = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics

        time_dict = {}
        total_training_time_in_minutes = 0
        sampling_time_minutes_list = []
        benchmark_time_minutes_list = []
        sampling_and_benchmarks_total_time_in_minutes = 0

        import time as timer
        start_tst_time = timer.time()
        start_total_training_time = timer.time()

        print('logging_iter', args.logging_iter)
        for step in range(1, 1 + 1):
            for i, data in enumerate(train_loader, 1):
                x = data["data"].float().to(args.device)
                train_coeffs = data['inter']
                obs = x[:, :, -1]  # Last column is time indices
                x_features = x[:, :, :-1]  # All columns except last (features only)
                time_indices = obs.long()  # Time indices for alignment

                time = torch.FloatTensor(list(range(24))).to(args.device)
                final_index = (torch.ones(x.shape[0]) * 23).to(args.device).float()
                h = embedder(time, train_coeffs, final_index)
                x_regular = recovery(h, obs)

                # NEW APPROACH: Align reduced input with full recovery output
                # Extract recovery values only at the time points that were kept
                batch_size, num_kept_points, num_features = x_features.shape
                x_regular_aligned = torch.zeros_like(x_features)

                for b in range(batch_size):
                    for t in range(num_kept_points):
                        time_idx = time_indices[b, t].long()
                        if 0 <= time_idx < x_regular.shape[1]:
                            x_regular_aligned[b, t, :] = x_regular[b, time_idx, :]

                # Now both tensors have the same shape, compute loss directly
                x_regular_flat = x_regular_aligned.reshape(-1, num_features)
                x_features_flat = x_features.reshape(-1, num_features)
                loss_e_t0 = _loss_e_t0(x_regular_flat, x_features_flat)
                loss_e_0 = _loss_e_0(loss_e_t0)
                optimizer_er.zero_grad()
                loss_e_0.backward()
                optimizer_er.step()
                torch.cuda.empty_cache()

                if step % 10 == 0:
                    print(
                        "step: "
                        + str(step)
                        + "/"
                        + str(args.first_epoch)
                        + ", loss_e: "
                        + str(np.round(np.sqrt(loss_e_t0.item()), 4))
                    )
                    logger.log('loss/ncde', np.round(np.sqrt(loss_e_t0.item()), 4))

        end_tst_time = timer.time()
        # Log the tst training time in minutes
        tst_training_time_in_minutes = (end_tst_time - start_tst_time) / 60
        logger.log('TST training time in minutes', tst_training_time_in_minutes)

        time_dict['tst_training_time_in_minutes'] = tst_training_time_in_minutes

        for epoch in range(init_epoch, args.epochs):
            print("Starting epoch %d." % (epoch,))
            # if epoch > 100:
            #     args.logging_iter = 10
            model.train()
            model.epoch = epoch
            logger.log_name_params('train/epoch', epoch)

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                # x = data[0].to(args.device)
                # x_ts = x[:, :, :-1]
                # times = x[:, :, -1]

                x = data["data"].float().to(args.device)
                obs = x[:, :, -1]
                x_ts = x[:, :, :-1]
                time = torch.FloatTensor(list(range(24))).to(args.device)
                final_index = (torch.ones(x.shape[0]) * 23).to(args.device).float()
                train_coeffs = data['inter']

                if random.random() < 0:
                    # FIXED: Use FULL imputed sequence for diffusion model training
                    h = embedder(time, train_coeffs, final_index)
                    x_recon_full = recovery(h, obs)  # This is FULL length (24)
                    
                    # Create proper mask based on originally observed vs missing time points
                    batch_size = x_recon_full.shape[0]
                    seq_len = x_recon_full.shape[1]  # Should be 24
                    num_features = x_recon_full.shape[2]
                    
                    # Create time-domain mask: 1 for originally observed, 0 for missing
                    time_mask = torch.zeros(batch_size, seq_len, num_features).to(args.device)
                    
                    for b in range(batch_size):
                        # obs contains the time indices that were originally kept
                        kept_time_indices = obs[b, :].long()  # Shape: [num_kept_points]
                        for t_idx in kept_time_indices:
                            if 0 <= t_idx < seq_len:
                                time_mask[b, t_idx, :] = 1.0
                    
                    # Convert time-domain mask to image-domain mask
                    mask_img = model.ts_to_img(time_mask)
                    
                    # Use full imputed sequence for diffusion model (like original NaN approach)
                    x_img = model.ts_to_img(x_recon_full)
                    optimizer.zero_grad()
                    loss = model.loss_fn_irregular(x_img, mask_img)
                    if len(loss) == 2:
                        loss, to_log = loss

                        for key, value in to_log.items():
                            logger.log(f'train/{key}', value, epoch)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    model.on_train_batch_end()
                else:
                    # FIXED: Use FULL imputed sequence for diffusion model training, not reduced sequence
                    h = embedder(time, train_coeffs, final_index)
                    x_recon_full = recovery(h, obs)  # This is FULL length (24)
                    
                    # Create proper mask based on originally observed vs missing time points
                    batch_size = x_recon_full.shape[0]
                    seq_len = x_recon_full.shape[1]  # Should be 24
                    num_features = x_recon_full.shape[2]
                    
                    # Create time-domain mask: 1 for originally observed, 0 for missing
                    time_mask = torch.zeros(batch_size, seq_len, num_features).to(args.device)
                    
                    for b in range(batch_size):
                        # obs contains the time indices that were originally kept
                        kept_time_indices = obs[b, :].long()  # Shape: [num_kept_points]
                        for t_idx in kept_time_indices:
                            if 0 <= t_idx < seq_len:
                                time_mask[b, t_idx, :] = 1.0
                    
                    # Convert time-domain mask to image-domain mask
                    mask_img = model.ts_to_img(time_mask)
                    
                    # For diffusion model: use FULL imputed sequence (like original NaN approach)
                    x_img = model.ts_to_img(x_recon_full)
                    
                    # --- convert the full reconstruction to image representation --- #
                    x_tilde_img = model.ts_to_img(x_recon_full)
                    loss = model.loss_fn_irregular(x_tilde_img, mask_img)
                    optimizer.zero_grad()
                    if len(loss) == 2:
                        loss, to_log = loss
                        for key, value in to_log.items():
                            logger.log(f'train/{key}', value, epoch)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    model.on_train_batch_end()

                    #############Recovery######################
                    h = embedder(time, train_coeffs, final_index)
                    x_regular = recovery(h, obs)

                    # NEW APPROACH: Align reduced input with full recovery output for x_ts
                    # x_ts has reduced shape, x_regular has full shape - need to align
                    batch_size, num_kept_points, num_features = x_ts.shape
                    x_regular_aligned = torch.zeros_like(x_ts)

                    for b in range(batch_size):
                        for t in range(num_kept_points):
                            time_idx = obs[b, t].long()  # obs contains time indices
                            if 0 <= time_idx < x_regular.shape[1]:
                                x_regular_aligned[b, t, :] = x_regular[b, time_idx, :]

                    # Now both tensors have the same shape, compute loss directly
                    x_regular_flat = x_regular_aligned.reshape(-1, num_features)
                    x_ts_flat = x_ts.reshape(-1, num_features)
                    loss_e_t0 = _loss_e_t0(x_regular_flat, x_ts_flat)

                    loss_e_0 = _loss_e_0(loss_e_t0)
                    loss_e = loss_e_0
                    optimizer_er.zero_grad()
                    loss_e.backward()
                    optimizer_er.step()
                    torch.cuda.empty_cache()

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0 and epoch > 1:
                start_net_single_sampling_time = timer.time()
                start_sampling_and_benchmarks_time = timer.time()
                gen_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        for data in tqdm(train_loader):
                            # sample from the model
                            x_img_sampled = process.sampling(sampling_number=data["original_data"].shape[0])
                            # --- convert to time series --
                            x_ts = model.img_to_ts(x_img_sampled)

                            # special case for temperature_rain dataset
                            if args.dataset in ['temperature_rain']:
                                x_ts = torch.clamp(x_ts, 0, 1)

                            gen_sig.append(x_ts.detach().cpu().numpy())

                end_net_single_sampling_time = timer.time()
                net_single_sampling_time_in_minutes = (end_net_single_sampling_time - start_net_single_sampling_time) / 60
                sampling_time_minutes_list.append(net_single_sampling_time_in_minutes)
                time_dict['net_single_sampling_time_in_minutes_average'] = sum(sampling_time_minutes_list) / len(sampling_time_minutes_list)
                # logger.log('net_single_sampling_time_in_minutes', net_single_sampling_time_in_minutes)

                gen_sig = np.vstack(gen_sig)

                start_benchmarks_time = timer.time()

                scores = evaluate_model_irregular(ori_data, gen_sig, args, calc_other_metrics=False)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

                # --- save checkpoint ---
                curr_score = scores['marginal_score_mean'] if 'marginal_score_mean' in scores else scores['disc_mean']

                if curr_score > best_score:
                    end_sampling_and_benchmarks_time = timer.time()
                    sampling_and_benchmarks_total_time_in_minutes += (end_sampling_and_benchmarks_time - start_sampling_and_benchmarks_time) / 60

                if curr_score < best_score:
                    new_scores = args.dataset != 'electricity' and args.seq_len != 10920
                    if new_scores:
                        new_scores = evaluate_model_irregular(ori_data, gen_sig, args, calc_other_metrics=False)

                    end_benchmarks_time = timer.time()
                    benchmarks_time_in_minutes = (end_benchmarks_time - start_benchmarks_time) / 60
                    benchmark_time_minutes_list.append(benchmarks_time_in_minutes)
                    time_dict['net_single_benchmark_time_in_minutes_average'] = sum(benchmark_time_minutes_list) / len(benchmark_time_minutes_list)
                    # logger.log('benchmarks_time_in_minutes', benchmarks_time_in_minutes)

                    end_sampling_and_benchmarks_time = timer.time()
                    sampling_and_benchmarks_total_time_in_minutes += (end_sampling_and_benchmarks_time - start_sampling_and_benchmarks_time) / 60

                    # if new_scores:
                    #     for key, value in new_scores.items():
                    #         logger.log(f'test/{key}', value, epoch)
                    #
                    #     pred_score = new_scores['pred_mean']
                    #     fid_score = new_scores['fid_mean']
                    #     correlation_score = new_scores['correlation_score_mean']

                    pred_score = None
                    fid_score = None
                    correlation_score = None

                    end_total_training_time = timer.time()
                    total_training_time_in_minutes = ((end_total_training_time - start_total_training_time) / 60) - sampling_and_benchmarks_total_time_in_minutes
                    print('total_training_time_in_minutes ', total_training_time_in_minutes, 'Epoch ', epoch)
                    # logger.log('sampling_and_benchmarks_total_time_in_minutes', sampling_and_benchmarks_total_time_in_minutes)

                    time_dict[f'disc_score_{curr_score}_hours'] = total_training_time_in_minutes / 60
                    time_dict['sampling_and_benchmarks_total_time_in_minutes'] = sampling_and_benchmarks_total_time_in_minutes
                    logger.log('total_time_in_hours', total_training_time_in_minutes / 60)
                    print(time_dict)
                    best_score = curr_score
                    ema_model = model.model_ema if args.ema else None
                    # if new_scores:
                    #     save_checkpoint(args=args, our_model=model, our_optimizer=optimizer, ema_model=ema_model, encoder=embedder, decoder=decoder, tst_optimizer=optimizer_er, disc_score=best_score, pred_score=pred_score, fid_score=fid_score, correlation_score=correlation_score)
                    #
                    # else:
                    #     save_checkpoint(args=args, our_model=model, our_optimizer=optimizer, ema_model=ema_model,
                    #                     encoder=embedder, decoder=decoder, tst_optimizer=optimizer_er,
                    #                     disc_score=best_score)

                if epoch > 300 and curr_score > 0.31:
                    return
        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_irregular()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
