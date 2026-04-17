import torch
import numpy as np
import os
import logging
from tqdm import tqdm

from utils.utils_args import parse_args_irregular
from utils.utils_data import gen_dataloader
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from metrics.memorization import compute_memorization_metric

def main():
    # 1. Parse Args & Config
    args = parse_args_irregular()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    # 2. Load Data (We need training data to check for memorization)
    logger.info(f"Loading dataset: {args.dataset}")
    train_loader, test_loader, _ = gen_dataloader(args)
    
    # Collect all real training data into a single numpy array
    real_data_list = []
    for batch in train_loader:
        # Batch is typically (data, label) or just data
        x = batch[0]
        real_data_list.append(x.numpy())
    
    real_data = np.concatenate(real_data_list, axis=0)
    # real_data shape: (N, T, C) or (N, C, T) -> TS2Vec expects (N, T, C) usually
    # Check shape from run_irregular.py: x = x[:, :, :-1] (removes time feature if present)
    # The dataloader returns (B, T, C) usually for time series
    
    # If the last dimension is time, we need to handle that. 
    # Looking at run_irregular.py: x = x[:, :, :-1] -> seems to remove timestamp
    if real_data.shape[-1] > args.input_channels: 
         real_data = real_data[..., :args.input_channels]
         
    logger.info(f"Real data shape: {real_data.shape}")

    # 3. Load Model
    logger.info("Loading model...")
    model = TS2img_Karras(args=args, device=args.device).to(args.device)
    
    if args.resume and args.model_save_path:
        # Try to load the checkpoint
        # This part assumes args.model_save_path points to the checkpoint file or dir
        if os.path.isfile(args.model_save_path):
            checkpoint = torch.load(args.model_save_path, map_location=args.device)
            model.load_state_dict(checkpoint['our_model_state_dict'])
            logger.info(f"Loaded checkpoint from {args.model_save_path}")
        else:
             logger.warning("No checkpoint file found at provided path. Using initialized model (random weights).")
    
    model.eval()
    
    # 4. Generate Samples
    # We want to generate roughly the same amount as we have real data, or a fixed amount (e.g. 1000)
    num_samples = min(2000, len(real_data)) 
    logger.info(f"Generating {num_samples} samples...")
    
    gen_sig = []
    
    # Batch size for generation
    gen_batch_size = args.batch_size
    num_batches = int(np.ceil(num_samples / gen_batch_size))
    
    with torch.no_grad():
        with model.ema_scope():
            process = DiffusionProcess(args, model.net,
                                       (args.input_channels, args.img_resolution, args.img_resolution))
            
            for _ in tqdm(range(num_batches)):
                current_batch_size = min(gen_batch_size, num_samples - len(gen_sig) * gen_batch_size)
                if current_batch_size <= 0: break
                
                # Sample
                # Note: DiffusionProcess.sampling typically takes sampling_number
                x_img_sampled = process.sampling(sampling_number=current_batch_size)
                
                # Convert to TS
                x_ts = model.img_to_ts(x_img_sampled)
                gen_sig.append(x_ts.detach().cpu().numpy())
    
    generated_data = np.vstack(gen_sig)
    logger.info(f"Generated data shape: {generated_data.shape}")
    
    # 5. Compute Memorization Metric
    plot_path = "memorization_hist.png"
    stats = compute_memorization_metric(
        real_data=real_data[:len(generated_data)], # Compare against subset for speed if needed, or full
        generated_data=generated_data,
        device=args.device,
        plot_path=plot_path
    )
    
    print("\n" + "="*30)
    print("MEMORIZATION ANALYSIS RESULTS")
    print("="*30)
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    print(f"Histogram saved to: {os.path.abspath(plot_path)}")
    print("="*30)

if __name__ == "__main__":
    main()
