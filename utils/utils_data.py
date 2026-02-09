import numpy as np
import os
import sys
import torch
import torch.utils.data as Data
from numpy.f2py.auxfuncs import throw_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data

def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')

class MujocoDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, data_name, missing_rate=0.0):
        import pathlib
        here = pathlib.Path(__file__).resolve().parent.parent
        base_loc = here / 'data'
        loc = pathlib.Path(base_loc)
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.samples = tensors['mujoco_irregular']
            self.original_sample = tensors['mujoco_regular']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            # Apply missing rate to samples
            generator = torch.Generator().manual_seed(56789)
            for i in range(len(self.samples)):
                removed_points = torch.randperm(self.samples[i].shape[0], generator=generator)[
                                 :int(self.samples[i].shape[0] * missing_rate)].sort().values
                self.samples[i][removed_points, :-1] = float('nan')  # Set missing data to NaN except time column

            self.size = len(self.samples)
        else:
            raise FileNotFoundError


    def __getitem__(self, index):
        return self.original_sample[index], self.samples[index]

    def __len__(self):
        return len(self.samples)

def sine_data_generation(no, seq_len, dim, missing_rate):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    # data = list()
    irregular_dataset = list()
    ori_data = list()
    generator = torch.Generator().manual_seed(56789)
    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        ori_data.append(temp.copy())

        # Create irregular data
        removed_points = torch.randperm(temp.shape[0], generator=generator)[:int(temp.shape[0] * missing_rate)].sort().values
        temp[removed_points] = float('nan')
        idx = np.array(range(seq_len)).reshape(-1, 1)
        temp = np.concatenate((temp, idx), axis=1)
        irregular_dataset.append(temp)

    return ori_data, irregular_dataset


def compute_sigma_from_schedule(timestep_fraction, sigma_min=0.002, sigma_max=80, rho=7):
    """
    Compute sigma from Karras/EDM diffusion schedule.
    
    This follows the paper's approach of using diffusion-schedule-based noise levels.
    
    Args:
      - timestep_fraction: float in [0, 1], where 0 = max noise, 1 = min noise
      - sigma_min: minimum sigma (default from model: 0.002)
      - sigma_max: maximum sigma (default from model: 80)
      - rho: schedule parameter (default from model: 7)
    
    Returns:
      - sigma: the noise level at the given timestep
    """
    sigma = (sigma_max ** (1 / rho) + timestep_fraction * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return sigma


def add_gaussian_noise(data, noise_level=0.1):
    """
    Add Gaussian noise to the data.

    Args:
      - data: input data (numpy array)
      - noise_level: standard deviation of the Gaussian noise

    Returns:
      - noisy_data: data with added Gaussian noise
    """
    noise = np.random.normal(0, noise_level, data.shape)  # Mean=0, Std=noise_level
    noisy_data = data + noise
    return noisy_data


def real_data_loading(data_name, seq_len, missing_rate, gaussian_noise_level=0, noise_timestep=None):
    """Load and preprocess real-world data.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length
      - missing_rate: fraction of data points to remove
      - gaussian_noise_level: standard deviation of Gaussian noise to add
      - noise_timestep: if provided, compute noise level from diffusion schedule (overrides gaussian_noise_level)

    Returns:
      - data: preprocessed data.
    """
    assert data_name in ['stock', 'energy', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity']

    if data_name == 'stock':
        ori_data = np.loadtxt('./data/stock.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('./data/energy.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTh1':
        ori_data = np.loadtxt('./data/ETTh1.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTh2':
        ori_data = np.loadtxt('./data/ETTh2.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTm1':
        ori_data = np.loadtxt('./data/ETTm1.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTm2':
        ori_data = np.loadtxt('./data/ETTm2.csv', delimiter=",", skiprows=1)
    elif data_name == 'weather':
        ori_data = np.loadtxt('./data/weather.csv', delimiter=",", skiprows=1)
    elif data_name == 'electricity':
        ori_data = np.loadtxt('./data/electricity.csv', delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    irregular_dataset = ori_data.copy()
    
    # Compute noise level from diffusion schedule if noise_timestep is provided
    if noise_timestep is not None:
        gaussian_noise_level = compute_sigma_from_schedule(noise_timestep)
        print(f"Using diffusion schedule noise: timestep={noise_timestep}, sigma={gaussian_noise_level:.6f}")
    
    if gaussian_noise_level > 0:
        irregular_dataset = add_gaussian_noise(ori_data, gaussian_noise_level)
    generator = torch.Generator().manual_seed(56789)

    removed_points = torch.randperm(ori_data.shape[0], generator=generator)[
                     :int(ori_data.shape[0] * missing_rate)].sort().values
    irregular_dataset[removed_points] = float('nan')
    total_length = len(ori_data)
    index = np.array(range(total_length)).reshape(-1, 1)

    # Preprocess the data
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the data (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    # Preprocess the data
    irregular_dataset = np.concatenate((irregular_dataset, index), axis=1)
    irregular_temp_data = []
    # Cut data by sequence length
    for i in range(0, len(irregular_dataset) - seq_len):
        _x = irregular_dataset[i:i + seq_len]
        irregular_temp_data.append(_x)

    # Mix the data (to make it similar to i.i.d)
    irregular_data = []
    for i in range(len(irregular_temp_data)):
        irregular_data.append(irregular_temp_data[idx[i]])

    return data, irregular_data


def real_data_loading_with_mix_missing_rates(data_name, seq_len, gaussian_noise_level=0, noise_timestep=None):
    """Load and preprocess real-world data with mixed missing rates.
       30% of data has 30% missing rate
       40% of data has 50% missing rate
       30% of data has 70% missing rate
    """
    assert data_name in ['stock', 'energy', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity']

    if data_name == 'stock':
        ori_data = np.loadtxt('./data/stock.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('./data/energy.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTh1':
        ori_data = np.loadtxt('./data/ETTh1.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTh2':
        ori_data = np.loadtxt('./data/ETTh2.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTm1':
        ori_data = np.loadtxt('./data/ETTm1.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTm2':
        ori_data = np.loadtxt('./data/ETTm2.csv', delimiter=",", skiprows=1)
    elif data_name == 'weather':
        ori_data = np.loadtxt('./data/weather.csv', delimiter=",", skiprows=1)
    elif data_name == 'electricity':
        ori_data = np.loadtxt('./data/electricity.csv', delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)
    
    # Compute noise level from diffusion schedule if noise_timestep is provided
    if noise_timestep is not None:
        gaussian_noise_level = compute_sigma_from_schedule(noise_timestep)
        print(f"Using diffusion schedule noise: timestep={noise_timestep}, sigma={gaussian_noise_level:.6f}")
    
    # Create irregular dataset base (apply noise if needed)
    irregular_base = ori_data.copy()
    if gaussian_noise_level > 0:
        irregular_base = add_gaussian_noise(ori_data, gaussian_noise_level)
    
    total_samples = ori_data.shape[0]
    
    # Calculate split indices
    split1 = int(total_samples * 0.3)
    split2 = int(total_samples * 0.7) # 30% + 40% = 70%
    
    # Create masks/missing values
    generator = torch.Generator().manual_seed(56789)
    
    irregular_dataset = irregular_base.copy()
    
    # Apply 30% missing rate to first chunk
    # Note: We apply missingness to the whole chunk, meaning for EACH row in the chunk, we drop 30% points?
    # No, the original logic drops points globally from the flattened array or per row?
    # Original logic: 
    # removed_points = torch.randperm(ori_data.shape[0], generator=generator)[:int(ori_data.shape[0] * missing_rate)]
    # This drops ROWS? No, shape[0] is time steps usually for these CSVs?
    # Let's check original logic carefully.
    # ori_data is (T, D) usually. 
    # removed_points = randperm(shape[0])...
    # irregular_dataset[removed_points] = nan
    # This means it drops TIME STEPS entirely? 
    # Yes, for multivariate time series in these CSVs, rows are timestamps.
    # So it drops entire rows (all features at that timestamp).
    
    # Chunk 1: indices 0 to split1
    # We need to drop 30% of points WITHIN this chunk
    chunk1_len = split1
    removed_points_1 = torch.randperm(chunk1_len, generator=generator)[:int(chunk1_len * 0.3)].sort().values
    # Adjust indices to be relative to the start (0)
    irregular_dataset[removed_points_1] = float('nan')
    
    # Chunk 2: indices split1 to split2
    # We need to drop 50% of points WITHIN this chunk
    chunk2_len = split2 - split1
    removed_points_2 = torch.randperm(chunk2_len, generator=generator)[:int(chunk2_len * 0.5)].sort().values
    # Adjust indices to be relative to split1
    irregular_dataset[split1 + removed_points_2] = float('nan')
    
    # Chunk 3: indices split2 to end
    # We need to drop 70% of points WITHIN this chunk
    chunk3_len = total_samples - split2
    removed_points_3 = torch.randperm(chunk3_len, generator=generator)[:int(chunk3_len * 0.7)].sort().values
    # Adjust indices to be relative to split2
    irregular_dataset[split2 + removed_points_3] = float('nan')
    
    # Add time index column
    index = np.array(range(total_samples)).reshape(-1, 1)
    
    # Preprocess regular data (sliding window)
    temp_data = []
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
        
    # Preprocess irregular data (sliding window)
    irregular_dataset_with_idx = np.concatenate((irregular_dataset, index), axis=1)
    irregular_temp_data = []
    for i in range(0, len(irregular_dataset_with_idx) - seq_len):
        _x = irregular_dataset_with_idx[i:i + seq_len]
        irregular_temp_data.append(_x)
        
    # Now we need to mix them, but we want to mix them such that we preserve the distribution of missing rates?
    # The original code mixes everything randomly:
    # idx = np.random.permutation(len(temp_data))
    # This is fine, the resulting dataset will have samples with different missing patterns/rates mixed together.
    
    idx = np.random.permutation(len(temp_data))
    data = []
    irregular_data = []
    
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
        irregular_data.append(irregular_temp_data[idx[i]])
        
    return data, irregular_data


# =============================================================================
# DiffEM Corruption and Conditioning Utilities
# =============================================================================

def apply_corruption(clean_data, missing_rate=0.3, gaussian_noise_level=0.0, 
                     noise_timestep=None, seed=None):
    """
    Apply corruption to clean data for DiffEM M-step training pairs.
    
    This function applies the same missingness logic and optional Gaussian noise
    as real_data_loading, but operates on already-loaded clean sequences.
    
    Args:
        clean_data: numpy array or torch tensor of shape (batch, seq_len, features)
                   or (seq_len, features) for single sequence
        missing_rate: fraction of time points to set as missing (NaN)
        gaussian_noise_level: standard deviation of Gaussian noise to add
        noise_timestep: if provided, compute noise level from diffusion schedule
                       (overrides gaussian_noise_level)
        seed: random seed for reproducibility (if None, uses current RNG state)
    
    Returns:
        corrupted_data: numpy array with same shape, missing values as NaN
        mask: boolean array, True where data is observed, False where missing
    """
    # Convert to numpy if tensor
    if isinstance(clean_data, torch.Tensor):
        clean_data = clean_data.numpy()
    
    # Handle single sequence vs batch
    single_sequence = clean_data.ndim == 2
    if single_sequence:
        clean_data = clean_data[np.newaxis, ...]
    
    batch_size, seq_len, features = clean_data.shape
    corrupted = clean_data.copy()
    
    # Compute noise level from diffusion schedule if noise_timestep is provided
    if noise_timestep is not None:
        gaussian_noise_level = compute_sigma_from_schedule(noise_timestep)
    
    # Add Gaussian noise if specified
    if gaussian_noise_level > 0:
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.normal(0, gaussian_noise_level, corrupted.shape)
        corrupted = corrupted + noise
    
    # Apply missingness to each sequence
    mask = np.ones((batch_size, seq_len), dtype=bool)
    
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
    
    for b in range(batch_size):
        num_to_remove = int(seq_len * missing_rate)
        if num_to_remove > 0:
            removed_points = torch.randperm(seq_len, generator=generator)[:num_to_remove].numpy()
            corrupted[b, removed_points, :] = np.nan
            mask[b, removed_points] = False
    
    if single_sequence:
        corrupted = corrupted[0]
        mask = mask[0]
    
    return corrupted, mask


def build_conditioning(irregular_ts, tst_completed_ts=None, device='cpu'):
    """
    Build conditioning tensors for the conditional diffusion model.
    
    Args:
        irregular_ts: torch tensor of shape (batch, seq_len, features) with NaN for missing
        tst_completed_ts: torch tensor of TST-completed sequence (same shape)
                         If None, will use zero-filling for missing values
        device: target device
    
    Returns:
        cond_ts: completed time series (TST output or zero-filled)
        cond_mask: binary mask indicating observed (1) vs completed/missing (0) points
    """
    if not isinstance(irregular_ts, torch.Tensor):
        irregular_ts = torch.tensor(irregular_ts, dtype=torch.float32)
    
    irregular_ts = irregular_ts.to(device)
    
    # Create mask: 1 for observed, 0 for missing
    # NaN check across features dimension
    cond_mask = ~torch.isnan(irregular_ts).any(dim=-1)  # (batch, seq_len)
    
    if tst_completed_ts is not None:
        cond_ts = tst_completed_ts.to(device)
    else:
        # Zero-fill missing values
        cond_ts = torch.nan_to_num(irregular_ts, nan=0.0)
    
    return cond_ts, cond_mask.float()


def prepare_em_training_pair(clean_x, model_ts_to_img, missing_rate=0.3, 
                              gaussian_noise_level=0.0, noise_timestep=None,
                              tst_embedder=None, tst_decoder=None, seed=None, device='cpu'):
    """
    Prepare a (clean, corrupted, conditioning) tuple for DiffEM M-step training.
    
    Args:
        clean_x: torch tensor of clean reconstructed data (batch, seq_len, features)
        model_ts_to_img: function to convert time series to image representation
        missing_rate: fraction of points to corrupt
        gaussian_noise_level: noise level for Gaussian corruption
        noise_timestep: if set, compute noise from diffusion schedule
        tst_embedder: TST encoder for completion (optional)
        tst_decoder: TST decoder for completion (optional)
        seed: random seed for reproducibility
        device: target device
    
    Returns:
        x_img: clean image representation (target)
        cond_img: conditioning image (completed irregular)
        cond_mask_img: mask image indicating observed pixels
        y_ts: corrupted time series (for reference)
    """
    # Apply corruption to get Y ~ Q(X)
    corrupted_np, mask_np = apply_corruption(
        clean_x.cpu().numpy(), 
        missing_rate=missing_rate,
        gaussian_noise_level=gaussian_noise_level,
        noise_timestep=noise_timestep,
        seed=seed
    )
    
    y_ts = torch.tensor(corrupted_np, dtype=torch.float32, device=device)
    obs_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    
    # Get TST completion if available
    if tst_embedder is not None and tst_decoder is not None:
        with torch.no_grad():
            padding_masks = ~torch.isnan(y_ts).any(dim=-1)
            h = tst_embedder(y_ts, padding_masks)
            completed_ts = tst_decoder(h)
    else:
        completed_ts = torch.nan_to_num(y_ts, nan=0.0)
    
    # Convert to images
    x_img = model_ts_to_img(clean_x)
    cond_img = model_ts_to_img(completed_ts)
    
    # Create mask image - need to expand mask to match image structure
    # The mask is (batch, seq_len), need to convert to image space
    # For now, use the NaN pattern from y_ts converted to image
    y_img_for_mask = model_ts_to_img(y_ts, pad_val=1.0)  # Use 1 as pad for mask
    cond_mask_img = (~torch.isnan(y_img_for_mask)).float()
    
    return x_img, cond_img, cond_mask_img, y_ts


class EMReconstructionDataset(torch.utils.data.Dataset):
    """
    Dataset for DiffEM M-step training that holds reconstructed clean data
    and generates corrupted pairs on-the-fly or from cache.
    """
    
    def __init__(self, reconstructed_data, missing_rate=0.3, 
                 gaussian_noise_level=0.0, noise_timestep=None,
                 precompute_corruption=False, seed=None):
        """
        Args:
            reconstructed_data: numpy array or tensor of reconstructed clean sequences
                               shape: (N, seq_len, features)
            missing_rate: corruption missing rate
            gaussian_noise_level: Gaussian noise level
            noise_timestep: if set, compute noise from diffusion schedule
            precompute_corruption: if True, generate all corruptions at init
            seed: base seed for reproducibility
        """
        if isinstance(reconstructed_data, torch.Tensor):
            self.clean_data = reconstructed_data.numpy()
        else:
            self.clean_data = np.array(reconstructed_data)
        
        self.missing_rate = missing_rate
        self.gaussian_noise_level = gaussian_noise_level
        self.noise_timestep = noise_timestep
        self.base_seed = seed
        
        self.corrupted_cache = None
        self.mask_cache = None
        
        if precompute_corruption:
            self._precompute_corruptions()
    
    def _precompute_corruptions(self):
        """Pre-compute all corruptions for the dataset."""
        self.corrupted_cache = []
        self.mask_cache = []
        
        for i in range(len(self.clean_data)):
            seed = self.base_seed + i if self.base_seed is not None else None
            corrupted, mask = apply_corruption(
                self.clean_data[i],
                missing_rate=self.missing_rate,
                gaussian_noise_level=self.gaussian_noise_level,
                noise_timestep=self.noise_timestep,
                seed=seed
            )
            self.corrupted_cache.append(corrupted)
            self.mask_cache.append(mask)
        
        self.corrupted_cache = np.array(self.corrupted_cache)
        self.mask_cache = np.array(self.mask_cache)
    
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, idx):
        clean = torch.tensor(self.clean_data[idx], dtype=torch.float32)
        
        if self.corrupted_cache is not None:
            corrupted = torch.tensor(self.corrupted_cache[idx], dtype=torch.float32)
            mask = torch.tensor(self.mask_cache[idx], dtype=torch.float32)
        else:
            seed = self.base_seed + idx if self.base_seed is not None else None
            corrupted_np, mask_np = apply_corruption(
                self.clean_data[idx],
                missing_rate=self.missing_rate,
                gaussian_noise_level=self.gaussian_noise_level,
                noise_timestep=self.noise_timestep,
                seed=seed
            )
            corrupted = torch.tensor(corrupted_np, dtype=torch.float32)
            mask = torch.tensor(mask_np, dtype=torch.float32)
        
        return clean, corrupted, mask


def save_reconstructions(reconstructions, save_path, em_iter):
    """
    Save reconstructed dataset to disk for a given EM iteration.
    
    Args:
        reconstructions: numpy array or tensor of reconstructed sequences
        save_path: base directory path
        em_iter: EM iteration number
    """
    import os
    
    iter_path = os.path.join(save_path, f'em_iter_{em_iter}')
    os.makedirs(iter_path, exist_ok=True)
    
    if isinstance(reconstructions, torch.Tensor):
        reconstructions = reconstructions.cpu().numpy()
    
    np.save(os.path.join(iter_path, 'reconstructions.npy'), reconstructions)
    print(f"Saved {len(reconstructions)} reconstructions to {iter_path}")


def load_reconstructions(save_path, em_iter):
    """
    Load reconstructed dataset from disk for a given EM iteration.
    
    Args:
        save_path: base directory path
        em_iter: EM iteration number
    
    Returns:
        reconstructions: numpy array of reconstructed sequences
    """
    import os
    
    iter_path = os.path.join(save_path, f'em_iter_{em_iter}')
    recon_file = os.path.join(iter_path, 'reconstructions.npy')
    
    if not os.path.exists(recon_file):
        raise FileNotFoundError(f"No reconstructions found at {recon_file}")
    
    return np.load(recon_file)


def gen_dataloader(args):
    if args.dataset == 'sine':
        args.dataset_size = 10000
        ori_data, irregular_data = sine_data_generation(args.dataset_size, args.seq_len, args.input_channels, args.missing_rate)
        ori_data = torch.Tensor(np.array(ori_data))
        ori_train_set = Data.TensorDataset(ori_data)
        irregular_data = torch.Tensor(np.array(irregular_data))
        irregular_train_set = Data.TensorDataset(irregular_data)

    elif args.dataset in ['stock', 'energy', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity']:
        noise_timestep = getattr(args, 'noise_timestep', None)
        if hasattr(args, 'mix_missing_rates') and args.mix_missing_rates:
            ori_data, irregular_data_np = real_data_loading_with_mix_missing_rates(args.dataset, args.seq_len, gaussian_noise_level=args.gaussian_noise_level, noise_timestep=noise_timestep)
        else:
            ori_data, irregular_data_np = real_data_loading(args.dataset, args.seq_len, missing_rate=args.missing_rate, gaussian_noise_level=args.gaussian_noise_level, noise_timestep=noise_timestep)
            
        ori_data = torch.Tensor(np.array(ori_data))
        ori_train_set = Data.TensorDataset(ori_data)
        irregular_data = torch.Tensor(np.array(irregular_data_np))
        irregular_train_set = Data.TensorDataset(irregular_data)

    elif args.dataset in ['mujoco']:
        train_set = MujocoDataset(args.seq_len, args.dataset, missing_rate=args.missing_rate)
        ori_data = list()
        irregular_data = list()
        for ori_data_b, irregular_data_b in train_set:
            ori_data.append(ori_data_b)
            irregular_data.append(irregular_data_b)
        ori_data = torch.Tensor(np.array(ori_data))
        ori_train_set = Data.TensorDataset(ori_data)
        irregular_data = torch.Tensor(np.array(irregular_data))
        irregular_train_set = Data.TensorDataset(irregular_data)

    train_loader = Data.DataLoader(dataset=irregular_train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    test_loader = Data.DataLoader(dataset=ori_train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)

    # For the time series benchmark, the entire dataset for both training and testing
    return train_loader, test_loader, None
