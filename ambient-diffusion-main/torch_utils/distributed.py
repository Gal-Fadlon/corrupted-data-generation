# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
from . import training_stats

#----------------------------------------------------------------------------

def init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    # Prefer CUDA > MPS > CPU. Use NCCL when CUDA is available; otherwise Gloo.
    use_cuda = torch.cuda.is_available()
    use_mps = (not use_cuda) and torch.backends.mps.is_available()

    backend = os.environ.get('TORCH_DISTRIBUTED_BACKEND')
    if backend is None:
        backend = 'nccl' if (os.name != 'nt' and use_cuda) else 'gloo'

    torch.distributed.init_process_group(backend=backend, init_method='env://')
    if use_cuda:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    device_type = 'cuda' if use_cuda else ('mps' if use_mps else None)
    sync_device = torch.device(device_type) if (get_world_size() > 1 and device_type) else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

#----------------------------------------------------------------------------

def should_stop():
    return False

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
