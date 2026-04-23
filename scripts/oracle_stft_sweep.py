"""Oracle STFT ceiling sweep (no model, no EM).

For a given dataset and sequence length, load the real train split,
build an STFTEmbedder with candidate (n_fft, hop_length) tilings,
compute the oracle roundtrip

    x_rt = img_to_ts(ts_to_img(x))

and score it with the same TS2Vec/GRU discriminator used by the full
pipeline. The goal is to expose the best-case ceiling of the STFT lift
BEFORE burning a 20 h EM run: if even the oracle roundtrip is
distinguishable from the real data, no amount of MMPS+EM can close it.

Usage:
    python scripts/oracle_stft_sweep.py \\
        --dataset ETTh1 --seq_len 24 \\
        --tiles 8,4 12,6 16,8 \\
        --n_samples 1024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.img_transformations import STFTEmbedder
from utils.utils_data import MinMaxScaler
from metrics.discriminative_torch import discriminative_score_metrics


def _load_dataset(data_name: str) -> np.ndarray:
    path = os.path.join('data', f'{data_name}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing dataset file: {path}')
    ori = np.loadtxt(path, delimiter=',', skiprows=1)
    ori = ori[::-1]
    ori = MinMaxScaler(ori)
    return ori


def _window_data(series: np.ndarray, seq_len: int, n_samples: int,
                 seed: int = 0) -> np.ndarray:
    total = len(series)
    rng = np.random.RandomState(seed)
    max_start = total - seq_len
    starts = rng.randint(0, max_start, size=n_samples)
    windows = np.stack([series[s:s + seq_len] for s in starts]).astype(np.float32)
    return windows


def _parse_tiles(raw: List[str]) -> List[Tuple[int, int]]:
    out = []
    for tok in raw:
        a, b = tok.split(',')
        out.append((int(a), int(b)))
    return out


def _oracle_roundtrip(windows: np.ndarray, n_fft: int, hop: int,
                      device: str) -> Tuple[np.ndarray, dict]:
    """Build an STFT embedder, cache zscore stats, return (x_rt, diag)."""
    x = torch.as_tensor(windows, dtype=torch.float32)
    _, seq_len, feat = x.shape

    native_h = n_fft // 2 + 1
    native_w = seq_len // hop + 1
    native_side = max(native_h, native_w)
    img_resolution = None
    for target in (8, 16, 32, 64):
        if native_side <= target:
            img_resolution = target
            break
    if img_resolution is None:
        raise ValueError(
            f'tile (n_fft={n_fft}, hop={hop}) produces native {native_h}x{native_w}, '
            f'no power-of-two img_resolution up to 64 fits.'
        )

    embedder = STFTEmbedder(
        device=device,
        seq_len=seq_len,
        n_fft=n_fft,
        hop_length=hop,
        img_resolution=img_resolution,
        scale_mode='zscore',
        pad_mode='reflect',
    )
    embedder.cache_min_max_params(x)

    with torch.no_grad():
        x_dev = x.to(device)
        img = embedder.ts_to_img(x_dev)
        x_rt = embedder.img_to_ts(img).detach().cpu().numpy().astype(np.float32)

        img_mean = img.mean().item()
        img_std = img.std().item()
        img_max = img.abs().max().item()
        sr_max = embedder.scale_real.max().item()
        sr_min = embedder.scale_real.min().item()
        rt_err = float(
            (torch.as_tensor(x_rt) - x.cpu()).norm().item()
            / max(x.norm().item(), 1e-8)
        )

    diag = {
        'n_fft': n_fft,
        'hop_length': hop,
        'img_resolution': img_resolution,
        'image_pixel_std': img_std,
        'image_pixel_mean': img_mean,
        'image_abs_max': img_max,
        'scale_real_max': sr_max,
        'scale_real_min': sr_min,
        'scale_ratio': sr_max / max(sr_min, 1e-12),
        'roundtrip_rel_err': rt_err,
    }
    return x_rt, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--n_samples', type=int, default=1024)
    parser.add_argument('--tiles', type=str, nargs='+',
                        default=['8,4', '12,6', '16,8'])
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default=None,
                        help='Optional JSON path for the full result dict')
    args = parser.parse_args()

    print(f'[oracle-sweep] dataset={args.dataset} seq_len={args.seq_len} '
          f'n_samples={args.n_samples} device={args.device}')

    series = _load_dataset(args.dataset)
    windows = _window_data(series, args.seq_len, args.n_samples, seed=args.seed)
    feat = windows.shape[-1]
    print(f'[oracle-sweep] loaded {windows.shape} features={feat}')

    tiles = _parse_tiles(args.tiles)
    results = []

    class _DiscArgs:
        device = args.device
        input_size = feat

    for n_fft, hop in tiles:
        print(f'\n[tile] n_fft={n_fft} hop={hop}')
        if hop > n_fft // 2 + 1:
            print('  skip: hop > n_fft//2+1 (COLA will fail catastrophically)')
            continue

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        try:
            x_rt, diag = _oracle_roundtrip(windows, n_fft, hop, args.device)
        except (RuntimeError, ValueError) as exc:
            print(f'  skip: {exc}')
            continue

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        disc = discriminative_score_metrics(windows, x_rt, _DiscArgs())

        diag['oracle_disc_mean'] = float(disc)
        cola_perfect = (hop == n_fft // 2)
        diag['cola_perfect_hann'] = cola_perfect

        print(f'  img_pixel_std={diag["image_pixel_std"]:.4f}  '
              f'scale_ratio={diag["scale_ratio"]:.1e}  '
              f'rt_rel_err={diag["roundtrip_rel_err"]:.2e}  '
              f'oracle_disc={disc:.4f}  cola={cola_perfect}')

        results.append(diag)

    results.sort(key=lambda d: d['oracle_disc_mean'])

    print('\n' + '=' * 72)
    print('ORACLE CEILING RANKING (lower disc = tighter lift)')
    print('=' * 72)
    print(f'{"n_fft":>6} {"hop":>4} {"img_std":>8} {"rt_err":>10} '
          f'{"scale_r":>8} {"COLA":>5} {"oracle_disc":>12}')
    for d in results:
        print(f'{d["n_fft"]:>6d} {d["hop_length"]:>4d} '
              f'{d["image_pixel_std"]:>8.4f} '
              f'{d["roundtrip_rel_err"]:>10.2e} '
              f'{d["scale_ratio"]:>8.1e} '
              f'{str(d["cola_perfect_hann"]):>5} '
              f'{d["oracle_disc_mean"]:>12.4f}')

    if results:
        winner = results[0]
        print('\n[oracle-sweep] WINNER: '
              f'n_fft={winner["n_fft"]} hop={winner["hop_length"]} '
              f'oracle_disc={winner["oracle_disc_mean"]:.4f}')
        if len(results) >= 2:
            runner_up = results[1]
            print('[oracle-sweep] RUNNER-UP (for V3): '
                  f'n_fft={runner_up["n_fft"]} hop={runner_up["hop_length"]} '
                  f'oracle_disc={runner_up["oracle_disc_mean"]:.4f}')

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n[oracle-sweep] saved full result to {args.output}')


if __name__ == '__main__':
    main()
