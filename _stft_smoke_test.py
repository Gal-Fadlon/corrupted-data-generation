"""End-to-end STFT smoke test.

Builds a tiny TS2img_Karras + DualSpaceMMPS pipeline with the STFT
embedder and verifies:
  1. Bootstrap M-step loss is in a normal range (~O(1)), NOT billions.
  2. One full MMPS sampling pass (obs-space CG) produces bounded output.
  3. One E-step-like round-trip (ts -> img -> MMPS -> img -> ts)
     has NO NaN/Inf and stays in reasonable range.
"""
import sys
import types
import torch
import numpy as np

sys.path.insert(0, '.')

from models.our import TS2img_Karras


class _A:
    """Minimal args proxy mirroring run_co_evolving_em.py defaults."""
    input_channels = 4
    img_resolution = 8
    seq_len = 24
    delay = 3
    embedding = 8
    batch_size = 4
    num_workers = 0
    ch_mult = [1, 2, 2, 4]
    unet_channels = 32
    attn_resolution = [8, 4, 2]
    diffusion_steps = 8
    ema = False
    ema_warmup = 100
    embedder = 'stft'
    stft_n_fft = 8
    stft_hop_length = 4
    ambient_concat_further_mask = False


args = _A()
device = 'cpu'
args.device = device

torch.manual_seed(0)
np.random.seed(0)

B, T, F = 8, args.seq_len, args.input_channels

# Synthetic clean data in [0, 1]
clean = 0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, T))[None, :, None] \
      + 0.1 * np.random.randn(B, T, F)
clean = np.clip(clean, 0, 1).astype(np.float32)

# Random mask (50% observed)
mask_np = np.random.rand(B, T) > 0.5

model = TS2img_Karras(args=args, device=device).to(device)

print(f"[SETUP] model num_features={model.num_features}  embedder={model.embedder_kind}")
print(f"[SETUP] EDMPrecond.img_channels = expected {2*args.input_channels} -> actual {model.net.img_channels}")

# Cache STFT stats
stats_tensor = torch.as_tensor(clean, dtype=torch.float32)
model.cache_embedder_stats(stats_tensor)
print(f"[SETUP] scale_real shape={tuple(model.ts_img.scale_real.shape)}  "
      f"scale_imag shape={tuple(model.ts_img.scale_imag.shape)}")
print(f"[SETUP] scale_real range=[{model.ts_img.scale_real.min().item():.3e}, "
      f"{model.ts_img.scale_real.max().item():.3e}]")

# --- Test 1: bootstrap M-step one forward pass ---
x_clean = torch.tensor(clean, dtype=torch.float32)
x_img = model.ts_to_img(x_clean)
print(f"\n[TEST 1] x_img shape={tuple(x_img.shape)}  "
      f"range=[{x_img.min().item():.3f}, {x_img.max().item():.3f}]  "
      f"std={x_img.std().item():.3f}  NaN={int(torch.isnan(x_img).sum())}")

loss, _ = model.loss_fn_irregular(x_img)
print(f"[TEST 1] Bootstrap loss = {loss.item():.4f}  (normal: ~0.5-1.5; bug: ~1e8+)")
assert torch.isfinite(loss), "Bootstrap loss is NaN/Inf!"
assert loss.item() < 100, f"Bootstrap loss {loss.item()} is too large!"

# --- Test 2: MMPS obs-space CG sampling with fresh (random) UNet ---
# Since the model is untrained, the denoiser is random. CG may not produce
# great imputations, but MUST stay finite / bounded.
from run_co_evolving_em import DualSpaceMMPS, off_manifold_energy_batch

target_shape = (model.num_features, args.img_resolution, args.img_resolution)
process = DualSpaceMMPS(
    args, model.net, target_shape,
    sigma_y_ratio=0.1,
    cg_iters=3,
    ts_to_img_fn=model.ts_to_img,
    img_to_ts_fn=model.img_to_ts,
    use_adaptive_sigma_y=True,
    use_consistency_projection=True,
    use_obs_space_cg=True,
    use_warm_start_cg=True,
    sigma_y_floor=0.01,
)

obs_ts = torch.tensor(clean, dtype=torch.float32)
mask_ts = torch.tensor(mask_np.astype(np.float32), dtype=torch.float32)
mask_ts_exp = mask_ts.unsqueeze(-1).expand(-1, -1, F)
# Zero unobserved
obs_ts = obs_ts * mask_ts_exp

x_obs_img = model.ts_to_img(obs_ts)
mask_img = model.ts_to_img(mask_ts_exp)[:, :1]

print(f"\n[TEST 2] Running MMPS obs-space CG...")
out_img = process.sampling_mmps(x_obs_img, mask_img, obs_ts=obs_ts, mask_ts=mask_ts_exp)
print(f"[TEST 2] output image range=[{out_img.min().item():.3f}, {out_img.max().item():.3f}]  "
      f"std={out_img.std().item():.3f}")
print(f"[TEST 2] NaN={int(torch.isnan(out_img).sum())}  "
      f"Inf={int(torch.isinf(out_img).sum())}  "
      f"|.|max={out_img.abs().max().item():.3e}")
assert torch.isfinite(out_img).all(), "MMPS output has NaN/Inf"
assert out_img.abs().max().item() < 1e5, f"MMPS output blew up: {out_img.abs().max().item():.3e}"

# --- Test 3: off-manifold energy (should be small after consistency projection) ---
e_off = off_manifold_energy_batch(out_img, model.img_to_ts, model.ts_to_img)
print(f"\n[TEST 3] E_off after projection = {e_off:.6e}  (should be ~0 after consistency proj)")
assert e_off < 1e-2, f"E_off = {e_off:.3e} is too large even with proj"

# --- Test 4: image-space CG path (regime b) ---
process_b = DualSpaceMMPS(
    args, model.net, target_shape,
    sigma_y_ratio=0.1,
    cg_iters=3,
    ts_to_img_fn=model.ts_to_img,
    img_to_ts_fn=model.img_to_ts,
    use_adaptive_sigma_y=True,
    use_consistency_projection=True,
    use_obs_space_cg=False,  # img-space CG
    use_warm_start_cg=True,
    sigma_y_floor=0.01,
)
print(f"\n[TEST 4] Running MMPS image-space CG (regime b)...")
out_img_b = process_b.sampling_mmps(x_obs_img, mask_img, obs_ts=obs_ts, mask_ts=mask_ts_exp)
print(f"[TEST 4] output image range=[{out_img_b.min().item():.3f}, {out_img_b.max().item():.3f}]  "
      f"NaN={int(torch.isnan(out_img_b).sum())}  "
      f"|.|max={out_img_b.abs().max().item():.3e}")
assert torch.isfinite(out_img_b).all(), "img-space CG output has NaN/Inf"

# --- Test 5: vanilla MMPS path (regime c) with tight sigma_y ---
process_c = DualSpaceMMPS(
    args, model.net, target_shape,
    sigma_y_ratio=0.01,
    cg_iters=3,
    ts_to_img_fn=model.ts_to_img,
    img_to_ts_fn=model.img_to_ts,
    use_adaptive_sigma_y=False,   # fixed sigma_y
    use_consistency_projection=False,
    use_obs_space_cg=False,
    use_warm_start_cg=False,
    sigma_y_floor=0.0,
)
print(f"\n[TEST 5] Running vanilla MMPS (regime c) with tight sigma_y=0.01...")
out_img_c = process_c.sampling_mmps(x_obs_img, mask_img, obs_ts=obs_ts, mask_ts=mask_ts_exp)
print(f"[TEST 5] output image range=[{out_img_c.min().item():.3f}, {out_img_c.max().item():.3f}]  "
      f"NaN={int(torch.isnan(out_img_c).sum())}  "
      f"|.|max={out_img_c.abs().max().item():.3e}")
assert torch.isfinite(out_img_c).all(), "vanilla MMPS output has NaN/Inf"

print("\n" + "="*60)
print("ALL SMOKE TESTS PASSED. STFT pipeline is numerically sound.")
print("="*60)
