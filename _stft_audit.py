"""Local audit of STFTEmbedder - checks linearity, roundtrip, and scale.

CG assumes G = A_ts ∘ img_to_ts is LINEAR (so that (Gx, y) = (x, G^T y)).
This test verifies that.
"""
import torch
from models.img_transformations import STFTEmbedder

device = 'cpu'
seq_len = 24
n_fft = 8
hop_length = 4
img_resolution = 8
F = 28
B = 4

torch.manual_seed(0)

emb = STFTEmbedder(device=device, seq_len=seq_len,
                   n_fft=n_fft, hop_length=hop_length,
                   img_resolution=img_resolution)

# Train data to set min/max
train = torch.randn(256, seq_len, F) * 0.5
emb.cache_min_max_params(train)

x1 = torch.randn(B, seq_len, F) * 0.5
x2 = torch.randn(B, seq_len, F) * 0.5
zero_ts = torch.zeros(B, seq_len, F)

# ----- Test 1: ts_to_img(0) should equal 0 if linear -----
img0 = emb.ts_to_img(zero_ts)
print(f"[Test 1] ts_to_img(0):  max|.| = {img0.abs().max().item():.4e}")
print(f"          If > 0, ts_to_img is AFFINE not linear (bias term present).")

# ----- Test 2: img_to_ts(0) should equal 0 if linear -----
zero_img = torch.zeros(B, 2*F, img_resolution, img_resolution)
ts0 = emb.img_to_ts(zero_img)
print(f"[Test 2] img_to_ts(0):  max|.| = {ts0.abs().max().item():.4e}")
print(f"          If > 0, img_to_ts is AFFINE not linear.")

# ----- Test 3: Linearity ts_to_img(a*x1 + b*x2) =?= a*ts_to_img(x1) + b*ts_to_img(x2) -----
a, b = 0.7, 0.3
lhs = emb.ts_to_img(a * x1 + b * x2)
rhs = a * emb.ts_to_img(x1) + b * emb.ts_to_img(x2)
err = (lhs - rhs).abs().max().item()
print(f"[Test 3] Linearity error ts_to_img: {err:.4e}")
print(f"          If > 1e-4, ts_to_img is NOT linear -> BREAKS CG")

# ----- Test 4: Roundtrip ts_to_img -> img_to_ts -----
x_img = emb.ts_to_img(x1)
x_rt = emb.img_to_ts(x_img)
rt_err = (x1 - x_rt).abs().max().item()
rt_rel = rt_err / x1.abs().max().item()
print(f"[Test 4] Roundtrip err max|ts - img_to_ts(ts_to_img(ts))| = {rt_err:.4e}")
print(f"          Relative: {rt_rel:.4e}")

# ----- Test 5: Image scale after normalization -----
print(f"[Test 5] ts_to_img(x1) range: [{x_img.min().item():.3f}, {x_img.max().item():.3f}]  "
      f"mean={x_img.mean().item():.3f}  std={x_img.std().item():.3f}")
print(f"          Padded region fraction: {(x_img == 0).float().mean().item():.3f}")

# ----- Test 6: Adjoint check (Gx, y) == (x, G^T y) -----
# G = img_to_ts, G^T = ts_to_img
# For a LINEAR G with adjoint G^T, <Gx, y>_ts == <x, G^T y>_img must hold.
x_img_rand = torch.randn(B, 2*F, img_resolution, img_resolution)
y_ts_rand = torch.randn(B, seq_len, F)
lhs_inner = (emb.img_to_ts(x_img_rand) * y_ts_rand).sum().item()
rhs_inner = (x_img_rand * emb.ts_to_img(y_ts_rand)).sum().item()
print(f"[Test 6] Adjoint inner products:  <Gx,y> = {lhs_inner:.4e}")
print(f"                                   <x,G^T y> = {rhs_inner:.4e}")
print(f"          Ratio = {lhs_inner/(rhs_inner+1e-12):.4f}  "
      f"(should be 1.0 if ts_to_img/img_to_ts are proper adjoint pair)")
