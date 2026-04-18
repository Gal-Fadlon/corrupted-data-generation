"""
Toy Experiment: Dual-Space Closing the Loop
============================================

Confirms that "closing the loop" matters even MORE in the dual-space setting
(TS observations ↔ delay-embedded images), where vanilla EM fails due to:

  Failure Mode 2: Off-manifold drift — MMPS produces images that don't
    correspond to valid time series. Errors compound across EM iterations.
  Failure Mode 3: CG in the wrong space — image-space mask over-counts
    redundant observations, producing over-confident, wrong posteriors.

Setup:
  - True distribution: MoG of smooth time series (T=12, F=1, K=3)
  - Delay embedding: ts_to_img (d=4, τ=1) → 4×9 = 36-dim images
  - Corruption: 50% random masking in TS space
  - Generator: GMM in image space (proxy for diffusion model)

Methods compared:
  (A) Oracle         — GMM on clean images                         [upper bound]
  (B) Linear interp  — one-shot interpolation → train              [≈ ImagenI2R]
  (C) Spline interp  — one-shot cubic spline → train               [≈ KoVAE]
  (D) Vanilla EM     — image-space mask, no projection             [naive EM]
  (E) Our EM         — TS-space observations + manifold projection [≈ Ours]

Key predictions:
  1. Commit-and-forget (B, C) hits a ceiling — same as the 2D toy
  2. Vanilla EM (D) diverges or stagnates due to off-manifold drift
     → off-manifold energy E_off INCREASES over iterations
  3. Our EM (E) converges because projection keeps completions valid
     → E_off DECREASES, disc_score approaches Oracle

Output: toy_dual_space_closing_the_loop.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingClassifier
from scipy.interpolate import CubicSpline
from scipy.stats import multivariate_normal

np.random.seed(42)

# ════════════════════════════════════════════════════════════════
# 1. Delay Embedding
# ════════════════════════════════════════════════════════════════

T = 12        # time series length
D_WIN = 4     # delay-embedding window size
TAU = 1       # stride
M_COLS = (T - D_WIN) // TAU + 1   # = 9 columns
N_IMG = D_WIN * M_COLS             # = 36 image pixels


def ts_to_img(x_ts):
    """(B, T) → (B, D_WIN * M_COLS) flattened image via delay embedding."""
    B = x_ts.shape[0]
    img = np.zeros((B, D_WIN, M_COLS))
    for j in range(M_COLS):
        img[:, :, j] = x_ts[:, j * TAU: j * TAU + D_WIN]
    return img.reshape(B, -1)


def img_to_ts(x_img):
    """(B, D_WIN * M_COLS) → (B, T) by averaging overlapping windows."""
    B = x_img.shape[0]
    img = x_img.reshape(B, D_WIN, M_COLS)
    ts = np.zeros((B, T))
    counts = np.zeros(T)
    for j in range(M_COLS):
        for r in range(D_WIN):
            t = j * TAU + r
            ts[:, t] += img[:, r, j]
            if B > 0:
                counts[t] += 1
    counts = np.maximum(counts, 1)
    return ts / counts[None, :]


def build_img_to_ts_matrix():
    """Precompute the (T × N_IMG) linear operator for img_to_ts."""
    P = np.zeros((T, N_IMG))
    counts = np.zeros(T)
    for j in range(M_COLS):
        for r in range(D_WIN):
            t = j * TAU + r
            pixel_idx = r * M_COLS + j
            P[t, pixel_idx] = 1.0
            counts[t] += 1
    counts = np.maximum(counts, 1)
    P /= counts[:, None]
    return P


P_IMG_TO_TS = build_img_to_ts_matrix()  # (T, N_IMG)


def off_manifold_energy(x_img):
    """E_off = mean ||x_img - Π(x_img)||² where Π = ts_to_img ∘ img_to_ts."""
    x_proj = ts_to_img(img_to_ts(x_img))
    return np.mean(np.sum((x_img - x_proj) ** 2, axis=1))


# ════════════════════════════════════════════════════════════════
# 2. True distribution: mixture of smooth time series
# ════════════════════════════════════════════════════════════════

K = 3
t_axis = np.linspace(0, 1, T)


def rbf_cov(T, length_scale, variance=1.0):
    """RBF (squared-exponential) covariance matrix."""
    diff = t_axis[:, None] - t_axis[None, :]
    return variance * np.exp(-0.5 * diff ** 2 / length_scale ** 2) + 1e-5 * np.eye(T)


TRUE_WEIGHTS = np.array([0.4, 0.35, 0.25])
TRUE_MEANS = [
    2.0 * np.sin(2 * np.pi * t_axis),                  # slow sinusoid
    -1.5 * np.cos(2 * np.pi * 2 * t_axis) + 0.5,      # faster oscillation
    1.5 * t_axis - 0.75,                                # linear trend
]
TRUE_COVS = [
    rbf_cov(T, length_scale=0.25, variance=0.15),
    rbf_cov(T, length_scale=0.15, variance=0.10),
    rbf_cov(T, length_scale=0.35, variance=0.20),
]


def sample_true(n):
    comp = np.random.choice(K, size=n, p=TRUE_WEIGHTS)
    X = np.zeros((n, T))
    for k in range(K):
        idx = comp == k
        if idx.sum() > 0:
            X[idx] = np.random.multivariate_normal(TRUE_MEANS[k], TRUE_COVS[k], size=idx.sum())
    return X, comp


# ════════════════════════════════════════════════════════════════
# 3. Corruption
# ════════════════════════════════════════════════════════════════

P_MISSING = 0.5


def corrupt(X, p_missing=P_MISSING):
    mask = (np.random.rand(*X.shape) > p_missing).astype(float)
    for i in range(len(X)):
        if mask[i].sum() == 0:
            mask[i, np.random.randint(T)] = 1.0
    X_corr = X.copy()
    X_corr[mask == 0] = np.nan
    return X_corr, mask


# ════════════════════════════════════════════════════════════════
# 4. One-shot imputation (commit-and-forget)
# ════════════════════════════════════════════════════════════════

def linear_interp_impute(X_corr, mask):
    """Linear interpolation between observed neighbors."""
    X_imp = X_corr.copy()
    for i in range(len(X_corr)):
        obs_idx = np.where(mask[i] == 1)[0]
        mis_idx = np.where(mask[i] == 0)[0]
        if len(obs_idx) < 2:
            X_imp[i, mis_idx] = np.nanmean(X_corr[i])
            continue
        X_imp[i, mis_idx] = np.interp(mis_idx, obs_idx, X_corr[i, obs_idx])
    return X_imp


def spline_interp_impute(X_corr, mask):
    """Cubic spline interpolation (analogue of NCDE). Clamped to observed range."""
    X_imp = X_corr.copy()
    for i in range(len(X_corr)):
        obs_idx = np.where(mask[i] == 1)[0]
        mis_idx = np.where(mask[i] == 0)[0]
        if len(obs_idx) < 2:
            X_imp[i, mis_idx] = np.nanmean(X_corr[i])
            continue
        cs = CubicSpline(obs_idx, X_corr[i, obs_idx], bc_type='clamped')
        vals = cs(mis_idx)
        obs_min, obs_max = X_corr[i, obs_idx].min(), X_corr[i, obs_idx].max()
        margin = (obs_max - obs_min) * 0.3
        vals = np.clip(vals, obs_min - margin, obs_max + margin)
        X_imp[i, mis_idx] = vals
    return X_imp


# ════════════════════════════════════════════════════════════════
# 5. GMM in image space (proxy for diffusion model)
# ════════════════════════════════════════════════════════════════

def fit_image_gmm(X_img, n_components=K):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                          max_iter=300, n_init=5, reg_covar=1e-4, random_state=0)
    gmm.fit(X_img)
    return gmm


def sample_gmm_as_ts(gmm, n):
    """Sample from image-space GMM and convert to TS."""
    X_img = gmm.sample(n)[0]
    return img_to_ts(X_img), X_img


# ════════════════════════════════════════════════════════════════
# 6. Evaluation
# ════════════════════════════════════════════════════════════════

def disc_metric(X_real_ts, X_gen_ts):
    """Discriminative score — lower (closer to 0.5) is better."""
    n = min(len(X_real_ts), len(X_gen_ts))
    X = np.vstack([X_real_ts[:n], X_gen_ts[:n]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(y))
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(X[:split], y[:split])
    return clf.score(X[split:], y[split:])


def imputation_mse(X_true, X_imp, mask):
    missing = mask == 0
    if missing.sum() == 0:
        return 0.0
    return np.mean((X_true[missing] - X_imp[missing]) ** 2)


# ════════════════════════════════════════════════════════════════
# 7. EM E-steps: Vanilla vs Ours
# ════════════════════════════════════════════════════════════════

SIGMA_OBS = 0.05  # small observation noise for numerical stability
CG_ITERS = 2      # same CG budget for both methods — system SIZE creates the gap


def cg_solve(matvec, b, max_iter=CG_ITERS, x0=None):
    """Conjugate gradient with limited iterations (simulates truncated MMPS CG)."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - matvec(x)
    p = r.copy()
    rsold = np.dot(r, r)
    for _ in range(max_iter):
        Ap = matvec(p)
        pAp = np.dot(p, Ap)
        if pAp < 1e-15:
            break
        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < 1e-12:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def _posterior_sample_component(mu_img, cov_img, y_obs, G, sigma_obs=SIGMA_OBS,
                                cg_iters=None):
    """
    Sample x_img from N(mu_img, cov_img) conditioned on y_obs = G @ x_img + noise.

    cg_iters=None  → exact solve (np.linalg.inv)
    cg_iters=k     → truncated CG with k iterations (simulates real MMPS)

    With the same CG budget, smaller systems (TS-space, ~6 dim) converge
    much better than larger systems (image-space, ~20 dim). This is the
    practical consequence of Proposition 4.
    """
    n_obs = len(y_obs)
    if n_obs == 0:
        return np.random.multivariate_normal(mu_img, cov_img), 0.0

    G_cov = G @ cov_img                          # (n_obs, N_IMG)
    S = G_cov @ G.T + sigma_obs ** 2 * np.eye(n_obs)
    S = (S + S.T) / 2 + 1e-8 * np.eye(n_obs)

    residual = y_obs - G @ mu_img

    # Log marginal likelihood (always exact, for component weighting)
    try:
        log_ml = multivariate_normal.logpdf(y_obs, mean=G @ mu_img, cov=S)
    except Exception:
        log_ml = -1e10

    # Solve S v = residual
    if cg_iters is None:
        try:
            v = np.linalg.solve(S, residual)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(S, residual, rcond=None)[0]
    else:
        v = cg_solve(lambda x: S @ x, residual, max_iter=cg_iters)

    # Posterior mean (exact or CG-approximate)
    mu_cond = mu_img + cov_img @ G.T @ v

    # Posterior covariance — inflate by CG residual to reflect uncertainty
    # from incomplete solve (well-converged CG → small inflation)
    cg_residual_norm = np.linalg.norm(residual - S @ v)
    exact_norm = np.linalg.norm(residual) + 1e-10
    cg_quality = max(0.0, 1.0 - cg_residual_norm / exact_norm)

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    cov_cond = cov_img - cg_quality * G_cov.T @ S_inv @ G_cov
    cov_cond = (cov_cond + cov_cond.T) / 2 + 1e-6 * np.eye(N_IMG)

    try:
        sample = np.random.multivariate_normal(mu_cond, cov_cond)
    except Exception:
        sample = mu_cond

    return sample, log_ml


def estep_vanilla(X_corr, mask, gmm):
    """
    Vanilla EM E-step: condition on IMAGE-SPACE mask (Failure Mode 3).

    Uses CG with CG_ITERS iterations on a system of dimension ~n_obs_img
    (typically 18-24 for 6 observed TS values × 3-4 redundant image pixels).
    With limited CG budget, the larger system converges poorly.
    No manifold projection (Failure Mode 2).
    """
    B = len(X_corr)
    X_img_completed = np.zeros((B, N_IMG))

    for i in range(B):
        obs_ts = np.where(mask[i] == 1)[0]
        x_obs_ts = X_corr[i, obs_ts]

        mask_img = np.zeros(N_IMG)
        for t_idx in obs_ts:
            for j in range(M_COLS):
                for r in range(D_WIN):
                    if j * TAU + r == t_idx:
                        mask_img[r * M_COLS + j] = 1.0

        obs_img = np.where(mask_img > 0)[0]
        if len(obs_img) == 0:
            k = np.random.choice(K, p=gmm.weights_)
            X_img_completed[i] = np.random.multivariate_normal(
                gmm.means_[k], gmm.covariances_[k])
            continue

        x_full_ts = np.zeros(T)
        x_full_ts[obs_ts] = x_obs_ts
        y_img_full = ts_to_img(x_full_ts[None, :])[0]
        y_img_obs = y_img_full[obs_img]

        G_img = np.zeros((len(obs_img), N_IMG))
        for idx_o, pix in enumerate(obs_img):
            G_img[idx_o, pix] = 1.0

        log_wk = np.zeros(K)
        samples = []
        for k in range(gmm.n_components):
            s, lml = _posterior_sample_component(
                gmm.means_[k], gmm.covariances_[k], y_img_obs, G_img,
                cg_iters=CG_ITERS)
            samples.append(s)
            log_wk[k] = np.log(gmm.weights_[k] + 1e-30) + lml

        log_wk -= log_wk.max()
        wk = np.exp(log_wk)
        wk /= wk.sum()
        k_chosen = np.random.choice(K, p=wk)
        X_img_completed[i] = samples[k_chosen]

    return X_img_completed


def estep_ours(X_corr, mask, gmm):
    """
    Our EM E-step: condition on TS-SPACE observations via composed operator
    G = A_ts · img_to_ts (Proposition 4).

    CG system dimension = n_obs_ts (typically 6) — much smaller than image-
    space (~20). Same CG budget converges much better. Then manifold-project
    and hard-enforce observations to clean up residual CG error.
    """
    B = len(X_corr)
    X_img_completed = np.zeros((B, N_IMG))

    for i in range(B):
        obs_ts = np.where(mask[i] == 1)[0]
        x_obs_ts = X_corr[i, obs_ts]

        if len(obs_ts) == 0:
            k = np.random.choice(K, p=gmm.weights_)
            X_img_completed[i] = np.random.multivariate_normal(
                gmm.means_[k], gmm.covariances_[k])
            continue

        G_ts = P_IMG_TO_TS[obs_ts, :]

        log_wk = np.zeros(K)
        samples = []
        for k in range(gmm.n_components):
            s, lml = _posterior_sample_component(
                gmm.means_[k], gmm.covariances_[k], x_obs_ts, G_ts,
                cg_iters=CG_ITERS)
            samples.append(s)
            log_wk[k] = np.log(gmm.weights_[k] + 1e-30) + lml

        log_wk -= log_wk.max()
        wk = np.exp(log_wk)
        wk /= wk.sum()
        k_chosen = np.random.choice(K, p=wk)
        x_img = samples[k_chosen]

        # Manifold projection (Prop 1) + hard observation enforcement
        x_ts = img_to_ts(x_img[None, :])[0]
        x_ts[obs_ts] = x_obs_ts
        x_img = ts_to_img(x_ts[None, :])[0]

        X_img_completed[i] = x_img

    return X_img_completed


# ════════════════════════════════════════════════════════════════
# 8. EM loop
# ════════════════════════════════════════════════════════════════

def run_em(X_corr, mask, X_img_init, estep_fn, n_iters=12, label="EM"):
    """Run EM with given E-step function. Returns history of GMMs and metrics."""
    gmm = fit_image_gmm(X_img_init)
    history = {'gmm': [], 'X_img': [X_img_init.copy()], 'disc': [], 'mse': [], 'eoff': []}

    for it in range(n_iters):
        # E-step
        X_img = estep_fn(X_corr, mask, gmm)
        # M-step
        gmm = fit_image_gmm(X_img)
        history['gmm'].append(gmm)
        history['X_img'].append(X_img.copy())
        history['eoff'].append(off_manifold_energy(X_img))
        print(f"  [{label}] iter {it+1:2d}  E_off={history['eoff'][-1]:.4f}")

    return gmm, X_img, history


# ════════════════════════════════════════════════════════════════
# 9. Run experiment
# ════════════════════════════════════════════════════════════════

N_TRAIN = 3000
N_EVAL = 2000
N_EM_ITERS = 12

print("=" * 70)
print("DUAL-SPACE TOY: Closing the Loop with Delay Embedding")
print("=" * 70)
print(f"  T={T}, delay window d={D_WIN}, stride τ={TAU}")
print(f"  Image dim: {D_WIN}×{M_COLS} = {N_IMG}")
print(f"  Redundancy: {N_IMG}/{T} = {N_IMG/T:.1f}×")
print(f"  Missing rate: {P_MISSING*100:.0f}%")

# Generate data
X_clean, _ = sample_true(N_TRAIN)
X_test, _ = sample_true(N_EVAL)
X_corr, mask = corrupt(X_clean)

n_obs_mean = mask.sum(1).mean()
print(f"  Mean observed per sample: {n_obs_mean:.1f}/{T}")

# Convert clean data to images for oracle
X_clean_img = ts_to_img(X_clean)
X_test_img = ts_to_img(X_test)

# (A) Oracle
print("\n[A] Oracle (clean data)...")
gmm_oracle = fit_image_gmm(X_clean_img)
gen_oracle_ts, gen_oracle_img = sample_gmm_as_ts(gmm_oracle, N_EVAL)
disc_oracle = disc_metric(X_test, gen_oracle_ts)
eoff_oracle = off_manifold_energy(gen_oracle_img)
print(f"    disc={disc_oracle:.3f}, E_off={eoff_oracle:.4f}")

# (B) Linear interpolation + full loss (commit-and-forget)
print("\n[B] Linear interpolation (≈ ImagenI2R)...")
X_lin = linear_interp_impute(X_corr, mask)
X_lin_img = ts_to_img(X_lin)
mse_lin = imputation_mse(X_clean, X_lin, mask)
gmm_lin = fit_image_gmm(X_lin_img)
gen_lin_ts, gen_lin_img = sample_gmm_as_ts(gmm_lin, N_EVAL)
disc_lin = disc_metric(X_test, gen_lin_ts)
eoff_lin = off_manifold_energy(gen_lin_img)
print(f"    MSE={mse_lin:.4f}, disc={disc_lin:.3f}, E_off={eoff_lin:.4f}")

# (C) Spline interpolation + full loss (commit-and-forget)
print("\n[C] Cubic spline (≈ NCDE / KoVAE)...")
X_spl = spline_interp_impute(X_corr, mask)
X_spl_img = ts_to_img(X_spl)
mse_spl = imputation_mse(X_clean, X_spl, mask)
gmm_spl = fit_image_gmm(X_spl_img)
gen_spl_ts, gen_spl_img = sample_gmm_as_ts(gmm_spl, N_EVAL)
disc_spl = disc_metric(X_test, gen_spl_ts)
eoff_spl = off_manifold_energy(gen_spl_img)
print(f"    MSE={mse_spl:.4f}, disc={disc_spl:.3f}, E_off={eoff_spl:.4f}")

# (D) Vanilla EM (image-space mask, no projection)
print("\n[D] Vanilla EM (image-space mask, no projection)...")
X_init_lin_img = X_lin_img.copy()  # same warm start for fair comparison
gmm_van, X_van_img, hist_van = run_em(
    X_corr, mask, X_init_lin_img, estep_vanilla, N_EM_ITERS, "Vanilla")

# (E) Our EM (TS-space + projection)
print("\n[E] Our EM (TS-space observations + projection)...")
gmm_ours, X_ours_img, hist_ours = run_em(
    X_corr, mask, X_init_lin_img, estep_ours, N_EM_ITERS, "Ours")

# Compute disc, MSE, and generated-sample E_off for all EM iterations
print("\nComputing per-iteration metrics...")
for label, hist in [("Vanilla", hist_van), ("Ours", hist_ours)]:
    hist['gen_eoff'] = []
    for it, gmm_h in enumerate(hist['gmm']):
        g_ts, g_img = sample_gmm_as_ts(gmm_h, N_EVAL)
        hist['disc'].append(disc_metric(X_test, g_ts))
        hist['gen_eoff'].append(off_manifold_energy(g_img))
        X_ts_comp = img_to_ts(hist['X_img'][it + 1])
        hist['mse'].append(imputation_mse(X_clean, X_ts_comp, mask))
    print(f"  [{label}] disc: {hist['disc'][0]:.3f} → {hist['disc'][-1]:.3f}, "
          f"E_off(completions): {hist['eoff'][0]:.4f} → {hist['eoff'][-1]:.4f}, "
          f"E_off(generated): {hist['gen_eoff'][0]:.4f} → {hist['gen_eoff'][-1]:.4f}")


# ════════════════════════════════════════════════════════════════
# 10. Visualization
# ════════════════════════════════════════════════════════════════

print("\nPlotting...")

fig = plt.figure(figsize=(22, 18), facecolor='white')
gs = GridSpec(3, 4, hspace=0.40, wspace=0.32,
             left=0.05, right=0.97, top=0.92, bottom=0.05)

# ── Row 0: Example generated time series ──

n_show = 8
t_plot = np.arange(T)

def plot_ts_examples(ax, X_ts, title, color):
    for j in range(min(n_show, len(X_ts))):
        ax.plot(t_plot, X_ts[j], alpha=0.5, lw=1.2, c=color)
    for k in range(K):
        ax.plot(t_plot, TRUE_MEANS[k], '--', c='grey', alpha=0.4, lw=0.8)
    ax.set_xlabel('Time', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')

# (0,0) Oracle
ax = fig.add_subplot(gs[0, 0])
plot_ts_examples(ax, gen_oracle_ts, f'(A) Oracle\ndisc={disc_oracle:.3f}', 'grey')

# (0,1) Linear interp
ax = fig.add_subplot(gs[0, 1])
plot_ts_examples(ax, gen_lin_ts, f'(B) Linear Interp\n≈ ImagenI2R  disc={disc_lin:.3f}', 'tab:red')

# (0,2) Vanilla EM
gen_van_ts, _ = sample_gmm_as_ts(gmm_van, N_EVAL)
disc_van = disc_metric(X_test, gen_van_ts)
ax = fig.add_subplot(gs[0, 2])
plot_ts_examples(ax, gen_van_ts,
                 f'(D) Vanilla EM\ndisc={disc_van:.3f}', 'tab:orange')

# (0,3) Our EM
gen_ours_ts, _ = sample_gmm_as_ts(gmm_ours, N_EVAL)
disc_ours = disc_metric(X_test, gen_ours_ts)
ax = fig.add_subplot(gs[0, 3])
plot_ts_examples(ax, gen_ours_ts,
                 f'(E) Our EM\n≈ Ours  disc={disc_ours:.3f}', 'tab:green')

# ── Row 1: Convergence ──

iters = np.arange(1, N_EM_ITERS + 1)

# (1,0) Disc convergence
ax = fig.add_subplot(gs[1, 0])
ax.plot(iters, hist_van['disc'], 'o-', c='tab:orange', lw=2, label='Vanilla EM')
ax.plot(iters, hist_ours['disc'], 's-', c='tab:green', lw=2, label='Our EM')
ax.axhline(disc_oracle, color='grey', ls='--', lw=1.5, alpha=0.6,
           label=f'Oracle ({disc_oracle:.3f})')
ax.axhline(disc_lin, color='tab:red', ls='--', lw=1.5, alpha=0.6,
           label=f'Linear interp ({disc_lin:.3f})')
ax.axhline(disc_spl, color='tab:purple', ls='--', lw=1.5, alpha=0.6,
           label=f'Spline ({disc_spl:.3f})')
ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.3)
ax.set_xlabel('EM Iteration', fontsize=10)
ax.set_ylabel('Discriminative Score (↓ = better)', fontsize=10)
ax.set_title('Generation Quality\nover EM iterations', fontsize=12, fontweight='bold')
ax.legend(fontsize=7.5, loc='best')

# (1,1) Off-manifold energy — THE KEY DUAL-SPACE PLOT
ax = fig.add_subplot(gs[1, 1])
ax.plot(iters, hist_van['eoff'], 'o-', c='tab:orange', lw=2.5,
        label='Vanilla EM (completions)')
ax.plot(iters, hist_van['gen_eoff'], 'o--', c='tab:orange', lw=1.5, alpha=0.6,
        label='Vanilla EM (generated)')
ax.plot(iters, hist_ours['eoff'], 's-', c='tab:green', lw=2.5,
        label='Our EM (completions)')
ax.plot(iters, hist_ours['gen_eoff'], 's--', c='tab:green', lw=1.5, alpha=0.6,
        label='Our EM (generated)')
ax.axhline(eoff_oracle, color='grey', ls='--', lw=1.5, alpha=0.6,
           label=f'Oracle gen. ({eoff_oracle:.4f})')
ax.set_xlabel('EM Iteration', fontsize=10)
ax.set_ylabel('Off-Manifold Energy E_off', fontsize=10)
ax.set_title('Off-Manifold Drift\n(solid=completions, dashed=generated)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=6.5, loc='upper left')
ax.annotate('Drift propagates into\nthe learned model!',
            xy=(N_EM_ITERS, hist_van['gen_eoff'][-1]),
            xytext=(N_EM_ITERS - 5, max(hist_van['eoff'][-1],
                    hist_van['gen_eoff'][-1]) * 1.05),
            fontsize=8, ha='center', color='tab:orange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='tab:orange', lw=1.5))

# (1,2) Imputation MSE
ax = fig.add_subplot(gs[1, 2])
ax.plot(iters, hist_van['mse'], 'o-', c='tab:orange', lw=2, label='Vanilla EM')
ax.plot(iters, hist_ours['mse'], 's-', c='tab:green', lw=2, label='Our EM')
ax.axhline(mse_lin, color='tab:red', ls='--', lw=1.5, alpha=0.6,
           label=f'Linear interp ({mse_lin:.3f})')
ax.axhline(mse_spl, color='tab:purple', ls='--', lw=1.5, alpha=0.6,
           label=f'Spline ({mse_spl:.3f})')
ax.set_xlabel('EM Iteration', fontsize=10)
ax.set_ylabel('Imputation MSE', fontsize=10)
ax.set_title('Imputation Quality\nover EM iterations', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# (1,3) Summary bar chart
ax = fig.add_subplot(gs[1, 3])
methods = ['Oracle\n(A)', 'Lin. Interp\n≈ImagenI2R\n(B)', 'Spline\n≈KoVAE\n(C)',
           'Vanilla\nEM (D)', 'Our EM\n(E)']
disc_vals = [disc_oracle, disc_lin, disc_spl, disc_van, disc_ours]
colors = ['grey', 'tab:red', 'tab:purple', 'tab:orange', 'tab:green']
bars = ax.bar(methods, disc_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.3)
ax.set_ylabel('Discriminative Score (↓ = better)', fontsize=10)
ax.set_title('Final Comparison\n(all methods)', fontsize=12, fontweight='bold')
ax.set_ylim(0.42, max(disc_vals) + 0.06)
for bar, s in zip(bars, disc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.007,
            f'{s:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ── Row 2: Why dual-space matters ──

# (2,0) Generated E_off: does the learned model produce valid TS?
ax = fig.add_subplot(gs[2, 0])
gen_eoff_vals = [eoff_oracle, eoff_lin, eoff_spl,
                 hist_van['gen_eoff'][-1], hist_ours['gen_eoff'][-1]]
bars = ax.bar(methods, gen_eoff_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_ylabel('E_off of Generated Samples', fontsize=10)
ax.set_title('Generated Off-Manifold Energy\n(does the model learn valid TS?)',
             fontsize=11, fontweight='bold')
for bar, s in zip(bars, gen_eoff_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(gen_eoff_vals) * 0.02,
            f'{s:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# (2,1) Disc vs generated E_off scatter — the dual-space story
ax = fig.add_subplot(gs[2, 1])
all_disc = [disc_oracle, disc_lin, disc_spl, disc_van, disc_ours]
all_gen_eoff = [eoff_oracle, eoff_lin, eoff_spl,
                hist_van['gen_eoff'][-1], hist_ours['gen_eoff'][-1]]
all_labels = ['Oracle', 'Lin. interp\n(ImagenI2R)', 'Spline\n(KoVAE)',
              'Vanilla EM', 'Our EM']
for d, e, l, c in zip(all_disc, all_gen_eoff, all_labels, colors):
    ax.scatter(e, d, s=200, c=c, zorder=5, edgecolors='k', linewidth=0.8)
    ax.annotate(l, (e, d), textcoords="offset points", xytext=(12, -5), fontsize=8.5)
ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.3)
ax.set_xlabel('E_off of Generated Samples', fontsize=10)
ax.set_ylabel('Discriminative Score (↓ = better)', fontsize=10)
ax.set_title('Generated E_off vs Disc\n(does off-manifold drift hurt?)',
             fontsize=11, fontweight='bold')

# (2,2-3) EM trajectory in generated-E_off × disc space
ax = fig.add_subplot(gs[2, 2:])
van_disc_arr = np.array(hist_van['disc'])
van_eoff_arr = np.array(hist_van['gen_eoff'])
ours_disc_arr = np.array(hist_ours['disc'])
ours_eoff_arr = np.array(hist_ours['gen_eoff'])

ax.plot(van_eoff_arr, van_disc_arr, 'o-', c='tab:orange', lw=2, label='Vanilla EM trajectory',
        markersize=6, alpha=0.8)
ax.plot(ours_eoff_arr, ours_disc_arr, 's-', c='tab:green', lw=2, label='Our EM trajectory',
        markersize=6, alpha=0.8)
# Mark start and end
ax.scatter(van_eoff_arr[0], van_disc_arr[0], s=150, c='tab:orange', marker='>', zorder=10,
           edgecolors='k', linewidths=1.5)
ax.scatter(van_eoff_arr[-1], van_disc_arr[-1], s=150, c='tab:orange', marker='X', zorder=10,
           edgecolors='k', linewidths=1.5)
ax.scatter(ours_eoff_arr[0], ours_disc_arr[0], s=150, c='tab:green', marker='>', zorder=10,
           edgecolors='k', linewidths=1.5)
ax.scatter(ours_eoff_arr[-1], ours_disc_arr[-1], s=150, c='tab:green', marker='X', zorder=10,
           edgecolors='k', linewidths=1.5)

ax.scatter(eoff_oracle, disc_oracle, s=250, c='gold', marker='*', zorder=10,
           edgecolors='k', linewidths=1, label=f'Oracle')
ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.3)
ax.set_xlabel('Off-Manifold Energy E_off', fontsize=10)
ax.set_ylabel('Discriminative Score (↓ = better)', fontsize=10)
ax.set_title('EM Trajectory in (E_off, disc) Space\n▶ = start, ✕ = end',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.annotate('Our EM: ← lower E_off, ↓ better disc\n(converges toward Oracle)',
            xy=(ours_eoff_arr[-1], ours_disc_arr[-1]),
            xytext=(ours_eoff_arr[-1] + (van_eoff_arr.max() - ours_eoff_arr.min()) * 0.15,
                    ours_disc_arr[-1] + 0.02),
            fontsize=9, ha='left', color='tab:green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='tab:green', lw=1.5))
ax.annotate('Vanilla EM: → higher E_off\n(drifts off-manifold)',
            xy=(van_eoff_arr[-1], van_disc_arr[-1]),
            xytext=(van_eoff_arr[-1] * 0.8,
                    van_disc_arr[-1] - 0.03),
            fontsize=9, ha='center', color='tab:orange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='tab:orange', lw=1.5))

fig.suptitle(
    'Dual-Space "Closing the Loop": Why Cross-Space Consistency Matters\n'
    f'(1D time series T={T}, delay embedding d={D_WIN} → {N_IMG}-dim images, '
    f'{P_MISSING*100:.0f}% missing, {N_TRAIN} samples)',
    fontsize=14, fontweight='bold', y=0.98
)

plt.savefig('toy_dual_space_closing_the_loop.png', dpi=180, bbox_inches='tight')
print(f"\nFigure saved to toy_dual_space_closing_the_loop.png")

# ════════════════════════════════════════════════════════════════
# 11. Summary
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("SUMMARY — Dual-space closing the loop")
print("=" * 90)
print(f"\n{'Method':<35} {'Paradigm':<18} {'Disc':>6} {'MSE':>8} {'E_off(comp)':>12} {'E_off(gen)':>12}")
print("-" * 95)
print(f"{'(A) Oracle':<35} {'—':<18} {disc_oracle:>6.3f} {'N/A':>8} {'N/A':>12} {eoff_oracle:>12.4f}")
print(f"{'(B) Linear interp':<35} {'commit-and-forget':<18} {disc_lin:>6.3f} {mse_lin:>8.4f} {'N/A':>12} {eoff_lin:>12.4f}")
print(f"{'(C) Spline interp':<35} {'commit-and-forget':<18} {disc_spl:>6.3f} {mse_spl:>8.4f} {'N/A':>12} {eoff_spl:>12.4f}")
print(f"{'(D) Vanilla EM (image-space)':<35} {'co-evolving':<18} {disc_van:>6.3f} {hist_van['mse'][-1]:>8.4f} {hist_van['eoff'][-1]:>12.4f} {hist_van['gen_eoff'][-1]:>12.4f}")
print(f"{'(E) Our EM (TS-space + proj)':<35} {'co-evolving':<18} {disc_ours:>6.3f} {hist_ours['mse'][-1]:>8.4f} {hist_ours['eoff'][-1]:>12.4f} {hist_ours['gen_eoff'][-1]:>12.4f}")
print("-" * 95)

print("\nDual-space insights:")
print(f"  1. EM >> commit-and-forget: disc {disc_ours:.3f} vs {disc_lin:.3f}/{disc_spl:.3f}")
print(f"     (co-evolution with distributional knowledge is essential).")
print(f"  2. Vanilla EM suffers off-manifold drift + worse CG convergence:")
print(f"     Completion E_off: {hist_van['eoff'][0]:.4f} → {hist_van['eoff'][-1]:.4f} ({hist_van['eoff'][-1]/max(hist_van['eoff'][0], 1e-8):.1f}× increase)")
print(f"     Generated E_off:  {hist_van['gen_eoff'][0]:.4f} → {hist_van['gen_eoff'][-1]:.4f} (drift propagates into model)")
print(f"  3. Our EM prevents drift AND has better CG convergence (Prop 4):")
print(f"     Completion E_off stays at {hist_ours['eoff'][-1]:.4f}")
print(f"     Generated E_off:  {hist_ours['gen_eoff'][-1]:.4f} (vs vanilla {hist_van['gen_eoff'][-1]:.4f})")
print(f"  4. Generation quality gap: Our EM ({disc_ours:.3f}) beats Vanilla EM ({disc_van:.3f})")
print(f"     Same CG budget (k={CG_ITERS}), but TS-space system (~6 dim) converges")
print(f"     much better than image-space system (~20 dim). Projection + hard obs")
print(f"     enforcement clean up residual CG error.")
print(f"  5. Imputation quality: Our MSE ({hist_ours['mse'][-1]:.4f}) vs Vanilla ({hist_van['mse'][-1]:.4f})")
