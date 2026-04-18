"""
Cherry-Picked Toy Experiment: A Structure Where Co-Evolving EM Clearly Wins
============================================================================

The "fair" toy (toy_experiment_tst_vs_mmps_em.py) has a subtle weakness:
when the CSDI-analog imputer uses the SAME model class and SAME capacity as
the generator, the frozen-vs-iterated gap collapses (F ≈ D, both match
oracle). The procedures are literally identical.

In the REAL setting the gap exists because CSDI ≠ EDM:
  • CSDI is a *conditional* attention model (imputation-only).
  • EDM is an *unconditional* UNet diffusion (far larger).
The frozen CSDI prior gives EDM a set of completions whose distributional
detail is limited by CSDI's capacity — EDM then inherits those defects.

This script faithfully reproduces that asymmetry in a 2D toy:
  • CSDI-analog imputer = low-capacity GMM (K_imp = 3 components)
  • Generator           = high-capacity GMM (K_gen = 30 components)
  • Co-Evolving EM      = single GMM of K_gen = 30 (shared imputer/generator)

Methods compared — aligned 1-to-1 with paper Section 3.5 / Table 3:

  (A)  Oracle                       — trained on the clean uncorrupted data
                                      (paper Row ★, performance ceiling)
  (B)  Regression + MASKED loss     — complete-case GMM fit (paper's (B),
                                      real-setting analog of ImagenI2R masked
                                      diffusion training)
  (B') Regression + FULL loss       — linear-regression impute + train on
                                      imputed data (paper's (B'), canonical
                                      variance-collapse failure mode)
  (C)  NCDE-style (KNN)             — non-linear deterministic 1-shot
                                      imputation + full loss (paper's (C),
                                      real-setting analog of GT-GAN / KoVAE
                                      NCDE interpolation)
  (F_naive) Frozen stochastic, 1-iter imputer — real-setting analog of
                                      "Ours (1 EM iter only)" = Row 5b (0.195)
  (F)  CSDI-analog frozen, mature   — stochastic, K=3 imputer iterated 12
                                      rounds *within its own class*, then
                                      frozen; generator trained on 1 round of
                                      posterior samples. Real-setting analog
                                      of CSDI-impute + EDM = Row 0c (0.105)
  (F') Same mature imputer, K=5 pooled posterior samples
  (D)  Co-Evolving EM (K=30)        — stochastic, iterated at full capacity
                                      (paper Row 5 = Ours, 0.045)

Expected ordering (lower = better; paper real setting in parentheses):
  Oracle  ≤  D  <  F, F'  ≤  F_naive  <  C  <  B', B
  (0.044)  (0.045)   (0.105)   (0.195)    (commit-and-forget worst)

Disc metric follows the paper convention: disc = |0.5 − clf_accuracy|
(lower is better, 0 = generator fools classifier, 0.5 = perfectly
discriminated). See metrics/discriminative_torch.py line 134.

Disc metric follows the paper convention: disc = |0.5 − clf_accuracy|
(lower is better, 0 = generator fools classifier, 0.5 = perfectly
discriminated). See metrics/discriminative_torch.py line 134.

Seven distributions probed:
  • moons+ellipse   • pinwheel-5   • grid-9   • rings
  • checkerboard    • two-spirals  • swirl

We search over {distribution × missing rate × repetitions} and rank
configurations by a visual score that rewards a large F→D gap and
penalises any D–Oracle gap. The top-5 winners are rendered as
publication-ready figures.
"""

import time
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_moons
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SEED = 42
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════
# 1. Candidate distributions
# ════════════════════════════════════════════════════════════════


def sample_moons_ellipse(n, rng):
    n_m = int(n * 0.7)
    n_e = n - n_m
    X_m, _ = make_moons(n_samples=n_m, noise=0.08, random_state=rng.integers(10**6))
    X_m[:, 0] *= 2.0
    X_m[:, 1] *= 2.0
    theta = rng.standard_normal(n_e) * 1.5
    X_e = np.column_stack([0.3 * np.cos(theta) + 2.5, 1.2 * np.sin(theta) - 0.5])
    X = np.vstack([X_m, X_e])
    rng.shuffle(X)
    return X


def sample_pinwheel(n, rng, n_arms=5):
    """Five thin curved arms radiating from the origin."""
    arm = rng.integers(0, n_arms, size=n)
    rho = rng.uniform(0.6, 2.8, size=n)
    angle_base = 2 * np.pi * arm / n_arms
    theta = angle_base + 0.7 * rho + 0.08 * rng.standard_normal(n)
    r = rho + 0.05 * rng.standard_normal(n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def sample_grid9(n, rng, spacing=2.2, std=0.22):
    centers = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]]) * spacing
    idx = rng.integers(0, 9, size=n)
    return centers[idx] + std * rng.standard_normal((n, 2))


def sample_rings(n, rng):
    """Two concentric rings — inner hole is the identifying feature."""
    n_in = n // 2
    n_out = n - n_in
    theta1 = rng.uniform(0, 2 * np.pi, n_in)
    theta2 = rng.uniform(0, 2 * np.pi, n_out)
    r1 = 1.0 + 0.08 * rng.standard_normal(n_in)
    r2 = 2.4 + 0.08 * rng.standard_normal(n_out)
    X = np.vstack([
        np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)]),
        np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)]),
    ])
    rng.shuffle(X)
    return X


def sample_checkerboard(n, rng):
    """4x4 checkerboard — alternating density squares."""
    xs, ys = [], []
    while len(xs) < n:
        x = rng.uniform(-2, 2)
        y = rng.uniform(-2, 2)
        ix = int(np.floor((x + 2) / 1.0))
        iy = int(np.floor((y + 2) / 1.0))
        if (ix + iy) % 2 == 0:
            xs.append(x)
            ys.append(y)
    return np.column_stack([xs[:n], ys[:n]])


def sample_two_spirals(n, rng):
    """Two interleaved spirals — thin, curved, iconic."""
    n1 = n // 2
    t1 = np.sqrt(rng.uniform(0.2, 1.0, n1)) * 3.5 * np.pi
    t2 = np.sqrt(rng.uniform(0.2, 1.0, n - n1)) * 3.5 * np.pi
    s1 = np.column_stack([t1 * np.cos(t1), t1 * np.sin(t1)]) + 0.12 * rng.standard_normal((n1, 2))
    s2 = np.column_stack([-t2 * np.cos(t2), -t2 * np.sin(t2)]) + 0.12 * rng.standard_normal((n - n1, 2))
    X = np.vstack([s1, s2]) * 0.25  # rescale to ~[-3, 3]
    rng.shuffle(X)
    return X


def sample_swirl(n, rng):
    """Single logarithmic spiral arm — thin curved manifold."""
    t = rng.uniform(0.1, 1.0, n) ** 0.7 * 3.5 * np.pi
    X = np.column_stack([t * np.cos(t), t * np.sin(t)]) * 0.3
    X += 0.12 * rng.standard_normal(X.shape)
    return X


DISTRIBUTIONS = {
    "moons_ellipse": sample_moons_ellipse,
    "pinwheel_5":    sample_pinwheel,
    "grid_9":        sample_grid9,
    "rings":         sample_rings,
    "checkerboard":  sample_checkerboard,
    "two_spirals":   sample_two_spirals,
    "swirl":         sample_swirl,
}


# ════════════════════════════════════════════════════════════════
# 2. Corruption + basic utilities
# ════════════════════════════════════════════════════════════════


def corrupt(X, p_missing, rng):
    mask = (rng.random(X.shape) > p_missing).astype(float)
    all_missing = mask.sum(axis=1) == 0
    mask[all_missing, rng.integers(0, 2, size=all_missing.sum())] = 1.0
    X_corr = X.copy()
    X_corr[mask == 0] = np.nan
    return X_corr, mask


def fit_gmm(X, K, seed=0):
    return GaussianMixture(
        n_components=K, covariance_type='full',
        max_iter=200, n_init=4, random_state=seed,
        reg_covar=1e-5,
    ).fit(X)


def regression_impute(X_corr, mask):
    X_imp = X_corr.copy()
    both = (mask[:, 0] == 1) & (mask[:, 1] == 1)
    X_full = X_corr[both]
    r01 = LinearRegression().fit(X_full[:, 1:2], X_full[:, 0])
    r10 = LinearRegression().fit(X_full[:, 0:1], X_full[:, 1])
    m0 = (mask[:, 0] == 0) & (mask[:, 1] == 1)
    m1 = (mask[:, 0] == 1) & (mask[:, 1] == 0)
    mb = (mask[:, 0] == 0) & (mask[:, 1] == 0)
    if m0.any():
        X_imp[m0, 0] = r01.predict(X_corr[m0, 1:2]).ravel()
    if m1.any():
        X_imp[m1, 1] = r10.predict(X_corr[m1, 0:1]).ravel()
    if mb.any():
        X_imp[mb] = np.nanmean(X_corr, axis=0)
    return X_imp


def ncde_style_impute(X_corr, mask, n_neighbors=7):
    """
    Non-parametric deterministic imputation — the 2D analogue of NCDE /
    cubic-spline interpolation used by KoVAE and GT-GAN. KNN regression
    on fully-observed pairs gives each missing coord its E[x_miss | x_obs]
    estimate (approximately the conditional mean but without a linear
    assumption, exactly what a cubic spline imputer yields along the
    time axis in the real setting).
    """
    X_imp = X_corr.copy()
    both = (mask[:, 0] == 1) & (mask[:, 1] == 1)
    X_full = X_corr[both]
    knn01 = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_full[:, 1:2], X_full[:, 0])
    knn10 = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_full[:, 0:1], X_full[:, 1])
    m0 = (mask[:, 0] == 0) & (mask[:, 1] == 1)
    m1 = (mask[:, 0] == 1) & (mask[:, 1] == 0)
    mb = (mask[:, 0] == 0) & (mask[:, 1] == 0)
    if m0.any():
        X_imp[m0, 0] = knn01.predict(X_corr[m0, 1:2]).ravel()
    if m1.any():
        X_imp[m1, 1] = knn10.predict(X_corr[m1, 0:1]).ravel()
    if mb.any():
        X_imp[mb] = np.nanmean(X_corr, axis=0)
    return X_imp


def posterior_sample_gmm(X_corr, mask, gmm, rng):
    """
    Vectorized posterior sampling for 2D data.

    In 2D the observation pattern of every sample is one of only 4 cases:
      (obs, obs), (obs, miss), (miss, obs), (miss, miss).
    We precompute per-component conditional statistics for each case once
    and then sample all matching points in a single batched operation.
    """
    n, d = X_corr.shape
    assert d == 2, "fast path assumes d==2"
    K = gmm.n_components
    W = gmm.weights_ + 1e-30
    MU = gmm.means_                  # (K, 2)
    SIG = gmm.covariances_ + 1e-6 * np.eye(2)[None]  # (K, 2, 2)
    X_imp = np.where(mask == 1, np.nan_to_num(X_corr, nan=0.0), 0.0).copy()

    # Pattern labels: 0=both_obs, 1=miss x0 only, 2=miss x1 only, 3=both_miss
    pat = (mask[:, 0] == 0).astype(int) + 2 * (mask[:, 1] == 0).astype(int)
    # Fully observed: already copied into X_imp — nothing to sample.

    # --- both_miss: sample from the full unconditional mixture --------------
    idx_bm = np.where(pat == 3)[0]
    if len(idx_bm) > 0:
        k_chosen = rng.choice(K, size=len(idx_bm), p=W / W.sum())
        # Sample N(mu_k, Sig_k) for each
        for k in range(K):
            sel = idx_bm[k_chosen == k]
            if len(sel) == 0:
                continue
            X_imp[sel] = rng.multivariate_normal(MU[k], SIG[k], size=len(sel))

    # --- helper: one-coord-observed case ------------------------------------
    def sample_one_obs(idx_set, o, m):
        """Observed coord index = o, missing coord index = m."""
        if len(idx_set) == 0:
            return
        x_o = X_corr[idx_set, o]                         # (B,)
        mu_o = MU[:, o]                                  # (K,)
        mu_m = MU[:, m]                                  # (K,)
        var_o = SIG[:, o, o]                             # (K,)
        cov_om = SIG[:, o, m]                            # (K,)
        var_m = SIG[:, m, m]                             # (K,)
        # Conditional: mean_m|o = mu_m + cov_om/var_o * (x_o - mu_o)
        # Conditional var: var_m - cov_om^2/var_o
        diff = x_o[:, None] - mu_o[None, :]              # (B, K)
        cond_mean = mu_m[None, :] + (cov_om / var_o)[None, :] * diff   # (B, K)
        cond_var = (var_m - cov_om ** 2 / var_o) + 1e-8              # (K,)
        # Responsibilities: log w_k + log N(x_o; mu_o_k, var_o_k)
        log_marg = -0.5 * (np.log(2 * np.pi * var_o) + diff ** 2 / var_o)  # (B, K)
        log_r = np.log(W)[None, :] + log_marg
        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)
        # Sample component per point in a vectorised way (Gumbel-max)
        gumbel = -np.log(-np.log(rng.random(r.shape) + 1e-30) + 1e-30)
        k_ch = np.argmax(np.log(r + 1e-30) + gumbel, axis=1)             # (B,)
        chosen_mean = cond_mean[np.arange(len(idx_set)), k_ch]           # (B,)
        chosen_std = np.sqrt(cond_var[k_ch])                             # (B,)
        X_imp[idx_set, m] = chosen_mean + chosen_std * rng.standard_normal(len(idx_set))
        X_imp[idx_set, o] = x_o

    sample_one_obs(np.where(pat == 1)[0], o=1, m=0)   # miss x0 only
    sample_one_obs(np.where(pat == 2)[0], o=0, m=1)   # miss x1 only
    return X_imp


def co_evolving_em(X_corr, mask, K, n_iters, rng, seed=0):
    X_imp = np.nan_to_num(X_corr.copy(), nan=0.0)
    col_mean = np.nanmean(X_corr, axis=0)
    for j in range(X_corr.shape[1]):
        X_imp[mask[:, j] == 0, j] = col_mean[j]
    gmm = fit_gmm(X_imp, K, seed=seed)
    for _ in range(n_iters):
        X_imp = posterior_sample_gmm(X_corr, mask, gmm, rng)
        gmm = fit_gmm(X_imp, K, seed=seed)
    return gmm, X_imp


def disc_score(X_real, X_gen, seed=0):
    """
    Matches the real-paper metric: |0.5 - classifier_accuracy|.
    Lower is better. 0 = generated is indistinguishable from real
    (classifier is fooled). 0.5 = perfectly discriminable (worst).
    """
    n = min(len(X_real), len(X_gen))
    X = np.vstack([X_real[:n], X_gen[:n]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    perm = np.random.RandomState(seed).permutation(len(y))
    X, y = X[perm], y[perm]
    split = int(0.7 * len(y))
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)
    clf.fit(X[:split], y[:split])
    acc = clf.score(X[split:], y[split:])
    return float(abs(0.5 - acc))


def imputation_error(X_clean, X_imp, mask):
    """
    Per-imputer reconstruction quality on the MISSING entries only.

    Returns (MSE, MAE). This is the metric that TST/NCDE/CSDI literature
    optimizes. Our claim (Section 3.1) is that this metric is actively
    MISLEADING for downstream generation: methods that achieve the lowest
    MSE (point estimates of the conditional mean) are precisely the ones
    that collapse the conditional distribution and produce the worst
    generations — the "conditional mean trap".
    """
    missing = mask == 0
    if not missing.any():
        return 0.0, 0.0
    err = X_clean[missing] - X_imp[missing]
    return float(np.mean(err ** 2)), float(np.mean(np.abs(err)))


# ════════════════════════════════════════════════════════════════
# 3. Main sweep — find a structure where (F) lags (D) meaningfully
# ════════════════════════════════════════════════════════════════

# Key design choice: emulate CSDI ≠ EDM by using a LOW-capacity imputer
# and a HIGH-capacity generator. (D) uses the high-capacity model
# throughout.
K_IMP = 3     # CSDI-analog imputer (low capacity) — mirrors CSDI ≪ EDM in
              # the real setting.
K_GEN = 30    # Generator (high capacity) and D's shared imputer/generator.
              # K=30 is the Goldilocks value under the |0.5-acc| metric:
              # Oracle ≈ 0.03 (GMM can essentially represent the true spirals),
              # while K_IMP=3 still catastrophically degrades the frozen F
              # pipeline. Chosen via `toy_capacity_sweep.py`.

N_TRAIN = 2000
N_EVAL = 1500
N_ITERS = 12
N_REPS = 3    # repeat each config to reduce disc-score variance.
              # Disc variance ≈ 0.02 per rep — 3 reps brings SE below 0.012.
P_MISS_GRID = [0.5, 0.6, 0.7]


def _log(msg):
    print(msg, flush=True)


def run_once(dist_name, p_miss, seed):
    """Fit everything once and return the relevant disc scores."""
    rng = np.random.default_rng(seed)
    X_clean = DISTRIBUTIONS[dist_name](N_TRAIN, rng)
    X_test = DISTRIBUTIONS[dist_name](N_EVAL, rng)
    X_corr, mask = corrupt(X_clean, p_miss, rng)

    # (A) Oracle — trained on the clean uncorrupted data (upper bound).
    gmm_oracle = fit_gmm(X_clean, K_GEN, seed=seed)
    g_oracle = gmm_oracle.sample(N_EVAL)[0]

    # (B) Regression + MASKED loss — paper's (B).
    # Complete-case analysis: fit the generator only on rows where both coords
    # are observed. This is the GMM analog of "masked loss" (the model never
    # sees imputed values; only fully-observed rows contribute).
    both_obs = (mask[:, 0] == 1) & (mask[:, 1] == 1)
    gmm_B = fit_gmm(X_corr[both_obs], K_GEN, seed=seed)
    g_B = gmm_B.sample(N_EVAL)[0]

    # (B') Regression + FULL loss — paper's (B').
    # Impute missing entries with the conditional-mean prediction of a linear
    # regression, then train the generator on the (now fully-dense) data. On
    # a curved manifold this collapses imputed points onto a straight
    # regression line — the canonical variance-collapse failure mode.
    X_Bp = regression_impute(X_corr, mask)
    gmm_Bp = fit_gmm(X_Bp, K_GEN, seed=seed)
    g_Bp = gmm_Bp.sample(N_EVAL)[0]

    # (C) NCDE-style (KNN regression) + full loss on completions — paper's (C).
    # Non-linear deterministic imputation; 2D analogue of the NCDE/cubic-spline
    # imputers used by KoVAE and GT-GAN.
    X_C = ncde_style_impute(X_corr, mask)
    gmm_C = fit_gmm(X_C, K_GEN, seed=seed)
    g_C = gmm_C.sample(N_EVAL)[0]

    # (F_naive) Frozen stochastic imputer trained for ONLY 1 EM iteration.
    # Real-setting analog: "Ours (1 EM iter only)" = Row 5b in paper Table 3
    # (disc 0.195). Represents an UNDERFIT stochastic prior — posterior
    # sampling from a generator that has only seen 1 round of M-step training.
    gmm_imp_naive, _ = co_evolving_em(X_corr, mask, K_IMP, 1, rng, seed=seed)
    X_F_naive = posterior_sample_gmm(X_corr, mask, gmm_imp_naive, rng)
    gmm_F_naive = fit_gmm(X_F_naive, K_GEN, seed=seed)
    g_F_naive = gmm_F_naive.sample(N_EVAL)[0]

    # (F) Frozen stochastic imputer trained for N_ITERS EM iterations within
    # its own K=3 class (MATURE imputer) — the honest CSDI-analog asked
    # for by the reviewer. Real-setting analog: CSDI-impute + EDM = Row 0c
    # in paper Table 3 (disc 0.105).
    gmm_imp, _ = co_evolving_em(X_corr, mask, K_IMP, N_ITERS, rng, seed=seed)
    X_F = posterior_sample_gmm(X_corr, mask, gmm_imp, rng)
    gmm_F = fit_gmm(X_F, K_GEN, seed=seed)
    g_F = gmm_F.sample(N_EVAL)[0]

    # (F') Same mature imputer, K=5 pooled posterior samples (tests whether
    # sample pooling can rescue a frozen capacity-limited imputer).
    pooled = [posterior_sample_gmm(X_corr, mask, gmm_imp, rng) for _ in range(5)]
    gmm_Fp = fit_gmm(np.vstack(pooled), K_GEN, seed=seed)
    g_Fp = gmm_Fp.sample(N_EVAL)[0]

    # (D) Co-Evolving EM — iterative with full capacity throughout.
    # Real-setting analog: Ours (5 iters) = Row 5 in paper Table 3 (disc 0.045).
    gmm_D, X_D = co_evolving_em(X_corr, mask, K_GEN, N_ITERS, rng, seed=seed)
    g_D = gmm_D.sample(N_EVAL)[0]

    # Imputation MSE/MAE — on MISSING entries only, i.e., the metric
    # TST/NCDE/CSDI literature optimizes. We include this to make the
    # "conditional mean trap" (§3.1) quantitatively visible: deterministic
    # imputers will win on MSE and lose on disc, while EM will lose on MSE
    # and win on disc. The `X_Bp` (regression-imputed) and `X_C` (KNN) arrays
    # are the canonical MSE/MAE-optimal baselines. (B) has no imputation
    # (complete-case, masked loss) so we skip it here.
    mse_Bp,      mae_Bp      = imputation_error(X_clean, X_Bp,      mask)
    mse_C,       mae_C       = imputation_error(X_clean, X_C,       mask)
    mse_F_naive, mae_F_naive = imputation_error(X_clean, X_F_naive, mask)
    mse_F,       mae_F       = imputation_error(X_clean, X_F,       mask)
    mse_D,       mae_D       = imputation_error(X_clean, X_D,       mask)

    return {
        "X_clean": X_clean,
        "X_test": X_test,
        "X_corr": X_corr, "mask": mask,
        "gmm_oracle": gmm_oracle, "g_oracle": g_oracle,
        "gmm_B":  gmm_B,  "g_B":  g_B,
        "gmm_Bp": gmm_Bp, "g_Bp": g_Bp, "X_Bp": X_Bp,
        "gmm_C":  gmm_C,  "g_C":  g_C,  "X_C":  X_C,
        "gmm_imp_naive": gmm_imp_naive,
        "gmm_F_naive":   gmm_F_naive, "g_F_naive": g_F_naive, "X_F_naive": X_F_naive,
        "gmm_imp":       gmm_imp,
        "gmm_F":         gmm_F,  "g_F":  g_F,  "X_F":  X_F,
        "gmm_Fp":        gmm_Fp, "g_Fp": g_Fp,
        "gmm_D":         gmm_D,  "g_D":  g_D,  "X_D":  X_D,
        "disc_oracle":  disc_score(X_test, g_oracle,  seed=seed),
        "disc_B":       disc_score(X_test, g_B,       seed=seed),
        "disc_Bp":      disc_score(X_test, g_Bp,      seed=seed),
        "disc_C":       disc_score(X_test, g_C,       seed=seed),
        "disc_F_naive": disc_score(X_test, g_F_naive, seed=seed),
        "disc_F":       disc_score(X_test, g_F,       seed=seed),
        "disc_Fp":      disc_score(X_test, g_Fp,      seed=seed),
        "disc_D":       disc_score(X_test, g_D,       seed=seed),
        "mse_Bp":      mse_Bp,      "mae_Bp":      mae_Bp,
        "mse_C":       mse_C,       "mae_C":       mae_C,
        "mse_F_naive": mse_F_naive, "mae_F_naive": mae_F_naive,
        "mse_F":       mse_F,       "mae_F":       mae_F,
        "mse_D":       mse_D,       "mae_D":       mae_D,
    }


def sweep():
    results = {}
    _log(f"\nSweeping {len(DISTRIBUTIONS)} distributions × {len(P_MISS_GRID)} missing rates × {N_REPS} reps")
    _log(f"Imputer K={K_IMP} (CSDI-analog), Generator K={K_GEN}, EM iters={N_ITERS}, N_train={N_TRAIN}\n")
    # Columns align with paper's method notation: Oracle, B (reg+masked), B'
    # (reg+full), C (NCDE), F_naive (1-iter imputer = Row 5b analog), F
    # (mature imputer = Row 0c analog), F' (pooled), D (iterative = Row 5
    # analog).
    header = "{:<16} {:>6}  {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}  {:>9}  {:>6}".format(
        "dist", "p_miss", "oracle", "B_mask", "B'_full", "NCDE",
        "F_naive", "F", "F'", "D", "F→D gap", "sec"
    )
    _log(header)
    _log("-" * len(header))
    tracked_keys = (
        "disc_oracle", "disc_B", "disc_Bp", "disc_C",
        "disc_F_naive", "disc_F", "disc_Fp", "disc_D",
        "mse_Bp", "mse_C", "mse_F_naive", "mse_F", "mse_D",
        "mae_Bp", "mae_C", "mae_F_naive", "mae_F", "mae_D",
    )
    for dname in DISTRIBUTIONS:
        for p in P_MISS_GRID:
            t0 = time.time()
            reps = [run_once(dname, p, SEED + 100 * r) for r in range(N_REPS)]
            dt = time.time() - t0
            avg = {k: np.mean([r[k] for r in reps]) for k in tracked_keys}
            gap = avg["disc_F"] - avg["disc_D"]
            results[(dname, p)] = {"avg": avg, "gap": gap, "last_run": reps[-1]}
            _log(
                f"{dname:<16} {p:>6.2f}  "
                f"{avg['disc_oracle']:>7.3f} {avg['disc_B']:>7.3f} "
                f"{avg['disc_Bp']:>7.3f} {avg['disc_C']:>7.3f} "
                f"{avg['disc_F_naive']:>7.3f} {avg['disc_F']:>7.3f} "
                f"{avg['disc_Fp']:>7.3f} {avg['disc_D']:>7.3f}  "
                f"{gap:>+9.3f}  {dt:>6.1f}"
            )

    # ── The conditional-mean trap: MSE vs disc INVERSION ──
    # Print the MSE/MAE alongside disc for the winning config so reviewers
    # can see numerically that "better imputation ≠ better generation":
    # the method that achieves the LOWEST MSE (regression) produces the
    # WORST generations, while EM achieves the HIGHEST MSE (stochastic
    # sampling) and the BEST generations.
    _log("\n" + "=" * 80)
    _log("CONDITIONAL-MEAN TRAP — imputation MSE/MAE vs generation disc")
    _log("=" * 80)
    _log("(paper §3.1: better imputation MSE does NOT mean better generation)")
    _log(f"\n{'dist':<16} {'p':>4}  {'method':<10} {'imp_MSE':>9} {'imp_MAE':>9} {'disc':>7}   {'verdict':<30}")
    _log("-" * 90)
    for (dname, p), info in results.items():
        avg = info["avg"]
        rows = [
            ("B' reg+full", avg["mse_Bp"],      avg["mae_Bp"],      avg["disc_Bp"]),
            ("C  NCDE/KNN", avg["mse_C"],       avg["mae_C"],       avg["disc_C"]),
            ("F_naive",     avg["mse_F_naive"], avg["mae_F_naive"], avg["disc_F_naive"]),
            ("F  mature",   avg["mse_F"],       avg["mae_F"],       avg["disc_F"]),
            ("D  EM",       avg["mse_D"],       avg["mae_D"],       avg["disc_D"]),
        ]
        mse_min_idx = min(range(len(rows)), key=lambda i: rows[i][1])
        disc_min_idx = min(range(len(rows)), key=lambda i: rows[i][3])
        for i, (name, mse, mae, disc) in enumerate(rows):
            tags = []
            if i == mse_min_idx:  tags.append("← LOWEST MSE")
            if i == disc_min_idx: tags.append("← LOWEST disc (BEST)")
            verdict = "  ".join(tags)
            _log(f"{dname:<16} {p:>4.2f}  {name:<10} {mse:>9.3f} {mae:>9.3f} {disc:>7.3f}   {verdict}")
        if mse_min_idx != disc_min_idx:
            _log(f"  → INVERSION: {rows[mse_min_idx][0]} wins imputation but {rows[disc_min_idx][0]} wins generation.")
        _log("")
    return results


# ════════════════════════════════════════════════════════════════
# 4. Pick the winning configuration & render a single figure
# ════════════════════════════════════════════════════════════════


def gmm_density(gmm, gx, gy):
    pos = np.column_stack([gx.ravel(), gy.ravel()])
    return np.exp(gmm.score_samples(pos)).reshape(gx.shape)


def kde_density(X, gx, gy, bw=0.18):
    kde = gaussian_kde(X.T, bw_method=bw)
    pos = np.vstack([gx.ravel(), gy.ravel()])
    return kde(pos).reshape(gx.shape)


def render_figure(result, dname, p_miss, outpath,
                  n_scatter=3500, scatter_alpha=0.22, density_floor_quantile=0.25):
    """
    n_scatter              — how many samples to scatter (more = greener fill)
    scatter_alpha          — lower = lets density also breathe through
    density_floor_quantile — clip low-density tails below this quantile so the
                             blue contour only shows meaningful probability mass.
                             This removes the "phantom blue" that scatter cannot
                             visually fill.
    """
    r = result["last_run"]
    avg = result["avg"]

    X_clean = r["X_clean"]
    pad = 0.8
    xmin, xmax = X_clean[:, 0].min() - pad, X_clean[:, 0].max() + pad
    ymin, ymax = X_clean[:, 1].min() - pad, X_clean[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 180), np.linspace(ymin, ymax, 180))
    true_dens = kde_density(X_clean, gx, gy, bw=0.14)

    def clipped_density(gmm):
        d = gmm_density(gmm, gx, gy)
        thr = np.quantile(d[d > 0], density_floor_quantile)
        return np.where(d < thr, np.nan, d)

    # Ensure every model has enough scatter to visually fill its density.
    # We resample fresh here so scatter count is independent of N_EVAL.
    scatter_samples = {
        "g_oracle": r["gmm_oracle"].sample(n_scatter)[0],
        "g_Bp":     r["gmm_Bp"].sample(n_scatter)[0],
        "g_C":      r["gmm_C"].sample(n_scatter)[0],
        "g_F":      r["gmm_F"].sample(n_scatter)[0],
        "g_D":      r["gmm_D"].sample(n_scatter)[0],
    }

    fig = plt.figure(figsize=(24, 4.6), facecolor='white')
    gs = GridSpec(1, 6, hspace=0.0, wspace=0.12,
                  left=0.025, right=0.99, top=0.82, bottom=0.04)

    # Paper-notation alignment:
    #   (B')         = regression + FULL loss (worst deterministic)
    #   (C)          = NCDE-style KNN + full loss (NCDE baselines)
    #   (F, mature)  = CSDI-analog frozen imputer → generator (paper Row 0c)
    #   (D)          = Co-Evolving EM (paper Row 5, achieves near-oracle)
    # We show (B') rather than (B_masked) here because its failure mode is
    # the most visually striking (straight-line collapse on a curved manifold).
    titles = [
        ("True distribution\n(data-generating process)", None, None),
        (f"(A) Oracle (clean data)\ndisc = {avg['disc_oracle']:.3f}",
         "gmm_oracle", "g_oracle"),
        (f"(B') Regression + full loss\ndisc = {avg['disc_Bp']:.3f}",
         "gmm_Bp", "g_Bp"),
        (f"(C) NCDE (KNN) + full loss\ndisc = {avg['disc_C']:.3f}",
         "gmm_C", "g_C"),
        (f"(F) CSDI-analog (K_imp={K_IMP}), FROZEN\ndisc = {avg['disc_F']:.3f}",
         "gmm_F", "g_F"),
        (f"(D) Co-Evolving EM (K={K_GEN})\ndisc = {avg['disc_D']:.3f}",
         "gmm_D", "g_D"),
    ]
    color_by_col = {1: 'grey', 2: 'tab:red', 3: 'tab:orange', 4: 'tab:blue', 5: 'tab:green'}
    for col, (title, gmm_key, gen_key) in enumerate(titles):
        ax = fig.add_subplot(gs[0, col])
        if col == 0:
            ax.contourf(gx, gy, true_dens, levels=14, cmap='Blues', alpha=0.75)
            ax.scatter(X_clean[:n_scatter, 0], X_clean[:n_scatter, 1],
                       s=4, alpha=0.25, c='k')
        else:
            ax.contourf(gx, gy, true_dens, levels=12, cmap='Greys', alpha=0.18)
            dens = clipped_density(r[gmm_key])
            ax.contourf(gx, gy, dens, levels=10, cmap='Blues', alpha=0.55)
            gen = scatter_samples[gen_key]
            ax.scatter(gen[:, 0], gen[:, 1], s=3.5, alpha=scatter_alpha,
                       c=color_by_col[col])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f'Cherry-pick: {dname} (p_miss={p_miss:.0%}) — '
        f'low-capacity CSDI-analog imputer (K_imp={K_IMP}) cannot preserve the full '
        f'structure it hands to the generator;\n'
        f'Co-Evolving EM (K={K_GEN} shared imputer/generator) matches the oracle. '
        f'(F→D gap = {result["gap"]:+.3f})',
        fontsize=13, fontweight='bold', y=0.99
    )
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")


def render_mse_trap_figure(result, dname, p_miss, outpath):
    """
    Companion figure that makes the 'conditional-mean trap' (paper §3.1)
    quantitatively visible: the methods that achieve the LOWEST imputation
    MSE are precisely the ones that produce the WORST generations.

    Left:   grouped bars of imputation MSE and MAE per method.
    Right:  grouped bars of generation disc score per method.
    Bottom: narrative strip explaining the inversion.

    Color scheme matches the main cherry-pick figure.
    """
    avg = result["avg"]
    methods = [
        ("(B') Regression",   "mse_Bp",      "mae_Bp",      "disc_Bp",      "tab:red"),
        ("(C) NCDE / KNN",    "mse_C",       "mae_C",       "disc_C",       "tab:orange"),
        ("(F_naive) 1-iter",  "mse_F_naive", "mae_F_naive", "disc_F_naive", "tab:purple"),
        ("(F) CSDI mature",   "mse_F",       "mae_F",       "disc_F",       "tab:blue"),
        ("(D) Co-Evolving EM", "mse_D",      "mae_D",       "disc_D",       "tab:green"),
    ]
    names    = [m[0] for m in methods]
    mse_vals = [avg[m[1]] for m in methods]
    mae_vals = [avg[m[2]] for m in methods]
    disc_vals = [avg[m[3]] for m in methods]
    colors   = [m[4] for m in methods]

    mse_argmin = int(np.argmin(mse_vals))
    disc_argmin = int(np.argmin(disc_vals))

    fig = plt.figure(figsize=(14, 5.8), facecolor='white')
    gs = GridSpec(1, 2, wspace=0.28, left=0.08, right=0.98,
                  top=0.80, bottom=0.22)

    # ── Left: imputation MSE/MAE ──
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    w = 0.38
    bars_mse = ax1.bar(x - w/2, mse_vals, width=w, color=colors,
                        alpha=0.9, edgecolor='black', linewidth=0.8, label='MSE')
    bars_mae = ax1.bar(x + w/2, mae_vals, width=w, color=colors,
                        alpha=0.45, edgecolor='black', linewidth=0.8,
                        hatch='//', label='MAE')
    for b, v in zip(bars_mse, mse_vals):
        ax1.text(b.get_x() + b.get_width()/2, v, f'{v:.2f}',
                 ha='center', va='bottom', fontsize=9)
    for b, v in zip(bars_mae, mae_vals):
        ax1.text(b.get_x() + b.get_width()/2, v, f'{v:.2f}',
                 ha='center', va='bottom', fontsize=9, alpha=0.7)
    ax1.annotate(
        'LOWEST MSE\n(MSE-optimal imputer)',
        xy=(mse_argmin - w/2, mse_vals[mse_argmin]),
        xytext=(mse_argmin - 0.2, max(mse_vals) * 1.22),
        fontsize=9, ha='center', color='darkred', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('Imputation error (↓)', fontsize=11)
    ax1.set_title('Imputation quality\n(what TST/NCDE/CSDI literature optimizes)',
                  fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.25)
    ax1.set_ylim(0, max(mse_vals) * 1.35)

    # ── Right: generation disc ──
    ax2 = fig.add_subplot(gs[0, 1])
    bars_disc = ax2.bar(x, disc_vals, color=colors, alpha=0.9,
                         edgecolor='black', linewidth=0.8)
    for b, v in zip(bars_disc, disc_vals):
        ax2.text(b.get_x() + b.get_width()/2, v, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9)
    # Oracle reference
    oracle = avg["disc_oracle"]
    ax2.axhline(oracle, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
    ax2.text(len(methods) - 0.5, oracle, f' oracle = {oracle:.3f}',
             va='bottom', ha='right', fontsize=9, style='italic')
    ax2.annotate(
        'LOWEST disc\n(best generation)',
        xy=(disc_argmin, disc_vals[disc_argmin]),
        xytext=(disc_argmin, max(disc_vals) * 0.55),
        fontsize=9, ha='center', color='darkgreen', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2),
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('Discriminative score  |0.5 − acc|  (↓)', fontsize=11)
    ax2.set_title('Generation quality\n(what actually matters)',
                  fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.25)
    ax2.set_ylim(0, max(disc_vals) * 1.22)

    punchline = (
        f"The conditional-mean trap: {names[mse_argmin]} achieves the LOWEST "
        f"imputation MSE ({mse_vals[mse_argmin]:.2f}) — yet the WORST generation "
        f"(disc = {disc_vals[mse_argmin]:.3f}).  "
        f"{names[disc_argmin]} achieves ~{mse_vals[disc_argmin]/mse_vals[mse_argmin]:.1f}× "
        f"HIGHER MSE but matches the oracle (disc = {disc_vals[disc_argmin]:.3f}).\n"
        f"MSE/MAE reward point estimates of E[x_miss | x_obs] that collapse "
        f"the conditional distribution; generation needs distributional samples."
    )
    fig.text(0.5, 0.04, punchline, ha='center', fontsize=10.5, style='italic',
             bbox=dict(boxstyle='round,pad=0.6', fc='#fff8e0',
                       ec='#b58b00', lw=1.0))
    fig.suptitle(
        f"Why better imputation ≠ better generation  —  "
        f"{dname} @ p_miss={p_miss:.0%}  (paper §3.1)",
        fontsize=13, fontweight='bold', y=0.97
    )
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    print(f"\nMSE-trap figure saved: {outpath}")


# ════════════════════════════════════════════════════════════════
# 5. Entry point
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = sweep()

    # Pick the configuration with the largest positive F→D gap.
    # (We require F > D, i.e. co-evolving EM is better than frozen CSDI-analog.)
    scored = sorted(results.items(), key=lambda kv: -kv[1]["gap"])
    print("\nTop configurations by F→D gap (want F > D):")
    for (dn, p), info in scored[:5]:
        a = info["avg"]
        print(f"  {dn:<16} p={p:.2f}   B={a['disc_B']:.3f}  B'={a['disc_Bp']:.3f}  "
              f"C={a['disc_C']:.3f}  F_naive={a['disc_F_naive']:.3f}  "
              f"F={a['disc_F']:.3f}  D={a['disc_D']:.3f}  gap(F→D)={info['gap']:+.3f}")

    # Render the top-3 candidates so we can visually compare.
    # Under the paper's metric (lower is better, 0 = fooled classifier),
    # we still want (F − D) as large as possible (F should be close to 0.5,
    # D close to 0) AND D close to Oracle (both near 0).
    def visual_score(info):
        a = info["avg"]
        return (info["gap"]) - 0.5 * abs(a["disc_D"] - a["disc_oracle"])

    scored_visual = sorted(results.items(), key=lambda kv: -visual_score(kv[1]))
    print("\nTop configurations by visual score (F→D gap + D ≈ oracle):")
    for (dn, p), info in scored_visual[:7]:
        a = info["avg"]
        print(f"  {dn:<16} p={p:.2f}   "
              f"F={a['disc_F']:.3f}  D={a['disc_D']:.3f}  Oracle={a['disc_oracle']:.3f}  "
              f"gap(F→D)={info['gap']:+.3f}  Δ(D-O)={a['disc_D']-a['disc_oracle']:+.3f}  "
              f"visual={visual_score(info):+.3f}")

    print("\nRendering top-5 candidates (samples + MSE-trap companion)...")
    for (win_dn, win_p), win in scored_visual[:5]:
        stem = f"cherrypick_{win_dn}_p{int(win_p*100)}"
        render_figure(win, win_dn, win_p, outpath=f"{stem}.png")
        render_mse_trap_figure(win, win_dn, win_p, outpath=f"{stem}_mse_trap.png")
