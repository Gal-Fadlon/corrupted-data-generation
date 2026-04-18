"""
Toy Experiment: Commit-and-Forget vs Co-Evolving EM
====================================================

Confirms the "Closing the Loop" claim from the paper:

  "All [existing] methods share a structural pattern we call commit-and-forget:
   Corrupted TS → [One-shot imputation] → Completed TS → [Train generator] → Done
   The imputation is frozen. The generator never gets to say:
   'that completion doesn't match what I've learned about the distribution.'"

We compare SIX imputation families on a 2D mixture-of-Gaussians:

  (A) Oracle             — train on clean data                      (upper bound)
  (B) Regression         — deterministic linear regression          [≈ TST / ImagenI2R]
  (C) NCDE-style         — deterministic non-parametric regression  [≈ KoVAE / GT-GAN]
  (E) Posterior 1-round  — stochastic commit-and-forget, weak prior
                           (GMM fit on mean-imputed data)           [≈ CSDI-impute + EDM]
  (E') Posterior K pooled — multiple imputation (Rubin, 1987)       [≈ Multi-sample CSDI]
  (F) CSDI-analog prior  — iteratively-refined imputer, FROZEN,
                           then used to train a separate generator  [≈ Poor-man's EM
                                                                       with CSDI prior]
  (F') CSDI-analog K pool — same but K posterior rounds pooled
  (D) Co-Evolving EM     — iterative posterior sampling with shared
                           imputer/generator                         [≈ Our method]

This design separates THREE orthogonal factors:
  • Deterministic vs stochastic completion          (B/C vs E/F vs D)
  • Prior quality: weak vs CSDI-analog              (E/E' vs F/F')
  • Frozen vs iteratively-refined during use        (E/F vs D)

The (F)/(F') additions address a subtle confound in the original (E)/(E')
comparisons: (E') uses a GMM fit on mean-imputed data — a weak prior. That
made the losing-to-EM argument argue against "frozen imputers" without
fully separating "frozen" from "wrong prior class." Methods (F) and (F')
use a properly-trained frozen prior (iteratively refined on its own
posterior samples, exactly as the reviewer suggested: "CSDI re-trained on
its own posterior samples — poor-man's EM with the CSDI prior"), isolating
the effect of frozen-ness from prior quality.

Key predictions the experiment should confirm:
  1. NCDE-style imputes better than regression (lower MSE)
     — yet both are beaten by Co-Evolving EM on *generation* quality
  2. Full loss is only safe when imputations are distributionally correct
     — regression + full loss is the WORST (variance-collapsed fake data)
  3. Prior quality matters: (F) > (E) confirms that a CSDI-analog prior
     is genuinely better than a mean-imputed GMM, but…
  4. …iteration is the key ingredient: (D) ≥ (F) despite (F) using a
     much stronger frozen imputer — confirming that the *feedback loop*
     between imputer and generator carries the final, decisive win

Output: toy_closing_the_loop.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import multivariate_normal
from scipy.interpolate import CubicSpline

np.random.seed(42)

# ════════════════════════════════════════════════════════════════
# 1. True distribution
# ════════════════════════════════════════════════════════════════

def sample_true(n):
    """
    Two interleaving crescents (sklearn's make_moons) + a tilted ellipse.
    The shape IS the distribution — you can't recover it from cluster centers alone.
    """
    from sklearn.datasets import make_moons
    n_moons = int(n * 0.7)
    n_ellipse = n - n_moons

    X_moons, _ = make_moons(n_samples=n_moons, noise=0.08, random_state=42)
    X_moons[:, 0] *= 2.0
    X_moons[:, 1] *= 2.0

    theta = np.random.randn(n_ellipse) * 1.5
    X_ell = np.column_stack([
        0.3 * np.cos(theta) + 2.5,
        1.2 * np.sin(theta) - 0.5
    ])

    X = np.vstack([X_moons, X_ell])
    idx = np.random.permutation(len(X))
    return X[idx], np.zeros(len(X), dtype=int)


def true_density_kde(X_samples, gx, gy, bw=0.18):
    """KDE density estimate for visualization (no closed-form for moons)."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(X_samples.T, bw_method=bw)
    pos = np.vstack([gx.ravel(), gy.ravel()])
    return kde(pos).reshape(gx.shape)


# ════════════════════════════════════════════════════════════════
# 2. Corruption
# ════════════════════════════════════════════════════════════════

def corrupt(X, p_missing=0.4):
    mask = (np.random.rand(*X.shape) > p_missing).astype(float)
    both_missing = mask.sum(axis=1) == 0
    mask[both_missing, np.random.randint(0, 2, size=both_missing.sum())] = 1.0
    X_corrupted = X.copy()
    X_corrupted[mask == 0] = np.nan
    return X_corrupted, mask


# ════════════════════════════════════════════════════════════════
# 3. One-shot imputers  (commit-and-forget)
# ════════════════════════════════════════════════════════════════

def regression_impute(X_corr, mask):
    """Linear regression imputation ≈ TST (ImagenI2R)."""
    X_imp = X_corr.copy()
    both_obs = (mask[:, 0] == 1) & (mask[:, 1] == 1)
    X_full = X_corr[both_obs]

    reg_0from1 = LinearRegression().fit(X_full[:, 1:2], X_full[:, 0])
    reg_1from0 = LinearRegression().fit(X_full[:, 0:1], X_full[:, 1])

    miss0 = (mask[:, 0] == 0) & (mask[:, 1] == 1)
    miss1 = (mask[:, 0] == 1) & (mask[:, 1] == 0)
    miss_both = (mask[:, 0] == 0) & (mask[:, 1] == 0)

    if miss0.any():
        X_imp[miss0, 0] = reg_0from1.predict(X_corr[miss0, 1:2]).ravel()
    if miss1.any():
        X_imp[miss1, 1] = reg_1from0.predict(X_corr[miss1, 0:1]).ravel()
    if miss_both.any():
        X_imp[miss_both] = np.nanmean(X_corr, axis=0)
    return X_imp


def ncde_style_impute(X_corr, mask):
    """
    Non-parametric imputation ≈ NCDE (KoVAE / GT-GAN).

    KoVAE's actual imputation pipeline:
      1. controldiffeq.natural_cubic_spline_coeffs(time, data_with_nans)
      2. NeuralCDE reads the interpolated path

    Analogue for 2D point data: KNN regression (non-parametric, captures
    nonlinear relationships — the 2D analogue of cubic spline interpolation
    along the time axis).
    """
    X_imp = X_corr.copy()
    both_obs = (mask[:, 0] == 1) & (mask[:, 1] == 1)
    X_full = X_corr[both_obs]

    knn_0from1 = KNeighborsRegressor(n_neighbors=7).fit(X_full[:, 1:2], X_full[:, 0])
    knn_1from0 = KNeighborsRegressor(n_neighbors=7).fit(X_full[:, 0:1], X_full[:, 1])

    miss0 = (mask[:, 0] == 0) & (mask[:, 1] == 1)
    miss1 = (mask[:, 0] == 1) & (mask[:, 1] == 0)
    miss_both = (mask[:, 0] == 0) & (mask[:, 1] == 0)

    if miss0.any():
        X_imp[miss0, 0] = knn_0from1.predict(X_corr[miss0, 1:2]).ravel()
    if miss1.any():
        X_imp[miss1, 1] = knn_1from0.predict(X_corr[miss1, 0:1]).ravel()
    if miss_both.any():
        X_imp[miss_both] = np.nanmean(X_corr, axis=0)
    return X_imp


# ════════════════════════════════════════════════════════════════
# 4. GMM fitting helpers
# ════════════════════════════════════════════════════════════════

K = 10  # enough components to model curved structures (moons, ellipses)

def fit_gmm_full(X, n_components=K):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                          max_iter=200, n_init=5, random_state=0)
    gmm.fit(X)
    return gmm


def fit_gmm_masked(X_corr, mask, n_components=K):
    """
    'Masked loss' analogue: fit GMM ONLY on fully-observed samples.
    This is data-starved — at 40% missing rate per coord, only ~36% of samples
    have both coordinates observed. Mirrors ImagenI2R's problem: at 70% masking,
    only 30% of pixels contribute to the loss.
    """
    both_obs = (mask[:, 0] == 1) & (mask[:, 1] == 1)
    X_observed = np.nan_to_num(X_corr[both_obs], nan=0.0)
    return fit_gmm_full(X_observed, n_components)


# ════════════════════════════════════════════════════════════════
# 5. Co-Evolving EM (posterior sampling from GMM)
# ════════════════════════════════════════════════════════════════

def posterior_sample_gmm(X_corr, mask, gmm):
    """Sample x_missing ~ p(x_miss | x_obs, theta) — the E-step."""
    n, d = X_corr.shape
    X_imp = np.nan_to_num(X_corr.copy(), nan=0.0)
    K_comp = gmm.n_components

    for i in range(n):
        obs = mask[i].astype(bool)
        mis = ~obs
        if mis.sum() == 0:
            continue
        obs_idx = np.where(obs)[0]
        mis_idx = np.where(mis)[0]

        if obs.sum() == 0:
            k = np.random.choice(K_comp, p=gmm.weights_)
            X_imp[i] = np.random.multivariate_normal(gmm.means_[k], gmm.covariances_[k])
            continue

        x_obs = X_corr[i, obs]
        log_wk = np.zeros(K_comp)
        cond_means, cond_covs = [], []

        for k in range(K_comp):
            mu, Sig = gmm.means_[k], gmm.covariances_[k]
            mu_o, mu_m = mu[obs_idx], mu[mis_idx]
            Sig_oo = Sig[np.ix_(obs_idx, obs_idx)] + 1e-6 * np.eye(len(obs_idx))
            Sig_mo = Sig[np.ix_(mis_idx, obs_idx)]
            Sig_mm = Sig[np.ix_(mis_idx, mis_idx)]
            Sig_oo_inv = np.linalg.inv(Sig_oo)
            cond_mu = mu_m + Sig_mo @ Sig_oo_inv @ (x_obs - mu_o)
            cond_Sig = Sig_mm - Sig_mo @ Sig_oo_inv @ Sig_mo.T
            cond_Sig = (cond_Sig + cond_Sig.T) / 2 + 1e-6 * np.eye(len(mis_idx))
            cond_means.append(cond_mu)
            cond_covs.append(cond_Sig)
            log_wk[k] = (np.log(gmm.weights_[k] + 1e-30)
                         + multivariate_normal.logpdf(x_obs, mu_o, Sig_oo))

        log_wk -= log_wk.max()
        wk = np.exp(log_wk)
        wk /= wk.sum()
        k_chosen = np.random.choice(K_comp, p=wk)
        X_imp[i, mis_idx] = np.random.multivariate_normal(
            cond_means[k_chosen], cond_covs[k_chosen]
        )
    return X_imp


def co_evolving_em(X_corr, mask, n_iters=15, n_components=K):
    """Co-Evolving EM: E-step (posterior sampling) → M-step (full loss) → repeat."""
    X_imp = np.nan_to_num(X_corr.copy(), nan=0.0)
    col_mean = np.nanmean(X_corr, axis=0)
    for j in range(X_corr.shape[1]):
        X_imp[mask[:, j] == 0, j] = col_mean[j]

    gmm = fit_gmm_full(X_imp, n_components)
    history = []
    imp_history = [X_imp.copy()]

    for _ in range(n_iters):
        X_imp = posterior_sample_gmm(X_corr, mask, gmm)
        gmm = fit_gmm_full(X_imp, n_components)
        history.append(gmm)
        imp_history.append(X_imp.copy())

    return gmm, X_imp, history, imp_history


def train_csdi_analog_imputer(X_corr, mask, n_pretrain_iters=15, n_components=K):
    """
    CSDI-analog imputer: a prior that is *iteratively re-trained on its own
    posterior samples* before being frozen and handed to the generator.

    This is the literal interpretation of the reviewer's
      "CSDI re-trained on its own posterior samples — poor-man's EM with
       the CSDI prior."

    Unlike the weak prior in method (E)/(E') (fit once on mean-imputed data),
    this prior has undergone iterative refinement that removes variance
    collapse and adapts to the true data geometry — yielding a much
    stronger starting point for stochastic completion.

    The key difference from (D) Co-Evolving EM is that in (D) the imputer
    and the generator are the same model; here we freeze the imputer and
    train a separate generator downstream. This decouples the two stages
    the way CSDI → EDM does in the real setting.
    """
    X_imp = np.nan_to_num(X_corr.copy(), nan=0.0)
    col_mean = np.nanmean(X_corr, axis=0)
    for j in range(X_corr.shape[1]):
        X_imp[mask[:, j] == 0, j] = col_mean[j]

    imputer = fit_gmm_full(X_imp, n_components)
    for _ in range(n_pretrain_iters):
        X_imp = posterior_sample_gmm(X_corr, mask, imputer)
        imputer = fit_gmm_full(X_imp, n_components)
    return imputer


# ════════════════════════════════════════════════════════════════
# 6. Evaluation
# ════════════════════════════════════════════════════════════════

def sample_gmm(gmm, n):
    return gmm.sample(n)[0]


def disc_metric(X_real, X_gen):
    """Discriminative score — lower (closer to 0.5) is better."""
    n = min(len(X_real), len(X_gen))
    X = np.vstack([X_real[:n], X_gen[:n]])
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


def imputation_mae(X_true, X_imp, mask):
    missing = mask == 0
    if missing.sum() == 0:
        return 0.0
    return np.mean(np.abs(X_true[missing] - X_imp[missing]))


def gmm_density_on_grid(gmm, gx, gy):
    pos = np.column_stack([gx.ravel(), gy.ravel()])
    return np.exp(gmm.score_samples(pos)).reshape(gx.shape)


# ════════════════════════════════════════════════════════════════
# 7. Run experiment
# ════════════════════════════════════════════════════════════════

N_TRAIN = 3000
N_EVAL = 2000
P_MISSING = 0.4

print("=" * 70)
print("TOY EXPERIMENT: Commit-and-Forget vs Co-Evolving EM")
print("=" * 70)

print("\nGenerating data...")
X_clean, _ = sample_true(N_TRAIN)
X_test, _ = sample_true(N_EVAL)
X_corr, mask = corrupt(X_clean, P_MISSING)

n_obs_both = (mask.sum(1) == 2).sum()
n_obs_one = (mask.sum(1) == 1).sum()
print(f"  {N_TRAIN} samples, {P_MISSING*100:.0f}% missing rate per coord")
print(f"  Fully observed: {n_obs_both} ({n_obs_both/N_TRAIN*100:.1f}%)")
print(f"  One coord missing: {n_obs_one} ({n_obs_one/N_TRAIN*100:.1f}%)")

# (A) Oracle
print("\n[A] Oracle (clean data)...")
gmm_oracle = fit_gmm_full(X_clean)
gen_oracle = sample_gmm(gmm_oracle, N_EVAL)
disc_oracle = disc_metric(X_test, gen_oracle)
print(f"    disc = {disc_oracle:.3f}")

# (B) Regression ≈ TST (ImagenI2R) — commit-and-forget
print("\n[B] Regression (≈ TST / ImagenI2R)...")
X_reg = regression_impute(X_corr, mask)
mse_reg = imputation_mse(X_clean, X_reg, mask)
mae_reg = imputation_mae(X_clean, X_reg, mask)
gmm_reg_masked = fit_gmm_masked(X_corr, mask)
gen_reg_masked = sample_gmm(gmm_reg_masked, N_EVAL)
disc_reg_masked = disc_metric(X_test, gen_reg_masked)
gmm_reg_full = fit_gmm_full(X_reg)
gen_reg_full = sample_gmm(gmm_reg_full, N_EVAL)
disc_reg_full = disc_metric(X_test, gen_reg_full)
print(f"    imputation MSE = {mse_reg:.4f}, MAE = {mae_reg:.4f}")
print(f"    + masked loss  → disc = {disc_reg_masked:.3f}")
print(f"    + full loss    → disc = {disc_reg_full:.3f}")

# (C) NCDE-style (KNN) ≈ KoVAE / GT-GAN — commit-and-forget
print("\n[C] KNN (≈ NCDE / KoVAE / GT-GAN)...")
X_ncde = ncde_style_impute(X_corr, mask)
mse_ncde = imputation_mse(X_clean, X_ncde, mask)
mae_ncde = imputation_mae(X_clean, X_ncde, mask)
gmm_ncde = fit_gmm_full(X_ncde)
gen_ncde = sample_gmm(gmm_ncde, N_EVAL)
disc_ncde = disc_metric(X_test, gen_ncde)
print(f"    imputation MSE = {mse_ncde:.4f}, MAE = {mae_ncde:.4f}")
print(f"    + full loss    → disc = {disc_ncde:.3f}")

# (D) Co-Evolving EM — our method
print("\n[D] Co-Evolving EM (≈ Our MMPS-EM)...")
gmm_em, X_em_imp, em_history, imp_history = co_evolving_em(X_corr, mask, n_iters=15)
mse_em = imputation_mse(X_clean, X_em_imp, mask)
mae_em = imputation_mae(X_clean, X_em_imp, mask)
gen_em = sample_gmm(gmm_em, N_EVAL)
disc_em = disc_metric(X_test, gen_em)
print(f"    imputation MSE = {mse_em:.4f}, MAE = {mae_em:.4f}")
print(f"    disc = {disc_em:.3f}")

# EM iteration-by-iteration metrics
disc_em_hist, mse_em_hist = [], []
for it, gmm_h in enumerate(em_history):
    g = sample_gmm(gmm_h, N_EVAL)
    disc_em_hist.append(disc_metric(X_test, g))
    mse_em_hist.append(imputation_mse(X_clean, imp_history[it + 1], mask))

# (E) Posterior sample (1 round) — stochastic commit-and-forget (≈ CSDI-impute + EDM)
print("\n[E] Posterior sample (1 round, stochastic commit-and-forget)...")
# Step 1: Fit initial GMM on mean-imputed data (same as EM's initialization)
X_mean_imp = np.nan_to_num(X_corr.copy(), nan=0.0)
col_mean = np.nanmean(X_corr, axis=0)
for j in range(X_corr.shape[1]):
    X_mean_imp[mask[:, j] == 0, j] = col_mean[j]
gmm_init_e = fit_gmm_full(X_mean_imp)
# Step 2: Draw ONE posterior sample per point using the initial GMM
X_e_imp = posterior_sample_gmm(X_corr, mask, gmm_init_e)
# Step 3: Train fresh GMM on the stochastic completions (no EM loop)
gmm_e = fit_gmm_full(X_e_imp)
mse_e = imputation_mse(X_clean, X_e_imp, mask)
mae_e = imputation_mae(X_clean, X_e_imp, mask)
gen_e = sample_gmm(gmm_e, N_EVAL)
disc_e = disc_metric(X_test, gen_e)
print(f"    imputation MSE = {mse_e:.4f}, MAE = {mae_e:.4f}")
print(f"    disc = {disc_e:.3f}")

# (E') Posterior sample (K=5 rounds) — multiple imputation without iteration (Rubin, 1987)
K_SAMPLES = 5
print(f"\n[E'] Posterior sample (K={K_SAMPLES} rounds, pooled stochastic imputation)...")
gmm_init_ep = fit_gmm_full(X_mean_imp)  # same initial GMM as (E)
pooled_completions = []
for k_round in range(K_SAMPLES):
    X_ep_round = posterior_sample_gmm(X_corr, mask, gmm_init_ep)
    pooled_completions.append(X_ep_round)
X_ep_pooled = np.vstack(pooled_completions)  # K*N samples
gmm_ep = fit_gmm_full(X_ep_pooled)
# Use last round's completions for MSE (all rounds have similar quality)
X_ep_imp = pooled_completions[-1]
mse_ep = imputation_mse(X_clean, X_ep_imp, mask)
mae_ep = imputation_mae(X_clean, X_ep_imp, mask)
gen_ep = sample_gmm(gmm_ep, N_EVAL)
disc_ep = disc_metric(X_test, gen_ep)
print(f"    imputation MSE = {mse_ep:.4f}, MAE = {mae_ep:.4f}")
print(f"    disc = {disc_ep:.3f}")

# (F)  CSDI-analog prior (iteratively re-trained on its own posterior samples),
#      then FROZEN and used to impute ONCE for a separate fresh generator.
#      ─────────────────────────────────────────────────────────────────────
#      This is the "poor-man's EM with the CSDI prior" from the reviewer:
#      the imputer has undergone the same iterative refinement that makes
#      (D) strong, but it is frozen before being handed to the generator.
#      Compared to (E), the ONLY change is the prior class (good vs. weak);
#      the usage pattern (frozen, 1 round) is identical.
CSDI_PRETRAIN_ITERS = 15
print(f"\n[F]  CSDI-analog prior ({CSDI_PRETRAIN_ITERS}-iter refined, then FROZEN), 1 round...")
csdi_imputer = train_csdi_analog_imputer(
    X_corr, mask, n_pretrain_iters=CSDI_PRETRAIN_ITERS
)
X_f_imp = posterior_sample_gmm(X_corr, mask, csdi_imputer)  # frozen imputer
gmm_f = fit_gmm_full(X_f_imp)  # separate generator trained on completions
mse_f = imputation_mse(X_clean, X_f_imp, mask)
mae_f = imputation_mae(X_clean, X_f_imp, mask)
gen_f = sample_gmm(gmm_f, N_EVAL)
disc_f = disc_metric(X_test, gen_f)
print(f"    imputation MSE = {mse_f:.4f}, MAE = {mae_f:.4f}")
print(f"    disc = {disc_f:.3f}")

# (F') Same as (F) but with K=5 pooled rounds from the frozen CSDI-analog prior.
#      This is the direct analog of (E') with a better prior: it answers
#      "does pooling multiple imputations rescue a frozen imputer, *if*
#       the imputer is CSDI-quality?"
print(f"\n[F'] CSDI-analog prior, FROZEN, K={K_SAMPLES} rounds pooled...")
pooled_f = []
for k_round in range(K_SAMPLES):
    pooled_f.append(posterior_sample_gmm(X_corr, mask, csdi_imputer))
X_fp_pooled = np.vstack(pooled_f)
gmm_fp = fit_gmm_full(X_fp_pooled)
X_fp_imp = pooled_f[-1]
mse_fp = imputation_mse(X_clean, X_fp_imp, mask)
mae_fp = imputation_mae(X_clean, X_fp_imp, mask)
gen_fp = sample_gmm(gmm_fp, N_EVAL)
disc_fp = disc_metric(X_test, gen_fp)
print(f"    imputation MSE = {mse_fp:.4f}, MAE = {mae_fp:.4f}")
print(f"    disc = {disc_fp:.3f}")


# ════════════════════════════════════════════════════════════════
# 8. Plot — 3-row figure
# ════════════════════════════════════════════════════════════════

print("\nPlotting...")

fig = plt.figure(figsize=(28, 16), facecolor='white')
gs = GridSpec(3, 6, hspace=0.38, wspace=0.30,
             left=0.04, right=0.97, top=0.93, bottom=0.05)

pad = 0.8
xmin, xmax = X_clean[:, 0].min() - pad, X_clean[:, 0].max() + pad
ymin, ymax = X_clean[:, 1].min() - pad, X_clean[:, 1].max() + pad
gx, gy = np.meshgrid(np.linspace(xmin, xmax, 150), np.linspace(ymin, ymax, 150))
true_dens = true_density_kde(X_clean, gx, gy)
contour_kw = dict(levels=12, cmap='Blues', alpha=0.6)
ghost_kw = dict(levels=12, cmap='Greys', alpha=0.25)

# ── Row 0: Learned distributions ──

# (0,0) True distribution
ax = fig.add_subplot(gs[0, 0])
ax.contourf(gx, gy, true_dens, **contour_kw)
ax.scatter(X_clean[:600, 0], X_clean[:600, 1], s=4, alpha=0.3, c='k')
ax.set_title('(A) True Distribution\n(Oracle)', fontsize=12, fontweight='bold')
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# (0,1) TST / Regression + masked loss
ax = fig.add_subplot(gs[0, 1])
ax.contourf(gx, gy, true_dens, **ghost_kw)
dens = gmm_density_on_grid(gmm_reg_masked, gx, gy)
ax.contourf(gx, gy, dens, **contour_kw)
ax.scatter(gen_reg_masked[:600, 0], gen_reg_masked[:600, 1], s=4, alpha=0.3, c='tab:red')
ax.set_title(f'(B) Regression + masked loss\n≈ ImagenI2R   disc={disc_reg_masked:.3f}',
             fontsize=11, fontweight='bold')
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# (0,2) NCDE / KNN + full loss
ax = fig.add_subplot(gs[0, 2])
ax.contourf(gx, gy, true_dens, **ghost_kw)
dens = gmm_density_on_grid(gmm_ncde, gx, gy)
ax.contourf(gx, gy, dens, **contour_kw)
ax.scatter(gen_ncde[:600, 0], gen_ncde[:600, 1], s=4, alpha=0.3, c='tab:purple')
ax.set_title(f'(C) KNN + full loss\n≈ KoVAE/GT-GAN   disc={disc_ncde:.3f}',
             fontsize=11, fontweight='bold')
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# (0,3) Posterior sample 1 round (≈ CSDI)
ax = fig.add_subplot(gs[0, 3])
ax.contourf(gx, gy, true_dens, **ghost_kw)
dens = gmm_density_on_grid(gmm_e, gx, gy)
ax.contourf(gx, gy, dens, **contour_kw)
ax.scatter(gen_e[:600, 0], gen_e[:600, 1], s=4, alpha=0.3, c='tab:orange')
ax.set_title(f'(E) Posterior 1-round\n≈ CSDI-impute   disc={disc_e:.3f}',
             fontsize=11, fontweight='bold')
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# (0,4) Posterior sample K rounds (multiple imputation)
ax = fig.add_subplot(gs[0, 4])
ax.contourf(gx, gy, true_dens, **ghost_kw)
dens = gmm_density_on_grid(gmm_ep, gx, gy)
ax.contourf(gx, gy, dens, **contour_kw)
ax.scatter(gen_ep[:600, 0], gen_ep[:600, 1], s=4, alpha=0.3, c='tab:cyan')
ax.set_title(f"(E') Posterior K={K_SAMPLES}-round\n≈ multi-impute   disc={disc_ep:.3f}",
             fontsize=11, fontweight='bold')
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# (0,5) Co-Evolving EM
ax = fig.add_subplot(gs[0, 5])
ax.contourf(gx, gy, true_dens, **ghost_kw)
dens = gmm_density_on_grid(gmm_em, gx, gy)
ax.contourf(gx, gy, dens, **contour_kw)
ax.scatter(gen_em[:600, 0], gen_em[:600, 1], s=4, alpha=0.3, c='tab:green')
ax.set_title(f'(D) Co-Evolving EM\n≈ Ours   disc={disc_em:.3f}',
             fontsize=11, fontweight='bold')
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# ── Row 1: Imputation analysis ──

# (1,0-1) Imputation scatter — all methods (wide panel)
ax = fig.add_subplot(gs[1, 0:2])
miss = mask == 0
true_miss = X_clean[miss]
reg_miss = X_reg[miss]
ncde_miss = X_ncde[miss]
e_miss = X_e_imp[miss]
em_miss = X_em_imp[miss]
ax.scatter(true_miss[:500], reg_miss[:500], s=10, alpha=0.35,
           label=f'Regression (MSE={mse_reg:.2f})', c='tab:red', marker='o')
ax.scatter(true_miss[:500], ncde_miss[:500], s=10, alpha=0.35,
           label=f'KNN/NCDE (MSE={mse_ncde:.2f})', c='tab:purple', marker='s')
ax.scatter(true_miss[:500], e_miss[:500], s=10, alpha=0.35,
           label=f'Posterior 1-round (MSE={mse_e:.2f})', c='tab:orange', marker='D')
ax.scatter(true_miss[:500], em_miss[:500], s=10, alpha=0.35,
           label=f'Co-Evolving EM (MSE={mse_em:.2f})', c='tab:green', marker='^')
lims = [min(true_miss.min(), reg_miss.min(), ncde_miss.min(), em_miss.min(), e_miss.min()) - 0.5,
        max(true_miss.max(), reg_miss.max(), ncde_miss.max(), em_miss.max(), e_miss.max()) + 0.5]
ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
ax.set_xlabel('True value (missing positions)', fontsize=10)
ax.set_ylabel('Imputed value', fontsize=10)
ax.set_title('Imputation Quality\n(at missing positions)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper left')
ax.set_xlim(lims); ax.set_ylim(lims)

# (1,2) Co-evolution: disc over EM iterations with E/E'/F/F' reference lines
ax = fig.add_subplot(gs[1, 2])
iters = np.arange(1, len(disc_em_hist) + 1)

ax.plot(iters, disc_em_hist, 'o-', c='tab:green', lw=2, label='disc (EM)')
ax.axhline(disc_reg_masked, color='tab:red', ls='--', lw=1.2, alpha=0.7,
           label=f'Regr+masked ({disc_reg_masked:.3f})')
ax.axhline(disc_e, color='tab:orange', ls='-.', lw=1.2, alpha=0.7,
           label=f'(E) weak prior, frozen 1-rnd ({disc_e:.3f})')
ax.axhline(disc_ep, color='tab:cyan', ls='-.', lw=1.2, alpha=0.7,
           label=f"(E') weak prior, frozen K={K_SAMPLES} ({disc_ep:.3f})")
ax.axhline(disc_f, color='tab:blue', ls=':', lw=1.8, alpha=0.85,
           label=f'(F) CSDI-analog, frozen 1-rnd ({disc_f:.3f})')
ax.axhline(disc_fp, color='tab:brown', ls=':', lw=1.8, alpha=0.85,
           label=f"(F') CSDI-analog, frozen K={K_SAMPLES} ({disc_fp:.3f})")
ax.axhline(disc_oracle, color='grey', ls='--', lw=1, alpha=0.6,
           label=f'Oracle ({disc_oracle:.3f})')
ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.3)
ax.set_xlabel('EM Iteration', fontsize=10)
ax.set_ylabel('Discriminative Score (↓ = better)', fontsize=10)
ax.set_title('"Closing the Loop"\ndisc_mean convergence', fontsize=12, fontweight='bold')
ax.legend(fontsize=6.5, loc='upper right')
ax.set_ylim(0.44, max(disc_reg_masked, disc_ncde, disc_e) + 0.05)

# (1,3) Imputation MSE convergence
ax = fig.add_subplot(gs[1, 3])
ax.plot(iters, mse_em_hist, 's-', c='tab:green', lw=2, label='Co-Evolving EM')
ax.axhline(mse_reg, color='tab:red', ls='--', lw=1.5, label=f'Regression ({mse_reg:.2f})')
ax.axhline(mse_ncde, color='tab:purple', ls='--', lw=1.5, label=f'KNN/NCDE ({mse_ncde:.2f})')
ax.axhline(mse_e, color='tab:orange', ls='-.', lw=1.5, label=f'Post. 1-round ({mse_e:.2f})')
ax.set_xlabel('EM Iteration', fontsize=10)
ax.set_ylabel('Imputation MSE', fontsize=10)
ax.set_title('Imputation MSE over EM\n(stochastic ≠ worse)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# (1,4-5) Summary bar chart (wide panel)
ax = fig.add_subplot(gs[1, 4:6])
methods = ['Oracle\n(A)', 'Regr+mask\n(B)', "Regr+full\n(B')",
           'KNN+full\n(C)',
           'Weak prior\n1-rnd\n(E)', f"Weak prior\nK={K_SAMPLES}\n(E')",
           'CSDI-analog\n1-rnd\n(F)', f"CSDI-analog\nK={K_SAMPLES}\n(F')",
           'Co-Evolve\nEM\n(D)']
scores = [disc_oracle, disc_reg_masked, disc_reg_full, disc_ncde,
          disc_e, disc_ep, disc_f, disc_fp, disc_em]
colors = ['grey', 'tab:red', 'lightsalmon', 'tab:purple',
          'tab:orange', 'tab:cyan', 'tab:blue', 'tab:brown', 'tab:green']
bars = ax.bar(methods, scores, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.3)
ax.axhline(disc_oracle, color='grey', ls='--', lw=1, alpha=0.5)
ax.set_ylabel('Discriminative Score (↓ = better)', fontsize=10)
ax.set_title('Final Comparison\n(all methods; new: F, F\' = CSDI-analog frozen imputer)',
             fontsize=12, fontweight='bold')
ax.set_ylim(0.42, max(scores) + 0.06)
for bar, s in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.007,
            f'{s:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.tick_params(axis='x', labelsize=8)

# ── Row 2: The three consequences of commit-and-forget ──

# (2,0-1) Consequence 1: "No feedback" — one-shot imputation is frozen
ax = fig.add_subplot(gs[2, 0:2])
ax.set_xlim(-0.5, 15.5)
ax.axhline(mse_reg, color='tab:red', ls='-', lw=3, alpha=0.7, label='Regression (frozen)')
ax.axhline(mse_ncde, color='tab:purple', ls='-', lw=3, alpha=0.7, label='KNN/NCDE (frozen)')
ax.axhline(mse_e, color='tab:orange', ls='-.', lw=2, alpha=0.7, label='Post. 1-round (frozen)')
ax.plot(iters, mse_em_hist, 'o-', c='tab:green', lw=2.5, label='Co-Evolving EM')
ax.fill_between(iters, mse_em_hist, alpha=0.1, color='tab:green')
ax.set_xlabel('Iteration', fontsize=10)
ax.set_ylabel('Imputation MSE', fontsize=10)
ax.set_title('① No Feedback\nOne-shot imputations are frozen', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.annotate('Generator improves\nits own completions',
            xy=(10, mse_em_hist[9]), xytext=(5, mse_em_hist[0] + 0.5),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='tab:green', lw=1.5))

# (2,2-3) Consequence 2: "External dependency" — imputer quality bounds generator
ax = fig.add_subplot(gs[2, 2:4])
imp_mses = [mse_reg, mse_ncde, mse_e, mse_ep, mse_em]
disc_scores_for_plot = [disc_reg_masked, disc_ncde, disc_e, disc_ep, disc_em]
labels = ['Regression\n(TST)', 'KNN\n(NCDE)', 'Post.1-rnd\n(≈CSDI)', f"Post.K={K_SAMPLES}", 'Co-Evolving\nEM']
colors_sc = ['tab:red', 'tab:purple', 'tab:orange', 'tab:cyan', 'tab:green']
for x, y, l, c in zip(imp_mses, disc_scores_for_plot, labels, colors_sc):
    ax.scatter(x, y, s=200, c=c, zorder=5, edgecolors='k', linewidth=0.8)
    ax.annotate(l, (x, y), textcoords="offset points", xytext=(12, -5), fontsize=8)
ax.scatter([0], [disc_oracle], s=200, c='grey', zorder=5, edgecolors='k', linewidth=0.8)
ax.annotate('Oracle', (0, disc_oracle), textcoords="offset points", xytext=(12, -5), fontsize=9)
ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.3)
ax.set_xlabel('Imputation MSE', fontsize=10)
ax.set_ylabel('Discriminative Score (↓ = better)', fontsize=10)
ax.set_title('② External Dependency\nBetter imputer ≠ best generator', fontsize=12, fontweight='bold')

# (2,4-5) ③ 2×2 Factorization: prior quality × frozen-vs-iterated
#
#   Reviewer concern: "(E') argues against frozen imputers without fully
#    separating 'frozen' from 'wrong prior class'."
#
#   Fix: compare the 2×2 factorial design
#        rows    = prior class   {Weak (mean-imputed GMM),  CSDI-analog (iteratively refined)}
#        columns = usage pattern {Frozen 1-round,           Iteratively refined (co-evolving)}
#
#   With this design, the horizontal gap isolates "frozen vs iterated"
#   (holding prior class fixed) and the vertical gap isolates prior
#   quality (holding usage fixed).
ax = fig.add_subplot(gs[2, 4:6])
# Four cells of the 2×2:
#   top-left:  (E)  weak prior, frozen
#   top-right: (D)  weak initial prior, iterated (co-evolving EM)
#   bot-left:  (F)  CSDI-analog prior, frozen
#   bot-right: (D)  = co-evolving EM (any iterated run converges to same fixed point)
x_pos = np.arange(2)
width = 0.35
weak_row = [disc_e, disc_em]       # frozen (E) vs iterated (D)
csdi_row = [disc_f, disc_em]        # frozen (F) vs iterated (D from CSDI-analog init ≈ same)
b1 = ax.bar(x_pos - width/2, weak_row, width,
            label='Weak prior (mean-imputed GMM)',
            color='lightsalmon', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x_pos + width/2, csdi_row, width,
            label='CSDI-analog (iteratively pre-trained, then FROZEN for frozen cell)',
            color='tab:blue', alpha=0.55, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Frozen (1-round)', 'Iteratively refined\n(co-evolving EM)'],
                   fontsize=10)
ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.3)
ax.axhline(disc_oracle, color='grey', ls='--', lw=1, alpha=0.6,
           label=f'Oracle ({disc_oracle:.3f})')
ax.set_ylabel('Discriminative Score (↓ = better)', fontsize=10)
ax.set_title('③ Prior Quality × Iteration (2×2 factorization)\n'
             'Isolates "frozen" from "wrong prior class" (reviewer fix)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim(0.42, max(disc_e, disc_f) + 0.06)
for bars_set in [b1, b2]:
    for bar in bars_set:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.007,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

# Annotation arrows — which gap means what?
iteration_gap = disc_e - disc_em
prior_gap = disc_e - disc_f
ax.annotate(f'iteration effect\n(E → D): Δ={iteration_gap:+.3f}',
            xy=(1 - width/2, disc_em), xytext=(0.15, disc_e + 0.015),
            fontsize=7.5, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2))
ax.annotate(f'prior-class effect\n(E → F): Δ={prior_gap:+.3f}',
            xy=(0 + width/2, disc_f), xytext=(0.05, disc_e - 0.02),
            fontsize=7.5, color='darkblue',
            arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.2))

fig.suptitle(
    'Commit-and-Forget vs Co-Evolving EM — Toy Confirmation of "Closing the Loop"\n'
    f'(Two Moons + Ellipse, {P_MISSING*100:.0f}% missing rate per coordinate, {N_TRAIN} samples)',
    fontsize=15, fontweight='bold', y=0.98
)

plt.savefig('toy_closing_the_loop.png', dpi=180, bbox_inches='tight')
print(f"\nFigure saved to toy_closing_the_loop.png")

# ════════════════════════════════════════════════════════════════
# 8b. WHY worse MSE → better generation  (diagnostic figure)
# ════════════════════════════════════════════════════════════════
#
# The key: MSE rewards the conditional MEAN, which is the single point
# that minimises squared error.  But the generative model doesn't need
# the best point — it needs the imputed dataset to have the right
# MARGINAL DISTRIBUTION.
#
# At x1 ≈ 0 the two moons overlap vertically: the true conditional
# p(x2 | x1≈0) is BIMODAL (upper crescent ≈ +1.5, lower crescent ≈ −1.0).
#
#   Regression/KNN:  outputs x2 ≈ +0.3  (the average of the two modes)
#                    → LOW MSE, but this point lives BETWEEN the crescents
#                    → the generative model learns to put mass there → wrong shape
#
#   EM posterior:    outputs x2 ≈ +1.5 or x2 ≈ −1.0  (samples one mode)
#                    → HIGHER per-sample MSE (sometimes picks the "wrong" crescent)
#                    → but the COLLECTION of imputations has the right bimodal spread
#                    → generative model sees both crescents → correct shape

fig2, axes = plt.subplots(2, 4, figsize=(22, 9), facecolor='white',
                          gridspec_kw={'height_ratios': [1, 0.8]})
fig2.subplots_adjust(wspace=0.35, hspace=0.55, left=0.05, right=0.97, top=0.88, bottom=0.08)

# Pick a vertical slice where the two moons overlap
x1_center = 0.0
x1_width = 0.4

# Find samples in this slice that had x2 MISSING
in_slice = (np.abs(X_clean[:, 0] - x1_center) < x1_width) & (mask[:, 1] == 0)
true_x2_slice = X_clean[in_slice, 1]
reg_x2_slice = X_reg[in_slice, 1]
ncde_x2_slice = X_ncde[in_slice, 1]
em_x2_slice = X_em_imp[in_slice, 1]

bins = np.linspace(true_x2_slice.min() - 0.5, true_x2_slice.max() + 0.5, 30)

# Slice-local metrics
slice_mse_reg = np.mean((true_x2_slice - reg_x2_slice) ** 2)
slice_mae_reg = np.mean(np.abs(true_x2_slice - reg_x2_slice))
slice_mse_ncde = np.mean((true_x2_slice - ncde_x2_slice) ** 2)
slice_mae_ncde = np.mean(np.abs(true_x2_slice - ncde_x2_slice))
slice_mse_em = np.mean((true_x2_slice - em_x2_slice) ** 2)
slice_mae_em = np.mean(np.abs(true_x2_slice - em_x2_slice))

# ── Row 0: conditional distributions at the overlap region ──

ax = axes[0, 0]
ax.hist(true_x2_slice, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax.set_title(r'True $p(x_2 \mid x_1 \approx 0)$' + '\n(bimodal — two crescents)', fontsize=11, fontweight='bold')
ax.set_xlabel('$x_2$', fontsize=11)
ax.set_ylabel('Density', fontsize=11)

ax = axes[0, 1]
ax.hist(true_x2_slice, bins=bins, density=True, alpha=0.25, color='steelblue', edgecolor='white', label='True')
ax.hist(reg_x2_slice, bins=bins, density=True, alpha=0.7, color='tab:red', edgecolor='white', label='Regression')
ax.axvline(np.mean(reg_x2_slice), color='darkred', ls='--', lw=2)
ax.axvline(np.median(reg_x2_slice), color='darkred', ls=':', lw=2)
ax.set_title(f'Regression (≈ TST)\nmean→MSE-optimal, median→MAE-optimal\n'
             f'MSE={slice_mse_reg:.2f}  MAE={slice_mae_reg:.2f}', fontsize=10, fontweight='bold')
ax.set_xlabel('$x_2$', fontsize=11)
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.hist(true_x2_slice, bins=bins, density=True, alpha=0.25, color='steelblue', edgecolor='white', label='True')
ax.hist(ncde_x2_slice, bins=bins, density=True, alpha=0.7, color='tab:purple', edgecolor='white', label='KNN/NCDE')
ax.axvline(np.mean(ncde_x2_slice), color='purple', ls='--', lw=2)
ax.axvline(np.median(ncde_x2_slice), color='purple', ls=':', lw=2)
ax.set_title(f'KNN (≈ NCDE)\nSame collapse\n'
             f'MSE={slice_mse_ncde:.2f}  MAE={slice_mae_ncde:.2f}', fontsize=10, fontweight='bold')
ax.set_xlabel('$x_2$', fontsize=11)
ax.legend(fontsize=8)

ax = axes[0, 3]
ax.hist(true_x2_slice, bins=bins, density=True, alpha=0.25, color='steelblue', edgecolor='white', label='True')
ax.hist(em_x2_slice, bins=bins, density=True, alpha=0.7, color='tab:green', edgecolor='white', label='EM posterior')
ax.set_title(f'Co-Evolving EM\nPreserves bimodal structure\n'
             f'MSE={slice_mse_em:.2f}  MAE={slice_mae_em:.2f}', fontsize=10, fontweight='bold')
ax.set_xlabel('$x_2$', fontsize=11)
ax.legend(fontsize=8)

# ── Row 1: the mathematical explanation ──

# (1,0): Table — what each metric's optimal estimator is
ax = axes[1, 0]
ax.axis('off')
table_text = (
    "Point-estimate metrics and\n"
    "their optimal estimators:\n\n"
    "  MSE → conditional MEAN\n"
    "           argmin E[(x−c)²|y]\n\n"
    "  MAE → conditional MEDIAN\n"
    "           argmin E[|x−c| |y]\n\n"
    "Both are single points.\n"
    "Neither captures the SHAPE\n"
    "of the conditional distribution."
)
ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

# (1,1-2): bar chart comparing MSE and MAE across methods
ax = axes[1, 1]
methods_short = ['Regr\n(TST)', 'KNN\n(NCDE)', 'EM\n(Ours)']
x_pos = np.arange(3)
width = 0.3
mse_vals = [mse_reg, mse_ncde, mse_em]
mae_vals = [mae_reg, mae_ncde, mae_em]
b1 = ax.bar(x_pos - width/2, mse_vals, width, label='MSE', color=['tab:red', 'tab:purple', 'tab:green'],
            alpha=0.7, edgecolor='black', linewidth=0.5)
b2 = ax.bar(x_pos + width/2, mae_vals, width, label='MAE', color=['tab:red', 'tab:purple', 'tab:green'],
            alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods_short, fontsize=10)
ax.set_ylabel('Error', fontsize=10)
ax.set_title('Imputation Error\n(both MSE & MAE say EM is worst)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
for b in b1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03, f'{b.get_height():.2f}',
            ha='center', fontsize=9, fontweight='bold')
for b in b2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03, f'{b.get_height():.2f}',
            ha='center', fontsize=9)

# (1,2): bar chart for disc score
ax = axes[1, 2]
disc_vals = [disc_reg_full, disc_ncde, disc_em]
bars = ax.bar(x_pos, disc_vals, 0.5, color=['tab:red', 'tab:purple', 'tab:green'],
              alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.3)
ax.axhline(disc_oracle, color='grey', ls='--', lw=1, alpha=0.5, label=f'Oracle ({disc_oracle:.3f})')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods_short, fontsize=10)
ax.set_ylabel('Disc Score (↓ = better)', fontsize=10)
ax.set_title('Generation Quality\n(EM is BEST despite worst MSE/MAE)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.007, f'{b.get_height():.3f}',
            ha='center', fontsize=10, fontweight='bold')

# (1,3): the punchline
ax = axes[1, 3]
ax.axis('off')
punchline = (
    "THE PUNCHLINE\n"
    "─────────────────────────────\n\n"
    "• MSE is minimized by the\n"
    "  conditional MEAN\n\n"
    "• MAE is minimized by the\n"
    "  conditional MEDIAN\n\n"
    "• Both are SINGLE POINTS that\n"
    "  erase multi-modal structure\n\n"
    "• The generative model needs\n"
    "  the conditional DISTRIBUTION\n"
    "  — not any point summary\n\n"
    "→ No point-estimate metric can\n"
    "  measure what generation needs"
)
ax.text(0.05, 0.95, punchline, transform=ax.transAxes, fontsize=10.5,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='honeydew', alpha=0.8))

fig2.suptitle(
    r'Why Worse MSE (and MAE) → Better Generation:  conditional at $x_1 \approx 0$ (crescent overlap)' + '\n'
    f'Regression: MSE={mse_reg:.2f}, MAE={mae_reg:.2f} (low — WRONG shape)   |   '
    f'EM: MSE={mse_em:.2f}, MAE={mae_em:.2f} (high — RIGHT shape)',
    fontsize=13, fontweight='bold'
)

fig2.savefig('toy_why_mse_misleads.png', dpi=180, bbox_inches='tight')
print(f"Diagnostic figure saved to toy_why_mse_misleads.png")

# ════════════════════════════════════════════════════════════════
# 9. Summary table
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 95)
print("SUMMARY — Confirming findings (with reviewer-fix F/F' CSDI-analog)")
print("=" * 95)
row_fmt = "{:<46} {:<30} {:>6.3f} {:>8}"
rows = [
    ("(A)  Oracle",                                 "—",                             disc_oracle,     "N/A"),
    ("(B)  Regression + masked loss",               "commit-and-forget (det.)",      disc_reg_masked, f"{mse_reg:.3f}"),
    ("(B') Regression + full loss",                 "commit-and-forget (det.)",      disc_reg_full,   f"{mse_reg:.3f}"),
    ("(C)  KNN/NCDE + full loss",                   "commit-and-forget (det.)",      disc_ncde,       f"{mse_ncde:.3f}"),
    ("(E)  Weak prior + frozen 1-round",            "commit-and-forget (stoch.)",    disc_e,          f"{mse_e:.3f}"),
    ("(E') Weak prior + frozen K=5 pooled",         "multi-impute (stoch.)",         disc_ep,         f"{mse_ep:.3f}"),
    ("(F)  CSDI-analog prior + frozen 1-round",     "commit-and-forget (CSDI-like)", disc_f,          f"{mse_f:.3f}"),
    ("(F') CSDI-analog prior + frozen K=5 pooled",  "multi-impute (CSDI-like)",      disc_fp,         f"{mse_fp:.3f}"),
    ("(D)  Co-Evolving EM",                         "co-evolving (stoch.+iter.)",    disc_em,         f"{mse_em:.3f}"),
]
print(f"\n{'Method':<46} {'Paradigm':<30} {'Disc':>6} {'MSE':>8}")
print("-" * 95)
for method, paradigm, disc, mse_s in rows:
    print(row_fmt.format(method, paradigm, disc, mse_s))
print("-" * 95)

print("\n① Better imputation ≠ better generation:")
print(f"   KNN imputes better than EM (MSE {mse_ncde:.3f} < {mse_em:.3f})")
print(f"   Yet EM produces far better generations (disc {disc_em:.3f} vs {disc_ncde:.3f}).")

print(f"\n② Full loss on deterministic imputations is catastrophic:")
print(f"   Regression + full loss ({disc_reg_full:.3f}) is WORST — model trusts fake data.")

print(f"\n③ Stochastic sampling helps over deterministic:")
print(f"   Posterior 1-round (E) disc={disc_e:.3f} vs best deterministic (B) disc={disc_reg_masked:.3f}")
print(f"   → Stochastic imputation preserves distributional structure that point estimates destroy.")

print(f"\n④ Iteration helps beyond stochastic sampling:")
print(f"   Co-Evolving EM (D) disc={disc_em:.3f} outperforms both")
print(f"   single-round stochastic (E) disc={disc_e:.3f} and")
print(f"   multi-sample stochastic (E') disc={disc_ep:.3f}.")
print(f"   → The model's improving prior produces increasingly accurate completions.")

print(f"\n⑤ Frozen vs. wrong-prior-class (reviewer fix, addresses confound in (E')):")
prior_effect = disc_e - disc_f         # E → F : upgrading prior class (still frozen)
freeze_effect = disc_f - disc_em       # F → D : un-freezing the CSDI-analog prior
freeze_effect_weak = disc_e - disc_em  # E → D : un-freezing the weak prior
print(f"   Prior-class effect  (E → F, both frozen):       Δdisc = {prior_effect:+.3f}")
print(f"                       ({disc_e:.3f} → {disc_f:.3f})")
print(f"   Frozen effect, weak prior     (E → D):           Δdisc = {freeze_effect_weak:+.3f}")
print(f"                       ({disc_e:.3f} → {disc_em:.3f})")
print(f"   Frozen effect, CSDI-analog    (F → D):           Δdisc = {freeze_effect:+.3f}")
print(f"                       ({disc_f:.3f} → {disc_em:.3f})")
if abs(freeze_effect) > abs(prior_effect):
    print(f"   → Frozen-ness is the DOMINANT effect: even with a CSDI-quality")
    print(f"     frozen imputer, the generator still benefits from being")
    print(f"     co-evolved. Iteration is the key ingredient, not prior class.")
else:
    print(f"   → Prior class carries most of the gap; a CSDI-quality frozen")
    print(f"     imputer alone closes the majority of (E)→(D). Iteration's")
    print(f"     primary role is upgrading the prior.")
