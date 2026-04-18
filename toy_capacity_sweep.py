"""
Capacity sweep for the generator GMM.

We want the Oracle disc score to approach 0.5 so that (D) can visibly
match the truth. A thin 1D spiral/arm needs many small Gaussians.
Sweep K_GEN ∈ {15, 30, 60, 100} on swirl & pinwheel_5, reporting:
  oracle disc   (ceiling)
  F     disc   (frozen low-capacity imputer → high-capacity generator)
  D     disc   (co-evolving with high-capacity shared model)
"""
import time
import numpy as np
from toy_experiment_cherrypick import (
    DISTRIBUTIONS, corrupt, fit_gmm, posterior_sample_gmm, disc_score, co_evolving_em
)

SEED = 42
N_TRAIN = 2000
N_EVAL = 1500
N_ITERS = 12
K_IMP = 3
CONFIGS = [
    ("swirl",       0.60),
    ("pinwheel_5",  0.70),
]
K_GEN_GRID = [15, 30, 60, 100]


def once(dname, p_miss, K_gen, seed):
    rng = np.random.default_rng(seed)
    X_clean = DISTRIBUTIONS[dname](N_TRAIN, rng)
    X_test = DISTRIBUTIONS[dname](N_EVAL, rng)
    X_corr, mask = corrupt(X_clean, p_miss, rng)

    oracle = fit_gmm(X_clean, K_gen, seed=seed)
    g_oracle = oracle.sample(N_EVAL)[0]

    imp, _ = co_evolving_em(X_corr, mask, K_IMP, N_ITERS, rng, seed=seed)
    X_F = posterior_sample_gmm(X_corr, mask, imp, rng)
    F = fit_gmm(X_F, K_gen, seed=seed)
    g_F = F.sample(N_EVAL)[0]

    D, _ = co_evolving_em(X_corr, mask, K_gen, N_ITERS, rng, seed=seed)
    g_D = D.sample(N_EVAL)[0]

    return dict(
        oracle=disc_score(X_test, g_oracle, seed=seed),
        F=disc_score(X_test, g_F, seed=seed),
        D=disc_score(X_test, g_D, seed=seed),
    )


def main():
    print(f"\n{'dist':<14} {'p':>5}  {'K_gen':>5}  {'Oracle':>7} {'F':>7} {'D':>7}  {'F-D':>7}  {'D-O':>7}  {'sec':>6}")
    print("-" * 72)
    for dname, p in CONFIGS:
        for K in K_GEN_GRID:
            t0 = time.time()
            r = once(dname, p, K, SEED)
            dt = time.time() - t0
            print(f"{dname:<14} {p:>5.2f}  {K:>5}  {r['oracle']:>7.3f} {r['F']:>7.3f} {r['D']:>7.3f}  "
                  f"{r['F']-r['D']:>+7.3f}  {r['D']-r['oracle']:>+7.3f}  {dt:>6.1f}")


if __name__ == "__main__":
    main()
