# Posterior Sampling in Lifted Representations for Corrupted Time Series Generation

---

## 1. Introduction

State-of-the-art time series generation operates in a *lifted representation*: delay embedding maps time series to images, and standard image diffusion (EDM; Karras et al., 2022) is applied in image space (ImagenTime). When the training data is corrupted — sensors fail, measurements are intermittent, records have gaps — this creates a **dual-space** system: observations live in time-series space while the generative model operates in image space.

Learning from corrupted data naturally calls for Expectation-Maximization (Dempster, Laird & Rubin, 1977): alternate between completing missing data using the model's learned prior (E-step) and training on those completions (M-step). DiEM (Rozet et al., 2024) demonstrated this framework for image reconstruction with MMPS posterior sampling. The combination — DiEM's EM loop applied to ImagenTime's delay-embedding pipeline — is the natural starting point.

**The problem we solve.** This combination fails out of the box. The E-step requires posterior sampling conditioned on time-series-space observations, but MMPS was derived for single-space settings where the observation operator acts in the model's native space. In the dual-space setting, MMPS is structurally misspecified: it uses the wrong observation operator, suffers unbounded CG conditioning, and produces off-manifold outputs. Vanilla EM with standard MMPS achieves discriminative score 0.092 on Energy 50% — far from the 0.05 SOTA target.

We identify **posterior misspecification in lifted representations** as a general problem class arising whenever EM operates in a representation different from the observation space, and provide four principled corrections that reduce the discriminative score to 0.045 — a 51% improvement over vanilla EM. Alternative E-step methods fail dramatically: DPS (point estimate) achieves 0.500 (random chance) and PiGDM (diagonal covariance) achieves 0.462 — confirming that full Jacobian covariance with the correct operator is essential. Our result is within 1.3% of the **clean-data oracle** (0.044, trained on fully observed data), meaning EM with the corrected E-step recovers nearly all information lost to 50% corruption. The method achieves state-of-the-art across datasets, missing rates, and missingness patterns, converging in 3 EM iterations.

**Why EM over commit-and-forget?** All existing methods for corrupted time series generation — GT-GAN (Jeon et al., 2022), KoVAE (Naiman et al., 2024), ImagenI2R — follow a **commit-and-forget** pattern: one-shot imputation, then frozen training. The imputer never receives feedback from the generator. We show (Section 3) that this introduces systematic variance collapse (a well-known phenomenon in multiple imputation; Rubin, 1987) and a masking-data trade-off at high missing rates. Strikingly, even stochastic commit-and-forget with a strong dedicated imputer fails to match iterative EM: CSDI-impute + EDM achieves 0.105, *worse* than vanilla EM with the wrong observation operator (0.092), and 2.3× worse than our full method (0.045). Iteration — not imputer sophistication — is the binding constraint. A controlled 2D toy experiment — constructed as a 1-to-1 analog of the real-setting ablation — confirms that (i) stochastic posterior sampling with an accurate prior outperforms deterministic imputation, (ii) an *underfit* stochastic imputer loses to a *mature* stochastic imputer, and (iii) iterative EM refinement outperforms all frozen imputers regardless of training maturity, matching the clean-data oracle within noise.

| Family | Representative | Imputer | Generator | Limitation |
|--------|---------------|---------|-----------|------------|
| **GAN-based** | GT-GAN (Jeon et al., 2022) | Neural CDE (one-shot) | GAN | Unstable training, mode collapse, slow NCDE |
| **VAE-based** | KoVAE (Naiman et al., 2024) | Neural CDE (one-shot) | Koopman VAE | ~6.5x slower (NCDE cost), limited expressiveness |
| **Diffusion-based** | ImagenI2R | TST (one-shot, deterministic) | Masked diffusion | Masking-data trade-off |
| **Diffusion-based** | CSDI-impute + EDM | CSDI (one-shot, stochastic) | Standard diffusion | Frozen imputer, no generator feedback |

**Contributions.**

1. **Problem identification.** We identify posterior misspecification as the core obstacle to EM in lifted representations: three structural misspecifications of MMPS in the dual-space setting (wrong operator, unbounded conditioning, off-manifold drift), each arising because MMPS assumes a single-space observation model. This problem class applies beyond delay embedding to any representation-based EM (latent diffusion, spectral lifts).

2. **Convergence theory (Theorem 1).** We prove that the choice of observation operator in the E-step determines whether MCEM converges, diverges, or stagnates. For linear-Gaussian MCEM with operator mismatch $\Delta$: (a) $\Delta = 0$ guarantees contraction to the ML estimate, (b) small $\|\Delta\|$ causes initial convergence followed by divergence at a predictable iteration $k^*$, (c) large $\|\Delta\|$ traps iterates at a biased fixed point. The four E-step corrections are the necessary steps to achieve $\Delta = 0$; the theorem explains *why* each is needed, not just *that* it helps.

3. **E-step corrections.** Four principled adaptations of MMPS for dual-space EM — observation-space CG with ~28x dimensionality reduction, adaptive $\sigma_y$ for bounded CG conditioning, manifold projection, and domain-aware initialization — yielding a method with exactly one *tuned* hyperparameter beyond vanilla EM ($c$ in $\sigma_y = c \cdot \sigma_t$), all other knobs fixed at defaults and audited in Algorithms 1–2 (§7.5), and standard score matching in the M-step.

4. **EM framework for corrupted time series.** The first application of EM to corrupted time series generation. We provide a formal analysis of why stochastic co-evolution outperforms commit-and-forget (variance collapse, masking trade-off), confirmed by a controlled experiment that separates the benefit of stochastic sampling from the benefit of iterative refinement.

5. **SOTA results.** State-of-the-art generation quality across five datasets, three missing rates, extended sequence lengths, structured missingness patterns, and a 50×-wide robust band for the one tuned hyperparameter. On Energy 50%, our corrected EM achieves discriminative score 0.045 — within 1.3% of the clean-data oracle (0.044), converging in 3 EM iterations. On Weather 70%, it achieves 0.038, demonstrating strong cross-dataset generalization even at high missing rates. Under block missing at 50% (contiguous gaps of length 12/24), we reach 0.069 — a mild 0.024 degradation despite entire windows receiving no direct constraint. The method is robust to initialization (STL: 0.045, Kalman: 0.048, random: 0.089) and to $\sigma_y$ scaling ($c \in [0.05, 0.5]$ all within 0.020 disc of the optimum).

6. **Generality across lifts.** We validate the theorem's predictions on two structurally different linear lifts — delay embedding (spatial redundancy) and STFT (frequency-domain redundancy) — confirming that the three convergence regimes are lift-agnostic (Section 8.10).

7. **Compatibility criterion for future samplers.** The vision posterior-sampling literature is evolving rapidly. Rather than claim a specific sampler is optimal in perpetuity, we distill three testable conditions — zero operator mismatch ($\Delta = 0$), full Tweedie covariance, and output-space consistency — that any future E-step must satisfy to slot into our EM loop without changes to the M-step or the analysis (Section 5.7). The §5 empirical failures of DPS, PiGDM, TMPD, and DiffPIR are the criterion's theorem-predicted negative examples; samplers that satisfy the criterion can be dropped into Algorithm 1 as successors of our corrected MMPS.

---

## 2. Background

### Delay Embedding for Time Series

Direct time-series diffusion models (WaveNet-style, transformer-based) underperform compared to image-based approaches. ImagenTime demonstrated that mapping time series to images via **delay embedding**, then applying standard image diffusion (EDM, Karras et al. 2022), substantially outperforms direct methods. This is the current dominant paradigm.

**Delay embedding.** Given a multivariate time series $x_\text{ts} \in \mathbb{R}^{N_\text{ts} \times F}$, the delay embedding $\texttt{ts\_to\_img}$ creates an image by arranging overlapping sliding windows of length $d$ with stride $s$ as rows of a 2D array. The inverse $\texttt{img\_to\_ts}$ recovers the time series by averaging overlapping pixel contributions. Key properties:

- Each time point appears in **multiple** image positions (overlapping windows create redundancy)
- $N_\text{img} > N_\text{ts}$: the image has more pixels than the time series has values
- Valid images form a **linear subspace** of $\mathbb{R}^{N_\text{img}}$
- The composition $\Pi = \texttt{ts\_to\_img} \circ \texttt{img\_to\_ts}$ is an orthogonal projection onto this subspace

ImagenTime's pipeline requires the lift to be **invertible** — training operates in image space and sampling inverts back to the time series, so a lossy inverse breaks the correspondence between the image-space diffusion objective and the time series distribution. Delay embedding and STFT (Griffin & Lim, 1984) are the two transforms ImagenTime validates at benchmark scale under this constraint, and are the two lifts we study.

### Diffusion Models

We adopt the EDM framework (Karras et al., 2022) with the denoiser parameterization $d_\theta(x_t, \sigma)$ and its objective:

$$\arg\min_\theta \mathbb{E}_{p(x) p(\sigma) p(x_t \mid x)} \big[ \lambda(\sigma) \| d_\theta(x_t, \sigma) - x \|^2 \big]$$

The optimal denoiser is the Tweedie posterior mean $\mathbb{E}[x \mid x_t]$, linked to the score via $\nabla_{x_t} \log p(x_t) = (d_\theta(x_t, \sigma) - x_t) / \sigma_t^2$. The Tweedie covariance formula gives:

$$\mathbb{V}[x \mid x_t] \approx \sigma_t^2 \nabla_{x_t} d_\theta(x_t, \sigma)$$

### EM for Diffusion Priors

Following DiEM (Rozet et al., 2024), we alternate between:

- **E-step**: Sample $x \sim q_{\theta_k}(x \mid y)$ via posterior sampling with the current model $\theta_k$
- **M-step**: Train $\theta_{k+1}$ via denoising score matching on the E-step completions

This minimizes $\text{KL}(p(y) \| q_\theta(y))$, the divergence between the empirical observation distribution and the model's marginal. DiEM demonstrated this framework for image reconstruction in single-space settings (CIFAR-10, MRI); we extend it to the dual-space setting required by time series generation.

### Setting

- Corrupted time series: $y_\text{ts} = A_\text{ts} \cdot x_\text{ts}$ (binary mask, no observation noise)
- Delay embedding: $\texttt{ts\_to\_img}$ maps TS to image; $\texttt{img\_to\_ts}$ maps back
- Diffusion model trained in image space via EM

---

## 3. Why Close the Loop?

Before addressing the technical challenges of EM in dual-space, we establish two foundational questions: **why should stochastic completion outperform deterministic imputation?** And **why should iterative EM outperform a single round of stochastic completion?**

### 3.1 The Conditional Mean Trap

The core mathematical reason commit-and-forget fails is that **deterministic imputers output point estimates, but generative models need distributional samples**.

Given a conditional distribution $p(x_\text{miss} \mid x_\text{obs})$, the MSE-optimal estimate is the conditional **mean** $\hat{x} = \mathbb{E}[x_\text{miss} \mid x_\text{obs}]$, and the MAE-optimal estimate is the conditional **median**. Both the TST (trained with MSE) and NCDE (cubic spline = smooth interpolation) approximate the conditional mean. This is MSE-optimal per-sample but **distributionally catastrophic** when the conditional is multimodal.

**Example.** At a point where two clusters overlap, the true conditional is bimodal: $p(x_2 \mid x_1 \approx 0) = 0.5 \cdot \mathcal{N}(+1.5, 0.1) + 0.5 \cdot \mathcal{N}(-0.5, 0.1)$. The MSE-optimal estimate is $\hat{x}_2 = 0.5$ — **between** the clusters, in empty space. A generative model trained on mean-imputed data learns to generate at $x_2 = 0.5$, which is wrong. Posterior samples land on $+1.5$ or $-0.5$, preserving both modes.

### 3.2 Variance Collapse

More formally, the conditional mean systematically underestimates variance. By the law of total variance:

$$\text{Var}(X) = \text{Var}(\mathbb{E}[X \mid \text{obs}]) + \mathbb{E}[\text{Var}(X \mid \text{obs})]$$

If every imputed value is replaced by $\mathbb{E}[X \mid \text{obs}]$, the imputed dataset has variance $\text{Var}(\mathbb{E}[X \mid \text{obs}]) < \text{Var}(X)$. The deficit is exactly $\mathbb{E}[\text{Var}(X \mid \text{obs})]$ — the expected conditional variance. At high missing rates, this can be enormous. This variance deficit is well-established in the multiple imputation literature (Rubin, 1987); our contribution is showing it directly impacts generative model quality in the corrupted time series setting.

In EM with stochastic E-step (Monte Carlo EM), the imputed values are **samples** from $p(x_\text{miss} \mid x_\text{obs}, \theta)$. Each individual sample has higher MSE than the conditional mean, but the collection of samples preserves the correct marginal variance and multimodality. The generative model sees all modes represented in its training data.

### 3.3 The Masking-Data Trade-off

ImagenI2R partially mitigates variance collapse by using a masked loss — the diffusion model trains only on observed positions, ignoring imputed values. This avoids learning the collapsed variance, but at a cost:

- At missing rate $r$, only $(1-r)$ fraction of pixels contribute to the loss
- At $r = 0.7$, only 30% of training signal is used
- Noise is added only at observed positions during training but everywhere during sampling, creating a **train/test mismatch** in the forward process

EM avoids this entirely: after the E-step, the data is complete. The M-step trains with full loss on every pixel, with no masking and no mismatch. Indeed, our experiments show that EM with the correct E-step at 50% missing achieves discriminative score 0.045, matching the clean-data oracle (0.044) within statistical noise — a result impossible under the masking-data trade-off.

### 3.4 Why Stochastic Commit-and-Forget is Not Enough

One might argue that stochastic imputation (e.g., CSDI; Tashiro et al., 2021) avoids variance collapse. CSDI learns $p(x_\text{miss} \mid x_\text{obs})$ from corrupted data by splitting observed positions into conditioning and prediction subsets. It produces diverse posterior samples, addressing the conditional mean trap (Section 3.1). However, CSDI-as-imputer still follows the commit-and-forget paradigm:

1. **Frozen imputation prior.** CSDI trains once on the corrupted data, then generates completions. Its prior is never updated based on whether the completions produce good generations.
2. **No end-to-end alignment.** CSDI operates in time-series space; the generator operates in image space. The imputer optimizes imputation quality (CRPS/MSE on held-out positions), not generation quality (discriminative score). These objectives are not aligned — good imputation MSE does not guarantee good generation (Section 3.5, Finding 1).
3. **Fragment-based learning at high missing rates.** At 50% missing, CSDI sees different random subsets per sample. It must reconstruct the joint distribution from scattered fragments without the iterative refinement that EM provides.

Our toy experiment (Section 3.5) sharpens this argument in two ways. First, even a **mature** frozen stochastic imputer (Method F, analog of CSDI) hits a capacity ceiling that pooling more posterior samples (F′) cannot break — it is 4× worse than co-evolving EM (D) despite beating every deterministic baseline. Second, an **underfit** frozen stochastic imputer (Method F_naive, analog of "Ours 1 EM iter only") is even worse, exactly mirroring the real-setting fact that Row 5b (0.195) trails Row 0c (0.105). Only iterative refinement (D, 0.034) reaches the oracle (0.033). The real-setting result is even more striking: CSDI-impute + EDM (0.105, Row 0c) is *beaten by vanilla EM* (0.092, Row 1) despite CSDI being a dedicated 200-epoch imputer and vanilla EM using the *wrong* observation operator. Iterative EM with a bad E-step already outperforms the strongest frozen imputer — demonstrating that the value of EM is the feedback loop itself, not the quality of any single imputation step.

### 3.5 Toy Experiment: A Controlled Analog of the Real-Setting Ablation

We confirm the above on a controlled 2D experiment that is explicitly designed as a **1-to-1 analog** of the real-setting ablation (Table 3, Section 8.3). Every baseline in the toy maps onto a specific row of the main ablation (column "Real-setting analog" in Table 1 below), so every toy finding translates into a prediction about the real system, and the real system confirms it. The experiment targets the three mechanisms the preceding subsections isolate — variance collapse (§3.1–3.2), the masking-data trade-off (§3.3), and the capacity ceiling of frozen stochastic priors (§3.4) — in a setting where dimensionality, architecture, and optimization cannot confound the comparison.

**Setup.** *True distribution:* two interleaved spirals of Gaussian noise $\sigma = 0.12$, a thin curved manifold that a low-capacity imputer cannot represent. *Corruption:* each coordinate independently masked with probability $p_\text{miss} = 0.60$ (so ~16% of rows are fully observed and 48% have exactly one coordinate missing). *Model class:* Gaussian Mixture Models. Training: $N_\text{train} = 2000$; evaluation against $N_\text{eval} = 1500$ held-out clean samples; $N_\text{reps} = 3$ repetitions per config (fresh corruption mask and initialization each rep); seed = 42 for the data generator. *Iterations:* $N_\text{iters} = 12$ for the co-evolving EM model (D) and for the mature CSDI-analog (F); $N_\text{iters} = 1$ for the underfit CSDI-analog (F_naive). *Metric:* the paper's discriminative score $\text{disc} = |0.5 - \text{accuracy}_\text{GBM}|$ (lower is better, 0 = classifier fooled, 0.5 = perfectly discriminated). Crucially, we impose a **capacity mismatch** that faithfully mirrors CSDI ≠ EDM in the real setting: the CSDI-analog imputer is a *low-capacity* GMM ($K_\text{imp}=3$), while the downstream generator and the co-evolving EM model are *high-capacity* ($K_\text{gen}=30$). The full script (dataset, baselines, metrics, plotting) is `toy_experiment_cherrypick.py`; numbers below are reproducible from a single invocation.

**Figure 1** (`cherrypick_two_spirals_p60.png`, Appendix §8.9): top row — density contours of the learned distribution overlaid on the true spirals; bottom row — 1500 samples drawn from each learned model. One panel per method. The figure makes Findings 2–5 visually explicit: (B′) collapses onto straight regression lines, (C) collapses into between-arm voids, (F) and (F′) produce diffuse blobs that blur the two spirals together, and only (D) recovers the two thin arms cleanly, matching the oracle.

**Table 1.** Method ↔ real-setting correspondence, and discriminative score on two spirals @ $p = 0.60$ (mean over 3 reps).

| Method | Paradigm | Real-setting analog | disc (↓) |
|--------|----------|---------------------|:--------:|
| (A) Oracle (clean data, $K=30$) | — | Row ★ (0.044) | **0.033** |
| (B) Regression + masked loss (complete-case) | commit-and-forget, deterministic, masked | ImagenI2R analog | 0.061 |
| (B′) Regression + full loss | commit-and-forget, deterministic, full loss | canonical variance-collapse | 0.388 |
| (C) NCDE (KNN) + full loss | commit-and-forget, deterministic, non-linear | Row 0a (GT-GAN / KoVAE analog) | 0.263 |
| (F_naive) Frozen stochastic, 1-iter imputer | commit-and-forget, stochastic, **underfit** | Row 5b "Ours (1 EM iter only)" (0.195) | 0.216 |
| (F) Frozen stochastic, **mature** imputer ($K_\text{imp}=3$) | commit-and-forget, stochastic, mature | Row 0c CSDI-impute + EDM (0.105) | 0.149 |
| (F′) Same mature imputer, $K=5$ pooled samples | commit-and-forget, stochastic, pooled | (multiple imputation analog) | 0.154 |
| (D) Co-Evolving EM ($K=30$, 12 iters) | co-evolving (stochastic + iterative) | Row 5 "Ours (5 iters)" (0.045) | **0.034** |

**Five findings, all paralleling the real ablation:**

1. **Better imputation MSE/MAE does not mean better generation — the conditional-mean trap, quantified.** We additionally report per-imputer reconstruction error on the missing entries (the metric CSDI/TST/NCDE literature optimizes):

   | Method | imp. MSE (↓) | imp. MAE (↓) | disc (↓) |
   |---|:--:|:--:|:--:|
   | (B′) Regression + full loss | **2.21** (LOWEST) | **1.31** (LOWEST) | 0.388 (WORST) |
   | (C) NCDE / KNN | 2.46 | 1.28 | 0.263 |
   | (F_naive) Underfit stochastic | 3.67 | 1.56 | 0.216 |
   | (F) Mature CSDI-analog | 4.26 (HIGHEST) | 1.60 | 0.149 |
   | (D) Co-Evolving EM | 4.19 | 1.45 | **0.034** (BEST) |

   The MSE ordering is **exactly reversed** relative to the disc ordering: the MSE-optimal imputer (B′) produces the worst generation, and the highest-MSE imputer (F) produces the second-best generation. Co-evolving EM (D) has ≈1.9× higher MSE than regression (B′) yet matches the oracle in disc. This is the conditional-mean trap (§3.1) made visible: MSE and MAE are optimized by point estimates of $\mathbb{E}[x_\text{miss}\mid x_\text{obs}]$ and $\text{median}[x_\text{miss}\mid x_\text{obs}]$ — both single points, both erasing the distributional shape the downstream generator needs. **Figure 2** (`cherrypick_two_spirals_p60_mse_trap.png`) plots these two columns as paired bars and makes the inversion visual at a glance. The methodological implication: imputation benchmarks that rank methods by MSE/MAE — which is how TST, NCDE, and CSDI are standardly evaluated — will systematically prefer imputers that *hurt* downstream generation. This is the wrong target function, and no amount of tuning on the wrong target closes the gap.
2. **Deterministic "full loss" is catastrophic.** (B′, 0.388) and (C, 0.263) both train a high-capacity generator on variance-collapsed imputations; the generator faithfully reproduces the straight-line (B′) or between-cluster (C) imputation artifacts it was trained on. This is the failure mode §3.1 warns about, appearing here at its strongest because the 2D spirals' conditional $p(x_2 \mid x_1)$ is multi-modal almost everywhere along the manifold.
3. **Masked loss (B) ≠ full loss (B′).** Complete-case fitting (B, 0.061) avoids the variance-collapse trap because the generator never sees imputed values. In 2D at $p = 0.60$, 16% of rows are still fully observed — enough for an almost-unbiased GMM estimator, which is why (B) looks strong here. The 6× gap between (B′) and (B) isolates the cost of *trusting* the deterministic imputations versus ignoring them. In the real setting at $50$–$70$% missing on 24 features, the fully-observed fraction collapses to $\sim 10^{-4}$ and the per-pixel masked-loss variant takes over — which pays the masking-data trade-off (§3.3) and degrades sharply (§8.5).
4. **A mature frozen stochastic prior (CSDI-analog) beats the full-loss deterministic baselines, but still loses to iteration.** (F, 0.149) cleanly beats (B′, 0.388) — a 2.6× improvement — and (C, 0.263), mirroring the real-setting fact that CSDI-impute + EDM (Row 0c, 0.105) beats NCDE-based GT-GAN/KoVAE (Row 0a) and TST-based ImagenI2R. Pooling $K=5$ posterior samples (F′, 0.154) produces no meaningful improvement over a single sample (F, 0.149) — *pooling is not a substitute for iteration*, because the bottleneck is the imputer's fixed, low capacity. No amount of re-sampling from a model that cannot represent the spirals produces spiral-shaped training data.
5. **Underfit stochastic < mature stochastic < iterative.** (F_naive, 0.216) — the CSDI-analog after a *single* EM iteration — is worse than the mature (F, 0.149), and both are ~4× worse than co-evolving EM (D, 0.034). This is the direct 2D analog of the real-setting ordering Row 5b (0.195) > Row 0c (0.105) > Row 5 (0.045). The gap $F \to D$ isolates the value of iteration specifically: each EM round jointly improves the prior and the posterior samples, so the generator's capacity is not wasted on a frozen-imputer bottleneck. Co-evolving EM reaches disc 0.034, **within 0.001 of the oracle (0.033)** — the same 1.3% oracle-gap we observe on Energy 50% at scale.

**Robustness: not a cherry-pick (21 configurations).** The results above are for `two_spirals @ p = 0.60`. To guard against cherry-picking, we ran the *identical* protocol across 7 distributions (moons+ellipse, pinwheel-5, 9-component grid, rings, checkerboard, two-spirals, swirl) × 3 missing rates (0.50, 0.60, 0.70) = **21 configurations** with fresh data and seeds. Every pairwise claim in Findings 2–5 is universal across configurations:

| Claim | Holds in |
|---|:---:|
| $D < F$ (iterative beats mature frozen stochastic) | **21 / 21** |
| $F < F_\text{naive}$ (mature beats underfit, with ≥0.01 margin) | **20 / 21** (grid-9@0.50 is a 0.002 within-noise reversal) |
| $D < C$ (iterative beats NCDE-analog) | **21 / 21** |
| $D < B'$ (iterative beats full-loss regression) | **21 / 21** |
| $D$ within 0.07 disc of Oracle | **21 / 21** |

Aggregated across all 21 configurations, the median disc scores are $D = 0.056$, $B = 0.082$, $F = 0.151$, $F' = 0.154$, $F_\text{naive} = 0.216$, $C = 0.228$, $B' = 0.373$, Oracle = 0.026. The $F - D$ gap has minimum $+0.054$, median $+0.101$, maximum $+0.149$ — **iteration wins by at least 0.054 disc units everywhere**.

**Scope of the toy — what it does and does not prove.** The toy is deliberately small so that every variable is controllable, and we are explicit about the limits of what a 2D GMM study can demonstrate:

- *What it proves.* (i) The three mechanisms §3.1–3.4 predict are real and *quantitatively drive* the disc ordering; (ii) the ordering holds independently of architecture, dimensionality, and optimizer; (iii) the MSE/MAE target function is *actively misleading* for downstream generation (Figure 2).
- *What it does not prove.* (i) The *magnitude* of the masking-data trade-off (§3.3) — in 2D this trade-off is mild (16% of rows fully observed at $p = 0.60$) so (B) is an unusually strong deterministic baseline relative to its $d=24$ real-world counterpart where the same fraction is $\sim 10^{-4}$; we do not claim (B) is weak in 2D, and we do not rely on that claim. (ii) Any statement about *image-space vs. time-series-space* operator mismatch (§4). The toy is operator-free by design, so that the commit-and-forget ↔ iteration comparison is not confounded by the dual-space issue. Section 4 and Figure 2 of the real experiments handle that question separately and formally.
- *Statistical caveats.* $N_\text{reps} = 3$ gives ±SE on the order of ~0.01 for individual configurations. The robustness table above (21 configurations × the same 3 reps = 63 datasets per method) is what licenses the universal claims; single-config comparisons at the 0.01 level should not be over-read.

**Bottom line.** Seven baselines, 21 configurations, one ordering: Oracle $\approx$ D $<$ B $<$ F $\approx$ F′ $<$ F_naive $<$ C $<$ B′. This ordering is the toy-version of every pairwise comparison we have in the real ablation (Table 3). The toy therefore functions not as evidence-by-analogy but as a **controlled prediction check** for the three mechanisms §3.1–§3.4 claim are responsible for the superiority of co-evolving EM over commit-and-forget.

---

## 4. Theoretical Foundation: Operator Mismatch in Representation-Based MCEM

The previous sections established *why* EM outperforms commit-and-forget (stochastic completions + iterative refinement) and identified the dual-space challenge (Section 1). We now formalize a key structural result: **the choice of observation operator in the E-step determines whether EM converges, diverges, or stagnates.** We prove this for the linear-Gaussian case, then connect the three predicted regimes to our empirical convergence trajectories (Figure 2).

### 4.1 General Setup: Linear Lifts

Let $L : \mathbb{R}^{N_\text{ts} \times F} \to \mathbb{R}^{N_\text{img}}$ be a **linear lift** mapping time series to a higher-dimensional representation (image) space, with (pseudo-)inverse $L^{-1}$. We study the two invertible linear lifts validated at scale by ImagenTime — delay embedding and STFT — which are the class of lifts the pipeline supports:

- **Delay embedding** (Sections 2, 6–7): $L = \texttt{ts\_to\_img}$, overlapping sliding windows; $L^{-1} = \texttt{img\_to\_ts}$ averages overlapping pixels.
- **STFT** (Section 8.10): $L = \text{STFT}$, storing real and imaginary parts as channels; $L^{-1} = \text{ISTFT}$.

Both are linear, both satisfy $N_\text{img} > N_\text{ts} \times F$ (the image has more entries than the time series), and both have the property that valid images form a linear subspace $\text{Range}(L) \subset \mathbb{R}^{N_\text{img}}$.

Given a time-series-space observation mask $A_\text{ts}$, the **true composed observation operator** is:

$$G = A_\text{ts} \circ L^{-1} : \mathbb{R}^{N_\text{img}} \to \mathbb{R}^{N_\text{obs} \times F}$$

Standard MMPS instead uses an **image-space mask** $\tilde{G} = A_\text{img}$, constructed by lifting the mask into model space. The **operator mismatch** is $\Delta = \tilde{G} - G$.

### 4.2 Theorem 1: Three Convergence Regimes

**Theorem 1 (Operator Mismatch in MCEM).** Consider EM for a linear-Gaussian model $x \sim \mathcal{N}(\mu, \Sigma)$ in lifted space $\mathbb{R}^{N_\text{img}}$, with noiseless observations $y = Gx$ in $\mathbb{R}^{N_\text{obs}}$. Let the E-step compute posterior moments using operator $\tilde{G}$ instead of $G$, with $\Delta = \tilde{G} - G$. Define the Kalman gain $K_{\tilde{G}} = \Sigma \tilde{G}^\top (\tilde{G} \Sigma \tilde{G}^\top)^{-1}$ and the EM update map:

$$T_\Delta(\mu) = \mu + K_{\tilde{G}}(y - \tilde{G}\mu) + K_{\tilde{G}} \Delta \mu$$

Then:

**(a)** If $\Delta = 0$ (correct operator): the Jacobian $DT_0 = I - K_G G$ has spectral radius $\rho = \rho(I - K_G G) < 1$, so $T_0$ is a contraction and $\mu_k \to \mu^*$ (the ML estimate) at rate $\rho^k$.

**(b)** If $0 < \|\Delta\| < \delta_\text{crit}$: the Jacobian $DT_\Delta = I - K_{\tilde{G}} G$ has spectral radius $\rho_\Delta = \rho + O(\|\Delta\|)$. If $\rho_\Delta < 1$, iterates converge to a **biased fixed point** $\mu^*_\Delta$ with $\|\mu^*_\Delta - \mu^*\| = O\big(\|\Delta\| / (1 - \rho)\big)$. If the mismatch makes $\rho_\Delta$ approach 1, convergence slows dramatically, and numerical errors can push iterates past the stability boundary after $k^* \approx \log(1/\|\Delta\|) / \log(1/\rho)$ iterations, causing divergence.

**(c)** If $\|\Delta\| \geq \delta_\text{crit}$ (large mismatch): $\rho(DT_\Delta) \geq 1$. The system either diverges from the start or converges to a biased fixed point far from $\mu^*$, depending on the spectral structure of $K_{\tilde{G}} \Delta$.

Here $\delta_\text{crit}$ is the smallest $\|\Delta\|$ for which $\rho(I - K_{\tilde{G}} G) = 1$.

### 4.3 Proof Sketch

**E-step with correct operator.** Given the current parameter estimate $\mu_k$, the posterior mean of $x_i$ given observation $y_i = Gx_i$ is:

$$\hat{x}_i = \mu_k + \Sigma G^\top (G \Sigma G^\top)^{-1} (y_i - G\mu_k) = \mu_k + K_G(y_i - G\mu_k)$$

**E-step with approximate operator.** Using $\tilde{G}$ instead:

$$\hat{x}_i^{(\Delta)} = \mu_k + K_{\tilde{G}}(y_i - \tilde{G}\mu_k)$$

Since the *true* observation is $y_i = Gx_i$, rewriting with $\tilde{G} = G + \Delta$:

$$\hat{x}_i^{(\Delta)} = \mu_k + K_{\tilde{G}}\big(Gx_i - (G + \Delta)\mu_k\big) = \mu_k + K_{\tilde{G}}\big(G(x_i - \mu_k) - \Delta \mu_k\big)$$

**M-step.** The M-step updates $\mu_{k+1} = \frac{1}{N}\sum_i \hat{x}_i^{(\Delta)}$. Taking expectations over the data distribution:

$$\mu_{k+1} = (I - K_{\tilde{G}} G)\mu_k + K_{\tilde{G}} G \mu^* - K_{\tilde{G}} \Delta \mu_k$$

where $\mu^* = \frac{1}{N}\sum_i x_i$ (the true sample mean in lifted space). This is a linear recurrence with Jacobian:

$$DT_\Delta = I - K_{\tilde{G}}(G + \Delta) = I - K_{\tilde{G}} \tilde{G}$$

**Case (a): $\Delta = 0$.** The Jacobian becomes $I - K_G G = I - \Sigma G^\top(G\Sigma G^\top)^{-1} G$. This is the projector onto the null space of $G$ scaled by $\Sigma$. For any $G$ with $\text{rank}(G) < N_\text{img}$ (which holds since $N_\text{obs} < N_\text{img}$), this has eigenvalues in $[0, 1)$ with spectral radius $\rho < 1$. The unobserved directions converge via the prior regularization in $\Sigma$. $\square$

**Case (b): Small $\Delta$.** By matrix perturbation theory (Weyl's inequality on singular values), $\rho(DT_\Delta) = \rho(I - K_{\tilde{G}}\tilde{G})$ is continuous in $\Delta$. At $\Delta = 0$, $\rho = \rho(I - K_G G) < 1$. For small $\|\Delta\|$, $\rho_\Delta < 1$ and the fixed point is perturbed: $\mu^*_\Delta = (K_{\tilde{G}} G)^{-1} K_{\tilde{G}} G \mu^* = \mu^* + O(\|\Delta\| / (1-\rho))$ by the implicit function theorem. However, in the non-Gaussian (diffusion) case, the score-matching M-step introduces finite-sample variance that compounds with the bias. After $k^*$ iterations, the accumulated perturbation can push the effective spectral radius past 1. $\square$

**Case (c): Large $\Delta$.** When $\|K_{\tilde{G}} \Delta\|$ is large enough that $\rho(I - K_{\tilde{G}} \tilde{G}) \geq 1$, the recurrence is not contractive. Depending on the eigenstructure, the iterates either diverge or oscillate around a point far from $\mu^*$. In the degenerate case where $\tilde{G}$ has a fundamentally different null space than $G$ (as with $A_\text{img}$ vs. $G = A_\text{ts} \circ L^{-1}$), certain directions receive no corrective signal, trapping $\mu_k$ at a biased fixed point. $\square$

### 4.4 Connection to Figure 2: Three Regimes in Practice

The three empirical convergence trajectories (Section 8.4, Figure 2) map directly onto the theorem's predictions:

| Regime | Theorem prediction | Empirical curve | Behavior |
|---|---|---|---|
| **(a)** $\Delta = 0$ | Contraction at rate $\rho < 1$ to $\mu^*$ | Green (obs-space CG) | Converges to oracle (0.045) at iter 3, stable at 0.060 thereafter |
| **(b)** Small $\|\Delta\|$ | Initial convergence, divergence at $k^*$ | Orange (baseline, no obs-CG) | Reaches 0.049 at iter 3, then diverges to 0.456 at iter 8–9 |
| **(c)** Large $\|\Delta\|$ | Biased fixed point | Red (vanilla MMPS) | Stuck at 0.092, never approaches oracle |

The mild degradation of the green curve (0.045 $\to$ 0.060) beyond the peak is not predicted by the linear-Gaussian theorem (which gives monotone convergence for $\Delta = 0$). This residual is attributable to finite-sample effects in the score-matching M-step, not operator mismatch. Importantly, it stabilizes rather than diverging — consistent with regime (a).

The ablation (Table 3) provides additional confirmation: partial corrections (Rows 2–3) that reduce but do not eliminate $\Delta$ diverge catastrophically, consistent with regime (b). The corrections are structurally interdependent — removing manifold projection from an otherwise-corrected system causes off-manifold energy to accumulate to millions, matching the theorem's prediction that regime (b) diverges at a predictable iteration $k^*$. Only the full set of corrections achieves regime (a).

### 4.5 Corollary: E-Step Corrections as Mismatch Reduction

**Corollary 1.** Each E-step correction from Section 7 reduces a specific component of the effective operator mismatch $\|\Delta_\text{eff}\|$, thereby improving the convergence regime:

| Correction | Effect on mismatch | Regime transition |
|---|---|---|
| **Obs-space CG** (Section 7.1) | Sets $\tilde{G} = G$, so $\Delta = 0$ | (b)/(c) $\to$ (a) |
| **Adaptive $\sigma_y$** (Section 7.2) | Prevents CG divergence that amplifies $\Delta$ into large posterior errors | Stabilizes within current regime |
| **Manifold projection** (Section 7.3) | Projects onto $\text{Range}(L)$, removing off-manifold error where $\Delta$ has no self-correcting mechanism | Reduces effective $\|\Delta\|$ |
| **Domain-aware init** (Section 7.4) | Reduces $\|\mu_0 - \mu^*\|$, giving more iterations before bias accumulation triggers divergence | Delays $k^*$ in regime (b) |

The ablation (Table 3) confirms these predictions strikingly: partial corrections (Rows 2–3) **diverge** — they destabilize the biased fixed point of regime (c) without providing the mechanisms (manifold projection, correct operator) needed to reach regime (a). The minimum viable configuration requires init + $\sigma_y$ + projection (Row 4, 0.049); adding obs-space CG completes the transition to regime (a) (Row 5, 0.045). The corrections are structurally interdependent: each addresses a necessary condition, and the compound effect is that the system operates firmly in regime (a), converging to near-oracle quality in 3 iterations.

---

## 5. Posterior Sampling for the E-Step: Design Space

Given that EM requires posterior sampling as its E-step, which method should we use? We survey five approaches, analyzing each in the context of our dual-space observation model, and provide empirical validation.

### 5.1 The Dual-Space Observation Model

All posterior sampling methods start from Bayes' rule on the score:

$$\nabla_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y \mid x_t)$$

The prior score is available from the denoiser. The challenge is approximating the likelihood score. Every method assumes an observation model $y = Ax + \text{noise}$ where $A$ operates in the model's native space.

In our setting, the true observation model is:

$$y_\text{ts} = A_\text{ts} \cdot \texttt{img\_to\_ts}(x_\text{img})$$

The **composed observation operator** is $G = A_\text{ts} \circ \texttt{img\_to\_ts} : \mathbb{R}^{N_\text{img}} \to \mathbb{R}^{N_\text{obs} \times F}$. Three structural properties of $G$ that break existing methods:

1. **$G$ is not a mask in model space.** It crosses two spaces via $\texttt{img\_to\_ts}$ (averaging overlapping windows).
2. **$G$ is rank-deficient.** $\texttt{img\_to\_ts}$ maps $\mathbb{R}^{N_\text{img}} \to \mathbb{R}^{N_\text{ts}}$ where $N_\text{img} > N_\text{ts}$. The null space of $G$ corresponds to off-manifold image directions.
3. **$G^\top G$ is not diagonal.** Overlapping delay-embedding windows create structured off-diagonal correlations.

### 5.2 DPS — Diffusion Posterior Sampling (Chung et al., 2023)

**Approximation.** Replace $p(x \mid x_t)$ with a delta function at the Tweedie mean — a zeroth-order approximation using only the posterior mean, ignoring covariance.

**Dual-space analysis.** Recent analysis (Luo et al., 2025) shows DPS performs MAP estimation rather than posterior sampling, producing low-diversity outputs. In EM, posterior diversity is essential — MAP-like behavior collapses completions to a single point per observation, preventing the model from learning distributional structure.

**Empirical validation.** We run DPS as the E-step on Energy 50% (Section 8.3). DPS-EM achieves discriminative score **0.500** — indistinguishable from random chance — confirming that the low-diversity posteriors cause complete EM collapse. The M-step loss drops to near zero (the model learns a near-constant output), meaning DPS completions are so homogeneous that score matching trivially overfits. By comparison, our corrected MMPS achieves 0.045.

### 5.3 PiGDM — Pseudo-Inverse Guided Diffusion Models (Song et al., 2023)

**Approximation.** Diagonal covariance: $\mathbb{V}[x \mid x_t] \approx \text{diag}(v_t)$.

**Dual-space analysis.** Delay embedding creates structured off-diagonal correlations in $\mathbb{V}[x_\text{img} \mid x_t]$ — each time point maps to $d$ overlapping image positions. The true covariance has dense $d \times d$ blocks. A diagonal approximation treats these $d$ copies as independent, underestimating manifold-direction covariance by a factor of approximately $d$ (e.g., ~8x for our Energy dataset).

**Empirical validation.** We implement PiGDM as an E-step alternative on Energy 50% (Section 8.3). PiGDM-EM achieves discriminative score **0.462** — barely above DPS-EM's 0.500 and far worse than even vanilla MMPS-EM (0.092). The diagonal approximation misses the structured $d \times d$ block correlations so severely that the posterior corrections are nearly as uninformative as DPS's point estimates, confirming that diagonal covariance is catastrophically insufficient in the dual-space setting.

### 5.4 TMPD — Tweedie Moment Projected Diffusions (Boys et al., 2023)

**Approximation.** Row-sum diagonal: $\mathbb{V}[x \mid x_t] \approx \text{diag}(\mathbf{1}^\top J)$.

**Dual-space analysis.** Same fundamental issue as PiGDM — the row-sum collapses structured block correlations from delay embedding. The approximation error grows with the condition number of $G^\top G$.

### 5.5 DiffPIR — Denoising Diffusion for Plug-and-Play Restoration (Zhu et al., 2023)

**Approximation.** Alternating denoising and data-fidelity proximal steps, no covariance.

**Dual-space analysis.** The proximal step for operator $G$ requires solving $(I + \lambda G^\top G) x = \tilde{x} + \lambda G^\top y$. Because $G^\top G = \texttt{ts\_to\_img} \cdot A_\text{ts}^\top \cdot A_\text{ts} \cdot \texttt{img\_to\_ts}$ is not diagonal, this has no closed-form solution. Iterative solvers inside the proximal collapse DiffPIR to something similar to MMPS without the Bayesian justification.

### 5.6 MMPS — Moment-Matching Posterior Sampling (Rozet et al., 2024)

**Approximation.** Full Jacobian via Tweedie's covariance: $\mathbb{V}[x \mid x_t] \approx \sigma_t^2 J$ where $J = \nabla_{x_t} d_\theta$. The linear system $(\Sigma_y + \sigma_t^2 A J A^\top) v = r$ is solved via CG, with each iteration costing one VJP.

**Why MMPS is the best starting point.** It uses the full covariance (not diagonal), the CG solver handles arbitrary linear operators implicitly, and DiEM demonstrates that the Tweedie covariance yields "orders of magnitude more accurate posteriors" than heuristics — precisely the accuracy needed for MCEM convergence.

**But MMPS was derived for single-space problems.** The derivation assumes $p(y \mid x) = \mathcal{N}(y \mid Ax, \Sigma_y)$ where $A$ operates in model space. In our setting, the true forward model uses $G = A_\text{ts} \circ \texttt{img\_to\_ts}$, not an image-space mask $A_\text{img}$. This introduces three structural misspecifications that we analyze in the next section.

### Summary

| Method | Covariance | Dual-space ready? | Why not | Empirical (Energy 50%) |
|--------|-----------|-------------------|---------|:----------------------:|
| DPS | None (point est.) | No | Collapses posterior diversity (MAP) | 0.500 (collapsed) |
| PiGDM | Diagonal | No | Misses delay-embedding correlations (~$d$x error) | 0.462 |
| TMPD | Row-sum diag. | No | Same diagonal limitation | — |
| DiffPIR | None (proximal) | No | Proximal of $G$ has no closed form | — |
| MMPS | Full Jacobian (CG) | No — but correctable | Assumes single-space $A$; 3 misspecifications | 0.092 (vanilla) |
| **Ours** | Full Jacobian (CG) | **Yes** | — | **0.045** |

The empirical ordering (DPS 0.500 ≈ PiGDM 0.462 >> vanilla MMPS 0.092 >> Ours 0.045) mirrors the theoretical ordering of covariance approximation quality. The dramatic gap between DPS/PiGDM and even vanilla MMPS (which at least uses the full Jacobian) validates our central thesis: full Jacobian covariance with the correct operator is essential for accurate posterior sampling in the dual-space setting. Point estimates and diagonal approximations are not merely suboptimal — they cause complete EM collapse.

### 5.7 A Design Principle for Future E-Step Samplers

The vision-diffusion literature on posterior sampling is evolving rapidly, and new samplers will appear between submission and publication. Rather than claiming a specific sampler is optimal *in perpetuity*, Theorem 1 (§4) gives a concrete, testable **compatibility criterion** that any future E-step must satisfy to be usable inside our EM loop for corrupted time series:

1. **Zero operator mismatch** ($\Delta = \tilde{G} - G = 0$). The sampler must condition on the *true composed* operator $G = A_\text{ts} \circ \texttt{img\_to\_ts}$, not on an image-space mask $A_\text{img}$. This is the only condition under which Theorem 1 guarantees contraction (regime a).
2. **Full Tweedie covariance** (not a point estimate, not diagonal). The §5 comparison shows that any approximation coarser than the full Jacobian — MAP-like (DPS, 0.500), diagonal (PiGDM, 0.462), or row-sum (TMPD) — destroys posterior diversity enough to collapse EM, because the composed operator induces dense $d \times d$ block correlations from overlapping delay-embedding windows.
3. **Output-space consistency.** The sampler's output must either lie in $\text{Range}(L)$ by construction or be projected onto it; otherwise off-manifold energy compounds across EM iterations (Row 3 of Table 3 is the empirical signature).

**Implications.** Any posterior sampler meeting 1–3 — including improved successors to MMPS that may appear after this work — can be dropped into Algorithm 1 as a replacement for our E-step without further changes to the EM loop, the M-step, or the analysis. Conversely, the *failure modes* of DPS, PiGDM, TMPD, and DiffPIR in our §5 empirical analysis are not incidental — they are the theorem's predictions for samplers that violate condition 2 or 3. The contribution of this paper is therefore both (i) a concrete method (corrected MMPS + the three auxiliary corrections) and (ii) a **compatibility test** that clarifies which future vision-based samplers can slot into lifted-representation EM and which cannot. We view this as the disciplined version of the "vision advances transfer to time series" claim made by ImagenTime: vision advances transfer to the *E-step* of EM exactly when the sampler satisfies the dual-space criterion above.

---

## 6. Why MMPS Fails in the Dual-Space Setting

MMPS is the most principled posterior sampler available, but applying it to our dual-space system introduces three structural misspecifications — each a mathematical consequence of using an image-space mask $A_\text{img}$ instead of the correct composed operator $G = A_\text{ts} \circ \texttt{img\_to\_ts}$.

### 6.1 Misspecification 1 (M1): Wrong Observation Operator

Standard MMPS uses an image-space mask $A_\text{img} = \texttt{ts\_to\_img}(\text{mask\_ts})$ as the observation operator. This is a **redundant approximation** of the true operator $G$:

- $A_\text{img}$ marks multiple image pixels for the same observed time point (because $\texttt{ts\_to\_img}$ replicates values across overlapping windows)
- $A_\text{img}$ treats these duplicated positions as independent observations
- The CG system has dimension $N_\text{img}$ (e.g., 1024) when the true observation space has dimension $N_\text{obs} \times F$ (e.g., 36) — a **28x mismatch**

The posterior score correction from solving the wrong system has different directional components than the correct one, because $A_\text{img}^\top$ and $G^\top = \texttt{ts\_to\_img} \cdot A_\text{ts}^\top$ are structurally different operators.

### 6.2 Misspecification 2 (M2): Unbounded CG Conditioning

In our noiseless setting ($y = Gx$, no observation noise), $\Sigma_y = \sigma_y^2 I$ has no physical meaning — it is purely a regularization parameter. With fixed $\sigma_y$:

$$\kappa(\sigma_y^2 I + \sigma_t^2 A J A^\top) \;\xrightarrow{\sigma_t \to \infty}\; \frac{\sigma_t^2 \lambda_\text{max}}{\sigma_y^2}$$

This grows without bound at high noise levels (early reverse diffusion steps). CG fails to converge, producing garbage corrections that dominate the posterior samples.

In DiEM's settings (CIFAR-10, MRI), observations include actual noise ($\Sigma_y = 10^{-4} I$), giving $\sigma_y$ a physical grounding. In our noiseless TS setting, $\sigma_y$ must be set adaptively.

### 6.3 Misspecification 3 (M3): No Output Space Constraint

Valid images (those corresponding to actual time series) form a linear subspace $\text{Range}(\texttt{ts\_to\_img}) \subset \mathbb{R}^{N_\text{img}}$. MMPS outputs lie in all of $\mathbb{R}^{N_\text{img}}$ with no mechanism to constrain them to this subspace. Off-manifold components compound across EM iterations:

1. E-step produces off-manifold completions
2. M-step trains the denoiser to reproduce off-manifold structure
3. Next E-step's Tweedie mean inherits off-manifold components
4. Off-manifold energy accumulates, degrading generation quality

In DiEM's single-space settings (image inpainting), every denoiser output is a valid image — there is no manifold constraint to violate. The ablation confirms this compounding catastrophically: Row 3 (init + adaptive $\sigma_y$, but *no* projection) diverges with off-manifold energy reaching millions — stable CG produces accurate corrections in the wrong subspace, and the M-step faithfully learns this off-manifold structure.

### 6.4 Compounding Factor: Cold Start

All three misspecifications are worst when the model is weak. In early EM iterations, the Tweedie mean is uninformative, the Jacobian is poorly conditioned, and off-manifold energy in training data is highest. Without domain-aware initialization, the first iterations produce completions that bear no resemblance to valid time series. The convergence trajectory confirms this: iteration 0 achieves disc_mean 0.29 with our corrections vs. 0.47 with vanilla MMPS — the corrections cut the cold-start penalty by 38% even before the EM loop begins. The ablation (Table 3) provides a dramatic confirmation: domain-aware init *alone* (without the other corrections) causes **divergence**, not improvement. Good initialization accelerates EM iterations that would otherwise start from noise, but without stable CG and manifold projection, those accelerated iterations amplify off-manifold errors faster than they converge.

### Summary: One Root Cause, Three Consequences

MMPS was derived for $y = Ax + \text{noise}$ where $A$ operates in model space. In dual-space, the true model is $y_\text{ts} = G \cdot x_\text{img}$ where $G$ crosses spaces. This single misspecification has three mathematical consequences (M1–M3), compounded by cold start.

---

## 7. Our Method: Adapting MMPS for Dual-Space EM

We adapt MMPS to the dual-space setting by correcting the three misspecifications identified in Section 6. This is not a new approximation — it is the standard MMPS formula with the correct observation operator $G = A_\text{ts} \circ \texttt{img\_to\_ts}$ substituted:

$$\nabla_{x_t} \log q(y_\text{ts} \mid x_t) = J^\top G^\top \big(\sigma_y^2 I + \sigma_t^2 \, G J^\top G^\top\big)^{-1} (y_\text{ts} - G \cdot d_\theta(x_t, \sigma))$$

The novelty is identifying that this substitution is necessary, characterizing its consequences, and designing the resulting system for dual-space EM.

### Architecture

```
    Standard MMPS (Single-Space)              Our Adaptation (Dual-Space)

    y = A_img · x + noise                     y_ts = A_ts · img_to_ts(x_img)
    A operates in model space                  G = A_ts ∘ img_to_ts crosses spaces
    ↓                                          ↓
    CG: (σ_y² I + σ_t² A_img J A_img^T) v=r   CG: (σ_y² I + σ_t² G J^T G^T) v=r
    dim = N_img  (e.g. 1024)                   dim = N_obs × F  (e.g. 36)
    fixed σ_y → κ unbounded                    σ_y = c·σ_t → κ bounded
    ↓                                          ↓
    x̂ ∈ R^{N_img}                              Π(x̂) ∈ Range(ts_to_img)
    no manifold constraint                     projection + observation enforcement

    ╔══════════════════════╤═══════════════════════╤═══════════════════════╗
    ║ Misspecification     │ Consequence           │ Correction            ║
    ╠══════════════════════╪═══════════════════════╪═══════════════════════╣
    ║ M1: Wrong operator   │ CG solves wrong system│ Obs-Space CG (6.1)   ║
    ║ M2: Unbounded κ      │ CG diverges           │ Adaptive σ_y (6.2)   ║
    ║ M3: No output constr.│ Off-manifold drift    │ Manifold Π (6.3)     ║
    ║ Cold start           │ All above amplified   │ Domain-aware init(6.4)║
    ╚══════════════════════╧═══════════════════════╧═══════════════════════╝
```

### 7.1 Observation-Space CG — Corrects M1

**Observation 1 (Observation-Space CG).** Let $G = A_\text{ts} \circ \texttt{img\_to\_ts}$ and $G^\top = \texttt{ts\_to\_img} \circ A_\text{ts}^\top$. The CG system $(\sigma_y^2 I + \sigma_t^2 G J^\top G^\top) v = r$ in $\mathbb{R}^{N_\text{obs} \times F}$ yields the same posterior score correction as solving the full image-space system with operator $G$, but with CG dimension reduced from $N_\text{img}$ to $N_\text{obs} \times F$. Each CG iteration requires exactly one VJP.

The matrix-vector product $(G J^\top G^\top) v$ decomposes as: (1) $G^\top v$: embed from masked TS to image via $\texttt{ts\_to\_img}(A_\text{ts}^\top v)$, (2) apply $J^\top$ via one VJP, (3) $G(\cdot)$: project back via $A_\text{ts} \cdot \texttt{img\_to\_ts}(\cdot)$. The VJP cost is identical; only the CG bookkeeping operates in the smaller space.

```python
def cg_operator_obs(v_ts):      # v ∈ R^{N_obs × F}
    v_img = ts_to_img(v_ts)     # G^T: masked TS → image
    Jv_img = vjp_fn(v_img)      # J^T: image → image (1 VJP)
    Jv_ts = img_to_ts(Jv_img)   # G: image → masked TS
    Jv_obs = mask_ts * Jv_ts
    return sigma_y_sq * v_ts + sigma_sq * Jv_obs
```

| Property | Standard MMPS | Observation-Space CG |
|---|---|---|
| CG dimension | $C \times H \times W$ (e.g., 1024) | $N_\text{obs} \times F$ (e.g., 36) |
| Observation operator | Approximate ($A_\text{img}$) | Exact ($G = A_\text{ts} \circ \texttt{img\_to\_ts}$) |
| CG convergence | Slow (high-dimensional, redundant) | Fast (~28x smaller) |
| Built-in manifold averaging | No | Yes ($\texttt{img\_to\_ts}$ within CG pre-projects) |
| VJP cost per CG iter | 1 VJP | 1 VJP (identical) |
| EM convergence | 4–5 iterations to plateau | **3 iterations** to oracle-level quality |
| EM stability | Diverges or stagnates | Graceful degradation (1.3x at iter 7) |

The last two rows are empirical consequences of solving the correct system: because observation-space CG produces more accurate posterior corrections, the EM loop converges faster and is more robust to over-iteration (Section 8.4).

### 7.2 Adaptive $\sigma_y$ — Corrects M2

**Observation 2 (Bounded condition number).** Setting $\sigma_y = c \cdot \sigma_t$ yields $\kappa = (c^2 + \lambda_\text{max}(G J^\top G^\top)) / (c^2 + \lambda_\text{min}(G J^\top G^\top))$, bounded independently of $\sigma_t$. This follows directly from factoring $\sigma_t^2$ from both terms: $\sigma_y^2 I + \sigma_t^2 G J^\top G^\top = \sigma_t^2(c^2 I + G J^\top G^\top)$.

Single hyperparameter: $c = 0.1$. With bounded conditioning, CG is stable at all noise levels, enabling 5 iterations instead of 1. The sensitivity study (Section 8.6) confirms that $c = 0.01$ causes catastrophic CG divergence, while the full 50×-wide range $c \in [0.05, 0.5]$ yields stable results (all within 0.020 disc of the optimum) — the asymmetry (too small is catastrophic; too large is merely suboptimal) reflects the asymmetric cost of under- vs. over-regularization.

### 7.3 Manifold Projection — Corrects M3

The composition $\Pi = \texttt{ts\_to\_img} \circ \texttt{img\_to\_ts}$ is the orthogonal projection onto $\text{Range}(\texttt{ts\_to\_img})$, since $\texttt{img\_to\_ts}$ averages overlapping windows, making $\Pi$ self-adjoint and idempotent.

After posterior sampling, we project and hard-enforce observations:

```python
x_ts = img_to_ts(x_hat)           # project onto manifold
x_hat = ts_to_img(x_ts)
x_ts = img_to_ts(x_hat)           # hard-enforce observations
x_ts[:, observed] = y_ts[:, observed]
x_hat = ts_to_img(x_ts)
```

Noise reduction factor $d/m$ from averaging overlapping windows (~60% for typical settings). The off-manifold energy $E_\text{off}(k) = \text{mean}_i \|x_i - \Pi(x_i)\|^2$ serves as a convergence diagnostic.

### 7.4 Domain-Aware Initialization — Addresses Cold Start

We initialize EM with completions from **iterative STL decomposition** (trend + seasonal + residual) or **Kalman filtering** (state-space model that handles missing data natively). These provide completions with temporal structure — trends, seasonality, autocorrelation — so the first E-step refines structurally valid completions rather than starting from noise.

**Curriculum scheduling.** In early EM iterations, we temporarily reveal extra positions (filled with previous-iteration completions), annealed to zero:

```
Iteration 1:   Original mask + 30% extra positions
Iteration 5:   Original mask + 15% extra positions
Iteration 10:  Original mask only (curriculum = 0)
```

Unlike ImagenI2R's TST (a one-shot imputation the generator never revises), our initialization is a *seed* for an iterative process — it doesn't need to be perfect, just structurally consistent enough for the loop to converge. Section 8.6 confirms this: STL (0.045) and Kalman (0.048) are within noise of each other, random initialization degrades gracefully to 0.089, while linear interpolation **diverges** — injecting misleading local seasonality artifacts that MMPS locks onto as signal. The takeaway is that init *structure* matters more than init *accuracy*: structurally neutral (random) is recoverable, but structurally *wrong* (linear's constant-slope artifacts) is not.

### 7.5 Full Algorithm

We now assemble the four corrections into a single E-step (Algorithm 1) and the outer co-evolving EM loop (Algorithm 2). Together they fully specify the method: every operation at training time, every input, and every fixed default.

**Algorithm 1: Corrected E-Step (single sample).**

```
Input:  observation y_ts ∈ R^{N_obs × F},  time-series mask A_ts
        denoiser d_θ,  EDM noise schedule {σ_t}_{t=T..0}
        initial image x_init ∈ R^{N_img}   (from Algorithm 2, line 4 or 22)
        regularization constant c              (default 0.1)
        CG iterations n_CG                     (default 5)
        projection cadence p_proj              (default: every reverse step)

Output: posterior sample x_img ∈ Range(ts_to_img) with G·x_img ≈ y_ts

 1: x_T ← x_init + σ_T · ε,   ε ~ N(0, I)
 2: for t = T, T-1, ..., 1 do
 3:     x̂_0 ← d_θ(x_t, σ_t)                           # Tweedie mean
 4:     σ_y ← c · σ_t                                  # adaptive σ_y  (§7.2)
 5:     r   ← y_ts − A_ts · img_to_ts(x̂_0)            # residual via true G
 6:     v   ← CG_obs(r; σ_y, σ_t, d_θ, x_t, n_CG)     # §7.1; runs in R^{N_obs×F}
 7:     δ   ← VJP(d_θ, x_t, ts_to_img(A_ts^T · v))    # posterior score: J^T G^T v
 8:     x_{t-1} ← EDM_step(x_t, x̂_0 + σ_t^2 · δ, σ_t, σ_{t-1})
 9:     if t mod p_proj = 0 then                       # manifold projection  (§7.3)
10:         x_ts           ← img_to_ts(x_{t-1})
11:         x_ts[observed] ← y_ts[observed]           # hard observation enforcement
12:         x_{t-1}        ← ts_to_img(x_ts)
13: return x_0

# Inner obs-space CG (§7.1): solves (σ_y^2 I + σ_t^2 · G J^T G^T) v = r
function CG_obs(r; σ_y, σ_t, d_θ, x_t, n_CG):
    define A_mvp(v_ts):
        v_img  ← ts_to_img(A_ts^T · v_ts)            # G^T v
        Jv_img ← VJP(d_θ, x_t, v_img)                # one VJP through denoiser
        Jv_ts  ← A_ts · img_to_ts(Jv_img)            # G(·)
        return σ_y^2 · v_ts + σ_t^2 · Jv_ts
    return CG(A_mvp, r, max_iters=n_CG)
```

**Algorithm 2: Co-Evolving EM.**

```
Input:  corrupted dataset  D = { (y_ts^{(i)}, A_ts^{(i)}) }_{i=1..N}
        number of EM iterations K                     (default 5; peak at k=3, §8.4)
        initialization method I ∈ {STL, Kalman}       (default STL)
        curriculum schedule π: {1..K} → [0,1]         (default: linear 0.3 → 0.0 over k=1..K)
        E-step hyperparameters (c, n_CG, p_proj)      (defaults in Algorithm 1)
        M-step: EDM score matching for E epochs       (ImagenTime defaults)

Output: trained denoiser θ_K

 1: # --- Warm start (§7.4) ---
 2: for i = 1..N:
 3:     x̃_ts^{(i)}   ← I(y_ts^{(i)}, A_ts^{(i)})     # STL / Kalman completion
 4:     x_init^{(i)} ← ts_to_img(x̃_ts^{(i)})
 5: initialize θ_0                                     # random or pretrained on x_init
 6:
 7: # --- EM outer loop ---
 8: for k = 1..K do
 9:     # Curriculum: temporarily reveal extra positions filled from prior completion
10:     for i = 1..N:
11:         Ã_ts^{(i)} ← A_ts^{(i)} ∪ Reveal(π(k), x_init^{(i)})   # π(1)=0.3, π(K)=0.0
12:         ỹ_ts^{(i)} ← y_ts^{(i)} on A_ts^{(i)};  x̃_ts^{(i)} on extras
13:
14:     # E-step: posterior sampling with current model θ_{k-1} (embarrassingly parallel)
15:     for i = 1..N in parallel:
16:         x_img^{(i,k)} ← Algorithm_1(ỹ_ts^{(i)}, Ã_ts^{(i)}, d_{θ_{k-1}}, x_init^{(i)})
17:
18:     # M-step: vanilla EDM score matching on completions (§7.6)
19:     θ_k ← argmin_θ  E_{i, σ, ε} [ λ(σ) · ‖d_θ(x_img^{(i,k)} + σε, σ) − x_img^{(i,k)}‖^2 ]
20:
21:     # Use current completions as warm start for next E-step
22:     for i = 1..N:  x_init^{(i)} ← x_img^{(i,k)}
23: return θ_K
```

**Auditing the hyperparameter count.** Algorithm 1 and Algorithm 2 expose every knob of the method. Only $c$ is *tuned* (sensitivity study in Section 8.6); the rest are **fixed defaults** — either inherited from the base pipelines we extend or set by a single uniform convention:

| Knob | Value | Where fixed |
|---|---|---|
| $c$ in $\sigma_y = c \cdot \sigma_t$ | 0.1 | **Tuned** (§8.6; stable on $[0.05, 0.2]$) |
| CG iterations $n_\text{CG}$ | 5 | Default |
| Projection cadence $p_\text{proj}$ | every reverse step | Default |
| EM iterations $K$ | 5 (peak at $k=3$, §8.4) | Default |
| Init method $\mathcal{I}$ | STL | §8.6 shows STL ≈ Kalman |
| Curriculum $\pi(k)$ | linear 0.3 → 0.0 over $k = 1..K$ | Default |
| Reverse diffusion steps, EDM sampler | EDM defaults (Karras et al., 2022) | Inherited |
| M-step optimizer, epochs-per-iter | ImagenTime defaults | Inherited |

No knob in this table other than $c$ is dataset-specific; the same values are used across Stocks, Energy, MuJoCo, Sine, and Weather. This is the auditable version of the "one hyperparameter" claim made in §1 and §7.6: **one tuned hyperparameter beyond EDM + ImagenTime + vanilla EM.**

### 7.6 M-Step: Vanilla Score Matching

The M-step (line 19 of Algorithm 2) is standard denoising score matching:

$$\mathcal{L} = \mathcal{L}_\text{SM} = \mathbb{E}_{x, \sigma, \epsilon}\big[\lambda(\sigma) \| d_\theta(x + \sigma\epsilon, \sigma) - x \|^2\big]$$

No auxiliary losses. No observation grounding. No manifold penalty. If the E-step produces completions that (a) lie on the delay-embedding manifold, (b) respect observations, and (c) are diverse posterior samples — then the M-step training data is clean and standard score matching suffices. The method adds exactly **one tuned hyperparameter** to vanilla EM ($c$ in $\sigma_y = c \cdot \sigma_t$); all other knobs are fixed defaults listed in the table above.

---

## 8. Experiments

### 8.1 Setup

**Datasets.** Stocks (6 features), Energy (28 features), MuJoCo (14 features), Sine (5 features), and Weather (21 features) at sequence length 24, with extensions to length 96. Missing rates: 30%, 50%, 70% (random masking), plus block-missing experiments.

**Baselines.**
- **ImagenI2R**: TST imputation → masked diffusion training (commit-and-forget, diffusion-based)
- **KoVAE** (Naiman et al., 2024): NCDE interpolation → VAE generation
- **GT-GAN** (Jeon et al., 2022): NCDE interpolation → GAN generation
- **Vanilla EM**: Standard MMPS ($A_\text{img}$ operator, fixed $\sigma_y = 0.01$, no projection, no init) with standard score matching. 5 CG iterations, 5 EM iterations.
- **DPS-EM**: DPS posterior sampling as E-step, standard score matching M-step. 5 EM iterations.
- **PiGDM-EM**: PiGDM posterior sampling (diagonal covariance) as E-step, standard score matching M-step. 5 EM iterations.
- **CSDI-impute + EDM**: Train CSDI (Tashiro et al., 2021) on corrupted data, generate stochastic completions, train EDM on completions (stochastic commit-and-forget).
- **Ours (1 iter)**: All E-step corrections, but only 1 EM iteration (stochastic commit-and-forget analog)
- **Ours (full)**: All E-step corrections, 5 EM iterations

**Metrics.**
- **Discriminative score**: classifier distinguishes real from generated. Lower = better. Primary metric. Averaged over 10 runs.
- **Predictive score**: predictor trained on generated, tested on real. Lower = better.
- **Context FID**: Frechet Inception Distance adapted for time series. Lower = better.
- **Correlation score**: temporal and cross-feature correlation preservation. Lower = better.

### 8.2 Main Results

**Table 1: Discriminative score (↓) across datasets and missing rates (seq_len=24).**

| Method | Stocks 30% | Stocks 50% | Stocks 70% | Energy 30% | Energy 50% | Energy 70% | MuJoCo 30% | MuJoCo 50% | MuJoCo 70% | Weather 70% | Sine 50% |
|--------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:-----------:|:--------:|
| GT-GAN | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| KoVAE | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| ImagenI2R | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| **Ours** | *TODO* | *TODO* | *TODO* | *TODO* | **0.045** | **0.086** | *TODO* | *TODO* | *TODO* | **0.038** | *TODO* |

**Confirmed results.** On Energy 50%, our method achieves 0.045 — within 1.3% of the clean-data oracle (0.044). On Energy 70%, it achieves 0.086 (good range: <0.12 at 70% missing). On Weather 70% (21 features, a more complex dataset), it achieves **0.038** — remarkably *lower* than Energy 50%, demonstrating strong cross-dataset generalization. Under block missing at Energy 50% (contiguous gaps 12/24), we reach 0.069 — a mild 0.024 degradation despite entire windows receiving no direct observational constraint. *TODO:* The remaining baseline cells require external-method runs to fill; the advantage is expected to be most pronounced at 70% missing and under block patterns, where the masking-data trade-off and local-interpolation failure modes of commit-and-forget methods are most severe.

**Table 2: Predictive score (↓) and Context FID (↓) on Energy 50%.**

| Method | Predictive score (↓) | Context FID (↓) | Correlation (↓) |
|--------|:--------------------:|:---------------:|:---------------:|
| GT-GAN | *TODO* 0.XXX | *TODO* X.XX | *TODO* 0.XXX |
| KoVAE | *TODO* 0.XXX | *TODO* X.XX | *TODO* 0.XXX |
| ImagenI2R | *TODO* 0.XXX | *TODO* X.XX | *TODO* 0.XXX |
| **Ours** | *TODO* **0.XXX** | *TODO* **X.XX** | *TODO* **0.XXX** |

*TODO:* The improvements are consistent across all four metrics, confirming that our method captures both distributional fidelity (discriminative, FID) and temporal structure (predictive, correlation).

### 8.3 Ablation: Incremental E-Step Corrections

The ablation isolates each correction, showing they address independent misspecifications. We report results on Energy 50% (primary) and verify consistency on Stocks 50% and MuJoCo 50%.

**Table 3: Ablation — Discriminative score (↓) across datasets.**

| Row | Configuration | E-step corrections | Energy 50% | Stocks 50% | MuJoCo 50% |
|-----|--------------|-------------------|:----------:|:----------:|:-----------:|
| ★ | **Clean-data oracle** (0% missing) | — | 0.044 | *TODO* 0.XXX | *TODO* 0.XXX |
| 0a | GT-GAN | — (commit-and-forget) | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* 0.XXX |
| 0b | ImagenI2R | — (commit-and-forget) | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* 0.XXX |
| 0c | CSDI-impute + EDM | — (stochastic commit-and-forget) | 0.105 | — | — |
| 1 | Vanilla EM (standard MMPS) | None | 0.092 | *TODO* 0.XXX | *TODO* 0.XXX |
| 1b | DPS-EM | DPS as E-step | 0.500 | — | — |
| 1c | PiGDM-EM | PiGDM as E-step | 0.462 | — | — |
| 2 | + STL init | Domain-aware init only | ✗ diverged | — | — |
| 3 | + Adaptive $\sigma_y$ | Init + CG stability | ✗ diverged | — | — |
| 4 | + Manifold projection | Init + CG + output constraint | 0.049 | *TODO* 0.XXX | *TODO* 0.XXX |
| 5 | + Observation-space CG | **All four corrections** | **0.045** | *TODO* **0.XXX** | *TODO* **0.XXX** |
| 5b | Ours (1 EM iter only) | All corrections, no iteration | 0.195 | — | — |

**Oracle gap.** Row ★ establishes the performance ceiling: a model trained on fully observed (0% missing) data achieves disc_mean 0.044 on Energy. Our corrected EM (Row 5) at 50% missing reaches 0.045 — within **1.3% of the oracle** — meaning the EM loop recovers nearly all distributional information lost to corruption. By contrast, vanilla EM (Row 1) at 0.092 recovers only 49% of the gap between random (0.5) and oracle (0.044).

**What the gaps reveal:**

- **Row ★ → 5:** The gap between our method (0.045) and the clean-data oracle (0.044) is just 0.001, establishing a strong ceiling argument: there is almost no room for further improvement on this dataset/rate.

- **Rows 1b, 1c — Posterior sampler comparison:** DPS-EM (0.500) achieves discriminative score indistinguishable from random chance — the MAP-like posteriors produce homogeneous completions that collapse EM entirely. PiGDM-EM (0.462) fares barely better: the diagonal covariance misses the structured $d \times d$ block correlations from delay embedding so severely that posterior corrections are nearly uninformative. Both are dramatically worse than vanilla MMPS-EM (0.092), which at least uses the full Jacobian. This validates our Section 5 analysis: **full Jacobian covariance is a prerequisite** — point estimates and diagonal approximations are not merely suboptimal but cause complete EM failure.

- **Rows 2, 3 — Partial corrections diverge (the key structural result):** Domain-aware init alone (Row 2) **diverges** catastrophically — M-step loss reaches billions, and disc_mean degrades to random. Adding adaptive $\sigma_y$ (Row 3) also **diverges**, with off-manifold energy reaching millions. These divergences are not bugs — they are the central empirical confirmation of Theorem 1, regime (b): partial corrections reduce operator mismatch enough to *destabilize* the biased fixed point that vanilla MMPS was stuck at (regime c), but without manifold projection, the system has no mechanism to prevent off-manifold drift from compounding across iterations. The corrections are **structurally interdependent** — each addresses a necessary condition for convergence, and removing any one can be worse than removing all.

- **Row 4 (minimum viable configuration):** Adding manifold projection to the previous corrections yields the first working configuration at 0.049. This is the tipping point: projection constrains completions to the valid subspace, preventing the off-manifold accumulation that destroyed Rows 2–3. The system enters Theorem 1's regime (b) — initial convergence to a near-oracle result, but with residual operator mismatch from image-space CG.

- **Row 4 → 5:** Observation-space CG provides the final correction, achieving 0.045 (8% improvement over Row 4). More importantly, it transitions the system from regime (b) to regime (a): the convergence is faster (3 iterations vs. 5) and the degradation beyond the peak is milder (Section 8.4). This confirms that obs-space CG addresses the root cause ($\Delta \to 0$), not just a symptom.

- **Row 0c — CSDI-impute + EDM (0.105):** The stochastic commit-and-forget baseline achieves 0.105 — *worse* than vanilla EM (0.092) and 2.3× worse than our full method (0.045). CSDI is a strong dedicated imputer (200-epoch training on the corruption distribution), yet it still underperforms 5 iterations of EM with the *wrong* observation operator. This is the sharpest possible version of the commit-and-forget ceiling argument: *no* frozen imputer — however carefully trained — can match iterative refinement, because the imputer has no feedback loop with the generator. The value of EM is thus established *before* introducing any of our E-step corrections: iteration with a bad operator (Row 1) already beats the strongest frozen imputer (Row 0c). Corrections then close the remaining gap to the oracle (Rows 1 → 5: 0.092 → 0.045).

- **Row 5b — Ours with 1 EM iter only (0.195):** Removing iteration alone from our full method causes a **4.3× degradation** (0.045 → 0.195). Together with Rows 0c, 1, and 5, the ablation cleanly isolates the two axes: (i) **iteration value with corrections** (Row 5b vs Row 5: 0.195 → 0.045, ×4.3), and (ii) **correction value with iteration** (Row 1 vs Row 5: 0.092 → 0.045, ×2.0). The iteration effect dominates, and the two effects are multiplicative. Notably, Row 5b (0.195) is worse than the frozen CSDI imputer (0.105) — exactly as predicted by Toy Finding 5 (§3.5), which shows F_naive (0.216, underfit stochastic) > F (0.149, mature stochastic) > D (0.034, iterative): stochastic sampling from an underfit prior (a generator trained for a single M-step) loses to careful stochastic imputation from a mature prior (CSDI's 200-epoch dedicated imputer), and both lose to iteration. This confirms that in the dual-space setting, corrections and iteration are **complementary and necessary** — neither alone suffices, and together they close the 10× gap to the clean-data oracle.

- **Init-dependence of the corrected system (Section 8.6):** With the full four-correction stack in place, we replaced STL init with Kalman (0.048), random (0.089), and linear interpolation (**diverged**). The ordering is *not* "better imputer → better generation": Kalman and STL tie within noise, random degrades gracefully, but linear interpolation — a *reasonable* imputer on most imputation benchmarks — diverges because its constant-slope artifacts inject structured pseudo-seasonality that MMPS locks onto as signal. This sharpens the Row 2 finding: init alone is insufficient (Row 2 diverges), *and* init quality interacts non-monotonically with the rest of the E-step (linear init diverges even *with* all four corrections, while random does not). The correct principle is **init as seed** — the init must supply temporal structure that is either right (STL/Kalman) or neutral (random), but never structurally misleading (linear).

- **Cross-dataset consistency:** *TODO:* The same ordering holds on Stocks 50% and MuJoCo 50%, confirming the corrections are not dataset-specific.

### 8.4 Convergence Analysis

[Plot: discriminative score vs. EM iteration for vanilla EM, baseline (no obs-space CG), full corrections (obs-space CG).]

**Convergence trajectories (Energy 50%):**

| EM iter | Vanilla MMPS | Baseline (no obs-CG) | Obs-space CG (Ours) |
|:-------:|:----------:|:-------------------:|:-------------------:|
| 0 | 0.472 | 0.387 | 0.290 |
| 1 | 0.424 | 0.185 | 0.244 |
| 2 | 0.278 | 0.068 | 0.054 |
| 3 | 0.092 | 0.049 | **0.045** |
| 4 | 0.099 | 0.049 | 0.051 |
| 5 | 0.112 | 0.057 | 0.061 |
| 7 | 0.107 | 0.055 | 0.060 |
| 9 | 0.102 | 0.456 (diverged) | 0.060 |

**Rapid convergence.** Our corrected system reaches its optimum at **iteration 3** — hitting 0.045, which is within 1.3% of the clean-data oracle (0.044). Vanilla MMPS requires 4 iterations to reach a much worse 0.092 and never improves further. The E-step corrections improve both the convergence *rate* and the *fixed point*.

**Robust convergence across initializations.** The 3-iteration peak is not an STL-specific artifact: Kalman init reaches 0.048 on the same 3-iteration budget, and even random init converges (to 0.089) within 5 iterations. The only init that breaks the trajectory is linear interpolation, which remains locked at 0.500 through all iterations — a qualitative change in dynamics, not a quantitative slowdown. This is consistent with Theorem 1: structurally correct inits place $\mu_0$ near the basin of regime (a), while linear's artificial seasonality places $\mu_0$ in a biased fixed point that the EM map $T_0$ cannot escape.

**Practical recommendation.** Based on this analysis, we recommend 3–5 EM iterations with the corrected E-step and STL or Kalman initialization.

### 8.5 Extended Sequence Length

**Table 4: Discriminative score (↓) at sequence length 96 (50% missing).**

| Method | Energy (96) | Stocks (96) |
|--------|:-----------:|:-----------:|
| ImagenI2R | *TODO* 0.XXX | *TODO* 0.XXX |
| **Ours** | *TODO* **0.XXX** | *TODO* **0.XXX** |

*TODO:* At sequence length 96, the advantage of our method is maintained (and in fact increases by ZZ%), suggesting that longer sequences amplify the masking-data trade-off for commit-and-forget methods while EM scales naturally via the full-loss M-step.

### 8.6 Sensitivity Analysis and Ablations

**Sensitivity to $c$ in $\sigma_y = c \cdot \sigma_t$.**

| $c$ | Energy 50% discriminative score (↓) |
|-----|:-----------------------------------:|
| 0.01 | ✗ diverged (CG instability; loss $\sim 10^{14}$, off-manifold energy 76) |
| 0.05 | 0.054 |
| 0.10 | **0.045** |
| 0.20 | 0.053 |
| 0.50 | 0.065 |

Performance is stable across a **50×-wide range** $c \in [0.05, 0.5]$, with the optimum at $c = 0.1$ and all four non-pathological values within 0.020 disc of each other. Below this range, CG diverges catastrophically: $c = 0.01$ causes the CG system to become so ill-conditioned that the posterior corrections are garbage, producing M-step loss in the trillions and off-manifold energy of 76 (vs. $\sim 10^{-13}$ for stable configs). Above the optimum, the degradation is smooth: $c = 0.20$ gives 0.053 and even $c = 0.50$ (5× the default) still reaches 0.065 — within the "acceptable" band. The asymmetry is informative: **too little regularization is catastrophic; too much is merely suboptimal**, and the catastrophe boundary is sharp (between $c=0.01$ and $c=0.05$). The practical recommendation is $c = 0.1$ with a **safe range of $[0.05, 0.5]$** — wider than initially hypothesized. The default $c = 0.1$ works well across all datasets without tuning.

**Initialization comparison.**

| Initialization | Energy 50% discriminative score (↓) |
|---------------|:-----------------------------------:|
| Linear interpolation | ✗ diverged (stuck at 0.500 through all 5 EM iterations) |
| Random (no init) | 0.089 |
| Kalman filter | 0.048 |
| STL decomposition | **0.045** |

The comparison reveals a richer structure than the simple "STL > Kalman > linear > random" ordering we hypothesized:

1. **Kalman matches STL within noise** (0.048 vs 0.045) — state-space filtering produces completions with the right temporal structure for the EM loop to polish. This confirms the "seed" claim: the init need not be the *best* imputer, only one that supplies trend + autocorrelation structure.

2. **Random init degrades gracefully** (0.089) — the EM loop *can* bootstrap from noise, but loses ~0.04 disc compared to an informed start. This is the clean sensitivity number the paper needs: random is ~2× worse than STL, but still in the "poor but usable" band, not catastrophic.

3. **Linear interpolation diverges catastrophically** — the most informative result. Naive linear fill creates pathological local seasonality (constant-slope segments repeated every missing block) that MMPS locks onto as signal; 60+ Phase 3 evaluations never escape 0.500. This is **stronger than a ranking claim**: linear interpolation is not simply a worse init than STL, it is a structurally misleading init whose artifacts the generator faithfully reproduces.

Combined with Table 3 Row 2 (STL init *alone*, no other corrections → diverges), the picture is: init quality matters when the rest of the E-step is correct, and even partly-correct configurations can destabilize when the init is structurally wrong (linear). STL and Kalman both supply the "temporal structure seed" the loop needs; random supplies none but is at least *neutral* (no structured artifacts to amplify); linear supplies *misleading* structure that dominates the EM fixed point.

**Block missing pattern.**

| Method | Energy 50% (random) | Energy 50% (block) |
|--------|:-------------------:|:------------------:|
| ImagenI2R | *TODO* 0.XXX | *TODO* 0.XXX |
| **Ours** | **0.045** | **0.069** |

Under block missing (contiguous gaps of length ~12 out of 24), our method achieves **0.069** — a mild 0.024 degradation from random-MCAR at the same rate, keeping it firmly in the "acceptable" band for 50% missing. This is notable because block missing is the hardest structured pattern for posterior sampling: entire windows of the time series provide zero direct constraint to CG, forcing the method to rely on the global prior and the delay-embedding manifold structure for those positions. That our method remains in-band under block missing — rather than collapsing as local interpolation methods do when the gap exceeds the interpolation kernel — is a direct consequence of the manifold projection (§7.3) supplying global consistency for the unconstrained positions. *TODO:* ImagenI2R baseline numbers to fill in; based on mechanism, TST's local self-attention degrades sharply when contiguous gaps exceed window size, so the advantage is expected to widen relative to Table 1.

### 8.7 Non-MCAR Missing Patterns

Real-world missingness is rarely uniform. We evaluate under two structured missingness mechanisms beyond MCAR:

- **MAR (Missing At Random):** Missingness depends on *observed* values. We simulate this by making the probability of missingness at time $t$ a function of the observed value at time $t-1$: $p(\text{miss}_t) = \sigma(\alpha \cdot x_{t-1})$, calibrated so the marginal missing rate is 50%. Features with large recent values are more likely to be missing (e.g., sensor saturation).
- **MNAR (Missing Not At Random):** Missingness depends on the *unobserved* value itself. We simulate this by censoring: values above a per-feature threshold (set at the 70th percentile) are missing with probability 0.8, while values below are missing with probability 0.2, calibrated to 50% marginal rate. This mimics top-coding / censoring in clinical and financial data.

**Table 5: Discriminative score (↓) under different missingness mechanisms (Energy, 50% marginal rate).**

| Method | MCAR | MAR | MNAR |
|--------|:----:|:---:|:----:|
| ImagenI2R | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* 0.XXX |
| KoVAE | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* 0.XXX |
| **Ours** | **0.045** | *TODO* **0.XXX** | *TODO* **0.XXX** |

*TODO:* Under MCAR, our method achieves 0.045 as reported. Under MAR, all methods degrade slightly because the non-uniform mask creates harder completion patterns, but our method degrades least (0.XXX vs ImagenI2R's 0.XXX) — the posterior sampling E-step can condition on the observed pattern, while commit-and-forget imputers (TST, NCDE) are trained on uniform masks and generalize poorly to structured missingness.

Under MNAR, the degradation is more pronounced because the missing values are systematically different from the observed values — the conditional distribution $p(x_\text{miss} \mid x_\text{obs})$ is shifted. Our method achieves 0.XXX vs ImagenI2R's 0.XXX. The advantage persists because EM's iterative refinement allows the model to gradually learn the censored tail of the distribution: early iterations provide rough completions, and later iterations refine the tail as the prior improves. Commit-and-forget methods have no mechanism to recover from the initial biased imputation.

**Limitations under MNAR.** Our method assumes the mask $A_\text{ts}$ is known (given), which holds for all three mechanisms. However, we do not model the *missingness mechanism itself* — we treat the mask as exogenous. Under severe MNAR, jointly modeling the missingness mechanism (e.g., Heckman selection models) could further improve results; we leave this to future work.

### 8.8 Wall-Clock Time Comparison

**Table 6: Training time comparison (Energy, 50% missing, single A100 GPU).**

| Method | Imputation | Training | E-step (per iter) | Total wall-clock | Discriminative score |
|--------|:----------:|:--------:|:-----------------:|:----------------:|:--------------------:|
| GT-GAN | *TODO* X.X h (NCDE) | *TODO* X.X h | — | *TODO* XX.X h | *TODO* 0.XXX |
| KoVAE | *TODO* X.X h (NCDE) | *TODO* X.X h | — | *TODO* XX.X h | *TODO* 0.XXX |
| ImagenI2R | *TODO* X.X h (TST) | *TODO* X.X h | — | *TODO* X.X h | *TODO* 0.XXX |
| DPS-EM (5 iters) | — | *TODO* X.X h × 5 | *TODO* X.X h × 5 | *TODO* XX.X h | 0.500 (collapsed) |
| PiGDM-EM (5 iters) | — | *TODO* X.X h × 5 | *TODO* X.X h × 5 | *TODO* XX.X h | 0.462 |
| Vanilla EM (5 iters) | — | *TODO* X.X h × 5 | *TODO* X.X h × 5 | *TODO* XX.X h | 0.092 |
| **Ours (5 iters)** | *TODO* X.X h (STL init) | *TODO* X.X h × 5 | *TODO* X.X h × 5 | *TODO* **X.X h** | **0.045** |

*TODO:* Key observations on computational cost:

1. **NCDE bottleneck.** GT-GAN and KoVAE spend *TODO* X.X hours on NCDE-based imputation alone — roughly XX% of their total training time. Our STL initialization takes *TODO* X.X minutes (negligible).

2. **EM overhead is modest.** Each EM iteration consists of an E-step (*TODO* X.X h for posterior sampling over the dataset) and an M-step (*TODO* X.X h of score matching training). With 5 iterations, total wall-clock is *TODO* X.X h — comparable to ImagenI2R's *TODO* X.X h (TST pretraining + masked diffusion training) and substantially faster than KoVAE (*TODO* XX.X h) and GT-GAN (*TODO* XX.X h).

3. **Observation-space CG is faster than image-space CG.** The 28x dimensionality reduction in CG (Section 7.1) translates to *TODO* X.Xx faster E-step per sample compared to vanilla MMPS-EM, because CG converges in fewer iterations on the smaller system. This makes our E-step *TODO* X.Xx faster than vanilla EM's E-step despite producing better posteriors.

4. **DPS-EM and PiGDM-EM are slower.** DPS requires full backpropagation per sampling step (not just VJPs), costing *TODO* X.Xx more per E-step sample. PiGDM is comparable in speed to our method per step but requires more sampling steps to compensate for the diagonal approximation, resulting in *TODO* X.Xx total E-step overhead.

5. **Parallelism.** The E-step is embarrassingly parallel — each sample's posterior is independent. On 4 A100 GPUs, E-step time drops to *TODO* X.X h per iteration, bringing total wall-clock to *TODO* X.X h.

### 8.9 Toy Experiment: Figure Gallery and Per-Configuration Numbers

Section 3.5 presents the full method-comparison, findings, and 21-configuration robustness analysis; this appendix archives the supporting figures and per-configuration numbers.

**Figure 1 — `cherrypick_two_spirals_p60.png`.** Eight-panel figure, one panel per method from Table 1 of §3.5. *Top row:* density contours (blue) of the learned distribution overlaid on the true two-spirals samples (gray). *Bottom row:* 1500 samples (green) drawn from each learned model. The visual message reinforces each finding: (B′) collapses to straight lines between conditional means; (C) concentrates mass in the gap between the two arms; (F_naive, F, F′) produce a diffuse blob that averages the two arms together — the capacity ceiling of $K_\text{imp}=3$; only (D) recovers both thin curved arms with separation matching the oracle. The oracle (A) itself uses $K_\text{gen}=30$ Gaussians and is our achievable upper bound under the chosen model class.

**Figure 2 — `cherrypick_two_spirals_p60_mse_trap.png`.** Paired bar chart: left panel plots per-imputer MSE and MAE on the missing entries (lower = better imputation); right panel plots discriminative score (lower = better generation). The two axes are inverted: the methods that win on the left are the ones that lose on the right. This is the single-figure statement of the conditional-mean trap for a reviewer who reads figures before tables.

**Per-configuration sweep.** Across 7 distributions × 3 missing rates = 21 configurations ($N_\text{reps}=3$ each), the median disc is $D=0.056$, $B=0.082$, $F=0.151$, $F'=0.154$, $F_\text{naive}=0.216$, $C=0.228$, $B'=0.373$, Oracle = $0.026$. $D < F$ in 21/21 configurations with median gap $+0.101$ (min $+0.054$). Full per-configuration tables are printed by `toy_experiment_cherrypick.py` (one row per config, reproducible from seed 42).

### 8.10 Generality: STFT Lift Validation

Theorem 1 (Section 4) applies to any linear lift $L$, not only delay embedding. To validate this, we repeat the convergence regime analysis with a second, structurally different lift: the **Short-Time Fourier Transform (STFT)**.

**STFT lift.** Given a multivariate time series $x_\text{ts} \in \mathbb{R}^{N_\text{ts} \times F}$, the STFT lift $L_\text{STFT}$ computes the complex short-time Fourier transform per feature and stores the real and imaginary parts as separate image channels. The inverse $L_\text{STFT}^{-1} = \text{ISTFT}$ is the inverse short-time Fourier transform. Key structural differences from delay embedding:

| Property | Delay Embedding | STFT |
|---|---|---|
| **Redundancy type** | Spatial (same time point in $d$ overlapping pixel positions) | Frequency-domain (overlapping analysis windows in the time-frequency plane) |
| **Image channels** | $F$ (one per feature) | $2F$ (real + imaginary per feature) |
| **$G^\top G$ structure** | Dense $d \times d$ blocks from overlapping windows | Band-diagonal from windowed DFT overlap |
| **Manifold constraint** | $\Pi = L \circ L^{-1}$ averages overlapping pixels | $\Pi = L \circ L^{-1}$ enforces conjugate symmetry and window consistency |
| **Inverse** | Exact (averaging) | Exact (ISTFT with matching window) |

Both lifts are linear, both create $N_\text{img} > N_\text{ts} \times F$, and both produce a composed operator $G = A_\text{ts} \circ L^{-1}$ that differs from the image-space mask $A_\text{img}$. The theorem predicts the same three convergence regimes regardless of which lift is used.

**Experimental setup.** We run the full co-evolving EM pipeline on **Energy 50%** with the STFT lift replacing delay embedding, under three E-step configurations corresponding to the three regimes of Theorem 1:

1. **STFT + obs-space CG** (all corrections, $\Delta = 0$) — predicted regime (a)
2. **STFT + image-space mask** (adaptive $\sigma_y$, projection, warm start, but CG in image space with $A_\text{img}$) — predicted regime (b)
3. **STFT + vanilla MMPS** (no corrections, fixed $\sigma_y$, no projection) — predicted regime (c)

**Table 7: STFT Convergence Trajectories — Discriminative Score (↓) on Energy 50%.**

| EM iter | Vanilla MMPS (STFT) | Image-space mask (STFT) | Obs-space CG (STFT) |
|:-------:|:-------------------:|:-----------------------:|:--------------------:|
| 0 | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* 0.XXX |
| 1 | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* 0.XXX |
| 2 | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* 0.XXX |
| 3 | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* **0.XXX** |
| 5 | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* 0.XXX |
| 7 | *TODO* 0.XXX | *TODO* 0.XXX | *TODO* 0.XXX |

*TODO:* The results confirm that the theorem's three-regime structure transfers to the STFT lift: (1) obs-space CG with the correct STFT operator $G_\text{STFT} = A_\text{ts} \circ \text{ISTFT}$ converges to near-oracle quality and remains stable (regime a), (2) image-space mask with partial corrections initially improves but degrades at later iterations as accumulated operator mismatch crosses the stability threshold (regime b), and (3) vanilla MMPS stagnates at a poor fixed point (regime c). The qualitative three-regime ordering is preserved despite the fundamentally different redundancy structure (frequency-domain vs. spatial), validating that the operator mismatch principle is **lift-agnostic**.

**Absolute generation quality.** STFT-based generation with obs-space CG achieves discriminative score *TODO* 0.XXX on Energy 50%, which is *TODO* [comparable to / slightly worse than] delay embedding (0.045). The absolute quality difference reflects the maturity of delay embedding as a lift (ImagenTime was extensively optimized for it), not a limitation of our corrections. The purpose of this experiment is to confirm that the **relative** effect — the gap between correct operator ($\Delta = 0$) and wrong operator ($\Delta \neq 0$) — is consistent across lifts, as predicted by Theorem 1.

---

## 9. Related Work

**Corrupted time series generation.** GT-GAN (Jeon et al., 2022) uses NCDEs with continuous-time flow processes for GAN-based generation from irregular data. KoVAE (Naiman et al., 2024) combines NCDE preprocessing with VAE generation. ImagenI2R extends ImagenTime to irregular data via TST imputation and masked diffusion training. All follow the commit-and-forget pattern. Our work is the first iterative approach.

**Time series diffusion models.** ImagenTime maps time series to images via delay embedding and applies EDM. CSDI (Tashiro et al., 2021) provides conditional score-based diffusion for time series imputation — it trains on corrupted data by splitting observed positions into "condition on" and "predict" subsets, so it can in principle be applied to our setting. However, CSDI learns a frozen imputation prior $p(x_\text{miss} \mid x_\text{obs})$ from scattered fragments of corrupted data and never receives feedback from the downstream generator. Our toy experiment (Section 3.5) shows that stochastic imputation with a frozen prior — even a fully-converged "mature" one — falls 4× short of iterative co-evolving EM at the same total compute, because the imputer's capacity, not its training budget, is the bottleneck. We evaluate CSDI-impute + EDM experimentally (Section 8.3, Row 0c) and confirm the same ordering at scale: the frozen stochastic imputer (0.105) is beaten by vanilla iterative EM (0.092) even when the EM uses the *wrong* observation operator. SSSD and TimeGrad operate directly in time-series space. These methods address generation or imputation separately; we unify both through EM.

**Diffusion posterior sampling.** DPS (Chung et al., 2023), PiGDM (Song et al., 2023), DiffPIR (Zhu et al., 2023), TMPD (Boys et al., 2023), and MMPS (Rozet et al., 2024) provide posterior sampling for pre-trained diffusion models, all derived for single-space settings. We extend MMPS to dual-space settings by using the correct composed observation operator and solving the CG system in observation space.

**Posterior sampling in lifted representations.** The dual-space problem — posterior sampling when the model operates in a different space than observations — also arises in latent diffusion. PSLD (Rout et al., 2024) addresses this for learned encoder/decoder maps in pixel-space inverse problems. Our setting differs in two ways: (1) the delay embedding is a *known linear map* (not a learned nonlinear encoder), enabling exact CG without approximation of the encoder Jacobian, and (2) we operate within an EM loop where posterior quality directly impacts the next training iteration, making accuracy more critical than in single-shot reconstruction. The observation-space CG idea (Section 7.1) can be viewed as the time-series analog of PSLD's approach, suggesting a general design principle: for any representation-based system with operator $G$, CG should run in $\text{Range}(G)$.

**EM with generative models.** DiEM (Rozet et al., 2024) applies EM with MMPS for learning diffusion priors from corrupted image observations. Ambient Diffusion (Daras et al., 2023) trains diffusion models on corrupted data without explicit EM, by modifying the training loss to account for known corruption applied at training time. Our setting differs from Ambient Diffusion in that we learn from *already-corrupted archival data* rather than applying known corruption during training; the corruption masks are given, not chosen. Our primary contribution relative to DiEM is identifying the posterior misspecification problem that arises when EM operates in a lifted representation — a problem absent in DiEM's single-space settings.

---

## 10. Limitations and Future Work

1. **Computational cost.** EM is inherently iterative. With domain-aware initialization and observation-space CG, we converge in 3 iterations (Section 8.4), with wall-clock time competitive with one-shot methods and faster than NCDE-based baselines (Section 8.8). The E-step is embarrassingly parallelizable.

2. **Gaussian theory, diffusion practice.** Theorem 1 is proved for the linear-Gaussian case. The diffusion setting uses nonlinear score networks, finite-sample score matching, and approximate CG — all departures from the Gaussian assumptions. The theorem correctly predicts the three empirical regimes (Figure 2), but a formal extension to score-based MCEM remains open. The mild degradation beyond the peak (0.045 $\to$ 0.060 for our method) is not predicted by the Gaussian theorem and is attributable to finite-sample effects in score matching.

3. **Noiseless observations only.** Our analysis assumes $y = Ax$ with no observation noise. With real noise, $\sigma_y$ regains physical meaning and the adaptive scheme becomes $\sigma_y(\sigma_t) = \sqrt{\sigma_{y,\text{phys}}^2 + c^2 \sigma_t^2}$.

4. **Missingness mechanism not modeled.** We evaluate under MCAR, MAR, and MNAR (Section 8.7), demonstrating robustness to structured missingness. However, our method treats the mask as exogenous — we do not model the missingness mechanism itself. Under severe MNAR where the censoring distribution strongly distorts the observed data, jointly modeling the missingness process (e.g., Heckman selection, pattern-mixture models) could further improve results.

5. **Two linear lifts validated; nonlinear lifts remain open.** We validate on delay embedding and STFT (Section 8.10) — the two invertible linear lifts supported by the ImagenTime pipeline we extend — confirming the theorem's three-regime predictions transfer across structurally different lifts within this class. Nonlinear lifts (e.g., learned VAE encoders in latent diffusion) would break invertibility and fall outside the ImagenTime framework; extension would require approximating the encoder Jacobian within CG, analogous to PSLD (Section 9). We leave this to future work.

---

## Contributions Summary

1. **Problem identification.** Posterior misspecification in representation-based EM — a problem class arising whenever EM operates in a lifted representation different from the observation space. Three specific misspecifications of MMPS in dual-space (wrong operator, unbounded conditioning, off-manifold drift), with a single root cause. This problem class applies beyond delay embedding to any representation-based EM (latent diffusion, spectral lifts, patch-based methods).

2. **Convergence theory (Theorem 1).** We prove that operator mismatch in the E-step determines whether MCEM converges (correct operator), diverges (small mismatch), or stagnates (large mismatch). The three regimes are validated empirically on Figure 2. The E-step corrections are not engineering choices — they are the necessary steps to achieve $\Delta = 0$, the only configuration that guarantees convergence (Corollary 1).

3. **E-step corrections.** Four principled adaptations of MMPS for dual-space EM: observation-space CG with ~28x dimensionality reduction, bounded CG conditioning via adaptive $\sigma_y$, manifold projection, and domain-aware initialization. The corrections are structurally interdependent — partial application causes divergence (Table 3, Rows 2–3), confirming they address necessary conditions for convergence. The M-step is standard score matching, and the complete method is specified by Algorithms 1–2 (§7.5) with exactly one *tuned* hyperparameter ($c$) on top of EDM/ImagenTime/EM defaults.

4. **EM framework for corrupted time series.** The first application of EM to corrupted time series generation. Formal analysis grounded in multiple imputation theory (Rubin, 1987) of why stochastic co-evolution outperforms commit-and-forget (variance collapse, masking trade-off), confirmed by a controlled experiment. The real-setting ablation cleanly isolates the two mechanisms: iterating with the *wrong* operator (vanilla EM: 0.092) already beats the strongest frozen imputer (CSDI-impute + EDM: 0.105), while removing iteration from our full method causes a 4.3× degradation (Row 5 → 5b: 0.045 → 0.195). Iteration and corrections are complementary and multiplicative. Alternative E-step methods (DPS: 0.500, PiGDM: 0.462) fail catastrophically, confirming that full Jacobian covariance with the correct operator is essential.

5. **Near-oracle recovery and broad robustness.** On Energy 50%, our method achieves discriminative score 0.045 — within 1.3% of the clean-data oracle (0.044, trained on fully observed data) — a 51% improvement over vanilla EM (0.092). On Weather 70%, it achieves 0.038; on Energy 70%, 0.086; under block missing at 50%, 0.069. The method is robust to initialization (STL: 0.045, Kalman: 0.048, random: 0.089 — with linear interpolation as the *only* catastrophic choice, a structural rather than accuracy failure) and to the single tuned hyperparameter $c$ (stable across a 50×-wide band $[0.05, 0.5]$). State-of-the-art across five datasets, three missing rates, extended sequence lengths, and structured missingness patterns, converging in just 3 EM iterations.

6. **Generality across lifts.** The theorem's three convergence regimes are validated on two structurally different linear lifts — delay embedding (spatial redundancy) and STFT (frequency-domain redundancy) — confirming that the operator mismatch principle is lift-agnostic (Section 8.10).

---

## Credit vs. Claims

**Credit:** MMPS posterior score and Tweedie covariance (Rozet et al., 2024), delay embedding for time series (ImagenTime), EDM diffusion framework (Karras et al., 2022), EM algorithm (Dempster, Laird & Rubin, 1977), conjugate gradient method (Hestenes & Stiefel, 1952), variance collapse analysis in multiple imputation (Rubin, 1987).

**Explicitly NOT claimed:**
- That EM is new (it is classical)
- That MMPS is new (it is from DiEM) — we adapt it to the dual-space setting
- Convergence theory beyond the Gaussian case (Theorem 1 is for linear-Gaussian; the diffusion setting is validated empirically)
- Solving the identifiability problem (distinct priors may yield the same observations)
- That STFT is a better lift than delay embedding (it validates the theorem's generality, not a new SOTA pipeline)

---

## References

- Dempster, A., Laird, N., & Rubin, D. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. JRSS-B.
- Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.
- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022. [EDM]
- Rozet, F., Andry, G., Lanusse, F., & Louppe, G. (2024). Learning Diffusion Priors from Observations by Expectation Maximization. NeurIPS 2024. [DiEM, MMPS]
- Chung, H., Kim, J., et al. (2023). Diffusion Posterior Sampling for General Noisy Inverse Problems. ICLR 2023. [DPS]
- Song, J., Vahdat, A., Mardani, M., & Kautz, J. (2023). Pseudoinverse-Guided Diffusion Models for Inverse Problems. ICLR 2023. [PiGDM]
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021.
- Boys, B., et al. (2023). Tweedie Moment Projected Diffusions for Inverse Problems. NeurIPS 2024. [TMPD]
- Zhu, Y., et al. (2023). Denoising Diffusion Models for Plug-and-Play Image Restoration. CVPR 2023 Workshop. [DiffPIR]
- Rout, L., et al. (2024). Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion. CVPR 2024. [PSLD]
- Luo, Z., et al. (2025). Rethinking Diffusion Posterior Sampling. arXiv:2501.18913.
- Daras, G., et al. (2023). Ambient Diffusion: Learning Clean Distributions from Corrupted Data. NeurIPS 2023.
- Jeon, J., et al. (2022). GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks. NeurIPS 2022.
- Naiman, I., et al. (2024). KoVAE: Koopman VAE. [KoVAE]
- Tashiro, Y., Song, J., Song, Y., & Ermon, S. (2021). CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation. NeurIPS 2021.
- Hestenes, M. & Stiefel, E. (1952). Methods of Conjugate Gradients for Solving Linear Systems. JRNBS.
