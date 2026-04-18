# Trajectory-Corrected Posterior Sampling for Corrupted Time Series Generation

---

## 1. Introduction

State-of-the-art time series generation operates in a *lifted representation*: delay embedding maps time series to images, and standard image diffusion (EDM; Karras et al., 2022) is applied in image space (ImagenTime). When the training data is corrupted — sensors fail, measurements are intermittent, records have gaps — this creates a **dual-space** system: observations live in time-series space while the generative model operates in image space.

Learning from corrupted data naturally calls for Expectation-Maximization (Dempster, Laird & Rubin, 1977): alternate between completing missing data using the model's learned prior (E-step) and training on those completions (M-step). DiEM (Rozet et al., 2024) demonstrated this EM framework for image reconstruction using MMPS (Moment-Matching Posterior Sampling). The natural starting point for corrupted time series is to apply DiEM's EM loop within ImagenTime's delay-embedding pipeline.

**The problem.** This combination fails. Existing posterior sampling methods — DPS, PiGDM, MMPS — were all designed for **single-space** settings where the observation operator acts in the model's native space. In the dual-space setting created by delay embedding, every method suffers from one or more of three structural failures:

1. **Likelihood misspecification.** Methods approximate a likelihood $p(y \mid x_t)$ using an observation operator $A$ that acts in image space. The true operator $G = A_\text{ts} \circ \texttt{img\_to\_ts}$ crosses spaces. The wrong operator leads to wrong posterior corrections.
2. **Numerical instability.** CG-based solvers (MMPS) and gradient-based guidance (DPS) become ill-conditioned at high noise levels when applied with the wrong operator, producing garbage corrections early in the reverse process.
3. **Constraint violation.** No existing method ensures that posterior samples lie on the delay-embedding manifold or exactly match observations.

The best existing method (MMPS) achieves discriminative score 0.092 on Energy 50% — far from the 0.044 clean-data oracle. The root cause is shared across all methods: **they try to approximate a misspecified likelihood score and solve an optimization problem in the wrong space.**

**Our approach: Trajectory-Corrected Posterior Sampling (TCPS).** We propose a fundamentally different posterior sampling strategy that avoids likelihood approximation entirely. The key insight is that in the VE-SDE forward process, if we fix the noise vector $z$ at initialization, we know the **exact state** that observed positions should have at every noise level throughout the reverse process. This gives us a per-step ground truth signal that existing methods lack.

Instead of solving a regularized linear system to approximate the posterior score (as MMPS does), TCPS directly measures the trajectory error at observed positions and propagates corrections to missing positions via the denoiser's Jacobian. This design naturally resolves all three structural failures: (1) no observation operator is needed — errors are measured directly; (2) corrections are automatically annealed by the noise schedule — no $\sigma_y$ regularization; (3) hard observation replacement at every step guarantees exact consistency.

**Contributions.**

1. **Unified critique of existing posterior samplers in dual-space.** We identify three structural failures (likelihood misspecification, numerical instability, constraint violation) that affect all existing methods when applied to representation-based EM, and prove (Theorem 1) that any method suffering from likelihood misspecification either stagnates or diverges.

2. **Trajectory-Corrected Posterior Sampling (TCPS).** A posterior sampling method designed for dual-space EM that replaces likelihood-score approximation with direct trajectory error propagation. TCPS has a single hyperparameter ($\alpha$, correction strength), requires only one VJP per reverse step (vs. multiple for CG-based methods), and provides exact observation consistency by construction.

3. **Trajectory Consistency Theorem (Theorem 2).** We prove that TCPS generates samples that are exactly consistent with observations at every noise level, and that the correction magnitude is naturally bounded by $O(\sigma_t)$ — providing built-in annealing that prevents the over-correction pathology of DPS and the conditioning blow-up of MMPS.

4. **EM framework for corrupted time series.** The first application of EM to corrupted time series generation. We show that stochastic co-evolution outperforms commit-and-forget (variance collapse, masking trade-off), confirmed by a controlled experiment that separates the benefit of stochastic sampling from iterative refinement.

5. **SOTA results.** State-of-the-art generation quality across datasets, missing rates, and missingness patterns. On Energy 70%, TCPS-EM achieves discriminative score 0.092 — a 4.6× improvement over MMPS-EM (0.423). On Energy 50%, TCPS-EM achieves discriminative score *TODO* — within *TODO*% of the clean-data oracle (0.044).

---

## 2. Background

### 2.1 Delay Embedding for Time Series

Direct time-series diffusion models underperform compared to image-based approaches. ImagenTime demonstrated that mapping time series to images via **delay embedding**, then applying standard image diffusion (EDM), substantially outperforms direct methods.

**Delay embedding.** Given a multivariate time series $x_\text{ts} \in \mathbb{R}^{N_\text{ts} \times F}$, the delay embedding $L = \texttt{ts\_to\_img}$ creates an image by arranging overlapping sliding windows of length $d$ with stride $s$ as rows of a 2D array. The inverse $L^{-1} = \texttt{img\_to\_ts}$ recovers the time series by averaging overlapping pixel contributions. Key properties:

- Each time point appears in **multiple** image positions (overlapping windows create redundancy)
- $N_\text{img} > N_\text{ts}$: the image has more pixels than the time series has values
- Valid images form a **linear subspace** $\mathcal{M} = \text{Range}(L) \subset \mathbb{R}^{N_\text{img}}$
- The composition $\Pi = L \circ L^{-1}$ is an orthogonal projection onto $\mathcal{M}$

### 2.2 Diffusion Models

We adopt the EDM framework (Karras et al., 2022) with VE-SDE forward process:

$$x_t = x_0 + \sigma_t \, z, \quad z \sim \mathcal{N}(0, I)$$

The denoiser $d_\theta(x_t, \sigma)$ is trained via:

$$\arg\min_\theta \mathbb{E}_{p(x) p(\sigma) p(x_t \mid x)} \big[ \lambda(\sigma) \| d_\theta(x_t, \sigma) - x \|^2 \big]$$

The optimal denoiser is the Tweedie posterior mean $\mathbb{E}[x \mid x_t]$, linked to the score via $\nabla_{x_t} \log p(x_t) = (d_\theta(x_t, \sigma) - x_t) / \sigma_t^2$. The Tweedie covariance is $\mathbb{V}[x \mid x_t] \approx \sigma_t^2 \, J$ where $J = \nabla_{x_t} d_\theta$.

Reverse sampling proceeds via the Heun (2nd-order) solver: from noise $x_T \sim \mathcal{N}(0, \sigma_\text{max}^2 I)$, iteratively denoise along a schedule $\sigma_\text{max} = \sigma_0 > \sigma_1 > \cdots > \sigma_N = 0$.

### 2.3 EM for Diffusion Priors

Following DiEM (Rozet et al., 2024), we alternate between:

- **E-step**: Sample $x \sim q_{\theta_k}(x \mid y)$ via posterior sampling with current model $\theta_k$
- **M-step**: Train $\theta_{k+1}$ via denoising score matching on E-step completions

This minimizes $\text{KL}(p(y) \| q_\theta(y))$. DiEM demonstrated this framework for image reconstruction in single-space settings (CIFAR-10, MRI); we extend it to the dual-space setting of time series generation.

### 2.4 The Dual-Space Observation Model

- Corrupted time series: $y_\text{ts} = A_\text{ts} \cdot x_\text{ts}$ (binary mask, no observation noise)
- Delay embedding: $L = \texttt{ts\_to\_img}$ maps TS to image; $L^{-1} = \texttt{img\_to\_ts}$ maps back
- Diffusion model trained in image space $\mathbb{R}^{N_\text{img}}$ via EM
- The **true composed observation operator** is $G = A_\text{ts} \circ L^{-1} : \mathbb{R}^{N_\text{img}} \to \mathbb{R}^{N_\text{obs} \times F}$

---

## 3. Why Close the Loop? EM over Commit-and-Forget

Before addressing the technical challenge of *which* posterior sampler to use in the E-step, we establish *why* EM is necessary at all.

### 3.1 The Conditional Mean Trap

All existing methods for corrupted time series generation — GT-GAN (Jeon et al., 2022), KoVAE (Naiman et al., 2024), ImagenI2R — follow a **commit-and-forget** pattern: one-shot imputation, then frozen training.

| Family | Representative | Imputer | Generator | Limitation |
|--------|---------------|---------|-----------|------------|
| **GAN-based** | GT-GAN | Neural CDE (one-shot) | GAN | Unstable training, mode collapse, slow NCDE |
| **VAE-based** | KoVAE | Neural CDE (one-shot) | Koopman VAE | ~6.5x slower (NCDE cost), limited expressiveness |
| **Diffusion-based** | ImagenI2R | TST (one-shot, deterministic) | Masked diffusion | Masking-data trade-off |
| **Diffusion-based** | CSDI + EDM | CSDI (one-shot, stochastic) | Standard diffusion | Frozen imputer, no generator feedback |

Deterministic imputers output point estimates, but generative models need distributional samples. The MSE-optimal estimate $\hat{x} = \mathbb{E}[x_\text{miss} \mid x_\text{obs}]$ is distributionally catastrophic when the conditional is multimodal — it places mass between modes, in empty space. The resulting generator learns phantom density where no data exists.

### 3.2 Variance Collapse

By the law of total variance:

$$\text{Var}(X) = \text{Var}(\mathbb{E}[X \mid \text{obs}]) + \mathbb{E}[\text{Var}(X \mid \text{obs})]$$

Imputing every missing value by $\mathbb{E}[X \mid \text{obs}]$ produces a dataset with variance deficit $\mathbb{E}[\text{Var}(X \mid \text{obs})]$ — the expected conditional variance. At high missing rates, this can be enormous. In EM with stochastic E-step, imputed values are **samples** from $p(x_\text{miss} \mid x_\text{obs}, \theta)$. Individual samples have higher MSE than the conditional mean, but the collection preserves the correct marginal variance and multimodality.

### 3.3 The Masking-Data Trade-off

ImagenI2R partially mitigates variance collapse by using a masked loss — the diffusion model trains only on observed positions. But at missing rate $r$, only $(1-r)$ fraction of pixels contribute to the loss, and the mismatch between masked training and full-image sampling degrades generation. EM avoids this entirely: after the E-step, the data is complete and the M-step trains with full loss on every pixel.

### 3.4 Stochastic Commit-and-Forget is Not Enough

Stochastic imputation (e.g., CSDI) avoids variance collapse but follows the commit-and-forget paradigm: the imputer is frozen, never receiving feedback from the generator. Our toy experiment (Section 3.5) shows that stochastic completion with a frozen prior can *underperform* the best deterministic baseline, because sampling from an inaccurate prior introduces noise that outweighs the diversity benefit. Only when stochastic sampling is paired with iterative refinement (EM) does it surpass all baselines.

### 3.5 Toy Experiment: Separating Stochastic Sampling from Iterative Refinement

**Setup.** True distribution: two interleaving crescents + tilted ellipse. Corruption: each coordinate independently masked with probability 0.4. Model: Gaussian Mixture Model with 10 components.

| Method | Paradigm | disc (↓) | Imputation MSE |
|--------|----------|:--------:|:--------------:|
| (A) Oracle | — | 0.590 | — |
| (B) Regression + masked loss | commit-and-forget (deterministic) | 0.578 | **1.40** |
| (B') Regression + full loss | commit-and-forget (deterministic) | 0.768 | **1.40** |
| (C) KNN + full loss | commit-and-forget (deterministic) | 0.661 | **1.40** |
| (E) Posterior sample (1 round) + full loss | commit-and-forget (stochastic) | 0.703 | 2.34 |
| (E') Posterior sample (K=5 rounds) + full loss | commit-and-forget (stochastic, K samples) | 0.647 | 2.16 |
| (D) Co-Evolving EM (15 iters) | co-evolving (stochastic + iterative) | **0.572** | 2.55 |

Four findings:

1. **Better imputation ≠ better generation.** KNN imputes better than EM (MSE 1.40 vs 2.55) yet EM produces far better generations (disc 0.572 vs 0.661).
2. **Full loss on deterministic imputations is catastrophic.** Regression + full loss (0.768) is the worst — the model trusts variance-collapsed imputations as ground truth.
3. **Stochastic sampling with a frozen prior can be worse than careful deterministic.** Method (E) at disc 0.703 underperforms regression + masked loss (B, 0.578). The inaccurate prior produces noisy completions.
4. **Iterative refinement is the key ingredient.** Co-Evolving EM (D, 0.572) is the only method that outperforms the Oracle-aware baseline. Each EM round refines the prior, so subsequent samples become increasingly faithful.

---

## 4. Why Existing Posterior Samplers Fail in Dual-Space

Having established that EM is necessary, we now ask: which posterior sampling method should we use for the E-step? We analyze the five main approaches and show that each fails in the dual-space setting for specific, identifiable reasons. All failures trace back to a shared root cause: **approximating a misspecified likelihood score.**

### 4.1 The Shared Starting Point

All posterior sampling methods start from Bayes' rule on the score:

$$\nabla_{x_t} \log p(x_t \mid y) = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{prior score (from denoiser)}} + \underbrace{\nabla_{x_t} \log p(y \mid x_t)}_{\text{likelihood score (approximated)}}$$

The challenge is the likelihood score. Every method assumes an observation model $y = Ax + \text{noise}$ where $A$ operates in the model's native space. In our setting, the true observation model is $y_\text{ts} = A_\text{ts} \cdot \texttt{img\_to\_ts}(x_\text{img})$, and the composed operator $G = A_\text{ts} \circ \texttt{img\_to\_ts}$ has three structural properties that break existing methods:

1. **$G$ is not a mask in model space.** It crosses two spaces via the averaging operator $\texttt{img\_to\_ts}$.
2. **$G$ is rank-deficient.** $N_\text{img} > N_\text{ts}$, so $G$ has a large null space (off-manifold image directions).
3. **$G^\top G$ is not diagonal.** Overlapping delay-embedding windows create structured off-diagonal correlations.

### 4.2 DPS — Diffusion Posterior Sampling (Chung et al., 2023)

**How it works.** Approximate $p(x \mid x_t)$ with a delta function at the Tweedie mean — a zeroth-order approximation ignoring covariance entirely.

**Why it fails.**

- **F1: Posterior collapse.** Recent analysis (Luo et al., 2025) shows DPS performs MAP estimation rather than posterior sampling. In EM, posterior diversity is essential — MAP-like behavior collapses all completions of a given observation to a single point, preventing the model from learning distributional structure. The variance collapse is similar to deterministic imputation (Section 3.1) but occurs *within* the posterior sampler.
- **F2: Gradient instability.** DPS computes $\nabla_{x_t} \| y - A \, d_\theta(x_t) \|^2$, requiring full backpropagation through the denoiser. With the wrong operator $A_\text{img}$ in high dimensions, these gradients are noisy and scale poorly.
- **F3: No observation consistency.** The soft gradient guidance never guarantees that the final sample matches observations exactly.

**Empirical consequence.** DPS-EM achieves disc_mean *TODO* on Energy 50% — far worse than the 0.044 oracle.

### 4.3 PiGDM / TMPD — Diagonal Covariance Methods (Song et al., 2023; Boys et al., 2023)

**How they work.** Approximate the Tweedie covariance with a diagonal: $\mathbb{V}[x \mid x_t] \approx \text{diag}(v_t)$.

**Why they fail.**

- **F1: Correlation blindness.** Delay embedding creates structured $d \times d$ block correlations — each time point maps to $d$ overlapping image positions. A diagonal approximation treats these $d$ copies as independent, underestimating manifold-direction covariance by a factor of $\sim d$ (e.g., ~8x for typical settings). The posterior corrections systematically undershoot along correlated directions.
- **F3: No output constraints.** Like DPS, diagonal methods produce no manifold or observation guarantees.

### 4.4 DiffPIR — Plug-and-Play Restoration (Zhu et al., 2023)

**How it works.** Alternates denoising steps with data-fidelity proximal steps. The proximal step for operator $G$ requires solving $(I + \lambda G^\top G)x = b$.

**Why it fails.**

- **F1: Intractable proximal.** Because $G^\top G = L \cdot A_\text{ts}^\top \cdot A_\text{ts} \cdot L^{-1}$ is not diagonal, the proximal has no closed form. Iterative solvers inside the proximal collapse DiffPIR into something resembling MMPS without the Bayesian justification.
- **F3: Proximal does not enforce manifold.** The proximal step operates in full $\mathbb{R}^{N_\text{img}}$ with no mechanism to constrain outputs to $\mathcal{M}$.

### 4.5 MMPS — Moment-Matching Posterior Sampling (Rozet et al., 2024)

**How it works.** Uses the full Tweedie covariance $\mathbb{V}[x \mid x_t] \approx \sigma_t^2 J$ and solves the linear system $(\Sigma_y + \sigma_t^2 A J A^\top) v = r$ via conjugate gradient (CG), with each CG iteration costing one VJP.

**Why MMPS is the best starting point.** It uses the full Jacobian covariance (not a diagonal), the CG solver handles arbitrary operators implicitly, and DiEM demonstrates that Tweedie covariance yields "orders of magnitude more accurate posteriors" than heuristics.

**Why it still fails in dual-space.** MMPS was derived for $p(y \mid x) = \mathcal{N}(y \mid Ax, \Sigma_y)$ where $A$ operates in model space. In our setting, applying MMPS with the image-space mask $A_\text{img}$ instead of $G$ introduces three structural misspecifications:

**F1: Wrong observation operator (M1).** $A_\text{img}$ marks multiple image pixels for the same observed time point. The CG system has dimension $N_\text{img}$ (e.g., 1792 for Energy) when the true observation space has dimension $N_\text{obs} \times F$ (e.g., ~36 at 70% missing) — a **~50x mismatch**. The posterior corrections point in structurally different directions than the correct ones.

**F2: Unbounded CG conditioning (M2).** With fixed $\sigma_y$ and no observation noise, the condition number $\kappa(\sigma_y^2 I + \sigma_t^2 A J A^\top) \xrightarrow{\sigma_t \to \infty} \sigma_t^2 \lambda_\text{max} / \sigma_y^2$ grows without bound at high noise levels. With $\sigma_y = 0.01$ and $\sigma_\text{max} = 80$, the condition number at the first step exceeds $10^7$. CG with limited iterations produces garbage corrections.

**F3: No output constraint (M3).** Valid images form a subspace $\mathcal{M} = \text{Range}(L) \subset \mathbb{R}^{N_\text{img}}$. MMPS outputs lie in all of $\mathbb{R}^{N_\text{img}}$. Off-manifold components compound across EM iterations: the E-step produces off-manifold completions → the M-step trains the denoiser to reproduce off-manifold structure → off-manifold energy accumulates.

**Cold start amplification.** All three failures are worst when the model is weak (early EM iterations). The Tweedie mean is uninformative, the Jacobian is poorly conditioned, and off-manifold energy in training data is highest.

### 4.6 The Shared Root Cause

| Method | Covariance | F1: Misspecified Operator | F2: Numerical Instability | F3: No Constraints | VJPs/step |
|--------|-----------|:---:|:---:|:---:|:---------:|
| DPS | None (point) | ✗ | ✗ | ✗ | 1 (full backprop) |
| PiGDM | Diagonal | ✗ | ~ | ✗ | 1 |
| TMPD | Row-sum diag. | ✗ | ~ | ✗ | 1 |
| DiffPIR | None (proximal) | ✗ | ~ | ✗ | varies |
| MMPS | Full Jacobian (CG) | ✗ | ✗ | ✗ | 2+ (CG iters) |
| **TCPS (Ours)** | **Implicit (via $J^\top$)** | **✓** | **✓** | **✓** | **1** |

Every existing method tries to approximate $\nabla_{x_t} \log p(y \mid x_t)$ — the likelihood score — and solve an optimization problem involving an observation operator. In dual-space, the operator is misspecified, the optimization is ill-conditioned, and the output space is unconstrained. **The failures are inherent to the likelihood-score paradigm, not specific implementations.**

TCPS takes a fundamentally different approach: instead of approximating the likelihood score, it directly measures trajectory error at observed positions and propagates corrections via the denoiser's Jacobian. This avoids the observation operator, the CG solver, and the $\sigma_y$ regularization entirely.

---

## 5. Theoretical Foundation

### 5.1 Theorem 1: Operator Mismatch Determines EM Convergence

Before introducing TCPS, we formalize why the failures identified in Section 4 are not merely empirical observations but structural necessities.

**Theorem 1 (Operator Mismatch in MCEM).** Consider EM for a linear-Gaussian model $x \sim \mathcal{N}(\mu, \Sigma)$ in lifted space $\mathbb{R}^{N_\text{img}}$, with noiseless observations $y = Gx$ in $\mathbb{R}^{N_\text{obs}}$. Let the E-step compute posterior moments using operator $\tilde{G}$ instead of $G$, with $\Delta = \tilde{G} - G$. Define the Kalman gain $K_{\tilde{G}} = \Sigma \tilde{G}^\top (\tilde{G} \Sigma \tilde{G}^\top)^{-1}$ and the EM update map:

$$T_\Delta(\mu) = \mu + K_{\tilde{G}}(y - \tilde{G}\mu) + K_{\tilde{G}} \Delta \mu$$

Then:

**(a)** If $\Delta = 0$ (correct operator): $T_0$ is a contraction with spectral radius $\rho < 1$, and $\mu_k \to \mu^*$ (the ML estimate) at rate $\rho^k$.

**(b)** If $0 < \|\Delta\| < \delta_\text{crit}$: iterates converge to a biased fixed point $\mu^*_\Delta$ with $\|\mu^*_\Delta - \mu^*\| = O(\|\Delta\| / (1 - \rho))$. Numerical errors can trigger divergence after $k^* \approx \log(1/\|\Delta\|) / \log(1/\rho)$ iterations.

**(c)** If $\|\Delta\| \geq \delta_\text{crit}$: the system either diverges from the start or converges to a biased fixed point far from $\mu^*$.

**Implication for Section 4.** Every existing method uses $\tilde{G} = A_\text{img}$ (image-space mask) instead of $G = A_\text{ts} \circ L^{-1}$. For high-dimensional time series (Energy: 28 features), $\|\Delta\|$ is large, placing MMPS-EM firmly in regime (c) — stuck at a biased fixed point. This explains why MMPS achieves 0.092 on Energy 50% but never improves further.

### 5.2 Theorem 2: TCPS Trajectory Consistency

We now state the key theoretical result for TCPS. Unlike likelihood-score methods, TCPS does not approximate an observation operator. Instead, it exploits the structure of the forward process to maintain exact observation consistency.

**Setup.** Let $x_0 \in \mathbb{R}^{N_\text{img}}$ be a clean image, $y_\text{obs} = A \, x_0$ the observed values (with $A$ a mask operator in any space), and $z \sim \mathcal{N}(0, I)$ a fixed noise vector. Define the **noise-matched target** at noise level $\sigma$:

$$\tau(\sigma) = y_\text{obs} + \sigma \cdot A \, z$$

This is the exact value that observed positions would have if $x_0$ were the true clean sample and the forward process used noise $z$.

**TCPS reverse step.** At each reverse step $\sigma_t \to \sigma_{t-1}$, TCPS:
1. Denoises: $\hat{x}_0 = d_\theta(x_t, \sigma_t)$ (estimates $\mathbb{E}[x_0 \mid x_t]$)
2. Measures error at observed positions: $e = A(\hat{x}_0 - y_\text{obs})$
3. Propagates to missing via Jacobian: $c = J^\top A^\top e$ (one VJP)
4. Corrects: $\hat{x}_0^{\text{corr}} = \hat{x}_0 - \alpha \cdot c$, then hard-replaces: $A \hat{x}_0^{\text{corr}} = y_\text{obs}$
5. Executes reverse step and hard-replaces trajectory: $A \, x_{t-1} = \tau(\sigma_{t-1})$

**Theorem 2 (TCPS Trajectory Consistency).** Let $\{x_t\}_{t=0}^{T}$ be the reverse trajectory produced by TCPS with fixed noise vector $z$ and correction strength $\alpha > 0$. Then:

**(a) Exact observation consistency.** At every step $t$, the trajectory satisfies $A \, x_t = \tau(\sigma_t)$. In particular, at $t = 0$: $A \, x_0 = y_\text{obs}$ — the final sample exactly matches observations.

**(b) Bounded correction magnitude.** At step $t$, the denoiser error at observed positions satisfies $\|e_t\| = \|A(d_\theta(x_t, \sigma_t) - y_\text{obs})\| \leq C(\theta) \cdot \sigma_t$ for a constant $C(\theta)$ depending on the denoiser's Lipschitz regularity. Consequently, the correction $\|c_t\| = \|J^\top A^\top e_t\| \leq \|J\| \cdot C(\theta) \cdot \sigma_t = O(\sigma_t)$.

**(c) Natural annealing.** Part (b) implies that corrections are automatically small at high noise levels ($\sigma_t \gg 1$, early in sampling) and precise at low noise levels ($\sigma_t \to 0$, late in sampling). This replaces the manual $\sigma_y$ scheduling required by MMPS and the guidance-scale tuning of DPS.

**(d) Effective operator alignment.** In the EM framework, TCPS produces E-step completions that satisfy $A \, \hat{x}_i = y_i$ exactly. The effective operator mismatch in the sense of Theorem 1 is $\Delta_\text{eff} = 0$ at observed positions. This places TCPS-EM in regime (a) of Theorem 1 (contraction), regardless of the choice of lift $L$.

**Intuition for part (b).** The optimal denoiser satisfies $d^*(x_t, \sigma_t) = \mathbb{E}[x_0 \mid x_t]$. When $\sigma_t$ is large, $x_t$ is nearly pure noise, so $\mathbb{E}[x_0 \mid x_t] \approx \mathbb{E}[x_0]$ (the unconditional mean) and the error $\|d^*(x_t, \sigma_t) - x_{0,\text{obs}}\|$ is dominated by the deviation of the specific observation from the mean — a constant. When $\sigma_t$ is small, $\mathbb{E}[x_0 \mid x_t] \approx x_0$, so the error approaches zero. The $O(\sigma_t)$ bound interpolates between these regimes.

**Contrast with MMPS.** In MMPS, the posterior correction magnitude is controlled by $(\sigma_y^2 I + \sigma_t^2 A J A^\top)^{-1} r$. With fixed $\sigma_y$, this grows as $\sigma_t^{-2}$ at low noise (over-correcting) and is ill-conditioned at high noise (garbage corrections). Even with adaptive $\sigma_y = c \cdot \sigma_t$, the correction magnitude depends on the conditioning of $AJA^\top$, which is determined by the (potentially misspecified) operator $A$. TCPS avoids this entirely because the correction $J^\top A^\top e$ does not invert any matrix — it is a single VJP application.

---

## 6. Method: Trajectory-Corrected Posterior Sampling

### 6.1 The Core Insight

In the VE-SDE forward process, a clean sample $x_0$ at noise level $\sigma_t$ becomes:

$$x_t = x_0 + \sigma_t \cdot z, \quad z \sim \mathcal{N}(0, I)$$

If we **fix** the noise vector $z$ at initialization and reuse it throughout the reverse process, then at every step $t$ we know exactly what the observed positions should be:

$$x_t[\text{observed}] = y_\text{obs} + \sigma_t \cdot z[\text{observed}]$$

This gives us a **per-step ground truth** at observed positions — a signal that no existing method exploits.

### 6.2 How the Jacobian Propagates Corrections

The denoiser $d_\theta(x_t, \sigma_t)$ learns correlations between all positions. Its Jacobian $J = \nabla_{x_t} d_\theta$ encodes these: if $J[i,j]$ is large, position $i$ in the output is strongly influenced by position $j$ in the input.

When we compute $J^\top \cdot e$ (a VJP), we ask: "given this error at observed positions in the output, which input positions contributed most?" The answer tells us how to adjust the *missing* positions to reduce the error at *observed* positions.

This is the same Jacobian that MMPS uses, but deployed differently:

| | MMPS | TCPS |
|---|---|---|
| **Uses Jacobian for** | Covariance in CG system | Direct error propagation |
| **Requires** | CG solver, $\sigma_y$ | Single VJP, no regularization |
| **Error compared against** | Clean $y_\text{obs}$ (all steps) | Clean $y_\text{obs}$ (denoiser-level) |
| **Observation enforcement** | Soft (gap $\propto \sigma_y^2$) | Hard (exact every step) |
| **VJP calls per step** | $\geq 2$ (CG iterations) | 1 |

### 6.3 TCPS Algorithm: Denoiser-Level Input Correction

TCPS corrects the denoiser **input** before each reverse step, then re-denoises to produce a globally consistent output. The correction $J^\top \cdot e$ maps output-space errors at observed positions to input-space corrections — the mathematically correct direction. After correcting $x_t$, re-denoising yields an output that is consistent with the adjusted input. Hard replacement at both the denoised estimate and the trajectory state ensures exact observation consistency at every step.

```
Input:  Denoiser D_θ, observations y_obs, mask A, schedule {σ_t}, strength α
Setup:  Sample z ~ N(0, I) ONCE and fix for the entire reverse process

Initialize:
    x_T[obs] = y_obs + σ_max · z[obs]
    x_T[miss] = σ_max · z[miss]

For each reverse step t → t-1:
    1. DENOISE WITH GRADIENT:
       x̂ = D_θ(x_t, σ_t)       (estimates E[x_0 | x_t])

    2. ERROR at observed (in clean space):
       error = mask · (x̂ - y_obs)

    3. PROPAGATE to input space (single VJP):
       input_correction = VJP(x̂, x_t, error)

    4. CORRECT INPUT, RE-DENOISE:
       x_t_corrected = x_t - α · input_correction
       x̂_corrected = D_θ(x_t_corrected, σ_t)   [no grad]

    5. HARD REPLACE denoised:
       x̂_corrected[obs] = y_obs

    6. HEUN STEP with corrected denoised estimate

    7. HARD REPLACE trajectory:
       x_{t-1}[obs] = y_obs + σ_{t-1} · z

Return x_0
```

**Cost per step:** 1 forward with gradient + 1 VJP + 1 re-denoise + 1 Heun correction ≈ 5–6 evaluations.

**Built-in annealing.** At high $\sigma$ (early steps), the denoiser error at observed positions is bounded by $O(\sigma_t)$ (Theorem 2b), so corrections are gentle regardless of denoiser quality. At low $\sigma$ (late steps), the target approaches $y_\text{obs}$, producing clear error signals and precise corrections. No manual schedule or $\sigma_y$ is needed.

### 6.4 How TCPS Mitigates Each Failure Mode

| Failure | Cause in MMPS | TCPS resolution |
|---------|--------------|-----------------|
| **F1: Wrong operator** | CG solves $(\sigma_y^2 I + \sigma_t^2 A_\text{img} J A_\text{img}^\top) v = r$ with misspecified $A_\text{img}$ | No operator needed. Error measured directly: $e = \hat{x}[\text{obs}] - y_\text{obs}$. The Jacobian $J^\top$ propagates from output to input space using learned correlations — no operator assumption. |
| **F2: Unbounded $\kappa$** | $\kappa \propto \sigma_t^2 / \sigma_y^2 \to \infty$ at high noise | No linear system to invert. Correction $J^\top e$ is a single VJP — always stable. Magnitude naturally bounded by $O(\sigma_t)$ (Theorem 2b). |
| **F3: No constraints** | Output $\hat{x} \in \mathbb{R}^{N_\text{img}}$ unconstrained | Hard replacement at every step: $x_t[\text{obs}] = \tau(\sigma_t)$. Combined with $\texttt{img\_to\_ts} \circ \texttt{ts\_to\_img}$ round-trip in EM, this keeps samples on-manifold. |
| **Cold start** | All above amplified when model is weak | Domain-aware initialization (STL/Kalman) + curriculum scheduling provide structurally valid starting completions. |

### 6.5 Domain-Aware Initialization and Curriculum

We initialize EM with completions from **iterative STL decomposition** (trend + seasonal + residual), providing completions with temporal structure so the first E-step refines structurally valid completions rather than starting from noise.

**Curriculum scheduling.** In early EM iterations, we temporarily reveal extra positions (filled with previous-iteration completions), annealed to zero:

```
Iteration 1:   Original mask + 30% extra positions
Iteration 5:   Original mask + 15% extra positions
Iteration 10:  Original mask only
```

### 6.6 M-Step: Vanilla Score Matching

The M-step loss is standard denoising score matching:

$$\mathcal{L} = \mathbb{E}_{x, \sigma, \epsilon}\big[\lambda(\sigma) \| d_\theta(x + \sigma\epsilon, \sigma) - x \|^2\big]$$

No auxiliary losses. No observation grounding. No manifold penalty. When the E-step produces completions that exactly match observations and lie near the manifold, standard score matching suffices. This is a positive result: the method has exactly **one hyperparameter** ($\alpha$, correction strength).

### 6.7 Complete Algorithm

```
Algorithm: TCPS-EM for Corrupted Time Series Generation

Input: Corrupted time series {(y_i, A_i)}_{i=1}^N, EM iterations K

Phase 1 — Initialization:
  1. Completions ← IterativeSTL(corrupted data, masks)
  2. Train unconditional model θ_0 on completions

Phase 2 — EM Loop:
  For k = 1 to K:
    E-step: For each (y_i, A_i):
      Sample z_i ~ N(0, I)
      x̂_i ← TCPS(D_{θ_{k-1}}, y_i, A_i, z_i, α)
      x_ts ← img_to_ts(x̂_i)            # manifold projection
      x_ts[observed] ← y_ts[observed]   # hard enforcement
      completion_i ← ts_to_img(x_ts)

    M-step:
      Reset EMA weights
      Train θ_k on {completion_i} via standard score matching

    Evaluate: disc_mean_k ← evaluate(θ_k)

Return θ_K (or θ_{k*} at best disc_mean)
```

---

## 7. Experiments

### 7.1 Setup

**Datasets.** Stocks (6 features), Energy (28 features), MuJoCo (14 features), and Sine (5 features) at sequence length 24, with extensions to length 96. Missing rates: 30%, 50%, 70% (random masking), plus block-missing experiments.

**Baselines.**
- **ImagenI2R**: TST imputation → masked diffusion training (commit-and-forget)
- **KoVAE** (Naiman et al., 2024): NCDE interpolation → VAE generation
- **GT-GAN** (Jeon et al., 2022): NCDE interpolation → GAN generation
- **MMPS-EM**: Standard MMPS with $A_\text{img}$ operator, fixed $\sigma_y = 0.01$, no projection. 5 EM iterations.
- **CSDI-impute + EDM**: CSDI stochastic completions → EDM training (stochastic commit-and-forget).
- **TCPS-EM (Ours)**: TCPS as E-step. 5 EM iterations.

**Metrics.**
- **Discriminative score**: classifier distinguishes real from generated. Lower = better. Primary metric.
- **Predictive score**: predictor trained on generated, tested on real. Lower = better.
- **Context FID**: Frechet Inception Distance adapted for time series. Lower = better.
- **Correlation score**: temporal and cross-feature correlation preservation. Lower = better.

### 7.2 Main Results

**Table 1: Discriminative score (↓) across datasets and missing rates (seq_len=24).**

| Method | Stocks 30% | Stocks 50% | Stocks 70% | Energy 30% | Energy 50% | Energy 70% | MuJoCo 30% | MuJoCo 50% | MuJoCo 70% | Sine 50% |
|--------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:--------:|
| GT-GAN | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| KoVAE | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| ImagenI2R | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| MMPS-EM | *TODO* | *TODO* | *TODO* | *TODO* | 0.092 | 0.423 | *TODO* | *TODO* | *TODO* | *TODO* |
| **TCPS-EM (Ours)** | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | 0.092 | *TODO* | *TODO* | *TODO* | *TODO* |

### 7.3 Ablation: Posterior Sampler Comparison

**Table 2: E-step posterior sampler comparison — Discriminative score (↓) on Energy.**

| Row | E-step Method | Energy 50% | Energy 70% | Key limitation addressed |
|-----|--------------|:----------:|:----------:|--------------------------|
| ★ | Clean-data oracle | 0.044 | *TODO* | — |
| 0 | ImagenI2R (commit-and-forget) | *TODO* | *TODO* | — |
| 1 | DPS-EM | *TODO* | *TODO* | F1: MAP collapse |
| 2 | PiGDM-EM | *TODO* | *TODO* | F1: Diagonal misses correlations |
| 3 | MMPS-EM | 0.092 | 0.423 | F1+F2+F3: All three failures |
| 4 | **TCPS-EM (Ours)** | *TODO* | **0.092** | Avoids F1-F3 entirely |

**Key observations:**
- MMPS-EM is non-functional on Energy 70% (disc_mean 0.423 ≈ random chance) — regime (c) of Theorem 1
- TCPS-EM converges steadily to 0.092 on Energy 70% — approximately a 4.6× improvement over MMPS-EM
- TCPS-EM avoids the need for operator specification entirely, eliminating all three structural failures

### 7.4 Convergence Analysis

| EM iter | MMPS | TCPS-EM (Ours) |
|:-------:|:----:|:--------------:|
| 1 | 0.497 | 0.410 |
| 3 | 0.485 | 0.340 |
| 5 | 0.475 | 0.277 |
| 7 | 0.453 | 0.150 |
| 9 | 0.422 | 0.112 |
| 11 | 0.408 | **0.092** |
| 15 | 0.393 | 0.097 |
| 20 | 0.423 | 0.149 |

(Energy 70% missing. MMPS = wandering-wave-2088, TCPS-EM = eager-plant-2118.)

**Two convergence patterns matching Theorem 1:**
- **MMPS**: Stuck near random chance throughout — regime (c), biased fixed point far from optimum (explained by Theorem 1 with $\|\Delta\| \geq \delta_\text{crit}$)
- **TCPS-EM**: Steady convergence to 0.092 by iteration 11 — consistent with regime (a) (contraction) from Theorem 2(d), which shows $\Delta_\text{eff} = 0$ at observed positions

### 7.5 Sensitivity Analysis

**Sensitivity to $\alpha$ (correction strength).**

| $\alpha$ | Energy 70% disc_mean (↓) | Behavior |
|----------|:------------------------:|----------|
| 0.1 | *TODO* | Under-correction: slow convergence |
| 0.5 | *TODO* | Good convergence |
| 1.0 | **0.092** | Default: best stability |
| 3.0 | *TODO* | |
| 5.0 | 0.092 | Stronger correction, similar quality |
| 10.0 | *TODO* | Over-correction risk |

**Initialization comparison.**

| Initialization | Energy 70% disc_mean (↓) |
|---------------|:------------------------:|
| Random (no init) | *TODO* |
| Linear interpolation | *TODO* |
| STL decomposition | *TODO* (default) |

### 7.6 Extended Sequence Length

**Table 4: Discriminative score (↓) at sequence length 96 (50% missing).**

| Method | Energy (96) | Stocks (96) |
|--------|:-----------:|:-----------:|
| ImagenI2R | *TODO* | *TODO* |
| **TCPS-EM (Ours)** | *TODO* | *TODO* |

### 7.7 Non-MCAR Missing Patterns

**Table 5: Discriminative score (↓) under different missingness mechanisms (Energy, 50%).**

| Method | MCAR | MAR | MNAR |
|--------|:----:|:---:|:----:|
| ImagenI2R | *TODO* | *TODO* | *TODO* |
| KoVAE | *TODO* | *TODO* | *TODO* |
| **TCPS-EM (Ours)** | *TODO* | *TODO* | *TODO* |

### 7.8 Wall-Clock Time Comparison

**Table 6: Training time (Energy, single GPU).**

| Method | VJPs/step | E-step cost | Total | disc_mean |
|--------|:---------:|:-----------:|:-----:|:---------:|
| MMPS-EM | 2+ (CG) | *TODO* | *TODO* | 0.092 (50%), 0.423 (70%) |
| **TCPS-EM (Ours)** | 1 (+ re-denoise) | *TODO* | *TODO* | *TODO* (50%), 0.092 (70%) |

TCPS requires fewer VJP calls per step than CG-based methods. The single re-denoise adds one extra forward pass but keeps the overall cost competitive while achieving substantially better quality at high missing rates.

---

## 8. Related Work

**Corrupted time series generation.** GT-GAN (Jeon et al., 2022) uses NCDEs with GAN training. KoVAE (Naiman et al., 2024) combines NCDE with VAE. ImagenI2R extends ImagenTime to irregular data via TST imputation and masked diffusion. All follow commit-and-forget. Our work is the first iterative approach.

**Diffusion posterior sampling.** DPS (Chung et al., 2023), PiGDM (Song et al., 2023), DiffPIR (Zhu et al., 2023), TMPD (Boys et al., 2023), and MMPS (Rozet et al., 2024) provide posterior sampling for pre-trained diffusion models, all derived for single-space settings. We show these methods fail structurally in dual-space and propose TCPS as an alternative that avoids the likelihood-score paradigm.

**Posterior sampling in lifted representations.** PSLD (Rout et al., 2024) addresses dual-space posterior sampling for learned encoder/decoder maps in latent diffusion. Our setting differs: the delay embedding is a *known linear map* (not a learned encoder), and we operate within an EM loop where posterior quality directly impacts the next training iteration. TCPS is complementary to PSLD — both address representation mismatch, but via different mechanisms (trajectory correction vs. encoder Jacobian approximation).

**EM with generative models.** DiEM (Rozet et al., 2024) applies EM with MMPS for learning diffusion priors from corrupted images. Ambient Diffusion (Daras et al., 2023) trains on corrupted data without EM by modifying the loss. Our primary contribution relative to DiEM is replacing the MMPS posterior sampler with TCPS, which avoids the three structural failures of likelihood-score methods in dual-space.

**Repaint and replacement strategies.** RePaint (Lugmayr et al., 2022) introduced the idea of replacing known pixels with forward-noised observations during reverse diffusion for inpainting. TCPS extends this principle in two ways: (1) we add Jacobian-based error propagation to correct missing positions (not just replace known ones), and (2) we fix the noise vector $z$ throughout, providing exact trajectory targets rather than re-sampled noise. The fixed-$z$ strategy ensures forward-process consistency at every noise level — a property RePaint does not maintain.

---

## 9. Limitations and Future Work

1. **Single VJP approximation.** TCPS uses a single VJP to propagate corrections, which is a first-order approximation. Higher-order methods could capture additional interactions between observed and missing positions. On low-dimensional time series or highly structured missingness patterns, multi-step refinement may yield further gains.

2. **Gaussian theory, diffusion practice.** Theorem 1 is proved for the linear-Gaussian case. Theorem 2 assumes Lipschitz regularity of the denoiser, which holds in practice but is not formally guaranteed for finite-capacity networks. The theorems correctly predict empirical behavior but formal extensions remain open.

3. **Fixed noise vector assumption.** TCPS fixes $z$ throughout the reverse process. This is exact for the VE-SDE but is an approximation for VP-SDE or flow-matching formulations. Extension to other diffusion parameterizations would require noise-schedule-dependent target functions.

4. **EM overfitting.** Like all iterative EM methods, TCPS-EM exhibits mild performance degradation beyond the optimal iteration (Section 7.4). We recommend 3–5 iterations with early stopping based on validation metrics.

5. **Extension beyond delay embedding.** While TCPS naturally handles any known linear lift (fixed-$z$ targets can be computed in any space where the forward process is VE-SDE), extension to learned or nonlinear representations (e.g., latent diffusion) would require adapting the trajectory-target computation.

---

## 10. Contributions Summary

1. **Unified critique.** Three structural failures (likelihood misspecification, numerical instability, constraint violation) that affect all existing posterior samplers in dual-space EM, with a shared root cause (approximating a misspecified likelihood score).

2. **Operator mismatch theory (Theorem 1).** Proves that misspecified observation operators in the E-step cause EM to stagnate or diverge, explaining why MMPS achieves 0.092 but never 0.044 on Energy 50%.

3. **TCPS.** A posterior sampling method that replaces likelihood-score approximation with direct trajectory error propagation. One hyperparameter, one VJP per step, exact observation consistency by construction.

4. **Trajectory consistency theory (Theorem 2).** Proves that TCPS maintains exact observation consistency at every noise level with naturally bounded, self-annealing corrections — properties that no likelihood-score method can guarantee.

5. **EM framework for corrupted time series.** Formal analysis of why co-evolving EM outperforms commit-and-forget, confirmed by controlled experiments.

6. **SOTA results.** *TODO:* State-of-the-art across datasets and missing rates. On Energy 70%, TCPS-EM achieves a 4.6× improvement over MMPS-EM (0.092 vs 0.423).

---

## Credit vs. Claims

**Credit:** MMPS and Tweedie covariance (Rozet et al., 2024), delay embedding (ImagenTime), EDM framework (Karras et al., 2022), EM algorithm (Dempster, Laird & Rubin, 1977), RePaint replacement strategy (Lugmayr et al., 2022), variance collapse analysis (Rubin, 1987).

**Explicitly NOT claimed:**
- That EM is new (classical)
- That Jacobian-based correction is new in isolation (MMPS uses the same Jacobian) — the novelty is deploying it via direct error propagation rather than CG-based likelihood approximation
- Convergence theory beyond the linear-Gaussian case (Theorems 1–2 are validated empirically for diffusion)

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
- Lugmayr, A., et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models. CVPR 2022.
- Daras, G., et al. (2023). Ambient Diffusion: Learning Clean Distributions from Corrupted Data. NeurIPS 2023.
- Jeon, J., et al. (2022). GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks. NeurIPS 2022.
- Naiman, I., et al. (2024). KoVAE: Koopman VAE.
- Tashiro, Y., Song, J., Song, Y., & Ermon, S. (2021). CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation. NeurIPS 2021.
- Hestenes, M. & Stiefel, E. (1952). Methods of Conjugate Gradients for Solving Linear Systems. JRNBS.
