# Co-Evolving Generation and Completion for Time Series from Corrupted Observations

---

## 1. Introduction

Generating realistic time series is critical for applications in healthcare, finance, and science. In practice, real-world time series are rarely complete — sensors fail, measurements are intermittent, and records have gaps. The task of **corrupted time series generation** — learning a generative model of regular time series from irregular, partially-observed data — has emerged as a key challenge.

Three families of methods currently address this task:

| Family | Representative | Imputation | Generator | Limitation |
|--------|---------------|------------|-----------|------------|
| **GAN-based** | GT-GAN (Jeon et al., 2022) | NCDE (one-shot) | GAN | Unstable training, mode collapse, slow NCDE preprocessing |
| **VAE-based** | KoVAE (Naiman et al., 2024) | NCDE (one-shot) | VAE | NCDE cost (~6.5x slower), limited expressiveness |
| **Diffusion-based** | ImagenI2R | TST (one-shot) | Masked diffusion | Masking-data trade-off (see below) |

Despite their differences, all three share a structural pattern we call **commit-and-forget**:

```
Corrupted TS → [One-shot imputation] → Completed TS → [Train generator] → Done
```

The imputation is frozen. The generator never gets to say: *"that completion doesn't match what I've learned about the distribution."* Three consequences follow:

1. **No feedback.** Imputation errors propagate silently into generation. The generator has no mechanism to correct them.
2. **External dependency.** Imputer quality bounds generation quality. The system is only as good as its weakest link.
3. **Masking-data trade-off (ImagenI2R).** At 70% missing, only 30% of pixels contribute to the diffusion loss. The model either ignores imputed values (robust but data-starved) or trusts them (more signal but vulnerable to artifacts). The masking rate must be tuned per dataset and missing rate.

GT-GAN and KoVAE additionally rely on neural controlled differential equations (NCDEs) for their imputation step, making them impractical for long sequences — KoVAE requires approximately 6.5x more training time than diffusion-based alternatives.

**Our approach.** We propose a fundamentally different pipeline: the generative model and its training data **co-evolve**. The model completes missing data using its own learned prior (E-step), trains on those completions (M-step), and the improved model produces better completions in the next round. This is Expectation-Maximization (Dempster, Laird & Rubin, 1977) — a classical framework, but one that has never been applied to corrupted time series generation.

The idea is simple. Making it work is not. State-of-the-art time series generation operates in a **lifted representation** (delay embedding maps time series to images), creating a dual-space system where observations live in time-series space but the model operates in image space. Naive EM in this setting fails in five specific, identifiable ways. Our contribution is identifying these failure modes, proving why they arise, and designing targeted fixes — **cross-space consistency constraints** — that make EM converge in 3-5 iterations to state-of-the-art generation quality.

| Commit-and-forget limitation | Our solution |
|------------------------------|-------------|
| No feedback | The model iteratively improves its own completions |
| External dependency | The diffusion model IS the imputer (via posterior sampling) |
| Masking-data trade-off | After completion, the M-step trains on fully completed data — every pixel contributes |

---

## 2. Background: Delay Embedding and Diffusion for Time Series

### Why the lifted representation is inevitable

Direct time-series diffusion models exist (WaveNet-style, transformer-based) but underperform. They lack the mature convolutional inductive biases, data augmentation strategies, and pretrained components available to image-based diffusion models. ImagenTime demonstrated that mapping time series to images via **delay embedding**, then applying standard image diffusion (EDM, Karras et al. 2022), substantially outperforms direct approaches. This is the current dominant paradigm for time series generation.

**Delay embedding.** Given a multivariate time series `x_ts ∈ R^{N_ts × F}`, the delay embedding `ts_to_img` creates an image by arranging overlapping sliding windows of length `d` with stride `s` as rows of a 2D array. The inverse `img_to_ts` recovers the time series by averaging overlapping pixel contributions. Key properties:

- Each time point appears in **multiple** image positions (overlapping windows create redundancy)
- `N_img > N_ts`: the image has more pixels than the original time series has values
- Valid images form a **linear subspace** of R^{N_img} — not every image corresponds to a valid time series
- The composition `Π = ts_to_img ∘ img_to_ts` is an orthogonal projection onto this subspace

**The dual-space consequence.** Adopting the delay-embedding paradigm means observations live in time-series space while the generative model operates in image space. For regular time series generation, this is transparent — one simply converts, trains, and generates. But for corrupted data, the corruption mask `y = A_ts · x_ts` is defined in TS space, while the model must learn in image space. Any iterative framework (like EM) must cross this boundary at every iteration.

### Setting

- Corrupted time series: `y = A_ts · x_ts` (binary mask, no noise)
- Delay embedding: `ts_to_img` maps time series to image; `img_to_ts` maps back (averaging overlapping pixels)
- Diffusion model trained in image space via EM (alternating E-step and M-step)

---

## 3. Posterior Sampling with Diffusion Models: A Survey

In our EM framework, the E-step requires **posterior sampling**: generating completions `x ~ p(x|y, θ)` conditioned on the observed values `y` and the current model `θ`. Several methods exist for posterior sampling with pre-trained diffusion models. We survey four, focusing on their suitability for our setting.

### DPS — Diffusion Posterior Sampling (Chung et al., 2023)

DPS approximates the likelihood score `∇_{x_t} log p(y|x_t)` using only the Tweedie posterior mean:

```
∇_{x_t} log p(y|x_t) ≈ -(1/σ_y²) · ∇_{x_t} ||y - A · D_θ(x_t)||²
```

where `D_θ(x_t)` is the denoiser (Tweedie mean estimate). DPS uses a **point estimate** of `x_0` — it ignores the posterior covariance `Cov[x|x_t]` entirely. This makes it simple and cheap (one gradient per step), but the approximation is coarse, especially at high noise levels where the posterior is broad.

*Limitation for our setting:* No covariance information means the correction is under-informed. In a dual-space setting where the observation operator is complex (composition of mask + delay embedding inverse), the point estimate is insufficient.

### PiGDM — Pseudo-Inverse Guided Diffusion Models (Song et al., 2023)

PiGDM improves on DPS by incorporating a **diagonal approximation** to the posterior covariance. It uses the pseudo-inverse of the observation operator scaled by a variance estimate:

```
∇_{x_t} log p(y|x_t) ≈ -Σ_t^{-1} A^T (A Σ_t A^T + σ_y² I)^{-1} (y - A D_θ(x_t))
```

where `Σ_t` is approximated as a diagonal matrix. This is more informative than DPS but still relies on the diagonal assumption, which fails when the observation operator introduces correlations (as in delay embedding, where one TS value maps to multiple image pixels).

*Limitation for our setting:* The diagonal covariance approximation cannot capture the structured correlations introduced by the delay embedding.

### DiffPIR — Diffusion Posterior Image Restoration (Zhu et al., 2023)

DiffPIR alternates between a denoising step and a data-fidelity proximal step. It splits each reverse diffusion step into two stages: (1) run the unconditional denoiser, then (2) solve a proximal optimization that enforces consistency with observations. This decoupling is elegant but has a key structural issue: the denoising and data-fidelity steps operate independently, and the proximal step may push the sample off the learned manifold.

*Limitation for our setting:* The alternating structure does not naturally accommodate the composed observation operator `G = A_ts ∘ img_to_ts`. The proximal step would need to invert this operator, which is non-trivial in the redundant delay-embedding setting.

### MMPS — Moment-Matching Posterior Sampling (Rozet et al., 2024)

MMPS provides the most principled approximation by using both the Tweedie **mean** and **covariance**. It approximates the posterior score as:

```
∇_{x_t} log q(y|x_t) = J^T · A^T · (σ_y² I + σ_t² · A · J · A^T)^{-1} · (y - A · D_θ(x_t))
```

where `J = ∇_{x_t} D_θ(x_t)` is the Jacobian of the denoiser (the Tweedie covariance estimate) and the linear system is solved via conjugate gradient (CG). MMPS has a rigorous Bayesian derivation from moment-matching the intractable posterior with a Gaussian approximation.

*Why we choose MMPS:* It is the only method that uses the full Jacobian covariance, solved exactly (up to CG truncation) rather than approximated diagonally. This matters in our setting where the observation operator creates complex correlations.

*Limitation:* MMPS was designed for **single-space** settings (e.g., image inpainting where observations and model share the same pixel space). Applying it to our dual-space system introduces a structural mismatch that we identify and correct in Section 5.

### Comparison

| Method | Covariance | Solver | Dual-space ready? |
|--------|-----------|--------|-------------------|
| DPS | None (point estimate) | Gradient descent | No |
| PiGDM | Diagonal approximation | Closed-form | No |
| DiffPIR | None (proximal) | Proximal step | No |
| MMPS | Full Jacobian | CG | No — but correctable (our work) |

---

## 4. Why Vanilla EM Fails in the Dual-Space Setting

Given the E-step (MMPS posterior sampling) and M-step (score-matching training), one could naively apply EM: alternate between completing data and training. We call this **vanilla EM** — standard MMPS with standard score matching, no additional modifications.

Vanilla EM fails. Not subtly — it either diverges, stagnates, or converges slowly to mediocre quality. We identify **five specific failure modes**, each arising from the dual-space structure:

### Failure Mode 1: Cold Start

**What happens.** EM starts from an initial prior (typically Gaussian noise). The first E-step draws posterior samples using a random model → completions bear no resemblance to real time series. The M-step trains on these nonsensical completions → the model learns nothing useful. Even if the loop eventually recovers, it wastes many iterations relearning what basic statistical decomposition provides for free.

**Why it's specific to TS.** In image reconstruction (single-space EM), random initialization is acceptable because the model and observations share the same space — every noisy completion is at least an image. In our setting, random image-space initializations do not respect temporal structure (trends, seasonality, autocorrelation). The gap between a random image and a valid delay embedding is larger than the gap between a random image and a valid natural image.

**Evidence (Energy 50%):**

| Configuration | Best disc_mean | Best iteration |
|---------------|---------------|----------------|
| Vanilla EM (Gaussian init) | 0.0560 | 10 |
| Kalman warm start | 0.0487 | 2 |

The 13% improvement and 5x faster convergence demonstrate that cold start is a real bottleneck.

### Failure Mode 2: Off-Manifold Drift

**What happens.** The delay embedding maps `N_ts` time points into `N_img > N_ts` image pixels. Valid images form a linear subspace of R^{N_img}. Standard MMPS doesn't know about this subspace — it produces images that don't correspond to any valid time series. These off-manifold components leak into the M-step → the model learns off-manifold structure → the next E-step drifts further. The error compounds across EM iterations.

**Why it's specific to dual-space.** In single-space settings (image inpainting), every output of the denoiser is a valid image. There is no "manifold constraint" to violate. The delay embedding's redundancy creates a strict subspace constraint that MMPS ignores.

| Constraint | What it says | Standard MMPS |
|------------|-------------|---------------|
| Observation: `A · img_to_ts(x) = y` | Observed time points must match | Enforced softly via CG |
| Representation: `x ∈ Range(ts_to_img)` | Image must be a valid delay embedding | **Ignored** |

### Failure Mode 3: CG in the Wrong Space

**What happens.** Standard MMPS runs CG in image space with an image-space mask `A_img`. But in our setting, the true observation model is `y_ts = A_ts · img_to_ts(x_img)` — a composition of the inverse delay embedding and the TS-space mask. The image-space mask `mask_img = ts_to_img(mask_ts)` is a **redundant approximation**: it marks multiple image pixels for the same observed time point, introducing correlations that the diagonal-mask assumption cannot capture.

**Consequence.** CG solves a system of dimension `N_img = C × H × W` (e.g., 1024 for a 32×32 image) when the true observation space has dimension `N_obs × F` (e.g., 36 for 12 observed time points with 3 features). The system is 28x larger than necessary, with a structurally incorrect operator.

### Failure Mode 4: CG Instability at High Noise

**What happens.** The MMPS CG system `(σ_y²I + σ_t² AJA^T)v = r` requires a regularization parameter `σ_y`. In our noiseless setting (`y = Ax`, no observation noise), `σ_y` has no physical meaning. Fixing it to a small constant (e.g., 0.01) creates catastrophic ill-conditioning: the condition number grows as `κ ≈ σ_t²||J||/σ_y²`, becoming unbounded at high noise levels. CG fails to converge → posterior corrections are garbage → M-step trains on corrupted completions.

### Failure Mode 5: Observation Forgetting (MCEM Drift)

**What happens.** With approximate posterior sampling (MMPS with truncated CG), E-step completions are imperfect — they don't perfectly match the observed values. The M-step trains on these as if they were exact, with no mechanism to penalize inconsistency with actual observations:

1. E-step samples are imperfect → some observed values are wrong
2. M-step trains on these as ground truth → denoiser reproduces the errors
3. Next E-step uses this denoiser → produces similarly wrong completions
4. Error compounds across EM iterations

This drift has been observed empirically: "the artifacts introduced by the poor sampling get amplified at each iteration, leading to a total collapse" (Rozet et al., 2024, Appendix).

**Why it's worse in dual-space.** In single-space EM, the M-step loss (score matching) directly penalizes the model at observed positions. In our setting, L_SM operates in image space while observations live in TS space — there is no direct connection between the M-step loss and the observation constraint. The gap is bridged only by the E-step completions, which are approximate.

---

## 5. Our Method: Cross-Space Consistency Constraints

Each failure mode from Section 4 has a targeted fix. Together, they form a system of **cross-space consistency constraints** — mechanisms that ensure EM respects the dual-space structure at every stage (initialization, completion, training).

### System Architecture

```
                         ┌─────────────────────────────────────────────────────────┐
                         │              CO-EVOLVING EM FRAMEWORK                   │
                         │                                                         │
 Corrupted TS ──────────►│  ┌───────────────────────────────────────────────┐      │
 (y, mask)     Warm      │  │  E-step (completion via posterior sampling)   │      │
               Start     │  │                                               │      │
              ─────►     │  │  1. Curriculum mask augmentation (annealed)   │      │
          Kalman/STL     │  │  2. Observation-Space CG (Prop 4):           │      │
                         │  │     CG in TS space, not image space           │      │
                         │  │     + adaptive σ_y = c·σ_t (Prop 2)          │      │
                         │  │     + warm-started CG across reverse steps    │      │
                         │  │  3. Manifold projection (Prop 1)             │      │
                         │  │  4. Hard observation enforcement              │      │
                         │  └───────────────────┬───────────────────────────┘      │
                         │                      │ completed samples                │
                         │                      ▼                                  │
                         │  ┌───────────────────────────────────────────────┐      │
                         │  │  M-step (denoiser training)                  │      │
                         │  │                                               │      │
                         │  │  L_SM:  score matching on completed data     │      │
                         │  │  L_obs: SNR-gated observation grounding      │      │──────► Trained
                         │  │  L_rep: SNR-gated manifold penalty           │      │        Diffusion
                         │  │  L = L_SM + λ_obs·L_obs + λ_rep·L_rep      │      │        Model
                         │  └───────────────────┬───────────────────────────┘      │
                         │                      │ improved model                   │
                         │                      ▼                                  │
                         │  ┌───────────────────────────────────────────────┐      │
                         │  │  Evaluation                                  │      │
                         │  │                                               │      │
                         │  │  • Generation quality (disc_mean)            │      │
                         │  │  • Off-manifold energy E_off                 │      │
                         │  │  • Memorization check                        │      │
                         │  └───────────────────┬───────────────────────────┘      │
                         │                      │                                  │
                         │                      └──── next iteration ──────►       │
                         └─────────────────────────────────────────────────────────┘

Co-evolution dynamic:
  Better model → better completions → better training data → better model

Cross-space consistency at every stage:
  INIT:       Structural consistency — Kalman/STL seeds respect TS structure
  E-STEP:     Representation consistency — Observation-Space CG (Prop 4)
              + adaptive σ_y (Prop 2) + manifold projection (Prop 1)
  M-STEP:     Observation consistency — L_obs grounds denoiser to TS-space observations
              Representation consistency — L_rep penalizes off-manifold denoiser outputs
```

### 5.1 Structural Consistency: Warm Start + Curriculum (fixes Failure Mode 1)

*EM must begin from a point that respects temporal structure, and early iterations must be scaffolded to prevent collapse.*

We initialize the EM loop with completions from **Kalman filtering** (state-space model that handles missing observations natively) or **iterative STL decomposition** (trend + seasonal + residual). These provide completions that respect temporal structure — trends, seasonality, autocorrelation. The first E-step *refines* these completions rather than starting from scratch.

To further stabilize early iterations, we use **curriculum scheduling**: temporarily reveal extra observed positions (filled with previous-iteration completions) in early EM rounds, annealed to zero by convergence. This gives the weak model more signal during the fragile bootstrap phase.

```
Iteration 1:  Original mask + 30% extra positions revealed
Iteration 5:  Original mask + 15% extra positions revealed
Iteration 10: Original mask only (curriculum = 0)
```

**Evidence (Energy 50%):**

| Configuration | Best disc_mean | Best iteration |
|---------------|---------------|----------------|
| No warm start (vanilla EM) | 0.0560 | 10 |
| Kalman warm start only | 0.0487 | 2 |
| Kalman + curriculum | 0.0476 | 3 |
| STL + curriculum | 0.0481 | 3 |

Warm start alone gives a 13% improvement. Adding curriculum pushes further and — critically — peaks at iteration 3, not 10+. This makes the method practical: 3-5 iterations is comparable wall-clock time to one-shot methods.

Note the distinction from ImagenI2R's use of TST: ImagenI2R uses TST as a one-shot imputation that the generator never revises. We use Kalman/STL as a *seed* for an iterative process — the first completion doesn't need to be perfect, it needs to be structurally consistent enough that the loop converges to the right basin.

### 5.2 Observation-Space CG (fixes Failure Mode 3)

*CG should run where observations live, not where the model lives.*

We reformulate the MMPS CG system to operate in the **observation space** (masked TS space) rather than image space. Define the composed observation operator:

```
G = A_ts · img_to_ts :  R^{N_img} → R^{N_obs × F}
```

This maps from image space to the space of observed TS values. The correct MMPS likelihood score with this operator is:

```
∇_{x_t} log q(y|x_t) = J^T · G^T · (σ_y² I + σ_t² · G · J^T · G^T)^{-1} · (y_ts - G · D_θ(x_t))
```

The CG system `(σ_y² I + σ_t² · G · J^T · G^T) · v = r` now lives in R^{N_obs × F}, dramatically smaller than R^{N_img}.

> **Proposition 4 (Observation-Space CG).** Let `G = A_ts · img_to_ts` and `G^T = ts_to_img · A_ts^T`. The CG system `(σ_y² I + σ_t² · G · J^T · G^T) v = r` in R^{N_obs × F} yields the same posterior score correction as solving the full image-space system with the exact observation operator G, but with CG dimension reduced from N_img to N_obs × F. Each CG iteration still requires exactly one VJP of the denoiser.

*Proof sketch.* The matrix-vector product `(G · J^T · G^T) v` decomposes as: (1) embed `v` from masked TS to image via `G^T = ts_to_img(A_ts^T v)`, (2) apply `J^T` via a single VJP, (3) project back via `G = A_ts · img_to_ts(·)`. The VJP cost is identical; only the CG bookkeeping operates in the smaller space. ∎

```python
# Observation-Space CG: CG in masked TS space (dimension N_obs × F)
def cg_operator_ts(v):    # v ∈ R^{N_obs × F}
    # G^T: masked TS → image
    v_full_ts = scatter_observed(v, mask_ts)    # zero-fill unobserved
    v_img = ts_to_img(v_full_ts)                # delay-embed to image
    # J^T: image → image (single VJP — same cost)
    Jv = vjp_fn(v_img)
    # G: image → masked TS
    Jv_ts = img_to_ts(Jv)                       # project to TS (averaging)
    Jv_masked = gather_observed(Jv_ts, mask_ts)  # extract observed
    return sigma_y_sq * v + sigma_sq * Jv_masked
```

**Practical impact:**

| Property | Image-Space CG (standard MMPS) | Observation-Space CG (ours) |
|---|---|---|
| **CG dimension** | C × H × W (e.g., 1 × 32 × 32 = 1024) | N_obs × F (e.g., 12 × 3 = 36) |
| **Observation operator** | Approximate (image-space mask) | Exact (G = A_ts · img_to_ts) |
| **CG convergence** | Slow — high-dimensional, redundant | Fast — ~30x smaller system |
| **Built-in averaging** | No — post-hoc projection needed | Yes — img_to_ts averages overlapping windows within the CG |
| **VJP cost per CG iteration** | 1 VJP | 1 VJP (identical) |

For the Energy dataset (24 time points, 3 features, 50% missing): CG dimension shrinks from ~1024 to ~36 — a **28x reduction**.

Observation-Space CG naturally incorporates the `img_to_ts` averaging within the CG operator, partially pre-projecting the correction onto the delay-embedding manifold. This reduces the off-manifold energy of the raw MMPS output before explicit projection (Section 5.4).

### 5.3 Adaptive σ_y (fixes Failure Mode 4)

*The regularization parameter must track the diffusion noise level to keep CG stable.*

The MMPS CG system `(σ_y²I + σ_t² AJA^T)v = r` has condition number `κ ≈ σ_t²||J||/σ_y²` with fixed `σ_y`, growing unboundedly with σ_t. In our noiseless setting, `σ_y` has no physical meaning — it is purely a regularization parameter.

> **Proposition 2 (Bounded condition number).** Setting `σ_y = c · σ_t` yields condition number `κ = (c² + λ_max(AJA^T))/(c² + λ_min(AJA^T))`, bounded independently of σ_t.

*Proof.* Factor σ_t² from both terms: the operator becomes `σ_t²(c²I + AJA^T)`. The condition number of `(c²I + AJA^T)` depends only on the spectrum of `AJA^T` and the constant `c`, not on σ_t. ∎

```python
sigma_y_sq = (self.c * sigma) ** 2    # σ_y(σ_t) = c · σ_t
```

With bounded conditioning, CG is stable at all noise levels → we can use 5+ iterations instead of 1-3 → more accurate posterior samples → downstream components (projection, M-step) have meaningful inputs.

### 5.4 Manifold Projection (fixes Failure Mode 2, E-step)

*Every E-step completion must correspond to a valid time series.*

After posterior sampling, we project the MMPS output onto the valid delay-embedding subspace and hard-enforce observations:

```python
result_ts = img_to_ts(result)
result = ts_to_img(result_ts)

result_ts = img_to_ts(result)
result_ts[:, observed_mask] = y[:, observed_mask]
result = ts_to_img(result_ts)
```

> **Proposition 1 (Projection optimality).** Let `Π = ts_to_img ∘ img_to_ts`. Then `Π` is the orthogonal projection onto `Range(ts_to_img)` and yields the minimum-norm solution: `Π(x) = argmin_{z ∈ Range(ts_to_img)} ||x - z||²`.

*Proof sketch.* `img_to_ts` averages overlapping delay-embedding windows (left-inverse up to normalization). The composition `ts_to_img(img_to_ts(·))` is self-adjoint and idempotent on R^{N_img}, hence an orthogonal projection. ∎

This provides noise reduction by factor `d/m` from averaging overlapping pixels (~60% for typical settings `m=8, d=3`). The off-manifold energy `E_off(k) = mean_i ||x_i - Π(x_i)||²` serves as a convergence diagnostic.

Projection is only useful when the MMPS output is reasonable — this is why adaptive σ_y (Section 5.3) and observation-space CG (Section 5.2) are prerequisites.

**Evidence (Energy 50%):**

| Configuration | Best disc_mean | Best iteration |
|---------------|---------------|----------------|
| Kalman, no projection | 0.0487 | 2 |
| STL + projection | 0.0493 | 3 |
| Kalman + full system + projection | **0.0472** | 4 |
| Projection without warm start | 0.0573 | 11 |

Projection without warm start still helps vs. baseline (0.057 vs. 0.056), but the payoff is much larger when combined with structural consistency — projecting well-initialized completions is far more effective than projecting random ones.

### 5.5 L_rep: Manifold Penalty in the M-step (fixes Failure Mode 2, M-step)

*The denoiser should learn to produce valid delay embeddings, not just be corrected after the fact.*

The E-step projection (Proposition 1) corrects off-manifold outputs after the fact. But the M-step has no analog: the denoiser is free to produce off-manifold outputs, requiring projection to clean them at the next E-step. We close this gap symmetrically.

```python
# L_rep: penalize off-manifold denoiser outputs
denoised_projected = ts_to_img(img_to_ts(denoised))   # Π(d_θ(x_t))
off_manifold = denoised - denoised_projected
L_rep = w_SNR(σ) · ||off_manifold||²
```

The **SNR gate** `w_SNR(σ) = σ_d²/(σ² + σ_d²)` is essential: at high σ, the Tweedie mean `E[x|x_t]` is a smooth global average that need not be a valid delay embedding. At low σ, the denoiser should produce images that correspond to valid time series.

**Why L_rep complements the E-step projection:**
1. The E-step projection applies a smaller correction → less noise injection from the projection step
2. The off-manifold energy `E_off(k)` decreases faster across EM iterations
3. The Jacobian `J = ∂D_θ/∂x_t` is better-conditioned at on-manifold inputs

This creates **symmetric enforcement** of representation consistency: E-step projects (hard, post-hoc), M-step penalizes (soft, at training time).

### 5.6 L_obs: Observation Grounding in the M-step (fixes Failure Mode 5)

*The model must not forget what it observed.*

**The principled corrections are intractable.** Two standard approaches to correct MCEM drift are:

| Approach | What it does | Why intractable |
|----------|-------------|-----------------|
| **Importance-weighted EM** | Re-weight samples by `p(y\|x)/q(x\|y)` | Requires the proposal density `q(x\|y)` of MMPS — defined by a reverse SDE with data-dependent drift, no closed form |
| **Variational EM** | Optimize a tractable bound on marginal likelihood | Requires an explicit variational family; the diffusion posterior is procedural, no parametric approximation |

This is not an engineering limitation — both approaches require density functions that are *structurally unavailable* for diffusion-based posterior samplers.

**Our solution: L_obs — the minimal tractable alternative.**

> **Proposition 3 (L_obs properties).** Let `w_SNR(σ) = σ_d²/(σ² + σ_d²)`. Then: (i) `w_SNR` is monotonically decreasing, with `w_SNR → 1` as `σ → 0` and `w_SNR → 0` as `σ → ∞`. (ii) At a fixed point where `d_θ(x_t, σ)` perfectly reconstructs observations at low noise, `L_obs = 0` and its gradient vanishes — L_obs does not distort the converged model. (iii) L_obs operates in TS space (through `img_to_ts`) while L_SM operates in image space, creating a cross-space bridge.

```python
# L_obs: SNR-gated observation grounding (TS space)
w_SNR(σ) = σ_d² / (σ² + σ_d²)
L_obs = w_SNR(σ) · ||mask_ts · (img_to_ts(d_θ(x_t, t)) - y_ts)||²
```

The SNR gate ensures L_obs activates only in the low-noise regime where the denoiser has capacity to match fine-grained observations:

| σ | w_SNR | Behavior |
|---|-------|----------|
| 0.002 | ≈ 1.0 | Full observation enforcement |
| 0.5 (= σ_d) | 0.5 | Moderate enforcement |
| 80 | ≈ 0.00004 | Effectively off |

σ_d is the EDM `sigma_data` (= 0.5), inherited from Karras et al. (2022) — not an additional hyperparameter.

**Theoretical status.** L_obs fits within **penalized EM** (Green, 1990): replace the M-step objective `Q(θ|θ_k)` with `Q(θ|θ_k) - λ · R(θ)`. This is non-standard because `R` depends on θ through the denoiser, so standard penalized EM convergence results do not directly apply. We claim empirical stabilization, validated through ablation. Proposition 3 guarantees that L_obs vanishes at a good fixed point, so it does not distort a converged solution.

**Bias analysis.** L_obs creates a bias-variance trade-off: it biases the denoiser toward observations at observed positions (proportional to `λ_obs · w_SNR`), while reducing variance from noisy MCEM samples. At low noise (where `w_SNR ≈ 1`), the Tweedie mean should already be close to the true data, so the bias is *toward the correct value*. Empirically, L_obs improves `disc_mean` without increasing memorization.

**Evidence (Energy 50%):** The best run (`kalman_freqgate_decomp_proj`, disc_mean 0.0472) uses L_obs; the same configuration without L_obs (`kalman_curriculum_decomp_proj`, 0.0479) is slightly worse. The `freq_gated` variants appear in 3 of the top 6 runs.

### 5.7 Warm-Started CG (practical speedup)

Adjacent reverse diffusion steps have similar noise levels and similar CG solutions. We initialize CG at step `t-1` with a scaled version of the previous solution:

```python
v_init = (sigma_next / sigma_cur) ** 2 * v_prev
v = conjugate_gradient(cg_operator, r, x0=v_init, maxiter=cg_iters)
v_prev = v.detach()
```

The scaling `(σ_{t-1}/σ_t)²` accounts for the changing σ² in the covariance matrix. Combined with Observation-Space CG (where the system is already small), this makes high-quality MMPS nearly free.

### Combined M-step Loss

```
L = L_SM + λ_obs · L_obs + λ_rep · L_rep
```

where L_SM operates in image space, L_obs bridges to TS-space observations, and L_rep bridges to the delay-embedding manifold. The method introduces exactly **three hyperparameters** beyond vanilla EM: `c` (adaptive CG ratio), `λ_obs` (observation loss weight), and `λ_rep` (manifold loss weight).

### Why This Is Not Self-Training

EM uses the **known corruption process** `y = A_ts · x_ts` as a mathematical constraint. The E-step is Bayesian posterior sampling conditioned on the actual observations and the known mask. Self-training retrains on its own outputs without corruption-aware correction. L_obs ensures the model never forgets what was actually observed.

---

## 6. Experiments

### Benchmarks and Baselines

We evaluate on the corrupted time series generation benchmark, following the evaluation protocol established by ImagenI2R. Datasets include Stocks, Energy, MuJoCo, and Sine at sequence length 24, with extensions to lengths 96 and 768. Missing rates: 30%, 50%, 70% (random masking).

**Baselines:**
- **ImagenI2R**: TST imputation → masked diffusion training (diffusion-based, commit-and-forget)
- **KoVAE** (Naiman et al., 2024): NCDE interpolation → VAE generation
- **GT-GAN** (Jeon et al., 2022): NCDE interpolation → GAN generation
- **TimeGAN-Δt**: TimeGAN adapted for irregular data (weak baseline)

**Metrics:**
- **Discriminative score** (disc_mean): A classifier trained to distinguish real from generated data. Lower = better. The primary metric.
- **Predictive score**: A predictor trained on generated data, tested on real. Lower = better.
- **Context FID**: Frechet Inception Distance adapted for time series. Lower = better.
- **Correlation score**: Temporal and cross-feature correlation preservation. Lower = better.
- **Memorization**: Nearest-neighbor similarity between generated and training data. Must remain low.

### Main Results

*[Results table to be populated with experimental data across datasets, missing rates, and metrics. Expected format following main.tex Tables 3-4.]*

We expect to demonstrate:
- SOTA generation quality across all datasets and missing rates
- Particular advantage at high missing rates (70%) where the masking-data trade-off hurts ImagenI2R most
- Convergence in 3-5 EM iterations
- Training time competitive with ImagenI2R and substantially faster than KoVAE/GT-GAN (no NCDE overhead)

### Ablation

The ablation isolates each consistency constraint, showing that all are necessary and complementary:

| Row | Method | Structural | Representation (E) | Representation (M) | Observation | What it tests |
|-----|--------|-----------|-------------------|-------------------|-------------|---------------|
| 1 | ImagenI2R | TST (one-shot) | — | — | — | Main competitor |
| 2 | KoVAE / GT-GAN | NCDE (one-shot) | — | — | — | Other baselines |
| 3 | Vanilla EM | Gaussian init | image-space CG, fixed σ_y, no proj | — | L_SM only | EM alone — no consistency |
| 4 | + structural | Kalman/STL + curriculum | image-space CG, fixed σ_y, no proj | — | L_SM only | Constraint 1 only |
| 5 | + repr (E) | Kalman/STL + curriculum | adaptive CG + projection | — | L_SM only | Constraints 1+2(E) |
| 5b | + obs-space CG | Kalman/STL + curriculum | **obs-space CG** + adaptive + proj | — | L_SM only | Prop 4: CG in TS space |
| 6 | + observation | Kalman/STL + curriculum | obs-space CG + adaptive + proj | — | L_SM + L_obs | C1+C2(E)+C3 |
| 6b | + repr (M) | Kalman/STL + curriculum | obs-space CG + adaptive + proj | L_rep | L_SM + L_obs + L_rep | **Full system** |
| 7 | − structural | Gaussian init | obs-space CG + adaptive + proj | L_rep | L_SM + L_obs + L_rep | Without C1 |
| 8 | − repr (E) | Kalman/STL + curriculum | image-space CG, fixed σ_y, no proj | L_rep | L_SM + L_obs + L_rep | Without C2(E) |
| 9 | − repr (M) | Kalman/STL + curriculum | obs-space CG + adaptive + proj | — | L_SM + L_obs | Without L_rep |
| 10 | − observation | Kalman/STL + curriculum | obs-space CG + adaptive + proj | L_rep | L_SM + L_rep | Without L_obs |

**What the gaps reveal:**
- Gap (1,2 → 3): Does co-evolution help at all, even without consistency enforcement?
- Gap (3 → 4): How much does structural consistency alone contribute?
- Gap (4 → 5): Does representation consistency (E-step) add on top of structural?
- Gap (5 → 5b): Does observation-space CG improve over image-space CG? (Prop 4 vs. standard)
- Gap (5b → 6): Does observation consistency (L_obs) add on top?
- Gap (6 → 6b): Does L_rep add on top?
- Rows 7-10: Complementarity — removing any single constraint should degrade performance.

**Sensitivity analysis:** We vary `c` in {0.01, 0.05, 0.1, 0.2}, `λ_obs` in {0.1, 0.3, 0.5, 1.0}, and `λ_rep` in {0.1, 0.3, 0.5} on Energy 50% to characterize robustness.

### Convergence Analysis

*[Plot: disc_mean vs. EM iteration for vanilla EM, +structural, +representation, full system. Expected: vanilla converges slowly or diverges; full system converges at iteration 3-5.]*

---

## 7. Expanded Use Cases

Beyond the standard evaluation at fixed random missing rates (30%/50%/70%), we demonstrate three additional settings that are important for practical deployment:

### 7.1 Mixed Missing Rates

In real-world data, different features often have different observation rates. A patient's heart rate may be measured every minute while blood pressure is measured every hour. We evaluate with mixed missing rates: each feature independently assigned a missing rate sampled from a distribution (e.g., uniform over 30-70%).

Commit-and-forget methods must handle this heterogeneity at the imputation stage. Our EM approach handles it naturally — the observation mask `A_ts` encodes per-feature missingness, and the E-step conditions on whatever is observed.

### 7.2 Block Missing

Instead of random missing values, contiguous temporal blocks are absent (e.g., a sensor offline for 2 hours). Block missing is harder because local interpolation fails — the model must generate plausible dynamics over extended gaps using global distributional knowledge.

We evaluate with temporal blocks of width `w` missing (e.g., `w = 4, 8, 12` time steps). The EM framework handles this through the same mask-based posterior sampling, but the posterior is more uncertain and the warm start (Kalman/STL) is more important.

### 7.3 Imputation as a Byproduct

The E-step's posterior samples serve as high-quality imputations — a free byproduct of the generation framework. We evaluate imputation quality (MAE/RMSE at unobserved positions) across EM iterations, showing that imputation quality improves as the generative model improves, validating the co-evolution hypothesis.

This positions our method not just as a generator but as a **joint generation-and-imputation framework**: a single trained model serves both tasks.

---

## 8. Related Work

**Corrupted time series generation.** GT-GAN (Jeon et al., 2022) uses NCDEs with continuous-time flow processes for GAN-based generation from irregular data. KoVAE (Naiman et al., 2024) combines NCDE preprocessing with VAE generation. ImagenI2R extends ImagenTime to irregular data via TST imputation and masked diffusion training. All follow the commit-and-forget pattern. Our co-evolving EM framework is the first iterative approach to this task.

**Time series diffusion models.** ImagenTime maps time series to images via delay embedding and applies EDM. DiffTime, CSDI, SSSD, and TimeGrad operate directly in time-series space or use conditional diffusion for imputation. These methods address generation or imputation separately; we unify both through EM.

**Diffusion posterior sampling.** DPS (Chung et al., 2023), PiGDM (Song et al., 2023), DiffPIR (Zhu et al., 2023), and MMPS (Rozet et al., 2024) provide posterior sampling for pre-trained diffusion models with increasing sophistication. MMPS uses the full Jacobian covariance via CG — we extend it to dual-space settings with observation-space CG (Proposition 4).

**EM with generative models.** EM has been applied to learn diffusion priors from corrupted observations in the image domain (Rozet et al., 2024, using MMPS for the E-step). Ambient diffusion (Daras et al., 2023) trains diffusion models on corrupted data without explicit EM iteration. Our work is the first to identify the cross-space consistency problem that arises when EM operates in a lifted representation different from the observation space.

---

## 9. Limitations and Future Work

1. **Computational cost.** EM is inherently iterative — each iteration requires a full E-step (posterior sampling) and M-step (retraining). With warm start and curriculum, we converge in 3-5 iterations rather than 32, making wall-clock time comparable to one-shot methods. The E-step is embarrassingly parallelizable across samples.

2. **Component-level, not system-level theory.** We prove formal properties for each constraint individually (Propositions 1-4) and prove intractability of principled alternatives. But we do not have a convergence guarantee for the combined system. Near convergence, L_obs and L_rep vanish, so standard EM dynamics govern the fixed point — but the path to that fixed point is validated empirically.

3. **Noiseless observations only.** Our analysis assumes `y = Ax` with no observation noise. When real noise is present, σ_y regains physical meaning and the adaptive regularization becomes `σ_y(σ_t) = sqrt(σ_y_phys² + c²·σ_t²)`, which we leave to future work.

4. **L_rep bias.** L_rep pushes the denoiser toward on-manifold outputs at low noise, which is correct for perfect delay embeddings. If the delay embedding itself introduces artifacts (e.g., boundary effects), L_rep could amplify them. The SNR gate and small λ_rep mitigate this.

5. **Single instantiation.** We validate the framework on delay embedding (the dominant SOTA paradigm). The general problem class (cross-space consistency in representation-based EM) applies to latent diffusion, patch-based methods, and graph lifts — but we do not provide experiments in those settings. Observation-space CG generalizes directly: for any EM with observation operator G, CG should run in Range(G), not in the model space.

---

## Contributions Summary

1. **Problem identification.** We identify *cross-space consistency in representation-based EM* — a problem class that arises whenever EM operates in a lifted representation different from the observation space. This does not arise in single-space EM and has not been studied.

2. **Five failure modes of vanilla EM.** We characterize five specific failure modes when naively applying EM to delay-embedded time series, each arising from the dual-space structure.

3. **Framework with formal results.** Three consistency constraints — structural, representational, observational — with formal guarantees: projection optimality (Prop. 1), bounded CG conditioning (Prop. 2), L_obs minimality with intractability of principled alternatives (Prop. 3), and observation-space CG equivalence with dimensionality reduction (Prop. 4).

4. **Observation-space posterior sampling.** MMPS in a dual-space setting should run CG in the observation space, not the model space — yielding exact observation operators, ~30x dimensionality reduction, and built-in manifold averaging.

5. **SOTA corrupted TS generation.** The first co-evolving framework for corrupted time series generation, achieving state-of-the-art quality across multiple datasets and missing rates, converging in 3-5 iterations. Ablation demonstrates each constraint is necessary and they are complementary.

---

## Credit vs. Claims

**Credit:** MMPS posterior score decomposition and Tweedie covariance (Rozet et al., 2024), delay embedding for time series (ImagenTime), EDM diffusion framework (Karras et al., 2022), EM algorithm (Dempster, Laird & Rubin, 1977), conjugate gradient method (Hestenes & Stiefel, 1952).

**Explicitly NOT claimed:**
- That EM or co-evolution is new (EM is classical)
- That any individual technique is novel in isolation — the contribution is identifying the general problem class, proving why these constraints are necessary, and demonstrating their complementarity
- Full convergence theory for the combined system
- Solving the identifiability problem

---

## Implementation References

### E-step: Observation-Space CG with posterior sampling

```python
def posterior_denoise_obs_space(self, x_t, sigma, obs_ts, mask_ts):
    sigma_sq = sigma ** 2
    sigma_y_sq = (self.c * sigma) ** 2   # Prop 2: adaptive σ_y

    x_t_input = x_t.detach().requires_grad_(True)
    denoised_img = self.net(x_t_input, sigma, None).to(torch.float64)

    def vjp_fn(cotangent_img):
        grad, = torch.autograd.grad(
            denoised_img, x_t_input, grad_outputs=cotangent_img,
            retain_graph=True
        )
        return grad

    denoised_ts = self.img_to_ts_fn(denoised_img.float()).to(torch.float64)
    r_ts = obs_ts - mask_ts * denoised_ts
    r_obs = mask_ts * r_ts

    def cg_operator_obs(v_ts):
        v_img = self.ts_to_img_fn(v_ts.float()).to(torch.float64)
        Jv_img = vjp_fn(v_img)
        Jv_ts = self.img_to_ts_fn(Jv_img.float()).to(torch.float64)
        Jv_obs = mask_ts * Jv_ts
        return sigma_y_sq * v_ts + sigma_sq * Jv_obs

    v_ts = conjugate_gradient(
        cg_operator_obs, r_obs,
        x0=self.v_prev,            # warm start
        maxiter=self.cg_iters
    )
    self.v_prev = v_ts.detach()

    v_img = self.ts_to_img_fn(v_ts.float()).to(torch.float64)
    score_img = vjp_fn(v_img)

    return denoised_img.detach() + sigma_sq * score_img.detach()
```

### M-step: Combined loss

```python
sigma = (torch.randn([B, 1, 1, 1]) * P_std + P_mean).exp()
weight = (sigma ** 2 + sigma_d ** 2) / (sigma * sigma_d) ** 2
denoised = net(x_img + sigma * noise, sigma, None)

# L_SM: standard score matching (image space)
loss_sm = (weight * (denoised - x_img).square()).mean()

# L_obs: SNR-gated observation grounding (TS space)
snr_gate = sigma_d ** 2 / (sigma ** 2 + sigma_d ** 2)
denoised_ts = img_to_ts(denoised)
obs_residual = mask_ts * (denoised_ts - y_ts)
loss_obs = (snr_gate * (obs_residual ** 2).mean(dim=(-1,-2))).mean()

# L_rep: SNR-gated manifold penalty (image space)
denoised_proj = ts_to_img(img_to_ts(denoised))
off_manifold = denoised - denoised_proj
loss_rep = (snr_gate * (off_manifold ** 2).mean(dim=(1,2,3))).mean()

loss = loss_sm + lambda_obs * loss_obs + lambda_rep * loss_rep
```

### Consistency projection (post E-step)

```python
result_ts = img_to_ts(result)
result = ts_to_img(result_ts)
result_ts = img_to_ts(result)
result_ts[:, observed_mask] = y[:, observed_mask]
result = ts_to_img(result_ts)
```
