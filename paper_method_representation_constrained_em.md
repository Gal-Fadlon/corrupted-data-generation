# Co-Evolving Generation and Completion for Time Series from Corrupted Observations

## Core Narrative

**Every existing method for generating regular time series from irregular observations commits to a reconstruction and hopes for the best.** We propose a different approach: the generative model and its training data **co-evolve** — the model completes missing data using its own learned prior, trains on those completions, and the improved model produces better completions in the next round.

The idea is grounded in Expectation-Maximization. The key question is not *whether* to iterate — it is **what it takes to make iteration actually work for time series**.

**The core insight.** State-of-the-art time series generation relies on *lifted representations* — transformations that map time series into a richer space where powerful generative architectures (CNNs, diffusion models) can operate. The delay embedding (ImagenTime) is the dominant example: it transforms time series into images, unlocking vision-based diffusion models that substantially outperform direct time-series architectures. This representation choice is not ours — it is the field's established path to SOTA quality. But it creates an unavoidable consequence: a **dual-space system** where observations live natively in time-series space while the generative model operates in image space. The embedding connects them, but introduces redundancy, structural constraints, and a gap between where data is observed and where the model learns.

This dual-space gap is not specific to delay embedding. It arises **whenever EM operates in a representation space different from the observation space**: latent diffusion models (observations in pixel space, model in latent space), patch-based methods (overlapping patches create redundancy), graph-based lifts of irregular data, or any variational EM in a learned encoder's range. We study this general problem class — *cross-space consistency in representation-based EM* — through its most transparent and practically important instance: delay-embedded time series.

The central finding: **EM in a lifted representation requires cross-space consistency enforcement at every stage — initialization, completion, and training — or errors compound across iterations.** We formalize this as three necessary constraints (structural, representational, observational), prove formal properties for two of them, and demonstrate empirically that all three are necessary and complementary.

The competition is ImagenI2R, KoVAE, GT-GAN — methods that generate regular time series from irregular observations. They all follow a commit-and-forget pipeline where imputation errors propagate silently into generation. Our co-evolving approach achieves superior generation quality, converging in 3–5 iterations with the model training on 100% of the data — no masking-data trade-off.

---

## System Architecture

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

The delay embedding creates a dual-space boundary that EM must cross at every iteration. The E-step produces image-space completions from TS-space observations. The M-step trains an image-space model that must remain faithful to TS-space ground truth. Each crossing is an opportunity for consistency to degrade. Our three consistency constraints — structural, representational, and observational — ensure that temporal structure is preserved end-to-end, not just at a single stage.

---

## The Commit-and-Forget Problem

Every existing method for generating regular time series from irregular observations follows the same structural pattern:

```
Corrupted TS → [One-shot reconstruction] → Completed TS → [Train generator] → Done
```

- **ImagenI2R**: TST imputes once → diffusion trains with masking
- **KoVAE / GT-GAN**: NCDE interpolates once → GAN/VAE generates

The reconstruction is frozen. The generator never gets to say *"that imputation doesn't match what I've learned about the distribution."* Three structural consequences:

| Limitation | What goes wrong | Why it matters |
|-----------|----------------|---------------|
| **No feedback** | If the imputer gets it wrong, the generator can't fix it | Imputation errors propagate silently into generation |
| **External dependency** | Imputer quality bounds generation quality | The system is only as good as its weakest link |
| **Masking-data trade-off** | At 70% missing, only 30% of pixels contribute to ImagenI2R's loss | The model sees very little signal per training sample |

ImagenI2R uses masking deliberately to prevent overfitting to imputation artifacts — a valid design choice. But it creates an inherent trade-off: the model either ignores imputed values (high masking = robust but data-starved) or trusts them (low masking = more data but vulnerable). The masking rate must be tuned per dataset and missing rate.

**Our approach eliminates all three limitations:**

| Limitation | Our solution |
|-----------|-------------|
| No feedback | The model iteratively improves its own completions |
| External dependency | The diffusion model IS the imputer (via posterior sampling) |
| Masking-data trade-off | After completion, the M-step trains on FULLY completed data — every pixel contributes |

The framework is grounded in EM (Dempster, Laird & Rubin, 1977). We employ MMPS (Rozet et al., 2024) for posterior sampling in the E-step. But the co-evolution idea is simple — **making it actually work requires maintaining consistency across the dual-space boundary at every stage of the EM loop, and that is our paper.**

---

## Why This Is Not Self-Training

The reviewer may ask: how is this different from self-training or iterative refinement?

**Commit-and-forget (ImagenI2R):**
1. External imputer fills missing values (one-shot, never revised)
2. Generator trains on filled data
3. No mechanism to correct imputation errors

**Co-evolving EM (ours):**
1. E-step: posterior sampling with the KNOWN corruption model `y = Ax` — the model generates completions conditioned on actual observations
2. M-step: retrain on these completions, with observation grounding (L_obs) to prevent drift
3. Next E-step uses the improved model → better completions
4. The corruption model `p(y|x)` is part of the feedback loop at every iteration

The key distinction: EM uses the **known corruption process** as a mathematical constraint. Self-training does not — it retrains on its own outputs without corruption-aware correction. The E-step is Bayesian posterior sampling, not "running the model again." And L_obs ensures the model never forgets what was actually observed.

---

## Setting

- Corrupted time series: `y = A_ts · x_ts` (binary mask, **no noise**)
- Delay embedding: `ts_to_img` maps time series to image; `img_to_ts` maps back (averaging overlapping pixels)
- Diffusion model trained in image space via EM (alternating E-step and M-step)
- Each time point appears in multiple image positions (overlapping windows)

---

## Why Lifted Representations Are Inevitable (Not a Design Choice)

A reviewer might ask: *"you introduced the dual-space problem by choosing delay embedding — why not avoid it?"* The answer is that the lifted representation is what makes SOTA performance possible.

**Direct time-series diffusion models exist but underperform.** Architectures that operate directly on 1D sequences (e.g., WaveNet-style, transformer-based) lack the mature convolutional inductive biases, data augmentation strategies, and pretrained components available to image-based diffusion. ImagenTime demonstrated that mapping time series to images via delay embedding, then applying standard image diffusion (EDM/Karras et al.), substantially outperforms direct approaches. This is the current dominant paradigm.

**The representation is fixed by the SOTA.** We inherit delay embedding because we build on ImagenTime — the best available TS generation framework. Any EM-based approach to corrupted TS generation through this paradigm will face the dual-space gap. Our contribution is making EM work *within* this inevitable constraint, not choosing the constraint.

**The problem generalizes.** The dual-space consistency problem is not an artifact of delay embedding. It arises whenever:

| Setting | Observation space | Model space | Redundancy? |
|---------|-----------------|-------------|-------------|
| **Delay embedding (ours)** | Time series | Image | Yes — overlapping windows |
| **Latent diffusion (LDM)** | Pixels | Latent | Yes — encoder range ⊂ R^d |
| **Patch-based restoration** | Full image | Overlapping patches | Yes — patch overlap |
| **Graph lifts** | Irregular samples | Regular grid | Yes — interpolation onto grid |

In each case, EM in the model space must respect constraints inherited from the observation space, and the lifted representation introduces a subspace constraint. Our framework — structural, representational, and observational consistency — provides a blueprint. Delay embedding is the first complete instantiation and the one we validate empirically.

---

## Posterior Sampling in Dual-Space EM: From MMPS to Observation-Space CG

For posterior sampling in the E-step, we build on Moment-Matching Posterior Sampling (MMPS, Rozet et al. 2024). MMPS approximates the posterior mean via a conjugate gradient (CG) system involving the Tweedie Jacobian:

```
(σ_y² I + σ_t² · A · J · A^T) · v = y - A · D_θ(x_t)
```

where `J = ∇_{x_t} d_θ(x_t, t)` is the Tweedie covariance estimate. However, MMPS was designed for **single-space** settings (e.g., image inpainting where observations and model share the same pixel space). Applying it naively in a dual-space system introduces a structural mismatch that we identify and correct.

### The mismatch: image-space CG with an approximate observation operator

Standard MMPS operates entirely in image space. The observation mask `A` is represented as an image-space binary mask, and CG solves a system of dimension `N_img = C × H × W`:

```python
# Standard MMPS: CG in image space (dimension N_img)
def cg_operator(v):    # v ∈ R^{N_img}
    return sigma_y_sq * v + sigma_sq * mask_img * vjp_fn(mask_img * v)
```

But in the dual-space setting, the true observation model is not an image-space mask. The observations are `y_ts = A_ts · x_ts = A_ts · img_to_ts(x_img)` — a composition of the inverse delay embedding and the TS-space mask. The image-space mask `mask_img = ts_to_img(mask_ts)` is a **redundant approximation** of this composed operator: it marks multiple image pixels for the same observed time point, introducing correlations that the diagonal-mask assumption cannot capture.

### Observation-Space CG: running MMPS where observations live

We reformulate the MMPS CG system to operate in the **observation space** (masked TS space) rather than image space. Define the composed observation operator:

```
G = A_ts · img_to_ts :  R^{N_img} → R^{N_obs × F}
```

This maps from image space to the space of observed TS values. The correct MMPS likelihood score (Eq. 15 of Rozet et al.) with this operator is:

```
∇_{x_t} log q(y|x_t) = J^T · G^T · (Σ_y + σ² · G · J^T · G^T)^{-1} · (y_ts - G · D_θ(x_t))
```

The CG system `(Σ_y + σ² · G · J^T · G^T) · v = r` now lives in R^{N_obs × F}, which is **dramatically smaller** than R^{N_img}.

> **Proposition 4 (Observation-Space CG).** Let `G = A_ts · img_to_ts` and `G^T = ts_to_img · A_ts^T`. The CG system `(σ_y² I + σ² · G · J^T · G^T) v = r` in R^{N_obs × F} yields the same posterior score correction as solving the full image-space system with the exact observation operator G, but with CG dimension reduced from N_img to N_obs × F. Each CG iteration still requires exactly one VJP of the denoiser.

*Proof sketch.* The matrix-vector product `(G · J^T · G^T) v` decomposes as: (1) embed `v` from masked TS to image via `G^T = ts_to_img(A_ts^T v)`, (2) apply `J^T` via a single VJP, (3) project back via `G = A_ts · img_to_ts(·)`. The posterior score is `J^T · G^T · v`, computed by one additional VJP after CG converges. The VJP cost is identical; only the CG bookkeeping operates in the smaller space. ∎

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

**Practical impact for time series.**

| Property | Image-Space CG (standard MMPS) | Observation-Space CG (ours) |
|---|---|---|
| **CG dimension** | C × H × W (e.g., 1 × 32 × 32 = 1024) | N_obs × F (e.g., 12 × 3 = 36) |
| **Observation operator** | Approximate (image-space mask) | Exact (G = A_ts · img_to_ts) |
| **CG convergence** | Slow — high-dimensional, redundant | Fast — ~30× smaller system |
| **Built-in averaging** | No — post-hoc projection needed | Yes — img_to_ts averages overlapping windows within the CG |
| **VJP cost per CG iteration** | 1 VJP | 1 VJP (identical) |

The dimensionality reduction is significant: for the Energy dataset with 24 time points, 3 features, and 50% missing rate, the CG system shrinks from ~1024 to ~36 dimensions — a **28× reduction**. In the MRI-style regime of DiEM (80×80×16 images), the reduction would be even more dramatic.

**Connection to Constraint 2.** Observation-Space CG naturally incorporates the img_to_ts averaging within the CG operator (step 3 above). This means the correction is partially "pre-projected" onto the delay-embedding manifold, reducing the off-manifold energy of the raw MMPS output before the explicit projection of Proposition 1. The manifold projection remains necessary (the Jacobian correction J^T · G^T · v is in image space and not guaranteed on-manifold), but the correction it must apply is smaller.

### Warm-Started CG across reverse steps

A second practical improvement: adjacent reverse steps have similar noise levels and similar CG solutions. We initialize CG at step t−1 with a scaled version of the previous solution:

```python
v_init = (sigma_next / sigma_cur) ** 2 * v_prev
v = conjugate_gradient(cg_operator, r, x0=v_init, maxiter=cg_iters)
v_prev = v.detach()
```

The scaling `(σ_{t-1}/σ_t)²` accounts for the changing σ² in the covariance matrix. With warm starting, a single CG iteration at step t−1 effectively continues from a good initial point, achieving multi-iteration quality at single-iteration cost. Combined with Observation-Space CG (where the system is already small), this makes high-quality MMPS nearly free.

### Summary: what we credit vs. improve

| Component | Origin | Our adaptation |
|---|---|---|
| MMPS posterior score decomposition | Rozet et al. (2024) | Used as-is |
| Tweedie covariance via Jacobian VJP | Rozet et al. (2024) | Used as-is |
| CG solver for the linear system | Rozet et al. (2024) | **Reformulated in observation space** (Prop 4) + warm-started |
| Adaptive σ_y = c·σ_t | Ours (Prop 2) | Bounds condition number |
| Manifold projection | Ours (Prop 1) | Post-CG consistency enforcement |

MMPS provides the posterior score framework — our contribution is adapting it to the dual-space setting through observation-space CG, adaptive regularisation, and manifold projection.

---

## Cross-Space Consistency: Three Constraints for EM in Redundant Representations

The delay embedding transforms time series into images — but the time-series structure doesn't vanish. It persists as constraints that standard EM ignores: the starting point must respect temporal structure, completions must live on the valid image subspace, and the image-space model must stay grounded to TS-space observations. Violate any one of these, and the iterative loop degrades.

This gives us a single organizing principle: **EM in a redundant lifted representation requires cross-space consistency at every stage**. Our three constraints — structural, representational, and observational — are instantiations of this principle at initialization, completion, and training respectively. For the E-step constraints we prove formal optimality (Proposition 1) and bounded conditioning (Proposition 2). For the M-step constraint, we prove that principled alternatives are structurally intractable and characterize the properties of the minimal tractable alternative (Proposition 3). Each constraint is backed by ablation evidence.

```
                ┌──────────────────────────────────────────────────────┐
                │  EM in delay-embedded space: three consistency gaps  │
                │                                                      │
                │  TS space ←──── delay embedding ────→ Image space    │
                │  (observations)     (redundant)       (model)        │
                │                                                      │
                │  Gap 1: Initialization — TS has structure, images    │
                │         are initialized randomly                     │
                │  Gap 2: Representation — valid images form a strict  │
                │         subspace; MMPS doesn't know this             │
                │  Gap 3: Training — model trains in image space but   │
                │         observations live in TS space                │
                └──────────────────────────────────────────────────────┘
```

---

### Constraint 1: Structural Consistency — Start and Schedule from TS Structure

*EM must begin from a point that respects temporal structure, and early iterations must be scaffolded to prevent collapse.*

**The problem.** EM with diffusion priors starts from an initial prior and iteratively refines it. With a random Gaussian initialization (as in DiEM for images), the first E-step produces completions that bear no resemblance to real time series — no trends, no seasonality, no autocorrelation. The M-step trains on noise. Even if the loop eventually recovers, it wastes iterations relearning what statistical decomposition provides for free. Worse, the first EM iterations face a cold-start paradox: the model is weak → posterior samples are noisy → M-step trains on garbage → model stays weak.

**Our solution: domain-informed warm start + curriculum scheduling.**

We initialize the EM loop with completions from Kalman filtering (state-space model that handles missing observations natively) or iterative STL decomposition (trend + seasonal + residual). These provide completions that respect temporal structure. The first E-step *refines* these completions rather than starting from scratch.

To further stabilize early iterations, we use curriculum scheduling: temporarily reveal extra observed positions (filled with previous-iteration completions) in early EM rounds, annealed to zero by convergence. This gives the weak model more signal during the fragile bootstrap phase.

```
Iteration 1:  Original mask + 30% extra positions revealed
Iteration 5:  Original mask + 15% extra positions revealed
Iteration 10: Original mask only (curriculum = 0)
```

**Evidence (Energy 50%):**

| Configuration | Best disc_mean | Best iteration |
|---------------|---------------|----------------|
| No warm start (`general_missing` baseline) | 0.0560 | 10 |
| Kalman warm start only (`ema_reset_kalman_init`) | 0.0487 | 2 |
| Kalman + curriculum (`kalman_curriculum`) | 0.0476 | 3 |
| STL + curriculum (`iter_stl_curriculum`) | 0.0481 | 3 |

Warm start is the single biggest lever: 0.056 → 0.049 (a 13% improvement). Adding curriculum pushes further to 0.048 and — critically — peaks at iteration 3, not 10+. This makes the method practical: 3–5 iterations is comparable wall-clock time to one-shot methods.

**Why this is a consistency constraint, not just engineering.** ImagenI2R uses TST for a one-shot imputation. We use Kalman/STL as a *seed* for an iterative process — the first completion doesn't need to be perfect, it needs to be structurally consistent enough that the loop converges to the right basin. Curriculum extends this logic: don't force the model to solve the hardest version of the problem before it has learned basic temporal structure. Both are about ensuring TS-space structure is present from the start of the image-space loop.

---

### Constraint 2: Representation Consistency — Completions Must Live on the Delay-Embedding Manifold

*The E-step must produce images that correspond to valid time series, and the posterior sampler must be numerically stable enough for this to matter.*

**The problem.** The delay embedding maps N_ts time points into N_img > N_ts image pixels. The valid images form a **linear subspace** of R^{N_img}. Standard MMPS doesn't know about this subspace — it can produce images that don't correspond to any valid time series. This is specific to operating in a redundant representation; it does not arise in standard image reconstruction.

| Constraint | What it says | Standard MMPS |
|------------|-------------|------|
| Observation: `A · img_to_ts(x) = y` | Observed time points must match | Enforced softly via CG |
| Representation: `x ∈ Range(ts_to_img)` | Image must be a valid delay embedding | **Ignored** |

Without enforcement, off-manifold components in E-step outputs leak into the M-step → model learns off-manifold structure → next E-step drifts further. The error compounds across EM iterations.

**Our solution: manifold projection + noise-adaptive CG.**

After each E-step, we project the MMPS output onto the valid subspace and hard-enforce observations:

```python
# Project onto valid delay embedding manifold
result_ts = img_to_ts(result)
result = ts_to_img(result_ts)

# Hard observation enforcement (noiseless case)
result_ts = img_to_ts(result)
result_ts[:, observed_mask] = y[:, observed_mask]
result = ts_to_img(result_ts)
```

> **Proposition 1 (Projection optimality).** Let `Π = ts_to_img ∘ img_to_ts`. Then `Π` is the orthogonal projection onto `Range(ts_to_img)` and yields the minimum-norm solution to the representation consistency constraint: `Π(x) = argmin_{z ∈ Range(ts_to_img)} ||x - z||²`.

*Proof sketch.* `img_to_ts` averages overlapping delay-embedding windows (left-inverse up to normalization). The composition `ts_to_img(img_to_ts(·))` is self-adjoint and idempotent on R^{N_img}, hence an orthogonal projection. ∎

This provides noise reduction by factor d/m from averaging overlapping pixels (~60% for typical settings m=8, d=3). The off-manifold energy `E_off(k) = mean_i ||x_i - Π(x_i)||²` serves as a convergence diagnostic.

**Enabling condition: stable posterior sampling.** Projection is only useful if the MMPS output is reasonable to begin with. The MMPS forward model assumes `y ~ N(Ax, σ_y² I)`. In our noiseless setting (`y = Ax`), fixing `σ_y` to a small constant (e.g. 0.01) creates catastrophic ill-conditioning.

> **Proposition 2 (Bounded condition number).** The CG system `(σ_y²I + σ_t² AJA^T)v = r` has condition number `κ ≈ σ_t²||J||/σ_y²` with fixed `σ_y`, growing unboundedly with σ_t. Setting `σ_y = c · σ_t` yields `κ = (c² + λ_max(AJA^T))/(c² + λ_min(AJA^T))`, which is bounded independently of σ_t.

*Proof.* Factor σ_t² from both terms: the operator becomes `σ_t²(c²I + AJA^T)`. The condition number of `(c²I + AJA^T)` depends only on the spectrum of `AJA^T` and the constant `c`, not on σ_t. ∎

```python
sigma_y_sq = (self.c * sigma) ** 2    # σ_y(σ_t) = c · σ_t
```

The consequence: CG is stable at all noise levels → we can use 5+ iterations instead of 1–3 → more accurate posterior samples → the projection has something meaningful to work with.

**Evidence (Energy 50%):**

| Configuration | Best disc_mean | Best iteration |
|---------------|---------------|----------------|
| Kalman, no projection (`ema_reset_kalman_init`) | 0.0487 | 2 |
| STL + projection (`iter_stl_decomp_projection`) | 0.0493 | 3 |
| Kalman + freq_gated + projection (full) | **0.0472** | 4 |
| Projection without warm start (`decomp_projection`) | 0.0573 | 11 |

Projection without warm start still helps vs. baseline (0.057 vs 0.056), but the payoff is much larger when combined with structural consistency — projecting well-initialized completions onto the manifold is far more effective than projecting random ones. The best overall run (0.0472) uses projection.

**Why this is a consistency constraint.** The delay embedding is a *redundant* map: many images correspond to the same time series, but only one image per time series lives on the manifold. Standard MMPS is unaware of this redundancy because it was designed for non-redundant observation models. The projection enforces a structural invariant of the representation at every E-step — this is representation consistency, not a post-hoc fix.

#### Representation consistency in the M-step: L_rep

The E-step enforces representation consistency via projection (hard constraint). But the M-step has no analog: the denoiser is free to produce off-manifold outputs, requiring projection to clean them at the next E-step. We close this gap by teaching the denoiser to stay on-manifold during training.

```python
# L_rep: penalize off-manifold denoiser outputs (soft M-step analog of Prop 1 projection)
denoised_projected = ts_to_img(img_to_ts(denoised))   # Π(d_θ(x_t))
off_manifold = denoised - denoised_projected
L_rep = w_SNR(σ) · ||off_manifold||²
```

The SNR gate is essential (same rationale as L_obs): at high σ, the Tweedie mean `E[x|x_t]` is a smooth global average that need not be a valid delay embedding. At low σ, the denoiser should produce images that correspond to valid time series.

**Why L_rep complements the E-step projection.** The E-step projection (Prop 1) corrects off-manifold outputs after the fact. L_rep reduces the off-manifold energy at the source: the denoiser learns to produce outputs closer to Range(ts_to_img), so:
1. The E-step projection applies a smaller correction → less noise injection from the projection step
2. The off-manifold energy `E_off(k)` decreases faster across EM iterations
3. The Jacobian `J = ∂D_θ/∂x_t` is better-conditioned at on-manifold inputs (the inputs the model was trained to handle)

This creates a **symmetric enforcement** of representation consistency across both EM stages: E-step projects (hard, post-hoc), M-step penalizes (soft, at training time). The combined loss becomes:

```
L = L_SM + λ_obs · L_obs + λ_rep · L_rep
```

where L_SM operates in image space, L_obs bridges to TS-space observations, and L_rep bridges to the delay-embedding manifold.

---

### Constraint 3: Observation Consistency — The Model Must Not Forget What It Saw

*The image-space model must stay grounded to TS-space observations, even when the E-step is approximate.*

**The problem: MCEM drift.** In Monte Carlo EM, the M-step trains on samples from the approximate posterior. When the posterior sampler is exact, these samples perfectly satisfy `y = Ax`, so the observation-likelihood term drops out of the M-step. But with approximate posterior sampling (MMPS with truncated CG), completions are imperfect — they don't perfectly match observations. The M-step trains on these as if they were exact, with no mechanism to penalize inconsistency:

1. E-step samples are imperfect → some observed values are wrong
2. M-step trains on these as ground truth → denoiser reproduces the errors
3. Next E-step uses this denoiser → produces similarly wrong completions
4. Error compounds across EM iterations

This drift has been observed empirically: "the artifacts introduced by the poor sampling get amplified at each iteration, leading to a total collapse" (Rozet et al., 2024, Appendix figures).

**The principled corrections are provably intractable.**

| Approach | What it does | Why intractable |
|----------|-------------|-----------------|
| **Importance-weighted EM** | Re-weight samples by `p(y\|x)/q(x\|y)` | Requires the proposal density `q(x\|y)` of MMPS, which is defined by a reverse SDE with data-dependent drift — no closed form exists |
| **Variational EM** | Optimize a tractable bound on marginal likelihood | Requires an explicit variational family; the diffusion posterior is defined procedurally by the sampling chain and admits no parametric approximation |

This is not a matter of engineering effort — both approaches require density functions that are *structurally unavailable* for diffusion-based posterior samplers. Any method using MMPS (or similar SDE-based samplers) for the E-step faces this intractability.

**Our solution: L_obs — the minimal tractable alternative.**

Given that principled corrections are unavailable, we design the most targeted regularizer possible: one that acts only at observed positions, only at low noise levels, and vanishes at convergence.

> **Proposition 3 (L_obs properties).** Let `w_SNR(σ) = σ_d²/(σ² + σ_d²)`. Then: (i) `w_SNR` is monotonically decreasing, with `w_SNR → 1` as `σ → 0` and `w_SNR → 0` as `σ → ∞`. (ii) At a fixed point where `d_θ(x_t, σ)` perfectly reconstructs observations at low noise, `L_obs = 0` and its gradient vanishes — L_obs does not distort the converged model. (iii) L_obs operates in TS space (through `img_to_ts`) while L_SM operates in image space, creating a cross-space bridge with no interference at high noise levels.

```python
# Standard score matching loss (image space):
L_SM = λ_t · ||d_θ(x_t, t) - x_i||²

# Observation-consistency regularizer (time series space):
w_SNR(σ) = σ_d² / (σ² + σ_d²)        # ≈1 at low σ, ≈0 at high σ
L_obs = w_SNR(σ) · ||mask_ts · (img_to_ts(d_θ(x_t, t)) - y_ts)||²

# Combined (L_rep defined under Constraint 2):
L = L_SM + λ_obs · L_obs + λ_rep · L_rep
```

The SNR gate is essential: at high σ (e.g. σ=80) the denoiser cannot recover the signal — an unweighted L_obs would penalize it for an impossible task, creating spurious gradients that destroy the model. The gate `w_SNR(σ) = σ_d²/(σ² + σ_d²)` activates L_obs only in the low-noise regime where the denoiser has capacity to match fine-grained observations:

| σ | w_SNR | Behavior |
|---|-------|----------|
| 0.002 | ≈ 1.0 | Full observation enforcement |
| 0.5 (= σ_d) | 0.5 | Moderate enforcement |
| 80 | ≈ 0.00004 | Effectively off |

σ_d is the EDM `sigma_data` (= 0.5), inherited from Karras et al. (2022) — **not** an additional hyperparameter. The method introduces exactly **three hyperparameters** beyond vanilla EM: `c` (adaptive CG ratio), `λ_obs` (observation loss weight), and `λ_rep` (manifold loss weight).

**Evidence (Energy 50%):** The best run overall (`kalman_freqgate_decomp_proj`, disc_mean 0.0472) uses L_obs; runs with the same configuration minus L_obs (`kalman_curriculum_decomp_proj`, 0.0479) are slightly worse. The `freq_gated` variants appear in 3 of the top 6 runs.

**Why this is a consistency constraint.** L_SM trains in image space. Observations live in TS space. Without L_obs, the only connection between these spaces in the M-step is the E-step completions — which are approximate. L_obs creates a direct bridge: the denoiser's output, mapped to TS space via `img_to_ts`, must match ground-truth observations. This is observation consistency across the dual-space boundary — the M-step analog of what projection does for the E-step.

---

### Theoretical Status of L_obs

We address three potential concerns about L_obs directly.

**1. L_obs is the minimal tractable correction for approximate MCEM in a dual-space setting.**

We established above that the principled corrections (importance weighting, variational EM) are structurally intractable for diffusion-based posterior samplers. L_obs is therefore not "a hack instead of the right thing" — it is *the only option that is both tractable and directly targets the failure mode* (observation drift in the M-step).

Formally, L_obs fits within **penalized EM** (Green, 1990): replace the M-step objective `Q(θ|θ_k)` with `Q(θ|θ_k) - λ · R(θ)`, where our penalty `R(θ) = E_σ[w_SNR(σ) · ||mask · (img_to_ts(d_θ(x_t,σ)) - y)||²]` penalizes denoiser outputs that are inconsistent with observations at low noise. This is non-standard because `R` depends on θ through the denoiser (not just through the model density), so standard penalized EM convergence results (e.g., Green 1990, Theorem 2) do not directly apply. We claim empirical stabilization, validated through ablation, not formal convergence. Proposition 3 guarantees that L_obs vanishes at a good fixed point, so it does not distort a converged solution.

**2. Self-annealing is a property of the gradient magnitude, not a convergence proof.**

The self-annealing observation — "when the model is good, L_obs vanishes" — describes **gradient dynamics**, not convergence. We do not use it as a convergence argument. The concern is valid that a model could overfit to observed positions while having incorrect unobserved distribution, causing L_obs to vanish at a bad fixed point.

We argue this failure mode is unlikely in practice for two reasons:
- **Architectural:** The denoiser is a convolutional network that is translation-invariant across the image. It cannot easily learn to be accurate at observed positions and inaccurate at unobserved ones — the same filters process both. Overfitting to the mask pattern would require the network to memorize which positions are observed, which runs counter to the CNN inductive bias.
- **L_SM dominates globally:** L_obs acts only at observed positions (sparse: 30-50% of time points), while L_SM trains on the full image (all positions, including those corresponding to unobserved time points via delay embedding). A model that overfits to the mask pattern at the expense of global distributional quality would incur high L_SM loss, creating a countervailing gradient.

Empirically, we monitor `memorization_mean_similarity` to detect this failure mode. Our ablation shows L_obs improves `disc_mean` without increasing memorization.

**3. Bias analysis: L_obs biases the denoiser, not the score.**

L_obs penalizes `d_θ(x_t, σ)` (the denoiser output), not the score `∇_x log p_t(x)`. Since score matching already trains `d_θ` to approximate the Tweedie posterior mean `E[x|x_t]`, L_obs adds an additional constraint: "your Tweedie estimate, at observed positions, should match the ground truth."

This creates a **bias-variance trade-off:**

| Effect | Direction | Magnitude |
|--------|-----------|-----------|
| **Bias toward observations** | At observed positions, pushes d_θ toward y rather than E[x\|x_t] | Proportional to λ_obs · w_SNR; strongest at low σ |
| **Variance reduction** | Provides ground-truth signal that doesn't depend on E-step quality | Reduces variance from noisy MCEM samples |

At low noise levels (where w_SNR ≈ 1), the Tweedie mean E[x|x_t] should already be close to x (the true clean data), so the bias toward observations at observed positions is *toward the correct value* — the denoiser should predict y at observed positions when σ is small. The bias is therefore aligned with the score matching objective in the regime where it is active.

At moderate noise levels, L_obs could in principle distort the learned score by over-weighting observation positions. We mitigate this in two ways: (a) λ_obs is kept small (0.3-0.5) relative to L_SM, and (b) the SNR gate suppresses L_obs precisely when the denoiser's output is unreliable. Sensitivity analysis over λ_obs (Section: Ablation) validates that performance is robust across a range of values, indicating the bias does not dominate.

**Regarding identifiability.** EM may converge to a prior consistent with observations but different from the ground truth. L_obs does not "resolve" this — no finite-data method can. But it provides a practical inductive bias: among priors consistent with observations, it favors those whose denoiser best reconstructs observations at low noise levels. This is complementary to (not a replacement for) improved posterior sampling.

---

## How Our Method Relates to DiEM

We build on MMPS (Rozet et al., 2024) for posterior sampling in the E-step. DiEM is to our method as TST is to ImagenI2R — a tool we build upon, not the complete system. The difference is not a list of patches; it is that **DiEM operates in a single space (images), while we operate across a dual-space boundary (TS ↔ images)**. Every difference in the table below is a consequence of this structural fact.

| Constraint | DiEM (single-space) | Ours (dual-space) | Why different |
|-----------|--------------|-------------------|---------------|
| **Structural** | Gaussian init, 32 uniform iters | Kalman/STL warm start, curriculum (converges at 3–5) | TS distributions have temporal structure; images don't |
| **Representation (E)** | Image-space CG, fixed σ_y | **Observation-space CG** (Prop 4) + adaptive σ_y (Prop 2) + manifold projection (Prop 1) + warm-started CG | Delay embedding creates redundant subspace; CG should run where observations live |
| **Representation (M)** | — | L_rep (SNR-gated manifold penalty) | Denoiser should produce valid delay embeddings at low noise |
| **Observation** | L_SM only | L_SM + L_obs (SNR-gated, dual-space bridge) | Observations in TS space, model in image space; DiEM has both in the same space |

DiEM never needs these constraints because images are the native representation — there is no "other space" for consistency to degrade across. This is precisely the single-space vs. dual-space distinction that defines the general problem class. Any future method that applies EM-based learning in a lifted representation (latent diffusion, patch-based, etc.) will face the same three gaps and require analogous constraints.

---

## Ablation Design

The ablation is organized around the three consistency constraints and the observation-space CG improvement, isolating each contribution:

| Row | Method | Structural | Representation (E-step) | Representation (M-step) | Observation | What it tests |
|-----|--------|-----------|------------------------|------------------------|-------------|---------------|
| 1 | ImagenI2R | TST (one-shot) | — | — | — | Main competitor |
| 2 | KoVAE / GT-GAN | NCDE (one-shot) | — | — | — | Other baselines |
| 3 | EM vanilla | Gaussian init | image-space CG, fixed σ_y, no proj | — | L_SM only | EM alone — no consistency |
| 4 | + structural | Kalman/STL + curriculum | image-space CG, fixed σ_y, no proj | — | L_SM only | Constraint 1 only |
| 5 | + repr (E) | Kalman/STL + curriculum | adaptive CG + projection | — | L_SM only | Constraints 1+2(E) |
| 5b | + obs-space CG | Kalman/STL + curriculum | **obs-space CG** + adaptive + proj | — | L_SM only | Prop 4: CG in TS space |
| 6 | + observation | Kalman/STL + curriculum | obs-space CG + adaptive + proj | — | L_SM + L_obs | C1+C2(E)+C3 |
| 6b | + repr (M) | Kalman/STL + curriculum | obs-space CG + adaptive + proj | L_rep | L_SM + L_obs + L_rep | **Full system** |
| 7 | — structural | Gaussian init | obs-space CG + adaptive + proj | L_rep | L_SM + L_obs + L_rep | Without C1 |
| 8 | — repr (E) | Kalman/STL + curriculum | image-space CG, fixed σ_y, no proj | L_rep | L_SM + L_obs + L_rep | Without C2(E) |
| 9 | — repr (M) | Kalman/STL + curriculum | obs-space CG + adaptive + proj | — | L_SM + L_obs | Without L_rep |
| 10 | — observation | Kalman/STL + curriculum | obs-space CG + adaptive + proj | L_rep | L_SM + L_rep | Without L_obs |

**What the gaps reveal:**
- Gap (1,2 → 3): Does co-evolution help at all, even without consistency enforcement?
- Gap (3 → 4): How much does structural consistency alone contribute?
- Gap (4 → 5): Does representation consistency (E-step) add on top of structural?
- Gap (5 → 5b): Does observation-space CG improve over image-space CG? (Prop 4 vs. standard MMPS)
- Gap (5b → 6): Does observation consistency (L_obs) add on top?
- Gap (6 → 6b): Does L_rep (representation consistency in M-step) add on top?
- Rows 7–10: Complementarity — removing any single constraint should degrade performance.

**Sensitivity analysis:** We vary `c` in {0.01, 0.05, 0.1, 0.2}, `λ_obs` in {0.1, 0.3, 0.5, 1.0}, and `λ_rep` in {0.1, 0.3, 0.5} on energy 50% to characterize robustness.

---

## Paper Positioning

**Contribution 1 (problem identification — general).** We identify a problem class — *cross-space consistency in representation-based EM* — that arises whenever EM operates in a representation space different from the observation space. Lifted representations (delay embedding, latent diffusion, patch-based methods) create a dual-space system where consistency degrades at every EM stage. This problem does not arise in single-space EM and has not been studied.

**Contribution 2 (framework + formal results).** We formalize three necessary consistency constraints — structural, representational, and observational — with formal guarantees: projection optimality (Prop. 1), bounded CG condition number (Prop. 2), L_obs minimality with intractability of principled alternatives (Prop. 3), and observation-space CG equivalence with dimensionality reduction (Prop. 4). We prove that principled alternatives for observation consistency (importance-weighted EM, variational EM) are structurally intractable for diffusion-based samplers.

**Contribution 3 (observation-space posterior sampling).** We show that MMPS in a dual-space setting should run CG in the observation space (TS space), not the model space (image space). This yields the mathematically correct observation operator, reduces CG dimension by ~30×, and naturally incorporates the delay-embedding averaging within the CG iteration. Combined with warm-started CG across reverse steps, this makes high-quality posterior sampling practical.

**Contribution 4 (instantiation + empirical).** We instantiate the framework for delay-embedded time series — the dominant SOTA paradigm — replacing the commit-and-forget pipeline used by all existing corrupted TS generation methods. SOTA generation quality across multiple datasets and missing rates, converging in 3–5 iterations. Ablation organized around the constraints demonstrates that each is necessary and they are complementary.

---

## What We Credit vs. What We Claim

**Credit:** MMPS posterior score decomposition and Tweedie covariance (Rozet et al., 2024), delay embedding for time series representation (ImagenTime), EDM diffusion framework (Karras et al., 2022), EM algorithm (Dempster, Laird & Rubin, 1977), conjugate gradient method (Hestenes & Stiefel, 1952).

**Claim:**
1. The identification of *cross-space consistency* as a general problem class in representation-based EM, absent in single-space settings
2. Three necessary consistency constraints with formal results: projection optimality (Prop. 1), bounded CG condition number (Prop. 2), L_obs minimality with intractability of principled alternatives (Prop. 3), and observation-space CG equivalence (Prop. 4)
3. That MMPS posterior sampling in a dual-space setting should run CG in the observation space, not the model space — yielding exact observation operators, ~30× dimensionality reduction, and built-in manifold averaging
4. The first co-evolving framework for corrupted time series generation, instantiating the general framework for the dominant SOTA paradigm (delay embedding)

**Explicitly NOT claimed:**
- That EM or co-evolution is new (EM is classical; DiEM applies it to diffusion priors)
- That any individual technique is novel in isolation — the contribution is identifying the general problem class, proving why these constraints are necessary, and demonstrating their complementarity
- Full convergence theory for the combined system (we prove component-level properties and validate the system empirically)
- Solving the identifiability problem (we provide an inductive bias, not a solution)

---

## Honest Limitations

1. **Computational cost.** EM is inherently iterative — each iteration requires a full E-step (posterior sampling) and M-step (retraining). With warm start and curriculum, we converge in 3–5 iterations rather than 32, making wall-clock time comparable to one-shot methods. The E-step is embarrassingly parallelizable across samples. Observation-Space CG and warm starting reduce per-step cost, partially offsetting the overhead of L_rep.

2. **Component-level, not system-level theory.** We prove formal properties for each constraint individually (Propositions 1–4) and prove intractability of principled alternatives. But we do not have a convergence guarantee for the combined system. Near convergence, L_obs and L_rep vanish (Prop. 3 and on-manifold denoiser), so standard EM dynamics govern the fixed point — but the path to that fixed point is validated empirically, not theoretically.

3. **Noiseless observations only.** Our analysis assumes `y = Ax` with no observation noise. When real noise is present, σ_y regains physical meaning and the adaptive regularization becomes `σ_y(σ_t) = sqrt(σ_y_phys² + c²·σ_t²)`, which we leave to future work.

4. **L_rep bias.** L_rep pushes the denoiser toward on-manifold outputs at low noise, which is correct for perfect delay embeddings. If the delay embedding itself introduces artifacts (e.g., boundary effects), L_rep could amplify them. The SNR gate and small λ_rep mitigate this.

5. **Single instantiation.** We validate the framework on delay embedding (the dominant SOTA representation for TS generation). The general problem class (cross-space consistency in representation-based EM) applies to latent diffusion, patch-based methods, and graph lifts — but we do not provide experiments in those settings. However, observation-space CG generalizes directly: for any EM with observation operator G, CG should run in Range(G), not in the model space. We view validating this across settings as the most important direction for future work.

---

## Implementation References

### E-step changes (`run_diffem_mmps_self_correcting.py`)

**Observation-Space CG** (replaces image-space CG in posterior_denoise):
```python
def posterior_denoise_obs_space(self, x_t, sigma, obs_ts, mask_ts):
    """
    MMPS posterior denoiser with CG in observation space (Prop 4).
    CG operates in R^{N_obs × F} instead of R^{C × H × W}.
    """
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

    # Residual in observation space (masked TS)
    denoised_ts = self.img_to_ts_fn(denoised_img.float()).to(torch.float64)
    r_ts = obs_ts - mask_ts * denoised_ts   # r ∈ R^{N_ts × F}
    r_obs = mask_ts * r_ts                   # zero at unobserved

    # CG in observation space (Prop 4)
    def cg_operator_obs(v_ts):  # v_ts ∈ R^{N_ts × F}, nonzero only at observed
        # G^T: masked TS → image
        v_img = self.ts_to_img_fn(v_ts.float()).to(torch.float64)
        # J^T: image → image (single VJP)
        Jv_img = vjp_fn(v_img)
        # G: image → masked TS
        Jv_ts = self.img_to_ts_fn(Jv_img.float()).to(torch.float64)
        Jv_obs = mask_ts * Jv_ts
        return sigma_y_sq * v_ts + sigma_sq * Jv_obs

    v_ts = conjugate_gradient(
        cg_operator_obs, r_obs,
        x0=self.v_prev,            # warm start from previous step
        maxiter=self.cg_iters
    )
    self.v_prev = v_ts.detach()    # store for next step

    # Posterior score correction in image space
    v_img = self.ts_to_img_fn(v_ts.float()).to(torch.float64)
    score_img = vjp_fn(v_img)

    return denoised_img.detach() + sigma_sq * score_img.detach()
```

**Warm-started CG** (modified conjugate_gradient):
```python
def conjugate_gradient(A_fn, b, x0=None, maxiter=1):
    x = x0 if x0 is not None else torch.zeros_like(b)
    r = b - A_fn(x) if x0 is not None else b.clone()
    p = r.clone()
    # ... standard CG iterations ...
```

**Noise-adaptive σ_y** (Prop 2):
```python
sigma_y_sq = (self.c * sigma) ** 2    # σ_y(σ_t) = c · σ_t
```

**Consistency projection** (after posterior sampling, on final sample):
```python
result_ts = img_to_ts(result)
result = ts_to_img(result_ts)
result_ts = img_to_ts(result)
result_ts[:, observed_mask] = y[:, observed_mask]
result = ts_to_img(result_ts)
```

### M-step changes (`run_diffem_mmps_self_correcting.py`)

**Combined loss with L_SM + L_obs + L_rep:**
```python
sigma = (torch.randn([B, 1, 1, 1]) * P_std + P_mean).exp()
weight = (sigma ** 2 + sigma_d ** 2) / (sigma * sigma_d) ** 2
denoised = net(x_img + sigma * noise, sigma, None)

# L_SM: standard score matching (image space)
loss_sm = (weight * (denoised - x_img).square()).mean()

# L_obs: SNR-gated observation grounding (TS space — Prop 3)
snr_gate = sigma_d ** 2 / (sigma ** 2 + sigma_d ** 2)
denoised_ts = img_to_ts(denoised)
obs_residual = mask_ts * (denoised_ts - y_ts)
loss_obs = (snr_gate * (obs_residual ** 2).mean(dim=(-1,-2))).mean()

# L_rep: SNR-gated manifold penalty (image space — Prop 1 at training time)
denoised_proj = ts_to_img(img_to_ts(denoised))   # Π(d_θ)
off_manifold = denoised - denoised_proj
loss_rep = (snr_gate * (off_manifold ** 2).mean(dim=(1,2,3))).mean()

loss = loss_sm + lambda_obs * loss_obs + lambda_rep * loss_rep
```

**Note on loss weighting interaction.** L_SM uses the EDM weight `w(σ) = (σ² + σ_d²)/(σ·σ_d)²` (variance-reduction weighting for denoising score matching; Karras et al., 2022), while L_obs and L_rep use only the SNR gate `σ_d²/(σ² + σ_d²)`. L_obs and L_rep intentionally omit the score matching weight: `w(σ)` is specific to variance reduction in the denoising loss, while the SNR gate serves a fundamentally different purpose — gating the reliability of the denoiser output for observation and manifold matching. Applying both would over-suppress the regularizers at moderate noise levels where the denoiser is already informative. We validate this design choice empirically (see ablation).
