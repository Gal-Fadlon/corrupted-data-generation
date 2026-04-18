# Paper Story: Spectral MMPS — Learning Diffusion Priors for Irregular Time Series via Frequency-Aware Posterior Sampling

## The Problem

Generating realistic, regularly-sampled time series from irregularly-observed data
is a fundamental challenge in healthcare, finance, and science. Real-world sensors
fail, patients miss appointments, markets close — the data we collect is almost
always incomplete.  The goal is to learn the true underlying distribution
p(x) from corrupted observations y = Ax + noise (where A is a binary mask
indicating which timesteps were observed) and sample complete, regular sequences.

---

## Landscape of Prior Methods and Their Limitations

### Generation 1: GT-GAN and KoVAE (GAN/VAE + NCDE)

The earliest methods — GT-GAN and KoVAE — use a two-stage pipeline:

1. **Complete** irregular sequences using Neural Controlled Differential Equations (NCDEs)
2. **Generate** from the completed data using a GAN or VAE

**Three fundamental limitations:**

| Limitation | Consequence |
|---|---|
| **Outdated generative backbones** (GANs, VAEs) | Surpassed by diffusion models on every generative benchmark; mode collapse, training instability |
| **Computationally expensive NCDE preprocessing** | 6.5x slower than diffusion-based approaches; impractical for long sequences |
| **Blind trust in completion** | The generator is trained on NCDE-completed data *as if it were real*. When completion errors occur — and they inevitably do at high missing rates — these errors are treated as ground truth. There is no mechanism to distinguish real observations from imputed artifacts |

The result: state-of-the-art irregular benchmarks remain on average **540% worse**
than regular benchmarks. These methods cannot bridge the gap.

### Generation 2: ImagenI2R (Diffusion + TST Completion + Masking)

Our published NeurIPS paper (ImagenI2R) addressed all three limitations of
Generation 1:

- **Diffusion backbone** via ImagenTime — maps time series to images, leverages
  powerful vision-based U-Net architectures
- **Efficient TST completion** replaces NCDE — no differential equations, 85%
  faster training
- **Masking during loss computation** — does not blindly trust the TST completion;
  only observed pixels contribute to the denoising loss

The key insight was **"completion + masking"**: the TST creates *natural*
neighborhoods in image space (avoiding the "unnatural zero-padding" problem where
conv kernels attend to meaningless zeros), while masking prevents over-reliance
on imputed values. This achieved 70% improvement in discriminative score and 85%
reduction in compute.

**But ImagenI2R has its own structural limitations:**

| Limitation | Where it manifests | Why it matters |
|---|---|---|
| **One-directional supervision** | The TST encoder-decoder continues to train alongside the diffusion model, but only on a reconstruction loss over *observed* values (MSE on non-NaN positions). The diffusion model's growing understanding of the full data distribution never flows back to improve how the TST imputes *missing* values | The TST gets better at reconstructing what it can already see, not at guessing what it cannot. Imputation errors at missing positions — the hard part — are never corrected |
| **No feedback loop** | The diffusion model consumes TST completions but has no mechanism to signal "this imputation looks wrong given what I've learned about the distribution" | The system's quality ceiling is bounded by the TST's ability to impute from observed values alone — a weak bottleneck for a powerful diffusion backbone |
| **Masking is loss-level only** | Conv kernels still *see* imputed pixels in their receptive field | If TST completions are systematically biased (e.g., smoothing out high-frequency details), the kernels learn from plausible-looking but incorrect neighborhoods |
| **Frequency-blind** | Neither TST nor diffusion distinguishes frequency bands | High-frequency components are hardest to reconstruct from sparse observations, yet receive identical treatment to easy low-frequency trends |
| **No self-assessment** | The system has no notion of confidence in its imputations | All completed values are treated equally — whether the TST is confident or guessing |

The fundamental issue: **ImagenI2R is an open-loop pipeline.** The TST imputes
based on observed-value reconstruction only; the diffusion model learns from
those imputations; but the diffusion model's distributional knowledge never
informs the TST's imputation of missing values. The two components cannot
co-evolve on the hard part of the problem — reconstructing what was never
observed.

---

## Theoretical Foundation: DiEM and MMPS

Rozet et al. (NeurIPS 2024) formalize the problem of training diffusion models
from corrupted observations as **Empirical Bayes via Expectation-Maximization**
(DiEM). Their framework provides two critical components that we build upon.

### Component 1: EM for Diffusion Priors

DiEM observes that learning a prior q_θ(x) from observations
y ~ N(Ax, Σ_y) is equivalent to minimizing KL(p(y) || q_θ(y)). The EM update
decomposes into:

- **E-step**: For each observation (y, A), sample x ~ q_{θ_k}(x | y, A) via
  posterior sampling using the current diffusion model as prior
- **M-step**: Train q_{θ_{k+1}}(x) on the posterior samples via standard
  denoising score matching

This is powerful because it produces a *proper* diffusion model at every
iteration — compatible with any downstream sampling scheme — unlike methods
that modify the denoising objective itself (AmbientDiffusion, GSURE).

### Component 2: Moment Matching Posterior Sampling (MMPS)

To sample from q_θ(x | y), DiEM proposes MMPS. The posterior score decomposes as:

    ∇ log p(x_t | y) = ∇ log p(x_t) + ∇ log p(y | x_t)

The likelihood score is estimated via a Gaussian approximation q(x | x_t) with
mean E[x | x_t] (from the denoiser) and covariance V[x | x_t] (from Tweedie's
formula via the denoiser Jacobian):

    ∇ log q(y | x_t) = J^T A^T (Σ_y + A V[x|x_t] A^T)^{-1} (y - A E[x|x_t])

where J = ∇_{x_t} d_θ(x_t, t). The key insight is that the matrix
(Σ_y + A V[x|x_t] A^T) is symmetric positive-definite, so it can be inverted
via the **conjugate gradient (CG) method** using only implicit matrix-vector
products (vector-Jacobian products from automatic differentiation). Truncating
CG to 1-3 iterations already yields major improvements over heuristic
covariance approximations.

### What DiEM Demonstrates

On corrupted CIFAR-10 (75% pixels missing), DiEM with Tweedie covariance
achieves FID 13.18, compared to AmbientDiffusion's FID 28.88 at only 60%
corruption. On accelerated MRI, it produces posterior samples that are
noticeably more detailed than prior methods.

### What DiEM Leaves Open

DiEM treats Σ_y as an **isotropic** matrix (σ_y² I) — the same observation
noise for every component of y. This is appropriate for images where pixels
are independent and identically corrupted. However, for time series:

| DiEM assumption | Time-series reality |
|---|---|
| **Isotropic Σ_y** | Low frequencies are well-constrained by sparse observations; high frequencies are severely under-determined. Treating them identically wastes the CG solver's capacity |
| **Single sample per observation** | With 1 sample, the E-step produces a point estimate. No uncertainty information flows to the M-step. DiEM notes that the MRI experiment uses 2 samples "as data augmentation" but does not exploit variance |
| **Fixed E-step procedure** | The same CG operator is used at every EM iteration regardless of how the model's reconstruction quality evolves |
| **Gaussian initialization** | Fitting a Gaussian prior by EM is fast but uninformed about temporal structure (trend, seasonality) |

These are not limitations of DiEM's theory — its framework explicitly permits
any PSD Σ_y and any number of posterior samples. They are *opportunities*
that the time-series domain makes available.

---

## Our Method: Spectral MMPS in an Adaptive EM Framework

We build on DiEM's theoretical foundation — EM for diffusion priors with MMPS
posterior sampling — and introduce **Spectral MMPS**, a frequency-aware variant
of MMPS that exploits the spectral structure of time series. We embed Spectral
MMPS inside an adaptive EM loop where the posterior sampling strategy evolves
across iterations based on self-assessed reconstruction quality.

### The Core Technical Contribution: Spectral MMPS

Our starting point is DiEM's CG operator for the MMPS linear system:

    CG operator(v) = Σ_y v + A V[x|x_t] A^T v        [DiEM, isotropic]

We replace the isotropic Σ_y with a **frequency-dependent** operator:

    CG operator(v) = Σ_y(f) v + σ_t² A J^T(A^T v)    [Ours, anisotropic]

where Σ_y(f) is diagonal in the DFT basis of the time series:

    Σ_y(f) = σ_y² · (1 + (f / f_cutoff)^n)

This is still a valid PSD matrix — DiEM's theory applies without modification.
But its effect is profound:

- **Low frequencies** (f << f_cutoff): Σ_y(f) ≈ σ_y² — the solver trusts
  observations tightly, just like standard MMPS
- **High frequencies** (f >> f_cutoff): Σ_y(f) >> σ_y² — the solver
  down-weights observational constraints and relies on the diffusion prior

This acts as an **automatic Wiener filter** inside the CG solve. The diffusion
prior handles what it knows (global structure), while observations anchor what
they can reliably constrain (low-frequency trends).

**Within a single diffusion chain**, f_cutoff scales with the noise level:

    f_cutoff(σ) = max(0.05, f_cutoff_base · (1 - σ/σ_max))

At high diffusion noise (early in the reverse chain), the cutoff is low — only
trends are trusted from observations. As noise decreases, the cutoff rises,
progressively trusting finer details. This mirrors the coarse-to-fine
generation dynamics of diffusion models.

### Two-Phase System Architecture

#### Phase 1: STL Warm-Start

Rather than DiEM's Gaussian initialization, we use classical STL decomposition
(trend + seasonal + residual) — a well-established statistical method that
explicitly captures temporal structure. This provides a much more informed
starting point for the diffusion model's initial training, without any neural
network overhead.

#### Phase 2: Adaptive EM with Spectral MMPS

Each EM iteration follows DiEM's structure (Algorithm 1) but with four
enhancements:

**Enhancement 1: Spectral MMPS E-step (improved posterior sampling)**

The E-step uses Spectral MMPS as described above. The anisotropic Σ_y(f)
replaces DiEM's isotropic σ_y²I in the CG operator. Everything else — Tweedie
covariance via VJP, CG truncation, reverse-process discretization — follows the
same MMPS logic. (We use a Heun 2nd-order ODE solver with the Karras EDM
schedule rather than DiEM's stochastic DDIM; both are valid discretizations of
the posterior reverse process.)

**Enhancement 2: Multi-sample E-step as data augmentation**

Instead of generating 1 sample per observation, we draw K posterior samples.
All K samples are concatenated into the training set (K×N total examples),
treating each posterior sample as an independent training example — a form
of data augmentation that preserves posterior diversity.

The inter-sample variance across the K samples serves as a diagnostic signal
(see Enhancement 4) but is not used to weight the loss (early experiments
showed confidence weighting caused mode collapse).

**Enhancement 3: Uniform M-step with EMA reset**

The M-step trains via standard denoising score matching with **uniform loss
weighting** on all K×N augmented samples. This avoids the mode collapse
observed with confidence-weighted losses while still benefiting from the
richer training signal provided by multiple posterior samples.

Before each M-step, the model's EMA weights are reset. This prevents stale
running averages from the previous iteration's distribution from biasing the
new training round.

**Enhancement 4: Adaptive spectral filter + curriculum + early stopping**

After each E-step, **spectral diagnosis** measures the inter-sample coefficient
of variation (CV) in the Fourier domain across the K posterior samples. One bin
per rfft frequency ensures no empty bins. Raw (un-normalised) CV is compared
against a threshold: high CV at a frequency means the K samples disagree there,
indicating unreliable reconstruction.

The f_cutoff_base for the next E-step adapts **bidirectionally**:
- If high-frequency bands are reliable (low CV) → cutoff increases
- If high-frequency bands become unreliable → cutoff **decreases**
- Blending is slow (factor 0.2) to prevent oscillation

A **progress-based curriculum** anneals extra revealed positions over the full
EM range: early iterations reveal ~30% of missing positions (filled from the
current best reconstruction); this linearly decays to zero by the final
iteration.

**Early stopping** monitors disc_mean across EM iterations. If no improvement
is observed for a configurable patience window (default 2 iterations), the
loop terminates and the best checkpoint is retained.

```
for each EM iteration:

    PLAN    → Adapt f_cutoff_base from last spectral diagnosis
            → Progress-based curriculum: anneal extra reveals

    ACT     → E-step: K × Spectral MMPS posterior samples
            → All K samples → augmented training set (K×N)

    OBSERVE → Spectral diagnosis: per-frequency inter-sample CV
            → E-step quality: observed-position residual MSE,
              spectral energy ratio (hi/lo)

    PLAN    → Update f_cutoff_base bidirectionally for next iteration

    ACT     → M-step: uniform denoising score matching on augmented data

    EVALUATE → disc_mean, early stopping check, log adaptation trajectory
```

#### Output

The EM loop retains the best model checkpoint (by disc_mean) and terminates
either when all iterations complete or when early stopping triggers. The best
EM model is the final output — no separate Phase 3 retraining is needed, as
the EM-trained model already constitutes a proper diffusion prior.

---

## The Relationship to DiEM — Precisely

Our system maps to DiEM's algorithms as follows:

| DiEM Component | Our Implementation |
|---|---|
| **Algorithm 1** (EM pipeline) | `agentic_em.py: agentic_em_main()` — same E/M alternation, plus adaptive planning between steps |
| **Algorithm 2** (DDIM posterior sampling) | `run_diffem_agentic_spectral_cg.py: mmps_sample()` — deterministic Heun 2nd-order ODE solver (Karras EDM schedule); DiEM uses stochastic DDIM (η=1), both are valid reverse-process discretizations |
| **Algorithm 3** (MMPS posterior score) | `posterior_denoise()` — identical structure: denoiser call, VJP, CG solve, score update. **Only change**: Σ_y → Σ_y(f) |
| **Algorithm 4** (CG method) | `conjugate_gradient()` — same iteration, truncated to 1 step by default |
| **Eq. 9** (EM M-step: minimize KL) | `m_step_augmented()` — denoising score matching with uniform loss on K×N augmented samples |
| **Eq. 12-13** (likelihood score via CG) | CG operator: `spectral_sigma_y(v) + σ_t² · mask · J^T(mask · v)` |
| **Initialization** (Gaussian EM) | STL decomposition (domain-specific warm-start) |

The code-level correspondence in the CG operator is exact:

```
DiEM:  cg_op(v) = σ_y² · v        + σ_t² · A · (J^T · A^T · v)
Ours:  cg_op(v) = Σ_y(f) · v      + σ_t² · mask · vjp(mask · v)
                  ↑                  ↑
                  anisotropic        identical to DiEM
                  (our contribution) (A = mask, J^T via autodiff)
```

Where `Σ_y(f) · v` is implemented as: img → ts → rfft → scale by
σ_y²(1 + (f/f_c)^n) → irfft → ts → img.

---

## How Each Limitation Is Addressed

| Previous Limitation | Our Solution | Theoretical Basis |
|---|---|---|
| **GT-GAN/KoVAE: outdated backbones** | Vision-based diffusion model (inherited from ImagenI2R) | — |
| **GT-GAN/KoVAE: expensive NCDE** | STL warm-start (zero training cost) + diffusion-based imputation | DiEM: EM for diffusion priors replaces external imputers |
| **GT-GAN/KoVAE: blind trust in completion** | Multi-sample augmented M-step + spectral self-diagnosis | K posterior samples preserve diversity; inter-sample CV detects unreliable bands |
| **ImagenI2R: one-directional supervision** | The diffusion model IS the imputer — its distributional knowledge drives imputation via MMPS | DiEM Eq. 9: posterior sampling with learned prior |
| **ImagenI2R: no feedback loop** | Iterative EM — imputations improve every iteration as the diffusion prior improves | DiEM: monotonically increasing log-evidence |
| **ImagenI2R: frequency-blind** | Anisotropic Σ_y(f) in Spectral MMPS + bidirectional adaptive f_cutoff | DiEM Eq. 12-13: Σ_y may be any PSD matrix |
| **ImagenI2R: no self-assessment** | K-sample inter-sample CV in Fourier domain → spectral diagnosis | Multi-sample MCEM |
| **ImagenI2R: fixed strategy** | Spectral diagnosis adapts f_cutoff bidirectionally; progress-based curriculum + early stopping | Data-driven adaptation across EM iterations |
| **DiEM: isotropic Σ_y** | Frequency-dependent Σ_y exploiting time-series spectral structure | Valid within DiEM's linear-Gaussian framework |
| **DiEM: single-sample E-step** | K samples as data augmentation → richer M-step training signal | Better approximation of E[·] in Eq. 9 |

---

## What Makes This Genuinely Novel

### Spectral MMPS: Domain-Aware Posterior Sampling

DiEM's MMPS uses isotropic observation noise because images have no preferred
basis for distinguishing "easy" from "hard" components. Time series do — the
frequency domain. Our Spectral MMPS is, to our knowledge, the first posterior
sampling scheme that adapts observation noise *per frequency band* within the
MMPS framework. The implementation requires only replacing one term in the CG
operator, but the effect is a built-in Wiener filter that makes the solver
frequency-aware without any additional networks or training.

### Adaptive EM, Not Static EM

Standard EM (including DiEM) follows the same E-step and M-step procedures
at every iteration. Our system is different: the spectral diagnosis after
each E-step feeds back into the f_cutoff_base for the next E-step. The
inter-sample Fourier CV drives bidirectional f_cutoff adaptation, while
a progress-based curriculum and early stopping prevent overfitting. The
system follows a different trajectory depending on what it discovers about
its own reconstruction quality.

This is grounded in the MCEM literature: the quality of posterior samples
is critical for EM convergence (Levine & Casella 2001, Caffo et al. 2005).
By monitoring reconstruction quality (spectral diagnosis) and imputation
uncertainty (multi-sample variance), we adapt the posterior sampling strategy
to improve sample quality where it matters most.

### Multi-Sample Augmentation + Spectral Self-Diagnosis

Running K samples and averaging is variance reduction. Our contribution goes
further: all K samples are used as **data augmentation** for the M-step (K×N
training examples), preserving posterior diversity rather than collapsing it
via averaging. The inter-sample variance in the Fourier domain provides a
per-frequency **spectral diagnosis** that drives the adaptive spectral filter
bidirectionally — allowing f_cutoff to both increase (as the model improves)
and decrease (when high-frequency bands become unreliable).

Additional self-assessment signals include observed-position residual MSE
(how well imputations match known values) and spectral energy ratio (detecting
high-frequency hallucination). Together with early stopping on disc_mean,
these give the system robust self-correction without confidence-weighted losses
(which caused mode collapse in earlier experiments).

---

## The Narrative Arc (for the paper)

1. **The problem is real and unsolved.** Generating regular time series from
   irregular observations matters for healthcare, finance, and science. Existing
   methods are either too slow (NCDE-based) or too rigid (static pipelines).

2. **One-directional pipelines hit a ceiling.** Whether you use NCDE (GT-GAN,
   KoVAE) or TST (ImagenI2R), the imputer only learns from observed-value
   reconstruction, never from the generator's distributional understanding. The
   generator can never surpass the imputer's ability to fill in missing values.

3. **DiEM shows the way: EM + MMPS.** Rozet et al. demonstrate that training
   diffusion priors from corrupted observations via EM — with MMPS for posterior
   sampling — produces proper diffusion models that dramatically outperform
   methods that modify the denoising objective. The theoretical machinery is
   sound: EM guarantees monotonically increasing log-evidence, MMPS leverages
   Tweedie's covariance for accurate posterior sampling, and CG makes it
   tractable in high dimensions.

4. **But the solver is domain-agnostic.** DiEM's MMPS uses isotropic Σ_y —
   the same observation noise for every component. This ignores a fundamental
   property of time series: low frequencies are well-constrained by sparse
   observations while high frequencies are severely under-determined. Treating
   them identically wastes the solver's capacity.

5. **Spectral MMPS exploits this structure.** By replacing isotropic Σ_y with
   frequency-dependent Σ_y(f) — a single-line change to the CG operator, fully
   permitted by DiEM's theory — we get a built-in Wiener filter that makes the
   posterior sampling frequency-aware.

6. **The adaptive loop makes EM self-correcting.** Multi-sample E-steps provide
   data augmentation and spectral self-diagnosis via inter-sample Fourier CV.
   This signal feeds back to bidirectionally adapt the spectral filter — turning
   static EM into a system that adjusts its imputation strategy based on
   measured performance, with early stopping to prevent EM overfitting.

7. **The result bridges the irregular-regular gap.** By combining DiEM's
   principled EM framework with domain-aware posterior sampling and adaptive
   self-correction, we approach the performance of models trained on
   fully-observed regular data, even from highly corrupted observations.

---

## Proposed Contributions (for the paper)

1. **Spectral MMPS**: We introduce a frequency-aware variant of moment matching
   posterior sampling (MMPS) that replaces isotropic observation noise with a
   frequency-dependent anisotropic operator Σ_y(f), acting as an automatic
   Wiener filter inside the CG solver. This exploits the spectral structure of
   time series within DiEM's theoretical framework without modification to its
   convergence guarantees.

2. **Adaptive EM with spectral self-diagnosis**: We propose an adaptive EM
   loop where inter-sample Fourier-domain CV across K posterior samples drives
   f_cutoff bidirectionally across iterations. Additional self-assessment
   (observed-position residual MSE, spectral energy ratio) and early stopping
   prevent EM overfitting.

3. **Multi-sample data augmentation**: We use all K posterior samples as
   independent training examples (K×N augmented dataset) with uniform loss,
   preserving posterior diversity better than averaging or confidence weighting.

4. **State-of-the-art irregular time series generation**: Our method achieves
   state-of-the-art results on irregular time series benchmarks, further
   narrowing the gap between irregular and regular generation performance.

---

## Clean Ablation Structure

The ablation cleanly separates DiEM-level components from our extensions:

| Ablation | What is removed | Tests the value of |
|---|---|---|
| Full system | Nothing | Upper bound |
| Isotropic Σ_y (standard MMPS) | Spectral anisotropy in CG operator | Spectral MMPS vs vanilla MMPS |
| K=1 + fixed f_cutoff + no curriculum | All adaptive/agentic extensions | The full adaptive layer on top of Spectral MMPS |
| K=1 only | Multi-sample augmentation → single sample M-step (but adaptive f_cutoff still active via spectral diagnosis with K=1 returning zeros) | Data augmentation benefit specifically |
| Fixed f_cutoff | Spectral diagnosis + bidirectional adaptive filter | Adaptive spectral planning |
| No curriculum | Progress-based reveal schedule | Curriculum support for early EM iterations |
| No early stopping | Patience-based termination | EM overfitting prevention |
| No EM (ImagenI2R) | The entire EM loop + all extensions | The closed-loop co-evolution vs open-loop pipeline |

The **isotropic Σ_y ablation** is particularly important: it isolates the
contribution of Spectral MMPS from the rest of the system, directly measuring
whether frequency-aware observation noise improves posterior sampling quality
within the same EM framework. This corresponds to replacing our CG operator
with DiEM's original one (σ_y² · v + σ_t² · mask · J^T(mask · v)).

Note on K=1: setting `--estep_n_samples 1` disables spectral diagnosis
(returns zeros), so f_cutoff stays at its initial value. To fully recover the
non-agentic Spectral MMPS baseline, also fix f_cutoff to a constant and
disable the curriculum.
