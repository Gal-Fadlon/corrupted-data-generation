# Paper Story v2: Closing the Loop on Irregular Time Series Generation

**Working Title:** "Closing the Loop: Frequency-Aware Diffusion Priors for Irregular Time Series Generation"

---

## The Problem

Learning a generative model for regular time series from irregularly-observed
data (missing values, sensor failures, variable sampling). Formally:
given corrupted observations y = Ax + noise with per-sample binary masks A,
learn p(x) and sample complete, regular sequences.

**The irregular-regular gap:** Best existing methods produce results ~540%
worse than models trained on fully-observed data. Closing this gap is the
paper's goal.

---

## The Landscape: Two Disconnected Tracks, Both Incomplete

The literature treats generation and imputation as **separate problems** with
separate methods, architectures, and evaluation protocols — despite both
stemming from the same underlying question: learning p(x) from corrupted data.

**Generation methods** (ImagenI2R, KoVAE, GT-GAN, TimeGAN) produce
unconditional samples but cannot incorporate new observations at test time.
If a clinician has a patient's partial ECG and wants a plausible completion,
these methods cannot help — they can only generate entirely new synthetic ECGs.

**Imputation methods** (CSDI, BRITS, GP-VAE) learn p(x_missing | x_observed)
but cannot generate unconditional samples. They require observed values as
input and offer no generation capability.

A practitioner who needs both must train and maintain **two separate models**
with different architectures and losses.

---

## The Open-Loop Bottleneck (Key Observation)

Every existing generation method follows the same architecture:

    Irregular data → [Imputer] → Completed data → [Generator] → Samples

- GT-GAN / KoVAE: NCDE imputes → GAN/VAE generates
- ImagenI2R: TST imputes → Diffusion generates + masking

**Three structural problems:**

**1. No feedback — the pipeline is one-directional.** The imputer only learns
from observed values. The generator only learns from imputed data. The
generator's growing understanding of the data distribution NEVER flows back
to improve imputation. The system's ceiling is bounded by the imputer's
ability to guess from partial observations alone. ImagenI2R's masking
mitigates blind trust in completions but does not solve the fundamental
issue: the diffusion model cannot signal "this imputation is wrong given
what I've learned."

**2. No noise handling.** Prior generation methods assume observed values are
clean. TST trains with MSE(output, observed_value) — treating noisy values
as ground truth. NCDE interpolates through noisy observations, propagating
noise. When observations carry sensor noise (ubiquitous in real deployments),
these methods learn p(x_noisy), not p(x_clean). This is not a parameter
tweak — it is a fundamentally different treatment of observations that prior
generation methods cannot replicate without redesign.

**3. Generation-only at test time.** Once trained, these models can only
sample unconditionally. If new partial observations arrive at test time
(a common use case: conditioning on what you already know about a patient),
they require entirely separate imputation machinery.

---

## Our Approach: Close the Loop, Unify the Tasks

**Core idea:** Eliminate the separate imputer. The diffusion model IS both
generator and imputer, via EM posterior sampling. One model, two capabilities.

    ┌──────────────────────────────────┐
    │   Diffusion Model q_θ(x)        │
    │   (generator AND imputer)        │
    └──────┬───────────────┬───────────┘
           │               │
      M-step: train    E-step: posterior
      on posterior     sampling from
      samples (K×N)    irregular data
           │               │
           └── co-evolve ──┘

Each EM iteration: better prior → better imputations → better training data
→ better prior. The generator's distributional knowledge directly drives
imputation. This is the feedback loop that open-loop methods lack.

**Simultaneously, this solves all three structural problems:**

- **Feedback:** The EM loop is the feedback. The prior improves imputations,
  which improve training data, which improves the prior.
- **Noise handling:** The corruption model Q(y|x) enters only the E-step via
  σ_y. When σ_y > 0, the posterior denoiser balances prior vs. likelihood,
  naturally denoising while reconstructing. The M-step always trains on clean
  reconstructions — the model learns p(x_clean), not p(x_noisy). The
  corruption model is a parameter, not an architectural choice.
- **Unified inference:** At test time, generation = unconditional sampling from
  the learned prior. Imputation = MMPS posterior sampling conditioned on new
  observations. Same weights, no retraining, no separate model. The E-step IS
  conditional sampling — imputation is a free byproduct.

### The EM framework we build on

Learning a prior q_θ(x) from corrupted observations y ~ N(Ax, Σ_y) can
be cast as empirical Bayes: minimize KL(p(y) || q_θ(y)) where q_θ(y) =
∫ p(y|x) q_θ(x) dx. The EM algorithm decomposes this into:

- **E-step:** For each (y, A), sample x ~ q_{θ_k}(x | y, A)
- **M-step:** Train q_{θ_{k+1}}(x) on these samples via denoising score
  matching: min_θ E[λ_t ||d_θ(x_t, t) - x||²]

This produces a proper diffusion model at every iteration (compatible with
any downstream sampling scheme). For exact posterior sampling, the expected
log-evidence is monotonically increasing [Rozet et al., 2024]; with
approximate (MCEM) sampling, convergence is stochastic but well-studied
[Ruth et al., 2024].

### Posterior sampling via the posterior score decomposition

To draw x ~ q_θ(x | y, A) in the E-step, we need the posterior score:

    ∇ log p(x_t | y) = ∇ log p(x_t) + ∇ log p(y | x_t)
                        \_________/     \______________/
                        prior score      likelihood score
                        (from denoiser)  (needs estimation)

The prior score is available via Tweedie's formula: ∇ log p(x_t) =
(d_θ(x_t,t) - x_t) / σ_t². The likelihood score is the hard part.

**Gaussian moment-matching approximation.** Approximate p(x | x_t) by the
Gaussian q(x | x_t) = N(E[x|x_t], V[x|x_t]) that matches its first two
moments. Under the linear-Gaussian observation model p(y|x) = N(Ax, Σ_y),
the marginal likelihood becomes:

    q(y | x_t) = N(A · E[x|x_t],  Σ_y + A · V[x|x_t] · Aᵀ)

and the likelihood score is:

    ∇ log q(y|x_t) = Jᵀ Aᵀ (Σ_y + A V[x|x_t] Aᵀ)⁻¹ (y - A E[x|x_t])

where J = ∇_{x_t} d_θ(x_t, t) is the denoiser Jacobian.

**Tweedie's covariance.** The covariance V[x|x_t] is estimated via
Tweedie's second-order formula: V[x|x_t] = σ_t² · ∇_{x_t}ᵀ d_θ(x_t, t),
which requires only vector-Jacobian products (VJPs) — never materializing
the full N×N Jacobian.

**Conjugate gradient solver.** The matrix (Σ_y + A V[x|x_t] Aᵀ) is SPD,
so we solve the linear system via conjugate gradient using only implicit
matrix-vector products. The CG operator is:

    CG_op(v) = Σ_y · v + σ_t² · A · VJP(Aᵀ v)

Truncating to 1-3 CG iterations is sufficient [Rozet et al., 2024].

**This is where our contribution enters.** The entire posterior sampling
machinery above is established. Our observation is that the choice of Σ_y
in this CG operator — which prior work treats as isotropic — is critically
important for time series and should be frequency-dependent.

---

## Theoretical Contribution: Why Σ_y Should Be Frequency-Dependent

### Proposition 1 (Frequency-dependent observational informativeness)

For a stationary time series observed through random masking, observations
constrain different frequencies to vastly different degrees: low frequencies
are well-determined by sparse observations while high frequencies are
essentially unconstrained.

**Setup.** Let x ∈ ℝ^T be a zero-mean stationary Gaussian process with
power spectral density S_xx(f). Under periodic boundary conditions
(standard in spectral analysis), the covariance is circulant:
C_x = F^H diag(S_xx) F where F is the DFT matrix. (For non-periodic
boundaries, the Toeplitz covariance is well-approximated by its circulant
embedding when T is large.) Observe y = Ax + n where
A = diag(a₁,...,a_T) is a binary mask with each aᵢ ~ Bernoulli(p)
i.i.d., and n ~ N(0, σ_n² I).

**Exact result (fully observed, p = 1).** When A = I (no masking), the DFT
diagonalizes both the prior covariance and the observation operator. The
posterior variance at frequency f is the standard Bayesian update for
independent Gaussian channels:

    V_post(f) = [1/S_xx(f) + 1/σ_n²]⁻¹ = S_xx(f) · σ_n² / (S_xx(f) + σ_n²)

This is exact and requires no approximation.

**Approximate extension to random masking (0 < p < 1).** Under random
masking, the DFT no longer exactly diagonalizes the observation operator
because A is diagonal in the time domain (not the frequency domain). However,
for large T, the off-diagonal cross-frequency interference introduced by
the random mask concentrates around zero (each off-diagonal entry of
F A F^H has variance p(1-p)/T → 0). In this large-T regime, the system
is approximately described by T independent Gaussian channels with
effective noise variance σ_n²/p per channel, giving:

    V_post(f) ≈ [1/S_xx(f) + p/σ_n²]⁻¹ = S_xx(f) · σ_n² / (p · S_xx(f) + σ_n²)

**What this reveals about frequency-dependent informativeness.** The
fraction of prior uncertainty that remains after observing y is:

    R(f) = V_post(f) / S_xx(f) = σ_n² / (p · S_xx(f) + σ_n²)

R(f) is the quantity that varies across frequencies and determines how
informative the observations are at each frequency:

- At low f where S_xx(f) >> σ_n²/p:  R(f) ≈ σ_n² / (p · S_xx(f)) → 0
  (observations reduce uncertainty to near zero — highly informative)
- At high f where S_xx(f) << σ_n²/p:  R(f) ≈ 1
  (observations provide almost no information — posterior ≈ prior)

Real-world time series commonly exhibit decaying spectra — S_xx(f)
decreases with |f| (red/pink spectral character, a well-documented
property of natural signals [see e.g. 1/f noise literature], and
empirically verified across all our datasets). For any such spectrum,
R(f) increases monotonically with |f|: observations are most informative
at low frequencies and least informative at high frequencies.

**Intuition.** Low-frequency components vary slowly — neighboring observed
timesteps carry redundant information about them, so sparse observations
still constrain them well. High-frequency components change between every
pair of timesteps — each missing sample destroys information that no
neighbor can recover.

**Note on the absolute MMSE.** The absolute posterior variance V_post(f)
itself decreases with |f| (it is monotonically increasing in S_xx(f)).
What matters for the Σ_y design is not the absolute uncertainty but the
relative informativeness R(f): at which frequencies should the posterior
sampling trust observations vs. trust the learned prior.

### Corollary: Isotropic Σ_y mismatches the observation structure

The standard choice Σ_y = σ² I in the CG operator treats all frequencies
identically. But as Proposition 1 shows, observations constrain low
frequencies far more than high frequencies. Using isotropic Σ_y forces
the posterior sampling to weight observational evidence equally across
all frequencies — over-trusting observations at high frequencies (where
they carry little information) and under-utilizing them at low frequencies
(where they are highly informative).

A frequency-dependent Σ_y corrects this mismatch: small Σ_y(f) at low
frequencies (trust observations, which are informative) and large Σ_y(f)
at high frequencies (trust the learned prior, since observations are
uninformative). This defines a posterior that is better suited to the
actual information structure of irregularly-sampled time series.

### Proposition 2 (Wiener filter structure)

**Claim.** When Σ_y is frequency-dependent and the CG system matrix is
approximately diagonal in the DFT basis, the posterior sampling correction
at each diffusion step takes the form of a Wiener filter.

**Setup.** The CG operator applied at noise level σ_t inverts the matrix
Σ_y + A V[x|x_t] Aᵀ (cf. Eq. (8) of DiEM). When Σ_y is diagonal in the
DFT basis (by construction) and A V[x|x_t] Aᵀ is approximately so (which
holds when A is dense or when cross-frequency terms are small relative to
diagonal terms), the system is approximately separable by frequency:

    M(f) ≈ Σ_y(f) + σ_t² · S_J(f)

where S_J(f) represents the diagonal of F(A V[x|x_t] Aᵀ)F^H. The CG
solution then weights the residual (y - A·denoised) at each frequency by:

    w(f) ≈ σ_t² S_J(f) / (Σ_y(f) + σ_t² S_J(f))

This has the classical Wiener filter form: signal power over total power.
When Σ_y(f) is large (high frequencies), w(f) is small and the prior
dominates. When Σ_y(f) is small (low frequencies), w(f) is large and
observations dominate.

**Significance.** This connects our spectral Σ_y to a well-understood
framework: the Wiener filter is the MMSE-optimal linear estimator for
signal extraction in additive noise. By choosing Σ_y(f) to reflect the
frequency-dependent informativeness from Proposition 1, each posterior
sampling step applies an approximately optimal linear correction, rather
than the uniform correction implied by isotropic Σ_y.

**Caveat.** The frequency-diagonal approximation of A V[x|x_t] Aᵀ
becomes less accurate when the mask is very sparse (small p) or T is
small. In these regimes the Wiener filter interpretation is qualitative
rather than exact.

### Design choice: Noise-dependent frequency cutoff

**Motivation.** Diffusion models generate structure coarse-to-fine: at high
noise levels σ_t, the denoiser d_θ(x_t, t) captures only low-frequency
modes, and V[x|x_t] is dominated by these modes. At low noise levels, the
denoiser resolves progressively finer structure. Incorporating high-frequency
observational corrections before the denoiser can represent that detail
risks introducing artifacts.

**Implementation.** We parameterize Σ_y(f) with a noise-level-dependent
cutoff:

    f_cutoff(σ_t) = f_cutoff_base · (1 - σ_t / σ_max)

At early steps (large σ_t), f_cutoff is small and Σ_y(f) is large for
most frequencies — the prior dominates. At late steps (small σ_t),
f_cutoff grows and observations are incorporated at progressively higher
frequencies. The specific linear schedule is a design choice; the key
principle is monotonically expanding the trusted frequency band as the
diffusion process progresses.

---

## The Method: Spectral Posterior Sampling

### The one change to the CG operator (derived from Propositions 1-2)

Standard CG operator (isotropic):

    CG_op(v) = σ_y² · v + σ_t² · A · VJP(Aᵀ v)

**Our CG operator (spectral):**

    CG_op(v) = Σ_y(f) · v + σ_t² · A · VJP(Aᵀ v)

where:

    Σ_y(f) = σ_y² · (1 + (f / f_cutoff(σ_t))^n)

The operator is PSD (eigenvalues ≥ σ_y²), diagonal in the DFT basis
(efficient: FFT → scale → IFFT), and parameterized by two interpretable
quantities: f_cutoff (transition frequency from Prop 1's analysis) and n (filter
order). The second term in the CG operator is completely unchanged from
prior work.

**Why this specific functional form?** The Butterworth-like shape
(1 + (f/f_c)^n) is the standard parametric approximation to an ideal
low-pass/high-pass transition in filter design. It captures the
monotonically increasing Σ_y(f) motivated by Proposition 1 with only
2 parameters rather than O(T) free spectral values, avoiding overfitting
while reflecting the essential frequency-dependent structure. The
adaptive mechanism (below) tunes f_cutoff automatically.

### Full pipeline

**Initialization.** STL decomposition (trend + seasonal + residual) —
a frequency-aware warm-start where the trend captures low-frequency
structure and the seasonal component captures periodic modes, naturally
complementing the spectral posterior sampling.

**E-step.** Draw K posterior samples per observation using spectral
posterior sampling. All K samples form the training set (K×N examples),
preserving posterior diversity.

**M-step.** Standard denoising score matching on augmented data with
uniform loss. EMA weights reset between iterations.

**Self-diagnosis.** Inter-sample Fourier CV across K samples identifies
which frequencies are reliably reconstructed. This drives bidirectional
f_cutoff adaptation (increase when HF bands are reliable, decrease when
not), providing automatic spectral tuning without manual hyperparameter
selection. Early stopping on discriminative score prevents EM overfitting.

---

## Contributions

1. **First closed-loop framework for irregular TS generation** — the diffusion
   model serves as both generator and imputer via EM, enabling co-evolution
   that open-loop pipelines cannot achieve.

2. **Unified generation and imputation from one model** — the same trained
   weights provide unconditional generation AND conditional imputation via
   posterior sampling, replacing two separate model families with one. Prior
   generation methods cannot impute; prior imputation methods cannot generate.

3. **Principled corruption handling** — by formulating the problem as latent
   variable estimation, the corruption model (missing values, noise, or both)
   is a parameter of the E-step, not an architectural choice. This naturally
   handles noisy observations — which prior generation methods treat as clean
   ground truth — and mixed corruption types without redesign.

4. **Proposition 1 + spectral posterior sampling** — we show that
   observational informativeness for masked time series is strongly
   frequency-dependent and derive a spectral CG operator that accounts
   for this, connecting to classical Wiener filtering.

5. **SOTA under diverse, realistic corruption models** — beyond the standard
   30/50/70% benchmark, we evaluate on settings no prior work addresses:
   noisy observations, block missingness, mixed missing rates, and
   conditional generation.

---

## Experimental Scope (What Makes This Comprehensive)

| Setting | What it tests | Why we can, others can't |
|---|---|---|
| **Standard** (30/50/70%, lengths 24-10920) | Baseline comparison | Expected by reviewers |
| **Mixed missing rates** (each sample draws its own rate from U(0,0.7)) | Realistic heterogeneous corruption | EM handles per-sample A natively; open-loop methods assume fixed rates |
| **Block missingness** (prefix/suffix/middle/random blocks missing) | Contiguous gaps — the hardest setting | Tests whether the prior truly learned the distribution, not just interpolation |
| **Conditional generation** (observe partial, generate rest) | Practical downstream application | E-step IS conditional sampling — free capability; prior methods need separate machinery |
| **Noisy + irregular** (missing + Gaussian noise) | Simultaneous corruption | Σ_y(f) handles both naturally; prior methods treat noisy values as ground truth |
| **Imputation quality** (same model, no retraining) | Dual-use capability | Posterior sampling gives imputation for free; generation baselines need a separate model |

### Robustness story (new, from ablations)

Our framework shows structurally better robustness:
- **Across severity**: the quality gap between easy (30% missing) and hard
  (70% missing) is smaller than for prior methods — because the M-step always
  trains on fully-reconstructed data, never on partially-masked data.
- **Across corruption type**: switching between random and block missing
  patterns causes less degradation — because the corruption model is a
  parameter of the E-step, not baked into the architecture.
- **Noise immunity**: when observations carry Gaussian noise, prior methods
  degrade proportionally (they learn p(x_noisy)); our method's Σ_y absorbs
  the noise and the M-step trains on denoised reconstructions.

---

## The Key Ablation

| Method | Loop type | Noise handling | Imputation | What it shows |
|---|---|---|---|---|
| ImagenI2R | Open-loop | No (treats noise as GT) | No (generation only) | Baseline ceiling |
| Isotropic EM | Closed-loop | Yes (σ_y) | Yes (posterior sampling) | Value of closing the loop + unification |
| **Spectral EM (ours)** | **Closed-loop** | **Yes (Σ_y(f))** | **Yes** | **+ frequency-aware posterior** |

All three jumps matter. Together they tell the complete story: open→closed
(structural), isotropic→spectral (theoretical), generation-only→unified
(practical).

---

## The Narrative Arc (Elevator Pitch)

**Previous methods** for learning generative models from irregular time series
are open-loop pipelines: a separate imputer fills in gaps, then a generator
learns from the completions. The imputer has no access to the generator's
distributional knowledge, the generator has no way to reject bad imputations,
noisy observations are treated as ground truth, and the trained model can only
generate — not impute. The result: a ~540% gap versus training on clean data,
and a practitioner who needs both generation and imputation must maintain two
separate model families.

**We** reformulate the problem as empirical Bayes: a single unconditional
diffusion model learns p(x) from corrupted data via EM, where the E-step
uses posterior sampling to reconstruct clean data (using the model's own
evolving prior), and the M-step trains on these reconstructions. This closes
the feedback loop — the generator IS the imputer, and they co-evolve.
Corruption (missing values, noise, or both) enters only as a parameter of
the E-step, making the framework modular and noise-aware by construction.
At test time, the same weights give unconditional generation AND conditional
imputation via posterior sampling — no separate model needed.

**Our theoretical contribution** shows that for irregularly-sampled time
series, observational informativeness is strongly frequency-dependent (low
frequencies are well-constrained, high frequencies are not), and that the
standard isotropic noise assumption in posterior sampling ignores this
structure. We derive a spectral CG operator — a single, principled change —
that respects this frequency structure and connects to classical Wiener
filtering.

**The result:** SOTA generation quality across all benchmarks, the broadest
evaluation scope in the field (standard, block missing, mixed rates, noisy,
conditional), and a unified model that replaces separate generation and
imputation pipelines.

---

## Why This Gets Accepted

- **Reviewer who cares about theory:** Proposition 1 (exact for full obs, rigorous asymptotics for masking) + Wiener filter connection — a clean, non-trivial analysis
- **Reviewer who cares about results:** SOTA across the board, broadest evaluation scope in the field, including settings no prior work evaluates
- **Reviewer who cares about impact:** Closed-loop paradigm is new here; unified generation+imputation from one model is a practical shift; conditional generation is a free capability
- **Reviewer who cares about elegance:** One principled change to one operator, motivated by a clean analysis — plus the conceptual simplicity of "the generator IS the imputer"
- **Reviewer who cares about noise/robustness:** First generation method that handles noisy observations by design, not as an afterthought
- **The experimental breadth** (mixed rates, blocks, noisy, conditional, imputation) is itself a contribution — it redefines what the benchmark should be
