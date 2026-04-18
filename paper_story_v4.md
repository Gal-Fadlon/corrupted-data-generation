# Paper Story v4: Frequency-Aware Diffusion for Corrupted Time Series

**Working Title:** "Spectral Posterior Sampling for Time Series Generation from Irregular Observations"

---

## 1. The Problem

Learning a generative model for regular time series from irregularly-observed
data (missing values, sensor failures, variable sampling). Formally:
given corrupted observations y = Ax + noise with per-sample binary masks A,
learn p(x) and sample complete, regular sequences.

**The irregular-regular gap:** Existing generation methods trained on
corrupted observations produce substantially worse results than models
trained on fully-observed data (we quantify this per-dataset and
per-metric in Table 1). Closing this gap is the paper's goal.

---

## 2. The Landscape: Generation and Imputation Are Artificially Separated

The literature treats generation and imputation as **separate problems** —
separate methods, architectures, evaluation protocols — despite both
stemming from the same underlying question: learning p(x) from corrupted
observations.

**Generation methods** (ImagenTime, KoVAE, GT-GAN, TimeGAN) produce
unconditional samples but cannot condition on new observations at test time.
A clinician with a patient's partial ECG cannot get a plausible completion —
only an entirely new synthetic ECG.

**Imputation methods** (CSDI, BRITS, GP-VAE) fill in missing values
conditioned on observations but cannot generate unconditionally. They
require observed values as input.

**Diffusion-based methods** (TSDiff, SSSD) have been applied to both tasks
but train separate models or fine-tune with task-specific objectives for
each. They do not learn a single unconditional prior from corrupted data
that serves both tasks through the same inference mechanism.

A practitioner who needs both generation and imputation from the same
corrupted training set must train and maintain **separate models**.

---

## 3. Three Structural Limitations of Existing Generation Methods

Every existing generation method follows the same open-loop architecture:

```
Irregular data → [Imputer] → Completed data → [Generator] → Samples
```

- GT-GAN / KoVAE: NCDE imputes → GAN/VAE generates
- ImagenTime: TST imputes → Diffusion generates + masking

We identify three structural problems with this paradigm:

**1. No feedback — the pipeline is one-directional.** The imputer learns
only from observed values. The generator learns only from imputed data.
The generator's growing understanding of the data distribution never
flows back to improve imputation. The system's ceiling is the imputer's
ability to guess from partial observations alone.

**2. No noise handling.** Prior generation methods assume observed values
are clean. TST trains with MSE(output, observed_value) — treating noisy
values as ground truth. NCDE interpolates through noisy observations,
propagating noise. When observations carry sensor noise (ubiquitous in
real deployments), these methods learn p(x_noisy), not p(x_clean). This
is not a parameter tweak — it requires fundamental redesign.

**3. Generation-only at test time.** Once trained, these models can only
sample unconditionally. If partial observations arrive at test time, they
require entirely separate imputation machinery.

**These are not independent weaknesses — they share a root cause:** the
separation of imputation and generation into two disconnected stages.

---

## 4. Our Approach

### What we build on (credited)

Rozet et al. (2024) showed that learning a prior q_θ(x) from corrupted
observations y ~ N(Ax, Σ_y) can be cast as empirical Bayes solved via EM:

- **E-step:** Sample x ~ q_{θ_k}(x | y, A) using the current prior
  via posterior sampling (MMPS with CG solver).
- **M-step:** Train q_{θ_{k+1}}(x) on these samples via standard
  denoising score matching.

This produces a proper diffusion model at every iteration — the M-step
is unmodified score matching, no custom training objective. The posterior
sampling uses Tweedie's covariance and a CG solver with operator:

```
CG_op(v) = Σ_y · v + σ_t² · A · VJP(Aᵀ v)
```

This framework was demonstrated for images with isotropic Σ_y = σ²I.
Notably, during the DiffEM review process, Reviewer XeV5 asked: "What
is the role of Σ_y? What if it's anisotropic or non-diagonal?" The
authors responded that any Σ_y can be absorbed by whitening. **We argue
the opposite: for time series, the structure of Σ_y carries critical
information that should not be whitened away.**

### What we contribute (claimed)

Applying the EM framework to time series with isotropic Σ_y already
solves problems 1 (feedback) and 3 (unified inference) by construction,
and partially solves problem 2 (noise enters via σ_y). **But it does not
address the frequency-dependent nature of time series observations** —
and this is where the framework underperforms for time series.

Our contribution is threefold:

**(a) Theory:** We prove that observational informativeness for masked
time series is strongly frequency-dependent (Proposition 1) and that
isotropic Σ_y creates a systematic mismatch. We derive a spectral CG
operator whose posterior corrections take the Wiener filter form
(Proposition 2).

**(b) Method:** We design a spectral CG operator (one equation change),
a noise-dependent frequency cutoff that aligns spectral corrections
with the diffusion coarse-to-fine schedule, frequency-aware
initialization via STL decomposition, and a curriculum that anneals
auxiliary observations across EM iterations.

**(c) Evaluation:** We establish the broadest evaluation scope in the
field — standard benchmarks plus block missingness, mixed missing rates,
noisy observations, conditional generation, and imputation from the
same model — settings no prior generation method addresses.

---

## 5. Theoretical Contribution: Why Time Series Demands Frequency-Aware Posteriors

### Proposition 1 (Frequency-dependent observational informativeness)

For a stationary time series observed through random masking, observations
constrain different frequencies to vastly different degrees: low frequencies
are well-determined by sparse observations while high frequencies are
essentially unconstrained.

**Setup.** Let x ∈ ℝ^T be a zero-mean stationary Gaussian process with
power spectral density S_xx(f). Under periodic boundary conditions,
the covariance is circulant: C_x = F^H diag(S_xx) F where F is the DFT
matrix. Observe y = Ax + n where A = diag(a₁,...,a_T) is a binary mask
with each aᵢ ~ Bernoulli(p), and n ~ N(0, σ_n² I).

**Exact result (fully observed, p = 1).** When A = I, the DFT
diagonalizes both the prior covariance and the observation operator.
The posterior variance at frequency f is:

```
V_post(f) = [1/S_xx(f) + 1/σ_n²]⁻¹ = S_xx(f) · σ_n² / (S_xx(f) + σ_n²)
```

**Approximate extension to random masking (0 < p < 1).** Under random
masking, the DFT no longer exactly diagonalizes the observation operator.
However, for large T, the cross-frequency interference concentrates
around zero (each off-diagonal entry of F A F^H has variance
p(1-p)/T → 0), yielding approximately T independent channels with
effective noise σ_n²/p:

```
V_post(f) ≈ S_xx(f) · σ_n² / (p · S_xx(f) + σ_n²)
```

**Scope and limitations of this result:**
- The approximation requires T >> 1/(p(1-p)). For our shortest
  benchmark (T = 24 at p = 0.3), the concentration is weaker; for
  T = 10920 it is tight. We verify empirically that the qualitative
  conclusion (monotonic frequency dependence) holds across all T.
- The stationarity assumption is used to diagonalize the prior in the
  DFT basis. Real time series may be non-stationary. However, the
  qualitative insight — slow variations are better constrained by
  sparse observations than fast variations — holds for any signal
  whose spectral energy decays with frequency, stationary or not.
  We verify this empirically by computing R(f) on all datasets.
- The Gaussian process assumption yields exact expressions. For
  non-Gaussian processes, the result characterizes the second-order
  (MMSE linear) structure and serves as a lower bound on uncertainty.

**The key quantity: relative informativeness.**

```
R(f) = V_post(f) / S_xx(f) = σ_n² / (p · S_xx(f) + σ_n²)
```

For time series with decaying spectra (S_xx(f) decreasing with |f| — a
well-documented property of natural signals, empirically verified across
all our datasets):

- Low f where S_xx(f) >> σ_n²/p:  R(f) → 0  (highly informative)
- High f where S_xx(f) << σ_n²/p: R(f) → 1  (uninformative)

**Intuition.** Low-frequency components vary slowly — neighboring observed
timesteps carry redundant information. High-frequency components change
between every pair of timesteps — each missing sample destroys information
no neighbor can recover.

### Corollary: Isotropic Σ_y mismatches time series observation structure

Σ_y = σ²I treats all frequencies identically, forcing the posterior to
weight observational evidence equally across all frequencies. Proposition 1
shows this over-trusts observations at high frequencies (uninformative)
and under-utilizes them at low frequencies (highly informative).

### Proposition 2 (Wiener filter structure)

When Σ_y is frequency-dependent and the CG system matrix is approximately
diagonal in the DFT basis, the posterior correction at each diffusion step
takes the Wiener filter form:

```
w(f) ≈ σ_t² S_J(f) / (Σ_y(f) + σ_t² S_J(f))
```

Signal power over total power. This connects our spectral design to the
MMSE-optimal linear estimator: when Σ_y(f) is large (high frequencies),
the prior dominates; when Σ_y(f) is small (low frequencies), observations
dominate.

**Caveat.** The frequency-diagonal approximation of A V[x|x_t] Aᵀ
becomes less accurate when the mask is very sparse or T is small. In
these regimes the Wiener interpretation is qualitative, not exact.

---

## 6. The Method: Spectral Posterior Sampling for Time Series

### 6.1 The spectral CG operator

Standard (isotropic):
```
CG_op(v) = σ_y² · v + σ_t² · A · VJP(Aᵀ v)
```

**Ours (spectral):**
```
CG_op(v) = Σ_y(f) · v + σ_t² · A · VJP(Aᵀ v)
```

where:
```
Σ_y(f) = σ_y² · (1 + (f / f_cutoff(σ_t))^n)
```

PSD, diagonal in DFT basis (FFT → scale → IFFT), parameterized by two
interpretable quantities: f_cutoff and filter order n.

**Theory-to-method bridge (honest accounting).** Proposition 1 gives
the functional form R(f) = σ_n² / (p · S_xx(f) + σ_n²), which depends
on the unknown S_xx(f). Rather than estimating S_xx from corrupted data
(introducing another source of error), we use a Butterworth
parameterization that captures the key qualitative property —
monotonically increasing Σ_y(f) — with only 2 free parameters. This is
a design choice motivated by the theory, not a direct consequence of it.
The filter order n controls steepness (default n = 2); f_cutoff is
determined by the noise-dependent schedule (Section 6.2).

### 6.2 Noise-dependent frequency cutoff

Diffusion models generate coarse-to-fine: at high noise levels, the
denoiser captures only low-frequency modes. Incorporating high-frequency
corrections before the denoiser can represent that detail introduces
artifacts. We tie the trusted frequency band to the diffusion noise level:

```
f_cutoff(σ_t) = max(0.05,  1.0 - σ_t / σ_max)
```

At early diffusion steps (large σ_t, σ_t/σ_max close to 1), f_cutoff
is small and Σ_y(f) is large for most frequencies — the prior dominates.
At late steps (small σ_t), f_cutoff approaches 1.0 and observations are
incorporated at progressively higher frequencies. The floor of 0.05
ensures the lowest frequencies always receive observational corrections.

When σ_t is very small (σ_t/σ_max < 0.05), the denoiser has already
resolved most structure and the posterior is nearly converged; we fall
back to isotropic Σ_y = σ_y²I to avoid unnecessary spectral
manipulation at this terminal stage.

The linear schedule is a simple default; we ablate against constant
and exponential schedules (Appendix), finding that the precise schedule
matters less than the principle of expanding the trusted band.

### 6.3 Frequency-aware initialization

Rather than initializing the EM from a random Gaussian prior, we
initialize from iterative STL decomposition of the corrupted data.
STL separates trend (low-frequency) from seasonal (periodic) from
residual, providing a warm-start naturally aligned with the spectral
structure our method exploits. The trend seeds low-frequency
reconstruction; the seasonal component seeds periodic modes. This
reduces EM iterations to convergence, which is important for
computational cost (Section 8).

### 6.4 Curriculum: annealing auxiliary observations

At early EM iterations, the prior is weak and posterior sampling
struggles with severely missing data. We use a curriculum that
temporarily reveals extra positions to ease the E-step:

At EM iteration k, we compute a progress ratio
`progress = min(1.0, (k+1) / (0.6 · total_iters))` and sample
additional positions with probability `reveal_prob = (1 - progress) · 0.3`.
The revealed positions are filled with values from the previous
iteration's reconstructions (not ground truth). This provides the
posterior sampler with denser observations early on, when it needs
the most help, and smoothly removes the scaffold as the prior
strengthens.

By the final third of EM iterations, reveal_prob reaches zero and the
E-step operates on the original corrupted data alone — ensuring the
final model is not dependent on the curriculum.

### 6.5 Full algorithm (in paper: Algorithm 1 in main text)

```
Algorithm: Spectral EM for Time Series Generation
─────────────────────────────────────────────────
Input: Corrupted observations {(yᵢ, Aᵢ)}ᵢ₌₁ᴺ, noise level σ_y
Output: Trained diffusion model q_θ(x)

Phase 1 — Initialization:
  1. Compute STL reconstructions of {(yᵢ, Aᵢ)}
  2. Train θ₀ on STL reconstructions via score matching

Phase 2 — EM iterations:
  3. For k = 0, 1, ..., K-1:
     a. Curriculum: augment masks with extra reveals from
        previous reconstructions (annealing to zero)
     b. E-step: For each (yᵢ, Aᵢ), draw one reconstruction
        xᵢ ~ q_{θ_k}(x | yᵢ, Aᵢ) using spectral CG posterior
        sampling with f_cutoff(σ_t) = max(0.05, 1 - σ_t/σ_max)
     c. Reset EMA weights
     d. M-step: Train θ_{k+1} on {xᵢ} via denoising score
        matching, warm-starting from θ_k
     e. Evaluate: Periodically compute generation metrics

Phase 3 — Final training:
  4. Train a fresh model on the final reconstructions {xᵢ}
     via standard unconditional training (full epoch budget)

Test time:
  Generation   → unconditional sampling from q_{θ_final}
  Imputation   → posterior sampling from q_{θ_final}(x | y_new, A_new)
  (same model, same weights, no retraining)
```

**Note on the M-step:** This is standard, unmodified denoising score
matching. The resulting model is a proper unconditional diffusion model —
no custom objective, no architectural changes — compatible with any
downstream sampling scheme.

**Note on Phase 3:** The final training from scratch on the last
reconstructions serves as a clean slate: the Phase 2 model may carry
artifacts from early (poor) EM iterations in its optimization state.
Phase 3 gives the best reconstructions a fresh training run.

---

## 7. Contributions

1. **Problem analysis.** We identify three structural limitations of
   existing generation methods for irregular time series and trace them
   to the separation of imputation and generation. This analysis is
   independent of our method and clarifies the design space.

2. **Spectral theory for time series posterior sampling.** We prove that
   observational informativeness under masking is frequency-dependent
   (Proposition 1) and derive a spectral CG operator connected to
   classical Wiener filtering (Proposition 2). This is the paper's
   core theoretical contribution and is absent from all prior
   applications of diffusion posterior sampling.

3. **Domain-specific adaptations.** STL-based frequency-aware
   initialization, noise-dependent frequency cutoff aligned with the
   diffusion coarse-to-fine schedule, and a curriculum that anneals
   auxiliary observations — each motivated by the spectral theory and
   contributing measurably in ablations.

4. **Unified generation + imputation from a single model.** Posterior
   sampling provides imputation as a free capability of the same
   trained prior. Unlike TSDiff/SSSD which require separate training
   procedures for each task, our model uses the same weights and
   inference mechanism for both.

5. **Broadened evaluation.** We evaluate under the broadest scope in
   the field: standard benchmarks (30/50/70%), plus block missingness,
   mixed missing rates, noisy observations, conditional generation, and
   imputation — settings no prior generation method addresses.

---

## 8. Addressing Computational Cost (Proactive)

**The concern (from DiffEM reviews):** EM iterations with posterior
sampling sound prohibitively expensive.

**Our response (four mitigating factors):**

1. **One sample per observation.** Unlike the original DiffEM (which
   draws K samples per observation), our E-step draws a single
   posterior sample per corrupted sequence. The training set size
   remains N (not K×N), keeping M-step cost constant.

2. **Warm-starting the M-step.** Each M-step initializes from the
   previous iteration's parameters — fine-tuning, not training from
   scratch. In DiffEM, 32 EM iterations on CIFAR-10 took ~4 days on
   4 A100s — comparable to AmbientDiffusion's single-pass training.

3. **STL initialization reduces EM iterations.** By warm-starting with
   frequency-aware STL reconstructions (not random noise), the prior
   starts closer to the target and requires fewer EM iterations to
   converge. [Report actual iteration count vs. random init in
   Table X.]

4. **The spectral CG overhead is marginal.** Our spectral CG operator
   replaces a scalar multiplication (σ_y² · v) with an FFT → scale →
   IFFT operation. This adds negligible cost per CG iteration (one
   rfft + irfft per step). MMPS with 1 CG iteration (our default)
   adds only ~16ms per diffusion step over DPS (Rozet et al., 2024),
   with ~10% more memory.

**Total wall-clock comparison.** [Report: our method vs. ImagenTime
vs. isotropic EM, in GPU-hours per dataset. The key message: the
overhead is a small constant factor, not an order of magnitude.]

---

## 9. Addressing Identifiability (Proactive)

**The concern (from DiffEM reviews):** When corruption is severe,
distinct priors p(x) can produce the same observation distribution
p(y). Can the EM converge to the wrong distribution?

**Our response:**

This is an inherent limitation of learning from corrupted data, shared
by all methods in the field (including AmbientDiffusion, DiffEM, and
open-loop approaches). The identifiability question is: which aspects
of p(x) can be recovered from p(y)?

Proposition 1 provides a precise answer for our setting: **low-frequency
components are identifiable; high-frequency components are not** (when
S_xx(f) << σ_n²/p, the posterior equals the prior). Our spectral Σ_y
explicitly encodes this: at frequencies where observations are
uninformative, the model relies on its inductive bias (the diffusion
architecture and score matching objective) rather than pretending to
extract information that isn't there. Isotropic methods, by contrast,
implicitly claim equal identifiability at all frequencies.

In practice, the diffusion architecture's inductive bias (smoothness,
temporal coherence) constrains the solution space at high frequencies.
Block missingness is the hardest test of identifiability — it removes
contiguous regions where the model must rely entirely on the learned
prior. We evaluate this setting explicitly.

---

## 10. Experimental Design

### Baselines

**Generation:** ImagenTime, KoVAE, GT-GAN, TimeGAN (open-loop),
isotropic EM (our closed-loop ablation without spectral).

**Imputation:** CSDI, BRITS, GP-VAE (to compare imputation quality
of our generation model against dedicated imputation methods).

**Diffusion time-series:** TSDiff (if applicable — TSDiff learns
from clean data, so it serves as an oracle upper bound, not a direct
competitor. We note this distinction clearly.)

### Settings

| Setting | What it tests | Why we can, others can't |
|---|---|---|
| **Standard** (30/50/70%, lengths 24-10920) | Baseline comparison | Expected by reviewers |
| **Mixed missing rates** (per-sample rate ~ U(0,0.7)) | Realistic heterogeneous corruption | EM handles per-sample A natively; open-loop methods assume fixed rates |
| **Block missingness** (prefix/suffix/middle/random blocks) | Contiguous gaps — the hardest setting; tests identifiability | Prior must have truly learned the distribution, not just interpolation |
| **Conditional generation** (observe partial, generate rest) | Practical downstream application | Posterior sampling is a free capability; prior methods need separate machinery |
| **Noisy + irregular** (missing + Gaussian noise) | Simultaneous corruption | Σ_y(f) handles both naturally; prior methods treat noisy values as ground truth |
| **Imputation quality** (same model, no retraining) | Dual-use capability | Generation baselines need a separate model |

### Robustness story

- **Across severity:** Quality gap between 30% and 70% missing is smaller
  than for prior methods — M-step always trains on fully-reconstructed data.
- **Across corruption type:** Switching between random and block missing
  causes less degradation — corruption model is a parameter, not architecture.
- **Noise immunity:** Prior methods degrade proportionally (learn p(x_noisy));
  our Σ_y absorbs noise and M-step trains on denoised reconstructions.

---

## 11. Ablation Design

### Full pipeline ablation

| Method | Feedback | Noise handling | Imputation | Spectral |
|---|---|---|---|---|
| ImagenTime (open-loop baseline) | No | No | No | No |
| Isotropic closed-loop (EM + σ²I) | Yes | Yes (σ_y) | Yes | No |
| **Ours (spectral closed-loop)** | **Yes** | **Yes (Σ_y(f))** | **Yes** | **Yes** |

Three jumps: open→closed (structural), isotropic→spectral (theoretical),
generation-only→unified (practical). Each contributes measurably.

### Independent spectral CG validation (isolating our contribution)

To validate the spectral CG independently from the EM loop:

**Fixed-prior experiment.** Take a diffusion model trained via EM
with isotropic Σ_y. Then at test time, perform posterior sampling
with: (a) isotropic CG, (b) spectral CG. This isolates the spectral
operator's contribution from the EM training dynamics. If spectral CG
improves reconstruction quality with the same prior, the benefit is
purely in posterior sampling — independent of training.

### Component ablations

| Component | Ablated against | What it shows |
|---|---|---|
| STL initialization | Random/Gaussian init | Faster convergence; fewer EM iterations |
| Noise-dependent cutoff | Constant cutoff across σ_t | Coarse-to-fine alignment matters |
| Filter order n | n = 1, 2, 4 | Robustness to steepness; n = 2 is default |
| Curriculum (extra reveals) | No curriculum | Stabilizes early EM iterations |
| EMA reset | No reset | Prevents M-step from stale momentum |
| Phase 3 (fresh training) | Use Phase 2 model directly | Clean slate improves final quality |

---

## 12. The Narrative Arc (Elevator Pitch)

**Previous methods** for learning generative models from irregular time
series are open-loop pipelines: a separate imputer fills in gaps, then
a generator learns from the completions. The imputer never benefits from
the generator's distributional knowledge, noisy observations are treated
as ground truth, and the trained model can only generate — not impute.

**The closed-loop alternative** — EM with diffusion priors (Rozet et al.,
2024) — addresses the structural limitations: the model's own evolving
prior drives reconstruction, corruption is a parameter, and the same model
generates and imputes. But the posterior sampling in this framework treats
observation noise as isotropic, which is appropriate for images but
wrong for time series.

**Our key insight** is that posterior sampling for time series requires
frequency-aware noise modeling. We prove that sparse observations
constrain low frequencies far more than high frequencies (Proposition 1),
that isotropic Σ_y ignores this structure, and that a spectral CG
operator yields Wiener-optimal corrections at each diffusion step
(Proposition 2). Combined with a noise-dependent frequency cutoff that
aligns with the diffusion coarse-to-fine schedule and frequency-aware
STL initialization, this produces a method that is both principled and
practical.

**Result:** SOTA across all benchmarks and the broadest evaluation scope
in the field, with a single model replacing separate generation and
imputation pipelines.

---

## 13. Preempting Reviewer Concerns

### "Isn't this just DiffEM applied to time series?"

No. DiffEM provides the EM framework (credited in Background, ~1 page).
Our contribution is showing *why* that framework fails for time series
without frequency-aware posterior sampling (Propositions 1-2) and
designing the spectral operator + domain adaptations that make it work.
The ablation (isotropic EM vs. spectral EM) quantifies this directly:
the closed-loop structure alone is insufficient; the spectral correction
is essential.

### "Proposition 1 assumes stationarity/Gaussianity/Bernoulli masking."

The exact result is for this setting. The qualitative conclusion — low
frequencies better constrained than high — holds for any signal with
decaying spectral energy, stationary or not. We verify this empirically
on all datasets, including non-stationary ones. The Butterworth
parameterization depends only on the qualitative monotonicity, not on
the specific functional form of S_xx(f).

### "The Butterworth shape is a heuristic, not a theorem."

Acknowledged explicitly in Section 6.1. Proposition 1 motivates the
monotonic increase of Σ_y(f) with frequency. The Butterworth is a
minimal parameterization of this property (2 parameters vs. O(T) free
spectral values). We ablate against alternative shapes and filter
orders; the key property is monotonicity, not the specific functional
form.

### "Computational cost of EM iterations."

Addressed in Section 8. One sample per observation (not K) + warm-
starting + STL init keep total cost within a small constant factor of
single-pass training.

### "What about identifiability?"

Addressed in Section 9. Proposition 1 precisely characterizes which
frequencies are identifiable and which must rely on inductive bias.
Block missingness experiments directly test this.

### "TSDiff/SSSD also do generation + imputation."

TSDiff and SSSD learn from clean data — they cannot handle the corrupted-
data setting. When clean data is available, they are not our competitors
but upper bounds. When only corrupted data is available (our setting),
they cannot be applied. We include TSDiff as an oracle comparison where
applicable.

### "What if test-time corruption differs from training?"

The corruption model (A, σ_y) is a parameter of the E-step during
training and of posterior sampling at test time. The learned prior
q_θ(x) is unconditional — it does not encode a specific corruption
pattern. Test-time posterior sampling can use any (A, σ_y), including
patterns not seen during training. Block missingness experiments test
this: the model trains on random masking but is evaluated on structured
block patterns.

### "The f_cutoff is fixed — what about dataset-specific spectral structure?"

The noise-dependent schedule f_cutoff(σ_t) = max(0.05, 1 - σ_t/σ_max)
is dataset-agnostic by design: it encodes only the universal principle
that the diffusion process resolves frequencies coarse-to-fine. The
filter order n (default 2) and σ_y control the steepness and scale.
We ablate these and find the method is robust across datasets — the
Butterworth monotonicity, not the precise cutoff value, is what matters.
If dataset-specific tuning is desired, n and σ_y can be set via
validation on corrupted data.

---

## 14. Positioning: What We Credit, What We Claim

**We credit:** The empirical Bayes / EM framework for diffusion priors
(Rozet et al., 2024) and the posterior sampling machinery (MMPS, CG
solvers, Tweedie's covariance). These appear in Background (~1 page),
clearly cited.

**We claim:**

1. **Problem analysis** — identifying three structural limitations
   of open-loop methods and their shared root cause.
2. **Spectral theory** — Propositions 1-2 showing why time series
   posterior sampling demands frequency-dependent treatment.
3. **Spectral CG operator** — one principled equation change to the
   CG operator, motivated by Props 1-2.
4. **Domain adaptations** — STL initialization, noise-dependent
   frequency cutoff, observation curriculum for EM stability.
5. **Broadened evaluation** — comprehensive realistic corruption
   settings that redefine what the benchmark should be.

**The framing:** We do not modify the posterior sampling machinery
(MMPS/CG). We modify **what enters it** (Σ_y → Σ_y(f)), motivated
by time-series-specific theory. This is a cleaner, more focused novelty
claim than modifying the sampling algorithm itself.

**In the paper itself:** The EM/MMPS machinery appears in Background.
The Method section focuses entirely on our contributions. The reader
should come away thinking "frequency-aware posterior sampling for time
series" — not "DiffEM applied to time series."

---

## 15. Why This Gets Accepted

| Reviewer type | What they see |
|---|---|
| **Theory** | Proposition 1 (exact + rigorous asymptotics) + Wiener filter connection — clean, non-trivial, time-series-specific |
| **Results** | SOTA across the board; broadest evaluation scope in the field |
| **Impact** | Unified generation+imputation from one model; closed-loop paradigm is new for TS; conditional generation is free |
| **Elegance** | One principled change to one operator; "the generator IS the imputer" |
| **Robustness** | First generation method that handles noisy observations by design; better degradation across severity and corruption types |
| **Computational** | Honest cost analysis; within small constant factor of baselines |

---

## 16. Known Limitations (to include in paper)

1. **Linear Gaussian observation model.** We require p(y|x) = N(Ax, Σ_y)
   with known A. Non-linear corruptions (quantization, clipping,
   non-linear sensor responses) are not covered.

2. **Spectral theory scope.** Proposition 1 is tightest for large T and
   i.i.d. Bernoulli masking. For short sequences or structured
   missingness (which we test experimentally but do not analyze
   theoretically), the quantitative predictions are approximate.

3. **Identifiability at high frequencies.** When corruption is severe,
   high-frequency content is filled by inductive bias, not data. The
   model cannot distinguish between the true high-frequency structure
   and plausible alternatives consistent with p(y).

4. **Computational overhead.** EM iterations add cost relative to
   single-pass methods. Warm-starting and STL initialization mitigate
   this, but the method is not suitable for settings where training
   budget is extremely constrained.

5. **Gaussian moment-matching approximation.** The posterior sampling
   relies on approximating p(x|x_t) as Gaussian. This is accurate for
   large σ_t (where the true posterior concentrates) and for small σ_t
   (where p(x|x_t) approaches a delta), but may be less accurate at
   intermediate noise levels for highly multi-modal distributions.

---

## Appendix: Differences from v3

**Structural changes (v3 → v4):**
- Credit boundary moved to Section 4 (before theory), not at the end
- Computational cost addressed proactively (Section 8)
- Identifiability addressed proactively (Section 9)
- Independent spectral CG validation added to ablation (Section 11)
- Preempting reviewer concerns section (Section 13) based on DiffEM
  rebuttal analysis
- Known limitations section (Section 16)
- ~540% claim removed (replaced with per-dataset quantification)
- TSDiff/SSSD addressed explicitly
- Theory-to-method bridge made honest (Butterworth acknowledged as
  design choice)
- Algorithm box specified for main text
- Approximation quality discussed in limitations

**Alignment with run_diffem_mmps_spectral_cg.py (v4 update):**
- Removed adaptive spectral tuning / inter-sample Fourier CV
  (this was the agentic version, not the core method)
- Removed early stopping via discriminative monitoring
  (the standard EM loop evaluates periodically but doesn't
  auto-stop)
- Removed multi-sample E-step (K samples). The actual method
  draws 1 reconstruction per observation per EM iteration.
- Added curriculum with annealing auxiliary observations
  (extra reveals from prior reconstructions, decaying to zero)
- Added Phase 3 (fresh unconditional training on final
  reconstructions)
- Fixed f_cutoff base = 1.0 (not externally adaptive)
- Made f_cutoff formula exact: max(0.05, 1.0 - σ_t/σ_max)
  with isotropic fallback when σ_ratio < 0.05
- Simplified contributions and positioning to match simpler method
