# Paper Story v6: Act-Observe-Adapt

**Working Title:** "Act, Observe, Adapt: A Self-Calibrating Paradigm
for Time Series Generation from Corrupted Data"

---

## The Core Selling Point

Every existing method for learning generative models from corrupted
time series is **static**: choose a pipeline, fix the hyperparameters,
run training, hope for the best. The pipeline is as follows:
irregular ts → imputer → regular ts → generator. The imputer doesn't
know what the generator needs. The generator doesn't know where the
imputer failed.
Nobody checks whether the reconstruction was faithful at 0.1 Hz vs
10 Hz. The system is blind to its own failure modes.

We propose a different paradigm — one where the learning process
**monitors its own posterior uncertainty, diagnoses where it
succeeds and fails, and adapts its behavior accordingly**. Inspired
by the Act-Observe-Plan-Evaluate loop in adaptive decision-making
systems, our method:

1. **Acts** — reconstructs corrupted data using the model's own
   evolving prior (posterior sampling)
2. **Observes** — diagnoses reconstruction quality per frequency band
   via posterior disagreement across multiple samples
3. **Plans** — adapts the spectral trust profile: tighten where
   evidence is strong, relax where the model is uncertain
4. **Evaluates** — assesses convergence and decides whether to
   continue, adjust, or stop

This is not a static pipeline. It is a **self-calibrating learning
loop** that allocates modeling effort where the data can support it
and relies on learned structure where it cannot.

The result: a single unconditional diffusion model that generates,
imputes, conditionally generates, and denoises time series — all
from one set of weights. No prior method in the time series
generation literature offers this.

---

## 1. The Problem

Learning generative models from irregular time series: missing
values, sensor noise, variable sampling rates. The data you have
is not the data you want to model.

Formally: given corrupted observations y = Ax + n (binary masking +
Gaussian noise), learn p(x) — the distribution of clean, complete
sequences — and sample from it.

The field treats this as multiple separate problems: generation
methods produce synthetic data but ignore corruption at test time;
imputation methods fill gaps but cannot generate; denoising methods
remove noise but assume complete observations. A practitioner
needing multiple capabilities trains multiple models.

**We argue this fragmentation stems from a deeper issue: existing
methods do not learn from their own mistakes.** They commit to a
reconstruction strategy, train on the result, and never revisit
whether that strategy was appropriate. The system cannot distinguish
between "I reconstructed this well" and "I guessed and got lucky."

---

## 2. The Static Pipeline Problem

### How every existing generative method works

All current generation approaches follow what we call the
**commit-and-forget** pattern:

```
Corrupted data → [Fixed reconstruction] → Completed data → [Generator training] → Done
```

The reconstruction strategy is chosen once and never revisited.
The generator trains on whatever comes out. Three consequences:

**1. No self-assessment.** The system cannot tell whether a
reconstruction was faithful or fabricated. A time series with a
missing week in the middle gets "completed" by interpolation —
the generator trusts this equally whether the gap was 2 points
(easy) or 200 points (speculative). The pipeline has no mechanism
to signal "I'm guessing here."

**2. No frequency awareness.** Corruption affects different frequency
bands differently. Missing 50% of samples at random barely affects
low-frequency trends (neighboring points carry redundant information)
but devastates high-frequency content (each missing sample is
unrecoverable). Static pipelines treat all frequencies identically —
they cannot distinguish "I know the trend" from "I'm fabricating
the texture."

**3. No adaptation.** Even if the generator learns that certain
reconstructions were implausible (e.g., generated samples never
resemble certain training reconstructions), this information never
flows back to improve reconstruction. The two stages are
disconnected: better generation knowledge cannot produce better
training data.

These are not independent flaws. They share a root cause: **the
learning process does not monitor or adapt its own behavior.** The
system is open-loop — it executes a fixed plan regardless of
intermediate outcomes.

### Specific instances

- **ImagenTime:** TST imputer commits to completions → diffusion
  trains on them. The masking augmentation mitigates blind trust
  but the imputer never improves from generation feedback.
- **GT-GAN / KoVAE:** NCDE interpolates → GAN/VAE generates.
  Interpolation quality bounds generation quality permanently.
- **All of the above:** Observed values treated as ground truth.
  When sensors are noisy, the model learns p(noisy), not p(clean).

---

## 3. A New Paradigm: Self-Calibrating Generative Learning

### The key shift

Instead of committing to a fixed reconstruction strategy, we
propose that the learning process should **continuously monitor its
own reconstruction quality and adapt accordingly**. This transforms
generative learning from a static pipeline into an adaptive loop.

The paradigm has four phases, repeated iteratively:

### Act: Reconstruct via Posterior Sampling

The model uses its own current prior to reconstruct corrupted
observations. This is not imputation (filling gaps with a separate
model) — it is posterior sampling: drawing x ~ p(x | y, A) where
p(x) is the model's own learned distribution.

Crucially, we draw **multiple samples** (K per observation). A
single reconstruction gives a point estimate. Multiple
reconstructions reveal **where the model is confident and where it
is uncertain.** This is the information the system needs to
self-calibrate.

### Observe: Diagnose Posterior Uncertainty per Frequency Band

Given K posterior samples, we compute inter-sample disagreement in
the frequency domain. For each frequency f, the coefficient of
variation across K samples tells us:

- **Low CV at frequency f:** the K reconstructions agree → the
  model is confident → observations constrain this frequency well
- **High CV at frequency f:** the K reconstructions disagree → the
  model is uncertain → observations provide little information here

This is a **spectral X-ray of the model's own epistemic state.**
It answers, for each frequency band, the question: "Am I
reconstructing or hallucinating?"

### Plan: Adapt the Observation Trust Profile

The diagnosis drives adaptation. The posterior sampling uses an
observation noise model Σ_y that controls how much to trust
observations vs. the learned prior at each frequency:

```
Σ_y(f) = σ_y² · (1 + (f / f_cutoff)^n)
```

The cutoff f_cutoff determines the boundary between "trust the
data" (below cutoff) and "trust the prior" (above cutoff). After
each Observe step, we adapt f_cutoff **bidirectionally**:

- If high-frequency bands show low disagreement → **raise** f_cutoff
  (the model can be trusted at higher frequencies now)
- If high-frequency bands show high disagreement → **lower** f_cutoff
  (the model is hallucinating — rely more on learned structure)

This adaptation is smooth (blended with momentum) and bounded.
The system automatically navigates the trust-observation frontier
as its prior improves across iterations.

### Evaluate: Assess and Decide

After each Act-Observe-Plan cycle, the updated reconstructions
train the model (standard score matching). Then we evaluate
generation quality against the clean reference distribution
(unconditional samples vs. original uncorrupted data):

- **Improving** → continue the loop
- **Plateaued** → the model has converged
- **Degrading** → early-stop, restore best checkpoint

The Evaluate step closes the outer loop: better evaluation informs
whether the Plan was effective, which influences subsequent
iterations.

### The complete loop

```
┌─────────────────────────────────────────────────────────┐
│                Self-Calibrating EM Loop                  │
│                                                         │
│  ┌──────┐    ┌─────────┐    ┌──────┐    ┌──────────┐   │
│  │ ACT  │───▶│ OBSERVE │───▶│ PLAN │───▶│ EVALUATE │──┐│
│  │      │    │         │    │      │    │          │  ││
│  │ K    │    │ Spectral│    │ Adapt│    │ Train +  │  ││
│  │poster│    │ disagree│    │ Σ_y  │    │ assess   │  ││
│  │ ior  │    │ ment    │    │      │    │          │  ││
│  │ samp │    │ per freq│    │ cutoff│   │ converge?│  ││
│  └──────┘    └─────────┘    └──────┘    └──────────┘  ││
│       ▲                                        │       ││
│       └────────────────────────────────────────┘       ││
│              (model improves → better Act)              │
│                                                         │
│  Outer loop: better prior → better reconstructions      │
│  Inner loop: diagnosis → better trust allocation        │
└─────────────────────────────────────────────────────────┘
```

**Two nested feedback loops:**
1. The **outer loop** (EM): better model → better reconstructions
   → better training data → better model
2. The **inner loop** (self-calibration): posterior disagreement →
   adapted trust → more faithful reconstructions

Static pipelines have neither. Standard EM has only the outer loop.
Our contribution is both.

---

## 4. Why This Paradigm Fits Corrupted Time Series

### Time series have structured identifiability

Although our method operates in image space via the delay embedding
(following the ImagenTime framework), the underlying signal structure
remains fundamentally temporal. The delay embedding rearranges a 1D
sequence into a 2D grid, but pixel correlations still reflect temporal
adjacency — and crucially, the temporal frequency hierarchy persists:
low frequencies are over-determined by sparse observations, high
frequencies are under-determined. This is not an accident — it follows
from the physics of sampling, and no change of representation can
eliminate it.

**Proposition 1 (Frequency-dependent informativeness).** For a
stationary process observed through random masking at rate p with
noise σ_n, the relative informativeness at frequency f is:

```
R(f) = σ_n² / (p · S_xx(f) + σ_n²)
```

For signals with decaying power spectra (universal in natural time
series): R(f) → 0 at low f (observations are highly informative),
R(f) → 1 at high f (observations are uninformative).

**This is why self-calibration matters for time series specifically.**
The boundary between "identifiable" and "hallucinated" is
frequency-dependent and shifts as the model improves. A static
method cannot track this boundary. Our Observe step measures it
empirically; our Plan step adapts to it.

### The Wiener-optimal structure

**Proposition 2.** When the observation noise model is frequency-
dependent, the posterior correction takes the Wiener filter form:

```
w(f) ≈ σ_t² S_J(f) / (Σ_y(f) + σ_t² S_J(f))
```

Signal power over total power — the minimum mean-square-error
linear estimator. This is not something we impose; it emerges
from making Σ_y frequency-aware. The Act step naturally performs
Wiener-optimal reconstruction when the Plan step has correctly
calibrated Σ_y.

### Connection to diffusion's coarse-to-fine schedule

Diffusion models already resolve structure from coarse to fine:
high noise levels (early steps) determine global shape; low noise
levels (late steps) add detail. Our noise-dependent cutoff aligns
with this:

```
f_cutoff(σ_t) = max(0.05, f_cutoff_base · (1.0 - σ_t / σ_max))
```

Early diffusion steps trust only the lowest frequencies. Late steps
incorporate progressively higher frequencies, up to the limit
determined by the Plan step's current calibration. When σ_t is very
small (σ_t/σ_max < 0.05), we fall back to isotropic — at near-zero
noise, all frequencies are directly observable.

---

## 5. Multi-Task Capabilities

The paradigm produces a standard unconditional diffusion model —
not a specialized pipeline. Because the self-calibrating loop
trains p(x) directly, the framework naturally supports multiple
inference modes:

| Inference mode         | Mechanism                              | Requires   |
| ---------------------- | -------------------------------------- | ---------- |
| **Generation**         | Unconditional sampling                 | Nothing    |
| **Imputation**         | Posterior sampling given (y, A)        | Obs + mask |
| **Conditional gen.**   | Posterior sampling given partial obs   | Partial    |
| **Denoising**          | Posterior sampling with Σ_y for noise  | Noisy obs  |

In practice, achieving strong performance on each task may require
task-specific training or incorporating multiple tasks during
training. The architectural framework supports all modes, but
the quality of each depends on the training configuration.

Our model also handles **new types of irregular time series** that
have never been tested before in previous works on generative models
from irregular time series, such as mixed missing rates, block
missingness, and more.

---

## 6. Domain Adaptations

### 6.1 Warm-start via STL decomposition

Rather than beginning the loop from a random prior (requiring many
Act-Observe-Plan cycles to converge), we initialize with iterative
STL decomposition of corrupted data. STL separates trend + seasonal
+ residual — a frequency-aligned warm-start that gives the first
Act step a reasonable prior to work with.

### 6.2 Curriculum: stabilizing early iterations

Early iterations face a cold-start problem: the prior is weak, so
posterior sampling is noisy, so the Observe step sees high
disagreement everywhere, so the Plan step trusts almost nothing.
We stabilize this by temporarily revealing extra positions (filled
with previous-iteration reconstructions), decaying from ~30% extra
to zero by the final iterations. This gives the early Act step more
to work with without contaminating the final model.

### 6.3 Full pipeline

```
Algorithm: Self-Calibrating EM for Corrupted Time Series
──────────────────────────────────────────────────────────
Input:  Corrupted observations {(yᵢ, Aᵢ)}, noise level σ_y
Output: Unified diffusion model q_θ(x)

Phase 1 — Warm-Start:
  1. Compute STL decomposition of corrupted data
  2. Train initial θ₀ on STL reconstructions

Phase 2 — Self-Calibrating EM Loop:
  3. For k = 0, ..., K-1:
     a. Curriculum: augment masks (annealing to zero)
     b. ACT: Draw K posterior samples per observation
        using spectral CG with current f_cutoff
     c. OBSERVE: Compute per-frequency inter-sample CV
     d. PLAN: Adapt f_cutoff from spectral diagnosis
     e. Train q_{θ_{k+1}} on K×N augmented samples
     f. EVALUATE: Assess generation quality;
        early-stop if no improvement for P iterations

Test time (supported inference modes):
  Generate     → unconditional sampling
  Impute       → posterior sampling given (y_new, A_new)
  Conditional  → posterior sampling with partial observations
  Denoise      → posterior sampling with Σ_y for noise level
  (task-specific or multi-task training may be needed per mode)
```

---

## 7. Contributions

1. **A self-calibrating paradigm for learning from corrupted data.**
   We introduce Act-Observe-Plan-Evaluate as an organizing principle
   for iterative generative learning. The model monitors its own
   posterior uncertainty in the frequency domain and adapts its
   observation trust profile accordingly. This transforms EM from a
   static iterative procedure into an adaptive one. No prior method
   in time series generation includes self-monitoring of
   reconstruction quality.

2. **Spectral self-diagnosis via posterior disagreement.** We show
   that inter-sample coefficient of variation in the Fourier domain
   provides a practical, per-frequency measure of model confidence.
   This diagnostic drives bidirectional adaptation of the spectral
   filter and doubles as a convergence indicator.

3. **Frequency-aware posterior sampling grounded in theory.** We
   prove that observational informativeness under masking is
   frequency-dependent (Proposition 1) and that frequency-aware Σ_y
   yields Wiener-optimal posterior corrections (Proposition 2). The
   adaptive loop calibrates this theoretically-motivated structure
   against empirical posterior uncertainty.

4. **Multi-task framework.** The paradigm produces a standard
   unconditional diffusion model whose architecture supports
   generation, imputation, conditional generation, and denoising.
   Achieving strong performance per task requires task-specific or
   multi-task training, but the framework is unified — no prior
   method in the time series generation literature supports this
   range of tasks within a single architecture.

5. **Broadest evaluation scope in the field.** Standard benchmarks
   (30/50/70% missing) plus block missingness, mixed per-sample
   rates, noisy observations, conditional generation, and imputation
   from the same model. The evaluation redefines the benchmark.

---

## 8. Experimental Design

### The multi-task table

The most powerful result: one framework across every task.

| Task                         | Best prior method          | Our method          |
| ---------------------------- | -------------------------- | ------------------- |
| Generation (30% missing)     | ImagenTime (pipeline)      | **Our framework**   |
| Generation (50% missing)     | ImagenTime                 | **Our framework**   |
| Generation (70% missing)     | ImagenTime                 | **Our framework**   |
| Generation (block missing)   | No prior results           | **Our framework**   |
| Generation (mixed rates)     | No prior results           | **Our framework**   |
| Generation (noisy + missing) | No prior results           | **Our framework**   |
| Imputation                   | CSDI (dedicated imputer)   | **Our framework**   |
| Conditional generation       | Not available              | **Our framework**   |

### The paradigm ablation (tells the whole story)

| Method                       | Loop   | Self-calibrating | Multi-task       |
| ---------------------------- | ------ | ---------------- | ---------------- |
| ImagenTime                   | Open   | No               | Gen only         |
| Static EM (isotropic)        | Closed | No               | Gen + Imp + Cond |
| Static EM (spectral, fixed)  | Closed | No (fixed Σ_y)   | Gen + Imp + Cond |
| **Ours (adaptive spectral)** | Closed | **Yes**          | Gen + Imp + Cond |

**What this reveals:**

1. **Open → Closed** (largest jump): Closing the loop is the
   structural win. Multi-task capability, noise handling, and
   self-improving training all come from this step.
2. **Isotropic → Fixed spectral** (consistent improvement): Making
   Σ_y frequency-aware improves reconstruction quality, grounded
   in Propositions 1-2.
3. **Fixed → Adaptive spectral** (the paradigm contribution): Self-
   calibration from posterior disagreement further improves quality
   and provides automatic convergence monitoring. The model knows
   which frequency bands it has learned and which it is still
   guessing.

### Settings

| Setting                                         | What it tests                  | Why the paradigm enables it                   |
| ----------------------------------------------- | ------------------------------ | --------------------------------------------- |
| **Standard** (30/50/70%, lengths 24-10920)      | Baseline comparison            | Expected by reviewers                         |
| **Mixed missing rates** (per-sample ~ U(0,0.7)) | Heterogeneous corruption       | Per-sample (A, Σ_y) is native to the loop     |
| **Block missingness** (contiguous gaps)          | Hardest setting; tests prior   | Self-calibration detects unreliable HF bands   |
| **Conditional generation**                      | Downstream application         | Posterior sampling = free capability           |
| **Noisy + irregular**                           | Simultaneous corruption        | Σ_y absorbs noise; static methods cannot      |
| **Imputation** (same model)                     | Dual-use                       | Learned p(x) supports both tasks natively     |

### What to show about the self-calibration

Key plots that validate the paradigm (beyond standard metric tables):

- **f_cutoff trajectory across EM iterations**: shows the system
  learning to trust higher frequencies as the prior improves.
  Different datasets / corruption levels produce different
  trajectories. This is the paradigm in action.
- **Per-frequency CV heatmap** (frequency × EM iteration): shows
  disagreement shrinking from high to low frequencies as training
  progresses. The "learning frontier" moving rightward is a
  compelling visual.
- **Spectral PSD comparison** (generated vs. ground truth): shows
  that adaptive calibration produces better spectral fidelity than
  fixed approaches, especially at the mid-frequency boundary.
- **EM convergence curve** with early-stopping: shows the Evaluate
  step correctly identifying when to stop.

---

## 9. Addressing Concerns

### "This is just EM with adaptive hyperparameters."

EM provides the outer loop. The self-calibration is the inner loop
— it uses the model's own posterior uncertainty to adapt the
observation model in the frequency domain. This is absent from all
prior EM applications. The distinction matters: standard EM with
fixed hyperparameters works (our ablation shows this), but it
cannot distinguish identifiable from non-identifiable frequency
content. The self-calibrating variant can — and this is
measurable.

### "The adaptation is a simple rule, not real agency."

Correct — and we do not claim agency. We claim *inspiration* from
the Act-Observe-Plan-Evaluate structure as an organizing principle
for adaptive iterative learning. The specific adaptation mechanism
(inter-sample Fourier CV → bidirectional cutoff adjustment) is
intentionally simple, interpretable, and grounded in the spectral
theory. Complexity is not the goal; appropriate adaptation is.

### "Isn't this just DiffEM applied to time series?"

DiffEM provides the EM-with-diffusion-posteriors framework, which
we credit. Our contributions are: (a) the self-calibration paradigm
— monitoring posterior disagreement to adapt the observation model,
absent from DiffEM; (b) the spectral theory motivating why
adaptation matters for time series specifically; (c) the unified
multi-task demonstration; (d) domain adaptations. The ablation
quantifies each.

### "Proposition 1 assumes stationarity."

Exact under stationarity + Bernoulli masking. The qualitative
conclusion (low frequencies better constrained than high) holds
for any signal with decaying spectral energy — verified on all
datasets. The self-calibration makes the theory's limitations less
consequential: even if the theoretical Σ_y shape is imperfect, the
Observe step detects the mismatch and the Plan step corrects it.
Theory provides the initial design; adaptation handles the rest.

### "How expensive is K-sample E-step?"

K posterior samples cost K× a single E-step. With K=3 (our default)
and warm-started M-steps, total training is ~3-4× a single-sample
EM baseline. We report exact GPU-hours per dataset. The self-
calibration itself (FFT + CV computation + cutoff update) is
negligible: <1% of E-step time.

### "Test-time corruption differs from training?"

The learned prior is unconditional — it does not encode corruption.
Posterior sampling at test time accepts any (A, Σ_y). Block
missingness experiments validate this: trained on random masking,
evaluated on structured blocks the model never saw.

---

## 10. Known Limitations

1. **Linear Gaussian observation model.** p(y|x) = N(Ax, Σ_y) with
   known A. Non-linear corruptions (quantization, clipping) not
   covered.
2. **K-sample cost.** Multi-sample E-step multiplies reconstruction
   cost by K. Effective but not free.
3. **Spectral theory scope.** Proposition 1 tightest for large T,
   i.i.d. masking. Short sequences tested empirically, not analyzed.
4. **Simple adaptation rule.** The cutoff adjustment is a scalar
   update from aggregate frequency statistics. More sophisticated
   per-frequency or per-sample adaptation is possible but not
   explored.
5. **Gaussian moment matching.** Posterior sampling approximation;
   may be less accurate for multi-modal distributions at
   intermediate noise levels.
6. **High-frequency identifiability.** Under severe corruption,
   high-frequency content relies on inductive bias. The Observe step
   correctly identifies this (high CV), but the generated content
   at those frequencies is learned structure, not data-driven.

---

## 11. The Narrative Arc

**The problem.** Learning from corrupted time series today uses
static pipelines: reconstruct once, train once, hope the
reconstruction was good enough. The system never checks.

**The insight.** The model can check. By drawing multiple posterior
samples, it can measure its own uncertainty — per frequency band,
per EM iteration. This transforms "Was my reconstruction faithful?"
from an unanswerable question into a measurable quantity.

**The paradigm.** Act (reconstruct via posterior sampling), Observe
(diagnose uncertainty in the frequency domain), Plan (adapt the
observation trust profile), Evaluate (assess and decide). Two
nested loops: EM improves the model; self-calibration ensures each
EM step is as faithful as the current evidence allows.

**The theory.** Why this matters specifically for time series:
observations constrain different frequencies to different degrees
(Proposition 1), and the optimal posterior correction is a Wiener
filter whose parameters the self-calibration loop learns to set
(Proposition 2).

**The result.** A single unconditional diffusion model that
replaces separate generation, imputation, and denoising pipelines.
The broadest evaluation scope in the field. And a paradigm where
the model doesn't just learn from corrupted data — it learns *how
well* it's learning from corrupted data.

---

## 12. Why This Gets Accepted

| Reviewer type  | What they see                                                                                            |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| **Novelty**    | New paradigm: self-calibrating generative learning from corrupted data. Not incremental over any prior.   |
| **Impact**     | First unified model for irregular TS. Replaces separate gen/imp/denoise pipelines. Practical benefit.     |
| **Theory**     | Props 1-2 ground the spectral design. Wiener filter emerges naturally. Honest about scope.                |
| **Method**     | The self-calibration is interpretable (CV → cutoff), grounded (theory), and ablated (three-row ablation). |
| **Results**    | SOTA generation + beats specialist imputers + broadest eval scope + novel settings.                       |
| **Elegance**   | "The model measures where it's guessing and stops pretending." One loop, many capabilities.               |

**The acceptance case in one sentence:** "We show that generative
learning from corrupted time series should not be static — the
model can monitor its own posterior uncertainty, adapt its
observation model, and in doing so produce a single model that
generates, imputes, and denoises better than dedicated methods for
each task."

---

## 13. Positioning vs. v5 Framing

The v5 "Closing the Loop" framing centers on **unification** —
one model does everything. The paradigm is EM; the spectral
correction is a domain adaptation.

The v6 "Act-Observe-Adapt" framing centers on **self-calibration**
as the key methodological idea. The unification is a consequence,
not the headline. The emphasis shifts to: the model knows what it
knows and what it doesn't, per frequency, per iteration.

**v6 is stronger when:**
- The adaptive spectral loop (K-sample diagnosis → f_cutoff
  adaptation) shows clear empirical gains over fixed spectral EM
- The f_cutoff trajectory / CV heatmap plots are visually compelling
- Reviewers value methodological novelty over application scope

**v5 is stronger when:**
- The multi-task unification is the dominant result
- The spectral adaptation gain is modest
- Reviewers care more about "what can the model do" than "how does
  it learn"

The ideal paper may combine both: the v6 paradigm as the method
story (Sections 1-4), the v5 multi-task table as the results story
(Sections 5-7). The question is which goes in the title and
abstract.

---

## 14. What We Credit vs. What We Claim

**We credit:**
- EM with diffusion posteriors (Rozet et al., 2024)
- MMPS/CG posterior sampling machinery
- The Act-Observe-Plan-Evaluate structure as a conceptual framework
  from adaptive decision-making

**We claim:**
1. The self-calibrating paradigm: monitoring posterior disagreement
   to adapt observation trust, applied to generative learning
2. Spectral self-diagnosis via inter-sample Fourier CV
3. The spectral theory (Props 1-2) motivating frequency-dependent
   trust for time series
4. A multi-task framework for generation, imputation, conditional
   generation, and denoising from corrupted data
5. Domain adaptations (STL init, curriculum, noise-dependent cutoff)
6. The broadest evaluation scope in the field
