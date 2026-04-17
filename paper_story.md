# Paper Story: One Model, Two Tasks — Unified Generation and Imputation from Corrupted Time Series

## The Gap in the Field

Time series with corrupted observations (missing values, sensor noise, or both) are
ubiquitous in healthcare, finance, and engineering. Two fundamental tasks arise from
this setting:

| Task | Goal | Output |
|------|------|--------|
| **Generation** | Learn the clean data distribution, produce new synthetic samples | Unconditional samples from p(x) |
| **Imputation** | Given a specific corrupted observation, reconstruct the missing/noisy values | Conditional reconstruction x given y |

Despite being two sides of the same coin, the literature treats them as **separate
problems** with separate methods, separate architectures, and separate evaluation
protocols.

---

## What Previous Methods Do

### Generation track

| Method | Architecture | Training | Inference | Limitations |
|--------|-------------|----------|-----------|-------------|
| **ImagenI2R** (Fadlon et al.) | TST encoder + GRU decoder + UNet | Two-stage: (1) TST completion on observed values, (2) diffusion with masked loss | Unconditional sampling | Cannot condition on new observations at test time. Cannot handle noisy observations (TST trains MSE on observed values, treats noise as ground truth). |
| **KoVAE** (Naiman et al.) | NCDE encoder + GRU decoder + Koopman prior | Sequential VAE with linear latent dynamics | Unconditional sampling from VAE | Linear dynamics assumption. NCDE interpolation required. No test-time conditioning. |
| **TimeGAN** (Yoon et al.) | GRU-based GAN | Adversarial + supervised + reconstruction loss | Unconditional sampling | Training instability, mode collapse. No corruption handling beyond simple preprocessing. |

**Common pattern**: All generation methods produce an unconditional model. At test
time, they sample blindly — they cannot incorporate new observations. If a clinician
has a patient's partial ECG and wants a plausible completion, these methods cannot
help. They can only generate entirely new synthetic ECGs.

### Imputation track

| Method | Architecture | Training | Inference | Limitations |
|--------|-------------|----------|-----------|-------------|
| **CSDI** (Tashiro et al.) | Conditional diffusion with attention | Score matching conditioned on observed values | Conditional sampling given observations | Learns only the conditional p(x_missing \| x_observed). Cannot generate unconditional samples. |
| **BRITS** (Cao et al.) | Bidirectional RNN | Reconstruction loss on observed values | Forward pass with observed values | Deterministic — single imputation, no diversity. No generation capability. |
| **GP-VAE** (Fortuin et al.) | VAE with Gaussian Process prior | VAE ELBO with GP temporal prior | Conditional reconstruction | Gaussian assumptions limit expressiveness. Not designed for unconditional generation. |

**Common pattern**: All imputation methods require observed values as input. They
learn p(x_missing | x_observed), not p(x). They cannot generate entirely new samples
from the data distribution.

### The disconnect

These two tracks address **the same underlying problem** — learning about the clean
data distribution p(x) from corrupted observations — but with different
architectures, different losses, different evaluation protocols, and different
assumptions. A practitioner who needs both generation and imputation must train and
maintain **two separate models**.

---

## Our Insight

Both generation and imputation follow from a single statistical formulation:

> **Given corrupted observations Y drawn from a known corruption process Q(Y|X),
> learn the clean data distribution p(X).**

- **Generation** = sample from the learned p(X)
- **Imputation** = compute the posterior p(X | Y) using p(X) and Q(Y|X)

These are not separate problems. They are two inference modes of the **same model**.

The key is that any model of p(X) immediately gives you both capabilities:
1. **Unconditional generation**: sample X ~ p(X)
2. **Conditional imputation**: sample X ~ p(X | Y) via posterior sampling (Bayes' rule)

The reason prior generation methods can't do imputation is not fundamental — it's
because they lack a mechanism for posterior sampling at test time. And prior
imputation methods can't generate because they only model the conditional, not the
joint.

---

## Our Framework

We train a single **unconditional diffusion model** that learns p(X) from corrupted
observations via Expectation-Maximization, then use MMPS posterior sampling to unlock
both generation and imputation from the same weights.

### Training: EM with MMPS

The corruption is a latent variable problem: we observe Y, want to learn p(X).

**E-step** — Reconstruct clean data via MMPS posterior sampling:
- Uses the current unconditional model as prior
- MMPS replaces the standard denoiser with a **posterior denoiser** that incorporates
  observation constraints with Jacobian-based covariance correction
- Handles any corruption expressible as a forward model: missing values (mask),
  noise (sigma_y), or both
- Produces fully reconstructed (no NaN) training data

**M-step** — Standard unconditional diffusion training on the reconstructions:
- No masking, no corruption-specific loss
- The model simply learns p(X) from complete data
- Clean learning signal regardless of corruption type

**Key property**: The corruption model Q(Y|X) only enters the E-step. The M-step is
always the same. This means the framework handles diverse corruptions without
architectural changes.

### Test time: Two inference modes, one model

| Mode | How | Use case |
|------|-----|----------|
| **Generation** | Standard unconditional sampling from the diffusion model | Synthetic data, augmentation, privacy |
| **Imputation** | MMPS posterior sampling conditioned on new observations | Reconstruction, denoising, gap-filling |

No retraining, no additional networks. The same weights serve both purposes.

---

## Why This Unification Matters

### 1. Practical: one model replaces two

A hospital deploying time series AI currently needs:
- A generation model for synthetic training data (privacy, augmentation)
- An imputation model for reconstructing patient records with gaps

Our framework provides both from a single trained model, halving infrastructure,
maintenance, and compute costs.

### 2. The imputation is a byproduct, not a cost

The E-step already does imputation during training — it reconstructs corrupted
observations using the current model. At test time, we simply expose this
capability. There is no additional training cost for the imputation functionality.

### 3. Noisy observations: a structural advantage

Prior generation methods (ImagenI2R, KoVAE) assume observed values are clean:
- TST trains: MSE(output, observed_value) — treats noisy values as ground truth
- NCDE interpolates through noisy observations, propagating noise

Our MMPS E-step explicitly models observation noise via sigma_y:
- The posterior denoiser balances prior (model) vs. likelihood (observations)
- When sigma_y > 0, it naturally denoises while reconstructing
- The M-step trains on denoised data, learning p(X_clean) not p(X_noisy)

This is not a parameter tweak — it's a fundamentally different treatment of
observations that prior generation methods cannot replicate without redesign.

### 4. More robust learned distribution

Empirically, our framework shows:
- **More robust to corruption severity**: the quality gap between easy (30% missing)
  and hard (70% missing) training conditions is smaller than for prior methods
- **More robust to corruption type**: switching between random and block missing
  patterns causes less degradation than for prior methods
- **SOTA absolute performance** across standard benchmarks

This robustness stems from the M-step always training on fully-reconstructed data
with standard loss — the corruption never directly enters the model's gradients.

---

## Contributions

1. **Unification of generation and imputation**: We show that a single unconditional
   diffusion model, trained via EM from corrupted observations, provides both
   unconditional generation and conditional imputation via posterior sampling — tasks
   previously requiring separate methods and architectures.

2. **Principled corruption handling**: By formulating corrupted time series
   generation as latent variable estimation, we obtain a modular framework where the
   corruption model (missing values, noise, or both) is a parameter of the E-step,
   not an architectural choice. This naturally handles noisy observations, which
   prior generation methods cannot.

3. **Domain-specific innovations**: STL-based iterative initialization and curriculum
   missing rate scheduling make EM practical and efficient for time series data.

4. **Comprehensive evaluation**: We evaluate on both generation metrics (disc score,
   FID, predictive score) AND imputation metrics (MSE, MAE), on standard benchmarks
   AND new corruption scenarios (noisy observations, block missing, mixed). SOTA
   results on generation with competitive imputation performance — from the same
   model.

---

## Key Experiments

### Experiment 1: Generation quality (standard benchmarks)
- Datasets: Sines, Stock, Energy, MuJoCo, Weather, ETT, Electricity
- Missing rates: 30%, 50%, 70%
- Metrics: Discriminative score, Context-FID, Predictive score, Correlation
- Baselines: ImagenI2R, KoVAE, GT-GAN, TimeGAN
- **Expected result**: SOTA across the board

### Experiment 2: Imputation quality (new evaluation)
- Same trained model as Experiment 1 (no retraining!)
- Given corrupted test sequences, run MMPS to reconstruct
- Compare reconstructions to clean ground truth
- Metrics: MSE, MAE per feature
- Baselines: CSDI, BRITS, GP-VAE (specialized imputation methods)
- **Expected result**: Competitive with specialized imputation methods while also
  providing generation (which they cannot)

### Experiment 3: Noisy observations
- Training data has Gaussian noise (sigma = 0.1, 0.2, 0.35, 0.45) with or without
  missing values
- Baselines must train on noisy data — their observed values include noise
- Our method sets sigma_y to match noise level
- **Expected result**: Our method generates clean-looking samples; baselines generate
  noisy-looking samples because they learned p(x_noisy)

### Experiment 4: Robustness across corruption types and severities
- Train each method on one corruption setting
- Evaluate generation quality (disc score)
- Show degradation curves across missing rates: our gap (B3-B1) < their gap (A3-A1)
- Show cross-type robustness: train on random, evaluate quality; train on block,
  evaluate quality — our performance more stable

### Experiment 5: Ablation study
- STL initialization vs. Gaussian initialization vs. no initialization
- Curriculum vs. no curriculum
- EMA reset vs. no reset
- Number of EM iterations
- sigma_y sensitivity
- CG iterations (1 vs. 3 vs. 5)

---

## Positioning vs. Related Work

| Paper | Relationship | Our advantage |
|-------|-------------|---------------|
| **DiffEM** (Hosseintabar et al.) | Inspired our EM framework. They train conditional model for images. | We use unconditional model + MMPS, applied to time series with domain-specific innovations (STL, delay embedding, curriculum). We demonstrate dual generation+imputation capability. |
| **ImagenI2R** (Fadlon et al.) | Previous SOTA for irregular TS generation. | We achieve better generation AND add imputation capability. Handle noisy observations. More robust across corruptions. Simpler architecture (no TST/decoder). |
| **KoVAE** (Naiman et al.) | Previous VAE baseline for irregular TS generation. | Better generation quality. Dual capability. No linear dynamics assumption. |
| **CSDI** (Tashiro et al.) | Conditional diffusion for TS imputation. | We match imputation quality while also providing generation. One model vs. their conditional-only model. |
| **EM-MMPS** (Rozet & Bhatt) | MMPS posterior sampling for images. | We bring MMPS to time series via delay embedding, add STL initialization, curriculum, and demonstrate the generation+imputation unification. |

---

## Paper Title Candidates

1. "One Model, Two Tasks: Unified Time Series Generation and Imputation from Corrupted Data via Diffusion EM"
2. "Learning from Corrupted Time Series: A Diffusion EM Framework for Joint Generation and Imputation"
3. "Generate and Reconstruct: Unified Diffusion Models for Corrupted Time Series"
