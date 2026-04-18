# DS-MMPS: Dual-Space Moment-Matching Posterior Sampling

---

## 1. The Dual-Space Observation Model

State-of-the-art time series generation operates in a **lifted representation**: delay embedding maps time series to images, and a standard image diffusion model (EDM) learns the distribution. This creates a dual-space system:

- **Observation space**: time series, $\mathbb{R}^{N_\text{ts} \times F}$
- **Model space**: images, $\mathbb{R}^{C \times H \times W}$
- **Lift**: $\texttt{ts\_to\_img} : \mathbb{R}^{N_\text{ts} \times F} \to \mathbb{R}^{C \times H \times W}$ (delay embedding)
- **Projection**: $\texttt{img\_to\_ts} : \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{N_\text{ts} \times F}$ (averaging overlapping windows)

Corrupted observations are defined in TS space: $y_\text{ts} = A_\text{ts} \cdot x_\text{ts}$ where $A_\text{ts}$ is a binary mask. The diffusion model operates in image space. Any EM framework must cross this boundary at every iteration: the E-step must produce posterior samples in image space conditioned on TS-space observations.

The **composed observation operator** is:

$$G = A_\text{ts} \circ \texttt{img\_to\_ts} : \mathbb{R}^{N_\text{img}} \to \mathbb{R}^{N_\text{obs} \times F}$$

This maps from image space (model space) to masked TS space (observation space). It is the correct operator for the forward model $y_\text{ts} = G \cdot x_\text{img}$.

Three structural properties of $G$ that break existing posterior sampling methods:

1. **$G$ is not a mask in model space.** It crosses two spaces via $\texttt{img\_to\_ts}$ (which averages overlapping delay-embedding windows). Standard methods assume $A$ operates within model space.

2. **$G$ is rank-deficient.** $\texttt{img\_to\_ts}$ maps from $\mathbb{R}^{N_\text{img}}$ to $\mathbb{R}^{N_\text{ts}}$ where $N_\text{img} > N_\text{ts}$, then $A_\text{ts}$ further reduces dimension. The null space of $G$ corresponds to off-manifold directions — images that have no valid time-series interpretation.

3. **$G^\top G$ is not diagonal.** Because $\texttt{img\_to\_ts}$ averages overlapping windows, $G^\top G = \texttt{ts\_to\_img} \cdot A_\text{ts}^\top \cdot A_\text{ts} \cdot \texttt{img\_to\_ts}$ has structured off-diagonal correlations.

---

## 2. Why Existing Posterior Sampling Methods Fail

All diffusion-based posterior sampling methods start from the same decomposition (Bayes' rule on the score):

$$\nabla_{x_t} \log p(x_t \mid y) = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{prior score (denoiser)}} + \underbrace{\nabla_{x_t} \log p(y \mid x_t)}_{\text{likelihood score (must approximate)}}$$

The prior score is available from the trained denoiser $d_\theta(x_t, \sigma)$ via Tweedie's formula. The challenge is approximating the likelihood score $\nabla_{x_t} \log p(y \mid x_t)$.

The core integral is:

$$p(y \mid x_t) = \int p(y \mid x) \, p(x \mid x_t) \, dx$$

Every method approximates $p(x \mid x_t)$ differently. We analyze each in the dual-space setting.

### 2.1 DPS — Diffusion Posterior Sampling (Chung et al., 2023)

**Approximation.** Replace $p(x \mid x_t)$ with a delta function at the Tweedie mean:

$$p(y \mid x_t) \approx \mathcal{N}\big(y \;\big|\; \mathcal{A}(\mathbb{E}[x \mid x_t]),\; \Sigma_y\big)$$

This is a **zeroth-order** approximation — it uses only the posterior mean, ignoring covariance entirely.

**General limitation.** When $\sigma_t$ is large, $p(x \mid x_t)$ is broad and the delta approximation is poor. The DiEM paper (Rozet et al., 2024) notes: "DPS is unstable, does not properly cover the support of the posterior $p(x \mid y)$ and often leads to samples $x$ which are inconsistent with the observation $y$." Recent analysis (Luo et al., 2025) shows DPS actually performs MAP estimation rather than posterior sampling, producing high-quality but low-diversity outputs.

**Dual-space failure.** In EM, posterior diversity is essential — the M-step trains on the empirical distribution of E-step samples. MAP-like behavior collapses each observation's completion to a single point, preventing the model from learning distributional structure. Additionally, the gradient $\nabla_{x_t} \|y_\text{ts} - G \cdot \mathbb{E}[x \mid x_t]\|^2$ requires backpropagation through $G = A_\text{ts} \circ \texttt{img\_to\_ts}$, but the point estimate provides no information about how uncertain the correction should be at different image positions — it applies the same correction strength everywhere regardless of whether a position contributes to observed or unobserved time points.

### 2.2 PiGDM — Pseudo-Inverse Guided Diffusion Models (Song et al., 2023)

**Approximation.** Use a **diagonal** approximation to the posterior covariance:

$$\mathbb{V}[x \mid x_t] \approx \text{diag}(v_t)$$

$$\nabla_{x_t} \log p(y \mid x_t) \approx -\Sigma_t^{-1} A^\top \big(A \, \text{diag}(v_t) \, A^\top + \Sigma_y\big)^{-1} (y - A \, \mathbb{E}[x \mid x_t])$$

This captures per-pixel variance but misses all correlations between pixels.

**General limitation.** The diagonal assumption is valid when the observation operator $A$ is a simple mask in model space and the data distribution has approximately independent pixel-level noise. It fails when the forward model introduces correlations between dimensions.

**Dual-space failure.** Delay embedding creates **structured off-diagonal correlations** in $\mathbb{V}[x_\text{img} \mid x_t]$. Each time point $x_\text{ts}[i]$ maps to $d$ overlapping image positions via $\texttt{ts\_to\_img}$ (where $d$ is the delay dimension). The true covariance has dense $d \times d$ blocks encoding this redundancy. A diagonal approximation treats these $d$ copies as independent, systematically underestimating covariance along the $\texttt{ts\_to\_img}$ manifold by a factor of approximately $d$. For our Energy dataset ($d = 8$), the pseudo-inverse correction is ~8x too weak in the manifold-aligned directions, making the posterior samples drift off-manifold.

### 2.3 TMPD — Tweedie Moment Projected Diffusions (Boys et al., 2023)

**Approximation.** Use a **row-sum diagonal** approximation to the Jacobian:

$$\mathbb{V}[x \mid x_t] \approx \text{diag}(\mathbf{1}^\top J) \quad \text{where } J = \nabla_{x_t} d_\theta(x_t, \sigma)$$

This aggregates each row of the Jacobian into a single scalar per dimension.

**General limitation.** The row-sum approximation is only valid when $\mathbb{V}[x \mid x_t]$ is nearly diagonal. For data lying on low-dimensional manifolds (the setting where MMPS excels, as shown in DiEM Figure 2), the covariance has strong off-diagonal structure that the row-sum collapses.

**Dual-space failure.** Same fundamental issue as PiGDM. The row-sum $\text{diag}(\mathbf{1}^\top J)$ destroys the structured block correlations from delay embedding. For inverse problems where $G^\top G$ has significant off-diagonal mass — as in our setting where $G^\top G = \texttt{ts\_to\_img} \cdot A_\text{ts}^\top \cdot A_\text{ts} \cdot \texttt{img\_to\_ts}$ — the approximation error grows with the condition number of $G^\top G$.

### 2.4 DiffPIR — Denoising Diffusion for Plug-and-Play Restoration (Zhu et al., 2023)

**Approximation.** Decouple denoising and data fidelity into alternating proximal steps:

$$\tilde{x} = d_\theta(x_t, \sigma_t) \qquad \text{(denoise)}$$
$$x_{t-1} = \text{prox}_{\lambda \|y - A \cdot \|^2}(\tilde{x}) \qquad \text{(data fidelity)}$$

No covariance information is used. The proximal step enforces observation consistency after the denoiser runs.

**General limitation.** The alternating structure means denoising and data fidelity operate independently. The proximal step can push the sample off the learned data manifold, and there is no covariance-based weighting to modulate correction strength by noise level.

**Dual-space failure.** The proximal operator for the composed observation model requires solving:

$$(I + \lambda G^\top G) \, x = \tilde{x} + \lambda G^\top y$$

Because $G^\top G = \texttt{ts\_to\_img} \cdot A_\text{ts}^\top \cdot A_\text{ts} \cdot \texttt{img\_to\_ts}$ is not diagonal and not block-diagonal, this system has **no closed-form solution**. One would need iterative solvers (CG, ADMM) inside the proximal step — at which point DiffPIR collapses to something similar to MMPS but without the Bayesian justification from moment matching. Additionally, the denoised output $\tilde{x}$ is not constrained to $\text{Range}(\texttt{ts\_to\_img})$, and the proximal step has no mechanism to enforce this constraint.

### 2.5 MMPS — Moment-Matching Posterior Sampling (Rozet et al., 2024)

**Approximation.** Use the **full Jacobian** via Tweedie's covariance formula:

$$\mathbb{V}[x \mid x_t] \approx \sigma_t^2 \, J \quad \text{where } J = \nabla_{x_t} d_\theta(x_t, \sigma)$$

$$\nabla_{x_t} \log q(y \mid x_t) = J^\top A^\top \big(\Sigma_y + \sigma_t^2 \, A \, J \, A^\top\big)^{-1} (y - A \, \mathbb{E}[x \mid x_t])$$

The linear system $(\Sigma_y + \sigma_t^2 \, A \, J \, A^\top) v = r$ is solved via conjugate gradient (CG), where each CG iteration requires exactly one vector-Jacobian product (VJP) of the denoiser. The Jacobian $J$ is never materialized.

**Why MMPS is the best starting point.** Among all surveyed methods, MMPS is uniquely suited to the dual-space setting because:

- It uses the **full covariance** (not diagonal or row-sum), capturing the structured correlations from delay embedding.
- The CG solver handles **arbitrary linear operators** implicitly — the operator $A$ never needs to be materialized or inverted, only applied via matrix-vector products.
- The DiEM paper demonstrates that using the true Tweedie covariance (vs. heuristics) yields "orders of magnitude more accurate posteriors" — precisely the accuracy needed for MCEM convergence.

**But MMPS was derived for single-space problems.** The derivation (DiEM, Eq. 7–10) assumes:

$$p(y \mid x) = \mathcal{N}(y \mid A x, \Sigma_y) \quad \text{where } A : \mathbb{R}^N \to \mathbb{R}^M \text{ operates in the model's native space}$$

In our setting, the true forward model is:

$$p(y_\text{ts} \mid x_\text{img}) = \mathcal{N}(y_\text{ts} \mid G \cdot x_\text{img}, \Sigma_y) \quad \text{where } G = A_\text{ts} \circ \texttt{img\_to\_ts}$$

Naively applying MMPS with an image-space mask $A_\text{img}$ instead of the correct composed operator $G$ introduces **three structural misspecifications**.

---

## 3. Three Misspecifications of MMPS in Dual-Space

### Misspecification 1 (M1): Wrong Observation Operator

**The problem.** Standard MMPS in the delay-embedding setting uses an image-space mask $A_\text{img} = \texttt{ts\_to\_img}(\text{mask\_ts})$ as the observation operator. This is a **redundant approximation** of the true operator $G = A_\text{ts} \circ \texttt{img\_to\_ts}$:

- $A_\text{img}$ marks **multiple image pixels** for the same observed time point (because $\texttt{ts\_to\_img}$ replicates values across overlapping windows).
- $A_\text{img}$ treats these duplicated positions as **independent observations**, inflating the effective number of constraints.
- The CG system $(\Sigma_y + \sigma_t^2 \, A_\text{img} \, J \, A_\text{img}^\top) v = r$ has dimension $N_\text{img}$ (e.g., $1 \times 32 \times 32 = 1024$), when the true observation space has dimension $N_\text{obs} \times F$ (e.g., $12 \times 3 = 36$) — a **28x mismatch**.

**Mathematical consequence.** The posterior score correction $J^\top A_\text{img}^\top v$ from solving the wrong system does not equal the posterior score correction $J^\top G^\top v'$ from the correct system. The error is not a simple scaling — it has different directional components because $A_\text{img}^\top$ and $G^\top = \texttt{ts\_to\_img} \cdot A_\text{ts}^\top$ are structurally different operators.

### Misspecification 2 (M2): Unbounded CG Conditioning

**The problem.** In our setting, observations are noiseless: $y_\text{ts} = A_\text{ts} \cdot x_\text{ts}$ with no observation noise. The parameter $\Sigma_y = \sigma_y^2 I$ has no physical meaning — it is purely a regularization parameter for the CG system. With fixed $\sigma_y$:

$$\kappa\big(\sigma_y^2 I + \sigma_t^2 \, A J A^\top\big) = \frac{\sigma_y^2 + \sigma_t^2 \lambda_\text{max}}{\sigma_y^2 + \sigma_t^2 \lambda_\text{min}} \;\xrightarrow{\sigma_t \to \infty}\; \frac{\sigma_t^2 \lambda_\text{max}}{\sigma_y^2}$$

This grows **without bound** as $\sigma_t$ increases. At high noise levels (early reverse diffusion steps), the CG system is catastrophically ill-conditioned. CG fails to converge in the allotted iterations, producing garbage corrections. The resulting posterior samples are dominated by numerical noise rather than the likelihood signal.

**Why this doesn't arise in DiEM's experiments.** In DiEM's CIFAR-10 and MRI settings, observations include actual noise ($\Sigma_y = 10^{-4} I$ or $10^{-6} I$), giving $\sigma_y$ a physical grounding. In our noiseless TS setting, $\sigma_y$ is arbitrary and must be set adaptively.

### Misspecification 3 (M3): No Output Space Constraint

**The problem.** The delay embedding maps $N_\text{ts}$ time points into $N_\text{img} > N_\text{ts}$ image pixels. Valid images (those corresponding to actual time series) form a **linear subspace** $\text{Range}(\texttt{ts\_to\_img}) \subset \mathbb{R}^{N_\text{img}}$. The MMPS posterior correction:

$$\hat{x} = d_\theta(x_t, \sigma) + \sigma_t^2 \cdot J^\top A^\top v$$

lies in $\mathbb{R}^{N_\text{img}}$ with no mechanism to constrain it to $\text{Range}(\texttt{ts\_to\_img})$. The off-manifold components (the null space of $\texttt{img\_to\_ts}$) are invisible to the observation constraint but enter the M-step training data.

**Compounding across EM iterations.**
1. E-step produces off-manifold completions.
2. M-step trains the denoiser to reproduce off-manifold structure.
3. Next E-step's Tweedie mean $\mathbb{E}[x \mid x_t]$ has off-manifold components → CG corrections inherit them.
4. Off-manifold energy accumulates, degrading generation quality.

**Why this doesn't arise in single-space EM.** In image inpainting (DiEM's CIFAR-10 setting), every output of the denoiser is a valid image — there is no "manifold constraint" to violate. The delay embedding's redundancy creates a strict subspace constraint that single-space MMPS ignores because it never needed to handle one.

### Compounding Factor: Cold Start

All three misspecifications are worst when the model is weak. In early EM iterations:

- The denoiser $d_\theta$ is poorly trained → the Tweedie mean $\mathbb{E}[x \mid x_t]$ is uninformative → CG corrections are large and unreliable.
- The Jacobian $J = \nabla_{x_t} d_\theta$ is poorly conditioned → the CG system inherits this conditioning on top of M2.
- Off-manifold energy in the M-step training data is highest because the model hasn't learned the valid subspace yet.

Without a domain-aware initialization, the first EM iterations produce completions that bear no resemblance to valid time series, creating a chicken-and-egg problem that may take many iterations to escape (if it does at all).

---

## 4. DS-MMPS: Correcting MMPS for Dual-Space EM

We propose **DS-MMPS** (Dual-Space Moment-Matching Posterior Sampling): an adaptation of MMPS that corrects misspecifications M1–M3 and provides a domain-aware initialization. DS-MMPS uses the correct observation model $y_\text{ts} = G \cdot x_\text{img}$ where $G = A_\text{ts} \circ \texttt{img\_to\_ts}$, yielding the correct likelihood score:

$$\nabla_{x_t} \log q(y_\text{ts} \mid x_t) = J^\top G^\top \big(\sigma_y^2 I + \sigma_t^2 \, G \, J^\top \, G^\top\big)^{-1} (y_\text{ts} - G \cdot d_\theta(x_t, \sigma))$$

This is not a new approximation — it is the standard MMPS formula with the correct operator substituted. The novelty is identifying that this substitution is necessary, proving its consequences, and designing the resulting system for the dual-space setting.

### Architecture

```
                Standard MMPS                              DS-MMPS (Ours)
                (Single-Space)                             (Dual-Space)

  Observation   y = A_img · x + noise                     y_ts = A_ts · img_to_ts(x_img)
  model         A operates in model space                  G = A_ts ∘ img_to_ts crosses spaces
                ↓                                          ↓
  CG system     (σ_y² I + σ_t² A_img J A_img^T) v = r    (σ_y² I + σ_t² G J^T G^T) v = r
                dim = N_img  (e.g. 1024)                   dim = N_obs × F  (e.g. 36)
                fixed σ_y → κ grows with σ_t               σ_y = c·σ_t → κ bounded
                ↓                                          ↓
  Output        x̂ ∈ R^{N_img}                             Π(x̂) ∈ Range(ts_to_img)
                (no manifold constraint)                   (orthogonal projection + obs enforcement)

  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║ Misspecification          │ Consequence              │ DS-MMPS Fix       ║
  ╠═══════════════════════════╪══════════════════════════╪═══════════════════╣
  ║ M1: Wrong operator        │ CG solves wrong system   │ Obs-Space CG      ║
  ║ M2: Unbounded κ           │ CG diverges at high σ_t  │ Adaptive σ_y      ║
  ║ M3: No output constraint  │ Off-manifold drift       │ Manifold Π        ║
  ║ Cold start                │ All above amplified      │ Domain-aware init ║
  ╚═══════════════════════════╧══════════════════════════╧═══════════════════╝
```

### 4.1 Observation-Space CG — Corrects M1

**Proposition 4 (Observation-Space CG).** Let $G = A_\text{ts} \circ \texttt{img\_to\_ts}$ and $G^\top = \texttt{ts\_to\_img} \circ A_\text{ts}^\top$. The CG system:

$$(\sigma_y^2 I + \sigma_t^2 \, G \, J^\top \, G^\top) \, v = r, \quad v \in \mathbb{R}^{N_\text{obs} \times F}$$

yields the same posterior score correction as solving the full image-space system with the exact observation operator $G$, but with CG dimension reduced from $N_\text{img}$ to $N_\text{obs} \times F$. Each CG iteration requires exactly one VJP of the denoiser — identical cost to standard MMPS.

**Proof sketch.** The matrix-vector product $(G \, J^\top \, G^\top) v$ decomposes as:

1. $G^\top v$: embed $v$ from masked TS to image via $\texttt{ts\_to\_img}(A_\text{ts}^\top v)$
2. $J^\top (\cdot)$: apply one VJP of the denoiser
3. $G(\cdot)$: project back via $A_\text{ts} \cdot \texttt{img\_to\_ts}(\cdot)$

The VJP cost is identical to standard MMPS; only the CG bookkeeping operates in the smaller space. $\square$

**Implementation:**

```python
def cg_operator_obs(v_ts):      # v ∈ R^{N_obs × F}
    # G^T: masked TS → image
    v_img = ts_to_img(v_ts)     # delay-embed to image
    # J^T: image → image (single VJP)
    Jv_img = vjp_fn(v_img)
    # G: image → masked TS
    Jv_ts = img_to_ts(Jv_img)
    Jv_obs = mask_ts * Jv_ts    # extract observed positions
    return sigma_y_sq * v_ts + sigma_sq * Jv_obs
```

**Practical impact:**

| Property | Standard MMPS | DS-MMPS (Obs-Space CG) |
|---|---|---|
| CG dimension | $C \times H \times W$ (e.g., 1024) | $N_\text{obs} \times F$ (e.g., 36) |
| Observation operator | Approximate ($A_\text{img}$) | Exact ($G = A_\text{ts} \circ \texttt{img\_to\_ts}$) |
| CG convergence | Slow (high-dimensional, redundant) | Fast (~28x smaller system) |
| Built-in averaging | No (post-hoc projection needed) | Yes ($\texttt{img\_to\_ts}$ within CG partially pre-projects onto manifold) |
| VJP cost per CG iter | 1 VJP | 1 VJP (identical) |

For the Energy dataset (24 time points, 3 features, 50% missing): CG dimension shrinks from ~1024 to ~36 — a **28x reduction**.

### 4.2 Adaptive $\sigma_y$ — Corrects M2

**Proposition 2 (Bounded condition number).** Setting $\sigma_y = c \cdot \sigma_t$ (with constant $c > 0$) yields:

$$\kappa = \frac{c^2 + \lambda_\text{max}(G J^\top G^\top)}{c^2 + \lambda_\text{min}(G J^\top G^\top)}$$

which is **bounded independently of $\sigma_t$**.

**Proof.** Factor $\sigma_t^2$ from both terms in the CG operator:

$$\sigma_y^2 I + \sigma_t^2 \, G J^\top G^\top = \sigma_t^2 (c^2 I + G J^\top G^\top)$$

The condition number of $(c^2 I + G J^\top G^\top)$ depends only on the spectrum of $G J^\top G^\top$ and the constant $c$, not on $\sigma_t$. $\square$

**Single hyperparameter:** $c = 0.1$ (default). With a floor: $\sigma_y = \max(c \cdot \sigma_t, \sigma_\text{floor})$.

```python
sigma_y = max(c * sigma_t, sigma_floor)   # c = 0.1, floor = 0.01
```

With bounded conditioning, CG is stable at all noise levels, enabling 5 CG iterations (vs. the 1 iteration that standard MMPS typically uses before instability). More iterations → more accurate posterior samples → better E-step completions.

### 4.3 Manifold Projection — Corrects M3

**Proposition 1 (Projection optimality).** The composition $\Pi = \texttt{ts\_to\_img} \circ \texttt{img\_to\_ts}$ is the orthogonal projection onto $\text{Range}(\texttt{ts\_to\_img})$ and yields the minimum-norm correction:

$$\Pi(x) = \arg\min_{z \in \text{Range}(\texttt{ts\_to\_img})} \|x - z\|^2$$

**Proof sketch.** $\texttt{img\_to\_ts}$ averages overlapping delay-embedding windows (left-inverse up to normalization). The composition $\texttt{ts\_to\_img}(\texttt{img\_to\_ts}(\cdot))$ is self-adjoint and idempotent on $\mathbb{R}^{N_\text{img}}$, hence an orthogonal projection. $\square$

After posterior sampling, we apply:

```python
# 1. Project onto delay-embedding manifold
x_ts = img_to_ts(x_hat)
x_hat = ts_to_img(x_ts)

# 2. Hard-enforce observations
x_ts = img_to_ts(x_hat)
x_ts[:, observed_mask] = y_ts[:, observed_mask]
x_hat = ts_to_img(x_ts)
```

**Noise reduction.** Averaging $d$ overlapping windows reduces off-manifold noise by factor $d/m$ (approximately 60% for typical settings $m=8, d=3$). The off-manifold energy $E_\text{off}(k) = \text{mean}_i \|x_i - \Pi(x_i)\|^2$ serves as a convergence diagnostic.

**Interaction with Obs-Space CG.** When using observation-space CG (Section 4.1), the $\texttt{img\_to\_ts}$ operation inside the CG operator already partially pre-projects corrections onto the manifold. This means the explicit projection step applies a smaller correction than it would with standard image-space CG — the two components are complementary, not redundant.

### 4.4 Domain-Aware Initialization — Addresses Cold Start

EM must begin from a point that respects temporal structure. We initialize completions with **iterative STL decomposition** (trend + seasonal + residual) or **Kalman filtering** (state-space model that handles missing observations natively). These provide completions that respect temporal properties — trends, seasonality, autocorrelation — so the first E-step refines structurally valid completions rather than starting from Gaussian noise.

**Curriculum scheduling:** In early EM iterations, we temporarily reveal extra positions (filled with previous-iteration completions), annealed to zero by convergence:

```
Iteration 1:   Original mask + 30% extra positions revealed
Iteration 5:   Original mask + 15% extra positions revealed
Iteration 10:  Original mask only (curriculum = 0)
```

This gives the weak early-iteration model more signal during the fragile bootstrap phase.

**Distinction from commit-and-forget.** ImagenI2R uses TST as a one-shot imputation that the generator never revises. We use STL/Kalman as a *seed* for an iterative process — the first completion doesn't need to be perfect, it needs to be structurally consistent enough that the loop converges to the right basin.

### M-Step: Vanilla Score Matching (No Modifications)

The M-step loss is standard denoising score matching:

$$\mathcal{L} = \mathcal{L}_\text{SM} = \mathbb{E}_{x, \sigma, \epsilon}\big[\lambda(\sigma) \| d_\theta(x + \sigma \epsilon, \sigma) - x \|^2\big]$$

No auxiliary losses. No observation grounding. No manifold penalty. The M-step operates purely in image space on the completed training data from the E-step.

**Why this suffices.** If the E-step produces completions that (a) lie on the delay-embedding manifold, (b) respect observations at observed positions, and (c) are diverse posterior samples rather than MAP points — then the M-step training data is clean and representative. Standard score matching learns the correct distribution. No M-step corrections are needed because the M-step was never the problem.

---

## 5. Why M-Step Regularization is Unnecessary (Informative Negative Result)

We initially hypothesized that two M-step losses would improve performance:

- **$\mathcal{L}_\text{obs}$**: SNR-gated observation grounding — penalizes the denoiser for producing outputs inconsistent with TS-space observations at low noise levels.
- **$\mathcal{L}_\text{rep}$**: SNR-gated manifold penalty — penalizes the denoiser for producing off-manifold outputs at low noise levels.

Both were theoretically motivated (Propositions in earlier drafts) and seemed to address real concerns (observation forgetting, off-manifold denoiser outputs).

**The ablation disproves both.**

| Configuration | disc_mean | M-step |
|---|---|---|
| DS-MMPS (E-step only) | **0.045** | $\mathcal{L}_\text{SM}$ |
| DS-MMPS + $\mathcal{L}_\text{obs}(0.1) + \mathcal{L}_\text{rep}(0.05)$ | 0.053 | $\mathcal{L}_\text{SM} + \lambda_\text{obs} \mathcal{L}_\text{obs} + \lambda_\text{rep} \mathcal{L}_\text{rep}$ |

Adding M-step losses makes performance **18% worse** (0.045 → 0.053). This holds even at gentle weights.

**Interpretation.** The M-step losses create a bias-variance trade-off that is net negative:

- $\mathcal{L}_\text{obs}$ biases the denoiser toward reproducing observed values at low noise — but this conflicts with score matching's objective of modeling the full distribution. The SNR gate helps, but any nonzero weight introduces a directional bias that the optimizer resolves by distorting the learned score function away from observations.

- $\mathcal{L}_\text{rep}$ biases the denoiser toward on-manifold outputs — but the projection at the E-step already ensures M-step training data is on-manifold. Penalizing the denoiser directly constrains its intermediate representations, limiting its capacity to model the data distribution.

**The deeper insight:** "observation forgetting" (Failure Mode 5 in the earlier analysis) is not an independent failure mode — it is a symptom of poor E-step quality. When the E-step is correct (DS-MMPS), completions already respect observations because observation-space CG enforces this within the posterior sampling process. There is nothing for the M-step to "forget" because it never receives inconsistent training data.

This is a **positive result**: the method is simpler than initially proposed. DS-MMPS introduces exactly **one hyperparameter** beyond vanilla EM: the CG ratio $c$ in $\sigma_y = c \cdot \sigma_t$.

---

## 6. Summary: The DS-MMPS Method

### Comparison Table

| Method | Covariance | Operator | Output Constraint | Dual-Space? |
|--------|-----------|----------|-------------------|-------------|
| DPS (Chung 2023) | None (point est.) | Any | No | No — collapses posterior diversity |
| PiGDM (Song 2023) | Diagonal | Linear | No | No — misses delay-embedding correlations |
| TMPD (Boys 2023) | Row-sum diag. | Linear | No | No — same diagonal limitation |
| DiffPIR (Zhu 2023) | None (proximal) | Proximal | No | No — proximal of $G$ has no closed form |
| MMPS (Rozet 2024) | Full (CG) | Linear | No | No — assumes single-space $A$ |
| **DS-MMPS (ours)** | **Full (CG)** | **$G = A \circ \Phi^{-1}$** | **$\Pi$ projection** | **Yes** |

### Experimental Validation (Energy Dataset, 50% Missing)

| Configuration | disc_mean | What changes |
|---|---|---|
| Vanilla EM (standard MMPS) | 0.092 | Baseline — all misspecifications present |
| + STL init + adaptive $\sigma_y$ + manifold $\Pi$ | 0.049 | Partial DS-MMPS (fixes M2, M3, cold start) |
| + Observation-Space CG | **0.045** | **Full DS-MMPS** (fixes M1, M2, M3, cold start) |
| + $\mathcal{L}_\text{obs} + \mathcal{L}_\text{rep}$ (M-step losses) | 0.053 | M-step modifications hurt |

The gap from 0.092 to 0.045 (**51% improvement**) is entirely from E-step corrections. M-step modifications are not only unnecessary but actively harmful.

### Generality: The Dual-Space Problem Class

DS-MMPS generalizes beyond delay embedding. The same problem arises whenever:

1. The model operates in space $\mathcal{X}$ (image, latent, spectral)
2. Observations live in space $\mathcal{Y}$ (time series, pixel, sensor)
3. A known linear map $\Phi : \mathcal{X} \to \mathcal{Y}$ connects them (e.g., $\texttt{img\_to\_ts}$, decoder, inverse FFT)
4. The observation operator is $G = A_\mathcal{Y} \circ \Phi$, not $A_\mathcal{X}$

Examples where DS-MMPS directly applies:

- **Latent diffusion models** (PSLD, Rout et al. 2024): model in latent space, observations in pixel space, $\Phi$ = decoder.
- **Spectral methods for PDEs**: model in spectral space, observations in physical space, $\Phi$ = inverse Fourier transform.
- **Patch-based models**: model in patch space, observations in full-image space, $\Phi$ = patch reassembly.

In all cases, CG should run in $\text{Range}(G)$ (observation space), not in $\mathcal{X}$ (model space).

---

## References

- Rozet, F., Andry, G., Lanusse, F., & Louppe, G. (2024). Learning Diffusion Priors from Observations by Expectation Maximization. NeurIPS 2024. [DiEM — introduces MMPS]
- Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C. (2023). Diffusion Posterior Sampling for General Noisy Inverse Problems. ICLR 2023. [DPS]
- Song, J., Vahdat, A., Mardani, M., & Kautz, J. (2023). Pseudoinverse-Guided Diffusion Models for Inverse Problems. ICLR 2023. [PiGDM]
- Boys, B., Girolami, M., Sherlock, C., Sherrill-Mix, S., & Sherrill-Mix, S. (2023). Tweedie Moment Projected Diffusions for Inverse Problems. NeurIPS 2024. [TMPD]
- Zhu, Y., Zhang, K., Liang, J., Cao, J., Wen, B., Timofte, R., & Van Gool, L. (2023). Denoising Diffusion Models for Plug-and-Play Image Restoration. CVPR 2023 Workshop. [DiffPIR]
- Rout, L., Chen, Y., Kumar, A., Caramanis, C., Shakkottai, S., & Chu, W. (2024). Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion. CVPR 2024. [PSLD]
- Luo, Z., et al. (2025). Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior. arXiv:2501.18913. [DPS does MAP]
- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022. [EDM]
