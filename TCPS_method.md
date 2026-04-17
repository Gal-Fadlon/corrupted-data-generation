# Trajectory-Corrected Posterior Sampling (TCPS)

## 1. High-Level Idea

When using diffusion models to impute missing values, the standard approach (MMPS) starts from pure noise and tries to steer the reverse process toward solutions consistent with the observations. It does this by modifying the denoiser output at each step using a soft optimization (CG solver) that balances observation fidelity against the model's uncertainty.

**TCPS takes a fundamentally different approach.** Instead of softly guiding the denoiser, we exploit a simple but powerful insight:

> We know the clean values at observed positions. Since the forward diffusion process is deterministic given a noise vector `z`, we know **exactly** what the observed positions should look like at **every noise level** throughout the reverse process.

By fixing `z` once and reusing it, we can **anchor the observed positions to their exact forward-process trajectory** at every reverse step. The observed positions never drift --- they always sit exactly where the forward process would have placed them. We then use the denoiser's Jacobian to propagate corrections from these anchored observed positions to the missing positions.

## 2. Motivation

### Why not just use MMPS?

MMPS works well but has several limitations:

1. **Soft observation constraint**: MMPS introduces a parameter `sigma_y` that controls how tightly observations are enforced. This is a modeling assumption (observation noise) that doesn't match our setting --- our observations are clean (no noise), but setting `sigma_y = 0` makes the CG solver unstable. As a result, MMPS never achieves exact observation consistency --- there is always a gap proportional to `sigma_y^2`.

2. **Expensive CG solver**: At each reverse step, MMPS solves a linear system `(sigma_y^2 I + sigma^2 * mask * J^T * mask) * v = r` via conjugate gradient. This requires multiple VJP calls per step.

3. **No trajectory anchoring**: MMPS does not control what happens to observed positions in the noisy trajectory `x_t`. The observed values at intermediate noise levels are whatever the posterior denoiser produces --- they can drift away from where the forward process would have placed them. This means the denoiser sees an inconsistent trajectory at observed positions across steps.

### The TCPS insight

In the VE-SDE forward process, a clean sample `x_0` at noise level `sigma_t` becomes:

```
x_t = x_0 + sigma_t * z,    where z ~ N(0, I)
```

During the E-step (posterior sampling / imputation), if we **fix** the noise vector `z` once at the start and reuse it throughout the reverse process, then at every step `t` we know exactly what the observed positions should be at that noise level:

```
target_t[observed] = y_obs + sigma_t * z[observed]
```

This gives us two things:

1. **Trajectory anchoring**: At every reverse step, we hard-replace the observed positions with `y_obs + sigma_t * z`. This means the observed positions always follow the exact forward-process trajectory. The denoiser always sees a physically consistent input at observed positions.

2. **Error signal for correction**: The denoiser `D_theta(x_t)` estimates the clean signal `E[x_0 | x_t]`. At observed positions, the clean signal is `y_obs`. Any deviation `D_theta(x_t)[obs] - y_obs` is an error we can measure and use to correct the missing positions.

### How the correction works

The denoiser `D_theta(x_t)` learns the mapping from noisy input to clean output. Its Jacobian `J = dD_theta / dx_t` encodes the **learned correlations** between all positions. If `J[i,j]` is large, it means position `i` in the output is strongly influenced by position `j` in the input.

When we compute `J^T * error` (a VJP), we are asking: "given this error at observed positions in the output, which input positions contributed to it?" The answer tells us how to adjust the **input** to reduce the error in the **output**. This is the mathematically correct direction for an input-space correction.

TCPS uses this at **every reverse diffusion step**. At each step `t`:

1. The denoiser produces `x_hat = D_theta(x_t)` --- its best estimate of the fully clean signal `x_0` from noisy `x_t` (this is a single network forward pass, not the full multi-step process)
2. At observed positions, this estimate should equal `y_obs`. The difference is the error.
3. We compute `J^T(error)` to get an input-space correction, apply it to `x_t`, and re-denoise to get a better clean estimate.
4. The Heun solver uses this corrected estimate to take one step down the noise schedule, producing `x_{t-1}`.
5. We hard-replace `x_{t-1}[obs]` with the exact noise-matched target `y_obs + sigma_{t-1} * z` to keep the trajectory anchored.

This is the same Jacobian that MMPS uses, but applied differently:

- **MMPS**: Solves a regularized linear system to find the optimal correction, balancing `sigma_y` against the Jacobian-based covariance. Requires CG solver with multiple VJP calls. Observed positions are softly constrained.
- **TCPS**: Directly applies `J^T * error` as an input-space correction, then re-denoises. Single VJP call, no CG, no `sigma_y`. Observed positions are hard-anchored to the exact forward-process trajectory.

## 3. Algorithm

```
Input:  Unconditional denoiser D_theta, observations y_obs, mask A,
        noise schedule {sigma_t}, correction strength alpha

Setup:  Sample z ~ N(0, I) once and FIX it for the entire reverse process

Initialize:
    sigma_0 = sigma_max
    x_T[observed] = y_obs + sigma_0 * z[observed]    (noised observations)
    x_T[missing]  = sigma_0 * z[missing]              (pure noise)

    Note: at sigma_max (e.g. 80), the observation signal y_obs is tiny
    compared to the noise. But the fixed z ensures that as sigma decreases,
    the observed positions smoothly converge to y_obs along the correct
    forward-process path.

For each reverse step t -> t-1:

    1. DENOISE WITH GRADIENT:
       x_input = x_t.detach().requires_grad()
       x_hat = D_theta(x_input, sigma_t)
       (x_hat is the network's estimate of fully clean x_0, not one
        step of denoising. The network is trained to predict x_0 from
        any noise level in a single forward pass.)

    2. COMPUTE DENOISER ERROR at observed positions:
       error = mask * (x_hat - y_obs)
       (The denoiser estimates E[x_0 | x_t]. At observed positions,
        x_0 = y_obs, so any deviation is an error.)

    3. PROPAGATE TO INPUT SPACE via Jacobian (single VJP):
       input_correction = VJP(x_hat, x_input, grad_outputs=error)
       (J^T maps the output-space error to an input-space correction:
        "how should I change x_t to reduce the error in D_theta(x_t)?")

    4. CORRECT INPUT:
       x_t_corrected = x_t - alpha * input_correction

    5. RE-DENOISE with corrected input (no gradients needed):
       x_hat_corrected = D_theta(x_t_corrected, sigma_t)
       (A new, better estimate of x_0 from the corrected input.)

    6. HARD REPLACE denoised at observed positions:
       x_hat_corrected[observed] = y_obs

    7. HEUN STEP using corrected denoised estimate:
       d = (x_t - x_hat_corrected) / sigma_t
       x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * d
       [+ Heun 2nd order correction, also using corrected denoise]

    8. TRAJECTORY ANCHORING — hard replace observed positions:
       if sigma_{t-1} > 0:
           x_{t-1}[observed] = y_obs + sigma_{t-1} * z[observed]
       else:
           x_{t-1}[observed] = y_obs

       (This is the key step that uses the fixed z. It ensures the
        observed positions in the noisy trajectory always match the
        exact forward process. The denoiser at the next step will see
        a physically consistent input at observed positions.)

Return x_0
```

**Cost per step**: 1 forward pass with gradient + 1 VJP + 1 re-denoise + 1 Heun correction = ~5-6 network evaluations (2 corrected denoise calls for Euler + Heun).

## 4. Comparison with MMPS

| Property | MMPS | TCPS |
|----------|------|------|
| **What is corrected** | Denoiser output (score) | Denoiser input x_t |
| **Correction method** | CG solver (iterative, multiple VJPs) | Single VJP + re-denoise |
| **Observation enforcement** | Soft (controlled by sigma_y) | Hard (exact replacement every step) |
| **Trajectory at observed positions** | Uncontrolled (drifts based on CG solution) | Anchored to exact forward process via fixed z |
| **Fixed noise vector z** | No | Yes (same z throughout reverse process) |
| **Hyperparameters** | sigma_y, cg_iters | alpha (correction strength) |
| **VJP calls per step** | 2+ (CG iterations) | 1 (+ extra forward pass for re-denoise) |
| **Observation consistency at x_0** | Approximate (gap ~ sigma_y^2) | Exact (hard replace) |

## 5. Why This Matters for the EM Framework

In the DiEM framework, the E-step quality directly determines the M-step training data. Poor posterior samples lead to a degraded model, which leads to worse posterior samples in the next iteration --- a compounding error.

TCPS improves the E-step in three ways:

1. **Exact observation consistency**: Every posterior sample exactly matches the observations at observed positions. MMPS only approximately matches (gap proportional to sigma_y^2).

2. **Trajectory anchoring**: The observed positions follow the exact forward-process path throughout the reverse process. This gives the denoiser a consistent, physically correct reference to propagate from at every noise level. MMPS has no such guarantee --- observed positions in the trajectory can drift.

3. **Input-space correction with re-denoising**: By correcting x_t in input space and re-denoising, we get a clean, internally consistent denoised estimate. The Jacobian J^T naturally maps output errors to the correct input-space direction, and the re-denoise step ensures the correction is properly translated through the network.

Together, these produce cleaner E-step samples, which means the M-step trains on better data, which improves the model faster across EM iterations.