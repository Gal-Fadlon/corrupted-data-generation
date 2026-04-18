# Paper Experiments Handoff — Full Context & Validation Guide

## Overview

This document describes **16 experiments** submitted to the BGU Slurm cluster for a NeurIPS paper on **iterative EM with diffusion priors for corrupted time series generation**. Each experiment fills a specific gap in the paper's ablation tables, sensitivity analyses, or cross-dataset validation.

**Goal for the reviewing LLM**: Given the results of these experiments (from WandB or cluster logs), fill in all TODO values in the paper, validate claims, and provide a full picture of what worked and what didn't.

---

## Key Concepts

- **Method**: We propose an EM loop where the E-step uses MMPS (Moment-Matching Posterior Sampling) to impute missing values, and the M-step trains a diffusion model on the completed data.
- **Dual-Space Problem**: Observations are in time-series space, but the generative model operates in image space (via delay embedding). Standard posterior samplers assume a single space.
- **Four E-step Corrections** (our contributions):
  1. **STL Initialization** (Section 6.4): Domain-aware warm start using iterative STL decomposition
  2. **Adaptive σ_y** (Section 6.2): Set σ_y = c·σ_t (c=0.1) to bound CG condition number
  3. **Manifold Projection** (Section 6.3): Project onto delay-embedding manifold after each reverse step
  4. **Observation-Space CG** (Section 6.1): Run CG in native TS space (dim 36) instead of image space (dim 1024)
- **Primary Metric**: `disc_mean` (discriminative score, lower is better). A classifier tries to distinguish real from generated — 0.5 = random, 0.0 = perfect.
- **EM Overfitting**: A phenomenon where disc_mean improves up to iteration 3-5 then degrades. Report **both** best and final values.

## Known Results (Before These Experiments)

| Configuration | Energy 50% disc_mean | Source |
|---|---|---|
| Clean-data oracle (0% missing) | 0.044 | Established baseline |
| Vanilla EM (standard MMPS, no corrections) | 0.092 | Row 1 in Table 3 |
| + Manifold projection (3 of 4 corrections) | 0.049 | Row 4 |
| + Obs-space CG (all 4 corrections = "Ours full") | **0.045** | Row 5 |
| + M-step aux losses (hurts) | 0.053 | Row 6 |

## Toy Experiment Results (Completed)

2D Gaussian mixture toy experiment confirming the paper's core claims:

| Method | Paradigm | disc (↓) | MSE |
|---|---|---|---|
| (A) Oracle | — | 0.590 | — |
| (B) Regression + masked loss (≈ ImagenI2R) | commit-and-forget (det.) | 0.578 | 1.40 |
| (B') Regression + full loss | commit-and-forget (det.) | 0.768 | 1.40 |
| (C) KNN + full loss (≈ KoVAE) | commit-and-forget (det.) | 0.661 | 1.40 |
| (E) Posterior 1-round (≈ CSDI) | commit-and-forget (stoch.) | 0.703 | 2.34 |
| (E') Posterior K=5 rounds | multi-impute (stoch.) | 0.647 | 2.16 |
| (D) Co-Evolving EM (≈ Ours) | co-evolving (stoch.+iter.) | **0.572** | 2.55 |

**Key finding**: Stochastic with frozen prior (E, E') is WORSE than careful deterministic (B). Only iterative EM (D) beats everything. This is a stronger result than expected — iteration, not stochasticity, is the key.

---

## Experiment Descriptions (Priority Order)

---

### TIER 1: Table 3 — Critical Ablation Gaps

These are the most important experiments. They fill gaps in the paper's central ablation table.

---

#### Experiment 1: `sbatch_paper_dps_em`
- **Table Row**: 1b
- **Script**: `run_diffem_dps.py`
- **What it tests**: DPS (Diffusion Posterior Sampling) as the E-step instead of MMPS. DPS uses a zeroth-order approximation: it computes ∇_x log p(y|x_t) ≈ -1/(2σ_y²) · ∇_x ||mask·D_θ(x_t) - y||². This is a MAP-like point estimate — no covariance information.
- **Config**: Energy 50%, 5 EM iters, `--dps_guidance_scale 1.0`, `--dps_sigma_y 0.01`, Phase 3
- **Expected outcome**: Substantially WORSE than vanilla MMPS-EM (0.092). DPS produces near-deterministic posteriors (mode-seeking), which is incompatible with EM's need for diverse completions. The lack of diversity means the M-step sees a collapsed distribution.
- **Paper claim to validate**: "MAP-like posteriors are incompatible with EM's requirements for diverse completions"
- **What to check**:
  - [ ] disc_mean at each EM iteration (expect: never drops below 0.1, possibly never below 0.15)
  - [ ] If disc_mean is near 0.5 at all iterations → DPS completions are so poor the model can't learn
  - [ ] If disc_mean improves but plateaus at ~0.15-0.20 → DPS partially works but diversity is insufficient
  - [ ] Compare to Row 1 (vanilla MMPS at 0.092) — DPS should be worse
- **WandB tags**: `paper`, `energy`, `50_missing_rate`, `dps_em`, `table3_row1b`

---

#### Experiment 2: `sbatch_paper_pigdm_em`
- **Table Row**: 1c
- **Script**: `run_diffem_pigdm.py`
- **What it tests**: PiGDM (Pseudo-Inverse Guided Diffusion Models) as the E-step. PiGDM uses a diagonal approximation of the Tweedie covariance: v_i = σ_t² · |J^T·1|_i. This captures SOME posterior structure (unlike DPS's point estimate) but misses cross-pixel correlations.
- **Config**: Energy 50%, 5 EM iters, `--pigdm_sigma_y 0.01`, Phase 3
- **Expected outcome**: Better than DPS-EM but worse than vanilla MMPS-EM (0.092). The diagonal covariance misses delay-embedding correlations, which are off-diagonal in image space.
- **Paper claim to validate**: "Diagonal covariance captures some posterior structure but misses the delay-embedding correlations"
- **What to check**:
  - [ ] disc_mean trajectory over 5 EM iters
  - [ ] Expected hierarchy: DPS-EM > PiGDM-EM > Vanilla MMPS (0.092) > Ours (0.045) (where > means worse)
  - [ ] If PiGDM-EM is surprisingly close to MMPS → the diagonal approximation is sufficient, which would weaken our obs-space CG argument
  - [ ] If PiGDM-EM is close to DPS → diagonal covariance adds little value
- **WandB tags**: `paper`, `energy`, `50_missing_rate`, `pigdm_em`, `table3_row1c`

---

#### Experiment 3: `sbatch_paper_ours_1iter`
- **Table Row**: 5b
- **Script**: `run_co_evolving_em.py`
- **What it tests**: Our FULL method (all 4 E-step corrections) but with only 1 EM iteration. This is the "stochastic commit-and-forget analog" — same quality E-step, but no iterative refinement. Directly isolates the value of iteration.
- **Config**: Energy 50%, `--em_iters 1`, all corrections enabled (adaptive σ_y, obs-space CG, consistency projection), Phase 3
- **Expected outcome**: Worse than full EM (0.045) but much better than vanilla EM (0.092). Should land around 0.06-0.08. The gap from this to 0.045 is PURELY the value of iteration.
- **Paper claim to validate**: Finding 4 from toy experiment: "Iterative refinement is the key ingredient"
- **What to check**:
  - [ ] Final disc_mean (only 1 EM iter, so no trajectory — just one evaluation)
  - [ ] Gap between this and Row 5 (0.045) quantifies iteration value
  - [ ] If gap is small (< 0.01) → iteration adds little, which HURTS the paper
  - [ ] If gap is large (> 0.02) → strong evidence for iterative EM, which HELPS the paper
  - [ ] Phase 3 disc_mean may be different from EM eval — report both
- **WandB tags**: `paper`, `energy`, `50_missing_rate`, `ours_1iter`, `table3_row5b`

---

#### Experiment 4: `sbatch_paper_csdi_baseline`
- **Table Row**: 0c
- **Script**: `run_csdi_impute_baseline.py`
- **What it tests**: CSDI (Conditional Score-based Diffusion for Imputation) trained on corrupted data, used to stochastically impute all missing values, then a fresh EDM trained on the completions. This is the strongest "stochastic commit-and-forget" baseline — CSDI produces diverse posterior samples but the imputer is frozen (never gets feedback from the generator).
- **Config**: Energy 50%, `--csdi_epochs 200`, `--csdi_nsample 1`, Phase 3 (fresh EDM on completions)
- **Expected outcome**: Unknown — this is the most uncertain experiment.
  - Toy experiment showed stochastic frozen < deterministic masked loss. But CSDI has a much better prior than the toy's mean-imputed GMM.
  - Could beat ImagenI2R (stochastic > deterministic when prior is good)
  - Should fall short of iterative EM (0.045) — frozen imputer ceiling
- **Paper claim to validate**: Section 3.4 "Why Stochastic Commit-and-Forget is Not Enough"
- **What to check**:
  - [ ] Final disc_mean after Phase 3
  - [ ] Compare to ImagenI2R baseline (Row 0b, TODO) — does stochastic help?
  - [ ] Compare to Ours 1-iter (Exp 3) — does a good E-step beat a dedicated imputer?
  - [ ] If CSDI beats ImagenI2R → stochastic imputation helps with good prior (section 3.4 prediction confirmed for real data)
  - [ ] If CSDI is worse than ImagenI2R → even stronger evidence that stochasticity needs iteration (aligns with toy finding 3)
  - [ ] If CSDI is close to Ours full (0.045) → problematic for the paper! Would mean frozen imputation is sufficient.
  - [ ] Check for NaN/crash — CSDI dependency `linear_attention_transformer` must be installed
- **WandB tags**: `paper`, `energy`, `50_missing_rate`, `csdi_baseline`, `table3_row0c`, `stochastic_commit_and_forget`
- **Potential failure mode**: Missing `linear_attention_transformer` pip package → immediate crash

---

#### Experiment 5: `sbatch_paper_ablation_row2_init`
- **Table Row**: 2
- **Script**: `run_co_evolving_em.py`
- **What it tests**: STL initialization ONLY — no adaptive σ_y, no manifold projection, no obs-space CG. Isolates the effect of domain-aware initialization vs random/Gaussian init.
- **Config**: Energy 50%, `--init_method stl`, `--no_adaptive_sigma_y`, `--no_consistency_projection`, `--no_obs_space_cg`, `--no_warm_start_cg`, 5 EM iters, Phase 3
- **Expected outcome**: Better than vanilla EM (0.092) but worse than Row 3. Expected range: 0.06-0.08. STL init gives a better starting point, but without CG stability corrections, later EM iterations may degrade.
- **Paper claim to validate**: "Domain-aware initialization alone reduces disc_mean from 0.092 to 0.0XX"
- **What to check**:
  - [ ] Best disc_mean and at which EM iteration
  - [ ] Gap from Row 1 (0.092) → quantifies init benefit
  - [ ] EM overfitting pattern (may be severe without adaptive σ_y to stabilize CG)
  - [ ] If result is 0.06-0.08 → validates the init contribution
  - [ ] If result is ≥ 0.09 → init alone doesn't help much (weakens the story)
  - [ ] If result is < 0.05 → init alone nearly solves the problem (weakens the other corrections' story)
- **WandB tags**: `paper`, `energy`, `50_missing_rate`, `ablation_row2`, `stl_init_only`

---

#### Experiment 6: `sbatch_paper_ablation_row3_sigma`
- **Table Row**: 3
- **Script**: `run_co_evolving_em.py`
- **What it tests**: STL init + adaptive σ_y — but NO manifold projection, NO obs-space CG. Isolates the cumulative effect of init + CG stability.
- **Config**: Energy 50%, `--init_method stl`, `--adaptive_sigma_y`, `--sigma_y_ratio 0.1`, `--sigma_y_floor 0.01`, `--no_consistency_projection`, `--no_obs_space_cg`, `--no_warm_start_cg`, 5 EM iters, Phase 3
- **Expected outcome**: Better than Row 2 (init only) but worse than Row 4 (0.049). Expected range: 0.05-0.07. Adaptive σ_y stabilizes CG, allowing more effective EM iterations.
- **Paper claim to validate**: "Adaptive σ_y further improves from 0.0XX to 0.0XX, stabilizing CG at high noise levels"
- **What to check**:
  - [ ] Best disc_mean and at which EM iteration
  - [ ] Gap from Row 2 → quantifies adaptive σ_y benefit
  - [ ] Reduced EM overfitting compared to Row 2 (adaptive σ_y should make later iterations more stable)
  - [ ] Verify the cumulative ordering: Row 1 (0.092) > Row 2 > Row 3 > Row 4 (0.049) > Row 5 (0.045)
- **WandB tags**: `paper`, `energy`, `50_missing_rate`, `ablation_row3`, `stl_init_adaptive_sigma`

---

### TIER 2: Cross-Dataset Validation

These confirm the method generalizes beyond Energy 50%.

---

#### Experiment 7: `sbatch_paper_energy_70`
- **Table**: Main results (Table 1), Energy 70% column
- **Script**: `run_co_evolving_em.py`
- **What it tests**: Our full method (all 4 corrections) on Energy at 70% missing rate. Higher missing rate is harder — tests whether the method degrades gracefully.
- **Config**: Energy 70%, all corrections, 5 EM iters, Phase 3
- **Expected outcome**: Worse than Energy 50% (0.045) but still good. Expected: 0.05-0.08. At 70% missing, only 30% of data is observed — a much harder task.
- **Paper claim to validate**: "The advantage is most pronounced at high missing rates, where masking-data trade-off is most severe"
- **What to check**:
  - [ ] disc_mean vs Energy 50% (0.045) — quantify degradation
  - [ ] If < 0.08 → strong result, validates robustness
  - [ ] Compare to ImagenI2R at 70% (TODO) — the gap should be LARGER than at 50% (our advantage grows with missing rate)
  - [ ] EM convergence: may need more iterations at 70%
- **WandB tags**: `paper`, `energy`, `70_missing_rate`, `cross_dataset`, `tier2`

---

#### Experiment 8: `sbatch_paper_weather_70`
- **Table**: Main results, Weather 70% column
- **Script**: `run_co_evolving_em.py`
- **What it tests**: Our full method on Weather dataset at 70% missing. Weather has 21 features (different from Energy's 28). Tests cross-dataset generalization.
- **Config**: Weather 70%, all corrections, 5 EM iters, Phase 3
- **Expected outcome**: Weather is typically harder than Energy. Expected disc_mean: 0.08-0.12 at 70% missing.
- **Paper claim to validate**: Method generalizes across datasets
- **What to check**:
  - [ ] disc_mean compared to Energy 70%
  - [ ] If weather results are much worse (> 0.15) → may indicate the method is dataset-specific
  - [ ] Check if the STL initialization works for weather patterns (different seasonality structure)
- **WandB tags**: `paper`, `weather`, `70_missing_rate`, `cross_dataset`, `tier2`

---

### TIER 3: Sensitivity Analysis

These validate robustness to hyperparameter choices and corruption patterns.

---

#### Experiments 9-12: σ_y Sensitivity (`sbatch_paper_sigma_c001` through `sbatch_paper_sigma_c050`)

- **Section**: 7.6 (Sensitivity to c in σ_y = c·σ_t)
- **Script**: `run_co_evolving_em.py`
- **What they test**: Varying the single hyperparameter c from 0.01 to 0.5 (default: 0.1). The paper claims the method has "exactly one hyperparameter beyond vanilla EM."

| Experiment | c value | sbatch file | Expected behavior |
|---|---|---|---|
| 9 | 0.01 | `sbatch_paper_sigma_c001` | Too tight — σ_y too small, CG ill-conditioned, may diverge |
| 10 | 0.05 | `sbatch_paper_sigma_c005` | Slightly tight, may work well |
| 11 | 0.20 | `sbatch_paper_sigma_c020` | Slightly loose, less accurate posteriors |
| 12 | 0.50 | `sbatch_paper_sigma_c050` | Too loose — posteriors too diffuse, slow convergence |

- **Known baseline**: c=0.1 gives disc_mean 0.045 (Row 5)
- **Paper claim to validate**: "Method is robust to c in [0.05, 0.2]" (this is the expected claim — currently TODO)
- **What to check**:
  - [ ] All four disc_mean values vs c=0.1 baseline (0.045)
  - [ ] If all c values give disc_mean < 0.06 → very robust (strong claim)
  - [ ] If only c=0.1 works → fragile (weakens the paper)
  - [ ] Shape of the curve: should be a broad U-shape with minimum near c=0.1
  - [ ] At c=0.01: check for NaN, divergence, or very high loss — CG may fail
  - [ ] At c=0.5: check for slow convergence (may need more EM iters)
- **WandB tags**: all have `paper`, `energy`, `50_missing_rate`, `sigma_sensitivity`, `tier3`

---

#### Experiments 13-15: Initialization Comparison (`sbatch_paper_init_random`, `sbatch_paper_init_linear`, `sbatch_paper_init_kalman`)

- **Section**: 7.6 (Initialization sensitivity)
- **Script**: `run_co_evolving_em.py`
- **What they test**: Different initialization methods, all with full corrections (A2 config). The default is STL.

| Experiment | Init method | sbatch file | Expected behavior |
|---|---|---|---|
| 13 | Random (Gaussian fill) | `sbatch_paper_init_random` | Worst — no temporal structure, slow convergence |
| 14 | Linear interpolation | `sbatch_paper_init_linear` | Better — preserves local trends, no seasonality |
| 15 | Kalman filter | `sbatch_paper_init_kalman` | Good — state-space model handles gaps natively |

- **Known baseline**: STL init gives disc_mean 0.045 (Row 5)
- **Paper claim to validate**: "Initialization is a seed for iteration — it doesn't need to be perfect, just structurally consistent enough for the loop to converge" (Section 6.4)
- **What to check**:
  - [ ] All three disc_mean values vs STL baseline (0.045)
  - [ ] Expected ordering: Random > Linear > Kalman ≈ STL (where > means worse)
  - [ ] If random init still achieves < 0.06 → VERY strong result: method is robust to init (validates "seed" claim)
  - [ ] If random init gives > 0.1 → init matters a lot (validates STL contribution but weakens "robustness" claim)
  - [ ] If Kalman matches STL → Kalman is an equally good alternative
  - [ ] Check Kalman for timeout/crash — Kalman fitting can be slow with timeouts at 5s per series
  - [ ] Compare convergence SPEED: does better init mean fewer EM iterations needed?
- **WandB tags**: all have `paper`, `energy`, `50_missing_rate`, `init_comparison`, `tier3`

---

#### Experiment 16: `sbatch_paper_block_missing`

- **Section**: 7.6 (Robustness to missing pattern)
- **Script**: `run_co_evolving_em.py`
- **What it tests**: Block missing (contiguous gaps of length 12) instead of MCAR (random point-wise). All other experiments use MCAR. Block missing is harder because there's no nearby observed data to interpolate from.
- **Config**: Energy 50%, `--missing_type block`, `--block_length 12`, all corrections, 5 EM iters, Phase 3
- **Expected outcome**: Worse than MCAR at same rate (0.045). Expected: 0.06-0.10. Block gaps are harder for CG-based imputation since the local information is absent.
- **Paper claim to validate**: Method handles different missing patterns
- **What to check**:
  - [ ] disc_mean vs MCAR baseline (0.045)
  - [ ] If < 0.08 → method is robust to missing pattern (good for paper)
  - [ ] If > 0.12 → block missing is a significant challenge (honest limitation)
  - [ ] Check STL initialization: block gaps may confuse STL decomposition
  - [ ] EM convergence: may be slower with block missing
- **WandB tags**: `paper`, `energy`, `50_missing_rate`, `block_missing`, `tier3`

---

## Validation Checklist (For ALL Experiments)

### For every experiment, report:

1. **disc_mean trajectory**: value at each EM iteration (custom_step 0, 1, 2, ...)
2. **Best disc_mean**: minimum value and at which step
3. **Final disc_mean**: last evaluated value (may be worse due to EM overfitting)
4. **Phase 3 disc_mean**: fresh model trained on final completions (may differ from EM eval)
5. **Runtime**: total wall-clock time
6. **GPU**: which hardware was allocated
7. **Crash/failure**: any errors, especially import errors or NaN issues

### Cross-experiment validation:

- [ ] **Ablation ordering** (Table 3): Confirm Row 1 > Row 2 > Row 3 > Row 4 > Row 5 (strictly monotone improvement)
- [ ] **Posterior sampler hierarchy** (Table 3): Confirm DPS-EM >> PiGDM-EM >> Vanilla MMPS >> Ours
- [ ] **Iteration value**: Gap between Row 5b (1 iter) and Row 5 (5 iters) quantifies iterative refinement
- [ ] **CSDI positioning**: Where does Row 0c fall relative to Row 0b (ImagenI2R) and Row 5?
- [ ] **σ_y robustness**: Broad U-shape or flat sensitivity curve
- [ ] **Init robustness**: All inits converge to similar quality (validates "seed" claim)
- [ ] **70% missing**: Graceful degradation from 50% results

### Paper claim validation matrix:

| Claim | Experiments that test it | What confirms it | What refutes it |
|---|---|---|---|
| Iterative EM > commit-and-forget | Exp 3, 4 | Row 5b >> Row 0c, 0b | Row 5b ≈ Row 5 (iteration doesn't help) |
| Obs-space CG is critical | Row 4 vs 5 (existing) | 8% improvement (confirmed) | — |
| DPS/PiGDM posteriors are insufficient | Exp 1, 2 | Both worse than vanilla MMPS (0.092) | PiGDM matches MMPS |
| Single hyperparameter c=0.1 is robust | Exp 9-12 | All c in [0.05,0.2] give < 0.06 | Only c=0.1 works |
| Init is just a "seed" | Exp 13-15 | Random init still < 0.06 | Random init > 0.10 |
| Method handles block missing | Exp 16 | disc_mean < 0.08 | disc_mean > 0.12 |
| Method generalizes to other datasets | Exp 7, 8 | Weather 70% < 0.12 | Weather 70% > 0.15 |
| Stochastic frozen imputer is insufficient | Exp 4 | CSDI >> Ours (0.045) | CSDI ≈ Ours |

---

## WandB Query Guide

All experiments are in WandB project `azencot-group/ts_corrupted`.

**To find these runs**, filter by tags containing `"paper"`:
```
filters: {"tags": {"$in": ["paper"]}}
```

**To find a specific experiment**, use the table row tag:
```
filters: {"tags": {"$in": ["table3_row1b"]}}
```

**To check for crashes**, filter by state:
```
filters: {"state": "crashed", "tags": {"$in": ["paper"]}}
```

**Metric paths**:
- `test/disc_mean` — primary metric (lower is better)
- `test/pred_score_mean` — predictive score
- `test/fid_score_mean` — FID score
- `test/correlation_score_mean` — correlation score
- `em/m_step_loss` — M-step training loss
- `em/iteration` — current EM iteration
- `em/e_step_samples` — number of imputed samples per E-step

---

## Slurm Job IDs (for log retrieval)

| Exp # | Job ID | Partition | sbatch file |
|---|---|---|---|
| 1 | 17016176 | rtx6000 | sbatch_paper_dps_em |
| 2 | 17016177 | rtx6000 | sbatch_paper_pigdm_em |
| 3 | 17016178 | rtx6000 | sbatch_paper_ours_1iter |
| 4 | 17016179 | rtx6000 | sbatch_paper_csdi_baseline |
| 5 | 17016180 | rtx6000 | sbatch_paper_ablation_row2_init |
| 6 | 17016181 | rtx6000 | sbatch_paper_ablation_row3_sigma |
| 7 | 17016190 | gpu | sbatch_paper_energy_70 |
| 8 | 17016191 | gpu | sbatch_paper_weather_70 |
| 9 | 17016192 | gpu | sbatch_paper_sigma_c001 |
| 10 | 17016193 | gpu | sbatch_paper_sigma_c005 |
| 11 | 17016194 | gpu | sbatch_paper_sigma_c020 |
| 12 | 17016195 | gpu | sbatch_paper_sigma_c050 |
| 13 | 17016196 | gpu | sbatch_paper_init_random |
| 14 | 17016197 | gpu | sbatch_paper_init_linear |
| 15 | 17016198 | gpu | sbatch_paper_init_kalman |
| 16 | 17016199 | gpu | sbatch_paper_block_missing |

Log files on cluster: `~/diffem/train-<JOBID>.out`

Note: During submission, Tier 2-3 jobs that originally targeted `rtx6000`/`azencot` were dynamically
switched to `main`/`normal` partition/QoS due to `MaxGRESPerAccount` limits, landing on the `gpu` partition.
Job IDs above reflect the final submitted state.

---

## Appendix A: Exact CLI Arguments per Experiment

Below are the exact Python command lines from each sbatch file. Use these to verify WandB config fields match.

### Exp 1 — DPS-EM (Row 1b)
```bash
python run_diffem_dps.py \
    --config ./configs/seq_len_24/energy.yaml \
    --missing_rate 0.5 \
    --wandb true \
    --tags "paper" "energy" "50_missing_rate" "dps_em" "table3_row1b" \
    --dps_guidance_scale 1.0 \
    --dps_sigma_y 0.01 \
    --em_iters 5 \
    --eval_all_metrics
```

### Exp 2 — PiGDM-EM (Row 1c)
```bash
python run_diffem_pigdm.py \
    --config ./configs/seq_len_24/energy.yaml \
    --missing_rate 0.5 \
    --wandb true \
    --tags "paper" "energy" "50_missing_rate" "pigdm_em" "table3_row1c" \
    --pigdm_sigma_y 0.01 \
    --em_iters 5 \
    --eval_all_metrics
```

### Exp 3 — Ours 1-iter (Row 5b)
```bash
python run_co_evolving_em.py \
    --config ./configs/seq_len_24/energy.yaml \
    --missing_rate 0.5 \
    --wandb true \
    --tags "paper" "energy" "50_missing_rate" "ours_1iter" "table3_row5b" \
    --adaptive_sigma_y \
    --sigma_y_ratio 0.1 \
    --sigma_y_floor 0.01 \
    --mmps_cg_iters 5 \
    --consistency_projection \
    --obs_space_cg \
    --no_warm_start_cg \
    --lambda_obs 0 \
    --lambda_rep 0 \
    --em_iters 1 \
    --phase3 \
    --eval_all_metrics
```

### Exp 4 — CSDI Baseline (Row 0c)
```bash
pip install -q linear_attention_transformer 2>/dev/null || true
python run_csdi_impute_baseline.py \
    --config ./configs/seq_len_24/energy.yaml \
    --missing_rate 0.5 \
    --wandb true \
    --tags "paper" "energy" "50_missing_rate" "csdi_baseline" "table3_row0c" "stochastic_commit_and_forget" \
    --csdi_epochs 200 \
    --csdi_nsample 1 \
    --eval_all_metrics
```

### Exp 5 — Ablation Row 2 (STL init only)
```bash
python run_co_evolving_em.py \
    --config ./configs/seq_len_24/energy.yaml \
    --missing_rate 0.5 \
    --wandb true \
    --tags "paper" "energy" "50_missing_rate" "ablation_row2" "stl_init_only" \
    --no_adaptive_sigma_y \
    --sigma_y_ratio 0.1 \
    --mmps_cg_iters 5 \
    --no_consistency_projection \
    --no_obs_space_cg \
    --no_warm_start_cg \
    --lambda_obs 0 \
    --lambda_rep 0 \
    --em_iters 5 \
    --phase3 \
    --eval_all_metrics \
    --init_method stl
```

### Exp 6 — Ablation Row 3 (STL init + adaptive σ_y)
```bash
python run_co_evolving_em.py \
    --config ./configs/seq_len_24/energy.yaml \
    --missing_rate 0.5 \
    --wandb true \
    --tags "paper" "energy" "50_missing_rate" "ablation_row3" "stl_init_adaptive_sigma" \
    --adaptive_sigma_y \
    --sigma_y_ratio 0.1 \
    --sigma_y_floor 0.01 \
    --mmps_cg_iters 5 \
    --no_consistency_projection \
    --no_obs_space_cg \
    --no_warm_start_cg \
    --lambda_obs 0 \
    --lambda_rep 0 \
    --em_iters 5 \
    --phase3 \
    --eval_all_metrics \
    --init_method stl
```

### Exp 7 — Energy 70%
```bash
python run_co_evolving_em.py \
    --config ./configs/seq_len_24/energy.yaml \
    --missing_rate 0.7 \
    --wandb true \
    --tags "paper" "energy" "70_missing_rate" "cross_dataset" "tier2" \
    --adaptive_sigma_y --sigma_y_ratio 0.1 --sigma_y_floor 0.01 \
    --mmps_cg_iters 5 --consistency_projection --obs_space_cg \
    --no_warm_start_cg --lambda_obs 0 --lambda_rep 0 \
    --em_iters 5 --phase3 --eval_all_metrics
```

### Exp 8 — Weather 70%
```bash
python run_co_evolving_em.py \
    --config ./configs/seq_len_24/weather.yaml \
    --missing_rate 0.7 \
    --wandb true \
    --tags "paper" "weather" "70_missing_rate" "cross_dataset" "tier2" \
    --adaptive_sigma_y --sigma_y_ratio 0.1 --sigma_y_floor 0.01 \
    --mmps_cg_iters 5 --consistency_projection --obs_space_cg \
    --no_warm_start_cg --lambda_obs 0 --lambda_rep 0 \
    --em_iters 5 --phase3 --eval_all_metrics
```

### Exp 9-12 — σ_y Sensitivity (c = 0.01, 0.05, 0.20, 0.50)

All identical to Exp 7 (A2 full config) on Energy 50% except `--sigma_y_ratio`:

| Exp | c value | `--sigma_y_ratio` | Tag |
|---|---|---|---|
| 9 | 0.01 | 0.01 | `c_001` |
| 10 | 0.05 | 0.05 | `c_005` |
| 11 | 0.20 | 0.2 | `c_020` |
| 12 | 0.50 | 0.5 | `c_050` |

Reference: c=0.1 is the default (Row 5, disc_mean 0.045).

### Exp 13-15 — Init Comparison (random, linear, kalman)

All identical to the A2 full config on Energy 50% except `--init_method`:

| Exp | Init method | `--init_method` | Tag |
|---|---|---|---|
| 13 | Random (Gaussian fill) | random | `random_init` |
| 14 | Linear interpolation | linear | `linear_init` |
| 15 | Kalman filter | kalman | `kalman_init` |

Reference: STL init is the default (Row 5, disc_mean 0.045).

### Exp 16 — Block Missing
```bash
python run_co_evolving_em.py \
    --config ./configs/seq_len_24/energy.yaml \
    --missing_rate 0.5 \
    --missing_type block \
    --block_length 12 \
    --wandb true \
    --tags "paper" "energy" "50_missing_rate" "block_missing" "tier3" \
    --adaptive_sigma_y --sigma_y_ratio 0.1 --sigma_y_floor 0.01 \
    --mmps_cg_iters 5 --consistency_projection --obs_space_cg \
    --no_warm_start_cg --lambda_obs 0 --lambda_rep 0 \
    --em_iters 5 --phase3 --eval_all_metrics
```

Reference: MCAR at same rate gives disc_mean 0.045 (Row 5).

---

## Appendix B: A2 Configuration (Our Full Method)

The "A2 config" is the final best configuration, used as the base for all `run_co_evolving_em.py` experiments:

| Parameter | Value | Purpose |
|---|---|---|
| `--adaptive_sigma_y` | enabled | σ_y = c·σ_t instead of fixed |
| `--sigma_y_ratio` | 0.1 | c value |
| `--sigma_y_floor` | 0.01 | minimum σ_y |
| `--mmps_cg_iters` | 5 | CG iterations per reverse step |
| `--consistency_projection` | enabled | Manifold projection after each step |
| `--obs_space_cg` | enabled | CG in observation space |
| `--no_warm_start_cg` | disabled | No warm-starting CG |
| `--lambda_obs` | 0 | No obs auxiliary loss |
| `--lambda_rep` | 0 | No rep auxiliary loss |
| `--em_iters` | 5 | 5 EM iterations (default) |
| `--phase3` | enabled | Fresh model on final completions |
| `--init_method` | stl | STL decomposition (default) |

---

## Convergence Reference Data (Existing)

These trajectories are already known (from prior runs) and serve as reference:

**Energy 50% — disc_mean over EM iterations:**

| EM iter | Vanilla MMPS (Row 1) | Baseline no obs-CG (Row 4) | Obs-space CG / Ours (Row 5) |
|:---:|:---:|:---:|:---:|
| 0 | 0.472 | 0.387 | 0.290 |
| 1 | 0.424 | 0.185 | 0.244 |
| 2 | 0.278 | 0.068 | 0.054 |
| 3 | 0.092 | 0.049 | **0.045** |
| 4 | 0.099 | 0.049 | 0.051 |
| 5 | 0.112 | 0.057 | 0.061 |
| 7 | 0.107 | 0.055 | 0.060 |
| 9 | 0.102 | 0.456 (diverged) | 0.060 |

Key observations:
- Peak at iteration 3 for all methods (EM overfitting after)
- Obs-space CG has mildest degradation (1.3x from peak)
- Baseline diverges at iteration 9 (catastrophic EM overfitting)

---

## Expected Timeline

Each experiment runs ~12-24 hours on RTX 6000. With 16 experiments and limited GPU slots, expect:
- Tier 1 (rtx6000, 6 jobs): 2-3 days (jobs queue behind account GPU limit)
- Tier 2-3 (gpu partition, 10 jobs): 3-5 days (general queue)

Check `squeue --me` and WandB dashboard periodically.

---

## What You (Reviewing LLM) Should Produce

Given the WandB results from these 16 experiments, produce:

### 1. Filled Tables

**Table 3 (Ablation)** — fill in ALL TODO values:
- Row 0c: CSDI disc_mean (from Exp 4)
- Row 1b: DPS-EM disc_mean (from Exp 1)
- Row 1c: PiGDM-EM disc_mean (from Exp 2)
- Row 2: STL init only disc_mean (from Exp 5)
- Row 3: STL + adaptive σ_y disc_mean (from Exp 6)
- Row 5b: Ours 1-iter disc_mean (from Exp 3)

**Table 1 (Main Results)** — fill Energy 70% column (from Exp 7), note Weather 70% (from Exp 8)

**Section 7.6 (Sensitivity)** — fill the σ_y sensitivity table and init comparison table from Exps 9-16

### 2. Claim Verification Report

For each claim in the paper, state CONFIRMED / PARTIALLY CONFIRMED / REFUTED with evidence:

1. "Iterative EM > commit-and-forget" (compare Row 5 vs 0b, 0c, 5b)
2. "MAP-like posteriors are incompatible with EM" (Row 1b value)
3. "Diagonal covariance misses delay-embedding correlations" (Row 1c value)
4. "Domain-aware init addresses cold start" (Row 1 → 2 gap)
5. "Adaptive σ_y stabilizes CG" (Row 2 → 3 gap)
6. "Single hyperparameter is robust" (σ_y sensitivity curve)
7. "Init is just a seed" (init comparison spread)
8. "Method handles block missing" (Exp 16 vs MCAR)
9. "Method generalizes across datasets" (Weather 70% result)

### 3. Anomaly Flags

Report any unexpected results:
- Orderings that violate expected hierarchy
- Experiments that crashed
- EM overfitting patterns (if best step != step 3)
- disc_mean values that are suspiciously good or bad
- Cross-dataset inconsistencies

### 4. Narrative Recommendations

Based on results, suggest:
- Which claims need softening or strengthening
- Where error bars / confidence intervals would help
- Any additional experiments needed to close gaps
