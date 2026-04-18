# Hidden-Distribution Experiment Matrix

This matrix targets the `energy` dataset at `50%` missingness and is ordered by
expected value first, then implementation cost. The first 13 runs are the
highest-priority batch.

## Common Flags

```bash
COMMON="--config ./configs/seq_len_24/energy.yaml --missing_rate 0.5 --wandb true"
```

## Priority 1: Ambient Delta Sweep

These reuse the strongest current stack with fixed Ambient corruption strength.

```bash
python run_diffem_mmps_kalman_freqgate_decomp_proj_ambient.py $COMMON --further_corrupt_delta 0.05 --tags hidden-dist energy50 ambient delta-005
python run_diffem_mmps_kalman_freqgate_decomp_proj_ambient.py $COMMON --further_corrupt_delta 0.10 --tags hidden-dist energy50 ambient delta-010
python run_diffem_mmps_kalman_freqgate_decomp_proj_ambient.py $COMMON --further_corrupt_delta 0.20 --tags hidden-dist energy50 ambient delta-020
python run_diffem_mmps_kalman_freqgate_decomp_proj_ambient.py $COMMON --further_corrupt_delta 0.30 --tags hidden-dist energy50 ambient delta-030
```

## Priority 2: Ambient Anneal Sweep

These test whether early EM should distrust reconstructions more than late EM.

```bash
python run_diffem_mmps_kalman_freqgate_decomp_proj_ambient_anneal.py $COMMON --ambient_delta_start 0.30 --ambient_delta_end 0.05 --tags hidden-dist energy50 ambient-anneal d30-005
python run_diffem_mmps_kalman_freqgate_decomp_proj_ambient_anneal.py $COMMON --ambient_delta_start 0.20 --ambient_delta_end 0.05 --tags hidden-dist energy50 ambient-anneal d20-005
python run_diffem_mmps_kalman_freqgate_decomp_proj_ambient_anneal.py $COMMON --ambient_delta_start 0.30 --ambient_delta_end 0.10 --tags hidden-dist energy50 ambient-anneal d30-010
```

## Priority 3: A-aware Trust Schedule

These are the highest-value new runs because they directly anneal trust from
observations toward the hidden distribution.

```bash
python run_diffem_mmps_kalman_freqgate_decomp_proj_aaware.py $COMMON --lambda_obs_start 1.0 --lambda_obs_end 0.5 --tags hidden-dist energy50 aaware l10-05
python run_diffem_mmps_kalman_freqgate_decomp_proj_aaware.py $COMMON --lambda_obs_start 1.0 --lambda_obs_end 0.3 --tags hidden-dist energy50 aaware l10-03
python run_diffem_mmps_kalman_freqgate_decomp_proj_aaware.py $COMMON --lambda_obs_start 0.8 --lambda_obs_end 0.3 --tags hidden-dist energy50 aaware l08-03
python run_diffem_mmps_kalman_freqgate_decomp_proj_aaware.py $COMMON --lambda_obs_start 1.0 --lambda_obs_end 0.5 --sigma_y_anneal --sigma_y_start 0.1 --sigma_y_end 0.01 --tags hidden-dist energy50 aaware l10-05 sigma-anneal
python run_diffem_mmps_kalman_freqgate_decomp_proj_aaware.py $COMMON --lambda_obs_start 1.0 --lambda_obs_end 0.3 --sigma_y_anneal --sigma_y_start 0.1 --sigma_y_end 0.01 --tags hidden-dist energy50 aaware l10-03 sigma-anneal
python run_diffem_mmps_kalman_freqgate_decomp_proj_aaware.py $COMMON --lambda_obs_start 0.8 --lambda_obs_end 0.3 --sigma_y_anneal --sigma_y_start 0.1 --sigma_y_end 0.01 --tags hidden-dist energy50 aaware l08-03 sigma-anneal
```

## Priority 4: Temporal-Block Corruption

These test whether longer-range temporal corruption teaches a cleaner hidden
distribution than i.i.d. masking.

```bash
python run_diffem_mmps_kalman_freqgate_decomp_proj_temporal_corrupt.py $COMMON --temporal_block_prob 0.05 --temporal_block_width 2 --tags hidden-dist energy50 temporal-block p005-w2
python run_diffem_mmps_kalman_freqgate_decomp_proj_temporal_corrupt.py $COMMON --temporal_block_prob 0.10 --temporal_block_width 4 --tags hidden-dist energy50 temporal-block p010-w4
python run_diffem_mmps_kalman_freqgate_decomp_proj_temporal_corrupt.py $COMMON --temporal_block_prob 0.15 --temporal_block_width 6 --tags hidden-dist energy50 temporal-block p015-w6
```

## Priority 5: Soft Imputed-Loss Schedule

These keep full reconstructed neighborhoods but only add a small late-stage loss
weight on imputed pixels.

```bash
python run_diffem_mmps_kalman_freqgate_decomp_proj_trust.py $COMMON --trust_mstep_mode soft_loss --imputed_loss_weight_start 0.0 --imputed_loss_weight_end 0.10 --tags hidden-dist energy50 trust soft-010
python run_diffem_mmps_kalman_freqgate_decomp_proj_trust.py $COMMON --trust_mstep_mode soft_loss --imputed_loss_weight_start 0.0 --imputed_loss_weight_end 0.20 --tags hidden-dist energy50 trust soft-020
python run_diffem_mmps_kalman_freqgate_decomp_proj_trust.py $COMMON --trust_mstep_mode soft_loss --imputed_loss_weight_start 0.0 --imputed_loss_weight_end 0.05 --tags hidden-dist energy50 trust soft-005
```

## Follow-up Seeds

After the initial 19 runs, promote the best candidate from the first 13 runs to
multi-seed validation:

```bash
# Replace <BEST_COMMAND> with the exact winning command from runs 1-13.
<BEST_COMMAND> --seed 1 --tags hidden-dist energy50 validate seed-1
<BEST_COMMAND> --seed 2 --tags hidden-dist energy50 validate seed-2
```

If the best run comes from the new trust-aware family instead, validate the top
trust run in the same way before expanding to MultiMask or bootstrap hybrids.
