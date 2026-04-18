# Hidden Dist Short Ablation (Energy 50%)

This is the short recovery batch used before any full-matrix relaunch.

- Runner set: ambient + ambient_anneal + aaware
- Fixed dataset: `energy`, `missing_rate=0.5`
- Short horizon: `em_iters=4`, `m_step_epochs=20`
- Tags include: `hidden_dist`, `short_ablation`, `no_golden`

## Submit

```bash
sbatch --array=0-7 sbatch_hidden_dist_short_ablation_energy_50_no_golden
```

## Array mapping

- `0`: ambient, parity (`--no-ambient_concat_further_mask`), `delta=0.05`, seed 0
- `1`: ambient, parity (`--no-ambient_concat_further_mask`), `delta=0.05`, seed 1
- `2`: ambient, mask-concat on, `delta=0.02`, seed 0
- `3`: ambient, mask-concat on, `delta=0.05`, seed 0
- `4`: ambient-anneal, mask-concat on, `delta=0.05 -> 0.02`, seed 0
- `5`: aaware, parity (`--no-ambient_concat_further_mask`), `lambda_obs=0.5 -> 0.2`, seed 0
- `6`: aaware, mask-concat on, `lambda_obs=0.5 -> 0.2`, seed 0
- `7`: aaware, parity (`--no-ambient_concat_further_mask`), `lambda_obs=0.5 -> 0.2`, seed 1
