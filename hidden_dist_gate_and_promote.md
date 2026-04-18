# Hidden Dist Gate & Promote

## 1) Gate short-ablation runs

Run after short batch starts logging eval points:

```bash
python gate_hidden_dist_runs.py --tag short_ablation --max-runs 100 --json-out hidden_dist_short_ablation_gate_results.json
```

Decision policy (implemented in script):

- `STOP` if best `test/disc_mean` at `custom_step<=3` is `>= 0.20`
- `PROMOTE` if best `test/disc_mean` at `custom_step<=6` is `< 0.10`
- `HOLD` otherwise

## 2) Promote only winning configurations

For promoted candidates, relaunch the matching config with long horizon:

- `--em_iters 20`
- `--m_step_epochs 50`
- multi-seed: `--seed 0`, `--seed 1`, `--seed 2`

Example (promoted ambient parity config):

```bash
python run_diffem_mmps_kalman_freqgate_decomp_proj_ambient.py \
  --config ./configs/seq_len_24/energy.yaml \
  --missing_rate 0.5 \
  --wandb true \
  --em_iters 20 \
  --m_step_epochs 50 \
  --seed 0 \
  --further_corrupt_delta 0.05 \
  --no-ambient_concat_further_mask \
  --tags "hidden_dist" "energy" "50_missing_rate" "promotion_long" "ambient" "parity" "seed0" "no_golden"
```

Repeat for seeds `1` and `2`, and for each promoted runner variant.
