#!/bin/bash
# Generate sbatch scripts for the paper's continuous-time resampling matrix.
# 21 runs total: Energy seq_len in {24,96} x rho in {30,50,70} x {vanilla,noobscg,full}
#                + Weather seq_len=24 x rho=50 x {vanilla,noobscg,full}.
#
# Already exists (do not overwrite):
#   sbatch_paper_energy_continuous_rho{30,50,70}         (seq24, full)
#   sbatch_paper_energy_continuous_rho50_noobscg         (seq24, noobscg)
#
# This script only writes the missing 17 scripts.

set -euo pipefail

write_script() {
    local outfile="$1"
    local jobname="$2"
    local variant="$3"    # vanilla | noobscg | full
    local seq_len="$4"    # 24 | 96
    local dataset="$5"    # energy | weather
    local rho_pct="$6"    # 30 | 50 | 70
    local rho_dec=$(awk "BEGIN {printf \"%.1f\", ${rho_pct}/100.0}")

    local config="./configs/seq_len_${seq_len}/${dataset}.yaml"
    local tags=("paper" "${dataset}" "seq${seq_len}" "continuous_rho${rho_pct}" "variant_${variant}" "continuous_matrix")

    # Variant-specific flag blocks
    local variant_flags=""
    case "${variant}" in
        vanilla)
            # Vanilla MMPS-EM = uses A_img (no obs-CG), no manifold projection, no init
            variant_flags="    --no_obs_space_cg \\
    --no_consistency_projection \\
    --init_method stl \\"
            ;;
        noobscg)
            # Drop only obs-CG; keep manifold projection and init (isolates C2 correction)
            variant_flags="    --no_obs_space_cg \\
    --consistency_projection \\
    --init_method kalman \\"
            ;;
        full)
            # Full corrections: obs-CG + manifold projection + Kalman init
            variant_flags="    --obs_space_cg \\
    --consistency_projection \\
    --init_method kalman \\"
            ;;
        *)
            echo "Unknown variant: ${variant}" >&2
            exit 1
            ;;
    esac

    cat > "${outfile}" <<EOF
#!/bin/bash
#SBATCH --partition main
#SBATCH --qos normal
#SBATCH --time 7-00:00:00
#SBATCH --job-name ${jobname}
#SBATCH --output train-%J.out
#SBATCH --gpus=1

echo \`date\`
echo -e "\nSLURM_JOBID:\t\t" \$SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" \$SLURM_JOB_NODELIST "\n\n"

cd "\${SLURM_SUBMIT_DIR}" || { echo "cd SLURM_SUBMIT_DIR failed"; exit 1; }
export PYTHONUNBUFFERED=1

### Paper continuous-time resampling matrix:
###   dataset=${dataset}, seq_len=${seq_len}, rho=${rho_dec}, variant=${variant}
module load anaconda
source activate idan2_env
python run_co_evolving_em_continuous.py \\
    --config ${config} \\
    --corruption_type continuous \\
    --n_obs_ratio ${rho_dec} \\
    --t_distribution uniform \\
    --interp_kernel linear \\
    --missing_rate 0.0 \\
    --wandb true \\
    --tags "${tags[0]}" "${tags[1]}" "${tags[2]}" "${tags[3]}" "${tags[4]}" "${tags[5]}" \\
    --adaptive_sigma_y \\
    --sigma_y_ratio 0.1 \\
    --sigma_y_floor 0.01 \\
    --mmps_cg_iters 5 \\
${variant_flags}
    --no_warm_start_cg \\
    --lambda_obs 0.5 \\
    --lambda_rep 0 \\
    --em_iters 12 \\
    --em_patience 2 \\
    --phase3 \\
    --eval_all_metrics
EOF
    chmod +x "${outfile}"
    echo "Wrote ${outfile}"
}

# Energy seq_len=24: rho={30,50,70} x {vanilla, noobscg (only 30/70 missing), full exists}
write_script "sbatch_paper_energy_continuous_rho30_vanilla" "paper_e_cont_s24_r30_v" "vanilla" 24 "energy" 30
write_script "sbatch_paper_energy_continuous_rho50_vanilla" "paper_e_cont_s24_r50_v" "vanilla" 24 "energy" 50
write_script "sbatch_paper_energy_continuous_rho70_vanilla" "paper_e_cont_s24_r70_v" "vanilla" 24 "energy" 70
write_script "sbatch_paper_energy_continuous_rho30_noobscg" "paper_e_cont_s24_r30_n" "noobscg" 24 "energy" 30
write_script "sbatch_paper_energy_continuous_rho70_noobscg" "paper_e_cont_s24_r70_n" "noobscg" 24 "energy" 70

# Energy seq_len=96: all 9 cells (3 rhos x 3 variants)
write_script "sbatch_paper_energy_continuous_seq96_rho30_vanilla" "paper_e_cont_s96_r30_v" "vanilla" 96 "energy" 30
write_script "sbatch_paper_energy_continuous_seq96_rho50_vanilla" "paper_e_cont_s96_r50_v" "vanilla" 96 "energy" 50
write_script "sbatch_paper_energy_continuous_seq96_rho70_vanilla" "paper_e_cont_s96_r70_v" "vanilla" 96 "energy" 70
write_script "sbatch_paper_energy_continuous_seq96_rho30_noobscg" "paper_e_cont_s96_r30_n" "noobscg" 96 "energy" 30
write_script "sbatch_paper_energy_continuous_seq96_rho50_noobscg" "paper_e_cont_s96_r50_n" "noobscg" 96 "energy" 50
write_script "sbatch_paper_energy_continuous_seq96_rho70_noobscg" "paper_e_cont_s96_r70_n" "noobscg" 96 "energy" 70
write_script "sbatch_paper_energy_continuous_seq96_rho30_full" "paper_e_cont_s96_r30_f" "full" 96 "energy" 30
write_script "sbatch_paper_energy_continuous_seq96_rho50_full" "paper_e_cont_s96_r50_f" "full" 96 "energy" 50
write_script "sbatch_paper_energy_continuous_seq96_rho70_full" "paper_e_cont_s96_r70_f" "full" 96 "energy" 70

# Weather seq_len=24 x rho=50 x all 3 variants
write_script "sbatch_paper_weather_continuous_rho50_vanilla" "paper_w_cont_s24_r50_v" "vanilla" 24 "weather" 50
write_script "sbatch_paper_weather_continuous_rho50_noobscg" "paper_w_cont_s24_r50_n" "noobscg" 24 "weather" 50
write_script "sbatch_paper_weather_continuous_rho50_full" "paper_w_cont_s24_r50_f" "full" 24 "weather" 50

echo ""
echo "Done. 17 new sbatch scripts written."
echo "Together with the 4 existing scripts this covers the full 21-run matrix."
echo ""
echo "Suggested remote submission (after rsync to ~/diffem):"
echo "  for f in sbatch_paper_{energy,weather}_continuous*; do sbatch \"\$f\"; done"
