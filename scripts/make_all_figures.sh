#!/usr/bin/env bash
# Regenerate all 8 paper figures (mock data where applicable).
#
# Usage:
#   bash scripts/make_all_figures.sh
#
# Outputs land in corrupted_lifted/figs/ as matching .pdf / .png pairs.
# Run from the repo root.

set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"

SCRIPTS=(
    scripts/plot_phase_diagram_theorem.py
    scripts/plot_operator_heatmap.py
    scripts/plot_off_manifold_energy.py
    scripts/plot_qualitative_gallery.py
    scripts/plot_posterior_diversity.py
    scripts/plot_sensitivity_heatmap.py
    scripts/plot_delay_embed_schematic.py
    scripts/plot_cg_conditioning.py
)

echo "Regenerating ${#SCRIPTS[@]} figures into corrupted_lifted/figs/..."
echo

for s in "${SCRIPTS[@]}"; do
    echo "=== $s ==="
    "$PY" "$s"
    echo
done

echo "Done. Figures written to corrupted_lifted/figs/:"
ls -1 corrupted_lifted/figs/fig_*.pdf 2>/dev/null || true
