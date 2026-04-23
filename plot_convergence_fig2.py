"""
Figure 2: Convergence + EM Overfitting (Section 7.4)

Plots disc_mean vs EM iteration for multiple methods:
  - Naive MMPS-EM
  - Baseline (all corrections except obs-space CG)
  - Obs-space CG (ours — full corrections)

Includes:
  - Dashed horizontal line for clean-data oracle (0.044)
  - Shaded degradation region after peak
  - Annotations for peak and degradation ratio

Data source: Table in Section 7.4 of paper_v3_unified.md
             (can be replaced with WandB API queries once runs complete)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# Convergence data from Section 7.4
em_iters = np.array([0, 1, 2, 3, 4, 5, 7, 9])

vanilla_mmps = np.array([0.472, 0.424, 0.278, 0.092, 0.099, 0.112, 0.107, 0.102])
baseline_no_obs_cg = np.array([0.387, 0.185, 0.068, 0.049, 0.049, 0.057, 0.055, 0.456])
obs_space_cg = np.array([0.290, 0.244, 0.054, 0.045, 0.051, 0.061, 0.060, 0.060])

oracle = 0.044

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [2, 1]})
fig.subplots_adjust(wspace=0.3)

# --- Left panel: full convergence trajectories ---
ax1.plot(em_iters, vanilla_mmps, 'o-', color='#d62728', lw=2, markersize=7,
         label='Naive MMPS-EM', zorder=3)
ax1.plot(em_iters, baseline_no_obs_cg, 's-', color='#ff7f0e', lw=2, markersize=7,
         label='Baseline (no obs-CG)', zorder=3)
ax1.plot(em_iters, obs_space_cg, 'D-', color='#2ca02c', lw=2.5, markersize=8,
         label='Obs-space CG (Ours)', zorder=4)

ax1.axhline(oracle, color='grey', ls='--', lw=1.5, alpha=0.7, label=f'Clean-data oracle ({oracle})')

# Shade EM overfitting region
ax1.axvspan(3, 9, alpha=0.05, color='red')
ax1.annotate('EM overfitting region', xy=(6, 0.42), fontsize=9, color='darkred',
             ha='center', style='italic')

# Mark peaks
for data, color, peak_iter in [(obs_space_cg, '#2ca02c', 3),
                                 (baseline_no_obs_cg, '#ff7f0e', 3),
                                 (vanilla_mmps, '#d62728', 3)]:
    ax1.scatter([peak_iter], [data[em_iters == peak_iter]], s=120, facecolors='none',
                edgecolors=color, linewidths=2.5, zorder=5)

ax1.set_xlabel('EM Iteration', fontsize=12)
ax1.set_ylabel('Discriminative Score (↓ better)', fontsize=12)
ax1.set_title('Convergence & EM Overfitting', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.set_xlim(-0.5, 9.5)
ax1.set_ylim(0.02, 0.50)
ax1.grid(True, alpha=0.2)

# --- Right panel: degradation ratio table ---
ax2.axis('off')
table_data = [
    ['Method', 'Peak', 'Iter 7', 'Ratio'],
    ['Obs-space CG (Ours)', '0.045 (i=3)', '0.060', '1.3×'],
    ['Baseline (no obs-CG)', '0.049 (i=3)', '0.055*', '1.1×*'],
    ['Naive MMPS-EM', '0.092 (i=4)', '0.107', '1.2×'],
]

colors_row = [['#e6e6e6']*4, ['#d4edda']*4, ['#fff3cd']*4, ['#f8d7da']*4]
table = ax2.table(cellText=table_data, cellColours=colors_row,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 2.0)

for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(fontweight='bold')
    cell.set_edgecolor('grey')

ax2.set_title('Degradation under EM Overfitting\n(* diverges at iter 8-9)',
              fontsize=12, fontweight='bold', pad=20)

fig.suptitle('Figure 2: EM Convergence and Overfitting (Energy 50%)',
             fontsize=15, fontweight='bold', y=1.02)

plt.savefig('fig2_convergence.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig2_convergence.png', dpi=180, bbox_inches='tight')
print("Saved fig2_convergence.pdf and fig2_convergence.png")
