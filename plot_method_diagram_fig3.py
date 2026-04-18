"""
Figure 3: Method Overview Diagram (Section 6)

Dual-space architecture diagram showing:
  Left:  Standard MMPS (Single-Space) — image-space mask, high-dim CG, no projection
  Right: Our Adaptation (Dual-Space)  — obs-space CG, adaptive σ_y, manifold projection

This generates a publication-quality schematic using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
fig.subplots_adjust(wspace=0.15)

for ax in [ax_left, ax_right]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

# Colors
c_obs = '#4ECDC4'
c_img = '#FF6B6B'
c_cg = '#45B7D1'
c_proj = '#96CEB4'
c_mstep = '#FFEAA7'
c_flow = '#DDA0DD'

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, text='', color='black'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, text, fontsize=8, color='grey', style='italic')

# ── LEFT: Standard MMPS ──
ax_left.set_title('Standard MMPS\n(Single-Space)', fontsize=14, fontweight='bold', pad=20)

draw_box(ax_left, 1, 8.2, 3, 1, 'Observations\ny = A_img · x', c_obs)
draw_arrow(ax_left, 2.5, 8.2, 2.5, 7.5)

draw_box(ax_left, 0.5, 6, 4, 1.5, 'CG System\n(σ²I + σ²_t A_img J A^T_img) v = r\ndim = N_img (1024)', c_cg)
draw_arrow(ax_left, 2.5, 6, 2.5, 5.3)

draw_box(ax_left, 1, 3.8, 3, 1.2, 'Denoised x̂\n∈ ℝ^{N_img}\n(no constraint)', c_img)
draw_arrow(ax_left, 2.5, 3.8, 2.5, 3.1)

draw_box(ax_left, 1, 1.5, 3, 1.4, 'M-Step\nScore Matching\non completions', c_mstep)

# Annotations for problems
ax_left.text(5.5, 7.2, '✗ Wrong operator\n   (A_img ≠ G)', fontsize=10, color='red',
             fontweight='bold')
ax_left.text(5.5, 5.8, '✗ CG dim: 1024\n   (28× redundant)', fontsize=10, color='red',
             fontweight='bold')
ax_left.text(5.5, 4.3, '✗ κ → ∞ as σ_t ↑\n   (fixed σ_y)', fontsize=10, color='red',
             fontweight='bold')
ax_left.text(5.5, 2.8, '✗ Off-manifold drift\n   (no projection)', fontsize=10, color='red',
             fontweight='bold')

# ── RIGHT: Our Adaptation ──
ax_right.set_title('Our Adaptation\n(Dual-Space)', fontsize=14, fontweight='bold', pad=20)

draw_box(ax_right, 1, 8.2, 3.5, 1, 'TS Observations\ny_ts = A_ts · img_to_ts(x)', c_obs)
draw_arrow(ax_right, 2.75, 8.2, 2.75, 7.5)

draw_box(ax_right, 0.3, 6, 4.5, 1.5, 'Obs-Space CG\n(σ²_y I + σ²_t G J^T G^T) v = r\ndim = N_obs×F (36)', c_cg)
ax_right.text(5.2, 6.5, 'σ_y = c·σ_t\n→ κ bounded', fontsize=9, color='#2ca02c',
              fontweight='bold')
draw_arrow(ax_right, 2.75, 6, 2.75, 5.3)

draw_box(ax_right, 0.5, 3.8, 4, 1.2, 'Manifold Projection\nΠ(x̂) = ts_to_img(img_to_ts(x̂))\n+ obs enforcement', c_proj)
draw_arrow(ax_right, 2.75, 3.8, 2.75, 3.1)

draw_box(ax_right, 1, 1.5, 3.5, 1.4, 'M-Step\nVanilla Score Matching\n(no aux losses needed)', c_mstep)

# EM loop arrow
draw_arrow(ax_right, 4.8, 1.8, 5.5, 1.8, color='#2ca02c')
ax_right.annotate('', xy=(5.5, 8.7), xytext=(5.5, 1.8),
                  arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2,
                                  connectionstyle='arc3,rad=-0.3'))
ax_right.text(6.2, 5.5, 'EM Loop\n(3 iters)', fontsize=11, color='#2ca02c',
              fontweight='bold', ha='center',
              bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.8))

# Annotations for fixes
ax_right.text(5.2, 7.8, '✓ Correct operator G', fontsize=10, color='#2ca02c',
              fontweight='bold')
ax_right.text(5.2, 5.5, '✓ CG dim: 36\n   (28× reduction)', fontsize=10, color='#2ca02c',
              fontweight='bold')
ax_right.text(5.2, 4.5, '✓ On-manifold output', fontsize=10, color='#2ca02c',
              fontweight='bold')

fig.suptitle('Figure 3: Correcting MMPS for Dual-Space EM',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('fig3_method_diagram.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig3_method_diagram.png', dpi=180, bbox_inches='tight')
print("Saved fig3_method_diagram.pdf and fig3_method_diagram.png")
