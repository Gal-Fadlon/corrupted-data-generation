"""Architecture figure for the Co-Evolving EM paper (NeurIPS style).

Clean, thumbnail-driven layout inspired by imageni2r/figs/arch_opt.pdf:

  * rounded rectangles with a thin black outline, no fill
  * each stage holds a small thumbnail plot (line / heatmap / U-Net icon)
  * a small flame marks trainable components; snowflake marks frozen
  * the reverse-diffusion block is outlined in BLUE and contains three
    red-outlined sub-boxes (M1 / M2 / M3) stacked vertically --- the three
    novel corrected-MMPS modules introduced in the paper
  * the score-matching block is outlined in GREEN
  * heavy black arrows; minimal text

Produces corrupted_lifted/figs/fig_architecture.pdf and .png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

# --------------------------------------------------------------------------
# Paths and canvas
# --------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "corrupted_lifted" / "figs"
OUT_STEM = "fig_architecture"

FIG_W, FIG_H = 15.0, 7.0
DPI = 200

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11.0,
    "axes.linewidth": 0.0,
})

BORDER = "#111111"
BORDER_RED = "#c33333"
BORDER_BLUE = "#1f4e79"
BORDER_GREEN = "#1f7a45"
FREEZE_ICON = "\u2744"


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def add_box(ax, x, y, w, h, edgecolor=BORDER, linewidth=1.3,
            facecolor="white", rounding=1.1, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def inset_axes(ax, x, y, w, h):
    """Child Axes at (x,y,w,h) in the parent's 0..100 data coords."""
    ap = ax.get_position()
    fx = ap.x0 + (x / 100.0) * ap.width
    fy = ap.y0 + (y / 100.0) * ap.height
    fw = (w / 100.0) * ap.width
    fh = (h / 100.0) * ap.height
    ins = ax.figure.add_axes([fx, fy, fw, fh])
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(False)
    return ins


def add_arrow(ax, x0, y0, x1, y1, color=BORDER, lw=1.7, style="-|>",
              connectionstyle="arc3,rad=0", zorder=3, mutation_scale=18):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style,
        color=color,
        linewidth=lw,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def draw_fire_icon(ax, cx, cy, size=1.0, color="#e74c3c",
                   edge_color="#8b0000", x_stretch=1.0):
    flame = np.array([
        (0.00, -0.90), (0.50, -0.40), (0.30,  0.05), (0.60,  0.10),
        (0.20,  0.85), (0.00,  0.35), (-0.25, 0.95), (-0.35, 0.25),
        (-0.55, -0.05), (-0.35, -0.55),
    ]) * size
    flame[:, 0] *= x_stretch
    flame[:, 0] += cx
    flame[:, 1] += cy
    ax.add_patch(Polygon(flame, closed=True, facecolor=color,
                         edgecolor=edge_color, linewidth=0.9, zorder=6))
    inner = np.array([
        (0.00, -0.45), (0.22, -0.10), (0.10,  0.25), (0.12,  0.45),
        (-0.04, 0.30), (-0.18, 0.40), (-0.10, 0.05), (-0.22, -0.15),
    ]) * size * 0.7
    inner[:, 0] *= x_stretch
    inner[:, 0] += cx
    inner[:, 1] += cy
    ax.add_patch(Polygon(inner, closed=True, facecolor="#fff59d",
                         edgecolor="none", zorder=7))


# --------------------------------------------------------------------------
# Thumbnails
# --------------------------------------------------------------------------
def thumb_irregular_ts(ax, seed=1):
    rng = np.random.default_rng(seed)
    n = 22
    t = np.arange(n)
    y = np.sin(t * 0.55) + 0.35 * rng.standard_normal(n)
    obs = rng.random(n) > 0.45
    ax.plot(t, y, color="#bbbbbb", lw=0.9, zorder=1)
    ax.scatter(t[obs], y[obs], s=9, color="#1b7a3c", zorder=3)
    ax.scatter(t[~obs], y[~obs], s=12, color="#c33333", marker="x",
               linewidths=1.1, zorder=3)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(y.min() - 0.3, y.max() + 0.3)


def thumb_stl(ax):
    t = np.linspace(0, 1, 60)
    trend    = 0.55 * np.sin(2 * np.pi * 0.8 * t) + 0.15
    seasonal = 0.80 * np.sin(2 * np.pi * 4 * t)
    resid    = 0.22 * np.sin(2 * np.pi * 9 * t) * np.exp(-t)
    ax.plot(t, trend + 1.3,  color="#d35400", lw=1.5)
    ax.plot(t, seasonal,     color="#1f6fb2", lw=1.5)
    ax.plot(t, resid - 1.3,  color="#777777", lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-2.0, 2.1)


def thumb_delay_embed(ax):
    n = 16
    t = np.arange(n)
    y = 0.8 * np.sin(t * 0.45)
    ax.plot(t, y, color="black", lw=1.2)
    ax.scatter(t, y, s=10, color="#1b7a3c")
    ax.annotate("", xy=(19.5, 0.0), xytext=(16.2, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))
    rng = np.random.default_rng(7)
    side = 5
    img = rng.standard_normal((side, side))
    ax.imshow(img, extent=(20.5, 26.0, -1.3, 1.3), cmap="viridis",
              aspect="auto", zorder=2)
    ax.add_patch(Rectangle((20.5, -1.3), 5.5, 2.6, fill=False,
                           edgecolor="black", linewidth=0.9, zorder=3))
    ax.set_xlim(-1, 27)
    ax.set_ylim(-1.6, 1.6)


def thumb_image(ax, seed=3, cmap="viridis"):
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((10, 10))
    ax.imshow(img, cmap=cmap)


def thumb_noise_image(ax, seed=11):
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((10, 10))
    ax.imshow(img, cmap="gray")


def thumb_unet(ax, color_frame=BORDER):
    ax.fill([0.02, 0.32, 0.32, 0.02], [0.10, 0.30, 0.70, 0.90],
            facecolor="#cfe2f3", edgecolor=color_frame, linewidth=0.9,
            zorder=2)
    ax.fill([0.32, 0.60, 0.60, 0.32], [0.30, 0.45, 0.55, 0.70],
            facecolor="#fde2b3", edgecolor=color_frame, linewidth=0.9,
            zorder=3)
    ax.fill([0.60, 0.92, 0.92, 0.60], [0.45, 0.10, 0.90, 0.55],
            facecolor="#cfe2f3", edgecolor=color_frame, linewidth=0.9,
            zorder=2)
    for yy in (0.30, 0.50, 0.70):
        ax.annotate("", xy=(0.90, yy), xytext=(0.04, yy),
                    arrowprops=dict(arrowstyle="-", color="#888888",
                                    lw=0.5, linestyle=":"),
                    zorder=1)
    ax.set_xlim(0, 0.95)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")


def thumb_completed_stack(ax, n=4, seed=5):
    rng = np.random.default_rng(seed)
    for k in range(n):
        dx = 0.09 * k
        dy = 0.09 * (n - 1 - k)
        img = rng.standard_normal((8, 8))
        ax.imshow(img, extent=(dx, dx + 0.80, dy, dy + 0.80),
                  cmap="viridis", aspect="auto", zorder=n - k)
        ax.add_patch(Rectangle((dx, dy), 0.80, 0.80, fill=False,
                               edgecolor="black", linewidth=0.8,
                               zorder=n - k + 0.5))
    ax.set_xlim(0, 1.12)
    ax.set_ylim(0, 1.12)
    ax.set_aspect("equal", "box")


def thumb_theta(ax, label=r"$\theta_{k+1}$"):
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=26,
            fontweight="bold", color="#1b5a8a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")


def thumb_ts_sample(ax, seed=9):
    rng = np.random.default_rng(seed)
    n = 30
    t = np.arange(n)
    y = 0.8 * np.sin(t * 0.45) + 0.12 * rng.standard_normal(n)
    ax.plot(t, y, color="black", lw=1.3)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(y.min() - 0.3, y.max() + 0.3)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    aspect_corr = FIG_H / FIG_W  # ~0.47

    # ------------------- Layout -------------------
    # Three rows; E-step is TALLER than the other two because it contains
    # the three stacked M1/M2/M3 sub-boxes.
    ROW_Y = {"warm": 82, "estep": 35, "mstep": 5}
    ROW_H = {"warm": 13, "estep": 38, "mstep": 13}
    THUMB_PAD = 1.0

    # Subtle lane shading
    for key in ("warm", "estep", "mstep"):
        y0 = ROW_Y[key] - 2.5
        ax.add_patch(FancyBboxPatch(
            (1.2, y0), 97.5, ROW_H[key] + 5.0,
            boxstyle="round,pad=0.3,rounding_size=1.4",
            facecolor="#f6f7f9", edgecolor="none", zorder=0))

    # Row labels
    row_label_kw = dict(ha="center", va="center", fontsize=12.0,
                        rotation=90, fontweight="bold", color="#333")
    ax.text(2.2, ROW_Y["warm"] + ROW_H["warm"] / 2,
            "Warm start", **row_label_kw)
    ax.text(2.2, ROW_Y["estep"] + ROW_H["estep"] / 2,
            r"E-step (iter $k$)", **row_label_kw)
    ax.text(2.2, ROW_Y["mstep"] + ROW_H["mstep"] / 2,
            "M-step", **row_label_kw)

    # ==================================================================
    # Row 1: Warm start
    # ==================================================================
    y = ROW_Y["warm"]
    box_h = ROW_H["warm"]
    warm_defs = [
        (r"irregular $y_{\mathrm{ts}}$",        thumb_irregular_ts, {}),
        ("STL / Kalman impute",                  thumb_stl,          {}),
        (r"delay embed $\mathcal{T}$",           thumb_delay_embed,  {}),
        (r"$x_{\mathrm{init}} \in \mathrm{Range}(\mathcal{L})$",
                                                 thumb_image,       {"seed": 3}),
    ]
    warm_w = [12, 14, 14, 12]
    warm_gap = 3.2
    warm_total = sum(warm_w) + (len(warm_w) - 1) * warm_gap
    warm_start_x = (100 - warm_total) / 2 + 1.5
    warm_pos = []
    cx = warm_start_x
    for w in warm_w:
        warm_pos.append(cx)
        cx += w + warm_gap

    for (label, fn, kw), xx, ww in zip(warm_defs, warm_pos, warm_w):
        add_box(ax, xx, y, ww, box_h)
        ins = inset_axes(ax, xx + THUMB_PAD, y + THUMB_PAD,
                         ww - 2 * THUMB_PAD, box_h - 2 * THUMB_PAD)
        fn(ins, **kw)
        ax.text(xx + ww / 2, y + box_h + 0.6, label,
                ha="center", va="bottom", fontsize=11)

    for i in range(len(warm_pos) - 1):
        add_arrow(ax, warm_pos[i] + warm_w[i], y + box_h / 2,
                  warm_pos[i + 1], y + box_h / 2)

    # ==================================================================
    # Row 2: E-step.  Big diffusion box on the left contains the U-Net
    # plus three vertically-stacked M1/M2/M3 sub-boxes.
    # ==================================================================
    y = ROW_Y["estep"]
    row_h = ROW_H["estep"]

    # x-layout for the five stages
    #   [x_T]  [diffusion (wide + tall, M1/M2/M3 stacked inside)]  [posterior]
    #   [decode]  [x_hat_ts]
    x_T_w = 11
    diff_w = 25
    post_w = 12
    dec_w = 11
    ts_w = 13
    gap = 3.0

    e_total = x_T_w + diff_w + post_w + dec_w + ts_w + 4 * gap
    e_start = (100 - e_total) / 2 + 1.0
    e_pos = []
    cx = e_start
    for w in (x_T_w, diff_w, post_w, dec_w, ts_w):
        e_pos.append(cx)
        cx += w + gap

    # -- x_T box --
    small_h = 13
    x_T_y = y + (row_h - small_h) / 2
    add_box(ax, e_pos[0], x_T_y, x_T_w, small_h)
    ins = inset_axes(ax, e_pos[0] + THUMB_PAD, x_T_y + THUMB_PAD,
                     x_T_w - 2 * THUMB_PAD, small_h - 2 * THUMB_PAD)
    thumb_noise_image(ins)
    ax.text(e_pos[0] + x_T_w / 2, x_T_y + small_h + 0.6,
            r"$x_T = x_{\mathrm{init}} + \sigma_T\epsilon$",
            ha="center", va="bottom", fontsize=11)

    # -- Big diffusion box (tall): U-Net on the left half, M1/M2/M3 on the
    #    right half stacked vertically.
    add_box(ax, e_pos[1], y, diff_w, row_h,
            edgecolor=BORDER_BLUE, linewidth=1.8, rounding=1.5)
    ax.text(e_pos[1] + diff_w / 2, y + row_h + 0.6,
            r"reverse diffusion $d_{\theta_k}$  (corrected MMPS)",
            ha="center", va="bottom", fontsize=11.5,
            fontweight="bold", color=BORDER_BLUE)

    # U-Net inset on left half
    unet_x = e_pos[1] + 1.4
    unet_w = 10.5
    unet_y = y + (row_h - unet_w) / 2  # keep aspect square-ish
    ins = inset_axes(ax, unet_x, unet_y, unet_w, unet_w)
    thumb_unet(ins, color_frame=BORDER_BLUE)
    ax.text(unet_x + unet_w / 2, unet_y - 1.2,
            r"score $d_{\theta_k}$  " + FREEZE_ICON,
            ha="center", va="top", fontsize=10, color=BORDER_BLUE)

    # Three stacked M1/M2/M3 sub-boxes on the right half
    sub_x = e_pos[1] + diff_w - 12.5
    sub_w = 11.5
    n_sub = 3
    sub_gap = 0.9
    # Available vertical space for the 3 sub-boxes, leaving internal margins.
    avail_h = row_h - 3.0
    sub_h = (avail_h - (n_sub - 1) * sub_gap) / n_sub
    sub_y_top = y + row_h - 1.5

    sub_titles = [
        "M1  obs-space CG",
        r"M2  adaptive $\sigma_y$",
        r"M3  manifold projection $\Pi$",
    ]
    sub_subs = [
        r"$(\sigma_y^2 I + \sigma_t^2 G J G^\top)\,v = r$"
        "  \n"
        r"in $\mathbb{R}^{N_{\mathrm{obs}}}$   ($\sim\!28\!\times$ smaller)",
        r"$\sigma_y = c\,\sigma_t$" + "\n"
        r"keeps $\kappa$ bounded $\forall\,\sigma_t$",
        r"$\Pi = \mathcal{L}\,\mathcal{L}^{-1}$" + "\n"
        r"project onto $\mathrm{Range}(\mathcal{L})$",
    ]

    for k in range(n_sub):
        syk = sub_y_top - (k + 1) * sub_h - k * sub_gap
        add_box(ax, sub_x, syk, sub_w, sub_h,
                edgecolor=BORDER_RED, linewidth=1.4, rounding=0.7,
                zorder=4)
        ax.text(sub_x + sub_w / 2, syk + sub_h - 1.0,
                sub_titles[k], ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=BORDER_RED,
                zorder=5)
        ax.text(sub_x + sub_w / 2, syk + 1.0,
                sub_subs[k], ha="center", va="bottom",
                fontsize=8.8, color="#333", zorder=5)

    # Light connector arrow from U-Net to the stack (decorative)
    add_arrow(ax, unet_x + unet_w + 0.3, y + row_h / 2,
              sub_x - 0.3, y + row_h / 2,
              color="#888", lw=1.0, mutation_scale=12)

    # -- posterior box (centered vertically on the diffusion row midline)
    post_y = y + (row_h - small_h) / 2
    add_box(ax, e_pos[2], post_y, post_w, small_h)
    ins = inset_axes(ax, e_pos[2] + THUMB_PAD, post_y + THUMB_PAD,
                     post_w - 2 * THUMB_PAD, small_h - 2 * THUMB_PAD)
    thumb_image(ins, seed=21)
    ax.text(e_pos[2] + post_w / 2, post_y + small_h + 0.6,
            r"posterior  $\hat{x}_{\mathrm{img}}^{(i,k)}$",
            ha="center", va="bottom", fontsize=11)

    # -- decode box
    add_box(ax, e_pos[3], post_y, dec_w, small_h)
    ax.text(e_pos[3] + dec_w / 2, post_y + small_h / 2,
            r"$\mathcal{T}^{-1}$" + "\n(img $\\to$ ts)",
            ha="center", va="center", fontsize=10.5, color="#333")
    ax.text(e_pos[3] + dec_w / 2, post_y + small_h + 0.6,
            "decode", ha="center", va="bottom", fontsize=11)

    # -- x_hat_ts box
    add_box(ax, e_pos[4], post_y, ts_w, small_h)
    ins = inset_axes(ax, e_pos[4] + THUMB_PAD, post_y + THUMB_PAD,
                     ts_w - 2 * THUMB_PAD, small_h - 2 * THUMB_PAD)
    thumb_ts_sample(ins)
    ax.text(e_pos[4] + ts_w / 2, post_y + small_h + 0.6,
            r"$\hat{x}_{\mathrm{ts}}^{(i,k)}$",
            ha="center", va="bottom", fontsize=11)

    # Row arrows along the midline of the smaller boxes
    mid_y = post_y + small_h / 2
    add_arrow(ax, e_pos[0] + x_T_w, mid_y, e_pos[1], mid_y)
    add_arrow(ax, e_pos[1] + diff_w, mid_y, e_pos[2], mid_y)
    add_arrow(ax, e_pos[2] + post_w, mid_y, e_pos[3], mid_y)
    add_arrow(ax, e_pos[3] + dec_w, mid_y, e_pos[4], mid_y)

    # ==================================================================
    # Row 3: M-step
    # ==================================================================
    y = ROW_Y["mstep"]
    box_h = ROW_H["mstep"]
    m_defs = [
        ("completed dataset",  thumb_completed_stack, {}),
        ("EDM score matching", thumb_unet, {"color_frame": BORDER_GREEN}),
        ("updated prior",      thumb_theta, {"label": r"$\theta_{k+1}$"}),
    ]
    m_w = [14, 18, 12]
    m_gap = 4.0
    m_total = sum(m_w) + (len(m_w) - 1) * m_gap
    m_start = (100 - m_total) / 2 + 1.0
    m_pos = []
    cx = m_start
    for w in m_w:
        m_pos.append(cx)
        cx += w + m_gap

    for i, ((label, fn, kw), xx, ww) in enumerate(
            zip(m_defs, m_pos, m_w)):
        ec = BORDER_GREEN if i == 1 else BORDER
        lw = 1.8 if i == 1 else 1.3
        add_box(ax, xx, y, ww, box_h, edgecolor=ec, linewidth=lw)
        ins = inset_axes(ax, xx + THUMB_PAD, y + THUMB_PAD,
                         ww - 2 * THUMB_PAD, box_h - 2 * THUMB_PAD)
        fn(ins, **kw)
        color = BORDER_GREEN if i == 1 else "#111"
        weight = "bold" if i == 1 else "normal"
        ax.text(xx + ww / 2, y + box_h + 0.6, label,
                ha="center", va="bottom", fontsize=11,
                color=color, fontweight=weight)

    # Fire icon on the score-matching box
    sm_x = m_pos[1]
    sm_w = m_w[1]
    draw_fire_icon(ax, sm_x + sm_w - 1.3, y + 2.0,
                   size=1.1, x_stretch=aspect_corr)

    for i in range(len(m_pos) - 1):
        add_arrow(ax, m_pos[i] + m_w[i], y + box_h / 2,
                  m_pos[i + 1], y + box_h / 2)

    # ==================================================================
    # Cross-row connectors
    # ==================================================================
    # Warm -> E-step (feed x_init as seed)
    warm_last_x = warm_pos[-1] + warm_w[-1] / 2
    e_first_x = e_pos[0] + x_T_w / 2
    add_arrow(
        ax,
        warm_last_x, ROW_Y["warm"],
        e_first_x, ROW_Y["estep"] + row_h / 2 + 5.5,
        connectionstyle="arc3,rad=-0.35",
        lw=1.6,
    )
    ax.text(67.0, ROW_Y["warm"] - 4.0,
            r"seed  $x_T = x_{\mathrm{init}} + \sigma_T\epsilon$",
            ha="center", va="center", fontsize=10.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.98),
            zorder=6)

    # E-step posterior -> M-step completed dataset
    post_bot_x = e_pos[2] + post_w / 2
    m_first_top_x = m_pos[0] + m_w[0] / 2
    add_arrow(
        ax,
        post_bot_x, post_y,
        m_first_top_x, ROW_Y["mstep"] + ROW_H["mstep"],
        connectionstyle="arc3,rad=0.22",
        lw=1.6,
    )
    ax.text((post_bot_x + m_first_top_x) / 2 + 5.0,
            (post_y + ROW_Y["mstep"] + ROW_H["mstep"]) / 2 - 1.0,
            r"batch $\{\hat{x}_{\mathrm{img}}^{(i,k)}\}_i$",
            ha="center", va="center", fontsize=10.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.95))

    # Feedback loop: theta_{k+1} -> diffusion box (next EM iter).
    theta_right_x = m_pos[-1] + m_w[-1]
    theta_mid_y = ROW_Y["mstep"] + ROW_H["mstep"] / 2
    diff_right_x = e_pos[1] + diff_w
    diff_mid_y = ROW_Y["estep"] + ROW_H["estep"] / 2

    loop = FancyArrowPatch(
        (theta_right_x + 0.3, theta_mid_y),
        (diff_right_x + 0.3, diff_mid_y),
        arrowstyle="-|>",
        color="#c23616",
        linewidth=2.0,
        mutation_scale=22,
        connectionstyle="arc3,rad=-0.45",
        zorder=5,
    )
    ax.add_patch(loop)
    ax.text(94.5, (theta_mid_y + diff_mid_y) / 2,
            r"$\theta_k \!\leftarrow\! \theta_{k+1}$" + "\n(next EM iter.)",
            ha="center", va="center", fontsize=10, color="#c23616",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#c23616", alpha=0.95, linewidth=0.8))

    # Score-matching loss under the M-step box
    ax.text(sm_x + sm_w / 2, y - 1.6,
            r"$\mathcal{L} = \mathbb{E}\!\left[\lambda(\sigma)\,"
            r"\|d_\theta(x + \sigma\epsilon, \sigma) - x\|^2\right]$",
            ha="center", va="top", fontsize=10, color=BORDER_GREEN)

    # ==================================================================
    # Save
    # ==================================================================
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
