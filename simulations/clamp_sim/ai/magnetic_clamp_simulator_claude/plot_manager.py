"""
Plot Manager
============
Creates all matplotlib figures used by the three UI panels:
  • B-H demagnetisation curve with load line & operating point
  • 2D sandwich visualisation with magnetic gradient overlay
  • LaTeX equation renderings (algebraic and numerical)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; figures embedded in tkinter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib import rcParams

from physics_engine import MATERIALS, MU_0

# ── Global matplotlib styling ──
rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.facecolor": "#1e1e2e",
    "figure.facecolor": "#1e1e2e",
    "axes.edgecolor": "#6c7086",
    "axes.labelcolor": "#cdd6f4",
    "xtick.color": "#cdd6f4",
    "ytick.color": "#cdd6f4",
    "text.color": "#cdd6f4",
    "axes.grid": True,
    "grid.color": "#313244",
    "grid.alpha": 0.6,
})

COLORS = {
    "bg":       "#1e1e2e",
    "surface":  "#313244",
    "text":     "#cdd6f4",
    "accent1":  "#89b4fa",   # blue
    "accent2":  "#f38ba8",   # red/pink
    "accent3":  "#a6e3a1",   # green
    "accent4":  "#fab387",   # peach
    "magnet":   "#cba6f7",   # mauve
    "cover_plastic": "#9399b2",
    "cover_steel":   "#74c7ec",
    "gap":      "#45475a",
}


# ═════════════════════════════════════════════════════════════════
#  Panel 2 Top:  B-H Curve
# ═════════════════════════════════════════════════════════════════

def create_bh_figure(results, material_name, figsize=(5.4, 3.6)):
    """2nd-quadrant demagnetisation curve + load line + operating point."""
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    mat = MATERIALS[material_name]
    H_bh = results["H_bh"]
    B_bh = results["B_bh"]
    H_ll = results["H_loadline"]
    B_ll = results["B_loadline"]
    Hm_op = results["Hm_op"]
    Bm_op = results["Bm_op"]

    # Plot demagnetisation curve
    ax.plot(H_bh / 1e3, B_bh, color=mat["color"], linewidth=2.5,
            label=f"{material_name} Demag. Curve", zorder=3)

    # Plot load line
    ax.plot(H_ll / 1e3, B_ll, color=COLORS["accent2"],
            linewidth=2, linestyle="--", label="Load Line", zorder=3)

    # Operating point
    ax.plot(Hm_op / 1e3, Bm_op, 'o', color=COLORS["accent3"],
            markersize=10, markeredgecolor='white', markeredgewidth=1.5,
            zorder=5, label=f"Operating Point")
    ax.annotate(
        f"  $O$({Hm_op / 1e3:.1f}, {Bm_op:.3f})",
        xy=(Hm_op / 1e3, Bm_op),
        fontsize=9, color=COLORS["accent3"],
        xytext=(15, 10), textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color=COLORS["accent3"], lw=1.2),
    )

    # Axis labels with LaTeX
    ax.set_xlabel(r"$H_m$ (kA/m)", fontsize=11)
    ax.set_ylabel(r"$B_m$ (T)", fontsize=11)
    ax.set_title("2nd Quadrant Demagnetisation", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7,
              facecolor=COLORS["surface"], edgecolor=COLORS["accent1"])
    ax.set_xlim(H_bh[-1] / 1e3 * 1.05, 0.5)
    ax.set_ylim(-0.05, mat["Br"] * 1.15)

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════
#  Panel 3 Top:  Setup Visualisation
# ═════════════════════════════════════════════════════════════════

def create_setup_figure(engine, results, show_gradient=True,
                        show_vectors=True, figsize=(5.4, 4.0)):
    """2D cross-section of the 5-layer sandwich with optional
    magnetic-field colour map and force vectors."""
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    L_mm  = engine.L * 1e3
    W_mm  = engine.W * 1e3
    tm_mm = engine.tm * 1e3
    tc_mm = engine.tc * 1e3
    g_half_mm = engine.g_min * 1e3 / 2.0
    alpha_half = engine.alpha_deg / 2.0

    cover_color = COLORS["cover_steel"] if engine.cover_is_steel \
                  else COLORS["cover_plastic"]

    def _add_rotated_rect(ax, cx, cy, w, h, angle, color, label=None,
                          alpha_fill=0.70, edgecolor='white'):
        rect = patches.Rectangle(
            (-w / 2, -h / 2), w, h,
            linewidth=1.4, edgecolor=edgecolor,
            facecolor=color, alpha=alpha_fill, label=label,
        )
        t = matplotlib.transforms.Affine2D().rotate_deg(angle).translate(cx, cy) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    # ── Component centres ──
    y_tc = g_half_mm + tc_mm / 2.0
    y_tm = g_half_mm + tc_mm + tm_mm / 2.0
    y_bc = -(g_half_mm + tc_mm / 2.0)
    y_bm = -(g_half_mm + tc_mm + tm_mm / 2.0)
    cx = W_mm / 2.0

    # Rotate centres around pivot at left-inner corner
    import math
    def rotate_point(px, py, pivot_x, pivot_y, angle_deg):
        a = math.radians(angle_deg)
        dx, dy = px - pivot_x, py - pivot_y
        return (pivot_x + dx * math.cos(a) - dy * math.sin(a),
                pivot_y + dx * math.sin(a) + dy * math.cos(a))

    if alpha_half > 0.01:
        px_t, py_t = 0, g_half_mm
        cx_tc_r, cy_tc_r = rotate_point(cx, y_tc, px_t, py_t, alpha_half)
        cx_tm_r, cy_tm_r = rotate_point(cx, y_tm, px_t, py_t, alpha_half)
        px_b, py_b = 0, -g_half_mm
        cx_bc_r, cy_bc_r = rotate_point(cx, y_bc, px_b, py_b, -alpha_half)
        cx_bm_r, cy_bm_r = rotate_point(cx, y_bm, px_b, py_b, -alpha_half)
    else:
        cx_tc_r, cy_tc_r = cx, y_tc
        cx_tm_r, cy_tm_r = cx, y_tm
        cx_bc_r, cy_bc_r = cx, y_bc
        cx_bm_r, cy_bm_r = cx, y_bm

    # ── Gradient map (behind everything) ──
    if show_gradient:
        X_mm, Y_mm, Bx, By, B_mag = engine.get_field_map(nx=70, ny=70)
        extent = [X_mm[0, 0], X_mm[0, -1], Y_mm[0, 0], Y_mm[-1, 0]]
        vmax = max(np.percentile(B_mag, 97), 0.01)
        im = ax.imshow(
            B_mag, extent=extent, origin='lower',
            cmap='inferno', alpha=0.50, aspect='auto',
            norm=Normalize(vmin=0, vmax=vmax),
            zorder=0, interpolation='bilinear',
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.82)
        cbar.set_label(r"$|\mathbf{B}|$ (T)", fontsize=9)
        cbar.ax.tick_params(labelsize=7)

    # ── Draw rectangles ──
    _add_rotated_rect(ax, cx_tm_r, cy_tm_r, W_mm, tm_mm,
                      alpha_half, COLORS["magnet"], label="Magnet")
    _add_rotated_rect(ax, cx_tc_r, cy_tc_r, W_mm, tc_mm,
                      alpha_half, cover_color,
                      label="Steel Cover" if engine.cover_is_steel else "Plastic Cover")
    _add_rotated_rect(ax, cx_bm_r, cy_bm_r, W_mm, tm_mm,
                      -alpha_half, COLORS["magnet"])
    _add_rotated_rect(ax, cx_bc_r, cy_bc_r, W_mm, tc_mm,
                      -alpha_half, cover_color)

    # ── Polarity arrows ──
    arr = "▼ N" if engine.is_attracting else "▼ N"
    ax.text(cx_tm_r, cy_tm_r, "N ▼ S", ha='center', va='center',
            fontsize=8, color='white', fontweight='bold', zorder=10)
    bot_label = "S ▲ N" if engine.is_attracting else "N ▼ S"
    ax.text(cx_bm_r, cy_bm_r, bot_label, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold', zorder=10)

    # ── Force vectors ──
    if show_vectors and results["F_total"] > 0:
        F_N = results["F_total"]
        arrow_len = min(tm_mm * 0.7, 8)
        arr_col = COLORS["accent2"]
        # Top: downward pull
        ax.annotate('', xy=(cx, g_half_mm + 0.5),
                    xytext=(cx, g_half_mm + 0.5 + arrow_len),
                    arrowprops=dict(arrowstyle='->', color=arr_col, lw=2.5),
                    zorder=15)
        ax.text(cx + 2, g_half_mm + 0.5 + arrow_len + 1.5,
                f"$F={F_N:.1f}$ N", fontsize=8, color=arr_col,
                ha='center', va='bottom', zorder=15,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS["bg"],
                          edgecolor=arr_col, alpha=0.85))
        # Bottom: upward pull
        ax.annotate('', xy=(cx, -g_half_mm - 0.5),
                    xytext=(cx, -g_half_mm - 0.5 - arrow_len),
                    arrowprops=dict(arrowstyle='->', color=arr_col, lw=2.5),
                    zorder=15)

    # ── Gap dimension line ──
    xd = W_mm + 3
    ax.annotate('', xy=(xd, g_half_mm), xytext=(xd, -g_half_mm),
                arrowprops=dict(arrowstyle='<->', color=COLORS["accent3"],
                                lw=1.5), zorder=15)
    ax.text(xd + 1, 0, f"gap\n{engine.g_min*1e3:.1f} mm",
            fontsize=7, color=COLORS["accent3"], va='center', zorder=15)

    # ── Axis limits & labels ──
    total_h = 2 * (tm_mm + tc_mm + g_half_mm)
    margin = total_h * 0.3
    ax.set_xlim(-margin, W_mm + margin + 10)
    ax.set_ylim(-total_h / 2 - margin, total_h / 2 + margin)
    ax.set_xlabel("Width (mm)", fontsize=9)
    ax.set_ylabel("Height (mm)", fontsize=9)
    ax.set_title("Symmetric Wedge Setup", fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='lower left', fontsize=7, framealpha=0.6,
              facecolor=COLORS["surface"], edgecolor=COLORS["accent1"])

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════
#  Panel 1 Bottom:  Algebraic Equations (LaTeX, no numbers)
# ═════════════════════════════════════════════════════════════════

def create_equations_algebraic(engine, figsize=(5.4, 3.6)):
    """Render the general physics equations in LaTeX (no numeric values)."""
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.axis('off')

    cover_note = "steel: $l_{g,eff} = g$" if engine.cover_is_steel \
        else "plastic: $l_{g,eff} = g + 2t_c$"
    polarity_note = "Attracting" if engine.is_attracting else "Repelling"

    lines = [
        (f"Config: {polarity_note},  {cover_note}",  0.96, False, 10, COLORS["accent4"]),
        ("Ampere's Law (no current):",                0.87, False, 10, COLORS["accent1"]),
        (r"$2\, H_m \, t_m + H_g \, l_{g,eff} = 0$", 0.79, True, 13, COLORS["text"]),
        ("Flux Conservation:",                         0.69, False, 10, COLORS["accent1"]),
        (r"$B_m \, A_m = B_g \, A_g$",                0.61, True, 13, COLORS["text"]),
        ("Gap Constitutive Relation:",                 0.51, False, 10, COLORS["accent1"]),
        (r"$B_g = \mu_0 \, H_g$",                     0.43, True, 13, COLORS["text"]),
        ("Load Line (Eq. 5 from paper):",              0.33, False, 10, COLORS["accent1"]),
        (r"$B_m = -\frac{A_{g,eff}}{A_m}\,"
         r"\frac{2\,\mu_0\, t_m}{l_{g,eff}}\, H_m$", 0.23, True, 13, COLORS["text"]),
        ("Clamping Force (Maxwell Stress):",           0.12, False, 10, COLORS["accent1"]),
        (r"$F = \frac{B_g^{2}\, A_g}{2\,\mu_0}$",    0.03, True, 13, COLORS["text"]),
    ]

    for txt, y_pos, is_math, fontsize, color in lines:
        ax.text(0.05, y_pos, txt, transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', color=color)

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════
#  Panel 2 Bottom:  Numerical Equations (LaTeX with numbers)
# ═════════════════════════════════════════════════════════════════

def create_equations_numerical(engine, results, figsize=(5.4, 3.6)):
    """Same equations with current numerical values substituted."""
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.axis('off')

    r = results
    tm_mm = engine.tm * 1e3
    lg_mm = r["lg_eff"] * 1e3
    Am_cm2 = r["Am"] * 1e4
    Ag_cm2 = r["Ag_eff"] * 1e4
    Hm = r["Hm_op"]
    Bm = r["Bm_op"]
    Hg = r["Hg"]
    Bg = r["Bg"]
    F  = r["F_total"]
    slope = r["slope"]

    lines = [
        ("Ampere's Law:",                    0.96, 10, COLORS["accent1"]),
        (rf"$2 \times ({Hm:.0f}) \times {tm_mm:.1f}$mm"
         rf" $+ {Hg:.0f} \times {lg_mm:.2f}$mm $= 0$",
                                             0.87, 11, COLORS["text"]),
        ("Flux Conservation:",               0.77, 10, COLORS["accent1"]),
        (rf"${Bm:.4f}$T $\times$ ${Am_cm2:.2f}$cm² $=$"
         rf" ${Bg:.4f}$T $\times$ ${Ag_cm2:.2f}$cm²",
                                             0.68, 11, COLORS["text"]),
        ("Load Line slope:",                 0.58, 10, COLORS["accent1"]),
        (rf"$B_m = ({slope:.2e}) \times H_m$",
                                             0.49, 12, COLORS["text"]),
        ("Operating Point:",                 0.39, 10, COLORS["accent1"]),
        (rf"$H_m = {Hm:.0f}$ A/m,   $B_m = {Bm:.4f}$ T",
                                             0.30, 12, COLORS["accent3"]),
        ("Gap Flux Density:",                0.20, 10, COLORS["accent1"]),
        (rf"$B_g = \mu_0 \times {Hg:.0f} = {Bg:.4f}$ T",
                                             0.11, 12, COLORS["text"]),
        ("Force:",                           0.02, 10, COLORS["accent1"]),
        (rf"$F = {Bg:.4f}^2 \times {Am_cm2:.2f}$cm²"
         rf" $/ (2\mu_0) = {F:.2f}$ N",
                                             -0.07, 12, COLORS["accent3"]),
    ]

    for txt, y_pos, fontsize, color in lines:
        ax.text(0.05, y_pos, txt, transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', color=color)

    fig.tight_layout()
    return fig
