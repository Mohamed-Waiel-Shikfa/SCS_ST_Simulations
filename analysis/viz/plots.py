"""Server-rendered figures for the design explorer.

Everything here re-runs the real solvers.  Nothing is a cartoon: the field map
is the actual FEM solution on the actual mesh, the force curve is the same
Maxwell-stress integral Stage 1 uses, and the pivot plot integrates the same
separations Stage 2 does.  If a figure and a number disagree, that is a bug and
it should be visible.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from axisym_fem import AxisymModel, axial_force, sample_B  # noqa: E402
from compat import trapezoid  # noqa: E402
from framework import (Design, _pivot_geometry, _regions,  # noqa: E402
                       material, stage1_magnetics, stage2_mechanics,
                       stage3_switching)
from module import build_module  # noqa: E402

BG = "#12151c"
FG = "#e6e9ef"
GRID = "#2a3040"
ACC = "#5aa9ff"
ACC2 = "#ff8f5a"
GOOD = "#4ec9a0"
BAD = "#ff6b6b"
WARN = "#ffc14d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "grid.color": GRID,
    "font.size": 9, "axes.titlesize": 10, "axes.grid": True,
    "grid.linewidth": 0.5, "grid.alpha": 0.5,
    "figure.autolayout": False,
    "legend.framealpha": 0.85, "legend.facecolor": "#1a1f2b",
    "legend.edgecolor": GRID,
})


def _mesh_size(dsg, fidelity):
    frac = 6.0 if fidelity == "screen" else 16.0
    floor = 0.2e-3 if fidelity == "screen" else 0.15e-3
    return max(min(dsg.d_mag, dsg.l_mag) / frac, floor)


# --------------------------------------------------------------------------
# Parallel + cached Stage 1.  A gap sweep is ten independent FEM solves and
# was taking 163 s serially, which is unusable behind a UI.  They are pure
# functions of (design, gap, fidelity), so they parallelise perfectly and
# cache exactly.
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _ckey(row, gap, fidelity, states):
    return json.dumps([sorted(row.items()), gap, fidelity, list(states)],
                      default=str)


def _stage1_worker(args):
    row, gap, fidelity, states = args
    from framework import Design as _D
    from framework import stage1_magnetics as _s1
    try:
        return _s1(_D(**{**row, "gap": gap}), fidelity=fidelity,
                   states=tuple(states))
    except RuntimeError:
        return None


def stage1_many(dsg, gaps, fidelity="screen", states=("attract", "repel"),
                workers=None):
    """Stage 1 at many gaps, in parallel, with a process-wide cache."""
    row = dsg.as_row()
    todo, out = [], {}
    for g in gaps:
        k = _ckey(row, g, fidelity, states)
        with _CACHE_LOCK:
            hit = _CACHE.get(k)
        if hit is not None:
            out[g] = hit
        else:
            todo.append(g)

    if todo:
        n = workers or max(1, min(os.cpu_count() or 1, len(todo), 12))
        payload = [(row, g, fidelity, list(states)) for g in todo]
        if n == 1 or len(todo) == 1:
            res = [_stage1_worker(p) for p in payload]
        else:
            with ProcessPoolExecutor(max_workers=n) as pool:
                res = list(pool.map(_stage1_worker, payload, chunksize=1))
        for g, r in zip(todo, res):
            out[g] = r
            with _CACHE_LOCK:
                _CACHE[_ckey(row, g, fidelity, states)] = r
    return out


_MECH = {}


def stage2_cached(dsg, fidelity):
    """(magnetics, mechanics, module) for a design, computed once.

    ``stage2_mechanics`` internally re-probes the repulsion at two extra gaps
    to build the pivot work integral, so calling it once per figure was paying
    for the same FEM solves several times over.
    """
    key = _ckey(dsg.as_row(), None, fidelity, ("mech",))
    with _CACHE_LOCK:
        hit = _MECH.get(key)
    if hit is not None:
        return hit
    mag = stage1_magnetics(dsg, fidelity=fidelity)
    mod = build_module(dsg, None)
    mech = stage2_mechanics(dsg, mag, mod=mod, fidelity=fidelity)
    out = (mag, mech, mod)
    with _CACHE_LOCK:
        _MECH[key] = out
    return out


_REFIT = {}


def experiment_fit():
    """Load the measured data and fit the two free parameters, once.

    The fit is a grid search over shim thickness and per-pull-off loss and
    takes about two and a half minutes.  It depends on nothing the user can
    change, so it is computed once per process.
    """
    if "v" not in _REFIT:
        import validate_vs_experiment as V
        nib, aln, err = V.load_data()
        Br, loss, sheet = V.refit(nib, aln)
        _REFIT["v"] = (nib, aln, err, Br, loss, sheet)
    return _REFIT["v"]


def warm_experiment_fit():
    """Kick the fit off in the background so the first view is not a wait."""
    threading.Thread(target=experiment_fit, daemon=True).start()


def _build(dsg, flip, fidelity):
    Rm = dsg.d_mag / 2
    ro = Rm + (dsg.r_clear + dsg.t_steel if dsg.circuit == "potcore" else 0.0)
    ns = 3 if fidelity == "screen" else 6
    rfar_k, zfar_k = (12, 10) if fidelity == "screen" else (25, 20)
    rfar = rfar_k * max(ro, Rm)
    m = AxisymModel(_regions(dsg, flip), rfar, zfar_k * dsg.l_mag,
                    _mesh_size(dsg, fidelity), n_slabs=ns)
    return m, m.solve(), rfar


# --------------------------------------------------------------------------
def field_map(dsg, state="attract", fidelity="screen"):
    """|B| in the (r, z) half-plane, with the geometry drawn on top.

    This is the solution Stage 1 integrates for force, not a redrawing of it.
    """
    flip = (state == "repel")
    m, sol, rfar = _build(dsg, flip, fidelity)

    Rm, Lm, gap = dsg.d_mag / 2, dsg.l_mag, dsg.gap
    ro = Rm + (dsg.r_clear + dsg.t_steel if dsg.circuit == "potcore" else 0.0)
    span_r = 3.4 * max(ro, Rm)
    pad = max(3e-3, 0.45 * Lm)
    z0, z1 = -Lm - pad - dsg.t_steel, gap + Lm + pad + dsg.t_steel

    nr, nz = 300, 340
    rr = np.linspace(1e-6, span_r, nr)
    zz = np.linspace(z0, z1, nz)
    R, Z = np.meshgrid(rr, zz)
    Br, Bz = sample_B(sol, R.ravel(), Z.ravel())
    B = np.hypot(Br, Bz).reshape(nz, nr)

    fig, ax = plt.subplots(figsize=(5.4, 6.4))
    # mirror across the axis so the picture reads as a real cross-section
    Bfull = np.hstack([B[:, ::-1], B])
    rfull = np.concatenate([-rr[::-1], rr])

    # A linear scale is useless here: the interior sits near 1 T and the
    # fringing field that actually produces force is two orders below it, so
    # everything outside the iron reads as black.  Gamma compression shows
    # both without misrepresenting either - the colourbar stays in tesla.
    vmax = float(np.nanpercentile(Bfull, 99.5))
    norm = matplotlib.colors.PowerNorm(gamma=0.45, vmin=0.0,
                                       vmax=max(vmax, 1e-6))
    im = ax.pcolormesh(rfull * 1e3, zz * 1e3, Bfull, cmap="magma",
                       norm=norm, shading="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("|B|  (T)")
    cb.outline.set_edgecolor(GRID)

    ax.contour(rfull * 1e3, zz * 1e3, Bfull,
               levels=np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8]) * vmax,
               colors="#ffffff", linewidths=0.4, alpha=0.30)

    def box(r0, r1, za, zb, ec, lw, ls):
        for sgn in (+1, -1):
            lo = min(sgn * r0, sgn * r1)
            ax.add_patch(Rectangle((lo * 1e3, za * 1e3),
                                   abs(r1 - r0) * 1e3, (zb - za) * 1e3,
                                   fill=False, edgecolor=ec, lw=lw, ls=ls,
                                   zorder=5))

    for g in m.regions:
        if g.kind == "magnet":
            box(g.rmin, g.rmax, g.zmin, g.zmax, ACC, 1.6, "-")
        elif g.kind == "steel":
            box(g.rmin, g.rmax, g.zmin, g.zmax, GOOD, 1.2, "--")

    ax.axhline(gap / 2 * 1e3, color=WARN, lw=1.0, ls=":", zorder=6)
    ax.text(-span_r * 1e3 * 0.97, gap / 2 * 1e3, "force plane ",
            color=WARN, fontsize=7, va="bottom", ha="left")

    JA, HA = m.region_state(sol, "A")
    F = axial_force(sol, gap / 2, r_max=0.9 * rfar,
                    n=1500 if fidelity == "screen" else 4000)
    hcj = material(dsg.material).Hcj

    ax.set_xlim(-span_r * 1e3, span_r * 1e3)
    ax.set_ylim(z0 * 1e3, z1 * 1e3)
    ax.set_xlabel("r  (mm)")
    ax.set_ylabel("z  (mm)")
    ax.set_title(f"{dsg.material}  {state.upper()}   |B|\n"
                 f"J {JA:.3f} T,  H {HA/1e3:.1f} kA/m "
                 f"({abs(HA)/hcj:.2f} Hcj),  F {abs(F):.2f} N",
                 fontsize=9)
    ax.set_aspect("equal")
    ax.grid(False)
    fig.text(0.01, 0.005,
             f"blue = magnet, green dashed = 1018 steel.  Colour is gamma "
             f"compressed so the fringing field is visible.\n"
             f"fidelity={fidelity}, mesh {_mesh_size(dsg,fidelity)*1e3:.2f} "
             f"mm, {m.mesh.t.shape[1]} elements",
             fontsize=6.5, color="#8b93a7")
    return fig


# --------------------------------------------------------------------------
def force_vs_gap(dsg, fidelity="screen", gaps_mm=(0.05, 0.1, 0.2, 0.4, 0.7,
                                                  1.0, 1.5, 2.0, 3.0, 4.0)):
    """Attraction and repulsion against gap, plus the asymmetry."""
    want = sorted(set(list(gaps_mm) + [dsg.gap * 1e3]))
    res = stage1_many(dsg, [g * 1e-3 for g in want], fidelity=fidelity)

    gaps, fa, fr, ja, jr = [], [], [], [], []
    for g in want:
        m = res.get(g * 1e-3)
        if m is None:
            continue
        gaps.append(g)
        fa.append(m["F_attract"])
        fr.append(m["F_repel"])
        ja.append(m["J_attract"])
        jr.append(m["J_repel"])
    gaps = np.array(gaps)
    fa, fr = np.array(fa), np.array(fr)

    fig, axes = plt.subplots(3, 1, figsize=(5.6, 7.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1.2, 1.2]))
    ax = axes[0]
    ax.plot(gaps, fa, "o-", color=ACC, lw=1.8, ms=4, label="attract")
    ax.plot(gaps, fr, "s-", color=ACC2, lw=1.8, ms=4, label="repel")
    ax.axvline(dsg.gap * 1e3, color=WARN, ls=":", lw=1.1)
    ax.text(dsg.gap * 1e3, float(fa.max()) * 0.96, " design gap", color=WARN,
            fontsize=7, va="top")
    ax.set_ylabel("force  (N)")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.set_title(f"{dsg.material}, D {dsg.d_mag*1e3:.2f} x L "
                 f"{dsg.l_mag*1e3:.2f} mm, {dsg.circuit}   "
                 f"[{fidelity}]", fontsize=9)

    ax = axes[1]
    ax.plot(gaps, fa / np.maximum(fr, 1e-9), "o-", color=GOOD, lw=1.7, ms=4)
    ax.axhline(1.0, color=GRID, lw=1)
    ax.set_ylabel("asymmetry\nFa / Fr")
    ax.axvline(dsg.gap * 1e3, color=WARN, ls=":", lw=1.1)

    ax = axes[2]
    ax.plot(gaps, ja, "o-", color=ACC, lw=1.6, ms=3.5, label="attract")
    ax.plot(gaps, jr, "s-", color=ACC2, lw=1.6, ms=3.5, label="repel")
    ax.set_ylabel("operating J  (T)")
    ax.set_xlabel("gap  (mm)")
    ax.axvline(dsg.gap * 1e3, color=WARN, ls=":", lw=1.1)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xscale("log")

    fig.text(0.01, 0.005,
             "J falls as the pair separates: each magnet loses the permeance "
             "its neighbour was providing.\nThat recovery on the repel side is"
             " why repulsion decays far more slowly than a fixed-strength "
             "model predicts.",
             fontsize=6.5, color="#8b93a7")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return fig


# --------------------------------------------------------------------------
def demag_curves(names, marks=()):
    """Second-quadrant intrinsic curves, with operating points marked.

    ``marks`` is a list of {material, H, J, label} so a computed operating
    point can be shown on the curve it came from.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    colours = [ACC, ACC2, GOOD, WARN, "#c48bff", "#5ad6ff"]
    for i, nm in enumerate(names):
        try:
            mat = material(nm)
        except (KeyError, ValueError):
            continue
        H = -np.linspace(0, mat.Hcj, 600)
        c = colours[i % len(colours)]
        ax.plot(H / 1e3, mat.J(H), lw=1.9, color=c,
                label=f"{nm}  Br {mat.Br:.2f} T, Hcj {mat.Hcj/1e3:.0f} kA/m")
        ax.plot(H / 1e3, mat.B(H), lw=1.0, color=c, ls="--", alpha=0.55)

    for mk in marks or []:
        try:
            mat = material(mk["material"])
        except (KeyError, ValueError):
            continue
        ax.plot(mk["H"] / 1e3, mk["J"], "o", ms=8, mfc="none", mew=2,
                color=BAD if mk.get("bad") else "#ffffff")
        ax.annotate(mk.get("label", ""), (mk["H"] / 1e3, mk["J"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=7.5,
                    color=BAD if mk.get("bad") else "#ffffff")

    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("H  (kA/m)")
    ax.set_ylabel("J  (T)      [dashed: B]")
    ax.set_title("Intrinsic demagnetisation curves", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.text(0.01, 0.005,
             "The flat top is what protects the OFF state; the knee is what "
             "makes a rod collapse when it self-demagnetises.",
             fontsize=6.5, color="#8b93a7")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# --------------------------------------------------------------------------
def pivot_work_plot(dsg, fidelity="screen"):
    """The pivot work integral, drawn against the barrier it must beat."""
    full, mech, mod = stage2_cached(dsg, fidelity)
    probe = [dsg.gap, 1e-3, 4e-3]
    res = stage1_many(dsg, probe, fidelity=fidelity, states=("repel",))
    if any(res[g] is None for g in probe):
        raise RuntimeError("magnet solve failed at one of the probe gaps")

    gp = np.array(probe)
    fp = np.array([res[g]["F_repel"] for g in probe])

    def F(s):
        s = np.atleast_1d(np.asarray(s, float))
        out = np.interp(s, gp, fp)
        tail = s > gp[-1]
        if np.any(tail):
            c_end = gp[-1] + dsg.l_mag
            out = np.where(tail, fp[-1] * (c_end / (s + dsg.l_mag)) ** 4, out)
        return out

    th = np.linspace(0, 2 * np.pi / dsg.n_gon, 200)
    s = _pivot_geometry(dsg.n_gon, dsg.r_face, th) + dsg.gap
    f = F(s)
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (f[1:] + f[:-1]) * np.diff(s))])

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.2))
    ax = axes[0]
    ax.plot(np.degrees(th), f, lw=2, color=ACC2)
    ax.fill_between(np.degrees(th), 0, f, color=ACC2, alpha=0.18)
    ax.set_ylabel("repulsion  (N)")
    ax.set_xlabel("roll angle  (deg)")
    ax.set_title(f"Pivot drive over one "
                 f"{np.degrees(2*np.pi/dsg.n_gon):.0f} deg step  "
                 f"(n={dsg.n_gon})   [{fidelity}]", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(np.degrees(th), s * 1e3, lw=1.2, color="#8b93a7", ls="--")
    ax2.set_ylabel("face gap  (mm)", color="#8b93a7")
    ax2.grid(False)

    ax = axes[1]
    ax.plot(np.degrees(th), cum * 1e3, lw=2, color=GOOD, label="work done")
    ax.axhline(mech["E_barrier"] * 1e3, color=BAD, lw=1.6, ls="--",
               label=f"gravity barrier {mech['E_barrier']*1e3:.2f} mJ")
    ok = mech["pivot_ratio"] >= 1.5
    ax.set_ylabel("energy  (mJ)")
    ax.set_xlabel("roll angle  (deg)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(f"work {mech['W_drive']*1e3:.2f} mJ  /  barrier "
                 f"{mech['E_barrier']*1e3:.2f} mJ  =  "
                 f"{mech['pivot_ratio']:.2f}   "
                 f"({'passes' if ok else 'FAILS'} the 1.5 constraint)",
                 fontsize=9, color=GOOD if ok else BAD)

    fig.text(0.01, 0.005,
             "Energy is necessary, not sufficient: it has to arrive as "
             "rotation about the pivot edge, and\nsome goes into sliding and "
             "friction. MuJoCo is the arbiter - see the Dynamics tab.",
             fontsize=6.5, color="#8b93a7")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# --------------------------------------------------------------------------
def pulse_waveform(dsg):
    """Coil current during the switching pulse, against the MMF it must reach.

    Series RLC discharge of the capacitor bank into the coil.  Which regime it
    is in matters: an overdamped loop cannot reach peak current before the
    bank empties, which is what killed the 0.1 mm coil in the built prototype.
    """
    sw = stage3_switching(dsg)
    L, R, C, V = sw["L_coil"], sw["R_coil"], dsg.c_cap, dsg.v_cap
    N = sw["n_turns"]

    alpha = R / (2 * L)
    w0 = 1.0 / np.sqrt(L * C)
    t = np.linspace(0, min(6.0 / max(alpha, 1e-9), 20e-3), 2000)
    if alpha < w0:
        wd = np.sqrt(w0**2 - alpha**2)
        i = V / (L * wd) * np.exp(-alpha * t) * np.sin(wd * t)
        regime = "underdamped (oscillatory)"
    elif abs(alpha - w0) / max(w0, 1e-9) < 1e-6:
        i = V / L * t * np.exp(-alpha * t)
        regime = "critically damped"
    else:
        s1 = -alpha + np.sqrt(alpha**2 - w0**2)
        s2 = -alpha - np.sqrt(alpha**2 - w0**2)
        i = V / (L * (s1 - s2)) * (np.exp(s1 * t) - np.exp(s2 * t))
        regime = "overdamped"

    mmf = i * N
    need = sw["mmf_need"]
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ax.plot(t * 1e3, mmf, lw=2, color=ACC)
    ax.axhline(need, color=BAD, ls="--", lw=1.5,
               label=f"MMF needed {need:.0f} A-turns")
    peak = float(np.max(mmf))
    ax.axhline(peak, color=GOOD, ls=":", lw=1.2,
               label=f"peak reached {peak:.0f} A-turns")
    ax.fill_between(t * 1e3, need, mmf, where=(mmf >= need),
                    color=GOOD, alpha=0.18)
    ax.set_xlabel("time  (ms)")
    ax.set_ylabel("MMF  (ampere-turns)")
    ok = peak >= need
    ax.set_title(f"Switching pulse - {regime}\n"
                 f"{N:.0f} turns, L {L*1e6:.1f} uH, R {R*1e3:.0f} mohm, "
                 f"{C*1e6:.0f} uF at {V:.0f} V   "
                 f"margin {peak/max(need,1e-9):.2f}x",
                 fontsize=9, color=GOOD if ok else BAD)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.text(0.01, 0.005,
             "An overdamped loop empties the bank before the current peaks - "
             "the failure mode of the built 0.1 mm coil.",
             fontsize=6.5, color="#8b93a7")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# --------------------------------------------------------------------------
def experiment_validation():
    """The calibrated force model against the measured pull-off data."""
    import validate_vs_experiment as V

    nib, aln, err, Br, loss, sheet = experiment_fit()

    gn = np.array([r[0] for r in nib])
    fn = np.array([r[1] for r in nib])
    ga = np.array([r[0] for r in aln])
    fa = np.array([r[1] for r in aln])
    mn = V.nib_model(gn, Br=Br)
    ma = V.alnico_model(np.array([round(g / 0.05) for g in ga]),
                        loss=loss, sheet_mm=sheet)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for ax, g, f, m, reps, title, unit in (
            (axes[0], gn, fn, mn, [r[2] for r in nib],
             "NdFeB blocks", "gap (mm)"),
            (axes[1], ga, fa, ma, [r[2] for r in aln],
             "Alnico 5 rods", "shim stack (mm)")):
        for x, rr in zip(g, reps):
            ax.plot([x] * len(rr), rr, ".", color="#5c6478", ms=4, zorder=1)
        ax.errorbar(g, f, yerr=err, fmt="o", color=ACC, ms=6, capsize=3,
                    elinewidth=1, label=f"measured (+/- {err} N)", zorder=3)
        ax.plot(g, m, "-", color=ACC2, lw=2, label="model", zorder=2)
        rms = float(np.sqrt(np.mean((np.array(m) - f) ** 2)))
        inside = int(np.sum(np.abs(np.array(m) - f) <= err))
        ax.set_title(f"{title}   RMS {rms:.3f} N   "
                     f"{inside}/{len(f)} inside error bars", fontsize=9.5)
        ax.set_xlabel(unit)
        ax.set_ylabel("force (N)")
        ax.legend(fontsize=7.5)

    fig.suptitle("Force engine vs measured data", fontsize=10)
    fig.text(0.01, 0.005,
             f"Fitted: NdFeB Br {Br:.3f} T (magnetised through the 20 mm "
             f"axis, not the 10 mm one), Alnico {loss*100:.1f} % loss per "
             f"pull-off on {sheet:.2f} mm shims. Readings were taken in "
             f"increasing-gap order, so losses accumulate down the column - "
             f"grey dots are the individual repeats.",
             fontsize=6.5, color="#8b93a7")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig
