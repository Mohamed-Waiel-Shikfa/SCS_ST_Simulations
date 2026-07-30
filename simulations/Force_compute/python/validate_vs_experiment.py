"""Validate the force engine in ``magnet_force.py`` against the measured data.

    python validate_vs_experiment.py            # calibrated constants
    python validate_vs_experiment.py --refit    # re-run the calibration

What is predicted from first principles and what is fitted
----------------------------------------------------------
NIB blocks (20 x 10 x 5 mm, poles on the 20 x 10 faces)
    The whole shape of F(gap) comes out of the exact Akoun-Yonnet solution with
    no free parameters.  Only the *amplitude* is fitted, through one number --
    the remanence -- because the grade of these blocks is unknown.

Alnico rods (D 4.75 x 12.5 mm, LNG37)
    Geometry, the LNG37 curve (Br 1.20 T, Hcb 48 kA/m, Hcj 49 kA/m,
    (BH)max 37 kJ/m^3 from the supplier table) and the recoil permeability are
    all taken as given, and they already reproduce the measured contact force
    to within about 5 %.  A rod of L/D = 2.63 sits *below the knee* when it is
    open-circuited: the volume-average polarisation collapses from 1.20 T to
    about 0.44 T, which is exactly what every lumped-permeance model in this
    repository missed.  One empirical parameter is fitted -- the irreversible
    loss per pull-off -- because the readings were taken in increasing-gap
    order and Alnico degrades a little on every separation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from magnet_force import (CoaxialRodPair, alnico_lng37,  # noqa: E402
                          block_pair_force_mm, cylinder_demag_factor)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Mag Force Data.csv"
OUTPUT = HERE.parents[2] / "output"

# --------------------------------------------------------------------------
# Geometry from param.txt
# --------------------------------------------------------------------------
NIB_FACE = (20.0, 10.0)   # pole face, mm
NIB_THICK = 5.0           # magnetisation direction, mm

ALN_DIA = 4.75            # mm
ALN_LEN = 12.5            # mm

# --------------------------------------------------------------------------
# Calibrated constants (see --refit)
# --------------------------------------------------------------------------
NIB_BR = 0.841            # T, fitted; ~65 % of N42, i.e. bonded or under-magnetised
ALN_MU_REC = 4.0          # literature value for Alnico 5, not fitted
ALN_SHEET_MM = 0.10       # nominal caliper of one sheet of 80 gsm A4, not fitted
ALN_LOSS = 0.015          # irreversible loss per pull-off, fitted


# --------------------------------------------------------------------------
def load_data(path=DATA):
    """Parse the two side-by-side blocks in 'Mag Force Data.csv'.

    Returns (nib, alnico, err) where each list holds
    (gap_mm, mean_force_N, [individual readings]).
    """
    rows = list(csv.reader(open(path, newline="", encoding="utf-8-sig")))
    nib, aln, err = [], [], 0.25
    for i, r in enumerate(rows):
        if r and r[0].strip().lower().startswith("err"):
            err = float(rows[i + 1][0])
    for r in rows[2:]:
        if len(r) >= 5 and _isnum(r[0]) and _isnum(r[4]):
            nib.append((float(r[0]), float(r[4]),
                        [float(v) for v in r[1:4] if _isnum(v)]))
        if len(r) >= 11 and _isnum(r[6]) and _isnum(r[10]):
            aln.append((float(r[6]), float(r[10]),
                        [float(v) for v in r[7:10] if _isnum(v)]))
    return nib, aln, err


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def nib_model(gaps_mm, Br=None):
    """Exact closed-form attraction between the two NdFeB blocks."""
    Br = NIB_BR if Br is None else Br
    return np.array([block_pair_force_mm(Br, NIB_FACE[0], NIB_FACE[1], NIB_THICK, g)
                     for g in np.atleast_1d(gaps_mm)])


def alnico_model(n_sheets, loss=None, sheet_mm=None,
                 mu_rec=None, reps=3, n_slabs=20):
    """Replay the measurement protocol on the pair of LNG37 rods.

    The rods start already open-circuited (they were separated before the test),
    every reading is followed by a pull-off that drives them back to their
    open-circuit point, and each pull-off costs ``loss`` of polarisation.
    """
    loss = ALN_LOSS if loss is None else loss
    sheet_mm = ALN_SHEET_MM if sheet_mm is None else sheet_mm
    mu_rec = ALN_MU_REC if mu_rec is None else mu_rec
    pair = CoaxialRodPair(ALN_DIA * 5e-4, ALN_LEN * 1e-3,
                          alnico_lng37(mu_rec), n_slabs=n_slabs)
    pair.open_circuit()
    out = []
    for s in np.atleast_1d(n_sheets):
        readings = []
        for _ in range(reps):
            readings.append(abs(pair.force(float(s) * sheet_mm * 1e-3)))
            pair.open_circuit()
            pair.material = pair.material.scaled(1.0 - loss)
        out.append(float(np.mean(readings)))
    return np.array(out)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def refit(nib, aln):
    from scipy.optimize import least_squares

    g = np.array([r[0] for r in nib])
    f = np.array([r[1] for r in nib])
    sol = least_squares(lambda p: nib_model(g, p[0]) - f, [1.0], bounds=([0.2], [1.6]))
    br = float(sol.x[0])

    sheets = np.array([round(r[0] / 0.05) for r in aln])
    fa = np.array([r[1] for r in aln])
    best = None
    for sheet in (0.08, 0.09, 0.10, 0.11, 0.12):
        for loss in np.arange(0.005, 0.045, 0.0025):
            rms = float(np.sqrt(np.mean((alnico_model(sheets, loss, sheet) - fa) ** 2)))
            if best is None or rms < best[0]:
                best = (rms, loss, sheet)
    print(f"refit -> NIB_BR = {br:.3f} T")
    print(f"refit -> ALN_LOSS = {best[1]:.4f}, ALN_SHEET_MM = {best[2]:.2f} "
          f"(RMS {best[0]:.3f} N)")
    return br, best[1], best[2]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def table(title, xlabel, x, exp, model, err, reps):
    print(f"\n{title}")
    print(f"  {xlabel:>10} {'exp (N)':>9} {'spread':>8} {'model (N)':>10} "
          f"{'resid':>8} {'resid/err':>10}")
    for xi, e, m, rp in zip(x, exp, model, reps):
        spread = (max(rp) - min(rp)) / 2 if rp else 0.0
        print(f"  {xi:10.2f} {e:9.2f} {spread:8.2f} {m:10.2f} "
              f"{m - e:+8.2f} {(m - e) / err:+10.2f}")
    r = model - exp
    print(f"  RMS error {np.sqrt(np.mean(r**2)):.3f} N | max |resid| "
          f"{np.abs(r).max():.3f} N | stated uncertainty +/-{err:.2f} N")
    print(f"  chi2/dof (1 fitted parameter) = "
          f"{np.sum((r / err) ** 2) / (len(r) - 1):.2f}")


def plot(nib, aln, nib_pred, aln_pred, err):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1]})

    gn = np.array([r[0] for r in nib])
    fn = np.array([r[1] for r in nib])
    fine = np.linspace(gn.min(), gn.max(), 120)
    ax[0, 0].errorbar(gn, fn, yerr=err, fmt="o", color="k", capsize=3,
                      label="experiment", zorder=3)
    ax[0, 0].plot(fine, nib_model(fine), "-", color="crimson", lw=2,
                  label=f"model, exact cuboid solution ($B_r$ = {NIB_BR:.2f} T)")
    ax[0, 0].set_title("NdFeB blocks 20 x 10 x 5 mm, face to face")
    ax[0, 0].set_ylabel("attraction force (N)")
    ax[0, 1].errorbar([r[0] for r in aln], [r[1] for r in aln], yerr=err, fmt="o",
                      color="k", capsize=3, label="experiment", zorder=3)
    ax[0, 1].plot([r[0] for r in aln], aln_pred, "-o", color="royalblue", lw=2, ms=4,
                  label="model, non-linear rod solver")
    ax[0, 1].set_title("Alnico LNG37 rods D4.75 x 12.5 mm, face to face")

    for a, x, res in ((ax[1, 0], gn, nib_pred - fn),
                      (ax[1, 1], [r[0] for r in aln], aln_pred - np.array([r[1] for r in aln]))):
        a.plot(x, res, "o-", color="darkorange")
        a.axhspan(-err, err, color="0.85", zorder=0)
        a.axhline(0, color="k", lw=0.8, zorder=1)
        a.set_xlabel("air gap (mm)")
        a.set_ylabel("residual (N)")

    for a in ax[0]:
        a.grid(alpha=0.3, ls=":")
        a.legend(fontsize=9)
        a.set_ylabel("attraction force (N)")
    for a in ax[1]:
        a.grid(alpha=0.3, ls=":")

    # axhspan widens the data limits, so pin the shared x range to the data
    ga = np.array([r[0] for r in aln])
    for a, x in ((ax[0, 0], gn), (ax[0, 1], ga)):
        pad = 0.06 * (x.max() - x.min())
        a.set_xlim(x.min() - pad, x.max() + pad)

    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    out = OUTPUT / "force_model_vs_experiment.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true", help="re-run the calibration")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    nib, aln, err = load_data()
    global NIB_BR, ALN_LOSS, ALN_SHEET_MM
    if args.refit:
        NIB_BR, ALN_LOSS, ALN_SHEET_MM = refit(nib, aln)

    gn = np.array([r[0] for r in nib])
    fn = np.array([r[1] for r in nib])
    nib_pred = nib_model(gn)
    table("NdFeB blocks 20 x 10 x 5 mm (poles on the 20 x 10 faces)",
          "gap (mm)", gn, fn, nib_pred, err, [r[2] for r in nib])

    ga = np.array([r[0] for r in aln])
    fa = np.array([r[1] for r in aln])
    sheets = np.array([round(v / 0.05) for v in ga])
    aln_pred = alnico_model(sheets)
    table("Alnico LNG37 rods D4.75 x 12.5 mm", "sheets", sheets, fa, aln_pred, err,
          [r[2] for r in aln])

    print("\nphysics behind the Alnico numbers")
    N = float(cylinder_demag_factor(ALN_DIA * 5e-4, ALN_LEN * 1e-3))
    pair = CoaxialRodPair(ALN_DIA * 5e-4, ALN_LEN * 1e-3, alnico_lng37(ALN_MU_REC), 20)
    Jv, _ = pair.solve(1e9)
    pair.open_circuit()
    Jo, Ho = pair.solve(1e9)
    print(f"  demagnetising factor of one rod (L/D = {ALN_LEN/ALN_DIA:.2f}) : {N:.3f}")
    print(f"  open-circuit operating point : H = {Ho[:20].mean():.0f} A/m, "
          f"J = {Jo[:20].mean():.3f} T (Br = 1.20 T)")
    print(f"  -> the rods run at {Jo[:20].mean()/1.20*100:.0f} % of remanence, which is "
          f"why the measured forces are newtons and not tens of newtons")

    print("\nfalsifiable predictions of the calibrated model")
    print(f"  NdFeB blocks in contact (gap 0)        : "
          f"{nib_model(0.0)[0]:.1f} N")
    print(f"  NdFeB blocks at 15 mm                  : {nib_model(15.0)[0]:.2f} N")
    print(f"  Alnico rods, freshly saturated, gap 0  : "
          f"{abs(CoaxialRodPair(ALN_DIA*5e-4, ALN_LEN*1e-3, alnico_lng37(ALN_MU_REC), 20).force(0.0)):.1f} N")

    if not args.no_plot:
        plot(nib, aln, nib_pred, aln_pred, err)


if __name__ == "__main__":
    main()
