r"""What is the 3-D solver good for, and where does it stop being trusted?

The 3-D solver exists because the axisymmetric FEM cannot represent two
modules meeting at an angle, and a pivoting module spends its whole manoeuvre
at an angle.  But a new solver is worth nothing until someone has tried to
break it, so this script does that and reports what it finds - including the
part that did not come out well.

Summary of the result, so it is not buried:

* **Magnets without a return path: verified.**  Against the validated
  one-dimensional magnetisation solver and against the axisymmetric FEM, the
  operating point agrees to about 2 % and the force to 2-5 %, converging under
  refinement and stable to a couple of per cent across discretisations at
  every angle tested.
* **Magnets inside a steel pot core: NOT verified, and the two solvers
  disagree.**  The magnet operating point still agrees to 1.3 %, but the force
  does not: the 3-D solver reads about 20 % high on attraction and three times
  high on repulsion.  The disagreement is unchanged when both solvers are
  forced to use linear iron, so it is not saturation modelling.  Since J
  agrees and F does not, the two models are splitting the flux differently
  between the pole face and the annulus rim, and the repelling force is a
  small difference between a large pole-to-pole repulsion and a large
  pole-to-rim attraction - so a modest error in that split becomes a large
  error in the total.

Neither model has experimental validation for the pot-core geometry - the
measured data in this repository is for bare rods - so this is an open
discrepancy rather than a proven bug in one of them.  The pipeline therefore
takes its magnitudes from the axisymmetric FEM, which is the older and cheaper
model, and uses the 3-D solver only for what it is verified to do.

Four bugs this script found, every one of which produced plausible numbers:

1.  The kernel's Hz term used atan2 rather than the principal branch, so a
    cube's self-demagnetising factor came out as -2/3 instead of 1/3.
2.  The magnet's irreversible-loss history latched onto the first iterate's
    overshoot, permanently demagnetising the magnet with a field that was
    never a physical operating point.
3.  An angled pair was rotated about the pole-face centres rather than the
    shared rim, so the two magnets interpenetrated beyond three degrees.
4.  Square tiles of a circular pole reached 25 % past the magnet radius and
    ran into the steel annulus, 50 to 96 overlapping cell pairs of it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

import fem3d  # noqa: E402
from magnet_force import CoaxialRodPair  # noqa: E402
from materials import material  # noqa: E402

D, L, MAT, GAP = 4.2e-3, 8.4e-3, "LNGT44", 0.1e-3
TS, RC = 1.0e-3, 0.5e-3


def axisym(flip, steel, gap=GAP, fidelity="normal"):
    from framework import Design, stage1_magnetics
    d = Design(material=MAT, d_mag=D, l_mag=L,
               circuit="potcore" if steel else "none",
               t_steel=TS if steel else 0.3e-3, r_clear=RC if steel else 0.0,
               gap=gap)
    out = stage1_magnetics(d, fidelity=fidelity,
                           states=("repel",) if flip else ("attract",))
    return (out["F_repel"] if flip else out["F_attract"],
            out["J_repel"] if flip else out["J_attract"])


def solve(states, steel, gap=GAP, angle=0.0, na=3, nz=4, ns=8):
    mat = material(MAT)
    cells = fem3d.epm_pair_cells(D, L, mat, gap,
                                 t_steel=TS if steel else 0.0,
                                 r_clear=RC if steel else 0.0,
                                 states=states, angle=angle, n_across=na,
                                 n_axial=nz, n_sect=ns, with_steel=steel)
    bad = fem3d.check_overlap(cells)
    sol = fem3d.solve3d(cells, tol=1e-6, max_iter=30)
    F, T = fem3d.force_on_body(sol, 1, 0, max_sub=16)
    return sol, F, T, len(bad)


def main():
    print("=" * 78)
    print("3-D SOLVER: WHAT IT IS VERIFIED TO DO")
    print("=" * 78)
    print(f"\n  {MAT}, {D*1e3:.1f} x {L*1e3:.1f} mm rod, {GAP*1e3:.2f} mm gap")

    print("\n  1.  Open-circuit magnetisation, against the validated 1-D "
          "solver")
    print("  " + "-" * 68)
    mat = material(MAT)
    J1d, _ = CoaxialRodPair(D / 2, L, mat, n_slabs=24).solve(1e3 * L)
    ref = float(np.mean(J1d))
    print(f"      1-D reference               J = {ref:.4f} T")
    for na, nz in ((1, 2), (2, 4), (3, 6), (4, 8)):
        cells = fem3d.epm_cells(np.zeros(3), np.array([1.0, 0.0, 0.0]), D, L,
                                mat, n_across=na, n_axial=nz,
                                with_steel=False)
        sol = fem3d.solve3d(cells)
        J, _ = sol.magnet_state()
        print(f"      3-D {na} rings x {nz:<2} ({len(cells):3d} cells)"
              f"   J = {J:.4f} T  {(J-ref)/ref*100:+5.1f} %")

    print("\n  2.  Force with no return path, against the axisymmetric FEM")
    print("  " + "-" * 68)
    fa_ref = axisym(False, False)[0]
    fr_ref = axisym(True, False)[0]
    print(f"      {'discretisation':<16} {'F attract':>11} {'err':>8} "
          f"{'F repel':>10} {'err':>8}")
    print(f"      {'axisym FEM':<16} {fa_ref:10.3f}N {'-':>8} "
          f"{fr_ref:9.3f}N {'-':>8}")
    for na, nz in ((1, 3), (2, 4), (3, 5)):
        _, Fa, _, _ = solve(fem3d.ATTRACT, False, na=na, nz=nz)
        _, Fr, _, _ = solve(fem3d.REPEL, False, na=na, nz=nz)
        print(f"      {f'3-D {na} rings x {nz}':<16} {abs(Fa[0]):10.3f}N "
              f"{(abs(Fa[0])-fa_ref)/fa_ref*100:+7.1f}% {abs(Fr[0]):9.3f}N "
              f"{(abs(Fr[0])-fr_ref)/fr_ref*100:+7.1f}%")

    print("\n  3.  Angular dependence - the reason this solver exists.")
    print("      No reference exists, so the test is stability under "
          "refinement.")
    print("  " + "-" * 68)
    degs = (0, 10, 22.5, 45)
    for states, name in ((fem3d.REPEL, "repel"), (fem3d.ATTRACT, "attract")):
        print(f"      {name}, |F| in N")
        print(f"      {'cells':<12}" + "".join(f"{d:>11.1f} deg"
                                               for d in degs))
        table = []
        for na, nz in ((1, 3), (2, 4), (3, 5)):
            row = []
            for deg in degs:
                _, F, _, ov = solve(states, False, angle=np.radians(deg),
                                    na=na, nz=nz)
                assert ov == 0, "cells overlap - the scene is not physical"
                row.append(float(np.linalg.norm(F)))
            table.append(row)
            print(f"      {f'{na} x {nz}':<12}" +
                  "".join(f"{v:15.4f}" for v in row))
        A = np.array(table)
        spread = (A.max(axis=0) - A.min(axis=0)) / A.mean(axis=0) * 100
        print(f"      {'spread':<12}" +
              "".join(f"{v:14.1f}%" for v in spread))

    print("\n  4.  With a steel pot core: the two solvers DISAGREE.")
    print("  " + "-" * 68)
    print(f"      {'state':<9} {'axisym F':>10} {'3-D F':>10} {'dF':>9} "
          f"{'axisym J':>10} {'3-D J':>8} {'dJ':>8}")
    for states, flip, name in ((fem3d.ATTRACT, False, "attract"),
                               (fem3d.REPEL, True, "repel")):
        fa, Ja = axisym(flip, True)
        sol, F, _, ov = solve(states, True)
        J3, _ = sol.magnet_state(body=0)
        print(f"      {name:<9} {fa:9.3f}N {abs(F[0]):9.3f}N "
              f"{(abs(F[0])-fa)/fa*100:+8.1f}% {Ja:10.3f} {J3:8.3f} "
              f"{(J3-Ja)/Ja*100:+7.1f}%   overlaps {ov}")

    print("""
      The magnet operating point agrees to about one per cent while the force
      does not, and forcing both solvers to use linear iron changes nothing,
      so it is not saturation.  Neither model is validated against measurement
      for this geometry - the experimental data here is for bare rods - so
      this is recorded as an open discrepancy rather than silently averaged
      away.

      Consequently the pipeline uses:
        * the AXISYMMETRIC FEM for every magnitude, unchanged;
        * the 3-D solver for the ANGULAR transfer function, computed with
          magnets only, where section 3 shows it is stable;
        * the 3-D solver for the volumetric field drawn in the viewer, which
          is the field it actually solved.""")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  ({time.time()-t0:.0f} s)")
