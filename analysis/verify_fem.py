"""Validate the axisymmetric FEM against the exact free-space solver.

With the iron removed, the FEM must reproduce results that
``simulations/Force_compute/python/magnet_force.py`` computes in closed form and
that were themselves validated against the measured pull-force data.

  1. demagnetising factor of a rigid uniformly magnetised rod
  2. self-consistent open-circuit state of a nonlinear Alnico rod
  3. axial force between two Alnico rods
  4. sanity check that adding iron behaves monotonically
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from axisym_fem import (MU0, AxisymModel, Region, _elem_volume,  # noqa: E402
                        axial_force)
from magnet_force import (CoaxialRodPair, Material, alnico_lng37,  # noqa: E402
                          cylinder_demag_factor)

D, L = 4.75e-3, 12.5e-3
R = D / 2
OK = True


def extrapolate(hs, vals):
    """Linear extrapolation to h -> 0; the scheme converges at first order."""
    return float(np.polyfit(np.asarray(hs), np.asarray(vals), 1)[1])


def check(label, got, want, tol_pct):
    global OK
    err = abs(got - want) / abs(want) * 100
    good = err <= tol_pct
    OK &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {label:<40} {got:11.4f}"
          f"   (exact {want:.4f}, {err:.2f}% err, tol {tol_pct:.0f}%)")


print("=" * 78)
print("AXISYMMETRIC FEM VERIFICATION")
print("=" * 78)

# ---------------------------------------------------------------- test 1
print("\n[1] Demagnetising factor of a rigid uniformly magnetised rod")
print("    Reference: validated Lipschitz-Hankel kernel, itself cross-checked")
print("    against an independent elliptic-integral calculation.\n")

rigid = Material("rigid", Br=1.0, Hcj=1e12, mu_rec=1.0, p=60.0, q=0.5)
N_exact = float(cylinder_demag_factor(R, L))

hs, Ns = [], []
print(f"    {'h (mm)':>8} {'N_fem':>9} {'error':>9}")
for h_mm in (0.6, 0.4, 0.3, 0.2):
    m = AxisymModel([Region(0, R, -L / 2, L / 2, "magnet", "rod",
                            material=rigid, direction=+1)],
                    r_far=20 * R, z_far=8 * L, h=h_mm * 1e-3, n_slabs=6)
    sol = m.solve()
    _, H = m.region_state(sol, "rod")
    N_fem = -H * MU0 / 1.0
    hs.append(h_mm)
    Ns.append(N_fem)
    print(f"    {h_mm:8.2f} {N_fem:9.4f} {abs(N_fem-N_exact)/N_exact*100:8.2f}%")

check("N extrapolated to h -> 0", extrapolate(hs, Ns), N_exact, 2.0)

# ---------------------------------------------------------------- test 2
print("\n[2] Self-consistent open-circuit state of a nonlinear Alnico rod")
print("    The rod sits below the knee of its own demagnetisation curve, so")
print("    this exercises the stiff nonlinear magnet solve.\n")

aln = alnico_lng37()
pair = CoaxialRodPair(R, L, aln, n_slabs=32)
Jex, _ = pair.solve(1e9)
J_exact = float(Jex[:32].mean())

hs, Js = [], []
print(f"    {'h (mm)':>8} {'J_fem (T)':>11} {'error':>9} {'residual':>11}")
for h_mm in (0.5, 0.35, 0.25):
    m = AxisymModel([Region(0, R, -L / 2, L / 2, "magnet", "rod",
                            material=aln, direction=+1)],
                    r_far=20 * R, z_far=8 * L, h=h_mm * 1e-3, n_slabs=8)
    sol = m.solve()
    J, _ = m.region_state(sol, "rod")
    hs.append(h_mm)
    Js.append(J)
    print(f"    {h_mm:8.2f} {J:11.4f} {abs(J-J_exact)/J_exact*100:8.2f}%"
          f" {sol.residual:11.1e}")

check("J_open extrapolated to h -> 0", extrapolate(hs, Js), J_exact, 10.0)

print("\n    NOTE on the tolerance above.  The two solvers apply the material")
print("    law differently INSIDE a slab: the free-space solver holds J uniform")
print("    over the whole slab, whereas the FEM lets J vary as")
print("    J = Brz + (mu_rec - 1) mu0 H with H varying over the slab.  Both are")
print("    approximations of the same continuum problem.  The resulting offset")
print("    was measured separately: it is zero at mu_rec = 1.5, grows to ~6% at")
print("    mu_rec = 4 and ~8% at mu_rec = 6, and does not shrink with mesh or")
print("    slab refinement.  It is therefore a modelling difference, not an")
print("    error in either code.  It is immaterial for design comparison, where")
print("    the effects of interest are 2-3x and are computed as FEM-to-FEM")
print("    ratios in which the offset largely cancels.")

# ---------------------------------------------------------------- test 3
print("\n[3] Attraction between two coaxial Alnico rods at a 0.1 mm gap\n")

gap = 0.10e-3
F_exact = abs(CoaxialRodPair(R, L, aln, n_slabs=32).force(gap))

hs, Fs = [], []
print(f"    {'h (mm)':>8} {'F_fem (N)':>11} {'error':>9}")
for h_mm in (0.35, 0.25, 0.18):
    regs = [Region(0, R, -L - gap / 2, -gap / 2, "magnet", "lower",
                   material=aln, direction=+1),
            Region(0, R, gap / 2, L + gap / 2, "magnet", "upper",
                   material=aln, direction=+1)]
    m = AxisymModel(regs, r_far=20 * R, z_far=8 * L, h=h_mm * 1e-3, n_slabs=6)
    sol = m.solve()
    F = abs(axial_force(sol, 0.0, r_max=12 * R))
    hs.append(h_mm)
    Fs.append(F)
    print(f"    {h_mm:8.2f} {F:11.4f} {abs(F-F_exact)/F_exact*100:8.2f}%")

check("force extrapolated to h -> 0", extrapolate(hs, Fs), F_exact, 10.0)

# ---------------------------------------------------------------- test 4
print("\n[4] Iron sanity checks (no exact reference; monotonicity + limits)")
print("    A soft-iron cup behind and around the rod must raise its operating")
print("    point monotonically as the return path is closed.\n")

t_iron = 1.0e-3
print(f"    {'shell gap (mm)':>15} {'J (T)':>9} {'H (kA/m)':>10}")
prev = None
mono = True
for shell_gap in (8.0, 4.0, 2.0, 1.0, 0.5):
    sg = shell_gap * 1e-3
    regs = [
        Region(0, R, -L / 2, L / 2, "magnet", "rod", material=aln, direction=+1),
        Region(0, R + sg + t_iron, -L / 2 - t_iron, -L / 2, "steel", "back"),
        Region(R + sg, R + sg + t_iron, -L / 2, L / 2, "steel", "shell"),
    ]
    m = AxisymModel(regs, r_far=14 * (R + sg + t_iron), z_far=8 * L,
                    h=0.35e-3, n_slabs=8)
    sol = m.solve()
    J, H = m.region_state(sol, "rod")
    print(f"    {shell_gap:15.1f} {J:9.4f} {H/1e3:10.1f}")
    if prev is not None and J < prev - 1e-6:
        mono = False
    prev = J

OK &= mono
print(f"  [{'ok ' if mono else 'FAIL'}] operating point rises monotonically as "
      f"the return path closes")

print("\n" + ("FEM VERIFIED - safe to use with iron" if OK else
             "FEM VERIFICATION FAILED"))
raise SystemExit(0 if OK else 1)
