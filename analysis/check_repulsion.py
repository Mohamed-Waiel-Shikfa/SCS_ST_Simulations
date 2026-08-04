"""Is the attract/repel asymmetry real, or partly a measurement artefact?

Two checks before designing around it.

1. RIGID magnets.  If the magnetisation cannot change, reversing one magnet
   negates every equivalent surface charge on it, so the force must reverse
   sign with EXACTLY equal magnitude.  Any asymmetry in the rigid case is an
   error in the force computation, not physics.

2. TRUNCATION.  The Maxwell stress is integrated over a finite disc in the
   mid-gap plane.  In the repel state flux is pushed sideways and spreads much
   further radially than in the attract state, so the integral may need a far
   larger radius to converge.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from axisym_fem import AxisymModel, Region, axial_force  # noqa: E402
from magnet_force import Material, alnico_lng37  # noqa: E402

Rm, Lm = 2.375e-3, 12.5e-3
GAP = 0.1e-3

RIGID = Material("rigid", Br=0.8, Hcj=1e12, mu_rec=1.0, p=60.0, q=0.5)
ALN = alnico_lng37()


def pair(mat, flip):
    return [Region(0, Rm, -Lm, 0.0, "magnet", "A", material=mat, direction=+1),
            Region(0, Rm, GAP, GAP + Lm, "magnet", "B", material=mat,
                   direction=(-1 if flip else +1))]


print("=" * 74)
print("[1] RIGID magnets: does |F_repel| equal |F_attract| ?")
print("=" * 74)
print("    If yes, the force routine is sound and the asymmetry seen with")
print("    Alnico is purely a magnetisation effect.\n")

RFAR = 60 * Rm
out = {}
for flip, lab in ((False, "ATTRACT"), (True, "REPEL")):
    m = AxisymModel(pair(RIGID, flip), RFAR, 20 * Lm, 0.3e-3, n_slabs=4)
    s = m.solve()
    J, H = m.region_state(s, "A")
    F = axial_force(s, GAP / 2, r_max=0.9 * RFAR, n=8000)
    out[lab] = F
    print(f"    {lab:>8}  J = {J:.4f} T   F = {F:+9.4f} N")
print(f"\n    |F_repel| / |F_attract| = "
      f"{abs(out['REPEL'])/abs(out['ATTRACT']):.4f}   (must be 1.000)")

print("\n" + "=" * 74)
print("[2] Does the force integral converge in radius?")
print("=" * 74)
print("    Alnico, both states, integrating the mid-plane stress out to")
print("    increasing radius.\n")

print(f"    {'r_max / R_mag':>14} {'attract (N)':>13} {'repel (N)':>12}")
sols = {}
for flip, lab in ((False, "ATTRACT"), (True, "REPEL")):
    m = AxisymModel(pair(ALN, flip), RFAR, 20 * Lm, 0.3e-3, n_slabs=6)
    sols[lab] = m.solve()

for frac in (5, 10, 20, 30, 45, 55):
    rmax = frac * Rm
    if rmax > 0.95 * RFAR:
        continue
    fa = axial_force(sols["ATTRACT"], GAP / 2, r_max=rmax, n=8000)
    fr = axial_force(sols["REPEL"], GAP / 2, r_max=rmax, n=8000)
    print(f"    {frac:14d} {fa:13.3f} {fr:12.3f}")

print("\n    (the earlier study integrated to about 15 x the OUTER radius;")
print("     if these columns are still moving, that number was truncated)")
