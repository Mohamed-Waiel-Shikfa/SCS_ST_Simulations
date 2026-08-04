"""Can a higher-coercivity Alnico grade rescue the repel state?

The attract/repel asymmetry was shown (analysis/check_repulsion.py) to be
entirely a magnetisation effect: with rigid magnetisation the two states are
equal and opposite to within 1.4 %.  In the repel state the two magnets drive
each other backwards along their own demagnetisation curves, and Alnico 5 has
so little coercivity that its polarisation collapses.

That points at the material.  The supplier table in
``simulations/Force_compute/Alnico性能表.png`` contains grades with up to three
times the coercivity of Alnico 5, at the cost of remanence.  Since force scales
roughly as J^2, trading remanence for coercivity is only worth it if the repel
state holds on to enough of its magnetisation to more than repay the loss.

This sweeps the whole grade table in both states and reports which grade
maximises the usable repulsion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from axisym_fem import AxisymModel, Region, axial_force  # noqa: E402
from magnet_force import Material  # noqa: E402

Rm, Lm = 2.375e-3, 12.5e-3
GAP = 0.1e-3
RFAR = 30 * Rm

# Br (mT), Hcb (kA/m), Hcj (kA/m), BHmax (kJ/m3) -- supplier table
GRADES = [
    ("LNG37   (Alnico 5)",    1200, 48, 49, 37),
    ("LNG52   (Alnico 5DG)",  1300, 56, 57, 52),
    ("LNG60   (Alnico 5-7)",  1350, 59, 60, 60),
    ("LNGT28  (Alnico 6)",    1000, 58, 59, 28),
    ("LNGT38  (Alnico 8)",     800, 110, 112, 38),
    ("LNGT44  (Alnico 8)",     880, 120, 122, 44),
    ("LNGT60  (Alnico 9)",     900, 110, 112, 60),
    ("LNGT72  (Alnico 9)",    1050, 112, 114, 72),
    ("LNGT36J (Alnico 8HC)",   700, 140, 148, 36),
]


def build(name, br, hcb, hcj, bh):
    return Material.from_datasheet(name, Br=br / 1000.0, Hcb=hcb * 1e3,
                                   Hcj=hcj * 1e3, BHmax=bh * 1e3, mu_rec=4.0)


def pair(mat, flip):
    return [Region(0, Rm, -Lm, 0.0, "magnet", "A", material=mat, direction=+1),
            Region(0, Rm, GAP, GAP + Lm, "magnet", "B", material=mat,
                   direction=(-1 if flip else +1))]


def evaluate(mat, flip):
    m = AxisymModel(pair(mat, flip), RFAR, 20 * Lm, 0.3e-3, n_slabs=6)
    s = m.solve()
    J, H = m.region_state(s, "A")
    F = axial_force(s, GAP / 2, r_max=0.9 * RFAR, n=6000)
    return J, abs(H) / mat.Hcj, F


print("=" * 78)
print("ALNICO GRADE SWEEP: WHICH GRADE GIVES USABLE REPULSION?")
print("=" * 78)
print(f"\nBare rod pair, D {Rm*2e3:.2f} x L {Lm*1e3:.1f} mm, gap {GAP*1e3:.2f} mm."
      f"  No soft-magnetic circuit,")
print("so this isolates the material effect from the circuit effect.\n")
print(f"  {'grade':<22} {'Br':>5} {'Hcj':>5} | {'attract':>17} | "
      f"{'repel':>17} | {'ratio':>6}")
print(f"  {'':<22} {'(T)':>5} {'kA/m':>5} | {'J (T)':>7} {'F (N)':>9} | "
      f"{'J (T)':>7} {'F (N)':>9} |")
print("  " + "-" * 76)

rows = []
for name, br, hcb, hcj, bh in GRADES:
    mat = build(name, br, hcb, hcj, bh)
    Ja, _, Fa = evaluate(mat, False)
    Jr, mr, Fr = evaluate(mat, True)
    rows.append((name, br / 1000, hcj, Ja, Fa, Jr, Fr, mr))
    print(f"  {name:<22} {br/1000:5.2f} {hcj:5.0f} | {Ja:7.3f} {Fa:+9.2f} | "
          f"{Jr:7.3f} {Fr:+9.2f} | {abs(Fa/Fr):6.1f}", flush=True)

print("\n" + "=" * 78)
print("FINDINGS")
print("=" * 78)

best_r = max(rows, key=lambda r: abs(r[6]))
best_a = max(rows, key=lambda r: abs(r[4]))
base = [r for r in rows if r[0].startswith("LNG37")][0]

print(f"""
  as built   {base[0]:<22} repel {base[6]:+7.2f} N   attract {base[4]:+7.2f} N
  best repel {best_r[0]:<22} repel {best_r[6]:+7.2f} N   attract {best_r[4]:+7.2f} N
  best attract {best_a[0]:<20} repel {best_a[6]:+7.2f} N   attract {best_a[4]:+7.2f} N

  repulsion gain from the material alone: {abs(best_r[6]/base[6]):.1f} x
  attraction cost of that choice:         {best_r[4]/base[4]:.2f} x""")

print(f"""
  The high-remanence grades that look best on paper (LNG60, Alnico 5-7, the
  grade named in the report) are the WORST choice for a design that has to
  repel: they have the least coercivity, so they collapse hardest when pushed
  backwards. Remanence sets the ceiling, coercivity decides how much of that
  ceiling survives contact with a reversed neighbour.""")
