"""Combine the two fixes: higher-coercivity grade AND a soft-magnetic circuit.

The grade sweep showed Alnico 9 (LNGT72) strictly dominates Alnico 5 (LNG37) in
both working states.  The return-path study showed a steel circuit doubles the
latching force and rescues the demagnetisation margin.  The two act on the same
mechanism - how far the magnet is pushed back down its own curve - so they are
not guaranteed to add.  This measures the combination.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from axisym_fem import AxisymModel, Region, axial_force  # noqa: E402
from magnet_force import Material  # noqa: E402

R_M, L_M = 2.375e-3, 12.5e-3
GAP = 0.1e-3
T_STEEL, R_GAP = 1.0e-3, 1.0e-3
R_OUT = R_M + R_GAP + T_STEEL

LNG37 = Material.from_datasheet("LNG37 (Alnico 5)", Br=1.20, Hcb=48e3,
                                Hcj=49e3, BHmax=37e3, mu_rec=4.0)
LNGT72 = Material.from_datasheet("LNGT72 (Alnico 9)", Br=1.05, Hcb=112e3,
                                 Hcj=114e3, BHmax=72e3, mu_rec=4.0)


def bare(mat, flip):
    return [Region(0, R_M, -L_M, 0.0, "magnet", "A", material=mat,
                   direction=+1),
            Region(0, R_M, GAP, GAP + L_M, "magnet", "B", material=mat,
                   direction=(-1 if flip else +1))]


def potcore(mat, flip):
    d = -1 if flip else +1
    return [
        Region(0, R_M, -L_M, 0.0, "magnet", "A", material=mat, direction=+1),
        Region(0, R_OUT, -L_M - T_STEEL, -L_M, "steel", "backA"),
        Region(R_M + R_GAP, R_OUT, -L_M, 0.0, "steel", "annA"),
        Region(0, R_M, GAP, GAP + L_M, "magnet", "B", material=mat,
               direction=d),
        Region(0, R_OUT, GAP + L_M, GAP + L_M + T_STEEL, "steel", "backB"),
        Region(R_M + R_GAP, R_OUT, GAP, GAP + L_M, "steel", "annB"),
    ]


def evaluate(regions, label, rfar, nslab):
    t = time.time()
    m = AxisymModel(regions, rfar, 20 * L_M, 0.35e-3, n_slabs=nslab)
    s = m.solve()
    J, H = m.region_state(s, "A")
    F = axial_force(s, GAP / 2, r_max=0.9 * rfar, n=6000)
    print(f"  {label:<34} {J:7.3f} {F:+10.2f}   {time.time()-t:5.0f}s",
          flush=True)
    return J, F


print("=" * 78)
print("COMBINED FIX: ALNICO 9 + STEEL RETURN PATH")
print("=" * 78)
print(f"\nSame rod geometry throughout: D {R_M*2e3:.2f} x L {L_M*1e3:.1f} mm,"
      f" gap {GAP*1e3:.2f} mm\n")
print(f"  {'configuration':<34} {'J (T)':>7} {'force (N)':>10}")
print("  " + "-" * 62)

r = {}
r["a"] = evaluate(bare(LNG37, False), "LNG37 bare,      ATTRACT", 30 * R_M, 6)
r["b"] = evaluate(bare(LNG37, True), "LNG37 bare,      REPEL", 30 * R_M, 6)
r["c"] = evaluate(bare(LNGT72, False), "LNGT72 bare,     ATTRACT", 30 * R_M, 6)
r["d"] = evaluate(bare(LNGT72, True), "LNGT72 bare,     REPEL", 30 * R_M, 6)
r["e"] = evaluate(potcore(LNGT72, False), "LNGT72 pot core, ATTRACT",
                  25 * R_OUT, 5)
r["f"] = evaluate(potcore(LNGT72, True), "LNGT72 pot core, REPEL",
                  25 * R_OUT, 5)

print("\n" + "=" * 78)
print("FINDINGS")
print("=" * 78)
print(f"""
                        attract      repel     asymmetry
  as built (LNG37 bare) {r['a'][1]:+8.2f} N {r['b'][1]:+9.2f} N   {abs(r['a'][1]/r['b'][1]):6.1f} : 1
  grade change only     {r['c'][1]:+8.2f} N {r['d'][1]:+9.2f} N   {abs(r['c'][1]/r['d'][1]):6.1f} : 1
  grade + steel circuit {r['e'][1]:+8.2f} N {r['f'][1]:+9.2f} N   {abs(r['e'][1]/r['f'][1]):6.1f} : 1

  attraction   {r['a'][1]:.2f} -> {r['e'][1]:.2f} N   ({r['e'][1]/r['a'][1]:.1f} x)
  repulsion    {abs(r['b'][1]):.2f} -> {abs(r['f'][1]):.2f} N   ({abs(r['f'][1]/r['b'][1]):.1f} x)
""")
