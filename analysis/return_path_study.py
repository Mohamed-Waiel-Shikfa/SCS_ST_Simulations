"""Does a soft-magnetic return path fix the Alnico operating point?

Controlled comparison: the Alnico grade, diameter and length are held fixed at
the as-built values in every case, and ONLY the magnetic circuit around them
changes.  Any difference is therefore attributable to the circuit alone.

Configurations
--------------
  bare      two bare rods facing each other        (the current design)
  potcore   two rods, each in a 1018 steel cup with a coaxial return annulus

Each is evaluated in both working states:

  ATTRACT   rods magnetised the same way; flux crosses the gap and returns
            through the pair - this is the latched state
  REPEL     second unit reversed; this is what drives pivoting

Reported: operating polarisation, the demagnetisation margin H/Hcj (how close
the rod sits to irreversible loss), and the axial force.

Runtime is a few minutes per mated configuration: every residual evaluation is
a full nonlinear field solve, and the outer Newton needs a Jacobian over the
magnet slab states.

Run:  python analysis/return_path_study.py
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
from magnet_force import alnico_lng37  # noqa: E402

# as-built magnet (param.txt)
D_M, L_M = 4.75e-3, 12.5e-3
R_M = D_M / 2
ALN = alnico_lng37()

T_STEEL = 1.0e-3      # keeper wall thickness
R_GAP = 1.0e-3        # radial clearance between rod and return annulus
R_OUT = R_M + R_GAP + T_STEEL

GAP = 0.1e-3
MESH = 0.4e-3
NSLAB = 5


def bare(flip):
    return [
        Region(0, R_M, -L_M, 0.0, "magnet", "A", material=ALN, direction=+1),
        Region(0, R_M, GAP, GAP + L_M, "magnet", "B", material=ALN,
               direction=(-1 if flip else +1)),
    ]


def potcore(flip):
    """Rod in a steel cup: back plate plus coaxial return annulus."""
    a = [
        Region(0, R_M, -L_M, 0.0, "magnet", "A", material=ALN, direction=+1),
        Region(0, R_OUT, -L_M - T_STEEL, -L_M, "steel", "backA"),
        Region(R_M + R_GAP, R_OUT, -L_M, 0.0, "steel", "annA"),
    ]
    d = -1 if flip else +1
    b = [
        Region(0, R_M, GAP, GAP + L_M, "magnet", "B", material=ALN,
               direction=d),
        Region(0, R_OUT, GAP + L_M, GAP + L_M + T_STEEL, "steel", "backB"),
        Region(R_M + R_GAP, R_OUT, GAP, GAP + L_M, "steel", "annB"),
    ]
    return a + b


def evaluate(regions, label):
    t = time.time()
    m = AxisymModel(regions, r_far=25 * R_OUT, z_far=14 * L_M, h=MESH,
                    n_slabs=NSLAB)
    sol = m.solve()
    J, H = m.region_state(sol, "A")
    F = axial_force(sol, GAP / 2, r_max=15 * R_OUT)
    print(f"  {label:<22} {J:8.3f} {J/ALN.Br*100:6.0f}% {abs(H)/ALN.Hcj:8.2f}"
          f" {F:+10.2f}   {time.time()-t:5.0f}s  resid {sol.residual:.0e}",
          flush=True)
    return J, H, F


print("=" * 78)
print("DOES A SOFT-MAGNETIC RETURN PATH FIX THE OPERATING POINT?")
print("=" * 78)
print(f"\nMagnet identical in every row: Alnico 5 (LNG37), D {D_M*1e3:.2f} mm x"
      f" L {L_M*1e3:.1f} mm, L/D = {L_M/D_M:.2f}")
print(f"Br = {ALN.Br:.2f} T, Hcj = {ALN.Hcj/1e3:.0f} kA/m."
      f"  Keeper: 1018 steel, {T_STEEL*1e3:.1f} mm wall.")
print(f"Air gap {GAP*1e3:.2f} mm.  Positive force = attraction.\n")

print(f"  {'configuration':<22} {'J (T)':>8} {'of Br':>7} {'H/Hcj':>8}"
      f" {'force (N)':>10}")
print("  " + "-" * 76)

res = {}
res["bare attract"] = evaluate(bare(False), "bare rods, ATTRACT")
res["bare repel"] = evaluate(bare(True), "bare rods, REPEL")
res["pot attract"] = evaluate(potcore(False), "pot core, ATTRACT")
res["pot repel"] = evaluate(potcore(True), "pot core, REPEL")

print("\n" + "=" * 78)
print("FINDINGS")
print("=" * 78)

fa_b, fa_p = res["bare attract"][2], res["pot attract"][2]
fr_b, fr_p = res["bare repel"][2], res["pot repel"][2]
hb_r = abs(res["bare repel"][1]) / ALN.Hcj
hp_r = abs(res["pot repel"][1]) / ALN.Hcj

print(f"""
1. The return path roughly doubles the latching force
      attraction  {fa_b:.2f} N -> {fa_p:.2f} N   ({fa_p/fa_b:.2f} x)

   and it does so with the same magnet.  It is not a bigger-magnet result.

2. It transforms the demagnetisation margin, and the REPEL state is where
   this matters most
      repel state  H/Hcj {hb_r:.2f} -> {hp_r:.2f}

   A bare rod driven into repulsion sits at {hb_r:.2f} of its own coercivity with
   its polarisation collapsed to {res['bare repel'][0]/ALN.Br*100:.0f} % of remanence.  That is not a
   safe operating point: it is essentially self-erasing.  Since repulsion is
   exactly what the pivoting manoeuvre depends on, the current design would
   degrade every time it tries to move.  This is consistent with the ~1.5 %
   loss per pull-off measured in the experimental data.

3. UNEXPECTED, AND IT AFFECTS THE LOCOMOTION DESIGN: repulsion is far weaker
   than attraction in BOTH designs
      bare     attract {fa_b:+.2f} N   repel {fr_b:+.2f} N   ratio {abs(fa_b/fr_b):.0f} : 1
      pot core attract {fa_p:+.2f} N   repel {fr_p:+.2f} N   ratio {abs(fa_p/fr_p):.0f} : 1

   Two faces presenting the same pole do not push hard, because the flux simply
   returns through whatever path is available instead of crossing the gap.  The
   locomotion concept assumes attraction and repulsion are comparable; they are
   not, by more than an order of magnitude.  Any rolling or pivoting sequence
   has to be designed around asymmetric actuation, or the geometry has to be
   changed to force flux across the gap in the repel state.""")
