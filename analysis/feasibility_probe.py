"""Is the feasible region non-empty?  Hand-reason a design and check.

Before running an optimiser it is worth confirming by construction that at
least one design satisfies every constraint.  If none does, the search will
burn its whole budget proving a negative and the right response is to revisit
a requirement rather than to run longer.
"""
import sys
sys.path.insert(0, "analysis")
import numpy as np
from framework import Design, evaluate, HOLD_MIN, PIVOT_MIN, MARGIN_LIMIT
from driver import select_driver
from framework import stage3_switching
from module import build_module

print("=" * 92)
print("FEASIBILITY PROBE: can ANY design satisfy every constraint?")
print("=" * 92)
print(f"\nconstraints: margin <= {MARGIN_LIMIT}, hold >= {HOLD_MIN}, "
      f"pivot >= {PIVOT_MIN}, cube <= 50 mm, electronics fit\n")

# Reasoning: hold_ratio = F_attract / (m g).  Mass grows with face count and
# magnet volume, force only with pole area, so few faces and a big pole is the
# way in.  n=8 gives 18 faces and the largest face (16.6 mm at r=20 mm).
# Alnico 9 for repulsion, potcore for the margin.
CANDIDATES = [
    ("n=8, L/D=0.7 (very squat)",
     dict(n_gon=8, r_face=20e-3, d_mag=9e-3, l_mag=6e-3, t_steel=1.0e-3,
          r_clear=0.6e-3, material="LNGT72", circuit="potcore", v_cap=90.0)),
    ("n=8, L/D=1.1",
     dict(n_gon=8, r_face=20e-3, d_mag=8e-3, l_mag=9e-3, t_steel=1.2e-3,
          r_clear=0.6e-3, material="LNGT72", circuit="potcore", v_cap=90.0)),
    ("n=8, L/D=1.5",
     dict(n_gon=8, r_face=20e-3, d_mag=7e-3, l_mag=10.5e-3, t_steel=1.2e-3,
          r_clear=0.6e-3, material="LNGT72", circuit="potcore", v_cap=90.0)),
    ("n=8, L/D=2.0",
     dict(n_gon=8, r_face=20e-3, d_mag=6e-3, l_mag=12e-3, t_steel=1.2e-3,
          r_clear=0.6e-3, material="LNGT72", circuit="potcore", v_cap=90.0)),
    ("n=8, L/D=1.5, thick keeper",
     dict(n_gon=8, r_face=20e-3, d_mag=6.5e-3, l_mag=10e-3, t_steel=2.0e-3,
          r_clear=0.5e-3, material="LNGT72", circuit="potcore", v_cap=90.0)),
    ("n=8, L/D=1.5, Alnico 8HC",
     dict(n_gon=8, r_face=20e-3, d_mag=7e-3, l_mag=10.5e-3, t_steel=1.2e-3,
          r_clear=0.6e-3, material="LNGT36J", circuit="potcore", v_cap=140.0)),
    ("n=12, L/D=1.5",
     dict(n_gon=12, r_face=20e-3, d_mag=5e-3, l_mag=7.5e-3, t_steel=1.0e-3,
          r_clear=0.5e-3, material="LNGT72", circuit="potcore", v_cap=90.0)),
    ("n=8, smaller module, L/D=1.4",
     dict(n_gon=8, r_face=15e-3, d_mag=6e-3, l_mag=8.5e-3, t_steel=1.0e-3,
          r_clear=0.5e-3, material="LNGT72", circuit="potcore", v_cap=90.0)),
]

for label, kw in CANDIDATES:
    d = Design(**kw)
    r = evaluate(d, fidelity="screen")
    ok = "FEASIBLE" if r["feasible"] else "no"
    mod = build_module(d)
    print(f"  {label:<30} {ok:>9}  Fa={r['F_attract']:6.2f} "
          f"Fr={r['F_repel']:5.2f} asym={r['asymmetry']:5.1f} "
          f"marg={r['margin'] if np.isfinite(r['margin']) else -1:.2f} "
          f"m={r['m_module']*1e3:4.0f}g hold={r['hold_ratio']:6.1f}")
    if r["violations"]:
        print(f"  {'':<30}           {r['violations'][:80]}")

print("""
  Two constraints pull in opposite directions.  Attraction follows pole AREA
  and module mass follows face count times magnet VOLUME, so the hold
  requirement wants short wide magnets.  The demagnetisation margin wants the
  opposite: a squat magnet has a large demagnetising factor and sits closer to
  its own coercivity.  The feasible region, if it exists, is the band where a
  closed steel circuit has pulled the margin down far enough that a
  moderately squat magnet still survives - which is why the circuit is not
  optional.""")
