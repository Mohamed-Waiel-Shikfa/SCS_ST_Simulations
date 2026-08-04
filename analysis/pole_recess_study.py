"""How much force does a shell wall across the pole face cost?

The module shell has to hold the magnets, but any plastic left in front of a
pole face adds to the magnetic gap TWICE over when two modules mate - once for
each module.  This measures the penalty directly with the Stage 1 FEM, because
it decides whether the pole must be an exposed aperture or can be covered.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from framework import Design, stage1_magnetics  # noqa: E402

BASE = dict(material="LNGT72", circuit="potcore", v_cap=70.0)

print("=" * 74)
print("COST OF RECESSING THE POLE FACE BEHIND THE SHELL WALL")
print("=" * 74)
print("""
Two mated modules each contribute their own recess, so a wall of thickness t
produces a magnetic gap of (contact gap + 2t).
""")
print(f"  {'wall (mm)':>10} {'mag gap (mm)':>13} {'attract (N)':>12}"
      f" {'repel (N)':>10} {'% of flush':>11}")
print("  " + "-" * 62)

f0 = None
for wall_mm in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
    gap = 0.1e-3 + 2 * wall_mm * 1e-3
    d = Design(gap=gap, **BASE)
    m = stage1_magnetics(d, fidelity="screen")
    if f0 is None:
        f0 = m["F_attract"]
    print(f"  {wall_mm:10.2f} {gap*1e3:13.2f} {m['F_attract']:12.2f}"
          f" {m['F_repel']:10.2f} {m['F_attract']/f0*100:10.0f}%", flush=True)

print("""
  A 2 mm shell wall across the pole - an unremarkable printed thickness - would
  leave a fraction of the force.  The pole face must be an open aperture in the
  shell, with the magnet or its pole piece flush with the outer surface.  This
  also sets the manufacturing tolerance that matters most: it is the flatness
  and protrusion of the pole face, not the overall module dimension.""")
