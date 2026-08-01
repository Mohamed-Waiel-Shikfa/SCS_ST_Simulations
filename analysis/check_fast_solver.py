"""Cross-check the fast exact solver against the FEM, including the repel state.

The FEM is accurate but far too slow to drive an optimiser (seconds to minutes
per evaluation).  The free-space solver is milliseconds.  Before using it for a
large material sweep, confirm it reproduces the FEM in both working states.

FEM reference values are from analysis/combined_fix.py.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from magnet_force import CoaxialRodPair, Material  # noqa: E402

R, L, GAP = 2.375e-3, 12.5e-3, 0.1e-3

MATS = {
    "LNG37": (1.20, 48e3, 49e3, 37e3),
    "LNGT72": (1.05, 112e3, 114e3, 72e3),
}
FEM = {("LNG37", "ATTRACT"): 5.96, ("LNG37", "REPEL"): -0.28,
       ("LNGT72", "ATTRACT"): 6.63, ("LNGT72", "REPEL"): -1.24}

print("Fast free-space solver vs nonlinear FEM, bare rod pair, 0.1 mm gap")
print(f"  {'grade':>8} {'state':>8} {'exact (N)':>10} {'FEM (N)':>9} {'diff':>8}")

t0 = time.time()
n = 0
for name, (br, hcb, hcj, bh) in MATS.items():
    mat = Material.from_datasheet(name, Br=br, Hcb=hcb, Hcj=hcj, BHmax=bh,
                                  mu_rec=4.0)
    for state, o in (("ATTRACT", +1), ("REPEL", -1)):
        p = CoaxialRodPair(R, L, mat, n_slabs=16).set_orientation(+1, o)
        f = -p.force(GAP)          # sign: positive = attraction
        n += 1
        ref = FEM[(name, state)]
        print(f"  {name:>8} {state:>8} {f:10.2f} {ref:9.2f} "
              f"{abs(f-ref)/abs(ref)*100:7.1f}%")

dt = time.time() - t0
print(f"\n  {n} evaluations in {dt:.2f} s  ->  {dt/n*1000:.0f} ms each")
print("  (the FEM took 1-565 s per evaluation for the same cases)")

print("""
  VERDICT: agreement is 1-9 % except LNGT72 ATTRACT, which is 20 % low - and
  critically the fast solver ranks LNG37 above LNGT72 on attraction while the
  FEM ranks them the other way round.  A model that reverses a ranking cannot
  be used to choose a material.

  Cause is the known sub-slab difference documented in verify_fem.py: the
  free-space solver holds J uniform across each axial slab, while the FEM lets
  it vary radially through the recoil permeability.  The discrepancy therefore
  grows with mu_rec, and mu_rec = 4.0 was assumed here for every grade, which
  is right for Alnico 5 but too high for the more anisotropic Alnico 8/9.

  So: the material sweep uses the FEM.  A bare rod pair costs only a couple of
  seconds there, and only the steel-circuit geometries are slow.  The fast
  solver stays in use for the experimental-data fit and for geometry sweeps at
  fixed material, where it is validated.""")
