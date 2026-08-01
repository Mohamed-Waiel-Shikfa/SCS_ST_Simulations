"""Verify the dynamics force model against the Stage 1 FEM.

The charge-disc model used in MuJoCo must reproduce the FEM it was calibrated
against, and must fall off with distance the same way - a model calibrated at
one gap that has the wrong gradient will give wrong dynamics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dynamics import (MU0, face_charges, magnetic_wrenches,  # noqa: E402
                      make_spec)
from framework import Design, stage1_magnetics  # noqa: E402
from module import build_module  # noqa: E402

BASE = dict(material="LNGT72", circuit="potcore", v_cap=70.0, a_module=60e-3)
OK = True

print("=" * 76)
print("DYNAMICS FORCE MODEL vs STAGE 1 FEM")
print("=" * 76)

d = Design(**BASE)
mag = stage1_magnetics(d, fidelity="screen")
mod, _, _ = build_module(d, None)
spec = make_spec(d, mag)

quat = np.array([1.0, 0.0, 0.0, 0.0])
posA = np.array([0.0, 0.0, 0.0])


def axial(gap, stB):
    posB = np.array([mod.a + gap, 0.0, 0.0])
    F, T = magnetic_wrenches(posA, quat, [1, 0, 0, 0, 0, 0], posB, quat, stB,
                             mod, spec)
    return F, T


print(f"\n[1] Calibration point (gap {d.gap*1e3:.2f} mm)\n")
for lab, stB, ref in (("attract", [0, -1, 0, 0, 0, 0], mag["F_attract"]),
                      ("repel", [0, +1, 0, 0, 0, 0], -mag["F_repel"])):
    F, T = axial(d.gap, stB)
    err = abs(abs(F[0]) - abs(ref)) / abs(ref) * 100
    good = err < 5.0
    OK &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {lab:<8} model {F[0]:+8.3f} N   "
          f"FEM {-ref if lab=='attract' else -ref:+8.3f} N   {err:.1f}% err")

print("\n[2] Does the model fall off like the FEM?")
print("    (both normalised to their own value at the calibration gap)\n")
print(f"  {'gap (mm)':>9} {'FEM (N)':>9} {'model (N)':>10} {'FEM rel':>9}"
      f" {'model rel':>10}")
f_fem0 = f_mod0 = None
worst = 0.0
for gap_mm in (0.1, 0.3, 0.6, 1.0, 2.0):
    gg = gap_mm * 1e-3
    m = stage1_magnetics(Design(gap=gg, **BASE), fidelity="screen")
    F, _ = axial(gg, [0, -1, 0, 0, 0, 0])
    if f_fem0 is None:
        f_fem0, f_mod0 = m["F_attract"], abs(F[0])
    rel_fem = m["F_attract"] / f_fem0
    rel_mod = abs(F[0]) / f_mod0
    worst = max(worst, abs(rel_mod - rel_fem))
    print(f"  {gap_mm:9.2f} {m['F_attract']:9.2f} {abs(F[0]):10.2f}"
          f" {rel_fem:9.3f} {rel_mod:10.3f}", flush=True)

good = worst < 0.20
OK &= good
print(f"\n  [{'ok ' if good else 'FAIL'}] worst deviation in normalised "
      f"fall-off: {worst:.3f} (tol 0.20)")

print("\n[3] Torque sign on a tilted neighbour")
print("    A face rotated off-axis must be pulled back into alignment.\n")
ang = np.radians(20.0)
qtilt = np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])
posB = np.array([mod.a + d.gap, 0.0, 0.0])
F, T = magnetic_wrenches(posA, quat, [1, 0, 0, 0, 0, 0], posB, qtilt,
                         [0, -1, 0, 0, 0, 0], mod, spec)
restoring = T[1] * (-1)
good = restoring > 0
OK &= good
print(f"  [{'ok ' if good else 'FAIL'}] tilt +20 deg about y -> torque_y "
      f"{T[1]:+.4f} N m ({'restoring' if good else 'destabilising'})")

print("\n" + ("DYNAMICS MODEL VERIFIED" if OK else "DYNAMICS MODEL FAILED"))
raise SystemExit(0 if OK else 1)
