"""Verify the dynamics force model against the Stage 1 FEM.

The charge-disc model used in MuJoCo must reproduce the FEM it was calibrated
against, and must fall off with distance the same way - a model calibrated at
one gap that has the wrong gradient will give wrong dynamics.

Rewritten for the n-gon module geometry (three orthogonal regular n-gon
prisms intersected, 3n-6 square faces).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dynamics import magnetic_wrenches, make_spec  # noqa: E402
from framework import Design, stage1_magnetics  # noqa: E402
from module import build_module  # noqa: E402

BASE = dict(material="LNGT72", circuit="potcore", v_cap=70.0,
            n_gon=8, r_face=19.4e-3, d_mag=4.2e-3, l_mag=8.4e-3)
OK = True

print("=" * 76)
print("DYNAMICS FORCE MODEL vs STAGE 1 FEM")
print("=" * 76)

d = Design(**BASE)
mag = stage1_magnetics(d, fidelity="screen")
mod = build_module(d, None)
spec = make_spec(d, mag)

N = mod.n_faces
IX_P = int(np.argmax(mod.normals @ np.array([1.0, 0, 0])))   # face facing +x
IX_M = int(np.argmin(mod.normals @ np.array([1.0, 0, 0])))   # face facing -x
print(f"\n  module: n={d.n_gon}, {N} faces, r_face {mod.r_face*1e3:.2f} mm, "
      f"mass {mod.mass*1e3:.1f} g")
print(f"  mating pair: A face {IX_P} {mod.normals[IX_P]}  <->  "
      f"B face {IX_M} {mod.normals[IX_M]}")


def state(idx, val):
    s = [0] * N
    s[idx] = val
    return s


quat = np.array([1.0, 0.0, 0.0, 0.0])
posA = np.array([0.0, 0.0, 0.0])
A_ON = state(IX_P, +1)


def axial(gap, stB):
    posB = np.array([2 * mod.r_face + gap, 0.0, 0.0])
    return magnetic_wrenches(posA, quat, A_ON, posB, quat, stB, mod, spec)


print(f"\n[1] Calibration point (gap {d.gap*1e3:.2f} mm)\n")
for lab, stB, ref in (("attract", state(IX_M, -1), mag["F_attract"]),
                      ("repel", state(IX_M, +1), mag["F_repel"])):
    F, T = axial(d.gap, stB)
    err = abs(abs(F[0]) - abs(ref)) / abs(ref) * 100
    good = err < 5.0
    OK &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {lab:<8} model {F[0]:+8.3f} N   "
          f"FEM {ref:8.3f} N   {err:.1f}% err")

print("\n[2] Does the model fall off like the FEM?")
print("    (both normalised to their own value at the calibration gap)\n")
print(f"  {'gap (mm)':>9} {'FEM (N)':>9} {'model (N)':>10} {'FEM rel':>9}"
      f" {'model rel':>10}")
f_fem0 = f_mod0 = None
worst = 0.0
for gap_mm in (0.1, 0.3, 0.6, 1.0, 2.0):
    gg = gap_mm * 1e-3
    m = stage1_magnetics(Design(gap=gg, **BASE), fidelity="screen")
    F, _ = axial(gg, state(IX_M, -1))
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
posB = np.array([2 * mod.r_face + d.gap, 0.0, 0.0])
F, T = magnetic_wrenches(posA, quat, A_ON, posB, qtilt,
                         state(IX_M, -1), mod, spec)
restoring = -T[1]
good = restoring > 0
OK &= good
print(f"  [{'ok ' if good else 'FAIL'}] tilt +20 deg about y -> torque_y "
      f"{T[1]:+.4f} N m ({'restoring' if good else 'destabilising'})")

print("\n[4] Newton's third law: wrench on A must oppose wrench on B\n")
FA, _ = magnetic_wrenches(posA, quat, A_ON, posB, quat, state(IX_M, -1),
                          mod, spec)
FB, _ = magnetic_wrenches(posB, quat, state(IX_M, -1), posA, quat, A_ON,
                          mod, spec)
resid = np.linalg.norm(FA + FB) / max(np.linalg.norm(FA), 1e-12)
good = resid < 1e-9
OK &= good
print(f"  [{'ok ' if good else 'FAIL'}] |F_A + F_B| / |F_A| = {resid:.2e}")

print("\n[5] Off-state faces must contribute nothing\n")
Foff, Toff = magnetic_wrenches(posA, quat, [0] * N, posB, quat, [0] * N,
                               mod, spec)
good = np.linalg.norm(Foff) < 1e-12 and np.linalg.norm(Toff) < 1e-12
OK &= good
print(f"  [{'ok ' if good else 'FAIL'}] all-off |F| = "
      f"{np.linalg.norm(Foff):.2e} N")

print("\n" + ("DYNAMICS MODEL VERIFIED" if OK else "DYNAMICS MODEL FAILED"))
raise SystemExit(0 if OK else 1)
