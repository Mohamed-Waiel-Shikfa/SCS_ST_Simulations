"""Verify the dynamics force model.

The model that drives MuJoCo must (a) reproduce the Stage 1 FEM where the FEM
applies, (b) fall off with distance the way the FEM does, (c) obey Newton's
third law, and (d) conserve energy.  The last one is not optional: an earlier
version rescaled a charge-disc wrench by a pose-dependent factor, which is not
the gradient of anything, and it pumped enough energy into a rolling module to
carry it over four gravitational barriers it could not otherwise cross.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dynamics import make_spec, pair_wrench  # noqa: E402
from framework import Design, stage1_magnetics  # noqa: E402
from module import build_module  # noqa: E402

BASE = dict(material="LNGT72", circuit="potcore", v_cap=70.0,
            n_gon=8, r_face=19.4e-3, d_mag=4.2e-3, l_mag=8.4e-3)
OK = True

print("=" * 76)
print("DYNAMICS FORCE MODEL")
print("=" * 76)

d = Design(**BASE)
mag = stage1_magnetics(d, fidelity="screen")
mod = build_module(d, None)
spec = make_spec(d, mag)

IX_P = int(np.argmax(mod.normals @ np.array([1.0, 0, 0])))
IX_M = int(np.argmin(mod.normals @ np.array([1.0, 0, 0])))
print(f"\n  module: n={d.n_gon}, {mod.n_faces} faces, "
      f"r_face {mod.r_face*1e3:.2f} mm, mass {mod.mass*1e3:.1f} g")
print(f"  mating pair: A face {IX_P} <-> B face {IX_M}")

quat = np.array([1.0, 0.0, 0.0, 0.0])
posA = np.array([0.0, 0.0, 0.0])


def axial(gap, mode):
    posB = np.array([2 * mod.r_face + gap, 0.0, 0.0])
    return pair_wrench(posA, quat, posB, quat, mod, spec,
                       [(IX_P, IX_M, mode)])


print(f"\n[1] Calibration point (gap {d.gap*1e3:.2f} mm)\n")
for mode, ref in (("attract", mag["F_attract"]), ("repel", mag["F_repel"])):
    F, T = axial(d.gap, mode)
    err = abs(abs(F[0]) - abs(ref)) / abs(ref) * 100
    good = err < 1.0
    OK &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {mode:<8} model {F[0]:+8.3f} N   "
          f"FEM {ref:8.3f} N   {err:.2f}% err")

print("\n[2] Does the model fall off like the FEM?\n")
print(f"  {'gap (mm)':>9} {'FEM (N)':>9} {'model (N)':>10} {'FEM rel':>9}"
      f" {'model rel':>10}")
f_fem0 = f_mod0 = None
worst = 0.0
for gap_mm in (0.1, 0.3, 0.6, 1.0, 2.0):
    gg = gap_mm * 1e-3
    m = stage1_magnetics(Design(gap=gg, **BASE), fidelity="screen")
    F, _ = axial(gg, "attract")
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

print("\n[3] Torque sign on a tilted neighbour\n")
ang = np.radians(20.0)
qtilt = np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])
posB = np.array([2 * mod.r_face + d.gap, 0.0, 0.0])
F, T = pair_wrench(posA, quat, posB, qtilt, mod, spec,
                   [(IX_P, IX_M, "attract")])
good = -T[1] > 0
OK &= good
print(f"  [{'ok ' if good else 'FAIL'}] tilt +20 deg about y -> torque_y "
      f"{T[1]:+.5f} N m ({'restoring' if good else 'destabilising'})")

print("\n[4] Newton's third law\n")
FA, _ = pair_wrench(posA, quat, posB, quat, mod, spec,
                    [(IX_P, IX_M, "attract")])
FB, _ = pair_wrench(posB, quat, posA, quat, mod, spec,
                    [(IX_M, IX_P, "attract")])
resid = np.linalg.norm(FA + FB) / max(np.linalg.norm(FA), 1e-12)
good = resid < 1e-9
OK &= good
print(f"  [{'ok ' if good else 'FAIL'}] |F_A + F_B| / |F_A| = {resid:.2e}")

print("\n[5] The force field must be conservative")
print("    Work around a closed loop in configuration space must vanish.")
print("    A path that separates, rotates, comes back and unrotates returns")
print("    the module to its starting pose, so a model that is the gradient")
print("    of a potential can do no net work on it.\n")


def pose(t):
    """Closed loop: out along x, tilt, back along x, untilt."""
    g = d.gap + 3e-3 * (1 - np.cos(2 * np.pi * t))
    a = np.radians(25.0) * np.sin(2 * np.pi * t)
    p = np.array([2 * mod.r_face + g, 0.0, 0.0])
    q = np.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])
    return p, q


N = 4000
W = 0.0
p0, q0 = pose(0.0)
for i in range(N):
    t0, t1 = i / N, (i + 1) / N
    pa, qa = pose(t0)
    pb, qb = pose(t1)
    Fa, Ta = pair_wrench(posA, quat, pa, qa, mod, spec,
                         [(IX_P, IX_M, "attract")])
    Fb, Tb = pair_wrench(posA, quat, pb, qb, mod, spec,
                         [(IX_P, IX_M, "attract")])
    dx = pb - pa
    # small rotation vector between qa and qb
    da = 2.0 * np.array([qb[1] - qa[1], qb[2] - qa[2], qb[3] - qa[3]])
    W += 0.5 * float((Fa + Fb) @ dx) + 0.5 * float((Ta + Tb) @ da)

scale = abs(mag["F_attract"]) * 3e-3          # a representative work scale
good = abs(W) / scale < 1e-3
OK &= good
print(f"  [{'ok ' if good else 'FAIL'}] loop work {W*1e6:+.3f} uJ, "
      f"{abs(W)/scale*100:.4f} % of the {scale*1e3:.2f} mJ work scale")

print("\n" + ("DYNAMICS MODEL VERIFIED" if OK else "DYNAMICS MODEL FAILED"))
raise SystemExit(0 if OK else 1)
