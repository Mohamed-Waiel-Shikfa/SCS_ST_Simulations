"""Run the pivot manoeuvre and compare it with the Stage 2 static estimate.

Stage 2 answers the question statically: the magnetic work available over the
roll against the gravitational barrier of lifting the centre of mass from the
face radius to the vertex radius.  That model assumes both driving faces
deliver their full force through the whole arc, which they cannot - the faces
separate as the module rotates and the force falls off steeply with gap.

This runs the manoeuvre in MuJoCo on the real polyhedron so the module has to
get its centre of mass over a real edge against a real contact, with the
magnetic wrench recomputed from the actual pose at every step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from driver import select_driver  # noqa: E402
from dynamics import make_spec  # noqa: E402
from framework import (Design, stage1_magnetics, stage2_mechanics,  # noqa: E402
                       stage3_switching)
from module import build_module, pivot_angle  # noqa: E402
from pivot import hull_vertices, run_pivot  # noqa: E402
from verify_best import from_csv  # noqa: E402

DESIGNS = {
    "baseline (LNG37 bare rod)": dict(
        material="LNG37", circuit="none", n_gon=8, r_face=19.4e-3,
        d_mag=4.75e-3, l_mag=12.5e-3, t_steel=0.5e-3, r_clear=0.0,
        gap=0.1e-3, wire_d=0.3e-3, v_cap=30.0, c_cap=10e-6),
}
CSV = HERE / "ga_front.csv"
if CSV.exists():
    for kw, _ in from_csv(CSV, "scalar", True, 1):
        DESIGNS["GA best scalar"] = kw
    for kw, _ in from_csv(CSV, "m_module", False, 1):
        DESIGNS["GA lightest"] = kw
    for kw, _ in from_csv(CSV, "pivot_ratio", True, 1):
        DESIGNS["GA best pivot margin"] = kw

print("=" * 84)
print("PIVOT MANOEUVRE: static estimate vs simulated")
print("=" * 84)

V = hull_vertices(8, 19.4e-3)
box = np.abs(V).max(axis=0)
print(f"\n  n=8 hull: {len(V)} vertices, bounding half-box "
      f"{box[0]*1e3:.2f} / {box[1]*1e3:.2f} / {box[2]*1e3:.2f} mm "
      f"(r_face 19.40 mm),  circumradius "
      f"{np.linalg.norm(V, axis=1).max()*1e3:.2f} mm")

for label, kw in DESIGNS.items():
    d = Design(**kw)
    mag = stage1_magnetics(d, fidelity="normal")
    sw = stage3_switching(d)
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=d.n_faces)
    mod = build_module(d, drv if drv.feasible else None)
    mech = stage2_mechanics(d, mag, mod=mod)
    spec = make_spec(d, mag, fidelity="screen")

    target = np.degrees(pivot_angle(d.n_gon))
    # torque of gravity resisting the tip, about the leading bottom edge
    tau_g = mod.mass * 9.81 * (mod.a_face / 2)
    print(f"\n{'-'*84}\n{label}")
    print(f"  mass {mod.mass*1e3:.0f} g,  Fa {mag['F_attract']:.2f} N,  "
          f"Fr {mag['F_repel']:.2f} N,  target pivot {target:.0f} deg")
    print(f"  Stage 2 static:  barrier {mech['E_barrier']*1e3:.2f} mJ,  "
          f"drive work {mech['W_drive']*1e3:.1f} mJ,  ratio "
          f"{mech['pivot_ratio']:.0f}")
    print(f"  tipping torque needed about the leading edge: "
          f"{tau_g*1e3:.2f} mN m;  repulsion at centre height offers "
          f"{mag['F_repel']*mod.r_face*1e3:.2f} mN m")

    for mode in ("repel", "reach"):
        tr = run_pivot(mod, spec, seconds=0.6, drive=mode)
        peak = max(abs(t["ang"]) for t in tr)
        settled = abs(tr[-1]["ang"])
        zmax = max(t["z"] for t in tr)
        dx = (tr[-1]["x"] - tr[0]["x"]) * 1e3
        verdict = ("PIVOTED" if settled > 0.7 * target
                   else ("rocked back" if peak > 5 else "no motion"))
        if settled > 1.5 * target:
            verdict = f"OVERSHOT ({settled/target:.1f} steps)"
        print(f"    drive={mode:<6} peak {peak:5.1f} deg, settled "
              f"{settled:5.1f} deg ({settled/target*100:3.0f} % of target), "
              f"rise {(zmax-mod.r_face)*1e3:+.2f} mm, moved {dx:+6.1f} mm"
              f"   -> {verdict}")
        if mode == "repel":
            t_step = next((t["t"] for t in tr if abs(t["ang"]) >= target),
                          None)
            if t_step:
                print(f"                   reaches one step at "
                      f"{t_step*1e3:.0f} ms, so the drive must be cut within "
                      f"that to stop cleanly")

print("\n  'reach' - energising the NEXT face pair round the ring to pull the")
print("  module over - does nothing at all.  That pair starts 11 mm apart at")
print("  n = 8, and the force at 11 mm is three orders below the force at")
print("  contact.  Pivoting has to be driven by REPULSION from the face the")
print("  module is already on; there is no useful 'reach ahead' mode.")

print("\n" + "=" * 84)
print("HOW MUCH n?")
print("A larger polygon lifts the centre of mass less per step, so the barrier")
print("falls; but it also adds faces, drivers and mass.")
print("=" * 84)
print(f"\n{'n':>4}{'faces':>7}{'mass g':>8}{'step':>7}{'lift mm':>9}"
      f"{'barrier uJ':>12}{'W_drive uJ':>12}{'ratio':>8}"
      f"{'sim settled':>13}{'of step':>9}")
base = DESIGNS.get("GA best scalar") or list(DESIGNS.values())[-1]
for n in (8, 12, 16, 20):
    kw = {**base, "n_gon": n}
    d = Design(**kw)
    mag = stage1_magnetics(d, fidelity="normal")
    sw = stage3_switching(d)
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=d.n_faces)
    if not drv.feasible:
        print(f"{n:4d}  driver infeasible for {d.n_faces} faces")
        continue
    mod = build_module(d, drv)
    mech = stage2_mechanics(d, mag, mod=mod)
    spec = make_spec(d, mag, fidelity="screen")
    tgt = np.degrees(pivot_angle(n))
    tr = run_pivot(mod, spec, seconds=0.8, drive="repel")
    settled = abs(tr[-1]["ang"])
    print(f"{n:4d}{d.n_faces:7d}{mod.mass*1e3:8.0f}{tgt:6.1f}d"
          f"{mech['lift']*1e3:9.2f}{mech['E_barrier']*1e6:12.0f}"
          f"{mech['W_drive']*1e6:12.0f}{mech['pivot_ratio']:8.2f}"
          f"{settled:12.1f}d{settled/tgt*100:8.0f}%", flush=True)
