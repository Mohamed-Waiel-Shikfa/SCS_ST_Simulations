"""The whole pipeline on one design, stage by stage, in the order it runs.

    Stage 0  module      the physical assembly
    Stage 1  magnetics   forces, margin, and the circuit the coil will drive
    Stage 2  switching   the transient, on that measured circuit
    Stage 3  mechanics   latching, holding, pivoting - skipped if 2 failed

The point of this script is the ORDER.  Each stage prints what it hands to the
next one, so it is visible that nothing downstream is guessing at something
upstream already computed.

    python analysis/run_pipeline.py
    python analysis/run_pipeline.py --fidelity normal
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from driver import select_driver  # noqa: E402
from framework import (Design, epm_outer_radius, score,  # noqa: E402
                       stage1_magnetics, stage2_switching, stage3_mechanics)
from module import build_module, pivot_angle  # noqa: E402

CASES = [
    ("as built", dict(material="LNG37", circuit="none", r_face=20e-3,
                      d_mag=4.75e-3, l_mag=12.5e-3, wire_d=0.3e-3,
                      n_layers=3, v_cap=30.0, c_cap=10e-6)),
    ("pot core, Alnico 8", dict(material="LNGT44", circuit="potcore",
                                r_face=19.4e-3, d_mag=4.2e-3, l_mag=8.4e-3,
                                t_steel=1.0e-3, r_clear=0.4e-3,
                                wire_d=0.25e-3, n_layers=6, v_cap=120.0,
                                c_cap=47e-6)),
    ("pulse train", dict(material="LNGT44", circuit="potcore",
                         r_face=19.4e-3, d_mag=4.2e-3, l_mag=8.4e-3,
                         t_steel=1.0e-3, r_clear=0.4e-3, wire_d=0.25e-3,
                         n_layers=6, v_cap=120.0, c_cap=47e-6,
                         pulse_mode="train", f_pulse=80e3, duty=0.5,
                         n_pulses=8)),
]


def run(label, kw, fidelity):
    d = Design(**kw)
    t0 = time.time()
    print(f"\n{'=' * 78}\n{label}  ({d.material}, {d.circuit}, "
          f"n={d.n_gon}, {d.pulse_mode})\n{'=' * 78}")

    # ---- Stage 0 --------------------------------------------------------
    mod0 = build_module(d)
    w = d.winding
    print(f"  0 MODULE     {mod0.n_faces} faces, 6 latching, "
          f"{mod0.mass*1e3:.0f} g, cube {d.bounding_cube*1e3:.0f} mm")
    print(f"               winding {w.n_layers} x {w.turns_per_layer} = "
          f"{w.n_turns} turns, build {w.build*1e3:.2f} mm, "
          f"{w.resistance:.3f} ohm")
    print(f"               -> EPM outer radius {epm_outer_radius(d)*1e3:.2f} "
          f"mm sets where the steel starts")

    # ---- Stage 1 --------------------------------------------------------
    mag = stage1_magnetics(d, fidelity=fidelity)
    print(f"  1 MAGNETICS  attract {mag['F_attract']:6.2f} N   "
          f"repel {mag['F_repel']:5.2f} N   "
          f"asym {mag['asymmetry']:4.1f}   margin {mag['margin']:.2f}")
    print(f"               -> n_eff {mag['n_eff']:.3f} measured from the "
          f"field solve; the coil will see "
          f"{(1-mag['n_eff'])*100:.0f} % of its own mmf")

    # ---- Stage 2 --------------------------------------------------------
    sw = stage2_switching(d, n_eff=mag["n_eff"])
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=d.n_faces)
    print(f"  2 SWITCHING  L {sw['L_coil']*1e6:6.1f} uH   "
          f"i_peak {sw['i_peak']:5.1f} A   "
          f"H {sw['h_peak']/1e3:6.0f} of {sw['h_need']/1e3:.0f} kA/m   "
          f"{'SWITCHES' if sw['switched'] else 'FAILS'}")
    print(f"               iron at {sw['b_steel_peak']:.2f} T"
          f"{' (saturated)' if sw['saturated'] else ''}, "
          f"{sw['e_drawn']*1e3:.0f} mJ drawn, "
          f"{sw['e_total_module']*1e3:.0f} mJ for the whole module")
    if drv.feasible:
        print(f"               driver: {drv.bom()[:64]}...  "
              f"{drv.mass*1e3:.0f} g, ${drv.price:.0f}")
    else:
        print(f"               driver: NONE - {drv.notes}")

    if not sw["switched"]:
        print(f"  3 MECHANICS  skipped: a module whose coil cannot reverse "
              f"its magnet is not a robot,")
        print(f"               so its gait is not worth simulating. "
              f"({time.time()-t0:.0f} s)")
        return

    # ---- Stage 3 --------------------------------------------------------
    mod = build_module(d, drv if drv.feasible else None)
    mech = stage3_mechanics(d, mag, mod=mod, fidelity=fidelity)
    print(f"  3 MECHANICS  mass {mech['m_module']*1e3:5.0f} g   "
          f"hold {mech['hold_ratio']:5.1f}x   "
          f"pivot {mech['pivot_ratio']:4.2f}   "
          f"(barrier {mech['E_barrier']*1e3:.1f} mJ, "
          f"drive {mech['W_drive']*1e3:.1f} mJ)")
    print(f"               pivot angle {np.degrees(pivot_angle(d.n_gon)):.1f}"
          f" deg, electronics {'fit' if mech['fits'] else 'DO NOT FIT'}")

    sc = score(d, mag, mech, sw, drv)
    verdict = "FEASIBLE" if sc["feasible"] else "infeasible"
    print(f"    SCORE      {verdict}   scalar {sc['scalar']:.4f}")
    for v in sc["violations"]:
        print(f"               - {v}")
    print(f"               ({time.time()-t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", default="screen",
                    choices=("screen", "normal"))
    a = ap.parse_args()
    print("=" * 78)
    print("FULL PIPELINE, IN ORDER")
    print("=" * 78)
    print(f"\n  fidelity: {a.fidelity}")
    print("  each stage prints what it hands to the next one")
    for label, kw in CASES:
        run(label, kw, a.fidelity)
    print("""
  The third case differs from the second only in the pulse programme, and it
  is there to show that the drive waveform is a design variable rather than a
  fixed capacitor dump.""")


if __name__ == "__main__":
    main()
