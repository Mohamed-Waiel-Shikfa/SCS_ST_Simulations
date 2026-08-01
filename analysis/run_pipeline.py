"""End-to-end pipeline run: material -> magnetics -> driver -> module -> dynamics.

Runs one design the whole way through and reports what survives each stage.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from driver import select_driver  # noqa: E402
from dynamics import make_spec, run_scenario  # noqa: E402
from framework import (Design, stage1_magnetics, stage3_switching)  # noqa: E402
from module import build_module  # noqa: E402

CASES = [
    ("as built    ", dict(material="LNG37", circuit="none", v_cap=30.0,
                          a_module=60e-3)),
    ("recommended ", dict(material="LNGT72", circuit="potcore", v_cap=70.0,
                          a_module=60e-3)),
]

print("=" * 84)
print("FULL PIPELINE")
print("=" * 84)

for label, kw in CASES:
    d = Design(**kw)
    t0 = time.time()

    print(f"\n{'='*84}\n{label.strip()}  ({d.material}, {d.circuit}, "
          f"a={d.a_module*1e3:.0f} mm)\n{'='*84}")

    # ---- Stage 1
    mag = stage1_magnetics(d, fidelity="screen")
    print(f"  1 MAGNETICS  attract {mag['F_attract']:6.2f} N   "
          f"repel {mag['F_repel']:5.2f} N   asym {mag['asymmetry']:5.1f} : 1   "
          f"margin {mag['margin']:.2f}")

    # ---- Stage 2
    sw = stage3_switching(d)
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=d.n_faces)
    if drv.feasible:
        print(f"  2 DRIVER     {drv.v_bank:.0f} V, {drv.i_peak:.0f} A peak, "
              f"{drv.mass*1e3:.0f} g, ${drv.price:.0f}, "
              f"recharge {drv.recharge_s*1e3:.0f} ms")
        print(f"               {drv.bom()}")
    else:
        print(f"  2 DRIVER     NOT BUILDABLE: {drv.notes}")
        continue

    # ---- Stage 3
    mod, fits, free = build_module(d, drv)
    print(f"  3 MODULE     {mod.summary()}")
    print(f"               parts: " +
          "  ".join(f"{k} {v*1e3:.0f}g" for k, v in mod.parts.items()))
    if not fits:
        print("               electronics do NOT fit in the shell cavity")

    # ---- Stage 4
    spec = make_spec(d, mag)
    print(f"  4 DYNAMICS   pole charge {spec.q_attract:.3e} (attract), "
          f"{spec.q_repel:.3e} (repel)")

    weight = mod.mass * 9.81
    print(f"               module weight {weight:.3f} N, "
          f"attraction/weight {mag['F_attract']/weight:.1f}")

    # scenario 1: does B stay attached to A when both are attracting?
    on = [1] * 6
    off = [0] * 6
    attract = [0, 1, 0, 0, 0, 0]      # B's -x face toward A
    tr = run_scenario(mod, spec, [1, 0, 0, 0, 0, 0], attract, seconds=0.3)
    sep0, sep1 = tr[0]["sep"], tr[-1]["sep"]
    print(f"               latch test: separation {sep0*1e3:+.2f} -> "
          f"{sep1*1e3:+.2f} mm  "
          f"{'HELD' if sep1 < 2e-3 else 'SEPARATED'}")

    # scenario 2: reversed polarity on B
    tr = run_scenario(mod, spec, [1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0],
                      seconds=0.3)
    print(f"               repel test: separation {tr[0]['sep']*1e3:+.2f} -> "
          f"{tr[-1]['sep']*1e3:+.2f} mm, "
          f"moved {(tr[-1]['sep']-tr[0]['sep'])*1e3:+.2f} mm")

    print(f"\n  ({time.time()-t0:.0f}s)")
