"""Verify the optimiser's chosen design at full fidelity, end to end.

Screening fidelity is used inside the GA for speed.  Any design that is going
to be believed has to be re-run at full FEM fidelity and taken through the
dynamics, because a coarse mesh is only justified for RANKING - it was never
justified for a final number.  Screening carries a median 4.2 % force error
(measured in screening_study.py), which is fine for ordering designs and not
fine for quoting one.

Designs are read from the GA's CSV rather than pasted in, so this cannot drift
out of step with the run it is supposed to be checking.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from driver import select_driver  # noqa: E402
from dynamics import make_spec, run_scenario  # noqa: E402
from framework import (Design, score, stage1_magnetics,  # noqa: E402
                       stage2_mechanics, stage3_switching)
from module import build_module, pivot_angle  # noqa: E402

DESIGN_KEYS = ("material", "circuit", "n_gon", "r_face", "d_mag", "l_mag",
               "t_steel", "r_clear", "gap", "wire_d", "v_cap", "c_cap")


def from_csv(path, key="scalar", reverse=True, n=1):
    """Pull the top-n feasible designs from a GA result CSV."""
    rows = [r for r in csv.DictReader(open(path)) if r["feasible"] == "True"]
    rows.sort(key=lambda r: float(r[key]), reverse=reverse)
    out = []
    for r in rows[:n]:
        kw = {}
        for k in DESIGN_KEYS:
            v = r[k]
            kw[k] = v if k in ("material", "circuit") else (
                int(v) if k == "n_gon" else float(v))
        out.append((kw, r))
    return out


BASELINE = dict(material="LNG37", circuit="none", n_gon=8, r_face=19.4e-3,
                d_mag=4.75e-3, l_mag=12.5e-3, t_steel=0.5e-3, r_clear=0.0,
                gap=0.1e-3, wire_d=0.3e-3, v_cap=30.0, c_cap=10e-6)


def run(label, kw, do_dynamics=True, screen=None):
    d = Design(**kw)
    print("\n" + "=" * 86)
    print(f"{label}")
    print("=" * 86)
    print(f"  geometry   n={d.n_gon} ({d.n_faces} faces), "
          f"r_face {d.r_face*1e3:.1f} mm, face {d.a_face*1e3:.1f} mm, "
          f"cube {d.bounding_cube*1e3:.1f} mm, "
          f"pivot {np.degrees(pivot_angle(d.n_gon)):.0f} deg")
    print(f"  magnet     {d.material}, D {d.d_mag*1e3:.2f} x L "
          f"{d.l_mag*1e3:.2f} mm (L/D {d.l_mag/d.d_mag:.2f}), {d.circuit}")

    t = time.time()
    mag = stage1_magnetics(d, fidelity="normal")
    sw = stage3_switching(d)
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=d.n_faces)
    mod = build_module(d, drv if drv.feasible else None)
    mech = stage2_mechanics(d, mag, mod=mod)
    sc = score(d, mag, mech, sw, drv)

    print(f"\n  1 MAGNETICS  attract {mag['F_attract']:6.2f} N   "
          f"repel {mag['F_repel']:5.2f} N   "
          f"asymmetry {mag['asymmetry']:4.1f} : 1")
    print(f"               J {mag['J_attract']:.3f} T attract / "
          f"{mag['J_repel']:.3f} T repel,  worst margin "
          f"{mag['margin']:.2f} of Hcj")
    if drv.feasible:
        print(f"  2 DRIVER     {drv.v_bank:.0f} V, {drv.i_peak:.0f} A peak, "
              f"{drv.mass*1e3:.0f} g, {drv.volume*1e6:.1f} cc, "
              f"${drv.price:.0f}, {drv.topology}")
        print(f"               {drv.bom()}")
    print(f"  3 MODULE     {mod.summary()}")
    print(f"               " +
          "  ".join(f"{k} {v*1e3:.0f}g" for k, v in mod.parts.items()) +
          f"   free {mod.free_volume*1e6:.1f} cc, fits={mod.fits}")
    print(f"  4 MECHANICS  hold {mech['hold_ratio']:.1f} x weight,  "
          f"pivot work/barrier {mech['pivot_ratio']:.2f},  "
          f"lift {mech['lift']*1e3:.2f} mm")
    print(f"\n  FEASIBLE: {sc['feasible']}"
          + ("" if sc["feasible"] else f"  ({'; '.join(sc['violations'])})"))
    if screen is not None:
        print(f"  screening said: Fa {float(screen['F_attract']):.2f} N, "
              f"Fr {float(screen['F_repel']):.2f} N, "
              f"pivot {float(screen['pivot_ratio']):.2f}"
              f"   ->  full fidelity moves Fa by "
              f"{(mag['F_attract']/float(screen['F_attract'])-1)*100:+.1f} %, "
              f"pivot by "
              f"{(mech['pivot_ratio']/float(screen['pivot_ratio'])-1)*100:+.1f} %")
    print(f"  ({time.time()-t:.0f}s at full fidelity)")

    if not do_dynamics or not drv.feasible:
        return d, mag, mod, sc

    spec = make_spec(d, mag, fidelity="screen")
    ix_p = int(np.argmax(mod.normals @ np.array([1.0, 0, 0])))
    ix_m = int(np.argmin(mod.normals @ np.array([1.0, 0, 0])))
    tr = run_scenario(mod, spec, [(ix_p, ix_m, "attract")], seconds=0.25)
    print(f"  5 DYNAMICS   latch: separation {tr[0]['sep']*1e3:+.2f} -> "
          f"{tr[-1]['sep']*1e3:+.2f} mm  "
          f"{'HELD' if tr[-1]['sep'] < 2e-3 else 'SEPARATED'}")
    tr = run_scenario(mod, spec, [(ix_p, ix_m, "repel")], seconds=0.25)
    print(f"               repel: moved {(tr[-1]['sep']-tr[0]['sep'])*1e3:+.1f}"
          f" mm in 0.25 s")
    return d, mag, mod, sc


def main():
    print("=" * 86)
    print("FULL-FIDELITY VERIFICATION OF THE OPTIMISED DESIGNS")
    print("=" * 86)

    csv_path = HERE / "ga_front.csv"
    run("BASELINE  (as built: Alnico 5, bare rod, 18 faces)", BASELINE)

    for kw, row in from_csv(csv_path, "scalar", True, 1):
        run("OPTIMISED (best scalar score of 994 evaluations)", kw, screen=row)

    # The lightest feasible design is the other end of the trade and is the one
    # a builder would actually reach for, so it gets checked too.  A front is
    # only useful if more than its extreme point survives scrutiny.
    for kw, row in from_csv(csv_path, "m_module", False, 1):
        run("LIGHTEST  (same front, minimum mass)", kw, screen=row)


if __name__ == "__main__":
    main()
