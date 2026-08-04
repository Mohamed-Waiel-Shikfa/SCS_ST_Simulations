"""Design-matrix runner for the EPM module.

Evaluates a batch of candidate designs through all three stages and appends the
results to ``analysis/design_matrix.csv``.  The run is resumable: designs
already present in the CSV are skipped, so a long sweep can be stopped and
continued.

Usage
-----
    python analysis/run_design_matrix.py screen      # broad Latin-hypercube
    python analysis/run_design_matrix.py grades      # one row per material
    python analysis/run_design_matrix.py report      # analyse what exists

The matrix is the input to the optimiser, and is also the record of what has
been explored, so it is committed to the repo rather than regenerated silently.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from framework import (MATERIALS, Design, evaluate,  # noqa: E402
                       stage1_magnetics, stage2_mechanics, stage3_switching)

CSV = HERE / "design_matrix.csv"

VENDOR = [k for k, v in MATERIALS.items() if v["src"] == "vendor"]

KEY_FIELDS = ("material", "d_mag", "l_mag", "circuit", "t_steel", "r_clear",
              "gap", "n_faces", "a_module", "wire_d", "v_cap", "c_cap")


def key(d):
    return "|".join(f"{getattr(d, f):.6g}" if isinstance(getattr(d, f), float)
                    else str(getattr(d, f)) for f in KEY_FIELDS)


def existing_keys():
    if not CSV.exists():
        return set()
    with open(CSV, newline="") as fh:
        return {"|".join(f"{float(r[f]):.6g}" if f not in ("material",
                                                           "circuit")
                         else r[f] for f in KEY_FIELDS)
                for r in csv.DictReader(fh)}


def append(rows):
    """Append rows to the matrix CSV.

    A canonical column order is used rather than the key order of whichever row
    happens to be first in the batch.  Pre-screened designs build their result
    dict by a different path from fully evaluated ones, so taking fieldnames
    from ``rows[0]`` silently wrote later batches into shifted columns.
    """
    fields = list(rows[0].keys()) if not CSV.exists() else None
    if CSV.exists():
        with open(CSV, newline="") as fh:
            fields = next(csv.reader(fh))
    missing = [k for r in rows for k in r if k not in fields]
    if missing:
        raise ValueError(f"row has columns absent from the header: "
                         f"{sorted(set(missing))}")
    with open(CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="raise")
        if fh.tell() == 0:
            w.writeheader()
        w.writerows(rows)


def run(designs, label, fidelity="screen"):
    done = existing_keys()
    todo = [d for d in designs if key(d) not in done]
    print(f"{label}: {len(designs)} designs, {len(designs)-len(todo)} already "
          f"done, {len(todo)} to run (fidelity={fidelity})")
    t0 = time.time()
    batch = []
    for i, d in enumerate(todo, 1):
        t1 = time.time()
        try:
            row = evaluate(d, fidelity=fidelity)
        except Exception as exc:                       # keep the sweep going
            print(f"  [{i}/{len(todo)}] FAILED {d.material}: {exc}", flush=True)
            continue
        batch.append(row)
        flag = "ok " if row["feasible"] else "vio"
        print(f"  [{i}/{len(todo)}] {flag} {d.material:<9} "
              f"D{d.d_mag*1e3:4.1f} L{d.l_mag*1e3:4.1f} {d.circuit:<8} "
              f"Fa={row['F_attract']:6.2f} Fr={row['F_repel']:5.2f} "
              f"asym={row['asymmetry']:5.1f} score={row['scalar']:.3f} "
              f"({time.time()-t1:.0f}s)", flush=True)
        if len(batch) >= 3:
            append(batch)
            batch = []
    if batch:
        append(batch)
    print(f"{label}: done in {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------
def grades():
    """One row per vendor grade, at the current geometry, both circuits."""
    out = []
    for m in VENDOR:
        for circuit in ("none", "potcore"):
            out.append(Design(material=m, circuit=circuit, v_cap=70.0))
    return out


def screen(n=60, seed=0):
    """Latin hypercube over the continuous axes, crossed with the categoricals.

    Only vendor grades are used, since only those can be ordered.
    """
    rng = np.random.default_rng(seed)

    def lhs(n, lo, hi):
        u = (rng.permutation(n) + rng.random(n)) / n
        return lo + u * (hi - lo)

    d_mag = lhs(n, 2.0e-3, 12.0e-3)
    l_mag = lhs(n, 5.0e-3, 30.0e-3)
    t_steel = lhs(n, 0.5e-3, 3.0e-3)
    gap = lhs(n, 0.05e-3, 0.6e-3)
    wire_d = lhs(n, 0.15e-3, 0.6e-3)
    v_cap = lhs(n, 20.0, 150.0)
    a_mod = lhs(n, 20e-3, 80e-3)
    mats = [VENDOR[i % len(VENDOR)] for i in rng.permutation(n)]
    circs = ["potcore" if i % 4 else "none" for i in range(n)]

    return [Design(material=mats[i], d_mag=d_mag[i], l_mag=l_mag[i],
                   circuit=circs[i], t_steel=t_steel[i], gap=gap[i],
                   wire_d=wire_d[i], v_cap=v_cap[i], a_module=a_mod[i])
            for i in range(n)]


# --------------------------------------------------------------------------
def pareto(rows, objectives):
    """Non-dominated set.  objectives: dict name -> +1 maximise / -1 minimise."""
    pts = []
    for r in rows:
        pts.append([float(r[k]) * s for k, s in objectives.items()])
    pts = np.array(pts)
    keep = []
    for i in range(len(pts)):
        dominated = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        if not dominated.any():
            keep.append(i)
    return [rows[i] for i in keep]


def report():
    if not CSV.exists():
        print("no design matrix yet - run 'grades' or 'screen' first")
        return
    rows = list(csv.DictReader(open(CSV, newline="")))
    for r in rows:
        r["feasible"] = r["feasible"] in ("True", "true", "1")
    feas = [r for r in rows if r["feasible"]]

    print(f"design matrix: {len(rows)} designs, {len(feas)} feasible\n")
    if not feas:
        print("  no feasible designs - loosening which constraint would help?")
        from collections import Counter
        c = Counter(v.split()[0] + " " + v.split()[1]
                    for r in rows for v in r["violations"].split("; ") if v)
        for k, n in c.most_common():
            print(f"    {n:4d} x {k}")
        return

    front = pareto(feas, {"F_attract": +1, "F_repel": +1, "asymmetry": -1,
                          "e_switch": -1, "m_module": -1})
    print(f"Pareto front ({len(front)} designs) on "
          f"[max attract, max repel, min asymmetry, min switch energy, min mass]\n")
    hdr = (f"  {'material':<9} {'D':>5} {'L':>5} {'circuit':<8} {'gap':>5} "
           f"{'Fa':>7} {'Fr':>6} {'asym':>6} {'E_sw':>7} {'mass':>7} {'score':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted(front, key=lambda x: -float(x["scalar"])):
        print(f"  {r['material']:<9} {float(r['d_mag'])*1e3:5.1f} "
              f"{float(r['l_mag'])*1e3:5.1f} {r['circuit']:<8} "
              f"{float(r['gap'])*1e3:5.2f} {float(r['F_attract']):7.2f} "
              f"{float(r['F_repel']):6.2f} {float(r['asymmetry']):6.1f} "
              f"{float(r['e_switch'])*1e3:7.1f} "
              f"{float(r['m_module'])*1e3:7.1f} {float(r['scalar']):6.3f}")

    best = min(feas, key=lambda r: float(r["asymmetry"]))
    print(f"\n  most symmetric feasible design: {best['material']}, "
          f"D{float(best['d_mag'])*1e3:.1f} L{float(best['l_mag'])*1e3:.1f}, "
          f"{best['circuit']}, asymmetry {float(best['asymmetry']):.1f} : 1")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "grades":
        run(grades(), "grades")
        report()
    elif cmd == "screen":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        run(screen(n), f"screen({n})")
        report()
    elif cmd == "report":
        report()
    else:
        print(__doc__)
