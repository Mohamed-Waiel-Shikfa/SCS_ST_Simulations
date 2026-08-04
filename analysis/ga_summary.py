"""Summarise a finished GA run: the front, the trades, and what won.

Reads the CSV the optimiser appends to, rather than the final population,
because NSGA-II's crowding operator preserves SPREAD - a strong design can be
dropped from the last generation to keep the front diverse.  The best design
found and the best design surviving are not the same thing, and both are worth
seeing.
"""

from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "ga_front.csv"


def f(r, k):
    return float(r[k])


rows = [r for r in csv.DictReader(open(CSV)) if r["feasible"] == "True"]
allrows = list(csv.DictReader(open(CSV)))

print("=" * 104)
print(f"GA RUN SUMMARY  ({CSV.name}: {len(allrows)} evaluated, "
      f"{len(rows)} feasible)")
print("=" * 104)

hdr = (f"{'#':>3}{'material':>9}{'n':>3}{'r_face':>8}{'D':>6}{'L':>6}"
       f"{'circuit':>9}{'Fa':>7}{'Fr':>6}{'asym':>6}{'mass':>7}{'hold':>6}"
       f"{'piv':>6}{'E_sw':>8}{'score':>7}")
rows.sort(key=lambda r: -f(r, "scalar"))
print("\nTop 10 by scalar score (across every evaluation)\n")
print(hdr)
print("-" * len(hdr))
for i, r in enumerate(rows[:10]):
    print(f"{i:3d}{r['material']:>9}{r['n_gon']:>3}{f(r,'r_face')*1e3:7.1f}m"
          f"{f(r,'d_mag')*1e3:6.2f}{f(r,'l_mag')*1e3:6.2f}{r['circuit']:>9}"
          f"{f(r,'F_attract'):7.2f}{f(r,'F_repel'):6.2f}"
          f"{f(r,'asymmetry'):6.1f}{f(r,'m_module')*1e3:6.0f}g"
          f"{f(r,'hold_ratio'):6.1f}{f(r,'pivot_ratio'):6.2f}"
          f"{f(r,'e_switch')*1e3:7.0f}m{f(r,'scalar'):7.3f}")

print("\nExtremes of the front - what each objective costs\n")
for label, key, rev in (("most attraction", "F_attract", True),
                        ("most repulsion", "F_repel", True),
                        ("lowest asymmetry", "asymmetry", False),
                        ("lightest", "m_module", False),
                        ("cheapest to switch", "e_switch", False),
                        ("best pivot margin", "pivot_ratio", True)):
    r = sorted(rows, key=lambda x: f(x, key), reverse=rev)[0]
    print(f"  {label:<20}{r['material']:>8} n={r['n_gon']:<3}"
          f"Fa {f(r,'F_attract'):6.2f}N  Fr {f(r,'F_repel'):5.2f}N  "
          f"asym {f(r,'asymmetry'):5.1f}  {f(r,'m_module')*1e3:4.0f}g  "
          f"piv {f(r,'pivot_ratio'):5.2f}  "
          f"E_sw {f(r,'e_switch')*1e3:5.0f}mJ")

print("\nWhat the search selected for\n")
print("  material :", ", ".join(
    f"{k} {v}" for k, v in
    collections.Counter(r["material"] for r in rows).most_common()))
print("  n_gon    :", ", ".join(
    f"n={k} {v}" for k, v in
    collections.Counter(r["n_gon"] for r in rows).most_common()))
print("  circuit  :", ", ".join(
    f"{k} {v}" for k, v in
    collections.Counter(r["circuit"] for r in rows).most_common()))

why = collections.Counter()
for r in allrows:
    if r["feasible"] != "True":
        v = r["violations"].split(";")[0].strip()
        v = v.split("[")[0].strip()
        for tag in ("demag margin", "open-circuit demag", "pivot", "hold",
                    "cube", "electronics do not fit", "no driver",
                    "eval failed", "solve failed"):
            if tag in v:
                why[tag] += 1
                break
        else:
            why[v[:34] or "unknown"] += 1
print("\nWhy the rest were rejected\n")
for k, v in why.most_common(10):
    print(f"  {v:5d}  {k}")

piv = [f(r, "pivot_ratio") for r in rows]
print(f"\nPivot margin on the feasible set: min {min(piv):.2f}, "
      f"median {sorted(piv)[len(piv)//2]:.2f}, max {max(piv):.2f}")
print("  (the constraint is 1.5, so the search is sitting ON the boundary -")
print("   it is spending every joule it can on the other objectives)")
