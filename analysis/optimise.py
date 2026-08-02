"""Stage 5: multi-objective optimisation of the module design (NSGA-II).

Why NSGA-II rather than a plain genetic algorithm
-------------------------------------------------
The objectives genuinely conflict and there is no defensible way to weight them
in advance.  More attraction and less mass pull against each other; higher
coercivity buys repulsion and costs switching energy; more faces make the pivot
cheaper but add magnets and mass.  A weighted-sum GA would silently pick one
compromise and hide the rest.  NSGA-II returns the whole Pareto front, so the
trade can be inspected and the final call made with the numbers visible.

Constraint handling
-------------------
Constrained-domination (Deb 2002): a feasible design always beats an infeasible
one, two infeasible designs are compared on total violation, and two feasible
designs are compared by Pareto dominance.  This is preferred to a penalty
because it needs no penalty weight, and because most of the search space here
is infeasible - the population has to be able to make progress while still
infeasible, by reducing violation, before it can start optimising.

Cost control
------------
Every evaluation runs a nonlinear FEM twice, so the pre-screen matters: it
rejects geometrically or electrically impossible designs in milliseconds, and
those are also the numerically stiff ones.  Results are cached by genome so
repeated individuals cost nothing, and the whole run is appended to the design
matrix.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from framework import (MATERIALS, Design, evaluate)  # noqa: E402

VENDOR = [k for k, v in MATERIALS.items() if v["src"] == "vendor"]

# ---- search space --------------------------------------------------------
# The magnet is encoded as FRACTIONS of the space actually available on a face
# rather than as absolute millimetres.  Absolute encoding makes almost every
# random genome geometrically impossible - in a 60-sample probe, 41 had an EPM
# wider than its face and 21 had one deeper than the module radius - so the
# search spends its whole budget rediscovering geometry instead of physics.
# With fractional encoding every genome is buildable by construction and the
# box bounds imply the coupling constraints.
#
# name, kind, low, high  (kind: "real", "cat")
GENOME = [
    ("material", "cat", VENDOR),
    ("circuit", "cat", ["none", "potcore"]),
    ("n_gon", "cat", [8, 12, 16, 20]),
    ("r_face", "real", 8e-3, 24e-3),
    ("d_frac", "real", 0.35, 1.00),     # EPM outer dia / available face width
    ("l_frac", "real", 0.15, 0.80),     # EPM depth / module radius
    ("f_clear", "real", 0.05, 0.35),    # share of EPM radius given to clearance
    ("f_steel", "real", 0.10, 0.45),    # share of EPM radius given to keeper
    ("gap", "real", 0.05e-3, 0.4e-3),
    ("wire_d", "real", 0.15e-3, 0.6e-3),
    ("v_cap", "real", 20.0, 200.0),
    ("c_cap", "real", 4.7e-6, 220e-6),
]

# objective name -> +1 maximise, -1 minimise
OBJECTIVES = {"f_attract": +1, "f_repel": +1, "asymmetry": -1,
              "e_switch": -1, "m_module": -1}

# soft constraints, evaluated as normalised shortfalls so the search can make
# gradient progress while still infeasible
SOFT = ("margin", "hold_ratio", "pivot_ratio", "cube", "fit")


# --------------------------------------------------------------------------
def random_genome(rng):
    g = {}
    for spec in GENOME:
        name, kind = spec[0], spec[1]
        if kind == "cat":
            g[name] = spec[2][rng.integers(len(spec[2]))]
        else:
            g[name] = float(rng.uniform(spec[2], spec[3]))
    return g


def to_design(g):
    """Decode a genome into a Design, resolving fractions into dimensions.

    The radial budget on a face is shared between the magnet, the clearance and
    the keeper wall.  Allocating it by FRACTIONS rather than subtracting
    absolute thicknesses matters: with absolute values the clearance and keeper
    can consume the entire budget on a small face and leave a sub-millimetre
    magnet inside a heavy steel cup - which the search then wastes evaluations
    on.  Proportional allocation keeps every decoded design sensible.
    """
    n = int(g["n_gon"])
    r_face = g["r_face"]
    a_face = 2.0 * r_face * np.tan(np.pi / n)

    # total radius available to the whole EPM assembly on one face
    r_out = 0.5 * 0.92 * a_face * g["d_frac"]

    if g["circuit"] == "potcore":
        # split the radius: magnet / clearance / keeper
        f_clear, f_steel = g["f_clear"], g["f_steel"]
        f_mag = max(1.0 - f_clear - f_steel, 0.25)
        tot = f_mag + f_clear + f_steel
        d_mag = 2.0 * r_out * f_mag / tot
        r_clear = r_out * f_clear / tot
        t_steel = r_out * f_steel / tot
    else:
        d_mag = 2.0 * r_out
        r_clear = 0.0
        t_steel = g["f_steel"] * 1.5e-3      # winding build only

    depth_max = 0.85 * r_face
    l_mag = max(g["l_frac"] * depth_max -
                (t_steel if g["circuit"] == "potcore" else 0.0), 1.0e-3)

    return Design(material=g["material"], circuit=g["circuit"], n_gon=n,
                  r_face=r_face, d_mag=max(d_mag, 1.0e-3), l_mag=l_mag,
                  t_steel=max(t_steel, 0.3e-3), r_clear=max(r_clear, 0.0),
                  gap=g["gap"], wire_d=g["wire_d"], v_cap=g["v_cap"],
                  c_cap=g["c_cap"])


def genome_key(g):
    return json.dumps({k: (round(v, 9) if isinstance(v, float) else v)
                       for k, v in sorted(g.items())})


# --------------------------------------------------------------------------
def sbx(p1, p2, rng, eta=15.0, p_cross=0.9):
    """Simulated binary crossover for reals, uniform choice for categoricals."""
    c1, c2 = dict(p1), dict(p2)
    if rng.random() > p_cross:
        return c1, c2
    for spec in GENOME:
        name, kind = spec[0], spec[1]
        if kind == "cat":
            if rng.random() < 0.5:
                c1[name], c2[name] = p2[name], p1[name]
            continue
        lo, hi = spec[2], spec[3]
        x1, x2 = p1[name], p2[name]
        if abs(x1 - x2) < 1e-14:
            continue
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else \
            (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        a = 0.5 * ((x1 + x2) - beta * abs(x2 - x1))
        b = 0.5 * ((x1 + x2) + beta * abs(x2 - x1))
        c1[name] = float(np.clip(a, lo, hi))
        c2[name] = float(np.clip(b, lo, hi))
    return c1, c2


def mutate(g, rng, eta=20.0, p_mut=None):
    """Polynomial mutation for reals, resample for categoricals."""
    out = dict(g)
    p = p_mut if p_mut is not None else 1.0 / len(GENOME)
    for spec in GENOME:
        name, kind = spec[0], spec[1]
        if rng.random() > p:
            continue
        if kind == "cat":
            out[name] = spec[2][rng.integers(len(spec[2]))]
            continue
        lo, hi = spec[2], spec[3]
        x = out[name]
        d1, d2 = (x - lo) / (hi - lo), (hi - x) / (hi - lo)
        u = rng.random()
        if u < 0.5:
            dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** \
                 (1 / (eta + 1)) - 1
        else:
            dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) *
                      (1 - d2) ** (eta + 1)) ** (1 / (eta + 1))
        out[name] = float(np.clip(x + dq * (hi - lo), lo, hi))
    return out


# --------------------------------------------------------------------------
def objective_vector(row):
    """Objectives as a minimisation vector."""
    v = []
    for k, sense in OBJECTIVES.items():
        x = row.get(k)
        if x is None or not np.isfinite(x):
            x = 1e12 if sense < 0 else -1e12
        v.append(-sense * float(x))
    return np.array(v)


def violation(row):
    """Total constraint violation, continuous and normalised; 0 when feasible.

    A count of broken rules gives the search no gradient - every infeasible
    design looks equally bad, so it cannot walk downhill towards the feasible
    region.  Since most of this space is infeasible, that is fatal.  Each
    constraint is therefore scored as a normalised shortfall.
    """
    if row.get("feasible"):
        return 0.0

    from framework import CUBE_MAX, HOLD_MIN, MARGIN_LIMIT, PIVOT_MIN

    v = 0.0
    txt = str(row.get("violations", ""))

    m = row.get("margin")
    if m is not None and np.isfinite(m) and m > MARGIN_LIMIT:
        v += (m - MARGIN_LIMIT) / MARGIN_LIMIT
    h = row.get("hold_ratio")
    if h is not None and np.isfinite(h) and h < HOLD_MIN:
        v += (HOLD_MIN - h) / HOLD_MIN
    p = row.get("pivot_ratio")
    if p is not None and np.isfinite(p) and p < PIVOT_MIN:
        v += (PIVOT_MIN - p) / PIVOT_MIN
    c = row.get("bounding_cube")
    if c is not None and np.isfinite(c) and c > CUBE_MAX:
        v += (c - CUBE_MAX) / CUBE_MAX
    if "electronics do not fit" in txt:
        v += 0.5
    if "open-circuit demag" in txt:
        v += 0.5
    if "no driver" in txt:
        v += 1.0
    if "eval failed" in txt or "solve failed" in txt:
        v += 3.0
    # anything unrecognised still has to count for something
    return v if v > 0 else 1.0


def dominates(fa, fb, va, vb):
    """Constrained domination (Deb 2002).

    A feasible design always beats an infeasible one; two infeasible designs
    are compared on total violation; two feasible ones by Pareto dominance.
    """
    if va > 0 or vb > 0:
        if va < vb:
            return True
        if vb < va:
            return False
        return False
    return bool(np.all(fa <= fb) and np.any(fa < fb))


def fast_nondominated_sort(F, V):
    n = len(F)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, dtype=int)
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(F[i], F[j], V[i], V[j]):
                S[i].append(j)
            elif dominates(F[j], F[i], V[j], V[i]):
                cnt[i] += 1
        if cnt[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(F, idx):
    d = np.zeros(len(idx))
    if len(idx) <= 2:
        return np.full(len(idx), np.inf)
    A = np.array([F[i] for i in idx])
    for m in range(A.shape[1]):
        order = np.argsort(A[:, m])
        d[order[0]] = d[order[-1]] = np.inf
        rng = A[order[-1], m] - A[order[0], m]
        if rng <= 0:
            continue
        for q in range(1, len(idx) - 1):
            d[order[q]] += (A[order[q + 1], m] - A[order[q - 1], m]) / rng
    return d


# --------------------------------------------------------------------------
@dataclass
class GAResult:
    population: list
    rows: list
    fronts: list
    history: list


def run_ga(pop_size=32, generations=12, seed=0, fidelity="screen",
           csv_path=None, verbose=True):
    rng = np.random.default_rng(seed)
    cache = {}
    n_eval = 0

    def ev(g):
        nonlocal n_eval
        k = genome_key(g)
        if k not in cache:
            try:
                cache[k] = evaluate(to_design(g), fidelity=fidelity)
            except Exception as exc:
                bad = to_design(g).as_row()
                bad.update(feasible=False, violations=f"eval failed: {exc}",
                           scalar=0.0, f_attract=0.0, f_repel=0.0,
                           asymmetry=np.inf, e_switch=np.inf,
                           m_module=np.inf, n_faces=0)
                cache[k] = bad
            n_eval += 1
        return cache[k]

    pop = [random_genome(rng) for _ in range(pop_size)]
    history = []
    t0 = time.time()

    for gen in range(generations + 1):
        rows = [ev(g) for g in pop]
        F = [objective_vector(r) for r in rows]
        V = [violation(r) for r in rows]
        fronts = fast_nondominated_sort(F, V)

        feas = [r for r in rows if r["feasible"]]
        best = max(feas, key=lambda r: r["scalar"]) if feas else None
        history.append(dict(gen=gen, n_feasible=len(feas), n_eval=n_eval,
                            best=best["scalar"] if best else 0.0))
        if verbose:
            msg = (f"  gen {gen:3d}  feasible {len(feas):3d}/{len(rows):3d}  "
                   f"front {len(fronts[0]):3d}  evals {n_eval:4d}  "
                   f"{time.time()-t0:5.0f}s")
            if best:
                msg += (f"   best: {best['material']:<8} n={best['n_gon']:2d} "
                        f"Fa={best['f_attract'] if 'f_attract' in best else best['F_attract']:5.2f} "
                        f"Fr={best['F_repel']:4.2f} "
                        f"asym={best['asymmetry']:4.1f} "
                        f"m={best['m_module']*1e3:4.0f}g")
            print(msg, flush=True)

        if gen == generations:
            break

        # ---- tournament selection on (rank, crowding)
        rank = np.zeros(len(pop), dtype=int)
        crowd = np.zeros(len(pop))
        for r_i, fr in enumerate(fronts):
            d = crowding(F, fr)
            for pos, i in enumerate(fr):
                rank[i] = r_i
                crowd[i] = d[pos]

        def pick():
            a, b = rng.integers(len(pop)), rng.integers(len(pop))
            if rank[a] != rank[b]:
                return pop[a if rank[a] < rank[b] else b]
            return pop[a if crowd[a] > crowd[b] else b]

        children = []
        while len(children) < pop_size:
            c1, c2 = sbx(pick(), pick(), rng)
            children.append(mutate(c1, rng))
            if len(children) < pop_size:
                children.append(mutate(c2, rng))

        # ---- elitist survival from parents + children
        allg = pop + children
        allr = [ev(g) for g in allg]
        AF = [objective_vector(r) for r in allr]
        AV = [violation(r) for r in allr]
        afronts = fast_nondominated_sort(AF, AV)

        newpop = []
        for fr in afronts:
            if len(newpop) + len(fr) <= pop_size:
                newpop.extend(allg[i] for i in fr)
            else:
                d = crowding(AF, fr)
                order = np.argsort(-d)
                for o in order[: pop_size - len(newpop)]:
                    newpop.append(allg[fr[o]])
                break
        pop = newpop

    rows = [ev(g) for g in pop]
    F = [objective_vector(r) for r in rows]
    V = [violation(r) for r in rows]
    fronts = fast_nondominated_sort(F, V)

    if csv_path:
        write_rows(csv_path, list(cache.values()))

    return GAResult(population=pop, rows=rows, fronts=fronts, history=history)


def write_rows(path, rows):
    from framework import ROW_FIELDS
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROW_FIELDS),
                           extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ROW_FIELDS})


# --------------------------------------------------------------------------
def report(res, top=12):
    rows = res.rows
    feas = [r for r in rows if r["feasible"]]
    print("\n" + "=" * 100)
    print(f"FINAL POPULATION: {len(rows)} designs, {len(feas)} feasible, "
          f"Pareto front {len(res.fronts[0])}")
    print("=" * 100)
    if not feas:
        print("\n  no feasible designs")
        return
    front = [rows[i] for i in res.fronts[0] if rows[i]["feasible"]]
    front.sort(key=lambda r: -r["scalar"])
    hdr = (f"  {'material':<9} {'n':>2} {'r_face':>7} {'D_mag':>6} "
           f"{'L_mag':>6} {'circuit':<8} {'Fa':>6} {'Fr':>5} {'asym':>5} "
           f"{'mass':>6} {'E_sw':>7} {'hold':>5} {'pivot':>6} {'score':>6}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in front[:top]:
        print(f"  {r['material']:<9} {r['n_gon']:2d} "
              f"{r['r_face']*1e3:6.1f}m {r['d_mag']*1e3:5.1f}m "
              f"{r['l_mag']*1e3:5.1f}m {r['circuit']:<8} "
              f"{r['F_attract']:6.2f} {r['F_repel']:5.2f} "
              f"{r['asymmetry']:5.1f} {r['m_module']*1e3:5.0f}g "
              f"{r['e_switch']*1e3:6.1f}m {r['hold_ratio']:5.1f} "
              f"{r['pivot_ratio']:6.1f} {r['scalar']:6.3f}")
    return front


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fidelity", default="screen")
    ap.add_argument("--csv", default=str(HERE / "design_matrix.csv"))
    a = ap.parse_args()

    print("=" * 100)
    print(f"NSGA-II  pop={a.pop}  generations={a.gens}  seed={a.seed}  "
          f"fidelity={a.fidelity}")
    print(f"objectives: " + ", ".join(
        f"{'max' if s > 0 else 'min'} {k}" for k, s in OBJECTIVES.items()))
    print("=" * 100)
    res = run_ga(a.pop, a.gens, a.seed, a.fidelity, a.csv)
    report(res)
