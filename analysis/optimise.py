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
Every evaluation runs a nonlinear FEM two to four times, so three things carry
the run: the pre-screen rejects impossible designs in milliseconds, results are
cached by genome so repeated individuals cost nothing, and evaluations within a
generation are spread across processes.  ``evaluate`` is a pure function of a
Design, so the parallelism is embarrassing - there is no shared state to guard.

Checkpointing
-------------
The population, the evaluation cache and the RNG state are written to disk
after every generation.  A run that is interrupted - a wall-clock limit on a
cloud runner, a laptop going to sleep - resumes from the last generation
instead of starting again, and the cache means even the interrupted generation
is not re-evaluated.  Because the RNG state is restored too, a resumed run
produces exactly the same sequence as an uninterrupted one.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from framework import (MATERIALS, Design, evaluate)  # noqa: E402
from materials import ALL_NAMES  # noqa: E402

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
    ("material", "cat", ALL_NAMES),
    ("circuit", "cat", ["none", "potcore"]),
    ("n_gon", "cat", [8, 12, 16, 20]),
    ("pulse_mode", "cat", ["single", "train"]),
    ("r_face", "real", 8e-3, 24e-3),
    ("d_frac", "real", 0.35, 1.00),     # EPM outer dia / available face width
    ("l_frac", "real", 0.15, 0.80),     # EPM depth / module radius
    ("f_clear", "real", 0.05, 0.35),    # share of EPM radius given to clearance
    ("f_steel", "real", 0.10, 0.45),    # share of EPM radius given to keeper
    ("gap", "real", 0.05e-3, 0.4e-3),
    ("wire_d", "real", 0.10e-3, 0.6e-3),
    ("n_layers", "int", 1, 16),         # a real winding variable at last
    ("v_cap", "real", 20.0, 300.0),
    ("c_cap", "real", 4.7e-6, 220e-6),
    ("f_pulse", "real", 2e3, 120e3),    # pulse-train frequency
    ("duty", "real", 0.10, 0.90),       # pulse-train duty cycle
    ("n_pulses", "int", 1, 12),
]

CONTINUOUS = [s[0] for s in GENOME if s[1] in ("real", "int")]
BOUNDS = {s[0]: (s[2], s[3]) for s in GENOME if s[1] in ("real", "int")}

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
        elif kind == "int":
            g[name] = int(rng.integers(spec[2], spec[3] + 1))
        else:
            g[name] = float(rng.uniform(spec[2], spec[3]))
    return g


def to_design(g):
    """Decode a genome into a Design, resolving fractions into dimensions.

    The radial budget on a face is shared between the magnet, the winding, the
    clearance and the keeper wall.  Allocating it by FRACTIONS rather than
    subtracting absolute thicknesses matters: with absolute values the
    clearance and keeper can consume the entire budget on a small face and
    leave a sub-millimetre magnet inside a heavy steel cup - which the search
    then wastes evaluations on.  Proportional allocation keeps every decoded
    design sensible.

    The winding is now part of that budget rather than free.  A deep coil eats
    the radius it needs from somewhere, and previously it ate it from nowhere.
    """
    n = int(g["n_gon"])
    r_face = g["r_face"]
    a_face = 2.0 * r_face * np.tan(np.pi / n)

    # total radius available to the whole EPM assembly on one face
    r_out = 0.5 * 0.92 * a_face * g["d_frac"]

    n_layers = int(round(g["n_layers"]))
    wire_d = g["wire_d"]
    build = n_layers * wire_d * 1.08 * 0.92          # coil.LAYER_PITCH etc

    if g["circuit"] == "potcore":
        f_clear, f_steel = g["f_clear"], g["f_steel"]
        f_mag = max(1.0 - f_clear - f_steel, 0.25)
        tot = f_mag + f_clear + f_steel
        avail = max(r_out - build, 0.5e-3)
        d_mag = 2.0 * avail * f_mag / tot
        r_clear = avail * f_clear / tot
        t_steel = avail * f_steel / tot
    else:
        d_mag = 2.0 * max(r_out - build, 0.5e-3)
        r_clear = 0.0
        t_steel = 0.3e-3

    depth_max = 0.85 * r_face
    l_mag = max(g["l_frac"] * depth_max -
                (t_steel if g["circuit"] == "potcore" else 0.0), 1.0e-3)

    return Design(material=g["material"], circuit=g["circuit"], n_gon=n,
                  r_face=r_face, d_mag=max(d_mag, 1.0e-3), l_mag=l_mag,
                  t_steel=max(t_steel, 0.3e-3), r_clear=max(r_clear, 0.0),
                  gap=g["gap"], wire_d=wire_d, n_layers=max(n_layers, 1),
                  v_cap=g["v_cap"], c_cap=g["c_cap"],
                  pulse_mode=g["pulse_mode"], f_pulse=g["f_pulse"],
                  duty=g["duty"], n_pulses=max(int(round(g["n_pulses"])), 1))


def genome_key(g):
    return json.dumps({k: (round(v, 9) if isinstance(v, float) else v)
                       for k, v in sorted(g.items())})


# --------------------------------------------------------------------------
# Parallel evaluation.  This has to be a module-level function so it can be
# pickled to a worker; Windows spawns fresh interpreters rather than forking.
def _eval_worker(args):
    g, fidelity = args
    try:
        row = evaluate(to_design(g), fidelity=fidelity)
    except Exception as exc:                       # noqa: BLE001
        row = to_design(g).as_row()
        row.update(feasible=False, violations=f"eval failed: {exc}",
                   scalar=0.0, F_attract=0.0, F_repel=0.0,
                   asymmetry=float("inf"), e_switch=float("inf"),
                   m_module=float("inf"), n_faces=0)
    return genome_key(g), row


# --------------------------------------------------------------------------
# Checkpointing.  NaN and infinity are not valid JSON, and json.dump writes
# them as bare NaN/Infinity which many parsers reject, so they are encoded
# explicitly and restored on load.
def _enc(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "__nan__"
        if math.isinf(x):
            return "__inf__" if x > 0 else "__-inf__"
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return _enc(float(x))
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _dec(x):
    if x == "__nan__":
        return float("nan")
    if x == "__inf__":
        return float("inf")
    if x == "__-inf__":
        return float("-inf")
    return x


def save_checkpoint(path, gen, pop, cache, rng, history, cfg):
    tmp = Path(str(path) + ".tmp")
    blob = dict(
        version=1, gen=gen, cfg=cfg, history=history,
        rng_state=rng.bit_generator.state,
        pop=[{k: _enc(v) for k, v in g.items()} for g in pop],
        cache=[[k, {kk: _enc(vv) for kk, vv in r.items()}]
               for k, r in cache.items()],
    )
    with open(tmp, "w") as fh:
        json.dump(blob, fh)
    tmp.replace(path)          # atomic: a killed process cannot leave a
                               # half-written checkpoint behind


def load_checkpoint(path):
    with open(path) as fh:
        blob = json.load(fh)
    pop = [{k: _dec(v) for k, v in g.items()} for g in blob["pop"]]
    cache = {k: {kk: _dec(vv) for kk, vv in r.items()}
             for k, r in blob["cache"]}
    return blob["gen"], pop, cache, blob["rng_state"], blob["history"], \
        blob.get("cfg", {})


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
        x1, x2 = float(p1[name]), float(p2[name])
        if abs(x1 - x2) < 1e-14:
            continue
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else \
            (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        a = 0.5 * ((x1 + x2) - beta * abs(x2 - x1))
        b = 0.5 * ((x1 + x2) + beta * abs(x2 - x1))
        c1[name] = _clip(name, kind, a, lo, hi)
        c2[name] = _clip(name, kind, b, lo, hi)
    return c1, c2


def _clip(name, kind, x, lo, hi):
    v = float(np.clip(x, lo, hi))
    return int(round(v)) if kind == "int" else v


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
        x = float(out[name])
        d1, d2 = (x - lo) / (hi - lo), (hi - x) / (hi - lo)
        u = rng.random()
        if u < 0.5:
            dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** \
                 (1 / (eta + 1)) - 1
        else:
            dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) *
                      (1 - d2) ** (eta + 1)) ** (1 / (eta + 1))
        out[name] = _clip(name, kind, x + dq * (hi - lo), lo, hi)
    return out


# --------------------------------------------------------------------------
# Local refinement
# --------------------------------------------------------------------------
def merit(row):
    """Single number a local search can climb.

    Feasible designs are ranked by the scalar score, which is positive.
    Infeasible ones are ranked by minus their total violation, which is
    negative, so any feasible design beats any infeasible one and an
    infeasible design can still walk downhill towards the feasible region.
    A local search needs a total order; the GA does not, which is why the two
    use different comparisons on the same evaluations.
    """
    if row.get("feasible"):
        return float(row.get("scalar") or 0.0)
    return -violation(row)


def local_refine(g0, evaluate_many, budget=40, step0=0.25, shrink=0.5,
                 min_step=0.01, verbose=False):
    """Compass search on the continuous genes, categoricals held fixed.

    Why a pattern search rather than a gradient method: the objective is a
    chain of nonlinear solves with no derivative available, it is mildly
    discontinuous wherever a driver component selection flips, and it is
    expensive.  Compass search needs no derivative, tolerates the
    discontinuities, and - the part that matters here - evaluates 2N
    independent trial points per iteration, so a whole iteration fits in one
    parallel batch.

    ``evaluate_many`` takes a list of genomes and returns a list of rows, so
    the caller supplies its own cache and process pool.
    """
    specs = [s for s in GENOME if s[1] in ("real", "int")]
    g = dict(g0)
    base = evaluate_many([g])[0]
    best_m = merit(base)
    used = 1
    step = step0
    history = [dict(n=used, merit=best_m, step=step)]

    while used < budget and step >= min_step:
        trials, tags = [], []
        for name, kind, lo, hi in specs:
            span = hi - lo
            for sgn in (+1, -1):
                cand = dict(g)
                x = float(g[name]) + sgn * step * span
                cand[name] = _clip(name, kind, x, lo, hi)
                if cand[name] == g[name]:
                    continue
                trials.append(cand)
                tags.append((name, sgn))
        if not trials:
            break
        if used + len(trials) > budget:
            trials = trials[: max(budget - used, 0)]
            tags = tags[: len(trials)]
        if not trials:
            break
        rows = evaluate_many(trials)
        used += len(trials)
        ms = [merit(r) for r in rows]
        k = int(np.argmax(ms))
        if ms[k] > best_m + 1e-12:
            g, best_m = trials[k], ms[k]
            if verbose:
                print(f"      local: {tags[k][0]} {tags[k][1]:+d} -> "
                      f"merit {best_m:.4f} ({used} evals)", flush=True)
        else:
            step *= shrink
        history.append(dict(n=used, merit=best_m, step=step))

    return g, best_m, used, history


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
    refined: dict = None
    cache: dict = None


def run_ga(pop_size=32, generations=12, seed=0, fidelity="screen",
           csv_path=None, verbose=True, workers=None, checkpoint=None,
           resume=False, refine_every=5, refine_top=2, refine_budget=16,
           final_refine=200):
    cfg = dict(pop_size=pop_size, generations=generations, seed=seed,
               fidelity=fidelity)
    rng = np.random.default_rng(seed)
    cache = {}
    history = []
    gen0 = 0

    if resume and checkpoint and Path(checkpoint).exists():
        gen0, pop, cache, rng_state, history, old_cfg = \
            load_checkpoint(checkpoint)
        for k in ("pop_size", "seed", "fidelity"):
            if old_cfg.get(k) != cfg[k]:
                raise SystemExit(
                    f"checkpoint was written with {k}={old_cfg.get(k)!r}, "
                    f"this run asks for {cfg[k]!r}.  Resuming would silently "
                    f"mix two different experiments; start a new run instead.")
        rng.bit_generator.state = rng_state
        # Resume AT the checkpointed generation, not after it.  The checkpoint
        # is written once that generation's population has been evaluated but
        # before selection consumes any randomness, so restarting at gen0
        # replays selection from exactly the state it would have had.  The
        # re-evaluation is free: every one of those genomes is in the cache.
        if verbose:
            print(f"  resumed from {checkpoint}: restarting generation "
                  f"{gen0}, {len(cache)} cached evaluations", flush=True)
    else:
        pop = [random_genome(rng) for _ in range(pop_size)]

    if workers is None:
        workers = max(1, min(os.cpu_count() or 1, pop_size * 2))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    n_eval = [len(cache)]
    t0 = time.time()

    def ev_many(genomes):
        """Evaluate a list of genomes, using the cache and the process pool."""
        todo, seen = [], set()
        for g in genomes:
            k = genome_key(g)
            if k not in cache and k not in seen:
                seen.add(k)
                todo.append(g)
        if todo:
            if pool is None:
                out = [_eval_worker((g, fidelity)) for g in todo]
            else:
                out = list(pool.map(_eval_worker,
                                    [(g, fidelity) for g in todo],
                                    chunksize=1))
            for k, row in out:
                cache[k] = row
            n_eval[0] += len(todo)
        return [cache[genome_key(g)] for g in genomes]

    try:
        for gen in range(gen0, generations + 1):
            rows = ev_many(pop)
            F = [objective_vector(r) for r in rows]
            V = [violation(r) for r in rows]
            fronts = fast_nondominated_sort(F, V)

            feas = [r for r in rows if r["feasible"]]
            best = max(feas, key=lambda r: r["scalar"]) if feas else None
            history.append(dict(gen=gen, n_feasible=len(feas),
                                n_eval=n_eval[0],
                                best=best["scalar"] if best else 0.0))
            if verbose:
                msg = (f"  gen {gen:3d}  feasible {len(feas):3d}/{len(rows):3d}"
                       f"  front {len(fronts[0]):3d}  evals {n_eval[0]:5d}  "
                       f"{time.time()-t0:5.0f}s")
                if best:
                    msg += (f"   best: {best['material']:<8} "
                            f"n={best['n_gon']:2d} "
                            f"Fa={best['F_attract']:5.2f} "
                            f"Fr={best['F_repel']:4.2f} "
                            f"asym={best['asymmetry']:4.1f} "
                            f"piv={best['pivot_ratio']:4.1f} "
                            f"m={best['m_module']*1e3:4.0f}g")
                print(msg, flush=True)

            if checkpoint:
                save_checkpoint(checkpoint, gen, pop, cache, rng, history, cfg)

            if gen == generations:
                break

            # ---- local refinement of the seeds, between generations
            # The GA is good at finding the right REGION and poor at the last
            # few per cent inside it, because polynomial mutation makes small
            # steps only by luck.  A compass search on the best few
            # individuals fixes that, and its refined genomes go back into the
            # population so the improvement is inherited rather than
            # discarded.  Every point it evaluates is recorded in the same
            # cache, so the refinements appear in the design matrix too.
            if refine_every and refine_top and gen and gen % refine_every == 0:
                order = sorted(range(len(pop)),
                               key=lambda i: -merit(rows[i]))
                gained = 0
                for i in order[:refine_top]:
                    g_new, m_new, used, _ = local_refine(
                        pop[i], ev_many, budget=refine_budget)
                    if m_new > merit(rows[i]) + 1e-12:
                        pop[i] = g_new
                        gained += 1
                if verbose and gained:
                    print(f"        local search improved {gained} seed(s), "
                          f"{n_eval[0]} evals total", flush=True)

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
            allr = ev_many(allg)
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

        rows = ev_many(pop)

        # ---- final refinement: hammer the overall best design
        # Between generations the local search gets a small budget, because it
        # is competing with the GA for evaluations.  Once the GA has finished
        # there is nothing left to compete with, so the winner gets a much
        # larger budget and a finer terminating step.  This is where the last
        # few per cent come from.
        refined = None
        if final_refine:
            k = int(np.argmax([merit(r) for r in rows]))
            before = merit(rows[k])
            g_new, m_new, used, hist = local_refine(
                pop[k], ev_many, budget=final_refine, step0=0.15,
                min_step=0.002, verbose=verbose)
            if verbose:
                print(f"\n  final local search on the best design: "
                      f"merit {before:.4f} -> {m_new:.4f} in {used} "
                      f"evaluations", flush=True)
            if m_new > before + 1e-12:
                pop[k] = g_new
                rows = ev_many(pop)
            refined = dict(index=k, before=before, after=m_new,
                           evals=used, history=hist)
            # Checkpoint AFTER the final refinement, not only before it.
            # Otherwise the 200 evaluations that produced the winning design -
            # including the winner itself - are absent from the saved state,
            # and the design matrix written from that state does not contain
            # the design the run reports as best.
            if checkpoint:
                save_checkpoint(checkpoint, generations, pop, cache, rng,
                                history, cfg)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    F = [objective_vector(r) for r in rows]
    V = [violation(r) for r in rows]
    fronts = fast_nondominated_sort(F, V)

    if csv_path:
        write_rows(csv_path, list(cache.values()))

    res = GAResult(population=pop, rows=rows, fronts=fronts, history=history)
    res.refined = refined
    res.cache = cache
    return res


def write_rows(path, rows, append=False):
    """Write the design matrix.

    Overwrites by default.  Appending was the old behaviour and it silently
    mixed schemas: a rerun after the row fields changed left the file with the
    previous run's columns for its first thousand rows and the new ones after,
    so the "best design" read back out of it was a stale row from a model that
    no longer exists.  A run writes its own matrix; use ``append`` only when
    deliberately accumulating separate sweeps.
    """
    from framework import ROW_FIELDS
    path = Path(path)
    mode = "a" if (append and path.exists()) else "w"
    with open(path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROW_FIELDS),
                           extrasaction="ignore")
        if mode == "w":
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
    ap = argparse.ArgumentParser(
        description="NSGA-II over the Magnobots design pipeline")
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fidelity", default="screen")
    ap.add_argument("--csv", default=str(HERE / "design_matrix.csv"))
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel evaluation processes "
                         "(default: one per core, capped at 2x population)")
    ap.add_argument("--checkpoint", default=str(HERE / "ga_state.json"),
                    help="checkpoint file written after every generation")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint instead of starting "
                         "a new run")
    ap.add_argument("--no-checkpoint", action="store_true")
    a = ap.parse_args()

    from compat import check_environment
    check_environment()

    ckpt = None if a.no_checkpoint else a.checkpoint
    workers = a.workers or max(1, min(os.cpu_count() or 1, a.pop * 2))

    print("=" * 100)
    print(f"NSGA-II  pop={a.pop}  generations={a.gens}  seed={a.seed}  "
          f"fidelity={a.fidelity}  workers={workers}")
    print(f"objectives: " + ", ".join(
        f"{'max' if s > 0 else 'min'} {k}" for k, s in OBJECTIVES.items()))
    if ckpt:
        print(f"checkpoint: {ckpt}"
              + ("  (resuming)" if a.resume else ""))
    print("=" * 100)
    res = run_ga(a.pop, a.gens, a.seed, a.fidelity, a.csv,
                 workers=workers, checkpoint=ckpt, resume=a.resume)
    report(res)
