"""Re-run the final local refinement from a checkpoint and rewrite the matrix.

The final aggressive refinement of the winning design happens after the last
generation, and an earlier version of ``run_ga`` did not checkpoint again
afterwards - so its 200 evaluations, including the winning design itself, were
absent from the saved state and therefore from the design matrix written out
of it.  ``optimise.run_ga`` now saves after refining; this script recovers a
run made before that fix without repeating the 50 generations.

    python analysis/finish_ga.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(HERE / "ga_state.json"))
    ap.add_argument("--csv", default=str(HERE / "ga_front.csv"))
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()

    import os
    from concurrent.futures import ProcessPoolExecutor

    import optimise as O

    gen, pop, cache, rng_state, history, cfg = \
        O.load_checkpoint(a.checkpoint)
    fidelity = cfg.get("fidelity", "screen")
    print(f"  checkpoint at generation {gen}: {len(pop)} individuals, "
          f"{len(cache)} evaluations cached")

    workers = max(1, min(a.workers, os.cpu_count() or 1))
    pool = ProcessPoolExecutor(max_workers=workers)
    n = [0]

    def ev_many(genomes):
        todo, seen = [], set()
        for g in genomes:
            k = O.genome_key(g)
            if k not in cache and k not in seen:
                seen.add(k)
                todo.append(g)
        if todo:
            out = list(pool.map(O._eval_worker,
                                [(g, fidelity) for g in todo], chunksize=1))
            for k, row in out:
                cache[k] = row
            n[0] += len(todo)
        return [cache[O.genome_key(g)] for g in genomes]

    try:
        rows = ev_many(pop)
        k = int(np.argmax([O.merit(r) for r in rows]))
        before = O.merit(rows[k])
        print(f"  refining individual {k}, merit {before:.4f}, "
              f"budget {a.budget}")
        t0 = time.time()
        g_new, m_new, used, _ = O.local_refine(pop[k], ev_many,
                                               budget=a.budget, step0=0.15,
                                               min_step=0.002, verbose=True)
        if m_new > before + 1e-12:
            pop[k] = g_new
        rows = ev_many(pop)
        print(f"\n  merit {before:.4f} -> {m_new:.4f} in {used} evaluations "
              f"({time.time()-t0:.0f} s, {n[0]} new)")
    finally:
        pool.shutdown(wait=True)

    rng = np.random.default_rng(cfg.get("seed", 0))
    rng.bit_generator.state = rng_state
    O.save_checkpoint(a.checkpoint, gen, pop, cache, rng, history, cfg)
    O.write_rows(a.csv, list(cache.values()))
    print(f"  {len(cache)} designs written to {a.csv}")

    F = [O.objective_vector(r) for r in rows]
    V = [O.violation(r) for r in rows]
    res = O.GAResult(population=pop, rows=rows,
                     fronts=O.fast_nondominated_sort(F, V), history=history)
    O.report(res, top=12)


if __name__ == "__main__":
    main()
