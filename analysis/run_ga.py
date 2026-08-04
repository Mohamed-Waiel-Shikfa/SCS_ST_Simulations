"""Run the full genetic algorithm on the revised pipeline.

50 generations of 30 individuals, with local refinement of the seeds between
generations and a hard local search on the winner at the end.  Checkpointed
every generation so an interrupted run resumes where it stopped rather than
starting again.

    python analysis/run_ga.py                  # start (or resume) the run
    python analysis/run_ga.py --resume         # explicitly resume
    python analysis/run_ga.py --gens 10        # a shorter run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--fidelity", default="screen")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--refine-every", type=int, default=5)
    ap.add_argument("--refine-top", type=int, default=2)
    ap.add_argument("--refine-budget", type=int, default=16)
    ap.add_argument("--final-refine", type=int, default=200)
    ap.add_argument("--csv", default=str(HERE / "ga_front.csv"))
    ap.add_argument("--checkpoint", default=str(HERE / "ga_state.json"))
    a = ap.parse_args()

    import optimise as O

    print("=" * 78)
    print("GENETIC ALGORITHM ON THE REVISED PIPELINE")
    print("=" * 78)
    print(f"\n  population {a.pop}, {a.gens} generations, seed {a.seed}, "
          f"{a.workers} workers, {a.fidelity} fidelity")
    print(f"  genome: {len(O.GENOME)} genes over "
          f"{len(O.GENOME[0][2])} materials")
    print(f"  local refinement: top {a.refine_top} every {a.refine_every} "
          f"generations, {a.refine_budget} evaluations each; "
          f"{a.final_refine} on the winner")
    print(f"  checkpoint: {a.checkpoint}\n", flush=True)

    t0 = time.time()
    res = O.run_ga(pop_size=a.pop, generations=a.gens, seed=a.seed,
                   fidelity=a.fidelity, workers=a.workers,
                   csv_path=a.csv, checkpoint=a.checkpoint,
                   resume=a.resume, refine_every=a.refine_every,
                   refine_top=a.refine_top, refine_budget=a.refine_budget,
                   final_refine=a.final_refine, verbose=True)
    dt = time.time() - t0

    print(f"\n  {len(res.cache)} distinct designs evaluated in "
          f"{dt/60:.1f} minutes")
    O.report(res, top=15)
    print(f"\n  design matrix written to {a.csv}")


if __name__ == "__main__":
    main()
