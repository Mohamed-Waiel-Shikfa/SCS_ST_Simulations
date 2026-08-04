"""Does the optimiser survive being interrupted?

Three things have to be true before this run can be handed to a machine that
might kill it: parallel evaluation must give the same answer as serial, a
resumed run must give the same answer as an uninterrupted one, and the
checkpoint must never be left half-written.

The determinism check is the one that matters.  A resume that silently diverges
is worse than a crash, because the run still finishes and the result still
looks plausible.

Note the ``__main__`` guard.  Windows spawns fresh interpreters for pool
workers rather than forking, and each one re-imports the main module; without
the guard this file would re-run itself in every worker.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from optimise import (genome_key, load_checkpoint,  # noqa: E402
                      run_ga, save_checkpoint)

POP, GENS, SEED = 8, 4, 7


def main():
    ok = True
    print("=" * 76)
    print("OPTIMISER CHECKPOINT / PARALLELISM")
    print("=" * 76)

    tmp = Path(tempfile.mkdtemp())

    print("\n[1] Serial and parallel runs must agree\n")
    a = run_ga(POP, GENS, SEED, "screen", None, verbose=False, workers=1,
               checkpoint=None)
    b = run_ga(POP, GENS, SEED, "screen", None, verbose=False, workers=4,
               checkpoint=None)
    ka = sorted(genome_key(g) for g in a.population)
    kb = sorted(genome_key(g) for g in b.population)
    good = ka == kb
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] final populations identical "
          f"({len(ka)} genomes)")

    print("\n[2] An interrupted run must resume to the same answer\n")
    ck = tmp / "ck.json"
    run_ga(POP, 2, SEED, "screen", None, verbose=False, workers=4,
           checkpoint=str(ck))
    c = run_ga(POP, GENS, SEED, "screen", None, verbose=False, workers=4,
               checkpoint=str(ck), resume=True)
    kc = sorted(genome_key(g) for g in c.population)
    good = kc == ka
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] resumed population matches the "
          f"uninterrupted run")
    if not good:
        print(f"        {len(set(ka) & set(kc))}/{len(ka)} genomes in common")

    print("\n[3] The cache must survive the round trip\n")
    gen, pop, cache, state, hist, cfg = load_checkpoint(str(ck))
    good = len(cache) > 0 and len(pop) == POP and cfg["seed"] == SEED
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] gen {gen}, {len(pop)} genomes, "
          f"{len(cache)} cached evaluations, cfg seed {cfg.get('seed')}")

    nan_ok = True
    for _, row in list(cache.items())[:5]:
        for k, v in row.items():
            if isinstance(v, str) and v.startswith("__") and v.endswith("__"):
                nan_ok = False
    ok &= nan_ok
    print(f"  [{'ok ' if nan_ok else 'FAIL'}] NaN/inf sentinels decoded back "
          f"to floats")

    print("\n[4] Resuming with different settings must be refused\n")
    try:
        run_ga(POP + 2, GENS, SEED, "screen", None, verbose=False, workers=1,
               checkpoint=str(ck), resume=True)
        good = False
    except SystemExit:
        good = True
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] mismatched population size "
          f"rejected")

    print("\n[5] A killed write must not corrupt the checkpoint\n")
    before = ck.read_text()
    try:
        save_checkpoint(str(ck), 0, [{"bad": object()}], {}, None, [], {})
    except Exception:                                   # noqa: BLE001
        pass
    after = ck.read_text()
    good = before == after and bool(json.loads(after))
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] checkpoint intact after a failed "
          f"write ({len(after)} bytes, still valid JSON)")

    print("\n" + ("CHECKPOINTING VERIFIED" if ok else "CHECKPOINTING FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
