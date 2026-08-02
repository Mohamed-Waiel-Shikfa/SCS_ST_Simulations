# Running the design pipeline elsewhere

Everything in `analysis/` is a batch job with a fixed seed and no
interactivity, so it can be handed to any machine with Python 3.11+ and the
pinned dependencies. This is what a remote or cloud agent needs to know.

## Setup

```bash
python -m pip install -r analysis/requirements.txt
python analysis/compat.py          # prints versions, warns on bad combinations
```

## Check the physics still holds before spending compute

These are fast and they gate everything else. If any of them fail, the
optimiser output is not worth having.

```bash
python simulations/Force_compute/python/magnet_force.py          # ~5 s
python simulations/Force_compute/python/validate_vs_experiment.py # ~20 s
python analysis/verify_fem.py                                     # ~2 min
python analysis/verify_dynamics.py                                # ~1 min
python analysis/verify_optimise.py                                # ~3 min
```

All five print a `... VERIFIED` line and exit 0.

## The long job

```bash
python analysis/optimise.py --pop 40 --gens 25 --seed 1 \
    --workers 16 --checkpoint analysis/ga_state.json \
    --csv analysis/design_matrix.csv
```

Roughly 1000 evaluations. Serial that is about an hour; with `--workers 16` it
is a few minutes of wall clock, since `evaluate()` is a pure function of a
`Design` and generations are evaluated as a batch.

**If it is interrupted, resume it.** A checkpoint is written after every
generation and holds the population, the full evaluation cache and the RNG
state:

```bash
python analysis/optimise.py --pop 40 --gens 25 --seed 1 \
    --workers 16 --checkpoint analysis/ga_state.json --resume
```

The resumed run is bit-identical to an uninterrupted one - `verify_optimise.py`
tests exactly that - and cached evaluations are not repeated, so nothing is
lost but the partial generation. The population size, seed and fidelity must
match the checkpoint or the run is refused rather than silently mixing two
experiments.

## Afterwards

Screening fidelity is only justified for *ranking* (Spearman 0.992 against
full fidelity, measured in `screening_study.py`). Any design that is going to
be believed must be re-run properly:

```bash
python analysis/verify_best.py     # edit BEST at the top to the chosen design
python analysis/verify_pivot.py    # MuJoCo, the locomotion claim
```

## What to hand back

- `analysis/ga_front.csv` - the Pareto front
- `analysis/design_matrix.csv` - every design evaluated, appended
- `analysis/ga_state.json` - the checkpoint, so the run can be extended
- the console log

## Known limitations to keep in view

- the pivot criterion is an energy bound; energy is necessary, not sufficient,
  and the MuJoCo run is the arbiter
- the OFF state under a neighbour's field has never been tested
- component data are typical catalogue values, not datasheet-confirmed
