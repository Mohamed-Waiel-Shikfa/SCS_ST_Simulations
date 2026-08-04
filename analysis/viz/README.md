# Design explorer

An interactive view of the whole pipeline: browse the design matrix, feed any
configuration back through the real solvers, and see each stage.

```bash
python analysis/viz/server.py            # then open http://127.0.0.1:5173
python analysis/viz/server.py --port 8080
```

Needs `flask` and `matplotlib` on top of `analysis/requirements.txt`.

## What it does

**Design matrix** &mdash; all 994 evaluated designs, sortable and filterable, with
a trade-off scatter on any pair of columns. Click a point or row to inspect
it; infeasible designs are dimmed and carry their violation text.

**Configure &amp; run** &mdash; edit any parameter and re-evaluate. A design loaded
from the matrix re-evaluates to *exactly* the stored numbers, because the form
keeps the full-precision value behind the rounded display; edit a field and it
switches to what you typed.

**Stage tabs** &mdash; field maps, force-vs-gap, the pivot work integral, the
switching pulse, the module geometry in 3D, MuJoCo traces, and the
experimental validation.

## The point of it

Everything is computed by the same functions the optimiser called. The server
holds no second copy of the physics. If a number here disagrees with
`verify_best.py`, that is a bug and not a rounding difference.

Two things are surfaced deliberately rather than hidden:

* **Fidelity.** Screening carries a measured 4.2 % median force error and is
  only justified for ranking. Every result is badged with which one produced
  it, and the field map footnote gives the actual mesh size and element count
  so "screening" is visible rather than abstract.
* **Constraints.** Each gauge shows the value against the limit it must meet,
  so an infeasible design shows *how far* off it is, not just that it failed.

## Costs

First render of a figure runs real solvers; afterwards it is cached per
process.

| view | first | cached |
|---|---|---|
| field map | ~6 s | instant |
| force vs gap | ~33 s | instant |
| pivot integral | ~45 s | instant |
| experimental validation | ~165 s | instant |
| evaluate (screening) | ~5 s | &mdash; |
| evaluate (full fidelity) | ~60 s | &mdash; |
| MuJoCo dynamics | ~60 s | &mdash; |

The validation fit starts in the background at server start, so it is usually
ready before it is opened. Gap sweeps are spread across processes.
