# Design explorer

An interactive view of the whole pipeline: browse the design matrix, feed any
configuration back through the real solvers, and watch each stage in the order
it actually runs.

```bash
python analysis/viz/server.py            # then open http://127.0.0.1:5173
python analysis/viz/server.py --port 8080
```

Needs `flask` and `matplotlib` on top of `analysis/requirements.txt`.  There
are no front-end dependencies at all - no build step, no CDN, no network
access at runtime.

## The tabs are the pipeline

The stage tabs are numbered and ordered because the pipeline is:

**0 Module** &mdash; the physical assembly, built first because everything
downstream measures itself against this geometry.  A cutaway 3-D view of every
component with its real dimensions: magnets, multi-layer coils, steel pole
cups and back plates, the capacitor bank, the battery, the driver board, and
the envelope reserved for electronics.  Each component kind can be toggled,
and the six latching faces are ringed &mdash; those are the only faces two
modules may join on, which is what keeps an assembly on a cubic lattice.

**1 Magnetics** &mdash; the axisymmetric field map and force-gap curve, plus a
full 3-D solve of two interacting modules at any relative pivot angle, with a
volumetric field viewer (point cloud, orthogonal slices, or vectors).  An
angle sweep plots force and torque against the pivot angle, which is the thing
no axisymmetric model can produce.

**2 Driver** &mdash; the transient switching circuit, integrated in time, on
the magnetic circuit the field solve measured.  Coil current, bank voltage,
field in the magnet and the gate waveform, against the switching threshold.
Optionally searches pulse programmes for one that reaches the threshold for
less energy.

**3 Mechanics** &mdash; runs only if switching succeeded.  Four rolling
configurations (horizontal, and vertical from the bottom, side and top),
optionally under all three drive schemes, each in its own MuJoCo viewport with
independent playback: play, pause, single step forward and back, scrub, and
loop.

**Design matrix** and **Configure & run** sit before the stage tabs, and
**Validation** after them.

## Things it does deliberately

**Hover anything.**  Every parameter label and every result key carries a
tooltip saying what the quantity is physically, which way the design moves
when it goes up, and what that costs.  The J-against-B distinction, `n_eff`,
the difference between `e_bank` and `e_required` - all the things that are not
obvious from the name.

**The local optimiser is a button.**  *Refine locally* encodes the current
form into a genome, runs the same compass search the GA runs between its
generations, and shows which parameters moved and by how much before you
accept it.

**Fidelity is always visible.**  Screening carries a measured 4.2 % median
force error and is only justified for ranking.  Every result is badged with
which produced it.

**Disagreements are shown, not smoothed.**  The 3-D solver and the
axisymmetric FEM disagree for pot-core geometries - 22 % on attraction and
about 2.6x on repulsion - and neither is validated against measurement for
that case; the field panel says so whenever steel is included.  A design whose winning material is a "lit" row is a research
result, not a bill of materials, and the material table marks it.

**A gated design says so.**  If switching fails, the Mechanics tab explains
that mechanics was not run and why, rather than showing an empty plot.

## The point of it

Everything is computed by the same functions the optimiser called.  The server
holds no second copy of the physics, and the front end holds none either - it
draws what the endpoints return.  If a number here disagrees with
`verify_best.py`, that is a bug and not a rounding difference.

## Costs

First render of a figure runs real solvers; afterwards it is cached per
process.

| view | first | cached |
|---|---|---|
| module assembly | ~2 s | instant |
| axisymmetric field map | ~6 s | instant |
| force vs gap | ~33 s | instant |
| 3-D field solve | ~5 s | instant |
| angle sweep (14 solves) | ~60 s | &mdash; |
| switching transient | ~8 s | &mdash; |
| rolling, one drive scheme | ~25 s | &mdash; |
| rolling, all three schemes | ~70 s | &mdash; |
| evaluate (screening) | ~6 s | &mdash; |
| evaluate (full fidelity) | ~60 s | &mdash; |
| experimental validation | ~165 s | instant |

The validation fit starts in the background at server start, so it is usually
ready before anyone opens the tab.

## Files

| file | what |
|---|---|
| `server.py` | endpoints; every one calls the real pipeline |
| `param_info.py` | what each parameter means, its effect and its cost |
| `plots.py` | matplotlib figures for the axisymmetric views |
| `static/scene3d.js` | the 3-D renderer: painter's algorithm on a 2-D canvas |
| `static/app.js` | tabs, forms, plots, playback |
| `static/style.css` | styling |

`scene3d.js` is about three hundred lines and replaces a WebGL library on
purpose: everything drawn is cylinders, tubes, boxes, a convex hull and point
clouds, and sharing one camera and one lighting model between the module
viewer, the field viewer and the MuJoCo playback keeps the three visually
consistent and identical under the mouse.
