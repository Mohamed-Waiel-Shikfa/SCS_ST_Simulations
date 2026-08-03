# Magnobots research log

## 2026-08-01 — EPM architecture review

### Correction to my earlier claim (I was wrong)

I previously wrote "you're not building an EPM" on the grounds that a single
Alnico rod flips N<->S but never turns off. That was wrong, and the literature
already in this repo settles it.

Marchese, Asada & Rus 2012 (`used articles/epm/epmClampersMIT.pdf`, p.2,
"A Comparison with Previous EPMs"), verbatim:

> "The EPMs described in this work differ from EPMs utilized in [1] and [17] in
> that a homogeneous permanent magnet core is employed. A homogeneous AlNiCo
> core facilitates establishing a continuum of clamping forces. Applications in
> [1] and [17] required a binary, on/off, modulation of attractive force
> favoring a heterogeneous core in which the polarity of AlNiCo was switched to
> cancel or reinforce the flux output of a NdFeB element."

So:

* The Knaian-style **AlNiCo + NdFeB** core is explicitly **binary on/off**. The
  AlNiCo only cancels or reinforces a fixed NdFeB. It cannot present a reversed
  external polarity, so two such faces can never repel. It is therefore
  **unusable for this project**, which needs repulsion for pivoting.
* A **homogeneous (pure-Alnico) core** gives a *continuum* of output, set by
  pulse energy. Off is reached by a reverse pulse that "reduces the remnant
  magnetism to a negligible level" (p.2).
* This was validated experimentally, not just proposed: pulse length swept
  7-35 ms, repeated on 7 separate occasions, producing a monotonic continuum of
  gap flux density and hence force (p.5, Fig. 12, Table III).

**Conclusion: the pure-Alnico, per-face polarity-controlled architecture is
correct and is the published state of the art. Keep it.**

### What the same paper says about the flux circuit

The self-demagnetisation finding stands, and this paper is direct evidence for
it rather than against it. Their design parameters (p.5, Table I):

| quantity | Marchese et al. | this project (as built) |
|---|---|---|
| material | cast Alnico 5 | Alnico 5 (LNG37) |
| magnet length `lm` | 76.2 mm | 12.5 mm |
| magnet area `Am` | 20.3 cm^2 (D_eq 50.8 mm) | 0.177 cm^2 (D 4.75 mm) |
| **aspect ratio L/D** | **1.5** | **2.63** |
| operating `Bm` | **1.2 T** (~89 % of Br) | 0.47 T (39 % of Br) |
| operating `Hm` | **24.7 kA/m** (~0.5 x Hcj) | 47 kA/m (0.96 x Hcj) |
| soft-magnetic path | 1018 steel keepers | none |

Their magnet is *squatter* than ours (L/D 1.5 vs 2.63) and yet holds 89 % of
remanence instead of 39 %, at half the coercivity instead of 96 % of it. The
entire difference is the steel keeper circuit. Quoting p.2:

> "Steel keepers (A) act to route flux around the EPMA and minimize fringing
> fields."

> "this assembly positions permanent magnet cores directly above and below the
> non-ferrous gap, lessening fringing flux and resulting in high clamping force
> to weight ratios."

Aspect ratio is thus the wrong lever. The right lever is closing the circuit.

### Switching

p.2: the magnetiser generates "an external magnetizing field several times the
material's coercivity", which supports the 3 x Hcj figure used in
`analysis/epm_feasibility_audit.py`. Their drive: 1400 turns total (350/coil,
16 AWG), 0.1 F at 76.4 V, ~29 ms pulses, and note they could not magnetise all
four elements at once - they pulsed them sequentially.

### Alnico grades (from `Alnico性能表.png`, verified)

The highest-coercivity grade in the supplier table is **LNGT36J (Alnico 8HC):
Br = 0.70 T, Hcj = 148 kA/m** - three times the coercivity of Alnico 5 at about
half the remanence. This is a genuine third option for resisting
self-demagnetisation, at the cost of a harder switch and lower ceiling force.
Worth putting in the design matrix as a material axis.

### Open questions

* Stability of a partially demagnetised state under a neighbour's stray field.
  Marchese demonstrates repeatability of *setting* the state, but their
  application never exposes a set magnet to a reversed external field from
  another EPM. In a lattice of modules it will be. This needs its own study.
* Whether an axisymmetric pot core or a planar two-rod U-core is the better
  fit for a square face.

---

## 2026-08-01 (later) — the repulsion problem

### The asymmetry is real, and it is a material problem

`analysis/check_repulsion.py` establishes two things before anything is designed
around the attract/repel asymmetry:

* with **rigid** magnetisation, |F_repel| / |F_attract| = 1.014, i.e. the two
  states are equal and opposite as the equivalent-charge argument requires. The
  force routine is therefore sound.
* the mid-plane Maxwell stress integral converges by r = 5 x R_magnet, so the
  earlier numbers were not truncated.

The entire asymmetry is therefore a **magnetisation** effect: in the repel state
the two magnets drive each other backwards along their own demagnetisation
curves. Alnico 5 has so little coercivity that its polarisation collapses from
0.85 T to 0.38 T, and force goes as J^2.

### Grade sweep (`analysis/grade_sweep.py`)

Whole supplier table, bare rods, both states. The result is a strict dominance,
not a trade-off:

| grade | Br | Hcj | attract | repel | ratio |
|---|---|---|---|---|---|
| LNG37 (Alnico 5) *as built* | 1.20 | 49 | +6.02 N | -0.28 N | 21.3 |
| LNG60 (Alnico 5-7) *named in report* | 1.35 | 60 | +9.76 N | -0.45 N | 21.7 |
| **LNGT72 (Alnico 9)** | **1.05** | **114** | **+6.60 N** | **-1.24 N** | **5.3** |
| LNGT36J (Alnico 8HC) | 0.70 | 148 | +2.87 N | -0.80 N | 3.6 |

**Alnico 9 beats Alnico 5 in BOTH states** - 4.4x the repulsion and 1.10x the
attraction - despite 12 % lower remanence, because 2.3x the coercivity means it
keeps far more of that remanence under load. Remanence sets the ceiling;
coercivity decides how much of the ceiling survives a reversed neighbour.

The high-remanence grades that look best on a datasheet (LNG60, the grade named
in the report and on the poster) are the *worst* choice for a design that has to
repel.

### Combined with the circuit (`analysis/combined_fix.py`)

|  | attract | repel | asymmetry |
|---|---|---|---|
| as built (LNG37, bare) | +5.96 N | -0.28 N | 21.3 : 1 |
| grade change only | +6.63 N | -1.24 N | 5.4 : 1 |
| grade + steel pot core | **+9.24 N** | **-1.44 N** | 6.4 : 1 |

Net: **attraction 1.6x, repulsion 5.1x**, and the asymmetry falls from 21:1 to
6.4:1. Incidentally the LNGT72 pot core also solves ~10x faster than the LNG37
one, because the magnet no longer sits on the knee - the numerics get easier
for the same reason the physics gets better.

Note a residual trade: LNG37 + pot core gives the highest raw attraction
(11.86 N) but only 0.72 N repel. Since repulsion is the binding constraint,
LNGT72 is the right pick.

### Recommendation

1. Change grade to **Alnico 9 (LNGT72)**. Cheapest possible change - it is a
   different part number, not a redesign - and it is a strict improvement.
2. Add a **1018 steel return path**. 1 mm wall is enough; it peaks at 1.54 T,
   well below saturation.
3. A ~6:1 asymmetry still remains. Locomotion sequencing should not assume
   symmetric push/pull; this needs to feed into the Stage 2 pivot model.


---

## 2026-08-02 - Module geometry, confirmed with the author

The module is the **intersection of three mutually orthogonal regular n-gon
prisms**, not a cube.

* cross-section is a regular n-gon with **n = 8 + 4k** (8, 12, 16, 20 ...);
  four-fold symmetry is needed so the three orthogonal rings close consistently
* **3n - 6 square faces** (18 / 30 / 42 / 54), one EPM on each
* **pivot angle = 360/n**, not 90 degrees.  This is the whole point of the
  geometry: a smaller step lifts the centre of mass less
* fits inside a **5 cm cube**; homogeneous, polygamous, lattice, solid state
* latch by attraction, locomote by repulsion

Consequence for the design space: `n_gon` and `r_face` replace any cube side
length, and n is a real design variable, not a styling choice.

A packaging error followed from getting this wrong the other way: the bounding
cube was computed as `2 r_face / cos(pi/n)`.  That expression is the polygon
CIRCUMRADIUS, which is the right quantity for the pivot lift but the wrong one
for the envelope - the axis directions are themselves ring normals, so the box
is exactly `2 r_face`.  The module was being reported 8 % larger than it is.

---

## 2026-08-02 - The solver was the bottleneck, and it was the wrong shape

The nonlinear magnet solve took 668 s for a single pot-core design at full
fidelity, and stalled outright on short rods.  Both problems had the same root
cause: the two nonlinearities were being solved together.

* the field is **linear** in the slab remanences for a frozen reluctivity, so
  the entire magnet coupling is an n-by-n permeance matrix M, obtained with n
  back-substitutions on one factorisation.  The material law then reduces to
  `x = t(Mx)` on n unknowns with an analytic Jacobian - no field solves at all.
* the iron is nonlinear in |B|, so M drifts.  That is an outer loop.

Solving them in that order is 100x faster (3-70 s), and the stalls disappear
because the Jacobian is no longer a finite difference taken across the knee of
the demagnetisation curve.

Two real bugs fell out:

1. **the iron loop returned stale fields.**  It updated the reluctivity and
   then returned the fields computed with the PREVIOUS one, so the permeance
   matrix and the fields described slightly different problems.  Hard floor of
   2e-3 T on the outer residual; deeply saturated pot cores looked
   unconvergeable.  One extra linear solve at the end fixes it.
2. **the screening mesh was tied to the far-field box, not the magnet.**  For a
   bare rod that gave 1.5 elements across the diameter and a 48 % force error,
   while pot cores - which have a larger outer radius - got 11 %.  A structural
   bias like that does not cancel in a ranking; it silently favours whichever
   architecture happens to be meshed better.

`screening_study.py` now measures this rather than assuming it: over 24 designs
spanning both architectures, screening at `h = min(D,L)/6` has a median error of
4.2 % and a **Spearman rank correlation of 0.992** against full fidelity, at
about 1/8 of the cost.  That is what makes the optimiser's use of a cheap
surrogate defensible.

---

## 2026-08-02 - The pivot does not work, and three models were hiding it

The pivot was the only claim in the pipeline that had never been simulated.
Running it in MuJoCo on the real polyhedron broke three separate models, every
one of which had been flattering the design.

**1. The static pivot criterion was optimistic by ~50x.**  It multiplied peak
force by arc length, assuming both driving faces held full force through the
whole 45 degree roll.  They cannot: the trailing pair separates as soon as the
module tips, and force falls off with gap far faster than arc length grows.

**2. A first correction was wrong the other way, by ~5x**, because it used the
analytic charge-disc fall-off.  As a repelling pair separates the two magnets
stop demagnetising each other and their polarisation **recovers**, so repulsion
decays much more slowly than a fixed-strength model predicts.  The fall-off has
to come from the FEM too.  With that, predicted work is within 25 % of the work
the simulation actually delivers, measured by integrating F.v over the run.

**3. `EPMSpec.force` clamped beyond its table**, because that is what numpy's
`interp` does.  A module 44 mm away was still being pushed at the 4 mm force -
an infinite energy source.  Modules sailed over four gravitational barriers
they had a quarter of the energy to cross.  Now a 1/r^4 dipole tail.

**4. The wrench model was not conservative.**  It rescaled a charge-disc sum by
a pose-dependent factor, which is not the gradient of anything, so it pumped
energy around a rolling cycle; and at contact its reference collapsed and the
sign flipped, blowing latched modules apart.  Rotating configurations now use a
central force between pole-face centres with the FEM magnitude, conservative by
construction, and `verify_dynamics` tests that by integrating work around a
closed loop in configuration space.

### The finding that survives all of that

With the corrected models, **neither the as-built design nor the previous
optimiser winner clears the pivot criterion**:

| design | pivot work / barrier | simulated step |
|---|---|---|
| as built (LNG37 bare rod, n=8) | 1.39 | rocks 5 deg, falls back |
| previous GA winner (LNGT44 pot core, n=8) | 1.24 | rocks 5 deg, falls back |

Energy is necessary but not sufficient: the work has to arrive as rotation
about the pivot edge, and part of it goes into sliding and friction.  A ratio
near 1 is not enough; the constraint is now set at 1.5 and the optimiser can
see it.

**Locomotion by magnetic repulsion alone is marginal at n = 8.**  Raising n
lowers the barrier (lift falls as `r(1/cos(pi/n) - 1)`: 1.60 mm at n=8, 0.24 mm
at n=20) but adds faces, drivers and mass.  That trade is now inside the GA
rather than assumed away.

---

## 2026-08-03 - The OFF state: safe from neighbours, not safe from itself

Per-face polarity control assumes a face can be switched off and STAY off.
That had never been tested, and it is load-bearing: if a live neighbour
re-magnetises an off face, latching, releasing and sequencing all fail.
`off_state_study.py` tests it, keeping the exact results separate from the
modelled ones.

### Neighbours are not the problem (exact FEM, no free parameters)

Field imposed inside an off magnet by a live neighbour 0.1 mm away:

| design | Hcj | H in OFF face | as fraction of Hcj |
|---|---|---|---|
| as built, LNG37 bare rod | 49 kA/m | 14.8 kA/m | **0.30** |
| LNGT44 pot core | 122 kA/m | 20.6 kA/m | **0.17** |
| LNGT72 pot core | 114 kA/m | 25.8 kA/m | **0.23** |

All comfortably below the coercivity, and the induced remanence estimated from
a Preisach-style virgin curve is 0-1.2 % of Br.  The off state survives its
neighbour, and it survives it for a satisfying reason: the flat top of an
Alnico demagnetisation curve means almost no domains have switching fields in
the lower part of the range.  **The same curve shape that makes the material
switchable also protects the off state** - up to about 0.8 Hcj, beyond which
the protection collapses.

A bonus: the steel return path SHIELDS the off face rather than funnelling
flux into it.  Same material and geometry, pot core versus bare rod, 0.17 Hcj
against 0.24 Hcj - the pot core sees **0.70x** the field.  The keeper gives the
neighbour's flux an easier path than through the magnet.

Caveat: the FEM is axisymmetric, so this is one neighbour.  A face in a full
lattice has several, and that superposition has not been computed.

### The magnet is the problem (exact, using the existing recoil model)

A magnet pulsed until J = 0 does **not** stay at zero.  When the pulse ends it
recoils along a line of slope mu_rec back to its operating point, and recoil
raises J:

| design | operating point | J after recoil | as % of Br | residual force | vs module weight |
|---|---|---|---|---|---|
| LNG37 bare rod | -14.8 kA/m | 0.129 T | 10.7 % | 0.018 N | 0.02 |
| LNGT44 pot core | -20.6 kA/m | 0.127 T | 14.5 % | 0.091 N | 0.10 |
| LNGT72 pot core | -25.8 kA/m | 0.111 T | 10.6 % | **0.422 N** | **0.48** |

So an untuned off pulse leaves LNGT72 holding nearly half the module's weight.
That is not enough to defeat gravity outright, but it is a large parasitic
force in a machine whose entire job is to attach and detach on command.

This is a **control** requirement, not a materials one: the pulse must
overshoot past J = 0 so the magnet lands at zero AFTER recoil.  Marchese et al.
arrived at their off state by empirically sweeping pulse length, which is
exactly consistent with this.

### The part that should worry the optimiser

The recoil is `mu0 (mu_rec - 1) (Hcj - |H_op|)`, so **higher coercivity means a
larger residual for the same pulse error**.  The GA is being driven towards
high-coercivity Alnico by the repulsion and pivot objectives, and that same
choice makes the off state harder to hit accurately.  Nothing in the current
objective set sees this.  It is a real tension and it is not yet priced.

### A solver bug this exposed

`solve()` scaled its convergence tolerance on `saturated()[0]` - slab 0 only.
A mixed configuration, one face off next to a live one, has J = 0 in the first
region, so the tolerance collapsed to 1e-13 and every such solve failed for a
purely numerical reason.  Now scaled on the strongest slab.  Worth noting that
this is the normal state of a module in a lattice, so it would have bitten any
attempt to model an assembly rather than an isolated pair.
