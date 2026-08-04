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

---

## 2026-08-03 - The GA run, and the pivot verdict reversed

994 evaluations, seed 1, pop 40, 25 generations.  519 feasible.  Verified at
FULL fidelity, not screening.

### What the search chose, unprompted

| axis | outcome |
|---|---|
| material | **LNGT72 (Alnico 9) in 494 of 519** feasible designs |
| circuit | **pot core in 519 of 519** - a bare rod never survives |
| n | n=12 in 296, n=8 in 223 |

The material result independently reproduces the hand analysis from the grade
sweep, which is worth something: two different routes to the same answer.

Rejection reasons: hold 197, **pivot 152**, demag margin 92, electronics do not
fit 31.  The pivot constraint is doing real work, and the feasible set sits on
its boundary - min 1.50, median 2.00.

### Full fidelity confirms screening, within its stated error

| design | Fa screen | Fa full | error | pivot screen | pivot full |
|---|---|---|---|---|---|
| best scalar | 27.82 N | 26.71 N | -4.0 % | 2.64 | 2.55 |
| lightest | 8.71 N | 8.17 N | -6.2 % | 2.31 | 2.25 |

`screening_study.py` predicted a 4.2 % median error.  It delivered 4-6 %.  The
surrogate is behaving as characterised, which is the point of having measured
it rather than assumed it.

**Best scalar:** LNGT72, n=8, r_face 21.6 mm, D 7.42 x L 7.00 mm, pot core.
26.7 N attract, 2.63 N repel, 142 g, holds 19x its weight.

**Lightest feasible:** LNGT72, n=8, r_face 18.3 mm, D 4.30 x L 4.03 mm.
8.17 N attract, 42 g, holds 20x its weight.  This is the one a builder would
reach for: a third of the mass for a third of the force, same hold ratio.

### The pivot verdict is REVERSED

The previous entry concluded the pivot does not work at n = 8.  **That was
wrong, and it was wrong because the force model was.**  With the conservative
pair wrench and the 1/r^4 far tail in place, every design pivots:

| design | ratio | simulated | verdict |
|---|---|---|---|
| as-built baseline | 1.39 | 43.8 deg of 45 | **pivots** (97 %) |
| GA lightest | 2.25 | 46.3 deg of 45 | **pivots** (103 %) |
| GA best scalar | 2.55 | 140.1 deg of 45 | overshoots, 3.1 steps |
| GA best pivot margin | 3.00 | 132.0 deg of 45 | overshoots, 2.9 steps |

The baseline pivots at ratio 1.39, below the 1.5 constraint.  So the constraint
is slightly conservative at the bottom - which is the safe direction, but worth
knowing.

### The new problem is control, not force

Holding the repel drive on and sweeping n, with the same magnet:

| n | ratio | simulated, as % of one step |
|---|---|---|
| 8 | 2.55 | 296 % |
| 12 | 3.78 | **100 %** |
| 16 | 4.91 | 960 % |
| 20 | 5.95 | 128 % |

Not monotonic, not close to it.  Tumbling is **chaotic**: whether the module
stops on the next face or carries on depends on exactly how it lands and
whether it catches an edge.  So the number of steps is not a design parameter
and cannot be made one by adding force.

The consequence is a timing requirement, and it is now quantified: one step
completes in **140-280 ms**, faster for the stronger designs.  The drive has to
be cut inside that window.  Locomotion is a pulsed, timed operation - it cannot
be done by switching a face on and waiting.

**This says `pivot_ratio` should be a BAND, not a floor.**  Something like
1.3 to 2.0.  Below it the module cannot climb; above it, the extra magnet mass
and switching energy buy nothing but a harder control problem.  The current
objective set rewards force without limit and the search duly spent 142 g
getting 26.7 N when 42 g and 8.2 N pivots just as well.

### A mode that does not work

Energising the NEXT face pair round the ring to pull the module over does
nothing at all - 0.0 degrees in every case.  At n = 8 that pair starts 11 mm
apart, and the force there is three orders below the contact force.  **Pivoting
must be driven by repulsion from the face the module is already standing on.**
There is no reach-ahead mode.


---

# Second pass: the pipeline rebuilt so each stage feeds the next

The first pass established the physics one stage at a time.  This pass fixed
the joins between them.  Almost every finding below is about something a stage
was ASSUMING because it could not see what the stage before it had computed.

## The material space was pre-judged

The table held Alnico and little else, so the central trade - coercivity buys
repulsion and costs switching energy - could only be explored over a threefold
range of Hcj.  It now spans fifty: 29 grades in seven families, everything
commercially available below 2000 kA/m, from Cunife and Vicalloy at 38 kA/m
through the Alnicos, Mn-Al-C and the ferrites to SmCo at the ceiling.

Density is now per-material and it matters more than expected.  Ferrite is
4900 kg/m^3 against Alnico's 7300 and SmCo's 8400, and magnet mass is a large
share of module mass, so the lighter grades get a mass credit that partly
offsets their weaker remanence.

The rare-earth rows are deliberately hopeless and are kept for that reason.
Sm2Co17-30 needs about thirty times the ampere-turns of Alnico in the same
geometry; leaving it in lets the optimiser demonstrate where the wall is
instead of the wall being asserted.

## The winding was not a real object

Turns came out as `(l_mag / d) * (build / d)` with `build` defaulting to the
KEEPER WALL THICKNESS.  Two things were wrong with that.  The winding depth
was welded to a structural dimension that has no reason to equal it, so the
search could not trade one against the other.  And nothing distinguished one
layer from many: turns was a smooth product, so the model never saw that each
new layer sits at a larger radius and therefore costs more copper per turn.

With the layers made explicit, **turns per ohm falls steadily with winding
depth** - 205 for one layer, 127 for twelve, on a 4.2 mm magnet.  A lumped
model misses this entirely and over-rewards deep coils.

The winding also occupies radial space, which it previously did not.  That
space comes out of the magnet or the keeper, and the steel annulus now starts
outside the copper rather than at the magnet surface.

## One number carries the whole magnetic circuit

The switching stage used to compute an air-cored solenoid inductance.  A real
EPM coil is wound on a magnet of recoil permeability 1.04 to 4.0, inside a
steel return path, next to a neighbour that is also a magnet.

All of that enters through a single quantity: the effective demagnetising
factor `n_eff` of the circuit the magnet actually sits in.  With
`n_eff = R_ext / (R_mag + R_ext)`,

    H_self  = -n_eff J / (mu0 mu_rec)          the operating point
    F_coil  =  H L / (1 - n_eff)               ampere-turns for a target H
    L_coil  =  N^2 (1 - n_eff) / R_mag

and the useful part is that **n_eff is measurable from the Stage 1 field solve
at no extra cost**: the solved state already reports the volume-averaged J and
H in the magnet, so `n_eff = -mu0 mu_rec H / J`.  The driver is now designed
against the circuit the FEM computed rather than a guessed fringing permeance.

For a bare rod this reduces to `n_eff = mu_rec * N_d`.  That factor of mu_rec
is the reason a high recoil permeability is not free: Alnico 5 with mu_rec = 4
sits four times further down its own load line than a ferrite of identical
shape.  The material changes the electromagnet's job as well as the threshold
it has to clear.

### The neighbour helps, but less than it first appeared

An early version modelled a latched neighbour as REPLACING the local return
path, which made switching while latched look much cheaper.  That is wrong:
the neighbour is a SECOND path in PARALLEL with the local fringe path, so it
can only lower n_eff, and it lowers it modestly - from 0.266 to 0.214 in a
representative pot core - because the neighbour's own magnet is a long
low-permeability leg.  Getting this backwards would have sized the driver
against a circuit that does not exist.

## Pulse trains

The drive was always a single capacitor discharge: close the switch, take the
first peak.  Integrating the loop in time instead allows a train, and a train
at the right frequency and duty **reaches the same switching field for about
80 % less energy out of the bank**, because the bank is not dumped into
resistance on one swing.  Pulse frequency, duty and count are now genome
variables rather than assumptions.

One bug worth recording: with no floor on the coil inductance, a degenerate
design drove the RK4 step past the range of a double and filled the trace with
NaN.  Because a NaN peak field compares false against the threshold, those
designs were silently marked unable to switch rather than raising anything -
a whole corner of the search space quietly removed.  Both a floor and an
explicit divergence flag are now in place.

## Three dimensions, and what it cost to trust them

The axisymmetric FEM can only see two EPMs on a shared axis.  A pivot is not
that: the driving pair rotates away from coaxial while a second pair swings
towards it, and there is no axis of revolution anywhere in the geometry.

`fem3d.py` is a magnetostatic method of moments on the magnetised bodies
alone - cells of uniform magnetisation whose field is available in closed
form, so there is no mesh in the air at all.  Verifying it found four bugs,
every one of which produced numbers that looked entirely plausible:

1. The kernel's Hz term used `atan2` rather than the principal branch.  The
   corner sum cancels the resulting jumps of pi only OUTSIDE the rectangle's
   shadow, so the error was invisible in the far field and gave a cube a
   self-demagnetising factor of -2/3 instead of 1/3.
2. The magnet's irreversible-loss history latched onto the first iterate's
   overshoot from the Br starting guess, permanently demagnetising the magnet
   with a field that was never a physical operating point.  It converged to
   J = 0.45 T where the validated one-dimensional solver gives 0.62 T.
3. An angled pair was rotated about the pole-face CENTRES rather than the
   shared rim, so beyond `asin(gap / r_pole)` - about three degrees - the two
   magnets interpenetrated.  Every angled result before this was meaningless,
   and the symptom was scatter by a factor of eight between discretisations
   rather than anything that looked like an error.
4. Square tiles of a circular pole reach 25 % past the magnet radius, so with
   a 0.5 mm clearance the outer tiles ran into the steel annulus - 50 to 96
   overlapping cell pairs.  Polar dicing fixed it, and an explicit
   separating-axis overlap check now guards every scene.

Simple iteration also had to be replaced by a damped Newton step with
continuation.  Soft iron has a susceptibility around 2000, so the fixed point
has a gain far above one and diverges; with a pot core the plain iteration was
still at a residual of 1.1 after two thousand passes while reporting plausible
numbers.  Continuation is needed on top, because two like poles a tenth of a
millimetre apart drive each other past the knee and the iteration flips
between fully magnetised and fully demagnetised.

### An open discrepancy, not resolved

After all of that: **for magnets with no return path the 3-D solver is
verified** - 4 to 7 % on the operating point against the validated 1-D solver,
within 4 % on attraction and 10 % on repulsion against the axisymmetric FEM,
and stable across discretisations to between 1 and 15 % at every angle,
tightest at the pivot angles that matter.

**With a steel pot core the two solvers disagree.**  The magnet operating
point still agrees to about 1 %, but the force does not: the 3-D solver reads
about 22 % high on attraction and roughly 2.6 times high on repulsion.  Forcing both
to use linear iron changes nothing, so it is not saturation.  Since J agrees
and F does not, the two models are splitting the flux differently between the
pole face and the annulus rim - and the repelling force is a small difference
between a large pole-to-pole repulsion and a large pole-to-rim attraction, so
a modest error in that split becomes a large error in the total.

Neither model is validated against measurement for the pot-core geometry: the
experimental data in this repository is for bare rods.  This is therefore
recorded as an open discrepancy rather than averaged away.  The pipeline takes
all magnitudes from the axisymmetric solver and uses the 3-D solver only for
the angular dependence, computed magnets-only, and for the volumetric field
shown in the viewer.

## Faces, and which ones are allowed to do what

Exactly six of the `3n - 6` faces may LATCH: the ones on the coordinate axes,
shared by two of the three rings.  Two modules joined axis-face to axis-face
have parallel frames and the assembly stays on a cubic lattice; a joint on any
other face would fix two modules at an oblique angle and the lattice would
never close again.  This holds for every n - eight faces or forty-two, always
six latching.  The other faces are what the module ROLLS on.

A subtle error here: the face on the neighbour that mates with face `n` is the
REFLECTION of `n` through the joint plane, `n - 2 (n . a) a`, not `-n`.  For
the mating faces themselves the two coincide, which is why it stayed hidden;
it only shows on the neighbouring pairs, where taking `-n` picks a face on the
far side of the module that never comes near anything.

## Rolling: gravity does not treat the cases alike

Four configurations are now simulated - one horizontal, and three vertical
(from the bottom, from the side, from the top) - and three drive schemes are
run rather than one being assumed.  The result for the current design:

* **push off** (the specified scheme: reverse the face-to-face pair AND the
  trailing neighbour) rotates the module - 44 of 45 degrees in 183 ms on the
  ground - but RELEASES it.  On a floor the ground catches it.  On a wall
  nothing does, and it falls.
* **trailing only** (keep the mating pair attracting) holds in every
  configuration but does not rotate at all: 0.1 degrees.  The latch is far too
  strong to pivot against.
* **reach** behaves like push off.

So for this design no scheme both rotates and stays attached, and wall
climbing does not work even though the static pivot bound is nearly met.  That
is a real result about the actuation scheme rather than a limitation of the
simulation, and it is the reason all three are run.

One modelling point that took a while: the working air gap is a MAGNETIC
quantity, the distance between pole faces.  Separating the two shells by it as
well removed the mechanical contact and with it the friction, and since the
magnetic pull between two side-by-side modules acts along the horizontal
joint, there was then nothing at all resisting gravity.  Every wall case fell
2.7 metres for a reason that had nothing to do with magnetics.

## The gate that pays for everything else

The stages now run module, magnetics, switching, mechanics, and **mechanics is
skipped entirely when switching fails**.  A module whose coil cannot reverse
its magnet is not a robot however well it holds, so simulating its gait is
wasted time - and mechanics is the expensive stage.  In a random population
roughly two thirds of designs never reach it.

## What the more accurate model says about the previous winner

The revised pipeline is harsher, and it should be: the winding now has mass
and takes radial space, the steel starts outside the copper, and densities are
per-material.  The design that was previously reported as the winner now comes
out at a pivot ratio of 1.46 against the required 1.5 - marginal rather than
feasible.  The binding constraints across the search are the pivot bound and
the electronics packing, in that order.

## The run, and one thing to be careful reading

1879 distinct designs over 50 generations of 30, seed 1, 42 minutes on 18
cores.  The population was entirely infeasible until generation 20 - the
constrained-domination machinery spent twenty generations driving total
violation down before anything crossed the line - and reached 30 of 30
feasible by generation 26.  The binding constraints throughout were the pivot
bound and the electronics packing, in that order.

The final local search moved the winner from a merit of 0.114 to 0.191 in 200
evaluations, walking r_face up twice, the clearance fraction down, and the
bank voltage up three times.  That is a 68 % improvement on a design the GA
had already converged on, and it is the clearest evidence that a
population-based search alone leaves real value on the table: polynomial
mutation only makes small steps by luck.

**Reading the transient of a Pareto design.**  The bank voltage is a free
variable and the energy objective normalises it away: `e_required` is the
energy needed at `v_need`, the voltage that just reaches the switching
threshold, and the driver is selected for `v_need` too.  So a design can sit
on the front with `v_cap` at its upper bound and a switching margin of
thirteen, and the transient recorded for it is then grossly over-driven -
the winning design reports 636 A and a nominal 9.7 T in the iron, which is
not a physical operating point but the simulation of a bank thirteen times
larger than the design needs.  The numbers that mean something for that design
are `v_need` (22 V), `e_switch` (10.5 mJ per face), and the driver actually
selected.  The `switched` gate deliberately stays at the specified `v_cap`,
because that is what makes `v_cap` a real design variable rather than a
derived one.

This is a reporting hazard rather than an error - the objective, the driver
selection and the mass budget are all consistent with `v_need` - but it is
worth knowing before quoting a peak current out of the matrix.
