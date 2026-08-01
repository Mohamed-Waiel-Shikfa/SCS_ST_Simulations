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
