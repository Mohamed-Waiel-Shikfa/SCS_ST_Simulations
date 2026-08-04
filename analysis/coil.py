"""The coil: a real multi-layer winding, and the magnetic circuit it drives.

What was wrong before
---------------------
The previous switching stage computed turns as ``(l_mag / d) * (build / d)``
with ``build`` defaulting to ``t_steel``.  Two things were wrong with that.

* The winding build was tied to the *keeper wall thickness*, which is a
  structural quantity with no reason to equal the space available for copper.
  The optimiser could not trade a taller winding against a thinner keeper
  because moving one moved the other.
* Nothing in the model distinguished one layer from many.  Turns came out as a
  smooth product, so the search never saw that the mean turn length grows with
  every layer added - the outer layers of a deep winding cost more copper per
  turn than the inner ones and are worth less.  With that effect missing the
  model has no reason to prefer any particular winding depth, and every design
  the search produced behaved like a single ideal layer.

This module makes the winding explicit: an integer number of layers, a real
turns-per-layer from the insulated wire diameter, and a wire length summed
layer by layer over the true mean turn diameter.

The magnetic circuit
--------------------
The ampere-turns a coil needs are not a property of the coil alone.  The same
NI drives a very different field into the magnet depending on what the flux
path looks like, and three things that were previously ignored change it by
more than a factor of two:

* **The steel.**  A pot core closes the return path, so the reluctance the coil
  works against is the magnet plus a short iron path instead of the magnet plus
  a long air path.  The same coil reaches the switching threshold at a fraction
  of the current.
* **The magnet's own permeability.**  Recoil permeability ranges from 1.04 for
  SmCo to 4.0 for Alnico 5 across the material table.  The magnet leg of the
  circuit is four times more reluctant for the rare earths than for Alnico even
  before their higher coercivity is considered, so the material affects the
  electromagnet twice over.
* **The neighbour.**  A latched module closes the circuit through its
  neighbour's steel and magnet.  Switching against a neighbour is not the same
  problem as switching in free space, and it is the one that actually happens.

Saturation is included: once the flux the coil is pushing exceeds what the
steel cross-section can carry, the iron stops helping and the incremental
reluctance climbs towards that of air.  That is what limits how much a bigger
capacitor can buy.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from axisym_fem import steel_nu  # noqa: E402
from magnet_force import cylinder_demag_factor  # noqa: E402

MU0 = 4.0e-7 * np.pi
RHO_CU_20C = 1.68e-8          # ohm m
ALPHA_CU = 0.00393            # per K
RHO_CU_MASS = 8960.0          # kg/m^3

# Enamel build.  Grade 2 (heavy) magnet wire is about 8 % over bare diameter
# at these sizes; grade 1 is about 5 %.  Grade 2 is assumed because the coil
# sees hundreds of volts across a few layers during the pulse.
INSULATION = 1.08

# Layer pitch.  Successive layers nest into the grooves of the one below
# rather than stacking squarely, so the radial pitch per layer is sqrt(3)/2 of
# the wire diameter for an ideal orthocyclic winding.  Real machine winding
# achieves somewhere between that and 1.0; 0.92 is a practical figure and is
# conservative in the direction that matters (it makes the coil taller).
LAYER_PITCH = 0.92


@dataclass
class Winding:
    """A physical multi-layer coil."""

    n_layers: int
    turns_per_layer: int
    n_turns: int
    wire_d: float             # bare copper diameter
    wire_d_ins: float         # over the enamel
    r_bore: float             # inner radius of the winding
    build: float              # radial thickness of the winding
    length: float             # axial length of the winding
    wire_length: float
    resistance: float         # at the design temperature
    mass: float
    fill_factor: float        # copper area / winding window area
    mean_turn_d: float

    @property
    def r_outer(self):
        return self.r_bore + self.build

    def summary(self):
        return (f"{self.n_layers} layers x {self.turns_per_layer} turns = "
                f"{self.n_turns} turns of {self.wire_d*1e3:.2f} mm, "
                f"build {self.build*1e3:.2f} mm, {self.resistance:.3f} ohm, "
                f"{self.mass*1e3:.1f} g, fill {self.fill_factor:.2f}")


def wind(r_bore, length, wire_d, n_layers, temp_c=25.0):
    """Lay ``n_layers`` layers of ``wire_d`` wire on a ``r_bore`` former.

    The mean turn diameter is accumulated layer by layer rather than taken at
    the mean radius of the whole winding.  For a deep coil on a small bore the
    difference is large: 8 layers of 0.3 mm wire on a 2 mm bore have an outer
    turn 2.2x longer than an inner one, so a lumped mean overstates the turns
    obtainable per ohm by about 15 %.
    """
    n_layers = max(int(n_layers), 1)
    d_ins = wire_d * INSULATION
    turns_per_layer = max(int(np.floor(length / d_ins)), 1)
    pitch = d_ins * LAYER_PITCH
    build = n_layers * pitch

    wire_length = 0.0
    for k in range(n_layers):
        r_mid = r_bore + (k + 0.5) * pitch
        wire_length += turns_per_layer * 2.0 * np.pi * r_mid

    a_wire = np.pi * (wire_d / 2.0) ** 2
    rho = RHO_CU_20C * (1.0 + ALPHA_CU * (temp_c - 20.0))
    R = rho * wire_length / a_wire
    mass = wire_length * a_wire * RHO_CU_MASS

    n_turns = n_layers * turns_per_layer
    window = build * length
    fill = (n_turns * a_wire) / window if window > 0 else 0.0
    mean_turn_d = wire_length / (n_turns * np.pi) if n_turns else 0.0

    return Winding(n_layers=n_layers, turns_per_layer=turns_per_layer,
                   n_turns=n_turns, wire_d=wire_d, wire_d_ins=d_ins,
                   r_bore=r_bore, build=build, length=length,
                   wire_length=wire_length, resistance=R, mass=mass,
                   fill_factor=fill, mean_turn_d=mean_turn_d)


def layers_for_build(build, wire_d):
    """How many layers fit in a given radial build."""
    return max(int(np.floor(build / (wire_d * INSULATION * LAYER_PITCH))), 1)


# --------------------------------------------------------------------------
# Magnetic circuit
# --------------------------------------------------------------------------
# The circuit is parametrised by ONE number: the effective demagnetising
# factor ``n_eff`` of the magnet inside whatever circuit it actually sits in.
#
# Derivation.  Inside the magnet B = J + mu0 mu_rec H, and around the loop
# H L + Phi R_ext = F_coil.  Eliminating Phi = A (J + mu0 mu_rec H) and writing
# R_mag = L / (mu0 mu_rec A) gives
#
#     H = [F_coil - A R_ext J] / (mu0 mu_rec A (R_mag + R_ext))
#
# so with n_eff := R_ext / (R_mag + R_ext) the two results that matter are
#
#     H_self  = -n_eff J / (mu0 mu_rec)        (no coil: the operating point)
#     F_coil  =  H L / (1 - n_eff)             (ampere-turns for a target H)
#     L_coil  =  N^2 (1 - n_eff) / R_mag
#
# Everything the steel, the working gap and a latched neighbour do to the coil
# enters through n_eff and nothing else.  That matters because n_eff is
# MEASURABLE from the Stage 1 FEM at no extra cost: the solved state already
# reports the volume-averaged J and H in the magnet, and
#
#     n_eff = -mu0 mu_rec H / J
#
# So the switching stage does not have to guess a fringing permeance - it reads
# the real one out of the field solve that has already been paid for.  The
# analytic estimate below exists only for the pre-screen, where no FEM has been
# run yet.
# --------------------------------------------------------------------------
@dataclass
class MagCircuit:
    """Lumped reluctance model of one EPM, optionally facing a neighbour."""

    n_eff: float              # effective demagnetising factor of the circuit
    a_magnet: float
    l_magnet: float
    mu_rec: float
    has_steel: bool
    has_neighbour: bool
    source: str = "estimate"  # "estimate" or "fem"
    a_steel: float = 0.0
    b_steel: float = 0.0      # flux density in the iron at the switching point

    @property
    def r_magnet(self):
        return self.l_magnet / (MU0 * self.mu_rec * self.a_magnet)

    @property
    def r_ext(self):
        n = min(max(self.n_eff, 1e-6), 1 - 1e-9)
        return self.r_magnet * n / (1.0 - n)

    @property
    def r_series(self):
        return self.r_magnet + self.r_ext

    def inductance(self, n_turns):
        """Coil inductance including the magnet and its return path."""
        return n_turns ** 2 * (1.0 - self.n_eff) / max(self.r_magnet, 1e-9)

    def h_in_magnet(self, mmf):
        """Field the coil drives into the magnet, per the loop equation."""
        return mmf * (1.0 - self.n_eff) / self.l_magnet

    def mmf_for_h(self, h_target):
        """Ampere-turns needed to drive ``h_target`` into the magnet."""
        return abs(h_target) * self.l_magnet / max(1.0 - self.n_eff, 1e-6)

    def operating_h(self, J):
        """Self-demagnetising field at polarisation ``J``, no coil."""
        return -self.n_eff * J / (MU0 * self.mu_rec)

    def summary(self):
        return (f"n_eff {self.n_eff:.3f} ({self.source}), R_mag "
                f"{self.r_magnet/1e6:.1f}M, R_ext {self.r_ext/1e6:.1f}M, "
                f"{(1-self.n_eff)*100:.0f} % of the coil mmf reaches the "
                f"magnet")


def n_eff_from_fem(J, H, mu_rec):
    """Effective demagnetising factor measured from a solved FEM state.

    ``J`` and ``H`` are the volume-averaged polarisation and axial field in the
    magnet, exactly what ``AxisymModel.region_state`` returns.
    """
    if J is None or abs(J) < 1e-9:
        return 0.5
    n = -MU0 * mu_rec * float(H) / float(J)
    return float(min(max(n, 1e-4), 0.999))


def _steel_reluctance(path_len, area, b_est):
    """Reluctance of an iron path at an estimated flux density.

    Nonlinear on purpose.  At 1.2 T the relative permeability of 1018 is about
    2000 and the iron is nearly free; at 1.9 T it is under 200 and the iron is
    most of the circuit.  A constant-mu model therefore promises a switching
    current that saturation will not deliver.
    """
    nu = float(steel_nu(np.array([abs(b_est)]))[0])
    return nu * path_len / max(area, 1e-12)


def estimate_n_eff(d_mag, l_mag, mu_rec, t_steel=0.0, r_clear=0.0,
                   gap=0.1e-3, has_steel=True, has_neighbour=False,
                   b_est=1.2):
    """Analytic n_eff for the pre-screen, before any FEM has been run.

    For a bare rod this reduces to ``n_eff = mu_rec * N_d`` with N_d the
    magnetometric demagnetising factor.  That factor of mu_rec is the reason a
    high recoil permeability is not free: Alnico 5 with mu_rec = 4 sits four
    times further down its own load line than a ferrite of identical shape.

    Accuracy against the FEM is measured in ``verify_circuit.py``.  It is good
    enough to reject hopeless designs and is never used for a quoted number -
    the pipeline overrides it with ``n_eff_from_fem`` as soon as Stage 1 has
    run.
    """
    r_m = d_mag / 2.0
    a_mag = np.pi * r_m ** 2
    r_magnet = l_mag / (MU0 * mu_rec * a_mag)

    if not (has_steel and t_steel > 0):
        nd = float(cylinder_demag_factor(r_m, l_mag))
        return float(min(max(mu_rec * nd, 1e-4), 0.999))

    r_out = r_m + r_clear + t_steel
    a_ann = np.pi * (r_out ** 2 - (r_m + r_clear) ** 2)
    a_back = np.pi * r_out ** 2
    r_steel = (_steel_reluctance(l_mag, a_ann, b_est) +
               _steel_reluctance(r_out, a_back, b_est))

    # The local return.  Flux leaving the pole face finds its way back to the
    # annulus rim through open air: the two are coplanar and separated by the
    # radial clearance, so this is Roters' quarter-cylinder in parallel with a
    # half-annulus, both taken over the pole perimeter.
    perim = 2.0 * np.pi * r_m
    g = max(r_clear, 0.15e-3)
    p_fringe = MU0 * perim * (0.26 + np.log(1.0 + 2.0 * l_mag / g) / np.pi)
    r_local = r_steel + 1.0 / max(p_fringe, 1e-15)

    if has_neighbour:
        # A mated neighbour offers a SECOND return path, in parallel with the
        # local one: across the working gap, through the neighbour's magnet
        # and steel, and back across the outer gap.  It cannot make the
        # circuit worse, but it helps less than it looks, because the
        # neighbour's own magnet is a long low-permeability leg - mu_rec = 4
        # is still only four times air.  Modelling it as a replacement for the
        # local return rather than a parallel path (which is what an earlier
        # version did) gets the sign of the effect wrong.
        r_neigh = (2.0 * gap / (MU0 * a_mag) + 2.0 * r_steel +
                   l_mag / (MU0 * mu_rec * a_mag))
        r_ext = r_local * r_neigh / (r_local + r_neigh)
    else:
        r_ext = r_local

    return float(min(max(r_ext / (r_magnet + r_ext), 1e-4), 0.999))


def circuit(d_mag, l_mag, mu_rec, t_steel=0.0, r_clear=0.0, gap=0.1e-3,
            has_steel=True, has_neighbour=False, b_est=1.2, n_eff=None,
            source="estimate"):
    """Build the lumped magnetic circuit seen by the coil.

    Pass ``n_eff`` to use a value measured from the FEM; leave it None to fall
    back on the analytic estimate.
    """
    r_m = d_mag / 2.0
    a_mag = np.pi * r_m ** 2
    if n_eff is None:
        n_eff = estimate_n_eff(d_mag, l_mag, mu_rec, t_steel, r_clear, gap,
                               has_steel, has_neighbour, b_est)
        source = "estimate"
    a_steel = (np.pi * ((r_m + r_clear + t_steel) ** 2 -
                        (r_m + r_clear) ** 2)
               if has_steel and t_steel > 0 else 0.0)
    return MagCircuit(n_eff=float(n_eff), a_magnet=a_mag, l_magnet=l_mag,
                      mu_rec=mu_rec,
                      has_steel=bool(has_steel and t_steel > 0),
                      has_neighbour=bool(has_neighbour), source=source,
                      a_steel=a_steel)


def steel_saturates(circ, mmf, br):
    """Does the iron saturate at this drive?

    Returns (flux density in the annulus, headroom factor).  Once the iron is
    past about 1.9 T it stops being a return path, the effective n_eff climbs
    and further ampere-turns buy far less field.  A linear model promises a
    switching current the steel will not pass.
    """
    if not circ.has_steel or circ.a_steel <= 0:
        return 0.0, np.inf
    phi = (mmf * (1.0 - circ.n_eff) / circ.l_magnet) * \
        MU0 * circ.mu_rec * circ.a_magnet + br * circ.a_magnet
    b = abs(phi) / circ.a_steel
    return float(b), float(1.95 / max(b, 1e-9))


if __name__ == "__main__":
    print("=" * 88)
    print("MULTI-LAYER WINDING AND THE MAGNETIC CIRCUIT IT DRIVES")
    print("=" * 88)

    print("\n  Winding a 4.2 mm magnet, 8.4 mm long, with 0.25 mm wire:\n")
    print(f"  {'layers':>7} {'turns':>7} {'build':>8} {'wire':>8} "
          f"{'R':>8} {'mass':>7} {'fill':>6} {'turns/ohm':>10}")
    print("  " + "-" * 70)
    for nl in (1, 2, 4, 6, 8, 12):
        w = wind(2.1e-3, 8.4e-3, 0.25e-3, nl)
        print(f"  {w.n_layers:7d} {w.n_turns:7d} {w.build*1e3:7.2f}m "
              f"{w.wire_length:7.2f}m {w.resistance:7.3f}o "
              f"{w.mass*1e3:6.2f}g {w.fill_factor:6.2f} "
              f"{w.n_turns/w.resistance:10.1f}")
    print("""
  turns/ohm falls as layers are added: each new layer sits at a larger radius
  so its turns cost more copper.  A model that treats the winding as a single
  lumped block misses this entirely and over-rewards deep coils.""")

    print("\n  Magnetic circuit, 4.2 x 8.4 mm magnet, 1 mm keeper:\n")
    print(f"  {'case':<34} {'mu_rec':>7} {'n_eff':>7} {'mmf reach':>10} "
          f"{'L (1000t)':>11} {'NI for 3 Hcj':>13}")
    print("  " + "-" * 88)
    for label, kw, mu, hcj in (
            ("bare rod, no steel", dict(has_steel=False), 4.0, 122e3),
            ("pot core, free space", dict(has_steel=True), 4.0, 122e3),
            ("pot core, latched neighbour",
             dict(has_steel=True, has_neighbour=True), 4.0, 122e3),
            ("pot core, ferrite (mu_rec 1.1)", dict(has_steel=True), 1.1,
             250e3),
            ("pot core, SmCo (mu_rec 1.05)", dict(has_steel=True), 1.05,
             1400e3)):
        c = circuit(4.2e-3, 8.4e-3, mu, t_steel=1.0e-3, r_clear=0.5e-3,
                    **kw)
        print(f"  {label:<34} {mu:7.2f} {c.n_eff:7.3f} "
              f"{(1-c.n_eff)*100:9.1f}% {c.inductance(1000)*1e3:10.2f}mH "
              f"{c.mmf_for_h(3*hcj):13.0f}")
    print("""
  The neighbour row is the finding, and it is smaller than it looks.  A mated
  module offers a second return path in parallel with the local fringe path,
  so it can only lower n_eff - but the neighbour's own magnet is a long
  low-permeability leg, so the improvement is modest.  What matters is that the
  old model ignored the neighbour entirely and therefore designed the driver
  for the wrong circuit.

  The mu_rec column is the other one.  For a bare rod n_eff is exactly
  mu_rec * N_d, so Alnico 5's recoil permeability of 4 puts it four times
  further down its own load line than a ferrite of identical shape.  The
  material changes the electromagnet's job as well as the threshold it has to
  clear - and the last two rows show the wall: SmCo needs 30x the ampere-turns
  of Alnico in the same geometry.

  Every one of these n_eff values is an ESTIMATE.  In the pipeline the number
  actually used is measured from the Stage 1 field solve via n_eff_from_fem,
  so the circuit the driver is designed against is the one the FEM computed.""")
