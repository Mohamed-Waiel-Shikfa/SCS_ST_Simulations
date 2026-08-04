"""Design-space definition and the evaluation stages for an EPM module.

This is the evaluation core the optimiser calls.  It is deliberately kept
separate from any search algorithm so the two can be validated independently.

Stage order, and why it is this order
-------------------------------------
Every stage is built on the output of the one before it, and nothing is
evaluated before the thing it depends on exists:

    Stage 0  module      the physical assembly: magnets, multi-layer coils,
                         steel, capacitor, battery, board, and what space is
                         left.  Everything downstream is measured against
                         this geometry.
    Stage 1  magnetics   attraction, repulsion, demagnetisation margin, and
                         the effective demagnetising factor n_eff of the real
                         magnetic circuit - which is what the driver needs.
    Stage 2  switching   the transient circuit, driven with the inductance and
                         field-per-ampere that Stage 1 measured, including the
                         steel, the magnet's own permeability and a latched
                         neighbour.  Pulse trains are searched here.
    Stage 3  mechanics   latching, holding, and the pivot energy balance.

Mechanics runs LAST and only if switching succeeded.  A design whose coil
cannot reverse its magnet is not a robot, whatever its holding force, so
simulating its gait is wasted time - and since mechanics is the expensive
stage, gating it is where most of the run time is saved.

Design notes
------------
* Material is a live variable over every commercially available class with
  Hcj below 2000 kA/m, not one family.  Coercivity buys repulsion and costs
  switching energy, and the point of the search is to find where that trade
  lands.
* The attract/repel asymmetry is carried as an explicit objective to minimise,
  because a design that latches strongly but cannot push is useless for
  locomotion.
* The demagnetisation margin is a hard constraint, not an objective.  A design
  that sits above ~0.8 of coercivity in its worst state erases itself in
  service, however good its forces look on paper.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from axisym_fem import AxisymModel, Region, axial_force  # noqa: E402
from coil import circuit as mag_circuit  # noqa: E402
from coil import estimate_n_eff, n_eff_from_fem, wind  # noqa: E402
from compat import trapezoid  # noqa: E402
from magnet_force import MU0, CoaxialRodPair  # noqa: E402
from materials import MATERIALS, material  # noqa: E402

G = 9.81
RHO_STEEL = 7870.0
RHO_SHELL = 1240.0      # PLA
RHO_CU = 1.68e-8


# --------------------------------------------------------------------------
@dataclass
class Design:
    """One candidate EPM module.

    Geometry is the intersection of three orthogonal regular ``n_gon`` prisms
    (see analysis/module.py).  ``r_face`` is the centre-to-pole-face distance,
    which with ``n_gon`` fixes every other dimension.
    """

    material: str = "LNGT72"
    d_mag: float = 4.75e-3       # magnet diameter
    l_mag: float = 12.5e-3       # magnet length
    circuit: str = "potcore"     # "none" or "potcore"
    t_steel: float = 1.0e-3      # keeper wall thickness
    r_clear: float = 1.0e-3      # radial clearance around the winding
    gap: float = 0.1e-3          # working air gap between mated faces
    n_gon: int = 8               # ring polygon: 8, 12, 16, 20
    r_face: float = 20e-3        # centre to pole face
    wire_d: float = 0.3e-3       # coil wire diameter
    n_layers: int = 4            # winding layers - a real variable now, not
                                 # borrowed from the keeper thickness
    v_cap: float = 70.0          # capacitor bank voltage
    c_cap: float = 10e-6         # capacitor bank capacitance
    pulse_mode: str = "single"   # "single" or "train"
    f_pulse: float = 20e3        # pulse-train frequency
    duty: float = 0.5            # pulse-train duty cycle
    n_pulses: int = 4

    @property
    def n_faces(self):
        return 3 * self.n_gon - 6

    @property
    def n_latch_faces(self):
        """Always six: the axis faces, which is what keeps the lattice cubic."""
        return 6

    @property
    def a_face(self):
        """Square face side length."""
        return 2.0 * self.r_face * np.tan(np.pi / self.n_gon)

    @property
    def winding(self):
        return wind(self.d_mag / 2.0, self.l_mag, self.wire_d, self.n_layers)

    @property
    def bounding_cube(self):
        """Smallest axis-aligned cube containing the module.

        The axis directions are ring normals, so the extent along each axis is
        exactly r_face.  r_face / cos(pi/n) is the polygon CIRCUMRADIUS, which
        governs the pivot lift, not the packaging envelope.
        """
        return 2.0 * self.r_face

    @property
    def r_vertex(self):
        """Centre to pivot edge - the radius the centre of mass swings on."""
        return self.r_face / np.cos(np.pi / self.n_gon)

    def as_row(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Stage 1: magnetics
# --------------------------------------------------------------------------
def _regions(dsg, flip):
    """Axisymmetric regions for one mated pair.

    The steel annulus starts OUTSIDE the winding, not at the magnet surface.
    The coil occupies real radial space - four to eight layers of 0.25 mm wire
    is one to two millimetres of it - and putting the iron where the copper
    actually is overstated the return path.
    """
    Rm, Lm, gap = dsg.d_mag / 2, dsg.l_mag, dsg.gap
    mat = material(dsg.material)
    d = -1 if flip else +1
    regs = [
        Region(0, Rm, -Lm, 0.0, "magnet", "A", material=mat, direction=+1),
        Region(0, Rm, gap, gap + Lm, "magnet", "B", material=mat, direction=d),
    ]
    if dsg.circuit == "potcore":
        ri = Rm + dsg.winding.build + dsg.r_clear
        ro = ri + dsg.t_steel
        regs += [
            Region(0, ro, -Lm - dsg.t_steel, -Lm, "steel", "backA"),
            Region(ri, ro, -Lm, 0.0, "steel", "annA"),
            Region(0, ro, gap + Lm, gap + Lm + dsg.t_steel, "steel", "backB"),
            Region(ri, ro, gap, gap + Lm, "steel", "annB"),
        ]
    return regs


def epm_outer_radius(dsg):
    """Outer radius of the whole EPM assembly on a face."""
    r = dsg.d_mag / 2 + dsg.winding.build
    if dsg.circuit == "potcore":
        r += dsg.r_clear + dsg.t_steel
    return r


def stage1_magnetics(dsg, mesh=None, n_slabs=None, fidelity="normal",
                     states=("attract", "repel")):
    """Attraction, repulsion and demagnetisation margin.

    Returns a dict.  ``margin`` is the worst |H|/Hcj over both states; above
    about 0.8 the magnet is being driven towards irreversible loss in service.

    ``fidelity`` trades accuracy for speed.  Screening runs are dominated by
    low-coercivity designs, where the magnet sits on the knee of its own curve
    and the nonlinear solve is stiff; a coarser mesh and fewer magnet slabs cut
    that cost by an order of magnitude.  Rankings are preserved because the
    discretisation error is systematic across designs, but any design that
    matters should be re-run at "normal" before it is believed.

    ``states`` restricts which operating points are solved.  The pivot work
    integral needs the repulsion curve at several gaps and nothing else, and
    solving the attracting state as well would double its cost for no use.
    """
    Rm = dsg.d_mag / 2
    ro = epm_outer_radius(dsg)
    hcj = MATERIALS[dsg.material]["Hcj"]
    mu_rec = MATERIALS[dsg.material]["mu_rec"]

    if fidelity == "screen":
        ns, rfar_k, zfar_k, nq = 3, 12, 10, 1500
        kw = dict(max_iter=10, continuation=False)
    else:
        ns, rfar_k, zfar_k, nq = 6, 25, 20, 4000
        kw = dict(max_iter=25, continuation=True)
    n_slabs = n_slabs or ns
    rfar = rfar_k * max(ro, Rm)

    if fidelity == "screen":
        # Tie the mesh to the MAGNET, not to the far-field box.  The earlier
        # rule (a fixed fraction of the outer boundary) collapsed for bare
        # rods: with no steel the far field is small, so the rule produced
        # h = 1.57 mm on a 4.75 mm rod - 1.5 elements across the diameter, and
        # a 48 % force error, while pot cores with their larger outer radius
        # got 11 %.  A structural bias like that does not cancel in a ranking:
        # it systematically favours whichever architecture happens to be
        # meshed better.  The mesh is graded, so resolving the magnet costs
        # far less than resolving the whole domain.
        h = mesh or max(min(dsg.d_mag, dsg.l_mag) / 6.0, 0.2e-3)
    else:
        h = mesh or max(min(dsg.d_mag, dsg.l_mag) / 16.0, 0.15e-3)

    out = {}
    for flip, tag in ((False, "attract"), (True, "repel")):
        if tag not in states:
            continue
        m = AxisymModel(_regions(dsg, flip), rfar, zfar_k * dsg.l_mag, h,
                        n_slabs=n_slabs)
        s = m.solve(**kw)
        J, H = m.region_state(s, "A")
        F = axial_force(s, dsg.gap / 2, r_max=0.9 * rfar, n=nq)
        out[f"J_{tag}"] = J
        out[f"margin_{tag}"] = abs(H) / hcj
        out[f"F_{tag}"] = abs(F)
        # The effective demagnetising factor of the circuit this magnet is
        # actually sitting in, read straight out of the solved state.  This is
        # what the switching stage needs and could not previously get: it
        # already contains the steel, the working gap, the neighbour and the
        # magnet's own recoil permeability, so the driver is designed against
        # the real magnetic circuit instead of an air-cored guess.
        out[f"n_eff_{tag}"] = n_eff_from_fem(J, H, mu_rec)

    if len(states) == 2:
        out["asymmetry"] = out["F_attract"] / max(out["F_repel"], 1e-9)
        out["margin"] = max(out["margin_attract"], out["margin_repel"])
        out["n_eff"] = out["n_eff_attract"]
    return out


# --------------------------------------------------------------------------
# Stage 2: mechanics
# --------------------------------------------------------------------------
def _pivot_geometry(n, r, theta):
    """Gap at the mating pole faces during a roll of ``theta``.

    Module B rests on the floor beside module A and tips forward over its
    leading bottom edge.  Returns the centre-to-centre distance between B's
    trailing pole face and A's mating pole face, which is the physical gap: it
    is zero at theta = 0 and grows as the module rotates away.
    """
    a = 2.0 * r * np.tan(np.pi / n)          # square face side
    E = np.array([a / 2, 0.0, 0.0])          # leading bottom edge, B frame
    C = np.array([0.0, 0.0, r])              # B centre before the roll

    def rot(v, t):
        c, s = np.cos(t), np.sin(t)
        return np.array([v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c])

    m_trail = C + np.array([-r, 0.0, 0.0])
    f_a = np.array([-r, 0.0, r])             # A's mating face centre
    th = np.atleast_1d(theta)
    return np.array([np.linalg.norm(E + rot(m_trail - E, t) - f_a)
                     for t in th])


def pivot_work(dsg, mag, n_theta=80, fidelity="screen", probe_gaps=(1e-3, 4e-3)):
    """Work the magnets can actually put into the roll.

    The static estimate this replaces multiplied peak force by arc length and
    assumed both driving faces held that force through the whole arc.  They
    cannot: the trailing pair separates as soon as the module tips.  MuJoCo
    runs showed the old estimate was optimistic by roughly fifty times.

    A first correction integrated force against separation using the ANALYTIC
    charge-disc fall-off, and that was wrong the other way, by about five
    times.  The reason is physical: as a repelling pair separates the two
    magnets stop demagnetising each other and their polarisation RECOVERS, so
    the repulsion decays much more slowly than a fixed-strength charge model
    predicts.  The fall-off therefore has to come from the FEM as well as the
    magnitude, which costs two extra Stage 1 solves.

    Beyond the last probe the pair is treated as two dipoles, force ~ 1/r^4.

    Only the trailing repulsion is counted.  A leading face pair could add an
    attractive assist, but where it sits during the roll depends on lattice
    conventions this model does not fix, so counting it would flatter the
    design.  The number below is a lower bound on the available drive.
    """
    gaps = [dsg.gap] + [g for g in probe_gaps if g > dsg.gap]
    forces = [mag["F_repel"]]
    for g in gaps[1:]:
        m = stage1_magnetics(Design(**{**dsg.as_row(), "gap": g}),
                             fidelity=fidelity, states=("repel",))
        forces.append(m["F_repel"])
    gaps = np.array(gaps)
    forces = np.array(forces)

    def F(s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        out = np.interp(s, gaps, forces)
        tail = s > gaps[-1]
        if np.any(tail):
            c_end = gaps[-1] + dsg.l_mag
            out = np.where(tail, forces[-1] * (c_end / (s + dsg.l_mag)) ** 4,
                           out)
        return out

    th = np.linspace(0.0, 2 * np.pi / dsg.n_gon, n_theta)
    s = _pivot_geometry(dsg.n_gon, dsg.r_face, th) + dsg.gap
    W = float(trapezoid(F(s), s))
    return dict(W_drive=max(W, 0.0), W_trail=W)


def stage3_mechanics(dsg, mag, mod=None, driver=None, fidelity="screen"):
    """Static feasibility of latching, hanging and pivoting.

    Runs LAST, and only when switching has already succeeded.  A module whose
    coil cannot reverse its magnet is not a robot however well it holds, so
    there is nothing to learn from its gait - and this is the expensive stage,
    so skipping it is where most of the search time is saved.

    Pivot model.  A module rolls onto its neighbour by rotating about the
    shared edge through the exterior angle 360/n.  Two quantities decide
    whether that is possible:

    * the gravity barrier - the centre of mass rises from r_face to
      R_vertex = r_face / cos(pi/n) at the midpoint of the roll, so the energy
      barrier is m g (R_vertex - r_face).  This is why the polygon matters: for
      a cube the rise is 41 % of the half-width, for a 16-gon only 2 %.
    * the magnetic drive - see ``pivot_work``, which integrates force against
      the separation each pair actually traverses.  An earlier version of this
      function multiplied peak force by arc length and overstated the drive by
      about fifty times.

    Also checks the module can hold its own weight hanging from one face.
    """
    from module import build_module

    if mod is None:
        mod = build_module(dsg, driver)

    m = mod.mass
    w = m * G
    n = dsg.n_gon
    r = dsg.r_face
    R_vertex = r / np.cos(np.pi / n)

    dE = m * G * (R_vertex - r)

    pw = pivot_work(dsg, mag, fidelity=fidelity)
    W_drive = pw["W_drive"]

    return dict(m_module=m, weight=w,
                r_vertex=R_vertex, lift=R_vertex - r,
                E_barrier=dE, W_drive=W_drive,
                W_trail=pw["W_trail"],
                pivot_ratio=W_drive / max(dE, 1e-12),
                hold_ratio=mag["F_attract"] / w,
                tau_drive=W_drive / max(2 * np.pi / n, 1e-9),
                fits=mod.fits, module=mod)


# older scripts import the previous name
def stage2_mechanics(dsg, mag, **kw):
    return stage3_mechanics(dsg, mag, **kw)


# --------------------------------------------------------------------------
# Stage 2: switching
# --------------------------------------------------------------------------
def stage2_switching(dsg, k_switch=3.0, v_max=400.0, n_eff=None,
                     search_pulse=False, r_series=0.05, n_steps=2500):
    """Can the coil reverse the magnet, and what does it cost?

    Everything here is built on Stage 1.  ``n_eff`` is the effective
    demagnetising factor MEASURED from the field solve, so the inductance and
    the field driven per ampere are those of the real magnetic circuit -
    including the steel return path, the magnet's own recoil permeability and
    a latched neighbour.  Falling back to the analytic estimate is only for the
    pre-screen, where no field solve has been paid for yet.

    What changed from the previous version, and why it matters:

    * The winding is a real multi-layer coil, so turns and resistance come
      from ``coil.wind`` rather than from a smooth product that could not tell
      one layer from eight.
    * The drive is integrated in time rather than reduced to an underdamped LC
      peak, so a PULSE TRAIN can be evaluated.  A train at the right frequency
      and duty reaches the same field for substantially less energy out of the
      bank, because the bank is not dumped into resistance on a single swing.
    * The threshold is a field in the magnet, not ampere-turns in free space.

    Two energies are reported and must not be confused.  ``e_bank`` is whatever
    the specified capacitor happens to store.  ``e_required`` is the bank
    energy actually needed to reach the switching threshold for THIS material
    and geometry, obtained by scaling the drive voltage until the field just
    suffices.  That is the quantity that belongs in an objective.
    """
    from circuit_sim import PulseProgram, best_program, simulate

    w = dsg.winding
    hcj = MATERIALS[dsg.material]["Hcj"]
    mu_rec = MATERIALS[dsg.material]["mu_rec"]
    h_need = k_switch * hcj

    if n_eff is None:
        n_eff = estimate_n_eff(dsg.d_mag, dsg.l_mag, mu_rec,
                               t_steel=dsg.t_steel,
                               r_clear=dsg.winding.build + dsg.r_clear,
                               gap=dsg.gap,
                               has_steel=dsg.circuit == "potcore",
                               has_neighbour=True)
    circ = mag_circuit(dsg.d_mag, dsg.l_mag, mu_rec, t_steel=dsg.t_steel,
                       r_clear=dsg.winding.build + dsg.r_clear, gap=dsg.gap,
                       has_steel=dsg.circuit == "potcore",
                       has_neighbour=True, n_eff=n_eff, source="fem")

    prog = PulseProgram(dsg.pulse_mode, f_pulse=dsg.f_pulse, duty=dsg.duty,
                        n_pulses=dsg.n_pulses)
    tr = simulate(circ, w.n_turns, w.resistance, dsg.c_cap, dsg.v_cap,
                  r_series=r_series, h_need=h_need, program=prog,
                  n_steps=n_steps)

    # voltage that would just reach the threshold.  Peak field is linear in
    # the drive voltage in both damping regimes while the iron is unsaturated,
    # so this scales directly; once the iron saturates it is optimistic, which
    # is why the saturation flag is carried through to the constraint set.
    v_need = dsg.v_cap * h_need / max(tr.h_peak, 1e-9)
    e_required = 0.5 * dsg.c_cap * v_need ** 2

    best = tr
    if search_pulse and tr.switched:
        _, best = best_program(circ, w.n_turns, w.resistance, dsg.c_cap,
                               dsg.v_cap, h_need, r_series=r_series)

    v_mag = np.pi * (dsg.d_mag / 2) ** 2 * dsg.l_mag
    e_hyst = 4.0 * MATERIALS[dsg.material]["Br"] * hcj * v_mag

    return dict(n_turns=w.n_turns, n_layers=w.n_layers,
                turns_per_layer=w.turns_per_layer,
                winding_build=w.build, wire_length=w.wire_length,
                coil_mass=w.mass, fill_factor=w.fill_factor,
                R_coil=w.resistance, L_coil=circ.inductance(w.n_turns),
                n_eff=circ.n_eff, n_eff_source=circ.source,
                i_peak=tr.i_peak, h_peak=tr.h_peak, h_need=h_need,
                mmf=tr.mmf_peak, mmf_need=circ.mmf_for_h(h_need),
                switch_ok=bool(tr.switched and v_need <= v_max),
                switched=bool(tr.switched),
                saturated=bool(tr.saturated),
                b_steel_peak=tr.b_steel_peak,
                v_need=v_need, switch_margin=tr.h_peak / max(h_need, 1e-9),
                e_hysteresis=e_hyst,
                e_bank=0.5 * dsg.c_cap * dsg.v_cap ** 2,
                e_drawn=tr.e_drawn, e_resistive=tr.e_resistive,
                e_required=e_required,
                e_total_module=e_required * dsg.n_faces,
                t_peak=tr.t_peak,
                pulse_program=best.meta.get("program", "single shot"),
                e_drawn_best=best.e_drawn,
                pulse_saving=(1.0 - best.e_drawn / max(tr.e_drawn, 1e-12))
                if tr.e_drawn > 0 else 0.0,
                transient=tr)


# backwards-compatible alias: several older scripts import this name
def stage3_switching(dsg, **kw):
    return stage2_switching(dsg, **kw)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
MARGIN_LIMIT = 0.80          # H/Hcj above this erases the magnet in service
HOLD_MIN = 3.0               # attraction must exceed 3x module weight
PIVOT_MIN = 1.5              # magnetic work must exceed the gravity barrier
CUBE_MAX = 50e-3             # the module must fit inside a 5 cm cube


def score(dsg, mag=None, mech=None, sw=None, drv=None):
    """Objectives and constraints for one design.

    Objectives are returned separately rather than collapsed into one number,
    so a multi-objective search can use them directly.  A scalar fallback is
    provided for single-objective methods.
    """
    mag = mag if mag is not None else stage1_magnetics(dsg)
    sw = sw if sw is not None else stage2_switching(dsg,
                                                    n_eff=mag.get("n_eff"))
    if drv is None:
        from driver import select_driver
        drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                            sw["n_turns"], sw["mmf_need"],
                            n_faces=dsg.n_faces)
    mech = mech if mech is not None else stage3_mechanics(
        dsg, mag, driver=drv if drv.feasible else None)

    violations = []
    if mag["margin"] > MARGIN_LIMIT:
        violations.append(f"demag margin {mag['margin']:.2f} > {MARGIN_LIMIT}")
    if not sw["switched"]:
        violations.append(
            f"coil reaches only {sw['h_peak']/1e3:.0f} kA/m of the "
            f"{sw['h_need']/1e3:.0f} kA/m needed to switch")
    if not drv.feasible:
        violations.append(f"no driver for {sw['v_need']:.0f} V")
    if mech["hold_ratio"] < HOLD_MIN:
        violations.append(f"hold {mech['hold_ratio']:.1f} < {HOLD_MIN}")
    if mech["pivot_ratio"] < PIVOT_MIN:
        violations.append(f"pivot {mech['pivot_ratio']:.2f} < {PIVOT_MIN}")
    if dsg.bounding_cube > CUBE_MAX:
        violations.append(f"cube {dsg.bounding_cube*1e3:.0f} mm > 50 mm")
    if not mech["fits"]:
        violations.append("electronics do not fit")
    if dsg.l_mag + (dsg.t_steel if dsg.circuit == "potcore" else 0) > \
            0.85 * dsg.r_face:
        violations.append("EPM deeper than the module radius")

    objectives = dict(
        f_attract=mag["F_attract"],            # maximise
        f_repel=mag["F_repel"],                # maximise
        asymmetry=mag["asymmetry"],            # minimise
        e_switch=sw["e_required"] * dsg.n_faces,   # minimise
        mass=mech["m_module"],                 # minimise
    )

    feasible = not violations
    scalar = 0.0 if not feasible else (
        (objectives["f_repel"] ** 0.5 * objectives["f_attract"] ** 0.5) /
        (objectives["asymmetry"] ** 0.25 *
         (objectives["e_switch"] * 1e3) ** 0.25 *
         (objectives["mass"] * 1e3) ** 0.25))

    return dict(objectives=objectives, violations=violations,
                feasible=feasible, scalar=scalar,
                magnetics=mag, mechanics=mech, switching=sw, driver=drv)


# --------------------------------------------------------------------------
# Cheap pre-screen
# --------------------------------------------------------------------------
def prescreen(dsg, sw=None, drv=None):
    """Reject hopeless designs in milliseconds, before paying for the FEM.

    Several constraints need no field solve:

    * pure geometry - the EPM must fit inside the module radius, and the module
      must fit inside the 5 cm cube;
    * driver feasibility - a circuit calculation;
    * the open-circuit demagnetisation margin of an isolated rod, available in
      closed form from the validated free-space solver.  A rod that already
      exceeds the limit with no neighbour present can only be worse once a
      neighbour reverses against it.

    This matters for cost as well as time: the low-coercivity designs are both
    the physically hopeless ones AND the numerically stiff ones, so screening
    them analytically removes most of the expensive FEM work.
    """
    reasons = []

    depth = dsg.l_mag + (dsg.t_steel if dsg.circuit == "potcore" else 0.0)
    if depth > 0.85 * dsg.r_face:
        reasons.append("EPM deeper than the module radius")
    r_out = epm_outer_radius(dsg)
    if 2 * r_out > 0.95 * dsg.a_face:
        reasons.append("EPM wider than the face")
    if dsg.bounding_cube > CUBE_MAX:
        reasons.append(f"cube {dsg.bounding_cube*1e3:.0f} mm > 50 mm")

    if sw is None:
        sw = stage2_switching(dsg, search_pulse=False)
    if not sw["switched"]:
        reasons.append(
            f"coil reaches only {sw['h_peak']/1e3:.0f} kA/m of the "
            f"{sw['h_need']/1e3:.0f} kA/m needed to switch")
    if drv is None:
        from driver import select_driver
        drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                            sw["n_turns"], sw["mmf_need"],
                            n_faces=dsg.n_faces)
    if not drv.feasible:
        reasons.append(f"no driver for {sw['v_need']:.0f} V")

    if reasons:
        return False, reasons

    mat = material(dsg.material)
    pair = CoaxialRodPair(dsg.d_mag / 2, dsg.l_mag, mat, n_slabs=12)
    try:
        _, H = pair.solve(1e3 * dsg.l_mag)
        margin_open = (float(np.mean(np.abs(H[:12]))) /
                       MATERIALS[dsg.material]["Hcj"])
    except RuntimeError:
        return False, ["free-space solve failed"]

    # a closed circuit can only improve on the open-circuit margin, so this
    # bound is valid only for the uncircuited case
    if dsg.circuit == "none" and margin_open > MARGIN_LIMIT:
        reasons.append(f"open-circuit demag margin {margin_open:.2f} "
                       f"> {MARGIN_LIMIT}")

    return (not reasons), reasons


ROW_FIELDS = (
    # design
    "material", "d_mag", "l_mag", "circuit", "t_steel", "r_clear", "gap",
    "n_gon", "r_face", "wire_d", "n_layers", "v_cap", "c_cap",
    "pulse_mode", "f_pulse", "duty", "n_pulses", "fidelity",
    # stage 0: the module
    "n_faces", "a_face", "bounding_cube", "m_module", "free_volume",
    "coil_mass", "turns_per_layer", "n_turns", "winding_build",
    # stage 1: magnetics
    "J_attract", "J_repel", "margin_attract", "margin_repel",
    "F_attract", "F_repel", "asymmetry", "margin", "n_eff",
    # stage 2: switching
    "R_coil", "L_coil", "i_peak", "h_peak", "h_need", "switch_margin",
    "switched", "saturated", "b_steel_peak", "mmf", "mmf_need", "v_need",
    "e_switch", "e_drawn", "pulse_program", "pulse_saving",
    # stage 3: mechanics
    "hold_ratio", "pivot_ratio", "E_barrier", "W_drive",
    # driver
    "drv_mass", "drv_price", "drv_cap", "drv_mosfet", "drv_topology",
    # scoring
    "feasible", "scalar", "violations",
)


def _row(dsg, fidelity, mag, mech, sw, drv, feasible, scalar, violations,
         mod=None):
    """Build a result row in a fixed column order.

    The order must not depend on which code path produced the row: a
    pre-screened design and a fully evaluated one have to line up in the CSV.
    """
    src = dict(dsg.as_row())
    src.update(fidelity=fidelity, n_faces=dsg.n_faces, a_face=dsg.a_face,
               bounding_cube=dsg.bounding_cube)
    if mod is not None:
        src.update(free_volume=mod.free_volume)
    src.update(mag)
    src.update(m_module=mech.get("m_module"),
               hold_ratio=mech.get("hold_ratio"),
               pivot_ratio=mech.get("pivot_ratio"),
               E_barrier=mech.get("E_barrier"), W_drive=mech.get("W_drive"))
    for k in ("R_coil", "L_coil", "i_peak", "h_peak", "h_need",
              "switch_margin", "switched", "saturated", "b_steel_peak",
              "mmf", "mmf_need", "v_need", "e_drawn", "pulse_program",
              "pulse_saving", "coil_mass", "turns_per_layer", "n_turns",
              "winding_build"):
        src[k] = sw.get(k)
    src["e_switch"] = sw["e_required"] * dsg.n_faces
    src.update(drv_mass=(drv.mass if drv and drv.feasible else None),
               drv_price=(drv.price if drv and drv.feasible else None),
               drv_cap=(drv.cap_name if drv and drv.feasible else None),
               drv_mosfet=(drv.mosfet_name if drv and drv.feasible else None),
               drv_topology=(drv.topology if drv and drv.feasible else None))
    src.update(feasible=feasible, scalar=scalar, violations=violations)
    return {k: src.get(k) for k in ROW_FIELDS}


_BLANK_MAG = dict(J_attract=np.nan, J_repel=np.nan, margin_attract=np.nan,
                  margin_repel=np.nan, F_attract=0.0, F_repel=0.0,
                  asymmetry=np.inf, margin=np.nan, n_eff=np.nan)


def evaluate(dsg, fidelity="normal", use_prescreen=True):
    """Run every stage in order and score.  Returns a flat dict.

    The order is the point.  Stage 0 builds the module, Stage 1 solves its
    field and measures the magnetic circuit, Stage 2 drives that circuit, and
    Stage 3 only runs if Stage 2 actually switched.  Roughly two thirds of a
    random population fails before mechanics, and mechanics is four fifths of
    the cost, so the gate is most of the run time.
    """
    from driver import select_driver
    from module import build_module

    # ---- Stage 0: the module.  Built first because everything measures
    # itself against this geometry - including the winding, whose radial build
    # sets where the steel starts and therefore what Stage 1 solves.
    mod0 = build_module(dsg)

    # cheap switching pass for the pre-screen, on the estimated circuit
    sw = stage2_switching(dsg, search_pulse=False)
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=dsg.n_faces)

    if use_prescreen:
        ok, why = prescreen(dsg, sw, drv)
        if not ok:
            mech = dict(m_module=mod0.mass, hold_ratio=0.0, pivot_ratio=0.0,
                        E_barrier=np.nan, W_drive=0.0, fits=mod0.fits)
            return _row(dsg, fidelity, dict(_BLANK_MAG), mech, sw, drv,
                        False, 0.0, "; ".join(why) + " [prescreen]", mod0)

    # ---- Stage 1: magnetics
    try:
        mag = stage1_magnetics(dsg, fidelity=fidelity)
    except RuntimeError as exc:
        # A stalled nonlinear solve is a property of the DESIGN, not a bug:
        # it happens when the magnet sits exactly on the knee of its own
        # demagnetisation curve, which is precisely the operating point the
        # margin constraint exists to forbid.  Record it as infeasible rather
        # than letting it abort the sweep.
        blank = dict(_BLANK_MAG, margin=1.0)
        mech = dict(m_module=mod0.mass, hold_ratio=0.0, pivot_ratio=0.0,
                    E_barrier=np.nan, W_drive=0.0, fits=mod0.fits)
        return _row(dsg, fidelity, blank, mech, sw, drv, False, 0.0,
                    f"magnet solve stalled on the knee ({exc})", mod0)

    # ---- Stage 2: switching, now on the circuit the FEM measured
    sw = stage2_switching(dsg, n_eff=mag.get("n_eff"))
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=dsg.n_faces)

    if not sw["switched"] or not drv.feasible:
        why = []
        if not sw["switched"]:
            why.append(f"coil reaches only {sw['h_peak']/1e3:.0f} kA/m of "
                       f"the {sw['h_need']/1e3:.0f} kA/m needed to switch")
        if not drv.feasible:
            why.append(f"no driver for {sw['v_need']:.0f} V")
        mech = dict(m_module=mod0.mass, hold_ratio=0.0, pivot_ratio=0.0,
                    E_barrier=np.nan, W_drive=0.0, fits=mod0.fits)
        return _row(dsg, fidelity, mag, mech, sw, drv, False, 0.0,
                    "; ".join(why) + " [no mechanics: switching failed]",
                    mod0)

    # ---- Stage 3: mechanics, only now that the thing can actually switch
    mod = build_module(dsg, drv)
    mech = stage3_mechanics(dsg, mag, mod=mod)
    sc = score(dsg, mag, mech, sw, drv)
    return _row(dsg, fidelity, mag, mech, sw, drv, sc["feasible"],
                sc["scalar"], "; ".join(sc["violations"]), mod)


