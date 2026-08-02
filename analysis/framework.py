"""Design-space definition and the three evaluation stages for an EPM module.

This is the evaluation core the optimiser will call.  It is deliberately kept
separate from any search algorithm so the two can be validated independently.

    Stage 1  magnetics   attraction, repulsion, demagnetisation margin
    Stage 2  mechanics   can a module actually latch, hold and pivot
    Stage 3  switching   can the coil reverse the magnet, and at what energy

Every stage returns physical quantities in SI units.  Scoring and constraint
handling live in ``score()`` at the bottom so the objectives can be changed
without touching the physics.

Design notes
------------
* Material is a live variable, not a fixed choice.  The material sweep showed a
  genuine interior optimum: coercivity buys repulsion and costs switching
  energy.
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
from magnet_force import MU0, CoaxialRodPair, Material  # noqa: E402

G = 9.81
RHO_ALNICO = 7300.0
RHO_STEEL = 7870.0
RHO_SHELL = 1240.0      # PLA
RHO_CU = 1.68e-8

# --------------------------------------------------------------------------
# Materials.  Vendor rows are from simulations/Force_compute/Alnico性能表.png
# and correspond to orderable parts.  "lit" rows are typical published values
# for the material class and would need a real datasheet before being designed
# in.  mu_rec: the Alnico 5 family recoils strongly (~4); the more anisotropic
# Alnico 8/9 are straighter (~2); ferrite is nearly linear (~1.1).
# --------------------------------------------------------------------------
MATERIALS = {
    #  name              Br     Hcb    Hcj   BHmax  mu_rec  source
    "LNG13":    dict(Br=0.68, Hcb=48e3, Hcj=51e3, BHmax=13e3, mu_rec=4.0, src="vendor"),
    "LNG37":    dict(Br=1.20, Hcb=48e3, Hcj=49e3, BHmax=37e3, mu_rec=4.0, src="vendor"),
    "LNG40":    dict(Br=1.25, Hcb=48e3, Hcj=49e3, BHmax=40e3, mu_rec=4.0, src="vendor"),
    "LNG52":    dict(Br=1.30, Hcb=56e3, Hcj=57e3, BHmax=52e3, mu_rec=4.0, src="vendor"),
    "LNG60":    dict(Br=1.35, Hcb=59e3, Hcj=60e3, BHmax=60e3, mu_rec=4.0, src="vendor"),
    "LNGT28":   dict(Br=1.00, Hcb=58e3, Hcj=59e3, BHmax=28e3, mu_rec=3.5, src="vendor"),
    "LNGT18":   dict(Br=0.58, Hcb=90e3, Hcj=92e3, BHmax=18e3, mu_rec=2.5, src="vendor"),
    "LNGT38":   dict(Br=0.80, Hcb=110e3, Hcj=112e3, BHmax=38e3, mu_rec=2.0, src="vendor"),
    "LNGT44":   dict(Br=0.88, Hcb=120e3, Hcj=122e3, BHmax=44e3, mu_rec=2.0, src="vendor"),
    "LNGT36J":  dict(Br=0.70, Hcb=140e3, Hcj=148e3, BHmax=36e3, mu_rec=2.0, src="vendor"),
    "LNGT60":   dict(Br=0.90, Hcb=110e3, Hcj=112e3, BHmax=60e3, mu_rec=2.0, src="vendor"),
    "LNGT72":   dict(Br=1.05, Hcb=112e3, Hcj=114e3, BHmax=72e3, mu_rec=2.0, src="vendor"),
    "FeCrCo28": dict(Br=1.05, Hcb=44e3, Hcj=46e3, BHmax=28e3, mu_rec=4.0, src="lit"),
    "FeCrCo42": dict(Br=1.20, Hcb=59e3, Hcj=62e3, BHmax=42e3, mu_rec=3.5, src="lit"),
    "Ferrite30": dict(Br=0.38, Hcb=175e3, Hcj=195e3, BHmax=27e3, mu_rec=1.1, src="lit"),
}

_MAT_CACHE = {}


def material(name):
    """Fitted Material object for a catalogue entry (cached: the fit is slow)."""
    if name not in _MAT_CACHE:
        d = MATERIALS[name]
        _MAT_CACHE[name] = Material.from_datasheet(
            name, Br=d["Br"], Hcb=d["Hcb"], Hcj=d["Hcj"], BHmax=d["BHmax"],
            mu_rec=d["mu_rec"])
    return _MAT_CACHE[name]


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
    r_clear: float = 1.0e-3      # radial clearance rod -> return annulus
    gap: float = 0.1e-3          # working air gap between mated faces
    n_gon: int = 8               # ring polygon: 8, 12, 16, 20
    r_face: float = 20e-3        # centre to pole face
    wire_d: float = 0.3e-3       # coil wire diameter
    v_cap: float = 70.0          # capacitor bank voltage
    c_cap: float = 10e-6         # capacitor bank capacitance

    @property
    def n_faces(self):
        return 3 * self.n_gon - 6

    @property
    def a_face(self):
        """Square face side length."""
        return 2.0 * self.r_face * np.tan(np.pi / self.n_gon)

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
    Rm, Lm, gap = dsg.d_mag / 2, dsg.l_mag, dsg.gap
    mat = material(dsg.material)
    d = -1 if flip else +1
    regs = [
        Region(0, Rm, -Lm, 0.0, "magnet", "A", material=mat, direction=+1),
        Region(0, Rm, gap, gap + Lm, "magnet", "B", material=mat, direction=d),
    ]
    if dsg.circuit == "potcore":
        ro = Rm + dsg.r_clear + dsg.t_steel
        regs += [
            Region(0, ro, -Lm - dsg.t_steel, -Lm, "steel", "backA"),
            Region(Rm + dsg.r_clear, ro, -Lm, 0.0, "steel", "annA"),
            Region(0, ro, gap + Lm, gap + Lm + dsg.t_steel, "steel", "backB"),
            Region(Rm + dsg.r_clear, ro, gap, gap + Lm, "steel", "annB"),
        ]
    return regs


def stage1_magnetics(dsg, mesh=None, n_slabs=None, fidelity="normal"):
    """Attraction, repulsion and demagnetisation margin.

    Returns a dict.  ``margin`` is the worst |H|/Hcj over both states; above
    about 0.8 the magnet is being driven towards irreversible loss in service.

    ``fidelity`` trades accuracy for speed.  Screening runs are dominated by
    low-coercivity designs, where the magnet sits on the knee of its own curve
    and the nonlinear solve is stiff; a coarser mesh and fewer magnet slabs cut
    that cost by an order of magnitude.  Rankings are preserved because the
    discretisation error is systematic across designs, but any design that
    matters should be re-run at "normal" before it is believed.
    """
    Rm = dsg.d_mag / 2
    ro = Rm + (dsg.r_clear + dsg.t_steel if dsg.circuit == "potcore" else 0.0)
    hcj = MATERIALS[dsg.material]["Hcj"]

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
        m = AxisymModel(_regions(dsg, flip), rfar, zfar_k * dsg.l_mag, h,
                        n_slabs=n_slabs)
        if fidelity == "screen":
            # No continuation fallback while screening.  The designs that stall
            # are the low-coercivity ones sitting on the knee, which are also
            # the ones that fail the demagnetisation constraint anyway, so
            # paying for an expensive recovery to confirm a rejection is waste.
            s = m.solve(**kw)
        else:
            # "normal" already escalates internally (Newton -> continuation ->
            # damped fixed point).  A second full retry on top of that just
            # doubles the cost of a design that is going to fail anyway.
            s = m.solve(**kw)
        J, H = m.region_state(s, "A")
        F = axial_force(s, dsg.gap / 2, r_max=0.9 * rfar, n=nq)
        out[f"J_{tag}"] = J
        out[f"margin_{tag}"] = abs(H) / hcj
        out[f"F_{tag}"] = F

    out["F_attract"] = abs(out["F_attract"])
    out["F_repel"] = abs(out["F_repel"])
    out["asymmetry"] = out["F_attract"] / max(out["F_repel"], 1e-9)
    out["margin"] = max(out["margin_attract"], out["margin_repel"])
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
                             fidelity=fidelity)
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
    W = float(np.trapz(F(s), s))
    return dict(W_drive=max(W, 0.0), W_trail=W)


def stage2_mechanics(dsg, mag, mod=None, driver=None, fidelity="screen"):
    """Static feasibility of latching, hanging and pivoting.

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

    Also checks the module can hold its own weight hanging from one face, and
    that a horizontal chain of modules does not tear at the root.
    """
    from module import build_module

    if mod is None:
        mod = build_module(dsg, driver)

    m = mod.mass
    w = m * G
    n = dsg.n_gon
    r = dsg.r_face
    R_vertex = r / np.cos(np.pi / n)

    # energy barrier to roll over the edge
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


# --------------------------------------------------------------------------
# Stage 3: switching
# --------------------------------------------------------------------------
def stage3_switching(dsg, k_switch=3.0, winding_build=None, v_max=200.0):
    """Can the coil reverse the magnet, and what does it cost?

    Peak current is taken from the underdamped LC limit, which the feasibility
    audit showed is the operative regime for a practical coil: the 0.1 mm
    winding failed because it was overdamped, not because it lacked turns.

    Two different energies are reported and they must not be confused.
    ``e_bank`` is whatever the specified capacitor happens to store, which says
    nothing about the design - it is the same for every material at a given
    driver.  ``e_required`` is the bank energy actually needed to reach the
    switching threshold for THIS material and geometry, obtained by scaling the
    drive voltage until the ampere-turns just suffice.  That is the quantity
    that belongs in an objective, because it is what a high-coercivity grade
    really costs.
    """
    d = dsg.wire_d
    build = winding_build or dsg.t_steel
    n_turns = (dsg.l_mag / d) * (build / d)
    mean_d = dsg.d_mag + build
    wire_len = n_turns * np.pi * mean_d
    area = np.pi * (d / 2) ** 2
    R = RHO_CU * wire_len / area
    L = MU0 * n_turns**2 * (np.pi * (mean_d / 2) ** 2) / (dsg.l_mag +
                                                          0.45 * mean_d)

    z0 = np.sqrt(L / dsg.c_cap)
    underdamped = R < 2 * z0
    # resistance always bounds the peak current, even for a very low
    # inductance coil where the LC impedance alone would predict a huge one
    i_peak = min(dsg.v_cap / z0 if z0 > 0 else np.inf,
                 dsg.v_cap / max(R, 1e-6))
    mmf = n_turns * i_peak

    hcj = MATERIALS[dsg.material]["Hcj"]
    mmf_need = k_switch * hcj * dsg.l_mag

    # voltage that would just reach the threshold (peak current is linear in V
    # in both damping regimes, so this scales directly)
    v_need = dsg.v_cap * mmf_need / max(mmf, 1e-12)
    e_required = 0.5 * dsg.c_cap * v_need**2

    v_mag = np.pi * (dsg.d_mag / 2) ** 2 * dsg.l_mag
    e_hyst = 4.0 * MATERIALS[dsg.material]["Br"] * hcj * v_mag

    return dict(n_turns=n_turns, R_coil=R, L_coil=L, i_peak=i_peak,
                underdamped=underdamped,
                mmf=mmf, mmf_need=mmf_need,
                switch_ok=v_need <= v_max,
                v_need=v_need, switch_margin=mmf / mmf_need,
                e_hysteresis=e_hyst,
                e_bank=0.5 * dsg.c_cap * dsg.v_cap**2,
                e_required=e_required,
                e_total_module=e_required * dsg.n_faces)


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
    sw = sw if sw is not None else stage3_switching(dsg)
    if drv is None:
        from driver import select_driver
        drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                            sw["n_turns"], sw["mmf_need"],
                            n_faces=dsg.n_faces)
    mech = mech if mech is not None else stage2_mechanics(
        dsg, mag, driver=drv if drv.feasible else None)

    violations = []
    if mag["margin"] > MARGIN_LIMIT:
        violations.append(f"demag margin {mag['margin']:.2f} > {MARGIN_LIMIT}")
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
    r_out = dsg.d_mag / 2 + (dsg.r_clear + dsg.t_steel
                             if dsg.circuit == "potcore" else 0.0)
    if 2 * r_out > 0.95 * dsg.a_face:
        reasons.append("EPM wider than the face")
    if dsg.bounding_cube > CUBE_MAX:
        reasons.append(f"cube {dsg.bounding_cube*1e3:.0f} mm > 50 mm")

    if sw is None:
        sw = stage3_switching(dsg)
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
    "n_gon", "r_face", "wire_d", "v_cap", "c_cap", "fidelity",
    # derived geometry
    "n_faces", "a_face", "bounding_cube",
    # stage 1
    "J_attract", "J_repel", "margin_attract", "margin_repel",
    "F_attract", "F_repel", "asymmetry", "margin",
    # stage 2
    "m_module", "hold_ratio", "pivot_ratio", "E_barrier", "W_drive",
    # stage 3
    "mmf", "mmf_need", "v_need", "e_switch",
    # driver
    "drv_mass", "drv_price", "drv_cap", "drv_mosfet",
    # scoring
    "feasible", "scalar", "violations",
)


def _row(dsg, fidelity, mag, mech, sw, drv, feasible, scalar, violations):
    """Build a result row in a fixed column order.

    The order must not depend on which code path produced the row: a
    pre-screened design and a fully evaluated one have to line up in the CSV.
    """
    src = dict(dsg.as_row())
    src.update(fidelity=fidelity, n_faces=dsg.n_faces, a_face=dsg.a_face,
               bounding_cube=dsg.bounding_cube)
    src.update(mag)
    src.update(m_module=mech.get("m_module"),
               hold_ratio=mech.get("hold_ratio"),
               pivot_ratio=mech.get("pivot_ratio"),
               E_barrier=mech.get("E_barrier"), W_drive=mech.get("W_drive"))
    src.update(mmf=sw["mmf"], mmf_need=sw["mmf_need"], v_need=sw["v_need"],
               e_switch=sw["e_required"] * dsg.n_faces)
    src.update(drv_mass=(drv.mass if drv and drv.feasible else None),
               drv_price=(drv.price if drv and drv.feasible else None),
               drv_cap=(drv.cap_name if drv and drv.feasible else None),
               drv_mosfet=(drv.mosfet_name if drv and drv.feasible else None))
    src.update(feasible=feasible, scalar=scalar, violations=violations)
    return {k: src.get(k) for k in ROW_FIELDS}


def evaluate(dsg, fidelity="normal", use_prescreen=True):
    """Run every stage and score.  Returns a flat dict for tabulation."""
    from driver import select_driver
    from module import build_module

    sw = stage3_switching(dsg)
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=dsg.n_faces)

    if use_prescreen:
        ok, why = prescreen(dsg, sw, drv)
        if not ok:
            blank = dict(J_attract=np.nan, J_repel=np.nan,
                         margin_attract=np.nan, margin_repel=np.nan,
                         F_attract=0.0, F_repel=0.0, asymmetry=np.inf,
                         margin=np.nan)
            mod = build_module(dsg, drv if drv.feasible else None)
            mech = dict(m_module=mod.mass, hold_ratio=0.0, pivot_ratio=0.0,
                        E_barrier=np.nan, W_drive=0.0, fits=mod.fits)
            return _row(dsg, fidelity, blank, mech, sw, drv, False, 0.0,
                        "; ".join(why) + " [prescreen]")

    try:
        mag = stage1_magnetics(dsg, fidelity=fidelity)
    except RuntimeError as exc:
        # A stalled nonlinear solve is a property of the DESIGN, not a bug:
        # it happens when the magnet sits exactly on the knee of its own
        # demagnetisation curve, which is precisely the operating point the
        # margin constraint exists to forbid.  Record it as infeasible rather
        # than letting it abort the sweep.
        blank = dict(J_attract=np.nan, J_repel=np.nan,
                     margin_attract=np.nan, margin_repel=np.nan,
                     F_attract=0.0, F_repel=0.0, asymmetry=np.inf,
                     margin=1.0)
        mod = build_module(dsg, drv if drv.feasible else None)
        mech = dict(m_module=mod.mass, hold_ratio=0.0, pivot_ratio=0.0,
                    E_barrier=np.nan, W_drive=0.0, fits=mod.fits)
        return _row(dsg, fidelity, blank, mech, sw, drv, False, 0.0,
                    f"magnet solve stalled on the knee ({exc})")

    mod = build_module(dsg, drv if drv.feasible else None)
    mech = stage2_mechanics(dsg, mag, mod=mod)
    sc = score(dsg, mag, mech, sw, drv)
    return _row(dsg, fidelity, mag, mech, sw, drv, sc["feasible"],
                sc["scalar"], "; ".join(sc["violations"]))

