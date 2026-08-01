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
    """One candidate EPM module."""

    material: str = "LNGT72"
    d_mag: float = 4.75e-3       # magnet diameter
    l_mag: float = 12.5e-3       # magnet length
    circuit: str = "potcore"     # "none" or "potcore"
    t_steel: float = 1.0e-3      # keeper wall thickness
    r_clear: float = 1.0e-3      # radial clearance rod -> return annulus
    gap: float = 0.1e-3          # working air gap between mated faces
    n_faces: int = 6             # EPMs per module
    a_module: float = 40e-3      # module side length
    wire_d: float = 0.3e-3       # coil wire diameter
    v_cap: float = 70.0          # capacitor bank voltage
    c_cap: float = 10e-6         # capacitor bank capacitance

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
        div, ns, rfar_k, zfar_k, nq = 7.0, 4, 15, 12, 2000
        kw = dict(max_iter=12, continuation=False)
    else:
        div, ns, rfar_k, zfar_k, nq = 14.0, 6, 25, 20, 4000
        kw = dict(max_iter=25, continuation=True)
    n_slabs = n_slabs or ns
    rfar = rfar_k * max(ro, Rm)
    h = mesh or max(min(dsg.d_mag, dsg.l_mag) / div, 0.15e-3)

    out = {}
    for flip, tag in ((False, "attract"), (True, "repel")):
        m = AxisymModel(_regions(dsg, flip), rfar, zfar_k * dsg.l_mag, h,
                        n_slabs=n_slabs)
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
def masses(dsg):
    """Magnet, steel, shell and total module mass."""
    Rm = dsg.d_mag / 2
    v_mag = np.pi * Rm**2 * dsg.l_mag
    if dsg.circuit == "potcore":
        ro = Rm + dsg.r_clear + dsg.t_steel
        v_steel = (np.pi * ro**2 * dsg.t_steel +
                   np.pi * (ro**2 - (Rm + dsg.r_clear)**2) * dsg.l_mag)
    else:
        v_steel = 0.0
    a = dsg.a_module
    v_shell = a**3 - (a * 0.9) ** 3
    m_unit = v_mag * RHO_ALNICO + v_steel * RHO_STEEL
    return dict(m_magnet=v_mag * RHO_ALNICO, m_steel=v_steel * RHO_STEEL,
                m_shell=v_shell * RHO_SHELL,
                m_module=dsg.n_faces * m_unit + v_shell * RHO_SHELL)


def stage2_mechanics(dsg, mag):
    """Static feasibility of latching, hanging and pivoting.

    Pivot model: a module tips about the shared edge of its mating face.  The
    driving torque comes from one face repelling and the opposite face
    attracting, both acting at roughly half the module width from the pivot
    edge; gravity resists through the centre of mass at the same lever arm.
    This is a static go/no-go, not a trajectory - Stage 2 proper (MuJoCo) will
    replace it once a geometry is chosen.
    """
    m = masses(dsg)
    w = m["m_module"] * G
    lever = dsg.a_module / 2

    tau_drive = (mag["F_repel"] + mag["F_attract"]) * lever
    tau_gravity = w * lever

    return dict(**m,
                weight=w,
                hold_ratio=mag["F_attract"] / w,            # can it hang?
                pivot_ratio=tau_drive / tau_gravity,        # can it tip?
                tau_drive=tau_drive, tau_gravity=tau_gravity)


# --------------------------------------------------------------------------
# Stage 3: switching
# --------------------------------------------------------------------------
def stage3_switching(dsg, k_switch=3.0, winding_build=None):
    """Can the coil reverse the magnet, and what does it cost?

    Peak current is taken from the underdamped LC limit, which the feasibility
    audit showed is the operative regime for a practical coil: the 0.1 mm
    winding failed because it was overdamped, not because it lacked turns.
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
    i_peak = dsg.v_cap / z0 if R < 2 * z0 else dsg.v_cap / R   # under/over
    mmf = n_turns * i_peak

    hcj = MATERIALS[dsg.material]["Hcj"]
    mmf_need = k_switch * hcj * dsg.l_mag

    v_mag = np.pi * (dsg.d_mag / 2) ** 2 * dsg.l_mag
    e_hyst = 4.0 * MATERIALS[dsg.material]["Br"] * hcj * v_mag
    e_bank = 0.5 * dsg.c_cap * dsg.v_cap**2

    return dict(n_turns=n_turns, R_coil=R, L_coil=L, i_peak=i_peak,
                mmf=mmf, mmf_need=mmf_need,
                switch_ok=mmf >= mmf_need,
                switch_margin=mmf / mmf_need,
                e_hysteresis=e_hyst, e_bank=e_bank,
                e_total_module=e_bank * dsg.n_faces)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
MARGIN_LIMIT = 0.80          # H/Hcj above this erases the magnet in service
HOLD_MIN = 3.0               # attraction must exceed 3x module weight
PIVOT_MIN = 1.2              # drive torque must exceed gravity by 20 %


def score(dsg, mag=None, mech=None, sw=None):
    """Objectives and constraints for one design.

    Objectives are returned separately rather than collapsed into one number,
    so a multi-objective search (NSGA-II) can use them directly.  A scalar
    fallback is provided for single-objective methods.
    """
    mag = mag or stage1_magnetics(dsg)
    mech = mech or stage2_mechanics(dsg, mag)
    sw = sw or stage3_switching(dsg)

    violations = []
    if mag["margin"] > MARGIN_LIMIT:
        violations.append(f"demag margin {mag['margin']:.2f} > {MARGIN_LIMIT}")
    if not sw["switch_ok"]:
        violations.append(f"unswitchable (MMF {sw['switch_margin']:.2f} of need)")
    if mech["hold_ratio"] < HOLD_MIN:
        violations.append(f"hold {mech['hold_ratio']:.1f} < {HOLD_MIN}")
    if mech["pivot_ratio"] < PIVOT_MIN:
        violations.append(f"pivot {mech['pivot_ratio']:.2f} < {PIVOT_MIN}")

    objectives = dict(
        f_attract=mag["F_attract"],            # maximise
        f_repel=mag["F_repel"],                # maximise
        asymmetry=mag["asymmetry"],            # minimise
        e_switch=sw["e_total_module"],         # minimise
        mass=mech["m_module"],                 # minimise
    )

    # scalar fallback: geometric mean of the "more is better" terms divided by
    # the "less is better" terms, so no single term can dominate by scale
    feasible = not violations
    scalar = 0.0 if not feasible else (
        (objectives["f_repel"] ** 0.5 * objectives["f_attract"] ** 0.5) /
        (objectives["asymmetry"] ** 0.25 *
         (objectives["e_switch"] * 1e3) ** 0.25 *
         (objectives["mass"] * 1e3) ** 0.25))

    return dict(objectives=objectives, violations=violations,
                feasible=feasible, scalar=scalar,
                magnetics=mag, mechanics=mech, switching=sw)


# --------------------------------------------------------------------------
# Cheap pre-screen
# --------------------------------------------------------------------------
def prescreen(dsg):
    """Reject hopeless designs in milliseconds, before paying for the FEM.

    Two of the constraints do not need FEM accuracy to evaluate:

    * switching feasibility is a circuit calculation and involves no field
      solve at all;
    * the demagnetisation margin of an isolated rod is available in closed form
      from the validated free-space solver, and a design whose rod already
      exceeds the limit with no neighbour present can only get worse once a
      neighbour reverses against it.

    This matters for cost as well as time: the low-coercivity designs are both
    the physically hopeless ones AND the numerically stiff ones, so screening
    them out analytically removes most of the expensive FEM work.

    Returns (ok, reasons).
    """
    reasons = []

    sw = stage3_switching(dsg)
    if not sw["switch_ok"]:
        reasons.append(f"unswitchable (MMF {sw['switch_margin']:.2f} of need)")

    mat = material(dsg.material)
    pair = CoaxialRodPair(dsg.d_mag / 2, dsg.l_mag, mat, n_slabs=12)
    try:
        _, H = pair.solve(1e3 * dsg.l_mag)      # isolated rod
        margin_open = float(np.mean(np.abs(H[:12]))) / MATERIALS[dsg.material]["Hcj"]
    except RuntimeError:
        return False, ["free-space solve failed"]

    # a closed circuit can only improve on the open-circuit margin, so this is
    # a valid lower bound only for the uncircuited case
    if dsg.circuit == "none" and margin_open > MARGIN_LIMIT:
        reasons.append(f"open-circuit demag margin {margin_open:.2f} "
                       f"> {MARGIN_LIMIT}")

    return (not reasons), reasons


def evaluate(dsg, fidelity="normal", use_prescreen=True):
    """Run all three stages and score.  Returns a flat dict for tabulation."""
    row = dsg.as_row()
    row["fidelity"] = fidelity

    if use_prescreen:
        ok, why = prescreen(dsg)
        if not ok:
            row.update(dict(J_attract=np.nan, J_repel=np.nan,
                            margin_attract=np.nan, margin_repel=np.nan,
                            F_attract=0.0, F_repel=0.0, asymmetry=np.inf,
                            margin=np.nan))
            sw = stage3_switching(dsg)
            mech = masses(dsg)
            row.update(dict(m_module=mech["m_module"], hold_ratio=0.0,
                            pivot_ratio=0.0, mmf=sw["mmf"],
                            mmf_need=sw["mmf_need"],
                            switch_margin=sw["switch_margin"],
                            e_switch=sw["e_total_module"]))
            row.update(dict(feasible=False, scalar=0.0,
                            violations="; ".join(why) + " [prescreen]"))
            return row

    mag = stage1_magnetics(dsg, fidelity=fidelity)
    mech = stage2_mechanics(dsg, mag)
    sw = stage3_switching(dsg)
    sc = score(dsg, mag, mech, sw)
    row.update({k: v for k, v in mag.items()})
    row.update(dict(m_module=mech["m_module"], hold_ratio=mech["hold_ratio"],
                    pivot_ratio=mech["pivot_ratio"]))
    row.update(dict(mmf=sw["mmf"], mmf_need=sw["mmf_need"],
                    switch_margin=sw["switch_margin"],
                    e_switch=sw["e_total_module"]))
    row.update(dict(feasible=sc["feasible"], scalar=sc["scalar"],
                    violations="; ".join(sc["violations"])))
    return row
