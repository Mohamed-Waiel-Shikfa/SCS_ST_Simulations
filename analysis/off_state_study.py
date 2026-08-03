"""Is the OFF state stable when a neighbour is switched on?

This is load-bearing for the whole architecture and had never been tested.
Per-face polarity control assumes a face can be switched off and STAY off.  But
an off face sits a fraction of a millimetre from a live one, and Alnico has low
coercivity by design - that is why it is switchable at all.  If the neighbour's
field re-magnetises the off face, then "off" is not a state the module can
hold, and latching, releasing and sequencing all fail.

Three questions, kept separate because they carry different confidence:

  [1] EXACT.  What field does a live neighbour impose inside an off magnet, as
      a fraction of Hcj?  This is a straight FEM calculation - the off magnet
      is a permeable body with zero remanence and the neighbour is the source.
      No extra modelling, no free parameters.

  [2] EXACT.  Does the steel return path shield the off face or funnel flux
      into it?  Same calculation, with and without the pot core.

  [3] MODELLED, so lower confidence.  How much remanence does that field
      actually induce?  This needs a virgin magnetisation curve, which the
      datasheet does not provide - a datasheet only describes the second
      quadrant starting from saturation.  The estimate below is derived from
      the demagnetisation curve under a Preisach-style switching-field
      argument, and is stated as such.

  [4] EXACT, using the recoil model already in magnet_force.  Where does a
      magnet pulsed to J = 0 actually settle once the pulse ends?  This is a
      different question from [1]-[3] and it has a practical answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "simulations" / "Force_compute" / "python"))

from axisym_fem import AxisymModel, Region  # noqa: E402
from framework import Design, material  # noqa: E402

BASE = dict(n_gon=8, r_face=19.4e-3, gap=0.1e-3, wire_d=0.25e-3,
            v_cap=90.0, c_cap=100e-6)

CASES = [
    ("as built    LNG37  bare rod",
     dict(material="LNG37", circuit="none", d_mag=4.75e-3, l_mag=12.5e-3,
          t_steel=0.5e-3, r_clear=0.0)),
    ("GA winner   LNGT44 pot core",
     dict(material="LNGT44", circuit="potcore", d_mag=4.2e-3, l_mag=8.4e-3,
          t_steel=1.0e-3, r_clear=0.6e-3)),
    ("high coerc  LNGT72 pot core",
     dict(material="LNGT72", circuit="potcore", d_mag=4.2e-3, l_mag=8.4e-3,
          t_steel=1.0e-3, r_clear=0.6e-3)),
]


def mixed_regions(dsg, off_first=True):
    """Region A switched OFF (zero remanence), region B live.

    ``strength = 0`` makes A's J(H) identically zero, so it enters the solve as
    a permeable body of relative permeability mu_rec carrying no source.  That
    is exactly the right description of a demagnetised magnet for the purpose
    of finding the field imposed on it.
    """
    Rm, Lm, gap = dsg.d_mag / 2, dsg.l_mag, dsg.gap
    live = material(dsg.material)
    off = live.scaled(0.0)
    mA, mB = (off, live) if off_first else (live, off)
    regs = [
        Region(0, Rm, -Lm, 0.0, "magnet", "A", material=mA, direction=+1),
        Region(0, Rm, gap, gap + Lm, "magnet", "B", material=mB,
               direction=+1),
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


def solve_mixed(dsg, h_frac=8.0):
    """Return (H_in_off, J_in_off, H_in_live) in SI units."""
    Rm = dsg.d_mag / 2
    ro = Rm + (dsg.r_clear + dsg.t_steel if dsg.circuit == "potcore" else 0.0)
    h = max(min(dsg.d_mag, dsg.l_mag) / h_frac, 0.2e-3)
    m = AxisymModel(mixed_regions(dsg), r_far=25 * max(ro, Rm),
                    z_far=20 * dsg.l_mag, h=h, n_slabs=6)
    s = m.solve()
    J_off, H_off = m.region_state(s, "A")
    J_on, H_on = m.region_state(s, "B")
    return H_off, J_off, H_on, J_on


def virgin_remanence(mat, H):
    """Remanence induced in a demagnetised magnet by a field of magnitude H.

    NOT from the datasheet.  A datasheet demagnetisation curve starts from
    saturation and says nothing about the virgin curve, so this is derived.

    Take a distribution of switching fields with cumulative fraction F(h).
    Starting saturated and applying -H flips every domain below H, giving
    J = Br (1 - 2F(H)), which is the measured demagnetisation curve - so the
    curve fit already determines F:

        F(H) = (1 - J_major(H) / Br) / 2

    An ideally demagnetised sample has, at each switching field, equal
    populations up and down.  Applying +H flips the down half of the fraction
    below H, so the induced remanence is Br F(H).

    Valid only for H < Hcj, because the fitted curve carries no information
    beyond that point.  It is a first-order estimate, not a measurement.
    """
    Hc = mat.Hcj
    Hm = np.minimum(np.abs(H), Hc)
    F = (1.0 - mat.J(-Hm) / mat.Br) / 2.0
    return mat.Br * F


def solve_force_with(dsg, mat_A, h_frac=8.0):
    """Attraction when face A carries a WEAKENED magnet and B is fully live.

    Used to price the residual left by an untuned off pulse.  A rescaled
    Material is passed in rather than a strength fraction so the caller stays
    responsible for deciding what "weakened" means.
    """
    from axisym_fem import axial_force

    Rm, Lm, gap = dsg.d_mag / 2, dsg.l_mag, dsg.gap
    live = material(dsg.material)
    ro = Rm + (dsg.r_clear + dsg.t_steel if dsg.circuit == "potcore" else 0.0)
    regs = [
        Region(0, Rm, -Lm, 0.0, "magnet", "A", material=mat_A, direction=+1),
        Region(0, Rm, gap, gap + Lm, "magnet", "B", material=live,
               direction=-1),
    ]
    if dsg.circuit == "potcore":
        regs += [
            Region(0, ro, -Lm - dsg.t_steel, -Lm, "steel", "backA"),
            Region(Rm + dsg.r_clear, ro, -Lm, 0.0, "steel", "annA"),
            Region(0, ro, gap + Lm, gap + Lm + dsg.t_steel, "steel", "backB"),
            Region(Rm + dsg.r_clear, ro, gap, gap + Lm, "steel", "annB"),
        ]
    rfar = 25 * max(ro, Rm)
    h = max(min(dsg.d_mag, dsg.l_mag) / h_frac, 0.2e-3)
    m = AxisymModel(regs, r_far=rfar, z_far=20 * Lm, h=h, n_slabs=6)
    s = m.solve()
    return abs(axial_force(s, dsg.gap / 2, r_max=0.9 * rfar, n=4000))


print("=" * 84)
print("OFF-STATE STABILITY UNDER A LIVE NEIGHBOUR")
print("=" * 84)

print("\n[1][2] Field imposed on an OFF face, and what the return path does")
print("       Both columns are exact FEM; no virgin-curve model involved.\n")
print(f"  {'design':<30}{'Hcj kA/m':>10}{'H in OFF':>11}{'|H|/Hcj':>10}"
      f"{'J leak':>9}")
rows = []
for label, kw in CASES:
    d = Design(**{**BASE, **kw})
    mat = material(d.material)
    H_off, J_off, H_on, J_on = solve_mixed(d)
    frac = abs(H_off) / mat.Hcj
    rows.append((label, d, mat, H_off, J_off, frac))
    print(f"  {label:<30}{mat.Hcj/1e3:10.0f}{H_off/1e3:9.1f}k{frac:10.2f}"
          f"{J_off:9.3f}", flush=True)

print("\n  The 'J leak' column is the polarisation the OFF magnet carries")
print("  purely by being permeable - induced, not remanent.  It vanishes when")
print("  the neighbour is switched off, so it is not a failure of the OFF")
print("  state, but it does mean an OFF face is never magnetically invisible.")

print("\n  Shielding check: does the pot core help?\n")
d_bare = Design(**{**BASE, **dict(material="LNGT44", circuit="none",
                                  d_mag=4.2e-3, l_mag=8.4e-3,
                                  t_steel=0.5e-3, r_clear=0.0)})
H_b, J_b, _, _ = solve_mixed(d_bare)
d_pot = Design(**{**BASE, **CASES[1][1]})
H_p, J_p, _, _ = solve_mixed(d_pot)
mat44 = material("LNGT44")
print(f"    LNGT44 bare rod : |H|/Hcj = {abs(H_b)/mat44.Hcj:.2f}")
print(f"    LNGT44 pot core : |H|/Hcj = {abs(H_p)/mat44.Hcj:.2f}"
      f"   ({abs(H_p)/max(abs(H_b),1e-9):.2f}x the bare rod)")

print("\n[3] Induced remanence  (MODEL - see docstring, not a measurement)\n")
print(f"  {'design':<30}{'|H|/Hcj':>10}{'J_induced':>12}{'as % of Br':>13}")
for label, d, mat, H_off, J_off, frac in rows:
    Jv = float(virgin_remanence(mat, abs(H_off)))
    print(f"  {label:<30}{frac:10.2f}{Jv:11.4f}T{Jv/mat.Br*100:12.1f}%")

print("\n  Why the answer is small even at a large field fraction: the top of")
print("  an Alnico demagnetisation curve is flat, so almost no domains have")
print("  switching fields in the lower part of the range.  The OFF state is")
print("  protected by the same curve shape that makes the material switchable")
print("  at all - but the protection collapses once |H| approaches Hcj.")

print("\n[4] Where does a pulsed-to-zero magnet actually settle?")
print("    A DC pulse that brings J to zero is applied and then removed.  The")
print("    magnet recoils along a line of slope mu_rec back to its operating")
print("    point, and recoil RAISES J.  So pulsing to J = 0 does not leave")
print("    the magnet at J = 0 once the pulse ends.\n")
print(f"  {'design':<30}{'H_op kA/m':>11}{'J after recoil':>16}{'% of Br':>10}")
for label, d, mat, H_off, J_off, frac in rows:
    # operating point of the OFF magnet, from the FEM above
    H_op = -abs(H_off)
    J_settle = float(mat.J_recoil(H_op, -mat.Hcj))
    print(f"  {label:<30}{H_op/1e3:10.1f}k{J_settle:15.4f}T"
          f"{J_settle/mat.Br*100:9.1f}%")

print("\n  This is a CONTROL problem, not a materials problem: the pulse has")
print("  to be tuned so the magnet lands at zero AFTER recoil, which means")
print("  overshooting past J = 0 during the pulse.  Marchese et al. reached")
print("  their off state empirically by sweeping pulse length, which is")
print("  consistent with exactly this.")

print("\n[5] What that residual costs, in force")
print("    An OFF face that is really at 10-15 % of Br is not off in any")
print("    useful sense unless the force it still exerts is negligible next")
print("    to the module's own weight.  Force goes roughly as J^2, so a 13 %")
print("    remanence should cost under 2 % of full force - but 'roughly' is")
print("    not good enough when the answer decides whether a module can let")
print("    go, so it is computed rather than scaled.\n")
print(f"  {'design':<30}{'F full':>9}{'F untuned':>11}{'as %':>7}"
      f"{'weight':>9}{'F/weight':>10}")
for label, d, mat, H_off, J_off, frac in rows:
    from driver import select_driver
    from framework import stage1_magnetics, stage3_switching
    from module import build_module

    full = stage1_magnetics(d, fidelity="screen", states=("attract",))
    frac_rem = float(mat.J_recoil(-abs(H_off), -mat.Hcj)) / mat.Br
    weak = material(d.material).scaled(frac_rem)
    res = solve_force_with(d, weak)

    sw = stage3_switching(d)
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=d.n_faces)
    mod = build_module(d, drv if drv.feasible else None)
    w = mod.mass * 9.81
    print(f"  {label:<30}{full['F_attract']:8.2f}N{res:10.3f}N"
          f"{res/full['F_attract']*100:6.1f}%{w:8.2f}N{res/w:10.2f}")

print("\n  A residual comparable to the module's own weight means an untuned")
print("  OFF face can still hold a module against gravity - which is exactly")
print("  the failure mode that matters, because the robot would be unable to")
print("  release.  A tuned pulse is therefore not an optimisation, it is a")
print("  requirement, and the driver has to be able to hit it repeatably.")
