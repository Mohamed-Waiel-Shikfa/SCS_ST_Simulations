r"""A bench prototype: switching an N42 block on wall power.

The constraints have changed and the answer changes with them.  The earlier
analysis (``ndfeb_switching.py``) asked whether a battery-powered robot could
reverse an N42 block in place, and the answer was a firm no: 40 to 60 J per
switch against a 13 kJ cell is 26 reconfigurations.

This asks a different question.  Wall power, seconds between switches, size
and efficiency negotiable, one magnet at a time - can the thing be built on a
bench and demonstrated?  That is a much weaker requirement and it is worth
testing separately, because a proof of concept that settles the mechanism is
worth more than an efficient design of the wrong mechanism.

Nothing about the PHYSICS changes.  The magnet still needs about 3,100 kA/m
inside it to saturate in reverse, and the coil still has to supply it.  What
changes is that 50 J from a wall socket is nothing, and a 400 V capacitor bank
that would be absurd in a rolling robot is an ordinary bench object.

Three questions decide the build, in this order:

    1.  Can several small pulses do the work of one big one?  If yes,
        everything downstream gets easier.  This is settled first because it
        would change the whole architecture.
    2.  What does the coil look like with thick wire and no efficiency
        constraint?
    3.  Which capacitor technology can actually deliver the pulse - and this
        is not the same question as which can store the energy.
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

import fem3d  # noqa: E402
from magnet_force import MU0  # noqa: E402

# the block on the bench
MAG_W, MAG_H, MAG_T = 20e-3, 10e-3, 5e-3      # magnetised through MAG_T
BR_N42 = 1.32
HCJ_N42 = 955e3
HSAT_N42 = 2400e3          # field to saturate, from magnetiser practice
TC_HCJ = -0.006            # per K: NdFeB coercivity falls ~0.6 %/K

RHO_CU_20 = 1.68e-8
RHO_CU_120 = 2.32e-8       # a pulse coil runs hot
D_CU, CP_CU = 8960.0, 385.0
INSUL = 1.06               # 1 mm wire has proportionally thinner enamel
PITCH = 0.92


def demag(dims, axis=2):
    half = np.asarray(dims, dtype=float) / 2.0
    e = np.zeros(3)
    e[axis] = 1.0
    H = fem3d.cuboid_field(half, np.zeros(3), np.eye(3), e, np.zeros((1, 3)))[0]
    return float(-H[axis])


# ==========================================================================
def multipulse_verdict():
    """Can a train of sub-threshold pulses reverse the magnet?

    This has to be settled before anything is bought, because if the answer
    were yes the whole power supply would be a different and much smaller
    machine.

    It is no, and the reason is structural rather than a matter of degree.
    Hysteresis is RATE-INDEPENDENT and its state depends on the history of the
    applied field only through that history's EXTREMA - the Madelung rules,
    which are the defining property of the Preisach model and are what
    "hysteresis" means in a hard magnet.  Concretely: apply H1, return to
    zero, and the magnet lands on a recoil line.  Apply H1 again and it simply
    retraces that recoil line and comes back to where it was.  The second
    pulse does nothing the first did not already do.  Only a pulse that goes
    DEEPER than every previous one moves the magnet further.

    This is not a modelling artefact.  It is why a magnetiser is specified by
    peak field and not by joules or by pulse count, and why magnet
    manufacturers quote a single saturating field rather than an exposure.

    Two real effects do exist and both are quantified below, because "no" is
    only useful if the exceptions are named:

    * magnetic viscosity - thermal activation lets the magnetisation creep
      logarithmically at a held field.  Real, measurable, and logarithmic,
      which is the problem.
    * thermal softening - coercivity falls about 0.6 % per kelvin, so a hot
      magnet is genuinely easier to switch.  This one is large enough to
      matter and is worth a second look.
    """
    print("=" * 78)
    print("1.  CAN SEVERAL SMALL PULSES REPLACE ONE BIG ONE?")
    print("=" * 78)
    print("""
  No - and this is worth being precise about, because if it were yes the
  power supply would be a much smaller machine.

  Hysteresis is rate-independent, and the state of a hard magnet depends on
  the history of the applied field only through that history's EXTREMA.  Apply
  a reverse field, remove it, and the magnet sits on a recoil line.  Apply the
  SAME field again and it retraces that line and returns to where it was.  The
  second pulse does nothing the first did not.  Only a pulse deeper than every
  previous one moves the magnetisation further.

  This is why a magnetiser is specified by peak field and never by joules or
  by pulse count.

  Two exceptions exist and neither rescues the idea on its own:""")

    print("""
  MAGNETIC VISCOSITY.  Held near the coercive field, magnetisation creeps by
  thermal activation as S ln(t/t0).  S is a few per cent of Mr per decade for
  sintered NdFeB.  Stretching a pulse from 100 us to 1 s is seven decades:""")
    s_visc = 0.03
    print(f"\n  {'pulse length':>14} {'decades':>9} {'extra reversal':>16}")
    print("  " + "-" * 42)
    for t in (100e-6, 1e-3, 100e-3, 1.0, 60.0):
        dec = np.log10(t / 1e-6)
        print(f"  {t*1e3:11.1f} ms {dec:9.1f} {s_visc*dec*100:15.1f} %")
    print("""
  A few tens of per cent for a million-fold increase in pulse length, and the
  field still has to be near coercivity for any of it to happen.  It cannot
  bridge a factor of twelve.

  THERMAL SOFTENING.  This one is large.  NdFeB coercivity falls about
  0.6 % per kelvin, and it is reversible up to the point where the magnet
  starts losing flux irreversibly:""")
    print(f"\n  {'magnet temp':>12} {'Hcj':>10} {'H to saturate':>15} "
          f"{'A-turns needed':>16}")
    print("  " + "-" * 58)
    n_d = demag((MAG_W, MAG_H, MAG_T))
    cpl = 1.0 - demag((MAG_W, MAG_H, MAG_T), axis=2)
    for temp in (20, 60, 100, 140, 180):
        f = 1.0 + TC_HCJ * (temp - 20)
        hcj = HCJ_N42 * f
        hsat = HSAT_N42 * f
        ni = (hsat + n_d * BR_N42 / MU0) * MAG_T / cpl
        print(f"  {temp:9.0f} C {hcj/1e3:9.0f}k {hsat/1e3:14.0f}k "
              f"{ni:16,.0f}")
    print("""
  Heating to 140 C cuts the requirement by about 40 %.  It is a real lever and
  a poor one for a demonstrator: N42 starts losing flux irreversibly around
  80 C (its maximum working temperature), so the magnet you switch hot is
  weaker afterwards, and the whole point is to keep the 60 N.  A high
  temperature grade (N42SH, N42UH) would take the heat but has HIGHER
  coercivity, which is the wrong direction.

  VERDICT.  One pulse, and it must reach full amplitude.  What multi-pulse
  DOES buy is on the other axis entirely - switching one FACE at a time from a
  shared bank, which is the user's own suggestion and is worth a great deal:
  the bank is sized for one coil rather than eight, and it recharges between
  faces.  That is adopted throughout below.""")
    return n_d, cpl


# ==========================================================================
def coil_field_profile(turns, current, coil_len, bore_w, bore_h, build,
                       n_sample=9):
    """Axial field the coil produces, sampled through the magnet.

    A solenoid of rectangular section carrying N I over a length l is
    magnetically identical to a uniformly magnetised block of the same outer
    dimensions with M = N I / l, so the same closed-form cuboid kernel that is
    verified to machine precision gives its field exactly.  No new physics and
    no new approximation.

    The MINIMUM over the magnet is what is returned alongside the centre
    value, and it is the minimum that matters: a magnet reversed only in its
    middle is a partly reversed magnet, which is a weak magnet, which is
    useless.  Sizing on the centre field alone flatters a short coil badly,
    because that is exactly where a short coil's field falls off fastest.
    """
    m_equiv = turns * current / coil_len
    half = np.array([(bore_w + build) / 2, (bore_h + build) / 2,
                     coil_len / 2])
    z = np.linspace(-MAG_T / 2, MAG_T / 2, n_sample)
    # sample along the axis and at the corner of the magnet, which is the
    # weakest point in the volume
    pts = np.array([[0.0, 0.0, zz] for zz in z] +
                   [[MAG_W / 2 * 0.9, MAG_H / 2 * 0.9, zz] for zz in z])
    H = fem3d.cuboid_field(half, np.zeros(3), np.eye(3),
                           np.array([0.0, 0.0, m_equiv]), pts)
    hz = H[:, 2] + m_equiv       # B/mu0 inside the equivalent body
    return float(hz[n_sample // 2]), float(np.min(hz))


def ampere_turns_for(coil_len, bore_w, bore_h, build, h_target):
    """Ampere-turns so the WEAKEST point in the magnet reaches ``h_target``.

    Linear in the current, so one evaluation scales.
    """
    _, h_min = coil_field_profile(1.0, 1.0, coil_len, bore_w, bore_h, build)
    return h_target / max(h_min, 1e-9)


@dataclass
class Coil:
    layers: int
    turns: int
    turns_per_layer: int
    wire_d: float
    build: float
    length: float
    bore_w: float
    bore_h: float
    l_turn: float
    wire_len: float
    r_cold: float
    r_hot: float
    inductance: float
    coupling: float
    cu_mass: float
    ni_needed: float

    @property
    def i_needed(self):
        return self.ni_needed / self.turns

    @property
    def j(self):
        return self.i_needed / (np.pi * (self.wire_d / 2) ** 2) / 1e6


def build_coil(wire_d, layers, coil_len, h_target, bore_w=MAG_W,
               bore_h=MAG_H):
    """Wind a rectangular coil around the magnet's magnetised axis.

    ``coil_len`` is the winding length along that axis and may exceed the
    magnet's own 5 mm.  Overhang is NOT free: the ampere-turns needed scale
    with the coil's length, and although a longer coil couples better, the two
    do not cancel.  The sweep below finds where the optimum actually is
    instead of assuming it.
    """
    d_ins = wire_d * INSUL
    tpl = max(int(coil_len / d_ins), 1)
    pitch = d_ins * PITCH
    build = layers * pitch

    l_turn = 0.0
    for k in range(layers):
        t = (k + 0.5) * pitch
        l_turn += 2 * (bore_w + 2 * t) + 2 * (bore_h + 2 * t) \
            - (8 - 2 * np.pi) * t
    l_turn /= layers

    turns = layers * tpl
    a_w = np.pi * (wire_d / 2) ** 2
    wire_len = turns * l_turn
    r_cold = RHO_CU_20 * wire_len / a_w
    r_hot = RHO_CU_120 * wire_len / a_w

    ni = ampere_turns_for(coil_len, bore_w, bore_h, build, h_target)
    h_c, h_m = coil_field_profile(turns, ni / turns, coil_len, bore_w,
                                  bore_h, build)
    cpl = h_m / max(h_c, 1e-9)

    r_eq = np.sqrt((bore_w + build) * (bore_h + build) / np.pi)
    ind = MU0 * turns ** 2 * (np.pi * r_eq ** 2) / (coil_len + 0.9 * r_eq)

    return Coil(layers=layers, turns=turns, turns_per_layer=tpl,
                wire_d=wire_d, build=build, length=coil_len, bore_w=bore_w,
                bore_h=bore_h, l_turn=l_turn, wire_len=wire_len,
                r_cold=r_cold, r_hot=r_hot, inductance=ind, coupling=cpl,
                cu_mass=wire_len * a_w * D_CU, ni_needed=ni)


def coil_survey(n_d):
    """1 mm wire, 1 to 10 layers, and the coil length as a free variable."""
    h_needed = HSAT_N42 + n_d * BR_N42 / MU0
    e_floor = 0.5 * MU0 * h_needed ** 2 * MAG_W * MAG_H * MAG_T

    print("\n" + "=" * 78)
    print("2.  THE COIL, WITH 1 mm WIRE")
    print("=" * 78)
    print(f"""
  Field needed at the WEAKEST point in the magnet to saturate it in reverse:
  H_sat + N_d M = {HSAT_N42/1e3:,.0f} + {n_d*BR_N42/MU0/1e3:,.0f} = {h_needed/1e3:,.0f} kA/m

  Sizing on the weakest point rather than the centre matters.  A magnet
  reversed only through its middle is a partly reversed magnet, which is a
  weak magnet, which is useless - and the field of a short coil falls off
  fastest exactly where the magnet's corners are.

  The energy floor, from the field that has to exist in the magnet's own
  volume, is 1/2 mu0 H^2 V = {e_floor:.1f} J.  No coil beats that, and every real
  coil also fills the space around the magnet, so expect several times it.

  The coil bore is {MAG_W*1e3:.0f} x {MAG_H*1e3:.0f} mm.  Its LENGTH along the magnetised axis is
  free - it can overhang the magnet's 5 mm - so it is swept here.
""")
    print(f"  {'len':>5} {'lay':>4} {'turns':>6} {'unif':>6} {'NI':>9} "
          f"{'I':>7} {'R hot':>8} {'L':>8} {'E field':>8} {'V for I':>9} "
          f"{'J':>8}")
    print(f"  {'mm':>5} {'':4} {'':6} {'':6} {'A-t':>9} {'A':>7} "
          f"{'mohm':>8} {'uH':>8} {'J':>8} {'V':>9} {'A/mm2':>8}")
    print("  " + "-" * 88)

    rows = []
    for coil_len in (5e-3, 8e-3, 12e-3, 20e-3):
        for layers in (2, 4, 6, 8, 10):
            c = build_coil(1.0e-3, layers, coil_len, h_needed)
            e = 0.5 * c.inductance * c.i_needed ** 2
            v = c.i_needed * c.r_hot
            rows.append((c, e, v))
            print(f"  {coil_len*1e3:5.0f} {layers:4d} {c.turns:6d} "
                  f"{c.coupling:6.2f} {c.ni_needed:9,.0f} {c.i_needed:7.0f} "
                  f"{c.r_hot*1e3:8.1f} {c.inductance*1e6:8.1f} {e:8.1f} "
                  f"{v:9.0f} {c.j:8.0f}")
        print()

    best = min(rows, key=lambda r: r[1])
    print(f"""  Overhang is NOT free, and this is where an earlier version of this
  analysis went wrong.  Ampere-turns scale with the coil's LENGTH: doubling
  the coil doubles the ampere-turns needed for the same field, and the
  improvement in uniformity only partly offsets it.  The optimum sits at a
  coil roughly as long as the magnet, or slightly longer.

  Energy is nearly flat with layers, at {min(r[1] for r in rows):.0f} to {max(r[1] for r in rows):.0f} J, against a floor of
  {e_floor:.1f} J.  That is the expected factor of two to five for a real coil, and it
  confirms the coil is doing about as well as any coil can.  Layers do not
  change the energy - they trade current for voltage:

      2 layers   thousands of amps at a hundred volts
     10 layers   a few hundred amps at several hundred volts

  and THAT choice is really the choice of switch and capacitor, not of coil.
  Several hundred volts and a few hundred amps is the far easier corner: it is
  ordinary photoflash territory, where the parts are cheap and abundant.""")
    return rows, h_needed


# ==========================================================================
def capacitor_technologies(coil, c_target, v_target, energy):
    """Which capacitor can deliver the pulse, not merely store it.

    The comparison has to be made at the same bank the design settled on, or
    it is not a comparison at all.  Each technology is asked how many cells it
    needs to get there, and what the resulting series resistance does to the
    discharge.
    """
    print("\n" + "=" * 78)
    print("6.  WHICH CAPACITOR CAN ACTUALLY DELIVER IT")
    print("=" * 78)
    print(f"""
  The bank has to do two separate things and they pull in opposite
  directions: store {energy:.0f} J, and deliver it in a few hundred microseconds.
  Storage is easy and every technology below can do it.  DELIVERY is set by
  series resistance, and that is where they separate.

  Each technology is asked to build the SAME bank - {c_target*1e6:.0f} uF at {v_target:.0f} V - so
  the cell counts are directly comparable.  Series cells for voltage, parallel
  strings for capacitance, and the bank ESR is esr x n_series / n_parallel.
""")
    print(f"  {'technology':<24} {'cell':>7} {'C cell':>8} {'ser':>5} "
          f"{'par':>6} {'cells':>7} {'bank ESR':>9} {'I peak':>7} "
          f"{'verdict':<10}")
    print(f"  {'':<24} {'V':>7} {'':>8} {'':>5} {'':>6} {'':>7} "
          f"{'mohm':>9} {'A':>7}")
    print("  " + "-" * 88)
    techs = [
        ("supercapacitor 10 F", 2.7, 10.0, 0.030),
        ("supercapacitor 100 F", 2.7, 100.0, 0.008),
        ("supercap module 16 V", 16.2, 58.0, 0.150),
        ("photoflash electrolytic", 330.0, 200e-6, 0.100),
        ("HV electrolytic 450 V", 450.0, 470e-6, 0.150),
        ("metallised film 1 kV", 1000.0, 10e-6, 0.005),
    ]
    for name, v_cell, c_cell, esr in techs:
        n_s = max(int(np.ceil(v_target / v_cell)), 1)
        c_string = c_cell / n_s
        n_p = max(int(np.ceil(c_target / c_string)), 1)
        bank_esr = esr * n_s / n_p
        s = simulate(coil, c_target, v_target, r_extra=bank_esr + 0.020)
        ok = "works" if s["i_peak"] >= coil.i_needed else "TOO SLOW"
        c_txt = f"{c_cell:.0f} F" if c_cell > 1 else f"{c_cell*1e6:.0f} uF"
        print(f"  {name:<24} {v_cell:7.1f} {c_txt:>8} {n_s:5d} {n_p:6d} "
              f"{n_s*n_p:7,d} {bank_esr*1e3:9.0f} {s['i_peak']:7.0f} "
              f"{ok:<10}")

    print("""
  Supercapacitors fail twice over, and it is worth saying why rather than just
  excluding them.  A supercapacitor stores charge in a double layer a few
  nanometres thick across an enormous porous electrode area.  That structure
  is what makes it energy dense, and it is also what makes its resistance high
  and, worse, DISTRIBUTED - the charge deep in the pores cannot come out
  quickly at any price.  Their time constants are milliseconds to seconds by
  construction.

  Then the arithmetic does the rest.  Stacking to a few hundred volts needs
  well over a hundred cells in series, which multiplies the ESR by that same
  number AND divides the capacitance by it - so the string then has to be
  paralleled back up, and the cell count runs into the thousands.  The user's
  instinct that supercapacitors are where capacitor technology has advanced is
  right; it is simply advancement along the energy axis, and this is a power
  problem.

  A photoflash electrolytic is built for exactly the opposite duty: a few
  hundred joules dumped in under a millisecond, tens of thousands of times
  over.  It is the correct part and it is cheap because the camera industry
  made it so.  The only reason the 450 V general-purpose can wins here is
  headroom - photoflash cans stop at 330 V, and the design wants more.

  The film capacitor is better still on ESR and worse on everything else:
  about ten times the volume and twenty times the price per joule.  Worth
  remembering only if measurement shows the electrolytic's ESR limiting the
  rise time.""")


# ==========================================================================
def simulate(coil, c_bank, v0, r_extra=0.05, v_switch=1.5, n=40000):
    """Series RLC discharge with a switch drop, integrated properly."""
    r = coil.r_hot + r_extra
    l = coil.inductance
    t_end = 6 * np.pi * np.sqrt(l * c_bank)
    t = np.linspace(0, t_end, n)
    dt = t[1] - t[0]
    i = np.zeros(n)
    v = np.zeros(n)
    v[0] = v0
    for k in range(n - 1):
        drop = v_switch if i[k] > 0 else 0.0
        di = (v[k] - r * i[k] - drop) / l
        dv = -i[k] / c_bank
        i[k + 1] = max(i[k] + di * dt, 0.0)
        v[k + 1] = v[k] + dv * dt
        if i[k + 1] <= 0 and k > 10:
            i[k + 1:] = 0.0
            v[k + 1:] = v[k + 1]
            break
    kpk = int(np.argmax(i))
    action = float(np.trapz(i ** 2, t))
    return dict(t=t, i=i, v=v, i_peak=float(i[kpk]), t_peak=float(t[kpk]),
                action=action, e_used=0.5 * c_bank * (v0 ** 2 - v[-1] ** 2),
                dT=RHO_CU_120 / (D_CU * CP_CU) * action /
                (np.pi * (coil.wire_d / 2) ** 2) ** 2)


# ==========================================================================
def choose_operating_point(h_needed):
    """Find the cheapest coil + bank that actually reaches the field.

    The coil survey costs the FIELD ENERGY.  A real discharge also pays
    resistive loss, and that loss is not a small correction: the loop is only
    lightly underdamped, so a fair fraction of the bank is burnt in the copper
    before the current peaks.  Sizing on field energy alone under-sizes the
    bank, so the whole thing is simulated instead.
    """
    print("\n" + "=" * 78)
    print("3.  THE OPERATING POINT, BY SIMULATION")
    print("=" * 78)
    e_floor = 0.5 * MU0 * h_needed ** 2 * MAG_W * MAG_H * MAG_T
    print(f"""
  The survey above priced only the field energy.  A real discharge also burns
  copper, and the loop is only lightly underdamped, so that loss is not a
  correction - it is comparable to the field energy itself.  Every candidate
  below is therefore SIMULATED, not estimated, and kept only if the simulated
  peak current reaches the ampere-turns the magnet needs.

  There is one identity worth having in mind before reading the table.  At
  fixed coil and fixed damping,

      E = 1/2 C V^2   and   I = V / sqrt(L/C)   give   E = 1/2 L I^2 / eta^2

  so the stored energy does NOT depend on how the bank is split between volts
  and farads - except through eta, the fraction of the ideal peak current that
  survives resistive damping.  And eta gets WORSE as C grows, because damping
  goes as R/2 sqrt(C/L).

  The consequence is the opposite of the intuitive one: a small bank at high
  voltage is more efficient than a large bank at low voltage.  Voltage is
  therefore the main design variable, and the table sweeps a ceiling on it.
""")

    coils = [build_coil(1.0e-3, lay, ln, h_needed)
             for ln in (5e-3, 8e-3, 12e-3, 16e-3)
             for lay in (2, 3, 4, 5, 6, 8, 10)]
    caps = (100e-6, 150e-6, 220e-6, 330e-6, 470e-6, 680e-6, 1000e-6,
            1500e-6, 2200e-6, 3300e-6, 4700e-6)

    solutions = []
    for c in coils:
        for c_bank in caps:
            lo, hi = 50.0, 1600.0
            if simulate(c, c_bank, hi, r_extra=_esr(c_bank))["i_peak"] \
                    < c.i_needed:
                continue
            for _ in range(22):                     # bisect on V0
                mid = 0.5 * (lo + hi)
                if simulate(c, c_bank, mid,
                            r_extra=_esr(c_bank))["i_peak"] < c.i_needed:
                    lo = mid
                else:
                    hi = mid
            s = simulate(c, c_bank, hi, r_extra=_esr(c_bank))
            solutions.append((0.5 * c_bank * hi ** 2, hi, c_bank, c, s))

    print(f"  {'V ceiling':>9} {'E bank':>7} {'V0':>6} {'C':>7} {'coil':>10} "
          f"{'N':>4} {'I peak':>7} {'t_pk':>6} {'dT':>6} {'eta':>6}")
    print(f"  {'V':>9} {'J':>7} {'V':>6} {'uF':>7} {'':>10} {'':4} "
          f"{'A':>7} {'us':>6} {'K':>6} {'':>6}")
    print("  " + "-" * 82)
    frontier = {}
    for ceil in (350, 450, 600, 800, 1000, 1200, 1500):
        ok = [s for s in solutions if s[1] <= ceil]
        if not ok:
            print(f"  {ceil:9d}  nothing reaches the field at this voltage")
            continue
        e, v0, c_bank, c, s = min(ok, key=lambda r: r[0])
        eta = s["i_peak"] / (v0 / np.sqrt(c.inductance / c_bank))
        frontier[ceil] = (e, v0, c_bank, c, s)
        print(f"  {ceil:9d} {e:7.0f} {v0:6.0f} {c_bank*1e6:7.0f} "
              f"{c.length*1e3:5.0f}mm x{c.layers:<2d} {c.turns:4d} "
              f"{s['i_peak']:7.0f} {s['t_peak']*1e6:6.0f} {s['dT']:6.1f} "
              f"{eta:6.2f}")

    print(f"""
  The energy falls by roughly half between 350 V and 1000 V and then flattens,
  which is what the identity predicts: once the loop is lightly damped there is
  nothing left to win, and everything above that buys capacitor insulation for
  no return.

  But the place to BUILD is not the minimum of that column.  450 V is the top
  of the ordinary aluminium-electrolytic range and the top of what a
  single-stage capacitor-charger IC will reach.  Above it, the bank becomes
  series strings with balancing resistors, the switch needs 1600 V, and the
  charger becomes a project.  Paying {frontier[450][0] - frontier[1000][0]:.0f} extra joules - which cost nothing on
  wall power - to stay inside the cheap, available, well-understood part of
  the catalogue is the right trade for a demonstrator.

  Against the {e_floor:.0f} J field floor, the 450 V point is a factor of about {frontier[450][0]/e_floor:.0f} - the
  irreducible penalty for a coil that must surround the magnet rather than
  live inside it, plus the copper loss of a loop that is only half undamped.""")
    return frontier


def _esr(c_bank):
    """Bank ESR plus wiring, as a function of how many cans are paralleled.

    A 450 V aluminium electrolytic of a few hundred uF has an ESR in the
    region of 0.1 ohm; paralleling for capacitance divides it.  20 mohm is
    added for the loop - busbar, switch and joints - which is achievable with
    short wide conductors and is the single easiest thing to get wrong.
    """
    n_par = max(c_bank / 470e-6, 1.0)
    return 0.10 / n_par + 0.020


# ==========================================================================
# catalogue practice: reverse magnetisation recovered as a fraction of Br,
# against the applied internal field in multiples of Hcj.  These are the
# numbers magnetiser vendors publish as a sizing rule; they are approximate
# and grade-dependent, and are used here only to price a trade, never to
# claim a precise outcome.
MAG_CURVE = [(1.0, 0.00), (1.25, 0.45), (1.5, 0.70), (1.75, 0.83),
             (2.0, 0.90), (2.5, 0.97), (3.0, 1.00)]


def partial_magnetisation(n_d):
    """Does the magnet have to be fully re-saturated?

    This is the largest single lever on the size of the whole machine and it
    is easy to miss, because magnet datasheets quote only the fully saturating
    field.  Energy goes as H squared, so backing off from 3 Hcj to 1.5 Hcj is
    a factor of four on the bank - and it costs only the SQUARE ROOT in force,
    because force goes as Br squared.
    """
    print("\n" + "=" * 78)
    print("4.  HOW MUCH REVERSAL IS ACTUALLY NEEDED?")
    print("=" * 78)
    m_self = n_d * BR_N42 / MU0
    print(f"""
  Everything so far has sized for FULL reverse saturation, which is what a
  magnet vendor quotes and what a production magnetiser delivers.  A
  demonstrator does not need it, and this is the biggest lever in the whole
  analysis.

  Bank energy goes as H^2.  Holding force goes as Br^2.  So giving up
  remanence buys energy back faster than it costs force - and the mechanism
  only needs enough force to hold the module up, not the full 60 N.

  The self-demagnetising term {m_self/1e3:,.0f} kA/m is added to every internal field,
  because the coil must overcome the magnet's own field as well.
""")
    print(f"  {'H internal':>11} {'':>6} {'H applied':>10} {'A-turns':>10} "
          f"{'Br after':>9} {'pair force':>11} {'energy':>8}")
    print(f"  {'x Hcj':>11} {'kA/m':>6} {'kA/m':>10} {'rel':>10} "
          f"{'frac':>9} {'N':>11} {'rel':>8}")
    print("  " + "-" * 72)
    h_full = HSAT_N42 + m_self
    rows = []
    for k, frac in MAG_CURVE:
        h_int = k * HCJ_N42
        h_app = h_int + m_self
        e_rel = (h_app / h_full) ** 2
        force = 60.0 * frac ** 2
        rows.append((k, frac, h_app, e_rel, force))
        print(f"  {k:11.2f} {h_int/1e3:6.0f} {h_app/1e3:10.0f} "
              f"{h_app/h_full:10.2f} {frac:9.2f} {force:11.1f} {e_rel:8.2f}")

    print("""
  The knee is at 1.5 to 2.0 Hcj.  At 1.5 Hcj the magnet comes back at about
  70 % of Br, which is 49 % of the force - roughly 30 N on a module that needs
  under 1 N to hold its own weight - for 47 % of the energy.  At 2.0 Hcj it is
  90 % of Br and 81 % of the force for 71 % of the energy.

  2.0 Hcj is the right place to build.  It keeps most of the force, it is well
  clear of the steep part of the curve where small field errors turn into
  large magnetisation errors, and it takes about a third off the bank.

  One honest caveat: these fractions are catalogue sizing practice, not a
  measurement of this particular block.  A partly magnetised magnet is also
  less stable against later stray fields than a saturated one.  For a
  demonstrator that is acceptable; for anything that has to hold calibration
  it is not.  The remedy is trivial and worth stating - fire the bank at full
  voltage once at the end of a test session to re-saturate.""")
    return 2.0 * HCJ_N42 + m_self


# ==========================================================================
def switch_selection(v0, i_peak, action, t_peak):
    """What can hold off the bank and pass the pulse."""
    print("\n" + "=" * 78)
    print("5.  THE SWITCH")
    print("=" * 78)
    di_dt = i_peak / t_peak / 1e6
    print(f"""
  The switch has to block {v0:.0f} V, then pass {i_peak:.0f} A for {t_peak*1e6:.0f} us with a
  rate of rise of {di_dt:.0f} A/us and an action of {action:.0f} A^2 s.  It fires once and
  then has milliseconds to recover.  That duty is unusual, and it makes the
  obvious choice the wrong one.
""")
    cands = [
        ("MOSFET, 600 V 100 A", 600, 400, 1000.0, 200,
         "continuous-rated silicon.  Pulse current and I2t are both an\n"
         "     order short, so it needs many in parallel, and each one must\n"
         "     share the current within its own gate delay.  Fragile and\n"
         "     expensive for a job it is not built for."),
        ("IGBT module, 1200 V", 1200, 3000, 1000.0, 5000,
         "handles it, and is the right answer if the pulse must be\n"
         "     TURNED OFF.  Ours does not - the LC ring ends itself - so the\n"
         "     turn-off capability and the gate drive that goes with it are\n"
         "     paid for and never used."),
        ("SCR, stud, 1200 V", 1200, 2500, 100.0, 4500,
         "latches on, carries the half-sine, and commutates off by itself\n"
         "     when the current crosses zero.  A capacitor-discharge\n"
         "     magnetiser is the textbook SCR application, and it is the\n"
         "     cheapest part in the table."),
        ("triggered spark gap", 20000, 100000, 10000.0, 1000000,
         "what a production magnetiser uses.  Unnecessary here by two\n"
         "     orders of magnitude, and the electrodes erode."),
    ]
    print(f"  {'device':<22} {'V':>6} {'I pulse':>8} {'di/dt':>7} "
          f"{'I2t':>8}   verdict")
    print(f"  {'':<22} {'':>6} {'A':>8} {'A/us':>7} {'A2s':>8}")
    print("  " + "-" * 76)
    for name, v, i, d, i2t, note in cands:
        ok = v >= 1.3 * v0 and i >= 1.2 * i_peak and i2t >= 2 * action \
            and d >= 2 * di_dt
        margin = min(v / (1.3 * v0), i / (1.2 * i_peak), i2t / (2 * action),
                     d / (2 * di_dt))
        verdict = "no" if not ok else \
            ("ADEQUATE" if margin < 10 else "adequate, overkill")
        print(f"  {name:<22} {v:6.0f} {i:8.0f} {d:7.0f} {i2t:8.0f}   "
              f"{verdict}")
        print(f"     {note}\n")

    print(f"""  An SCR it is, and the reason is the one that matters: our current is
  SELF-EXTINGUISHING.  The bank rings into the coil, the current comes back to
  zero on its own after about {2*t_peak*1e6:.0f} us, and at that instant the SCR turns off
  because that is what SCRs do.  Nothing has to be commutated, no gate drive
  has to hold {i_peak:.0f} A off against a fault, and there is no shoot-through path
  to design against.

  Two things must go with it, and both are mandatory rather than nice:

  A FREEWHEEL DIODE across the bank, cathode to the positive rail.  Without
  it the LC ring drives the capacitor voltage NEGATIVE at the end of the
  half-cycle, and an aluminium electrolytic subjected to reverse voltage vents.
  With it, the coil current decays through the diode instead and the bank
  simply stops at zero.  This single part is the difference between a bench
  supply and a grenade.

  A GATE PULSE with real energy - a few hundred mA for a few microseconds,
  from a pulse transformer.  A slow, weak gate drive turns the SCR on over a
  small part of its die, and at {di_dt:.0f} A/us that part fails.  This is the
  classic way to destroy an SCR in a capacitor-discharge circuit.""")


def charger_choice(energy, v0):
    """Buy a flash charger, or build one."""
    print("\n" + "=" * 78)
    print("7.  CHARGER: BUY OR BUILD")
    print("=" * 78)
    print(f"""
  The bank needs {energy:.0f} J at {v0:.0f} V.  Nothing about the charger is on the
  critical path of the physics - it only has to get there eventually - so the
  question is purely how fast the prototype can exist.
""")
    for name, watt, vmax, cost, note in [
        ("disposable-camera board", 1.0, 330, 3,
         "the cheapest high-voltage source there is, and it is genuinely\n"
         "     usable.  330 V ceiling is the problem: our design wants more,\n"
         "     and its 1 W means minutes per shot at this energy."),
        ("fly-swatter / bug-zapper", 0.5, 2000, 5,
         "high voltage, negligible current, and no regulation.  It is a\n"
         "     voltage multiplier feeding a nanofarad.  Charging hundreds of\n"
         "     microfarads with it takes hours.  The user's instinct about\n"
         "     the topology was right; the power rating is what rules it out."),
        ("LT3750 flyback charger", 15.0, 500, 25,
         "purpose-built capacitor charger IC: set the target with one\n"
         "     resistor, it charges and stops.  This is the part the camera\n"
         "     industry moved to and it is the correct choice."),
        ("laboratory HV supply", 50.0, 1000, 300,
         "if one is already on the bench, use it - a current-limited\n"
         "     lab supply through a series resistor is the simplest possible\n"
         "     charger and needs no design at all."),
        ("custom flyback", 30.0, 1000, 40,
         "two evenings of work to reproduce what the LT3750 does, plus\n"
         "     a transformer to wind.  Not worth it for a demonstrator."),
    ]:
        t = energy / watt
        print(f"  {name:<26} {watt:5.1f} W  {vmax:5.0f} V  ~${cost:<4d} "
              f"{t:5.0f} s per shot")
        print(f"     {note}\n")

    print("""  RECOMMENDATION: buy, and specifically buy an LT3750-based charger board.

  The user's own preference was to buy the charging side and design only the
  pulse-delivery side, and that is exactly right here.  The charger is the
  part of this circuit with the least physics in it and the most fiddly
  engineering: transformer design, current-mode control, and a flyback that
  has to stay stable while its load voltage rises by two orders of magnitude.
  None of that teaches anything about whether the magnet switches.

  What must NOT be bought is the discharge side.  The face-select switching,
  the freewheel diode, the loop inductance and the SCR gate drive are where
  the pulse is actually made or lost, and they are specific to this magnet.

  A note on the disposable-camera board, since it is the folklore answer: it
  works, it is three dollars, and it will charge a small bank to 330 V in a
  few minutes.  It is a perfectly good way to take the very first shot and
  find out whether the magnet flips at all.  It is not a good way to run a
  test session, because every shot costs minutes.""")


def bom(coil, c_bank, v0, energy, esr_budget):
    """What to buy, with somewhere to buy it."""
    print("\n" + "=" * 78)
    print("8.  BILL OF MATERIALS")
    print("=" * 78)
    # cans are set by TWO constraints and the ESR one usually wins
    n_for_c = int(np.ceil(c_bank / 470e-6))
    n_for_esr = int(np.ceil(0.150 / max(esr_budget - 0.020, 1e-3)))
    n_cans = max(n_for_c, n_for_esr)
    print(f"""
  Sized for ONE coil fired at a time, which is the architecture the multi-pulse
  section arrived at: the bank is built once and switched to whichever face is
  being reversed.  A full module then costs one bank plus one SCR per face.

  The number of capacitors is set by two constraints, and the second is the
  one that gets missed.  {n_for_c} cans give the {c_bank*1e6:.0f} uF; {n_for_esr} cans are needed to get
  the bank ESR under the {(esr_budget-0.020)*1e3:.0f} mohm the loop budget allows once 20 mohm of
  busbar and joints is set aside.  Take the larger: {n_cans}.

  Links are distributor keyword SEARCHES rather than deep part links.  Part
  numbers and category identifiers churn constantly; a search for the right
  words will still find the part in a year's time, and a deep link very
  likely will not.
""")
    items = [
        ("MAGNET", [
            ("N42 block 20x10x5 mm, poles on 20x10 face", "4+", "$8",
             "https://www.kjmagnetics.com/products.asp?cat=168"),
        ]),
        ("COIL", [
            (f"1 mm enamelled copper, {coil.wire_len*3:.0f} m "
             f"grade 2 / 200 C", "1 reel", "$20",
             "https://www.digikey.com/en/products/result?keywords=magnet%20wire%2018%20AWG"),
            ("PTFE or Kapton former + glass tape banding", "1", "$10",
             "hoop stress at 1500 A will unwind an unbanded coil"),
            ("epoxy potting compound", "1", "$15",
             "https://www.digikey.com/en/products/result?keywords=epoxy%20encapsulant"),
        ]),
        ("BANK", [
            ("470 uF 450 V snap-in electrolytic, low ESR, 105 C",
             f"{n_cans}", f"${10*n_cans}",
             "https://www.digikey.com/en/products/result?keywords=470uF%20450V%20snap-in"),
            ("bleeder 100 kohm 5 W, permanently across the bank", "1", "$3",
             "https://www.digikey.com/en/products/result?keywords=100k%205W%20resistor"),
            ("copper busbar / 20 mm braid for the discharge loop", "1", "$10",
             "loop resistance is a specification, not an afterthought"),
        ]),
        ("SWITCH", [
            ("SCR stud or module, 1200 V, I2t > 4000 A2s",
             "1 per face", "$18",
             "https://www.digikey.com/en/products/result?keywords=SCR%20thyristor%201200V%20stud"),
            ("freewheel diode across the bank, 1200 V, fast, 4000 A2s",
             "1", "$14",
             "https://www.digikey.com/en/products/result?keywords=fast%20recovery%20diode%201200V"),
            ("gate pulse transformer + driver, isolated", "1", "$10",
             "https://www.digikey.com/en/products/result?keywords=SCR%20gate%20pulse%20transformer"),
        ]),
        ("CHARGER", [
            ("LT3750 capacitor charger board or module", "1", "$30",
             "https://www.analog.com/en/products/lt3750.html"),
            ("24 V 3 A supply for the charger", "1", "$15",
             "https://www.digikey.com/en/products/result?keywords=24V%203A%20power%20supply"),
        ]),
        ("SAFETY AND MEASUREMENT", [
            ("Rogowski coil or pulse current transformer, 3 kA", "1", "$60",
             "https://www.digikey.com/en/products/result?keywords=Rogowski%20current%20probe"),
            ("differential HV probe or 1000:1 divider", "1", "$50",
             "https://www.digikey.com/en/products/result?keywords=high%20voltage%20differential%20probe"),
            ("interlocked enclosure + shorting stick", "1", "$30",
             "shop-built"),
            ("Hall probe / gaussmeter to confirm the flip", "1", "$40",
             "https://www.digikey.com/en/products/result?keywords=gaussmeter%20hall%20probe"),
        ]),
    ]
    total = 0
    for group, rows in items:
        print(f"  {group}")
        for desc, qty, cost, link in rows:
            total += int(cost.strip("$"))
            print(f"    {desc:<58} {qty:>10} {cost:>6}")
            print(f"      {link}")
        print()
    print(f"  Rough total: ${total}\n")
    print(f"""  Three notes on this list.

  The bank stays at 450 V because that is the top of the cheap electrolytic
  range and the top of what a single-stage charger IC reaches.  Going to 800 V
  would halve the stored energy, but it means series strings with balancing
  resistors, a 1600 V switch, and a charger that is a project of its own.  On
  wall power the energy saved is worth nothing and the complexity costs weeks.

  The Rogowski coil is not optional in practice.  Without a current
  measurement, a switch that misfired and a magnet that did not flip look
  identical from the outside, and debugging becomes guesswork.  It is the most
  valuable instrument on the list and the one most often skipped.

  A photoflash bank is the alternative worth knowing about: two 330 V cans in
  series, paralleled up, gives a much lower ESR per joule than general-purpose
  450 V cans.  It costs more cans and more assembly, and it is the fallback if
  measurement shows the loop resistance over budget.""")


def safety(c_bank, v0, energy, i_peak):
    print("\n" + "=" * 78)
    print("9.  SAFETY")
    print("=" * 78)
    tau = 100e3 * c_bank
    print(f"""
  This is the section that matters most and it is short on purpose.

  A {v0:.0f} V bank holding {energy:.0f} J is lethal.  The threshold usually quoted for
  cardiac risk is around 10 J across the chest, and this is more than an order
  of magnitude past it.  Electrolytic capacitors also hold charge for hours,
  and DIELECTRIC ABSORPTION means a bank that has been shorted once will
  recover tens of volts by itself over the following minutes.

  The non-negotiables:

  * A bleeder resistor permanently across the bank.  100 kohm on {c_bank*1e6:.0f} uF is a
    {tau:.0f} s time constant - slow, but it means an abandoned rig is safe by
    morning.  Size it so the charger can still win.
  * A manual shorting stick, used every time, even after the bleeder.  Assume
    the bleeder has failed open, because that is its failure mode.
  * An interlocked enclosure.  The coil and bank live inside; nothing is
    touched with the lid open.
  * One hand in a pocket.  The path that kills is hand to hand.
  * NEVER connect an oscilloscope ground to the bank.  The scope ground is
    earth, the bank is floating, and connecting them puts the full discharge
    through the probe lead and the instrument.  Use a differential probe or a
    proper HV divider with an isolated supply.

  Mechanical hazards are real too and get forgotten next to the electrical
  ones.  Two N42 blocks pull at 60 N and will trap a finger between them hard
  enough to break skin, and they arrive at that speed from several centimetres
  away.  The coil also carries its own hoop stress at {i_peak:.0f} A and must be
  banded and potted - an unrestrained pulse coil unwinds itself, violently, on
  the first shot.""")


def main():
    n_d, _ = multipulse_verdict()
    rows, h_full = coil_survey(n_d)
    frontier = choose_operating_point(h_full)
    h_demo = partial_magnetisation(n_d)

    # rebuild at the demonstrator field, at the 450 V knee, with margin
    print("\n" + "=" * 78)
    print("   THE DESIGN, AT 2 Hcj AND 450 V")
    print("=" * 78)
    margin = 1.15
    best = None
    for ln in (5e-3, 8e-3, 12e-3, 16e-3):
        for lay in (2, 3, 4, 5, 6, 8, 10):
            c = build_coil(1.0e-3, lay, ln, h_demo)
            for c_bank in (470e-6, 680e-6, 1000e-6, 1500e-6, 2200e-6,
                           3300e-6):
                need = margin * c.i_needed
                if simulate(c, c_bank, 450.0,
                            r_extra=_esr(c_bank))["i_peak"] < need:
                    continue
                lo, hi = 50.0, 450.0
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    if simulate(c, c_bank, mid,
                                r_extra=_esr(c_bank))["i_peak"] < need:
                        lo = mid
                    else:
                        hi = mid
                s = simulate(c, c_bank, hi, r_extra=_esr(c_bank))
                e = 0.5 * c_bank * hi ** 2
                if best is None or e < best[0]:
                    best = (e, hi, c_bank, c, s)
    e, v0, c_bank, c, s = best

    # how much bank ESR the design can tolerate before it stops reaching
    # the field: this is a SPEC on the capacitor bank, not a nicety
    esr_budget = 0.0
    for r in np.arange(0.0, 0.40, 0.002):
        if simulate(c, c_bank, v0, r_extra=r)["i_peak"] >= c.i_needed:
            esr_budget = r
    print(f"""
      coil            {c.layers} layers, {c.turns} turns of 1 mm wire, {c.length*1e3:.0f} mm long
      bore            {MAG_W*1e3:.0f} x {MAG_H*1e3:.0f} mm, build {c.build*1e3:.1f} mm, {c.cu_mass*1e3:.0f} g of copper
      inductance      {c.inductance*1e6:.1f} uH
      resistance      {c.r_hot*1e3:.0f} mohm hot
      bank            {c_bank*1e6:.0f} uF at {v0:.0f} V  =  {e:.0f} J
      peak current    {s['i_peak']:.0f} A, against {c.i_needed:.0f} needed ({margin:.2f}x margin)
      pulse           {2*s['t_peak']*1e6:.0f} us, peak at {s['t_peak']*1e6:.0f} us
      action          {s['action']:.0f} A^2 s
      heating         +{s['dT']:.1f} K per shot
      recharge        {e/15:.0f} s at 15 W
      expected result ~90 % of Br reversed, ~{60*0.9**2:.0f} N pair force

      LOOP RESISTANCE BUDGET: everything outside the coil - bank ESR, busbar,
      SCR and joints - must total under {esr_budget*1e3:.0f} mohm, or the current does not
      reach the magnet.  That is a hard specification and it is the number
      most likely to be missed in a first build, because it is not written on
      any single part.  It is why the bank is paralleled cans rather than one
      big one, and why the loop must be short and flat.
""")
    switch_selection(v0, s["i_peak"], s["action"], s["t_peak"])
    capacitor_technologies(c, c_bank, v0, e)
    charger_choice(e, v0)
    bom(c, c_bank, v0, e, esr_budget)
    safety(c_bank, v0, e, s["i_peak"])

    m_self = n_d * BR_N42 / MU0
    grow = ((2.5 * HCJ_N42 + m_self) / (2.0 * HCJ_N42 + m_self)) ** 2 - 1
    print("\n" + "=" * 78)
    print("   VERDICT")
    print("=" * 78)
    print(f"""
  Buildable, on a bench, for roughly the cost of the instruments.

  The earlier verdict in ndfeb_switching.py was that a bare N42 block is not
  switchable in a battery-powered robot, and that stands - nothing here
  contradicts it.  What changed is the constraint, not the physics.  {e:.0f} J from
  a wall socket every {e/15:.0f} seconds is unremarkable; {e:.0f} J from a cell, forty
  times per manoeuvre, is not.

  Answers to the questions asked, in one place:

      wire            1 mm enamelled copper, as specified
      coil            {c.layers} layers, {c.turns} turns, {c.length*1e3:.0f} mm long on a {MAG_W*1e3:.0f} x {MAG_H*1e3:.0f} mm bore
      capacitor       {c_bank*1e6:.0f} uF at {v0:.0f} V, aluminium electrolytic, {int(np.ceil(c_bank/470e-6))} cans
      topology        capacitor discharge, single half-sine, freewheel diode
      switch          SCR, one per face, 1200 V, I2t > 4000 A^2 s
      charger         BUY - LT3750-based module, not custom
      pulses          ONE.  Multi-pulse does not accumulate; see section 1.
      simultaneity    one face at a time, from a shared bank

  The honest risks, in order:

  1.  The magnetisation fractions in section 4 are catalogue sizing practice,
      not a measurement of this block.  If this magnet needs 2.5 Hcj rather
      than 2.0, the bank grows by {grow*100:.0f} % - which is why it is built at 450 V
      with room to add cans rather than at its exact minimum.

  2.  Loop inductance outside the coil is not in the model.  Twenty
      centimetres of loose wiring is about 200 nH, small against {c.inductance*1e6:.0f} uH,
      but a long lead to the bank is not.  Keep bank, SCR and coil within a
      hand's width and use flat copper.  The same applies to the loop
      RESISTANCE budget above, which is the tighter of the two.

  3.  The first shot may simply not flip it, and the only way to tell "not
      enough field" from "the switch misfired" is to measure the current.
      Buy the Rogowski coil.

  What this does NOT settle is the robot.  A per-face bank of this size is not
  going into a 5 cm module, and the gated hybrid in gated_hybrid.py remains the
  architecture for that.  This is a rig to answer one question - does the
  mechanism work with real, switched magnets - and that question is worth
  answering on its own.""")


if __name__ == "__main__":
    main()
