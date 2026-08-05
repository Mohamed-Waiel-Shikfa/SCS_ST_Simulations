r"""Can an N42 magnet be switched in place by a coil wound around it?

The hypothesis under test
-------------------------
Drop the Alnico core and the steel cup.  Take the N42 block that is already
known to work mechanically - 20 x 10 x 5 mm, poles on the 20 x 10 faces, so
magnetised through the 5 mm axis - wrap a coil directly around it, and reverse
it in place.  The magnet is its own core.

This is an attractive idea because it keeps the force level that has already
been demonstrated on the bench.  Everything below is an attempt to find out
what it would take, from first principles, with no assumption either way.

The three numbers that decide it
--------------------------------
1.  **How much field is needed.**  Set by the material and by the shape,
    through the demagnetising factor.
2.  **How many ampere-turns that costs.**  Set by the field, the magnet
    length and how well the coil couples to it.
3.  **What those ampere-turns cost in energy.**  This is the one that decides
    the question, and it turns out to be almost independent of how the coil is
    wound.

Why the shape matters as much as the material
---------------------------------------------
A magnet in open circuit sits in its own demagnetising field, H_d = -N_d M.
For this block magnetised through its short axis N_d is large, and the
consequence cuts both ways during a reversal:

* at the START, the magnet is still forward-magnetised, its own demag field is
  already pushing it backwards, and the coil only has to make up the
  difference to reach the coercivity.  This part is easy.
* at the END, the magnet has reversed, so its demag field now points forward
  and OPPOSES further reversal.  The coil has to overcome the full saturating
  field plus the magnet's own demag field.

The second condition is the one that sizes the coil, and it is the reason a
partial reversal is cheap and a complete one is not.  A partially reversed
magnet is a weak magnet, so for an electropermanent magnet the expensive
condition is the one that counts.
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
from magnet_force import MU0, cuboid_pair_force  # noqa: E402

RHO_CU_20 = 1.68e-8            # ohm m
RHO_CU_100 = 2.24e-8           # ohm m, hot - a pulse coil does not stay cold
D_CU = 8960.0                  # kg/m^3
CP_CU = 385.0                  # J/(kg K)
INSULATION = 1.08              # grade 2 enamel build
LAYER_PITCH = 0.92             # layers nest into the grooves below
FILL = INSULATION * LAYER_PITCH


# --------------------------------------------------------------------------
# Materials.  H_sat is the field the manufacturer specifies to magnetise the
# grade to saturation, which is the honest figure for "fully reversed" - not
# Hcj, which is only the field at which the polarisation passes through zero.
# A magnet driven to exactly Hcj has no magnetisation at all; one driven a
# little past it is weakly and unpredictably magnetised.
# --------------------------------------------------------------------------
@dataclass
class Grade:
    name: str
    Br: float          # T
    Hcj: float         # A/m
    H_sat: float       # A/m, field to saturate
    mu_rec: float
    note: str = ""


GRADES = [
    Grade("NdFeB N42", 1.32, 955e3, 2400e3, 1.05,
          "the block on the bench"),
    Grade("SmCo5-16", 0.83, 1200e3, 3200e3, 1.05, ""),
    Grade("Ferrite Y30", 0.38, 195e3, 800e3, 1.10, ""),
    Grade("MnAlC", 0.55, 240e3, 900e3, 1.20, ""),
    Grade("Alnico LNGT44", 0.88, 122e3, 480e3, 2.0,
          "the grade the current pipeline picked"),
    Grade("Alnico LNG52", 1.30, 57e3, 280e3, 4.0,
          "highest remanence Alnico"),
    Grade("Alnico LNG37", 1.20, 49e3, 250e3, 4.0,
          "the grade in the measured data"),
]


def demag_factor(dims, axis=2):
    """Exact demagnetising factor of a cuboid along one axis.

    From the same closed-form kernel that is verified to machine precision
    against quadrature and against the analytic 1/3 for a cube.
    """
    half = np.asarray(dims, dtype=float) / 2.0
    e = np.zeros(3)
    e[axis] = 1.0
    H = fem3d.cuboid_field(half, np.zeros(3), np.eye(3), e, np.zeros((1, 3)))[0]
    return float(-H[axis])


def coil_coupling(l_coil, a, b):
    """Fraction of NI/l that reaches the centre of a finite coil.

    A solenoid carrying n ampere-turns per metre is magnetically identical to
    a uniformly magnetised body with M = n, so the field at its centre is
    (1 - N_d) n - the same demagnetising factor, used the other way round.
    For a long thin coil this tends to 1; for a pancake it tends to 0, which
    is exactly the regime a coil around a 5 mm thick magnet is in.
    """
    return 1.0 - demag_factor((a, b, l_coil), axis=2)


def reversal_field(g, n_d):
    """Coil field needed to reverse the magnet, at both ends of the process.

    Returns (H_start, H_end) in A/m, both as magnitudes.
    """
    m_s = g.Br / MU0
    h_start = abs(-g.Hcj + n_d * m_s)      # just reaching coercivity
    h_end = g.H_sat + n_d * m_s            # saturating the other way
    return h_start, h_end


# --------------------------------------------------------------------------
@dataclass
class Coil:
    wire_d: float
    n_turns: int
    n_layers: int
    turns_per_layer: int
    build: float
    length: float
    l_turn: float
    resistance: float
    inductance: float
    cu_area: float
    mass: float


def wind_around(a, b, l_coil, wire_d, n_layers):
    """Wind ``n_layers`` around a rectangular former of section a x b."""
    d_ins = wire_d * INSULATION
    tpl = max(int(l_coil / d_ins), 1)
    pitch = d_ins * LAYER_PITCH
    build = n_layers * pitch
    l_turn = 0.0
    for k in range(n_layers):
        t = (k + 0.5) * pitch
        l_turn += 2 * (a + 2 * t) + 2 * (b + 2 * t) - (8 - 2 * np.pi) * t
    l_turn /= n_layers
    n = n_layers * tpl
    a_w = np.pi * (wire_d / 2) ** 2
    wire_len = n * l_turn
    r = RHO_CU_100 * wire_len / a_w
    # inductance of a short thick coil, Wheeler-style with an equivalent radius
    r_eq = np.sqrt(a * b / np.pi) + build / 2
    ind = MU0 * n ** 2 * (np.pi * r_eq ** 2) / (l_coil + 0.9 * r_eq)
    return Coil(wire_d=wire_d, n_turns=n, n_layers=n_layers,
                turns_per_layer=tpl, build=build, length=l_coil,
                l_turn=l_turn, resistance=r, inductance=ind,
                cu_area=n * a_w, mass=wire_len * a_w * D_CU)


def drive_requirements(coil, ni_target, t_pulse=None):
    """What it takes to push ``ni_target`` ampere-turns through this coil."""
    i = ni_target / coil.n_turns
    j = i / (np.pi * (coil.wire_d / 2) ** 2) / 1e6      # A/mm^2
    v_res = i * coil.resistance                          # to hold it resistively
    # a capacitor discharge that just reaches this peak, critically damped:
    # C = L / (R/2)^2 puts the circuit on the boundary, and the peak current is
    # then about 0.37 V/Z0 with Z0 = R/2
    c_crit = 4.0 * coil.inductance / coil.resistance ** 2
    z0 = coil.resistance / 2.0
    v_cap = i * z0 / 0.37
    e_cap = 0.5 * c_crit * v_cap ** 2
    t_rise = np.pi / 2 * np.sqrt(coil.inductance * c_crit)
    t = t_pulse if t_pulse else t_rise
    # adiabatic temperature rise over the pulse
    dt_k = RHO_CU_100 / (D_CU * CP_CU) * (j * 1e6) ** 2 * t
    return dict(i=i, j=j, v_res=v_res, p_res=i * v_res, c_crit=c_crit,
                v_cap=v_cap, e_cap=e_cap, t_rise=t_rise, dT=dt_k,
                e_res=i ** 2 * coil.resistance * t)


def ampere_turns_from_voltage(coil, volts):
    """Steady ampere-turns a voltage source delivers into this winding."""
    return coil.n_turns * volts / coil.resistance


# --------------------------------------------------------------------------
def main():
    A, B, T = 20e-3, 10e-3, 5e-3        # the block on the bench
    n_d = demag_factor((A, B, T), axis=2)

    print("=" * 78)
    print("CAN AN N42 BLOCK BE SWITCHED IN PLACE?")
    print("=" * 78)
    print(f"\n  magnet 20 x 10 x 5 mm, poles on the 20 x 10 faces")
    print(f"  so it is magnetised through its SHORTEST axis, and")
    print(f"  demagnetising factor N_d = {n_d:.3f}\n")
    print(f"  {'if it were magnetised through':<32} {'N_d':>7}")
    print("  " + "-" * 42)
    for lbl, dims, ax in ((" 5 mm (as specified)", (A, B, T), 2),
                          ("10 mm", (A, T, B), 2),
                          ("20 mm", (B, T, A), 2)):
        print(f"  {lbl:<32} {demag_factor(dims, ax):7.3f}")
    print("""
  That first number is already the whole problem in miniature.  A block
  magnetised through its short axis is the worst case for self-demagnetisation,
  and N_d appears twice below: once making the magnet harder to saturate in
  reverse, and once making the coil around it inefficient.""")

    g42 = GRADES[0]
    m_s = g42.Br / MU0
    print(f"\n  N42 open circuit:  M_s = {m_s/1e6:.3f} MA/m,  "
          f"self-demag field {n_d*m_s/1e3:,.0f} kA/m")
    print(f"                     that is {n_d*m_s/g42.Hcj*100:.0f} % of its "
          f"own coercivity, sitting on the bench doing nothing")

    print("\n  Pull force of two of them, poles facing (rigid magnetisation):")
    for gap in (0.05e-3, 0.1e-3, 0.5e-3, 1e-3):
        f = abs(cuboid_pair_force(g42.Br, g42.Br, (A, B, T), (A, B, T),
                                  (0, 0, T + gap)))
        print(f"     {gap*1e3:4.2f} mm gap   {f:6.1f} N")

    # ---- what field is needed -------------------------------------------
    print("\n" + "=" * 78)
    print("1.  THE FIELD NEEDED TO REVERSE IT")
    print("=" * 78)
    print(f"\n  {'grade':<16} {'Hcj':>8} {'H_sat':>9} {'start':>9} "
          f"{'finish':>10}   {'NI no steel':>12} {'NI + steel':>11}")
    print(f"  {'':16} {'kA/m':>8} {'kA/m':>9} {'kA/m':>9} {'kA/m':>10}   "
          f"{'A-turns':>12} {'A-turns':>11}")
    print("  " + "-" * 88)
    coupling = coil_coupling(T, A, B)
    for g in GRADES:
        h0, h1 = reversal_field(g, n_d)
        ni_open = h1 * T / coupling
        ni_steel = g.H_sat * T
        print(f"  {g.name:<16} {g.Hcj/1e3:8.0f} {g.H_sat/1e3:9.0f} "
              f"{h0/1e3:9.0f} {h1/1e3:10.0f}   {ni_open:12,.0f} "
              f"{ni_steel:11,.0f}")

    print(f"""
  Read the N42 row across.  Starting the reversal needs only
  {reversal_field(g42, n_d)[0]/1e3:,.0f} kA/m, because the magnet's own demagnetising field is
  already doing most of the work.  FINISHING it needs
  {reversal_field(g42, n_d)[1]/1e3:,.0f} kA/m, because once the magnet has flipped, that same
  demagnetising field has flipped with it and now resists.

  The coil coupling makes it worse.  A coil {T*1e3:.0f} mm long around a
  {A*1e3:.0f} x {B*1e3:.0f} mm section is a pancake, not a solenoid: only
  {coupling*100:.0f} % of its ampere-turns per metre appears as field at the centre.
  That is a factor of {1/coupling:.1f} on top of everything else.""")

    ni42 = reversal_field(g42, n_d)[1] * T / coupling
    ni_alnico = reversal_field(GRADES[4], n_d)[1] * T / coupling
    print(f"""
  Against the Alnico grade the current pipeline settled on, switching this
  N42 block needs {ni42/ni_alnico:.1f} times the ampere-turns - and since resistive
  energy goes as the square of ampere-turns, {(ni42/ni_alnico)**2:.0f} times the energy.""")

    # ---- the gauge question ---------------------------------------------
    print("\n" + "=" * 78)
    print("2.  DOES THE WIRE GAUGE DECIDE ANYTHING?")
    print("=" * 78)
    print(f"""
  Wind the same window - {T*1e3:.0f} mm long, 1.0 mm build - with different wire and
  ask each one for {ni42:,.0f} ampere-turns:
""")
    print(f"  {'wire':>6} {'turns':>7} {'R':>9} {'L':>9} {'I needed':>10} "
          f"{'V needed':>10} {'peak power':>12} {'J':>11}")
    print(f"  {'mm':>6} {'':7} {'ohm':>9} {'uH':>9} {'A':>10} {'V':>10} "
          f"{'kW':>12} {'A/mm2':>11}")
    print("  " + "-" * 82)
    for wd in (0.05e-3, 0.1e-3, 0.2e-3, 0.4e-3, 0.8e-3, 1.6e-3):
        nl = max(int(1.0e-3 / (wd * FILL)), 1)
        c = wind_around(A, B, T, wd, nl)
        d = drive_requirements(c, ni42)
        print(f"  {wd*1e3:6.2f} {c.n_turns:7d} {c.resistance:9.2f} "
              f"{c.inductance*1e6:9.1f} {d['i']:10.1f} {d['v_res']:10.0f} "
              f"{d['p_res']/1e3:12.0f} {d['j']:11.0f}")

    p_fixed = None
    for wd in (0.1e-3, 0.8e-3):
        nl = max(int(1.0e-3 / (wd * FILL)), 1)
        c = wind_around(A, B, T, wd, nl)
        d = drive_requirements(c, ni42)
        p_fixed = d["p_res"]
    print(f"""
  The peak power column barely moves.  That is not a coincidence and it is the
  single most useful fact in this whole analysis:

      P = (NI)^2 * rho * l_turn / (k * A_window)

  Ampere-turns and the window geometry set the power.  The gauge only chooses
  how that power is split between volts and amps - thin wire gives a
  high-voltage low-current version of exactly the same problem, thick wire a
  low-voltage high-current one.  **You cannot wind your way out of an
  ampere-turns requirement.**""")

    # ---- why 0.1 mm failed on Alnico ------------------------------------
    print("\n" + "=" * 78)
    print("3.  WHY 0.1 mm WIRE ON ALNICO DID NOT WORK")
    print("=" * 78)
    print("""
  Same question the other way round: what does a given supply voltage actually
  deliver?  Substituting R = rho N l_turn / a_wire and N a_wire = k A_window,

      NI = V * k * A_window / (N * rho * l_turn)

  Ampere-turns are inversely proportional to the NUMBER OF TURNS at fixed
  voltage.  Adding turns of thin wire adds resistance faster than it adds
  turns.  Below, an Alnico LNG37 rod 4.75 mm across and 12.5 mm long, wound
  full, driven from a bench supply:
""")
    a_rod = b_rod = 4.75e-3
    l_rod = 12.5e-3
    g37 = GRADES[6]
    n_d_rod = demag_factor((a_rod, b_rod, l_rod), axis=2)
    cpl_rod = coil_coupling(l_rod, a_rod, b_rod)
    ni_need = reversal_field(g37, n_d_rod)[1] * l_rod / cpl_rod
    print(f"    rod demag factor {n_d_rod:.3f}, coil coupling {cpl_rod:.3f}")
    print(f"    ampere-turns needed to reverse it: {ni_need:,.0f}\n")
    print(f"  {'wire':>6} {'turns':>7} {'R':>8}   {'NI at 12 V':>11} "
          f"{'at 48 V':>9} {'at 400 V':>9}   {'V for full reversal':>20}")
    print("  " + "-" * 84)
    for wd in (0.1e-3, 0.2e-3, 0.4e-3, 0.8e-3):
        nl = max(int(1.5e-3 / (wd * FILL)), 1)
        c = wind_around(a_rod, b_rod, l_rod, wd, nl)
        v_need = ni_need * c.resistance / c.n_turns
        print(f"  {wd*1e3:6.2f} {c.n_turns:7d} {c.resistance:8.2f}   "
              f"{ampere_turns_from_voltage(c, 12):11,.0f} "
              f"{ampere_turns_from_voltage(c, 48):9,.0f} "
              f"{ampere_turns_from_voltage(c, 400):9,.0f}   "
              f"{v_need:17,.0f} V")
    print("""
  So 0.1 mm wire was never the problem, and nor was the number of layers.  A
  bench supply at 12 or 24 V into that winding produces a few hundred
  ampere-turns where several thousand are needed - short by more than an order
  of magnitude, which is why nothing happened at all rather than something
  partial.  The same coil driven from a few hundred volts would have worked.

  This is worth being precise about, because "thinner wire, more turns" is the
  intuitive fix and it is the wrong direction: it RAISES the voltage needed.
  What a low-voltage supply wants is FEWER turns of THICKER wire and a lot of
  current.  What actually solves it is abandoning the steady supply entirely
  and discharging a capacitor, which is what the next section is about.""")

    # ---- energy ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("4.  WHAT IT COSTS IN ENERGY")
    print("=" * 78)
    print("""
  Nothing above has been decisive yet: high volts and high amps are both
  survivable for a few tens of microseconds.  Energy is what decides it,
  because energy has to be stored in a capacitor that has to fit in the robot
  and be recharged from a battery the robot has to carry.

  There is a floor that no coil design can get under.  To reverse the magnet
  the field H has to exist in at least the magnet's own volume, and

      E_field = 1/2 * mu0 * H^2 * V

  is the energy in that field whatever produces it.  Real coils also spill
  field outside the magnet and lose energy resistively on the way, so the
  floor is optimistic by a factor of a few.
""")
    print(f"  {'grade':<16} {'H needed':>10} {'floor, magnet':>14} "
          f"{'x3 realistic':>14} {'per 8-face module':>18}")
    print(f"  {'':16} {'kA/m':>10} {'volume, J':>14} {'J':>14} {'J':>18}")
    print("  " + "-" * 78)
    v_mag = A * B * T
    for g in (GRADES[0], GRADES[4], GRADES[6]):
        h = reversal_field(g, n_d)[1]
        e = 0.5 * MU0 * h ** 2 * v_mag
        print(f"  {g.name:<16} {h/1e3:10,.0f} {e:14.2f} {3*e:14.2f} "
              f"{3*e*8:18.1f}")

    print("""
  A capacitor discharge, solved properly rather than estimated.  The coil is
  0.4 mm wire, 22 turns in a 1 mm build - the entry from the table above that
  needs the least extreme voltage and current together:
""")
    c = wind_around(A, B, T, 0.4e-3, max(int(1.0e-3 / (0.4e-3 * FILL)), 1))
    print(f"    {c.n_turns} turns, R = {c.resistance*1e3:.0f} mohm, "
          f"L = {c.inductance*1e6:.1f} uH, needs "
          f"{ni42/c.n_turns:,.0f} A")
    print(f"\n  {'C':>8} {'V needed':>10} {'stored':>9} {'I peak':>9} "
          f"{'t to peak':>11} {'dT rise':>9}")
    print(f"  {'uF':>8} {'V':>10} {'J':>9} {'A':>9} {'us':>11} {'K':>9}")
    print("  " + "-" * 62)
    i_need = ni42 / c.n_turns
    for cap in (22e-6, 47e-6, 100e-6, 220e-6, 470e-6):
        v, ipk, tpk, act = _rlc_for_peak(c.resistance, c.inductance, cap,
                                         i_need)
        j = i_need / (np.pi * (c.wire_d / 2) ** 2)
        dt = RHO_CU_100 / (D_CU * CP_CU) * act / (np.pi *
                                                  (c.wire_d / 2) ** 2) ** 2
        print(f"  {cap*1e6:8.0f} {v:10,.0f} {0.5*cap*v*v:9.1f} {ipk:9,.0f} "
              f"{tpk*1e6:11.0f} {dt:9.0f}")

    print("""
  Tens of joules per switch, at a kilovolt and a couple of thousand amps.  For
  comparison, that is the same class of pulse a bench magnetiser delivers -
  commercial NdFeB magnetisers are 1 to 3 kJ machines running 1 to 3 kV into
  fixtures drawing 5 to 30 kA, and they are the size of a filing cabinet.
  That is not a coincidence: this IS magnetising an NdFeB magnet, and the
  physics does not care that the fixture happens to be inside a robot.""")

    # ---- the fly swatter -------------------------------------------------
    print("\n" + "=" * 78)
    print("5.  THE FLY SWATTER")
    print("=" * 78)
    e_needed = 0.5 * 100e-6 * _rlc_for_peak(c.resistance, c.inductance,
                                            100e-6, i_need)[0] ** 2
    print(f"""
  The instinct is a good one and the topology is right: an electric fly swatter
  is a battery, an oscillator, a small step-up transformer and a voltage
  multiplier, which is exactly the architecture of a capacitor-discharge
  magnetiser.  It is the ENERGY, not the voltage, that does not carry over.

  A swatter charges a small capacitor - tens of nanofarads at one to three
  kilovolts - because all it has to do is break down an air gap.  Its stored
  energy is:
""")
    print(f"  {'swatter-class bank':<26} {'energy':>10}   "
          f"{'switches of this magnet':>24}")
    print("  " + "-" * 66)
    for cap, v in ((10e-9, 2000), (100e-9, 2000), (1e-6, 2000), (1e-6, 3000)):
        e = 0.5 * cap * v * v
        print(f"  {cap*1e9:8.0f} nF at {v:5.0f} V     {e*1e3:8.1f} mJ   "
              f"{e/e_needed:24.4f}")
    print(f"""
  Four to six orders of magnitude short.  A swatter also has a deliberately
  high source impedance - it is designed to make a spark, not to push amps
  into an inductor - so even its small charge cannot be delivered fast enough
  to matter.

  What DOES carry over is the silicon.  The same job done properly is a
  flyback capacitor-charger controller, which is the grown-up version of the
  swatter's oscillator: an LT3750 or LT3751 charging a real bank through a
  real transformer.  Those parts are used in camera flash and defibrillators,
  and a defibrillator is the right mental model for the energy scale here -
  200 J at 2 kV, hand-held, but the capacitor alone is the size of a deck of
  cards and it takes seconds to recharge.

  Eight faces at {e_needed:.0f} J is {e_needed*8:.0f} J per reconfiguration.  A 3.7 V 1000 mAh
  cell holds {3.7*3.6*1000/1000:.0f} kJ, so about {3.7*3.6*1000/(e_needed*8):.0f} full reconfigurations per charge,
  before any of it is spent on moving.""")


def _rlc_for_peak(r, l, c, i_target):
    """Solve the series RLC discharge for the voltage giving ``i_target``.

    The response is linear in the initial voltage, so one integration at 1 V
    scales to any target.  Returns (V, I_peak, t_peak, action) where action is
    the integral of i^2 dt that sets the temperature rise.
    """
    alpha = r / (2 * l)
    w0 = 1.0 / np.sqrt(l * c)
    t_end = 6.0 / alpha if alpha > 0 else 10 / w0
    t_end = min(t_end, 20 * np.pi / w0)
    t = np.linspace(0, t_end, 20000)
    if alpha < w0:
        wd = np.sqrt(w0 ** 2 - alpha ** 2)
        i_unit = np.exp(-alpha * t) * np.sin(wd * t) / (wd * l)
    else:
        b = np.sqrt(alpha ** 2 - w0 ** 2)
        i_unit = np.exp(-alpha * t) * np.sinh(b * t) / (b * l)
    k = int(np.argmax(i_unit))
    scale = i_target / i_unit[k]
    i = i_unit * scale
    # only the first half-cycle matters: after that a real driver has
    # crowbarred or the diode has taken over
    stop = k * 3 if k * 3 < len(t) else len(t) - 1
    action = float(np.trapz(i[:stop] ** 2, t[:stop]))
    return scale, float(i[k]), float(t[k]), action


if __name__ == "__main__":
    main()

