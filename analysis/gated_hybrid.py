r"""The gated hybrid EPM: sizing, driver and bill of materials.

Why this architecture
---------------------
``ndfeb_switching.py`` rules out reversing an N42 block in place: it needs
about 47,000 ampere-turns and 40 to 60 joules, which is a bench magnetiser.
``epm_architectures.py`` shows that a face which can only be switched on and
off is nevertheless sufficient, because with N42-class magnets the next face
round the ring can pull a module up onto its neighbour with a five-fold energy
margin.

That combination points at one design.  Keep the N42 - it is what makes the
force, and it is never switched.  Put a small low-coercivity magnet beside it
in a shared steel circuit, and switch THAT.  With the two magnets parallel,
both drive flux out through the pole faces and the module holds.  With them
antiparallel, the flux circulates inside the device between the two magnets
and almost none reaches the outside world.

    parallel:      NdFeB -> pole -> [neighbour] -> pole -> Alnico -> yoke
    antiparallel:  NdFeB -> pole -> Alnico -> yoke, entirely internal

This is the arrangement Knaian described for electropermanent connectors, and
the reason it is so much cheaper to switch is that the Alnico sits in an
essentially closed magnetic circuit.  A magnet in a closed circuit has almost
no self-demagnetising field, so the ampere-turns needed are just H_sat times
its length, with none of the shape penalty that made the bare N42 block so
expensive.

What this file computes
-----------------------
The flux balance that sets the two magnet cross-sections, the switching
requirement for the Alnico leg, a coil that meets it, the capacitor discharge
that drives the coil, and a bill of materials with real parts.
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

from magnet_force import MU0  # noqa: E402

RHO_CU = 2.24e-8          # ohm m, hot
FILL = 0.65               # round wire in a rectangular window, realistic
B_SAT_STEEL = 1.9         # T, where 1018 stops helping

# ---- the two magnets ------------------------------------------------------
NDFEB = dict(name="N42", Br=1.32, Hcj=955e3, rho=7500)
# The switchable leg wants the HIGHEST remanence it can get, because its flux
# has to cancel the NdFeB's in the off state.  In a closed circuit its shape
# no longer matters, so the low-coercivity Alnico 5 grades - which are useless
# in an open circuit - become the right choice, and they are also the cheapest
# to switch.
SWITCH = dict(name="Alnico LNG52", Br=1.30, Hcj=57e3, H_sat=280e3, rho=7300)


@dataclass
class Design:
    a_pole: float          # area of ONE pole face, m^2
    b_pole: float          # target flux density at the pole, T
    l_switch: float        # length of the switchable leg, m
    depth: float           # available depth into the module, m

    @property
    def flux(self):
        return self.b_pole * self.a_pole

    @property
    def a_nd(self):
        """NdFeB cross-section: it carries the flux on its own in the off
        state, so it is sized on its own remanence."""
        return self.flux / NDFEB["Br"]

    @property
    def a_sw(self):
        """The switchable leg must be able to cancel the NdFeB exactly.

        Under-size it and the off state leaks; over-size it and the on state
        is no stronger but the switching costs more.  Matching the FLUX, not
        the area, is the condition.
        """
        return self.flux / SWITCH["Br"]

    @property
    def mmf_switch(self):
        """Ampere-turns to saturate the switchable leg in a closed circuit.

        A small allowance is added for the yoke and the joints; the circuit is
        closed but not perfect.
        """
        return SWITCH["H_sat"] * self.l_switch * 1.25


def size_coil(dsg, window_build, wire_d, n_turns=None):
    """A coil round the switchable leg, in the window available beside it."""
    r_core = np.sqrt(dsg.a_sw / np.pi)
    window = dsg.l_switch * window_build
    a_cu = window * FILL
    if n_turns is None:
        n_turns = int(a_cu / (np.pi * (wire_d / 2) ** 2))
    a_wire = np.pi * (wire_d / 2) ** 2
    l_turn = 2 * np.pi * (r_core + window_build / 2)
    wire_len = n_turns * l_turn
    r_coil = RHO_CU * wire_len / a_wire
    i_need = dsg.mmf_switch / n_turns
    # inductance: the closed steel circuit gives a high effective permeability
    # until the iron saturates, which is what makes the pulse slow rather than
    # what makes it hard
    mu_eff = 400.0
    l_path = 2 * dsg.l_switch + 2 * np.sqrt(dsg.a_pole)
    ind = mu_eff * MU0 * n_turns ** 2 * dsg.a_sw / l_path
    return dict(n_turns=n_turns, wire_d=wire_d, r_core=r_core,
                build=window_build, l_turn=l_turn, wire_len=wire_len,
                r_coil=r_coil, inductance=ind, i_need=i_need,
                j=i_need / a_wire / 1e6,
                cu_mass=wire_len * a_wire * 8960)


def drive(dsg, coil, v_bank):
    """What the pulse actually has to do to the magnetic circuit.

    Reversing a magnet in a closed steel circuit is a flux-swing problem, not
    a current problem, and the two constraints are separate:

    * the CURRENT has to reach the ampere-turns that push the Alnico past its
      saturating field - that is what decides whether it reverses at all;
    * the VOLT-SECONDS have to cover the flux swing, N * dPhi, because that is
      the integral of the back-emf the changing flux generates.

    A constant-inductance model gets this badly wrong.  The small-signal
    inductance of a coil on a closed steel path is enormous - tens of
    millihenries here - and predicts that the current takes milliseconds to
    rise and never gets there.  It does get there, because by the time the
    ampere-turns are anywhere near the switching threshold the steel is deeply
    saturated and the inductance has collapsed towards its air value.  That
    collapse is the mechanism, not an inconvenience.
    """
    d_phi = 2.0 * SWITCH["Br"] * dsg.a_sw          # full reversal
    n = coil["n_turns"]
    volt_seconds = n * d_phi
    t_pulse = volt_seconds / v_bank

    # inductance once the iron is saturated: the coil sees the magnet's own
    # recoil permeability and the leakage path, not the steel
    l_sat = coil["inductance"] * 4.0 / 400.0
    l_max = volt_seconds / coil["i_need"]          # must be below this
    i_reached = v_bank * t_pulse / l_sat

    e_field = 0.5 * l_sat * coil["i_need"] ** 2
    e_res = coil["i_need"] ** 2 * coil["r_coil"] * t_pulse
    e_hyst = 4.0 * SWITCH["Br"] * SWITCH["Hcj"] * dsg.a_sw * dsg.l_switch
    e_total = e_field + e_res + e_hyst
    a_wire = np.pi * (coil["wire_d"] / 2) ** 2
    dt_k = RHO_CU / (8960 * 385) * (coil["i_need"] / a_wire) ** 2 * t_pulse
    return dict(d_phi=d_phi, volt_seconds=volt_seconds, t_pulse=t_pulse,
                l_sat=l_sat, l_max=l_max, i_reached=i_reached,
                e_field=e_field, e_res=e_res, e_hyst=e_hyst, e_total=e_total,
                dT=dt_k, ok=l_sat <= l_max)


# --------------------------------------------------------------------------
BOM = [
    ("capacitor bank", "Nichicon UCY2C101MHD, 100 uF 160 V electrolytic",
     "one per module, shared by all faces", "~$1.5"),
    ("charger", "Analog Devices LT3750 flyback capacitor charger controller",
     "the fly-swatter oscillator done properly; camera-flash part",
     "~$6"),
    ("charger transformer", "Wurth 750310471 or LT3750 demo winding",
     "off-the-shelf flyback for the LT3750", "~$4"),
    ("bridge FETs", "Infineon IRFB4615PbF, 150 V 33 A TO-220",
     "4 shared, full bridge so the coil can be driven both ways", "~$1.5 ea"),
    ("face select FETs", "Infineon IPD070N10N3 G, 100 V 60 A DPAK",
     "2 per face; multiplexed so the bridge is shared", "~$1 ea"),
    ("gate drivers", "TI UCC27201A, 120 V half-bridge driver",
     "2 for the bridge, plus face select logic", "~$2 ea"),
    ("MCU", "Espressif ESP32-C3-MINI-1", "sequencing and radio", "~$3"),
    ("magnet, fixed", "NdFeB N42, sintered, Ni-Cu-Ni plated",
     "one per face, sized below", "~$1 ea"),
    ("magnet, switchable", "Alnico 5 (LNG52) rod, cast",
     "one per face, sized below", "~$2 ea"),
    ("yoke and poles", "AISI 1018 low-carbon steel, laser cut and stacked",
     "flux path; 1.9 T saturation", "~$1/face"),
    ("coil wire", "Elektrisola grade 2 enamelled copper, 0.315 mm",
     "about 6 m per face", "~$0.2/face"),
]


def main():
    # ---- what the face has to deliver ------------------------------------
    # matching the bench model: a 20 x 10 mm face, two poles side by side
    a_face = 20e-3 * 10e-3
    a_pole = a_face / 2 * 0.85          # two poles, minus the gap between
    dsg = Design(a_pole=a_pole, b_pole=0.95, l_switch=12e-3, depth=16e-3)

    print("=" * 78)
    print("GATED HYBRID EPM: SIZING AND DRIVER")
    print("=" * 78)
    d_round = 2 * np.sqrt(dsg.a_nd / np.pi) * 1e3
    d_sq = np.sqrt(dsg.a_nd) * 1e3
    print(f"""
  Face footprint 20 x 10 mm, split into two poles of
  {a_pole*1e6:.0f} mm2 each, driven to {dsg.b_pole:.2f} T.

  Flux per pole            {dsg.flux*1e6:8.1f} uWb
  NdFeB cross-section      {dsg.a_nd*1e6:8.1f} mm2   ({d_round:.1f} mm round, or {d_sq:.1f} mm square)
  Alnico cross-section     {dsg.a_sw*1e6:8.1f} mm2   (matched on FLUX, not area)
  Alnico length            {dsg.l_switch*1e3:8.1f} mm
""")
    f_pole = dsg.b_pole ** 2 * (2 * a_pole) / (2 * MU0)
    print(f"  Holding force at contact, both poles: "
          f"B^2 A / 2 mu0 = {f_pole:.0f} N")
    print(f"  (the bench model's bare N42 pair gives about 60 N, so this is "
          f"the same class)")

    print(f"\n  Ampere-turns to switch the Alnico leg: "
          f"{dsg.mmf_switch:,.0f}")
    print(f"  For comparison, reversing the bare N42 block in the same module")
    print(f"  needed 46,800 - a factor of {46843/dsg.mmf_switch:.0f} - because that magnet is in")
    print(f"  open circuit and this one is not.")

    # ---- coil ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  Coil options in a 2 mm build beside the Alnico leg")
    print("-" * 78)
    print(f"  {'wire':>6} {'turns':>7} {'R':>8} {'L':>9} {'I need':>8} "
          f"{'J':>9}  {'V for 30 A':>11}")
    print(f"  {'mm':>6} {'':7} {'ohm':>8} {'mH':>9} {'A':>8} {'A/mm2':>9}  "
          f"{'V':>11}")
    print("  " + "-" * 66)
    best = None
    for wd in (0.15e-3, 0.2e-3, 0.25e-3, 0.315e-3, 0.4e-3, 0.5e-3):
        c = size_coil(dsg, 2e-3, wd)
        print(f"  {wd*1e3:6.3f} {c['n_turns']:7d} {c['r_coil']:8.2f} "
              f"{c['inductance']*1e3:9.2f} {c['i_need']:8.1f} {c['j']:9.0f}  "
              f"{c['i_need']*c['r_coil']:11.0f}")
        if best is None and c["i_need"] < 40:
            best = c

    c = size_coil(dsg, 2e-3, 0.315e-3)
    print(f"""
  Take the 0.315 mm winding: {c['n_turns']} turns, {c['r_coil']:.2f} ohm, needs {c['i_need']:.0f} A.
  That is an ordinary coil - about {c['wire_len']:.0f} m of wire weighing {c['cu_mass']*1e3:.1f} g.""")

    # ---- driver ----------------------------------------------------------
    print("\n" + "-" * 78)
    print("  The pulse: a flux swing, not just a current")
    print("-" * 78)
    d0 = drive(dsg, c, 150.0)
    print(f"""
  Reversing the Alnico leg swings its flux by 2 Br A = {d0['d_phi']*1e6:.0f} uWb, and the
  coil has to supply N times that in volt-seconds: {d0['volt_seconds']*1e3:.1f} mV s.
""")
    print(f"  {'V bank':>8} {'pulse':>9} {'I needed':>9} {'L_sat':>9} "
          f"{'L limit':>9} {'dT':>7} {'energy':>9} {'ok?':>5}")
    print(f"  {'V':>8} {'us':>9} {'A':>9} {'mH':>9} {'mH':>9} {'K':>7} "
          f"{'mJ':>9}")
    print("  " + "-" * 70)
    chosen = None
    for v in (48, 100, 150, 200, 300):
        d = drive(dsg, c, v)
        print(f"  {v:8.0f} {d['t_pulse']*1e6:9.0f} {c['i_need']:9.1f} "
              f"{d['l_sat']*1e3:9.2f} {d['l_max']*1e3:9.2f} {d['dT']:7.1f} "
              f"{d['e_total']*1e3:9.0f} {'yes' if d['ok'] else 'NO':>5}")
        # 150 V is the engineering choice rather than the lowest-energy one.
        # Energy falls with voltage because the pulse gets shorter and the
        # resistive term is the only one that scales with time - but it
        # asymptotes to the field plus hysteresis floor, so there is little
        # left to win above 150 V, and 150 V is the last rung where ordinary
        # 200 V MOSFETs and electrolytics apply.
        if v == 150 and d["ok"]:
            chosen = (v, d)

    if chosen:
        v, d = chosen
        e_face = d["e_total"] * 1.5          # bank is not fully discharged
        print(f"""
  Where the energy goes at {v:.0f} V:

      magnetic field in the saturated circuit   {d['e_field']*1e3:6.0f} mJ
      resistive loss in the winding             {d['e_res']*1e3:6.0f} mJ
      hysteresis in the Alnico itself           {d['e_hyst']*1e3:6.0f} mJ
      ----------------------------------------------------
      delivered                                 {d['e_total']*1e3:6.0f} mJ
      bank, allowing for partial discharge      {e_face*1e3:6.0f} mJ

  Per module, eight faces: {e_face*8:.1f} J for a complete reconfiguration.
  A 3.7 V 1000 mAh cell holds 13.3 kJ, so about {13320/(e_face*8):,.0f} reconfigurations
  per charge - against {13320/506:.0f} for the bare-N42 scheme.  That factor of
  {(506/8)/e_face:.0f} is the whole argument for this architecture.

  The bank: {e_face*1e3:.0f} mJ at {v:.0f} V wants
  C = 2E/V^2 = {2*e_face/v**2*1e6:.0f} uF, so a {2*e_face/v**2*1e6*1.5:.0f} uF part with margin.  One bank
  serves all eight faces, pulsed in sequence.""")
        cap_uf = 2 * e_face / v ** 2 * 1e6 * 1.5
    else:
        v, cap_uf = 150.0, 100.0

    # ---- bill of materials ----------------------------------------------
    print("\n" + "=" * 78)
    print("  BILL OF MATERIALS, one module")
    print("=" * 78)
    print(f"\n  {'item':<20} {'part':<46} {'cost':>9}")
    print("  " + "-" * 76)
    for item, part, note, cost in BOM:
        print(f"  {item:<20} {part:<46} {cost:>9}")
        print(f"  {'':<20} {note}")
    print(f"""
  Costs are order-of-magnitude single-unit prices and every part number wants
  checking against a live distributor page before anything is ordered - the
  point of the list is that nothing here is exotic.  The one part that would
  surprise someone reading the fly-swatter idea is the LT3750: it is the same
  function as the swatter's oscillator, built for camera flash, and it will
  charge a {cap_uf:.0f} uF bank to {v:.0f} V in a few tens of milliseconds.""")

    print("""
  What is NOT in the list, and why:

    no high-voltage bridge   the bare-N42 scheme needed a kilovolt and two
                             thousand amps, which means IGBT modules and film
                             capacitors and creepage clearances.  At 150 V and
                             30 A this is an ordinary motor-driver problem.

    no per-face capacitor    one bank is shared; faces are pulsed in sequence
                             through the select FETs.  A pulse is under a
                             millisecond and a move takes a fraction of a
                             second, so sequencing costs nothing.""")


if __name__ == "__main__":
    main()
