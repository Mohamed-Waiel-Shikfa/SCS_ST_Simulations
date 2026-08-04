"""Stage 2: the capacitor-discharge driver, built from real components.

Switching energy is not a hard constraint.  A high-coercivity magnet is fine
provided a driver can actually be built for it, so this stage selects real
parts and returns what they cost in mass, volume and money.  Those numbers then
feed the module geometry in Stage 3 rather than an arbitrary voltage ceiling
being imposed in Stage 1.

Topology
--------
Each EPM face needs bidirectional current, so the coil sits in a full H-bridge
across a capacitor bank.  The bank is charged from the module battery by a
flyback capacitor-charger controller - the same part family used for camera
flash, which is exactly this duty cycle.  One bank and one charger are shared
by all faces of a module; the H-bridge is per face, since faces must be
switched independently.

    battery --> charger --> C_bank --> H-bridge (x n_faces) --> coil

What is checked
---------------
* the bank stores enough energy, with margin, to deliver the required
  ampere-turns
* the capacitor can actually source the peak current (ESR limits it, and a
  large electrolytic is not automatically a good pulse source)
* the MOSFETs survive the bank voltage and the peak current within their
  pulsed rating
* the charger can recharge between switching events fast enough for locomotion

COMPONENT DATA are typical catalogue values for widely available parts, used to
establish feasibility and rough mass/volume budgets.  Every part here is real
and orderable, but the exact figures must be confirmed against a datasheet
before anything is purchased.  They are deliberately conservative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Capacitors.  Aluminium electrolytic "photoflash" types are purpose-built for
# repeated high-current pulse discharge; film types have far lower ESR per unit
# capacitance but are bulkier.
#   name, C (F), V rating, ESR (ohm), mass (kg), volume (m^3), price (USD)
# --------------------------------------------------------------------------
CAPACITORS = [
    ("elec 100u/63V",   100e-6,  63, 0.35, 6.0e-3, 2.5e-6, 0.5),
    ("elec 220u/63V",   220e-6,  63, 0.22, 9.0e-3, 4.0e-6, 0.7),
    ("elec 470u/63V",   470e-6,  63, 0.14, 14e-3,  7.0e-6, 1.1),
    ("elec 100u/160V",  100e-6, 160, 0.30, 12e-3,  5.5e-6, 1.4),
    ("elec 220u/160V",  220e-6, 160, 0.19, 20e-3,  9.5e-6, 2.2),
    ("elec 100u/250V",  100e-6, 250, 0.28, 22e-3,  11e-6,  2.8),
    ("elec 220u/250V",  220e-6, 250, 0.18, 34e-3,  18e-6,  4.5),
    ("photoflash 80u/330V", 80e-6, 330, 0.15, 30e-3, 16e-6, 5.0),
    ("photoflash 160u/330V", 160e-6, 330, 0.10, 48e-3, 27e-6, 7.5),
    ("film 10u/250V",    10e-6, 250, 0.010, 15e-3, 9.0e-6, 3.2),
    ("film 22u/250V",    22e-6, 250, 0.008, 26e-3, 16e-6,  5.4),
    ("film 10u/400V",    10e-6, 400, 0.012, 24e-3, 15e-6,  4.6),
]

# --------------------------------------------------------------------------
# MOSFETs for the H-bridge.  I_pulse is the datasheet pulsed drain current,
# typically about 4x the continuous rating for the short pulses used here.
#   name, V_DS, I_pulse (A), R_DS_on (ohm), mass (kg), volume (m^3), price
# --------------------------------------------------------------------------
MOSFETS = [
    # surface mount - the only realistic option at high face counts
    ("IRLML6344 SOT-23",   30,   20, 0.029, 0.03e-3, 0.02e-6, 0.30),
    ("IRF7416 SO-8",       30,   40, 0.020, 0.10e-3, 0.08e-6, 0.60),
    ("BSC0902NS PG-TDSON", 30,  400, 0.0009, 0.05e-3, 0.03e-6, 1.40),
    ("SQJ850EP PowerPAK",  60,  240, 0.0075, 0.12e-3, 0.10e-6, 1.50),
    ("BSC160N10 PG-TDSON",100,  180, 0.016, 0.05e-3, 0.03e-6, 1.20),
    ("IPB65R110 D2PAK",   650,   72, 0.110, 0.90e-3, 0.55e-6, 3.60),
    ("STD5N52U DPAK",     525,   16, 0.950, 0.30e-3, 0.20e-6, 1.10),
    # through hole
    ("IRF3205 TO-220",     55,  390, 0.008, 2.0e-3,  1.2e-6,  0.90),
    ("IRFB4110 TO-220",   100,  720, 0.0037, 2.0e-3, 1.2e-6,  2.10),
    ("IRFP260N TO-247",   200,  200, 0.040, 6.0e-3,  3.5e-6,  2.40),
    ("IXTH30N50L TO-247", 500,  120, 0.200, 6.0e-3,  3.5e-6,  8.50),
]

# Capacitor-charger controllers (flyback, current-mode).  P_charge is the
# realistic average charging power into the bank.
#   name, V_out_max, P_charge (W), mass, volume, price
CHARGERS = [
    ("LT3750 + xfmr",   500, 6.0, 4.0e-3, 2.2e-6, 9.0),
    ("LT3751 + xfmr",  1000, 20.0, 7.0e-3, 4.0e-6, 14.0),
    ("MC34063 boost",    60, 2.0, 1.5e-3, 0.9e-6, 1.2),
]

GATE_DRIVER = ("IR2110 half-bridge", 2.0e-3, 1.0e-6, 2.5)   # per half-bridge
GATE_DRIVER_SMD = ("UCC27201 SO-8", 0.10e-3, 0.09e-6, 1.8)
MCU = ("ESP32-C3 module", 3.0e-3, 2.0e-6, 3.0)
BATTERY_WH_PER_KG = 150.0
BATTERY_WH_PER_L = 250.0
SWITCH_EVENTS = 200          # switching events per charge, per face

# Topology.  A full H-bridge on every face needs 4 devices per face, which for
# an 18- to 42-face module is 72 to 168 transistors - physically impossible in
# through-hole packages inside a 5 cm module.  A real design multiplexes: one
# shared H-bridge drives the bank, and each face has a pair of series select
# switches.  That trades the ability to switch faces simultaneously (they must
# be pulsed in sequence) for a large saving in device count and volume.
TOPOLOGIES = {
    "bridge_per_face": dict(fets_per_face=4, shared_fets=0, gd_per_face=2,
                            simultaneous=True),
    "multiplexed": dict(fets_per_face=2, shared_fets=4, gd_per_face=1,
                        simultaneous=False),
}


@dataclass
class Driver:
    cap_name: str
    n_caps: int
    mosfet_name: str
    charger_name: str
    topology: str
    n_fets: int
    v_bank: float
    c_bank: float
    e_bank: float
    i_peak: float
    mass: float
    volume: float
    price: float
    recharge_s: float
    feasible: bool
    notes: str

    def bom(self):
        return (f"{self.n_caps} x {self.cap_name}; {self.n_fets} x "
                f"{self.mosfet_name} ({self.topology}); {self.charger_name}; "
                f"{MCU[0]}")


def select_driver(v_need, l_coil, r_coil, n_turns, mmf_need, n_faces=18,
                  switches_per_second=2.0, margin=1.25, v_max=400.0):
    """Choose the lightest real driver that can switch this coil.

    ``v_need`` is the bank voltage that just reaches the switching threshold
    (from Stage 3 of the framework); ``margin`` is applied on top so the design
    is not sitting exactly on the boundary.
    """
    v_target = max(v_need * margin, 5.0)
    if not np.isfinite(v_target) or v_target > v_max:
        return _no_driver(v_target, f"required {v_target:.0f} V is out of range")

    best = None
    for cname, C, vrat, esr, cmass, cvol, cprice in CAPACITORS:
        if vrat < v_target:
            continue
        for n_caps in (1, 2, 3, 4):
            c_tot = C * n_caps
            esr_tot = esr / n_caps
            z0 = np.sqrt(l_coil / c_tot) if l_coil > 0 else 0.0
            r_tot = r_coil + esr_tot
            underdamped = r_tot < 2 * z0
            # A low-inductance coil does NOT draw unbounded current: the loop
            # resistance (winding plus capacitor ESR) always limits it.  Taking
            # the LC impedance alone gave hundreds of kiloamps for short fat
            # coils and rejected every real MOSFET.
            i_peak = min(v_target / z0 if z0 > 0 else np.inf,
                         v_target / max(r_tot, 1e-6))
            if n_turns * i_peak < mmf_need:
                continue

            e_bank = 0.5 * c_tot * v_target**2
            recharge_budget = 1.0 / switches_per_second

            for tname, topo in TOPOLOGIES.items():
                n_fets = topo["fets_per_face"] * n_faces + topo["shared_fets"]
                n_gd = topo["gd_per_face"] * n_faces
                for mname, vds, ipulse, rds, mmass, mvol, mprice in MOSFETS:
                    if vds < v_target * 1.2 or ipulse < i_peak * 1.2:
                        continue
                    gd = (GATE_DRIVER_SMD if "TO-2" not in mname
                          else GATE_DRIVER)
                    for chname, vmax, pchg, chmass, chvol, chprice in CHARGERS:
                        if vmax < v_target:
                            continue
                        recharge = e_bank / pchg
                        if recharge > recharge_budget:
                            continue

                        # battery sized for a finite number of moves, not for
                        # continuous switching
                        batt_wh = e_bank * SWITCH_EVENTS * n_faces / 3600.0
                        batt_wh = max(batt_wh, 0.5)
                        b_mass = batt_wh / BATTERY_WH_PER_KG
                        b_vol = batt_wh / BATTERY_WH_PER_L * 1e-3

                        mass = (n_caps * cmass + n_fets * mmass + chmass +
                                n_gd * gd[1] + MCU[1] + b_mass)
                        vol = (n_caps * cvol + n_fets * mvol + chvol +
                               n_gd * gd[2] + MCU[2] + b_vol)
                        price = (n_caps * cprice + n_fets * mprice + chprice +
                                 n_gd * gd[3] + MCU[3])

                        cand = Driver(cname, n_caps, mname, chname, tname,
                                      n_fets, v_target, c_tot, e_bank, i_peak,
                                      mass, vol, price, recharge, True,
                                      "underdamped" if underdamped
                                      else "overdamped")
                        if best is None or cand.volume < best.volume:
                            best = cand
    if best is None:
        return _no_driver(v_target, "no combination of available parts meets "
                                    "the requirement")
    return best


def _no_driver(v, why):
    return Driver("none", 0, "none", "none", "none", 0, v, 0, 0, 0,
                  np.inf, np.inf, np.inf, np.inf, False, why)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from framework import Design, MATERIALS, stage3_switching

    print("=" * 100)
    print("DRIVER SELECTION FROM REAL COMPONENTS")
    print("=" * 100)
    print("\n18-face module (n=8 rhombicuboctahedron), 2 switching events per "
          "second.\n")
    print(f"  {'material':<9} {'V need':>7} {'I peak':>7} {'capacitor':<20}"
          f" {'mosfet':<20} {'topology':<16} {'#fet':>5} {'mass':>7}"
          f" {'vol':>8} {'price':>7}")
    print("  " + "-" * 116)

    for m in [k for k, v in MATERIALS.items() if v["src"] == "vendor"]:
        d = Design(material=m, circuit="potcore", v_cap=70.0, n_gon=8)
        sw = stage3_switching(d)
        drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                            sw["n_turns"], sw["mmf_need"], n_faces=d.n_faces)
        if not drv.feasible:
            print(f"  {m:<9} {sw['v_need']:7.0f}   NO FEASIBLE DRIVER: "
                  f"{drv.notes}")
            continue
        print(f"  {m:<9} {sw['v_need']:7.0f} {drv.i_peak:7.1f} "
              f"{drv.cap_name + ' x' + str(drv.n_caps):<20} "
              f"{drv.mosfet_name:<20} {drv.topology:<16} {drv.n_fets:5d} "
              f"{drv.mass*1e3:6.0f}g {drv.volume*1e6:7.1f}cc "
              f"${drv.price:6.2f}")

    print("""
  The topology column is the finding.  A full H-bridge on every face needs 4
  transistors per face - 72 of them on an 18-face module, 168 on a 42-face one
  - which is physically impossible inside a 5 cm shell in through-hole
  packages.  The multiplexed topology shares one bridge and gives each face a
  pair of series select switches, cutting device count roughly in half and
  volume by far more once surface-mount parts are used.

  The cost is that faces can no longer be pulsed simultaneously; they must be
  switched in sequence.  Since a pulse is tens of microseconds and a
  reconfiguration move is a fraction of a second, that is an acceptable trade -
  but it is a real constraint on any control scheme that assumes all faces
  change state at once.""")
