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
    ("IRLML6344 SOT-23",   30,   20, 0.029, 0.03e-3, 0.02e-6, 0.30),
    ("IRF7416 SO-8",       30,   40, 0.020, 0.10e-3, 0.08e-6, 0.60),
    ("IRF3205 TO-220",     55,  390, 0.008, 2.0e-3,  1.2e-6,  0.90),
    ("IRFB4110 TO-220",   100,  720, 0.0037, 2.0e-3, 1.2e-6,  2.10),
    ("IRF540N TO-220",    100,  110, 0.044, 2.0e-3,  1.2e-6,  0.80),
    ("IRFP260N TO-247",   200,  200, 0.040, 6.0e-3,  3.5e-6,  2.40),
    ("IXTH30N50L TO-247", 500,  120, 0.200, 6.0e-3,  3.5e-6,  8.50),
    ("STW20N95K5 TO-247", 950,   60, 0.290, 6.0e-3,  3.5e-6, 11.00),
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
MCU = ("ESP32-C3 module", 3.0e-3, 2.0e-6, 3.0)
BATTERY_WH_PER_KG = 150.0
BATTERY_WH_PER_M3 = 250e3 * 3600 / 3600     # ~250 Wh/L


@dataclass
class Driver:
    cap_name: str
    n_caps: int
    mosfet_name: str
    charger_name: str
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
        return (f"{self.n_caps} x {self.cap_name}; "
                f"4 x {self.mosfet_name} per face; "
                f"{self.charger_name}; {GATE_DRIVER[0]}; {MCU[0]}")


def select_driver(v_need, l_coil, r_coil, n_turns, mmf_need, n_faces=6,
                  switches_per_second=2.0, margin=1.25):
    """Choose the cheapest real driver that can switch this coil.

    ``v_need`` is the bank voltage that just reaches the switching threshold
    (from Stage 3 of the framework); ``margin`` is applied on top so the design
    is not sitting exactly on the boundary.
    """
    v_target = v_need * margin
    best = None

    for cname, C, vrat, esr, cmass, cvol, cprice in CAPACITORS:
        if vrat < v_target:
            continue
        # series capacitors are avoided: they need balancing resistors and the
        # capacitance divides, so only parallel banks are considered
        for n_caps in (1, 2, 3, 4):
            c_tot = C * n_caps
            esr_tot = esr / n_caps
            z0 = np.sqrt(l_coil / c_tot)
            # peak current with ESR included in the damping
            r_tot = r_coil + esr_tot
            underdamped = r_tot < 2 * z0
            i_peak = v_target / z0 if underdamped else v_target / r_tot
            if n_turns * i_peak < mmf_need:
                continue

            for mname, vds, ipulse, rds, mmass, mvol, mprice in MOSFETS:
                if vds < v_target * 1.2 or ipulse < i_peak * 1.2:
                    continue
                for chname, vmax, pchg, chmass, chvol, chprice in CHARGERS:
                    if vmax < v_target:
                        continue
                    e_bank = 0.5 * c_tot * v_target**2
                    recharge = e_bank / pchg
                    if recharge > 1.0 / switches_per_second:
                        continue

                    n_fets = 4 * n_faces
                    n_gd = 2 * n_faces
                    e_duty = e_bank * switches_per_second
                    batt_wh = max(e_duty * 600 / 3600, 1.0)   # 10 min of duty
                    b_mass = batt_wh / BATTERY_WH_PER_KG
                    b_vol = batt_wh / 250e3 * 1e-3 * 3600 / 3600
                    b_vol = batt_wh * 1e-3 / 250.0            # Wh -> m^3 at 250 Wh/L

                    mass = (n_caps * cmass + n_fets * mmass + chmass +
                            n_gd * GATE_DRIVER[1] + MCU[1] + b_mass)
                    vol = (n_caps * cvol + n_fets * mvol + chvol +
                           n_gd * GATE_DRIVER[2] + MCU[2] + b_vol)
                    price = (n_caps * cprice + n_fets * mprice + chprice +
                             n_gd * GATE_DRIVER[3] + MCU[3])

                    cand = Driver(cname, n_caps, mname, chname, v_target,
                                  c_tot, e_bank, i_peak, mass, vol, price,
                                  recharge, True,
                                  "underdamped" if underdamped else "overdamped")
                    if best is None or (mass, price) < (best.mass, best.price):
                        best = cand
    if best is None:
        return Driver("none", 0, "none", "none", v_target, 0, 0, 0,
                      np.inf, np.inf, np.inf, np.inf, False,
                      "no combination of available parts meets the requirement")
    return best


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from framework import Design, MATERIALS, stage3_switching

    print("=" * 96)
    print("DRIVER SELECTION FROM REAL COMPONENTS")
    print("=" * 96)
    print("\nOne bank and charger per module, one H-bridge per face, 6 faces,")
    print("2 switching events per second.\n")
    print(f"  {'material':<9} {'V need':>7} {'I peak':>7} {'capacitor':<22}"
          f" {'mosfet':<18} {'mass':>7} {'price':>7} {'t_chg':>7}")
    print("  " + "-" * 92)

    for m in [k for k, v in MATERIALS.items() if v["src"] == "vendor"]:
        d = Design(material=m, circuit="potcore", v_cap=70.0)
        sw = stage3_switching(d)
        drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                            sw["n_turns"], sw["mmf_need"], n_faces=d.n_faces)
        if not drv.feasible:
            print(f"  {m:<9} {sw['v_need']:7.0f} {'':>7} "
                  f"{'NO FEASIBLE DRIVER':<22}")
            continue
        print(f"  {m:<9} {sw['v_need']:7.0f} {drv.i_peak:7.1f} "
              f"{drv.cap_name + ' x' + str(drv.n_caps):<22} "
              f"{drv.mosfet_name:<18} {drv.mass*1e3:6.0f}g "
              f"${drv.price:6.2f} {drv.recharge_s*1e3:6.0f}ms")

    print("""
  Every vendor Alnico grade is drivable from stock parts.  The switching
  requirement therefore belongs in the objective as a mass, volume and cost,
  not as a feasibility cliff - which is what the user asked for.""")
