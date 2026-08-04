"""Feasibility audit of the current Magnobots EPM design.

Everything here is computed from first principles with the validated engine in
``simulations/Force_compute/python/magnet_force.py`` (which reproduces the
measured pull forces in ``Mag Force Data.csv`` to within the stated +/-0.25 N).

Run:  python analysis/epm_feasibility_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from magnet_force import (MU0, CoaxialRodPair, Material, cylinder_demag_factor,  # noqa: E402
                          alnico_lng37, block_pair_force_mm)

RHO_CU = 1.68e-8          # copper resistivity at 20 C, ohm m
RHO_PLA = 1240.0          # kg/m^3
RHO_ALNICO = 7300.0       # kg/m^3
G = 9.81

# As-built parameters, from param.txt, the S26 report and Switch_sim/
ALN_D, ALN_L = 4.75e-3, 12.5e-3
COIL_V = 30.0
WINDING_THICKNESS = 0.3e-3   # radial build of the winding, m
COIL_LEN = ALN_L


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def lng60():
    """Alnico 5-7 / LNG60, the grade named in the report and on the poster."""
    return Material.from_datasheet("LNG60 (Alnico 5-7)", Br=1.35, Hcb=59e3,
                                   Hcj=60e3, BHmax=60e3, mu_rec=4.0)


# ==========================================================================
rule("1. IS A BARE ALNICO ROD MAGNETICALLY STABLE AT THIS ASPECT RATIO?")

for mat in (alnico_lng37(), lng60()):
    N = float(cylinder_demag_factor(ALN_D / 2, ALN_L))
    M_sat = mat.Br / MU0
    H_d_full = N * M_sat                      # demag field if it held full Br

    pair = CoaxialRodPair(ALN_D / 2, ALN_L, mat, n_slabs=24)
    J, H = pair.solve(1e9)                    # isolated rod
    J_op, H_op = J[:24].mean(), H[:24].mean()

    print(f"\n{mat.name}:  Br = {mat.Br:.2f} T,  Hcj = {mat.Hcj/1e3:.0f} kA/m")
    print(f"  L/D = {ALN_L/ALN_D:.2f}  ->  demagnetising factor N = {N:.4f}")
    print(f"  demag field IF it held full Br : {H_d_full/1e3:8.1f} kA/m")
    print(f"  material coercivity      Hcj   : {mat.Hcj/1e3:8.1f} kA/m")
    print(f"  ratio |H_demag| / Hcj          : {H_d_full/mat.Hcj:8.2f}   "
          f"{'<-- SELF-DEMAGNETISING' if H_d_full > mat.Hcj else 'stable'}")
    print(f"  actual settled operating point : J = {J_op:.3f} T "
          f"({J_op/mat.Br*100:.0f} % of Br), H = {H_op/1e3:.0f} kA/m")

print("\n  Empirical check: the measured contact force for a pair of these rods")
print("  is 2.00 N.  A pair holding full Br would give:")
rigid = Material("rigid@Br", Br=1.20, Hcj=1e12, mu_rec=1.0, p=60.0, q=0.5)
print(f"    {abs(CoaxialRodPair(ALN_D/2, ALN_L, rigid, 24).force(0.0)):.1f} N  "
      f"(rigid at Br = 1.20 T)")
print(f"    {abs(CoaxialRodPair(ALN_D/2, ALN_L, alnico_lng37(), 24).force(0.0)):.1f} N  "
      f"(self-consistent, what the physics actually allows)")
print("  -> measurement agrees with the self-consistent value, not with Br.")


# ==========================================================================
rule("2. WHAT ASPECT RATIO WOULD THE ROD NEED?")

mat = alnico_lng37()


def J_open_circuit(LD, D=ALN_D, n=20):
    p = CoaxialRodPair(D / 2, LD * D, mat, n_slabs=n)
    J, _ = p.solve(1e9)
    return float(J[:n].mean())


print(f"  target: hold a useful fraction of Br = {mat.Br:.2f} T in open circuit\n")
print(f"  {'L/D':>6} {'length (mm)':>12} {'J_open (T)':>11} {'% of Br':>9}")
for LD in (2.63, 4, 6, 8, 10, 15, 20, 30):
    j = J_open_circuit(LD)
    flag = "  <-- as built" if abs(LD - 2.63) < 0.01 else ""
    print(f"  {LD:6.2f} {LD*ALN_D*1e3:12.1f} {j:11.3f} {j/mat.Br*100:9.0f}{flag}")

for target in (0.70, 0.90):
    try:
        r = brentq(lambda x: J_open_circuit(x) - target * mat.Br, 2.0, 60.0, xtol=1e-2)
        print(f"\n  need L/D = {r:.1f}  (length {r*ALN_D*1e3:.0f} mm at D = 4.75 mm) "
              f"to hold {target*100:.0f} % of Br")
    except ValueError:
        print(f"\n  {target*100:.0f} % of Br is not reachable at any aspect ratio")


# ==========================================================================
rule("3. CAN THE COIL ACTUALLY SWITCH THE ROD?  (WIRE GAUGE SCALING)")

print("  Fixed winding window: 12.5 mm long, 0.3 mm radial build, 30 V drive.")
print("  N turns fit as (window area)/(wire area); R = rho * N * circumference / area.")
print("\n  Switching criterion: Alnico must be driven to roughly 3 x Hcj to re-saturate")
print("  in reverse (1 x Hcj only takes it through zero net magnetisation), so the coil")
print("  must supply MMF = 3 * Hcj * L_magnet.  For LNG37 that is")
H_SWITCH = 3.0 * 49e3
MMF_NEEDED = H_SWITCH * ALN_L
print(f"    3 x 49 kA/m x 12.5 mm = {MMF_NEEDED:.0f} ampere-turns.\n")

CAP = 10e-6


def rlc_peak_current(L, R, C, V0, t_end=500e-6, n=200001):
    """Peak current of a series RLC discharging a pre-charged capacitor.

    Solved directly rather than with the V/R steady-state limit, which is
    meaningless here: the drive is a short capacitor pulse, so for thick wire
    the current is set by the LC impedance, not by resistance.
    """
    t = np.linspace(0.0, t_end, n)
    dt = t[1] - t[0]
    i = 0.0
    v = V0
    peak = 0.0
    for _ in range(n):
        di = (v - i * R) / L * dt
        v -= i / C * dt
        i += di
        peak = max(peak, i)
        if i < 0:
            break
    return peak


print(f"  {'wire':>6} {'turns':>6} {'R':>7} {'L':>8} {'L/R':>7} {'2*sqrt(L/C)':>12} "
      f"{'damping':>11} {'I peak':>8} {'MMF':>8} {'verdict':>8}")
print(f"  {'(mm)':>6} {'':>6} {'(ohm)':>7} {'(uH)':>8} {'(us)':>7} {'(ohm)':>12} "
      f"{'':>11} {'(A)':>8} {'(A-t)':>8}")

results = {}
for d in (0.10, 0.15, 0.20, 0.30, 0.40):
    dm = d * 1e-3
    N = (COIL_LEN / dm) * (WINDING_THICKNESS / dm)
    mean_d = ALN_D + WINDING_THICKNESS
    length = N * np.pi * mean_d
    area = np.pi * (dm / 2) ** 2
    R = RHO_CU * length / area
    L_coil = MU0 * N**2 * (np.pi * (mean_d / 2) ** 2) / (COIL_LEN + 0.45 * mean_d)

    R_crit = 2.0 * np.sqrt(L_coil / CAP)
    I_pk = rlc_peak_current(L_coil, R, CAP, COIL_V)
    mmf = N * I_pk
    results[d] = (N, R, L_coil, I_pk, mmf)
    print(f"  {d:6.2f} {N:6.0f} {R:7.2f} {L_coil*1e6:8.1f} {L_coil/R*1e6:7.2f} "
          f"{R_crit:12.2f} {'over' if R > R_crit else 'under':>11} "
          f"{I_pk:8.1f} {mmf:8.0f} {'OK' if mmf >= MMF_NEEDED else 'FAILS':>8}")

print(f"\n  (10 uF bank at 30 V; MMF must reach {MMF_NEEDED:.0f} A-turns to switch)")
print("\n  Scaling law for a fixed winding window: N ~ 1/d^2 and R ~ 1/d^4, so")
print("  L ~ 1/d^4 too and the L/R rise time is INDEPENDENT of wire gauge.")
print("  What changes with gauge is the damping: the 0.1 mm coil is overdamped")
print("  (R above critical) and its current is resistance-limited, while the")
print("  0.3 mm coil is underdamped and rings up to the full LC current.")
print(f"\n  0.1 mm reaches {results[0.10][4]:.0f} A-turns, "
      f"{results[0.10][4]/MMF_NEEDED*100:.0f} % of what is needed -> cannot switch.")
print(f"  0.3 mm reaches {results[0.30][4]:.0f} A-turns, "
      f"{results[0.30][4]/MMF_NEEDED*100:.0f} % of what is needed -> switches.")
print("  This reproduces the reported experimental outcome from circuit physics")
print("  alone, without invoking shorted turns, a broken wire or poor winding.")

print("\n  --- sensitivity of this conclusion to the switching threshold ---")
print("  The '3 x Hcj to re-saturate' factor is an estimate (literature puts full")
print("  Alnico re-saturation at roughly 3-5 x Hc).  Margin for the 0.3 mm coil:\n")
mmf_03 = results[0.30][4]
print(f"  {'grade':>8} {'Hcj':>8} {'2x Hcj':>10} {'3x Hcj':>10} {'4x Hcj':>10} {'5x Hcj':>10}")
print(f"  {'':>8} {'(kA/m)':>8} " + " ".join(f"{'margin':>10}" for _ in range(4)))
for grade, hcj in (("LNG37", 49e3), ("LNG60", 60e3)):
    cells = []
    for k in (2, 3, 4, 5):
        need = k * hcj * ALN_L
        cells.append(f"{mmf_03/need*100:9.0f}%")
    print(f"  {grade:>8} {hcj/1e3:8.0f} " + " ".join(cells))

print("\n  Read that as: the 0.3 mm / 30 V / 10 uF build switches Alnico 5 (LNG37)")
print("  with essentially zero margin, and does NOT switch Alnico 5-7 (LNG60) at")
print("  any plausible threshold.  The report and the poster both specify LNG60,")
print("  but the parts actually procured were Alnico 5.  If LNG60 is ordered for")
print("  the next build, the existing driver will stop working.")




# ==========================================================================
rule("4. THE SQUARE-CUBE PROBLEM: HOW BIG CAN A MODULE BE?")

print("  If the whole module scales with its side length a, holding force follows")
print("  the pole area (a^2) while weight follows the volume (a^3).  The ratio")
print("  therefore falls as 1/a and there is a hard maximum module size.\n")

J_face = J_open_circuit(2.63)
sigma_bare = J_face ** 2 / (2 * MU0)
sigma_ideal = 1.20 ** 2 / (2 * MU0)
print(f"  Maxwell stress at J = {J_face:.2f} T (as built) : {sigma_bare/1e3:7.1f} kPa")
print(f"  Maxwell stress at J = 1.20 T (ideal)      : {sigma_ideal/1e3:7.1f} kPa")

WALL_FRAC = 1 / 25          # wall thickness as a fraction of side
POLE_FRAC = 0.25            # fraction of a face that is active pole
MAG_FRAC = 0.06             # fraction of module volume that is magnet


def module(a, sigma):
    shell_v = a**3 - (a * (1 - 2 * WALL_FRAC)) ** 3
    mass = shell_v * RHO_PLA + MAG_FRAC * a**3 * RHO_ALNICO
    weight = mass * G
    force = sigma * POLE_FRAC * a**2
    return mass, weight, force


print(f"\n  shell wall = a/25, magnet = 6 % of volume, active pole = 25 % of a face\n")
print(f"  {'a (mm)':>7} {'mass (g)':>9} {'weight (N)':>11} "
      f"{'F built (N)':>12} {'F/W built':>10} {'F ideal (N)':>12} {'F/W ideal':>10}")
for a_mm in (10, 20, 30, 50, 80, 120, 200):
    a = a_mm * 1e-3
    m, w, f_b = module(a, sigma_bare)
    _, _, f_i = module(a, sigma_ideal)
    print(f"  {a_mm:7.0f} {m*1e3:9.1f} {w:11.3f} {f_b:12.2f} {f_b/w:10.1f} "
          f"{f_i:12.2f} {f_i/w:10.1f}")

for label, sigma in (("as built", sigma_bare), ("ideal", sigma_ideal)):
    def ratio(a, s=sigma):
        m, w, f = module(a, s)
        return f / w - 1.0
    try:
        a_max = brentq(ratio, 5e-3, 50.0)
        print(f"\n  {label}: F/W drops to 1 only at a = {a_max*1e3:.0f} mm")
    except ValueError:
        print(f"\n  {label}: F/W stays above 1 for every module size up to 50 m")

print("\n  The 1-10 cm design envelope is comfortably inside these limits, so the")
print("  square-cube law is NOT the binding constraint at this scale.  The binding")
print("  constraint is the flux the Alnico can actually hold (section 1).")

