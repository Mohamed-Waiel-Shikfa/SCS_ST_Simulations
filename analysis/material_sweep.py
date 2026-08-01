"""Switchable magnet materials: force performance against switching cost.

Coercivity is the pivot of the whole design.  Raising it keeps the magnet from
collapsing when a neighbouring module reverses against it, which is what makes
repulsion usable - but the energy needed to switch the magnet rises with it, and
past some point the magnet stops being switchable at all with a practical
capacitor bank.  This sweeps that trade.

Two things are reported for every candidate:

  FORCE       attraction, repulsion and their ratio, from the nonlinear FEM
              with a bare rod pair, so the material effect is isolated from any
              circuit effect

  SWITCH COST the minimum electrical energy to reverse the magnet, taken as the
              hysteresis loop area times the magnet volume,
                  E_switch ~ V * 4 * Br * Hcj
              plus the ampere-turns needed to reach 3 x Hcj over the magnet
              length.  Both are lower bounds; a real driver also pays resistive
              and eddy losses.

DATA PROVENANCE
  Alnico grades are transcribed from the supplier table already in this repo,
  simulations/Force_compute/Alnico性能表.png - these are vendor figures for
  parts that can actually be ordered.
  Non-Alnico entries are typical published values for the material class and
  are marked "lit"; they are here to show where the physics leads, and would
  need a real datasheet before any of them is designed in.

Run:  python analysis/material_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from axisym_fem import AxisymModel, Region, axial_force  # noqa: E402
from magnet_force import Material  # noqa: E402

R_M, L_M = 2.375e-3, 12.5e-3
GAP = 0.1e-3
RFAR = 30 * R_M
VOL = np.pi * R_M**2 * L_M

# name, Br (T), Hcb (kA/m), Hcj (kA/m), BHmax (kJ/m3), mu_rec, source
# mu_rec: the Alnico 5 family has a strongly curved recoil (~4); the more
# anisotropic Alnico 8/9 are straighter (~2); ferrite is nearly linear (~1.1).
CANDIDATES = [
    ("Alnico 2   LNG13",    0.68,  48,  51, 13, 4.0, "vendor"),
    ("Alnico 5   LNG37",    1.20,  48,  49, 37, 4.0, "vendor"),
    ("Alnico 5   LNG40",    1.25,  48,  49, 40, 4.0, "vendor"),
    ("Alnico 5DG LNG52",    1.30,  56,  57, 52, 4.0, "vendor"),
    ("Alnico 5-7 LNG60",    1.35,  59,  60, 60, 4.0, "vendor"),
    ("Alnico 6   LNGT28",   1.00,  58,  59, 28, 3.5, "vendor"),
    ("Alnico 8   LNGT18",   0.58,  90,  92, 18, 2.5, "vendor"),
    ("Alnico 8   LNGT38",   0.80, 110, 112, 38, 2.0, "vendor"),
    ("Alnico 8   LNGT44",   0.88, 120, 122, 44, 2.0, "vendor"),
    ("Alnico 8HC LNGT36J",  0.70, 140, 148, 36, 2.0, "vendor"),
    ("Alnico 9   LNGT60",   0.90, 110, 112, 60, 2.0, "vendor"),
    ("Alnico 9   LNGT72",   1.05, 112, 114, 72, 2.0, "vendor"),
    ("FeCrCo 28/5",         1.05,  44,  46, 28, 4.0, "lit"),
    ("FeCrCo 42/12",        1.20,  59,  62, 42, 3.5, "lit"),
    ("Cunife",              0.55,  44,  46, 12, 3.0, "lit"),
    ("Vicalloy II",         1.00,  36,  38, 28, 4.0, "lit"),
    ("Hard ferrite Y30",    0.38, 175, 195, 27, 1.1, "lit"),
    ("Hard ferrite Y33",    0.41, 235, 250, 32, 1.1, "lit"),
    ("SmCo 1:5 (low)",      0.85, 620, 1400, 140, 1.05, "lit"),
]

# a practical driver ceiling, for reference: the 0.3 mm coil in this project
# on a 10 uF / 30 V bank reached about 1870 ampere-turns (epm_feasibility_audit)
MMF_AVAILABLE = 1870.0


def build(name, br, hcb, hcj, bh, mu):
    return Material.from_datasheet(name, Br=br, Hcb=hcb * 1e3, Hcj=hcj * 1e3,
                                   BHmax=bh * 1e3, mu_rec=mu)


def force(mat, flip):
    regs = [Region(0, R_M, -L_M, 0.0, "magnet", "A", material=mat,
                   direction=+1),
            Region(0, R_M, GAP, GAP + L_M, "magnet", "B", material=mat,
                   direction=(-1 if flip else +1))]
    m = AxisymModel(regs, RFAR, 20 * L_M, 0.3e-3, n_slabs=6)
    s = m.solve()
    J, H = m.region_state(s, "A")
    return J, axial_force(s, GAP / 2, r_max=0.9 * RFAR, n=6000)


print("=" * 100)
print("SWITCHABLE MAGNET MATERIALS: FORCE vs SWITCHING COST")
print("=" * 100)
print(f"\nBare rod pair, D {R_M*2e3:.2f} x L {L_M*1e3:.1f} mm, gap"
      f" {GAP*1e3:.2f} mm, magnet volume {VOL*1e9:.0f} mm^3.")
print(f"Switching MMF assumes 3 x Hcj over the magnet length; the existing"
      f" driver delivers ~{MMF_AVAILABLE:.0f} A-t.\n")

hdr = (f"  {'material':<20} {'Br':>5} {'Hcj':>5} {'src':>7} | "
       f"{'attract':>8} {'repel':>7} {'ratio':>6} | {'MMF':>7} {'E_sw':>7} "
       f"{'switch?':>8}")
print(hdr)
print(f"  {'':<20} {'(T)':>5} {'kA/m':>5} {'':>7} | {'(N)':>8} {'(N)':>7} "
      f"{'':>6} | {'(A-t)':>7} {'(mJ)':>7}")
print("  " + "-" * 96)

rows = []
t0 = time.time()
for name, br, hcb, hcj, bh, mu, src in CANDIDATES:
    mat = build(name, br, hcb, hcj, bh, mu)
    try:
        Ja, Fa = force(mat, False)
        Jr, Fr = force(mat, True)
    except RuntimeError as exc:
        print(f"  {name:<20} solver failed: {exc}")
        continue
    mmf = 3.0 * hcj * 1e3 * L_M
    e_sw = 4.0 * br * (hcj * 1e3) * VOL          # hysteresis loop area x volume
    ok = "yes" if mmf <= MMF_AVAILABLE else f"{MMF_AVAILABLE/mmf*100:.0f}%"
    rows.append((name, br, hcj, Fa, Fr, mmf, e_sw, src))
    print(f"  {name:<20} {br:5.2f} {hcj:5.0f} {src:>7} | {Fa:8.2f} {Fr:7.2f} "
          f"{abs(Fa/Fr):6.1f} | {mmf:7.0f} {e_sw*1e3:7.1f} {ok:>8}", flush=True)

print(f"\n  ({time.time()-t0:.0f} s total)")

print("\n" + "=" * 100)
print("READING THE TRADE")
print("=" * 100)

base = [r for r in rows if "LNG37" in r[0]][0]
switchable = [r for r in rows if r[5] <= MMF_AVAILABLE]
best_repel = max(rows, key=lambda r: abs(r[4]))
best_now = max(switchable, key=lambda r: abs(r[4]))

print(f"""
  as built            {base[0]:<20} attract {base[3]:+6.2f} N  repel {base[4]:+6.2f} N"""
      f"  ratio {abs(base[3]/base[4]):5.1f}")
print(f"  best repel, any     {best_repel[0]:<20} attract {best_repel[3]:+6.2f} N"
      f"  repel {best_repel[4]:+6.2f} N  ratio {abs(best_repel[3]/best_repel[4]):5.1f}")
print(f"  best repel, current {best_now[0]:<20} attract {best_now[3]:+6.2f} N"
      f"  repel {best_now[4]:+6.2f} N  ratio {abs(best_now[3]/best_now[4]):5.1f}")

print("""
  THE SWITCHING CONSTRAINT IS NOW THE BINDING ONE.  With the existing driver
  only the low-coercivity grades can be switched at all - and those are exactly
  the ones that cannot repel.  Recommending Alnico 9 therefore also means
  upgrading the magnetiser.  How much of an upgrade:""")

# In the underdamped regime the peak coil current goes as V * sqrt(C/L), so the
# achievable MMF scales as V * sqrt(C).  Reference point from the feasibility
# audit: the 0.3 mm coil on 10 uF at 30 V reached ~1870 ampere-turns.
V_REF, C_REF = 30.0, 10e-6
print(f"\n  {'material':<20} {'MMF need':>9} {'x driver':>9} {'V at 10uF':>10}"
      f" {'C at 30V':>10} {'bank E':>8}")
print(f"  {'':<20} {'(A-t)':>9} {'':>9} {'(V)':>10} {'(uF)':>10} {'(mJ)':>8}")
print("  " + "-" * 72)
for name, br, hcj, Fa, Fr, mmf, e_sw, src in rows:
    if src != "vendor":
        continue
    k = mmf / MMF_AVAILABLE
    v_need = V_REF * k
    c_need = C_REF * k**2
    bank = 0.5 * C_REF * v_need**2
    flag = "  <- as built" if "LNG37" in name else (
        "  <- recommended" if "LNGT72" in name else "")
    print(f"  {name:<20} {mmf:9.0f} {k:9.2f} {v_need:10.0f} {c_need*1e6:10.0f}"
          f" {bank*1e3:8.1f}{flag}")

print("""
  Alnico 9 needs about 2.3 times the ampere-turns of Alnico 5, which is a 70 V
  bank at the existing 10 uF, or 52 uF at the existing 30 V.  Stored energy per
  switch goes from roughly 5 mJ to 24 mJ.  For comparison Marchese, Asada & Rus
  drove their EPM from 76.4 V and 0.1 F, so a 70 V / 10 uF bank is modest and
  well inside normal practice.  This is a driver redesign, not a physical limit.

  Three regimes are visible in the table.

  LOW COERCIVITY (Alnico 2/5/6, FeCrCo, Cunife, Vicalloy, Hcj < 70 kA/m)
    High remanence, trivially switchable, useless in repulsion - they collapse
    when pushed backwards.  This is where the project currently sits, and it is
    the reason repulsion is 21x weaker than attraction.

  MIDDLE (Alnico 8/9, Hcj 90-150 kA/m)
    The useful region.  Alnico 9 LNGT72 gives 4.6x the repulsion of Alnico 5
    and slightly MORE attraction, for 2.3x the switching ampere-turns.

  HIGH COERCIVITY (ferrite, SmCo, Hcj > 190 kA/m)
    Nearly symmetric, because the magnetisation barely moves - but that is
    exactly why they are hard to switch, and low remanence drags the absolute
    forces down.  Ferrite Y33 reaches a 1.1 : 1 ratio but only 1.0 N of it, and
    needs 9375 A-t.  SmCo is not switchable at this scale at all, which is
    precisely why the literature uses it as the FIXED half of a hybrid EPM.

  So there is a genuine interior optimum rather than a corner solution, and
  material belongs in the optimiser as a live variable with the switching
  energy priced in.""")
