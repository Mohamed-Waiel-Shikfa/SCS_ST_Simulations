"""Permanent-magnet materials available to the search.

Scope
-----
An electropermanent magnet only works if a coil pulse can reverse the magnet,
so the material has to be *switchable*.  That rules out nothing a priori
except by cost: the ampere-turns needed scale directly with intrinsic
coercivity, so the ceiling here is set at

    Hcj <= HCJ_MAX = 2000 kA/m

which admits every commercially available class from the very soft
Cunife/Vicalloy family through Alnico, FeCrCo, MnAlC and the ferrites, up to
SmCo and the low-coercivity end of NdFeB.  Whether a given grade is *usable*
is not decided here - it is decided downstream, by whether a driver can be
built for it.  Several grades in this table are deliberately hopeless for
that reason; leaving them in lets the optimiser demonstrate where the wall is
instead of the wall being asserted.

Why the ceiling is not lower
----------------------------
The earlier version of this file carried Alnico only, which pre-judged the
central trade.  Coercivity buys two things - repulsion, because a magnet that
resists its neighbour's field keeps its polarisation, and demagnetisation
margin - and costs one, switching energy, which goes as Hcj^2 through the
capacitor bank.  With one material family in the table the search could only
move along that trade over a 3x range of Hcj.  It now spans 50x.

Columns
-------
Br      remanence, T
Hcb     normal coercivity (B = 0), A/m
Hcj     intrinsic coercivity (J = 0), A/m
BHmax   energy product, J/m^3
mu_rec  recoil permeability, dimensionless.  This is the slope the magnet
        returns along after being pushed down its curve, and it controls how
        much polarisation recovers when a repelling neighbour moves away.
        Alnico 5 recoils strongly (~4); the anisotropic Alnico 8/9 grades are
        straighter (~2); ferrite and the rare earths are essentially linear
        (~1.05).
rho     density, kg/m^3.  Not cosmetic: ferrite is 4900 against Alnico's 7300
        and SmCo's 8400, and magnet mass is a large share of module mass.
src     "vendor" - a row from the supplier table in Force_compute/, orderable
        as-is.  "lit" - typical published values for the grade, which need a
        real datasheet before anything is designed in.

The distinction matters and is surfaced in the UI: a design whose winning
material is a "lit" row is a research result, not a bill of materials.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from magnet_force import Material  # noqa: E402

HCJ_MAX = 2000e3


def _m(Br, Hcb, Hcj, BHmax, mu_rec, rho, family, src, note=""):
    return dict(Br=Br, Hcb=Hcb, Hcj=Hcj, BHmax=BHmax, mu_rec=mu_rec,
                rho=rho, family=family, src=src, note=note)


# --------------------------------------------------------------------------
MATERIALS = {
    # ---- Alnico, isotropic cast (LNG) --------------------------------------
    # Vendor rows from simulations/Force_compute/Alnico performance table.
    "LNG13":    _m(0.68, 48e3, 51e3, 13e3, 4.0, 7300, "AlNiCo", "vendor"),
    "LNG37":    _m(1.20, 48e3, 49e3, 37e3, 4.0, 7300, "AlNiCo", "vendor",
                   "the grade in the measured experimental data"),
    "LNG40":    _m(1.25, 48e3, 49e3, 40e3, 4.0, 7300, "AlNiCo", "vendor"),
    "LNG52":    _m(1.30, 56e3, 57e3, 52e3, 4.0, 7300, "AlNiCo", "vendor"),
    "LNG60":    _m(1.35, 59e3, 60e3, 60e3, 4.0, 7300, "AlNiCo", "vendor"),
    # ---- Alnico, anisotropic (LNGT) ----------------------------------------
    "LNGT18":   _m(0.58, 90e3, 92e3, 18e3, 2.5, 7300, "AlNiCo", "vendor"),
    "LNGT28":   _m(1.00, 58e3, 59e3, 28e3, 3.5, 7300, "AlNiCo", "vendor"),
    "LNGT36J":  _m(0.70, 140e3, 148e3, 36e3, 2.0, 7300, "AlNiCo", "vendor"),
    "LNGT38":   _m(0.80, 110e3, 112e3, 38e3, 2.0, 7300, "AlNiCo", "vendor"),
    "LNGT44":   _m(0.88, 120e3, 122e3, 44e3, 2.0, 7300, "AlNiCo", "vendor"),
    "LNGT60":   _m(0.90, 110e3, 112e3, 60e3, 2.0, 7300, "AlNiCo", "vendor"),
    "LNGT72":   _m(1.05, 112e3, 114e3, 72e3, 2.0, 7300, "AlNiCo", "vendor"),

    # ---- Fe-Cr-Co: Alnico-like but ductile and machinable -------------------
    "FeCrCo12": _m(0.80, 30e3, 32e3, 12e3, 4.5, 7600, "FeCrCo", "lit"),
    "FeCrCo28": _m(1.05, 44e3, 46e3, 28e3, 4.0, 7600, "FeCrCo", "lit"),
    "FeCrCo42": _m(1.20, 59e3, 62e3, 42e3, 3.5, 7600, "FeCrCo", "lit"),

    # ---- Very soft "semi-hard" alloys: cheap to switch, weak ---------------
    "Cunife":   _m(0.55, 42e3, 44e3, 12e3, 3.0, 8600, "semi-hard", "lit",
                   "drawable into wire; historically used for meter magnets"),
    "Vicalloy2": _m(0.90, 36e3, 38e3, 16e3, 3.5, 8100, "semi-hard", "lit"),

    # ---- Mn-Al-C: rare-earth-free, light, mid coercivity --------------------
    "MnAlC":    _m(0.55, 200e3, 240e3, 45e3, 1.2, 5100, "MnAlC", "lit",
                   "lowest density in the table by a wide margin"),

    # ---- Hard ferrite: cheap, light, and nearly linear ----------------------
    "FerriteY25": _m(0.36, 170e3, 180e3, 25e3, 1.10, 4900, "Ferrite", "lit"),
    "FerriteY30": _m(0.38, 175e3, 195e3, 27e3, 1.10, 4900, "Ferrite", "lit"),
    "FerriteY33": _m(0.41, 220e3, 235e3, 32e3, 1.08, 4950, "Ferrite", "lit"),
    "FerriteY35": _m(0.42, 235e3, 250e3, 34e3, 1.08, 4950, "Ferrite", "lit"),
    "FerriteBond": _m(0.23, 160e3, 240e3, 10e3, 1.15, 3600, "Ferrite", "lit",
                      "injection moulded; can be formed to the pole shape"),

    # ---- Sm-Co: strong, hot, and expensive to switch ------------------------
    "SmCo5-16":  _m(0.83, 620e3, 1200e3, 127e3, 1.05, 8300, "SmCo", "lit"),
    "SmCo5-20":  _m(0.92, 680e3, 1400e3, 160e3, 1.05, 8300, "SmCo", "lit"),
    "Sm2Co17-24": _m(1.00, 720e3, 1600e3, 190e3, 1.04, 8400, "SmCo", "lit"),
    "Sm2Co17-30": _m(1.12, 820e3, 1990e3, 240e3, 1.04, 8400, "SmCo", "lit",
                     "at the Hcj ceiling; included to show where the wall is"),

    # ---- NdFeB at the low-coercivity end ------------------------------------
    "NdFeB-bond": _m(0.68, 380e3, 760e3, 80e3, 1.15, 6000, "NdFeB", "lit"),
    "NdFeB-N35":  _m(1.18, 860e3, 955e3, 263e3, 1.05, 7500, "NdFeB", "lit",
                     "the strongest thing that is still nominally switchable"),
}

for _name, _row in MATERIALS.items():
    if _row["Hcj"] > HCJ_MAX:
        raise ValueError(f"{_name}: Hcj {_row['Hcj']:.0f} exceeds the "
                         f"{HCJ_MAX:.0f} A/m ceiling")

FAMILIES = sorted({m["family"] for m in MATERIALS.values()})
VENDOR = [k for k, v in MATERIALS.items() if v["src"] == "vendor"]
ALL_NAMES = list(MATERIALS)

_FIT_CACHE = {}


def material(name):
    """Fitted ``Material`` for a catalogue entry.

    The fit solves for the two shape parameters of the intrinsic curve so it
    reproduces the catalogue Hcb and (BH)max, and is slow enough to be worth
    caching - the optimiser asks for the same dozen materials thousands of
    times.
    """
    if name not in _FIT_CACHE:
        d = MATERIALS[name]
        _FIT_CACHE[name] = Material.from_datasheet(
            name, Br=d["Br"], Hcb=d["Hcb"], Hcj=d["Hcj"], BHmax=d["BHmax"],
            mu_rec=d["mu_rec"])
    return _FIT_CACHE[name]


def density(name):
    return MATERIALS[name]["rho"]


def switching_class(name):
    """Rough guide to what it takes to switch this grade.

    Purely descriptive - the driver stage decides feasibility - but useful for
    labelling a table so a reader is not surprised that SmCo has no driver.
    """
    h = MATERIALS[name]["Hcj"]
    if h < 60e3:
        return "trivial"
    if h < 160e3:
        return "easy"
    if h < 320e3:
        return "moderate"
    if h < 900e3:
        return "hard"
    return "impractical"


if __name__ == "__main__":
    import numpy as np

    print("=" * 96)
    print("PERMANENT MAGNET MATERIALS AVAILABLE TO THE SEARCH")
    print("=" * 96)
    print(f"\n  ceiling: Hcj <= {HCJ_MAX/1e3:.0f} kA/m     "
          f"{len(MATERIALS)} grades in {len(FAMILIES)} families\n")
    print(f"  {'grade':<13} {'family':<10} {'Br':>6} {'Hcb':>8} {'Hcj':>8} "
          f"{'BHmax':>7} {'mu_r':>5} {'rho':>6} {'switching':<12} {'src':<7}")
    print("  " + "-" * 92)
    for k, v in sorted(MATERIALS.items(),
                       key=lambda kv: (kv[1]["family"], kv[1]["Hcj"])):
        print(f"  {k:<13} {v['family']:<10} {v['Br']:6.2f} "
              f"{v['Hcb']/1e3:7.0f}k {v['Hcj']/1e3:7.0f}k "
              f"{v['BHmax']/1e3:6.0f}k {v['mu_rec']:5.2f} {v['rho']:6.0f} "
              f"{switching_class(k):<12} {v['src']:<7}")

    print("\n  curve fit check (J at H = -Hcj/2, and B = 0 crossing):")
    print("  " + "-" * 92)
    bad = 0
    for k in MATERIALS:
        m = material(k)
        H = -np.linspace(0.0, m.Hcj, 20001)
        B = m.B(H)
        idx = int(np.argmax(B <= 0.0))
        hcb_fit = -H[idx] if idx else m.Hcj
        want = MATERIALS[k]["Hcb"]
        err = abs(hcb_fit - want) / want * 100
        flag = "" if err < 5 else "   <-- poor fit"
        bad += err >= 5
        print(f"  {k:<13} Hcb fit {hcb_fit/1e3:7.1f}k  vs "
              f"datasheet {want/1e3:7.1f}k   ({err:4.1f} %){flag}")
    print(f"\n  {len(MATERIALS)-bad}/{len(MATERIALS)} grades fit their "
          f"datasheet Hcb to better than 5 %.")
