"""Is the screening fidelity good enough to RANK designs?

The optimiser evaluates ~1000 designs at "screen" fidelity and trusts the
ordering.  That is only legitimate if the screening error is systematic - if
it is design-dependent, the GA is optimising noise.

This measures screen-vs-normal on a deliberately diverse set of designs
(bare rods and pot cores, short and long, weak and strong grades) and reports
both the error and, more importantly, the RANK correlation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from framework import Design, stage1_magnetics  # noqa: E402

CASES = []
for mat in ("LNG37", "LNGT44", "LNGT72", "FeCrCo28"):
    for circ in ("none", "potcore"):
        for d_mm, l_mm in ((4.0, 4.0), (4.75, 12.5), (7.0, 7.0)):
            CASES.append(dict(material=mat, circuit=circ, n_gon=8,
                              r_face=19.4e-3, d_mag=d_mm * 1e-3,
                              l_mag=l_mm * 1e-3, t_steel=1.0e-3,
                              r_clear=0.6e-3, gap=0.1e-3, wire_d=0.25e-3,
                              v_cap=90.0, c_cap=100e-6))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def run(fidelity, frac=None, verbose=False):
    """frac=None uses the built-in mesh rule; otherwise h = min(D,L)/frac."""
    fa, fr, t0 = [], [], time.time()
    for i, kw in enumerate(CASES):
        d = Design(**kw)
        h = None if frac is None else max(min(d.d_mag, d.l_mag) / frac, 0.15e-3)
        t1 = time.time()
        try:
            m = stage1_magnetics(d, fidelity=fidelity, mesh=h)
            fa.append(m["F_attract"])
            fr.append(m["F_repel"])
            tag = f"Fa {m['F_attract']:6.2f}"
        except RuntimeError:
            fa.append(np.nan)
            fr.append(np.nan)
            tag = "STALL     "
        if verbose:
            print(f"    [{i+1:2d}/{len(CASES)}] {d.material:<9}{d.circuit:<9}"
                  f"D{d.d_mag*1e3:.2f} L{d.l_mag*1e3:5.2f}  {tag}"
                  f"  {time.time()-t1:5.1f}s", flush=True)
    return np.array(fa), np.array(fr), time.time() - t0


def report(label, res, fa_n, fr_n):
    fa_s, fr_s, t_s = res
    ok = np.isfinite(fa_s) & np.isfinite(fa_n)
    if not ok.any():
        print(f"{label:<24} all stalled")
        return
    err = np.abs(fa_s[ok] - fa_n[ok]) / fa_n[ok] * 100
    asym_s = fa_s[ok] / np.maximum(fr_s[ok], 1e-9)
    asym_n = fa_n[ok] / np.maximum(fr_n[ok], 1e-9)
    print(f"{label:<24}{t_s:6.0f}s{int(np.sum(~np.isfinite(fa_s))):7d}"
          f"{np.median(err):10.1f}%{np.max(err):10.1f}%"
          f"{spearman(fa_s[ok], fa_n[ok]):9.3f}"
          f"{spearman(asym_s, asym_n):11.3f}", flush=True)


print("=" * 88)
print(f"SCREENING FIDELITY STUDY  ({len(CASES)} designs)")
print("=" * 88)

fa_n, fr_n, t_n = run("normal", verbose=True)
print(f"\nreference (normal): {t_n:.0f}s, "
      f"{int(np.sum(~np.isfinite(fa_n)))} stalls, "
      f"F_attract {np.nanmin(fa_n):.2f} .. {np.nanmax(fa_n):.2f} N")

print(f"\n{'variant':<24}{'time':>7}{'stall':>7}{'med|err|':>11}"
      f"{'max|err|':>11}{'rho(Fa)':>9}{'rho(asym)':>11}")
report("screen (current rule)", run("screen"), fa_n, fr_n)
for frac in (4, 6, 8, 12):
    report(f"screen h=min(D,L)/{frac}", run("screen", frac), fa_n, fr_n)

