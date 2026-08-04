"""Stage 3b: transient simulation of the switching circuit.

This replaces the closed-form "underdamped LC peak current" estimate that the
switching stage used before.  That estimate answered one question - could the
peak current possibly reach the threshold - and got three things wrong that
matter.

1.  **It ignored the neighbour.**  Switching happens while latched, and a
    mated module changes the magnetic circuit the coil drives.  The old model
    computed a free-space inductance and designed the driver against it.

2.  **It ignored the steel and the material permeability.**  Coil inductance
    was ``mu0 N^2 A / l`` - an air-cored solenoid.  A real EPM coil is wound on
    a magnet of recoil permeability 1.04 to 4.0 sitting inside a steel return
    path, which changes the inductance by two orders of magnitude and the
    ampere-turns needed by a factor of three.  Worse, it changes them by
    DIFFERENT factors for different materials, so a comparison between grades
    made with an air-cored model is not just offset, it is mis-ordered.

3.  **It only ever considered one pulse.**  Close the switch, let the LC ring,
    take the first peak.  There is no reason the drive has to be a single
    shot: a train of shorter pulses at a chosen frequency and duty can reach
    the same peak field for less delivered energy, because the coil current
    does not have to be built from zero every time and the resistive loss goes
    as the integral of i^2, not of i.

What is solved here
-------------------
The series loop

    L(i) di/dt + (R_coil + R_esr + R_ds_on) i + v_c = 0
    C dv_c/dt = -i

with the switch state set by a pulse program, integrated by RK4 with a step
tied to the smaller of the LC period and the pulse width.  ``L`` comes from the
magnetic circuit built in ``coil.py``, whose reluctance is measured from the
Stage 1 field solve, and it FALLS as the steel saturates - which is what limits
what a bigger capacitor can buy.

The output that matters is the peak field driven into the magnet, ``H_peak``,
against the switching threshold, plus the energy actually drawn from the bank.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from axisym_fem import steel_nu  # noqa: E402

MU0 = 4.0e-7 * np.pi

# Smallest inductance the integrator will accept.  A real coil of a few
# hundred turns cannot be below this, and without a floor the RK4 step
# overflows for degenerate designs.
L_FLOOR = 1e-9

# Largest current the model will admit.  Nothing in this design space can
# deliver a kiloamp; a trace that reaches it has diverged, and the design is
# recorded as unable to switch rather than allowed to poison the objective
# with a NaN.
I_MAX = 5.0e3


# --------------------------------------------------------------------------
@dataclass
class PulseProgram:
    """How the bridge is driven.

    ``mode``:
      * ``"single"``  - close the switch once and let the loop ring.  This is
        the classical capacitor-discharge EPM drive.
      * ``"train"``   - a square-wave gate at ``f_pulse`` with duty ``duty``
        for ``n_pulses`` cycles.  Between pulses the coil current freewheels
        through the body diodes, so it decays rather than reversing.
    """

    mode: str = "single"
    f_pulse: float = 20e3
    duty: float = 0.5
    n_pulses: int = 6

    def gate(self, t):
        if self.mode == "single":
            return 1.0
        T = 1.0 / max(self.f_pulse, 1.0)
        if t >= self.n_pulses * T:
            return 0.0
        return 1.0 if (t % T) < self.duty * T else 0.0

    def duration(self):
        if self.mode == "single":
            return None
        return self.n_pulses / max(self.f_pulse, 1.0)

    def label(self):
        if self.mode == "single":
            return "single shot"
        return (f"{self.n_pulses} pulses at {self.f_pulse/1e3:.0f} kHz, "
                f"{self.duty*100:.0f} % duty")


@dataclass
class Transient:
    t: np.ndarray
    i: np.ndarray
    v_c: np.ndarray
    h_mag: np.ndarray          # field driven into the magnet
    gate: np.ndarray
    i_peak: float
    h_peak: float
    mmf_peak: float
    e_drawn: float             # energy taken out of the bank
    e_resistive: float
    t_peak: float
    b_steel_peak: float
    saturated: bool
    switched: bool = False
    h_need: float = 0.0
    meta: dict = field(default_factory=dict)

    def summary(self):
        return (f"i_peak {self.i_peak:.1f} A at {self.t_peak*1e6:.0f} us, "
                f"H_peak {self.h_peak/1e3:.0f} kA/m "
                f"({self.h_peak/max(self.h_need,1e-9):.2f}x needed), "
                f"{self.e_drawn*1e3:.1f} mJ drawn, "
                f"{self.e_resistive/max(self.e_drawn,1e-12)*100:.0f} % lost "
                f"in resistance")


def _inductance(circ, n_turns, i, sat_knee):
    """Coil inductance at current ``i``, falling as the iron saturates.

    Below the knee the iron is nearly free and the inductance is the
    small-signal value from the magnetic circuit.  Above it the iron stops
    carrying the incremental flux and the coil sees progressively more air, so
    L is rolled off towards the value it would have with no return path at
    all.  This is the mechanism that stops a bigger capacitor buying more
    field.

    The floor matters more than it looks.  A design with n_eff close to one -
    a bare rod of a high recoil-permeability grade - has almost no inductance,
    and an unfloored L put di/dt past the range of a double and filled the
    trace with NaN.  Because a NaN peak field compares false against the
    threshold, those designs were silently marked unable to switch instead of
    raising anything, which quietly removed a whole corner of the search
    space.
    """
    l0 = max(circ.inductance(n_turns), L_FLOOR)
    if not circ.has_steel or sat_knee <= 0 or not np.isfinite(sat_knee):
        return l0
    x = abs(i) / sat_knee
    return max(l0 * (0.06 + 0.94 / (1.0 + x ** 2)), L_FLOOR)


def simulate(circ, n_turns, r_coil, c_bank, v_bank, r_series=0.05,
             program=None, h_need=0.0, t_end=None, n_steps=6000,
             b_sat=1.95):
    """Integrate the switching loop.

    ``circ`` is a ``coil.MagCircuit`` - ideally one whose ``n_eff`` was
    measured from the Stage 1 FEM, so the inductance and the field-per-ampere
    are the ones the real magnetic circuit gives, including the steel, the
    magnet's own permeability and any latched neighbour.
    """
    program = program or PulseProgram()
    l0 = max(circ.inductance(n_turns), 1e-12)
    r_tot = max(r_coil + r_series, 1e-4)

    # current at which the annulus reaches saturation
    if circ.has_steel and circ.a_steel > 0:
        phi_sat = b_sat * circ.a_steel
        h_sat = phi_sat / max(MU0 * circ.mu_rec * circ.a_magnet, 1e-18)
        i_sat = max(circ.mmf_for_h(h_sat) / max(n_turns, 1), 1e-6)
    else:
        i_sat = np.inf

    omega = 1.0 / np.sqrt(l0 * c_bank) if l0 * c_bank > 0 else 1e9
    t_lc = 2.0 * np.pi / omega
    if t_end is None:
        t_end = program.duration() or min(1.2 * t_lc, 5e-3)
        t_end = max(t_end, 3.0 * l0 / r_tot, 5e-6)
        t_end = min(t_end, 20e-3)
    t_end = float(np.clip(t_end, 1e-6, 20e-3))
    # resolve the LC ring itself, not merely the window: a very low inductance
    # rings far faster than the window suggests and an under-resolved RK4 step
    # on a stiff loop is what produced the NaNs
    n_steps = int(np.clip(max(n_steps, 40.0 * t_end / max(t_lc, 1e-12)),
                          200, 60000))
    dt = t_end / n_steps
    if program.mode == "train":
        dt = min(dt, 1.0 / max(program.f_pulse, 1.0) / 60.0)
        n_steps = int(np.clip(t_end / dt, 200, 200000))
        dt = t_end / n_steps

    def deriv(t, y):
        i, vc = y
        g = program.gate(t)
        L = _inductance(circ, n_turns, i, i_sat)
        if g > 0.5:
            di = (vc - r_tot * i) / L
        else:
            # freewheel: the coil drives its current through the diodes into
            # the bank, recovering some energy and decaying
            di = (-r_tot * i - np.sign(i) * 1.0) / L
        dv = -i / c_bank if g > 0.5 else 0.0
        return np.array([di, dv])

    y = np.array([0.0, float(v_bank)])
    T = np.empty(n_steps + 1)
    I = np.empty(n_steps + 1)
    V = np.empty(n_steps + 1)
    Gt = np.empty(n_steps + 1)
    T[0], I[0], V[0], Gt[0] = 0.0, 0.0, v_bank, program.gate(0.0)
    e_res = 0.0
    diverged = False
    for k in range(n_steps):
        t = k * dt
        k1 = deriv(t, y)
        k2 = deriv(t + dt / 2, y + dt / 2 * k1)
        k3 = deriv(t + dt / 2, y + dt / 2 * k2)
        k4 = deriv(t + dt, y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(y)) or abs(y[0]) > I_MAX:
            diverged = True
            y = np.array([float(np.clip(np.nan_to_num(y[0]), -I_MAX, I_MAX)),
                          float(np.nan_to_num(y[1]))])
            T[k + 1:], I[k + 1:], V[k + 1:] = t + dt, y[0], y[1]
            Gt[k + 1:] = 0.0
            break
        if y[0] < 0 and program.mode == "train":
            y[0] = 0.0                      # diodes block reverse current
        e_res += r_tot * y[0] ** 2 * dt
        T[k + 1], I[k + 1], V[k + 1] = t + dt, y[0], y[1]
        Gt[k + 1] = program.gate(t + dt)

    mmf = n_turns * I
    H = np.array([circ.h_in_magnet(m) for m in mmf])
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    kpk = int(np.argmax(np.abs(H)))
    i_peak = float(np.max(np.abs(np.nan_to_num(I))))
    b_steel = (abs(mmf[kpk]) * (1 - circ.n_eff) / circ.l_magnet *
               MU0 * circ.mu_rec * circ.a_magnet / circ.a_steel
               ) if circ.a_steel > 0 else 0.0

    e_drawn = 0.5 * c_bank * (v_bank ** 2 - float(V[-1]) ** 2)
    return Transient(t=T, i=np.nan_to_num(I), v_c=np.nan_to_num(V), h_mag=H,
                     gate=Gt, i_peak=i_peak, h_peak=float(abs(H[kpk])),
                     mmf_peak=float(abs(mmf[kpk])),
                     e_drawn=float(np.nan_to_num(max(e_drawn, 0.0))),
                     e_resistive=float(np.nan_to_num(e_res)),
                     t_peak=float(T[kpk]),
                     b_steel_peak=float(np.nan_to_num(b_steel)),
                     saturated=bool(i_peak > i_sat),
                     switched=bool(np.isfinite(H[kpk]) and
                                   abs(H[kpk]) >= h_need and not diverged),
                     h_need=float(h_need),
                     meta=dict(L0=l0, i_sat=float(i_sat), r_tot=r_tot,
                               t_lc=t_lc, diverged=diverged,
                               program=program.label()))


def best_program(circ, n_turns, r_coil, c_bank, v_bank, h_need,
                 r_series=0.05, freqs=(5e3, 10e3, 20e3, 40e3, 80e3),
                 duties=(0.2, 0.35, 0.5, 0.7), n_pulses=(2, 4, 8),
                 objective="energy"):
    """Search pulse programs for the cheapest one that still switches.

    The single shot is always evaluated as the baseline.  A train wins when it
    reaches the same field for less energy out of the bank, which happens
    because the bank is not fully discharged into resistance on the first
    swing.
    """
    base = simulate(circ, n_turns, r_coil, c_bank, v_bank,
                    r_series=r_series, h_need=h_need,
                    program=PulseProgram("single"))
    best, best_key = base, None
    if base.switched:
        best_key = (0, base.e_drawn)

    for f in freqs:
        for d in duties:
            for n in n_pulses:
                p = PulseProgram("train", f_pulse=f, duty=d, n_pulses=n)
                tr = simulate(circ, n_turns, r_coil, c_bank, v_bank,
                              r_series=r_series, h_need=h_need, program=p)
                if not tr.switched:
                    continue
                key = (0, tr.e_drawn) if objective == "energy" else \
                    (0, -tr.h_peak)
                if best_key is None or key < best_key:
                    best, best_key = tr, key
    return base, best


if __name__ == "__main__":
    from coil import circuit, wind

    print("=" * 82)
    print("TRANSIENT SWITCHING CIRCUIT")
    print("=" * 82)

    D, L = 4.2e-3, 8.4e-3
    w = wind(D / 2, L, 0.25e-3, 6)
    hcj = 122e3
    h_need = 3.0 * hcj

    print(f"\n  coil: {w.summary()}")
    print(f"  threshold: 3 x Hcj = {h_need/1e3:.0f} kA/m\n")
    print(f"  {'magnetic circuit':<30} {'L':>9} {'i_peak':>8} {'H_peak':>10} "
          f"{'switch':>7} {'energy':>9}")
    print("  " + "-" * 78)
    for label, kw in (("bare rod", dict(has_steel=False)),
                      ("pot core, free space", dict(has_steel=True)),
                      ("pot core, latched neighbour",
                       dict(has_steel=True, has_neighbour=True))):
        c = circuit(D, L, 4.0, t_steel=1.0e-3, r_clear=0.5e-3, **kw)
        tr = simulate(c, w.n_turns, w.resistance, 47e-6, 120.0,
                      h_need=h_need)
        print(f"  {label:<30} {c.inductance(w.n_turns)*1e6:8.1f}u "
              f"{tr.i_peak:7.1f}A {tr.h_peak/1e3:9.0f}k "
              f"{'yes' if tr.switched else 'NO':>7} "
              f"{tr.e_drawn*1e3:8.1f}mJ")

    print("\n  pulse programme search on the latched circuit:\n")
    c = circuit(D, L, 4.0, t_steel=1.0e-3, r_clear=0.5e-3, has_steel=True,
                has_neighbour=True)
    base, best = best_program(c, w.n_turns, w.resistance, 47e-6, 120.0,
                              h_need)
    print(f"    single shot : {base.summary()}")
    print(f"    best train  : {best.meta['program']}")
    print(f"                  {best.summary()}")
    if best.e_drawn < base.e_drawn:
        print(f"\n    the train reaches the threshold for "
              f"{(1-best.e_drawn/max(base.e_drawn,1e-12))*100:.0f} % less "
              f"energy out of the bank.")
