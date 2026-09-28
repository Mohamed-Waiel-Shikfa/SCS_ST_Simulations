"""
N52 magnet / foil-coil pulse simulator -- ONE file, TWO interactive figures.

Run
---
    python -m pip install numpy scipy matplotlib
    python magnet_pulse_sim.py
    python magnet_pulse_sim.py --check     # numerical + widget regression tests
    python magnet_pulse_sim.py --no-gui    # calculate and print; no windows

No project modules, downloaded data, web service, or other local file is needed.
Edit Config below. Units in the calculations are SI; plot labels state conversions.

What this actually predicts
--------------------------
A passive magnetoquasistatic, linear-recoil REFERENCE discharge. This is not a
calibrated irreversible-magnetization simulation or a component safety approval.
It includes finite-width foil, geometrically calculated self/mutual inductance,
temperature-dependent copper resistance, a conducting-magnet eddy-current model,
two conducting SCRs, capacitor ESR/ESL, and correct DC-bus freewheeling.

The supplied N52 sheet has Br/Hcb/Hcj MINIMUM specifications, NOT an N52 intrinsic
hysteresis loop. Its plotted loop is explicitly for N42. It cannot determine an
N52 switching-field distribution, dynamic reversal law, or final remanence.
Inventing those would make a more complicated model, not a more accurate one.
No fitted "percent flipped", saturation factor, or prescribed magnet back-EMF is
used here. Beyond irreversible switching, BOTH the circuit and internal-field
reference can deviate from the real experiment. Measure the loop/waveforms, or
use a characterized nonlinear 3-D material model, before claiming a reversal.

Source ledger -- attached datasheets, with page numbers
------------------------------------------------------
[C] TDK B43657, February 2026, pp. 3, 5, 12, 19:
    B43657C5787M05#: 780 uF +/-20%, 450 V, 25.4 x 80 mm, about 48 g.
    ESR_typ = 45 mOhm at 360 Hz AND 60 C; 170 mOhm at 120 Hz AND 20 C.
    Approximate ESL = 20 nH. General-purpose grade, not a specified magnetizer
    pulse capacitor. Ripple-current ratings are NOT kA pulse-current ratings.
    The 45 mOhm value is held constant as requested; it is neither a measured
    pulse ESR nor a temperature/frequency model. Do not combine the two ESR
    points to fit frequency dependence: their temperatures differ.
    https://www.tdk-electronics.tdk.com/ (search B43657)
[S] Littelfuse/IXYS CLA100E1200HB, 20210601d, pp. 2, 3, 5:
    Package V_T0 = 0.82 V, r_T = 5.2 mOhm at T_vj=150 C, explicitly labelled
    "for power loss calculation only". Two SCRs conduct, so the bridge drop is
    2*V_T0 + 2*r_T*I -- each term counted ONCE.
    The 2.7 mOhm value on p. 3 is DIE LEVEL, not a replacement for package r_T.
    Gate delay <=2 us. Holding current <=0.1 A, latching <=0.15 A.
    1200 V blocking. I_TSM=1100 A (45 C), 935 A (150 C), 10 ms half sine;
    I^2t=6050 / 4370 A^2 s sml those same nonrepetitive conditions.
    di/dt=150 A/us repetitive at I_T=300 A, 500 A/us nonrepetitive at 100 A,
    with specified gate drive; t_q=150 us typical under specified reverse bias.
    No sqrt(time) extension of these ratings is used to "approve" a kA pulse.
    https://www.littelfuse.com/ (search CLA100E1200HB)
[M] Eclipse Magnetics "NdFeB Magnets / Neodymium Iron Boron Magnets", pp. 1-4:
    N52 minimum Br=1.430 T, Hcb=796 kA/m, Hcj=875 kA/m, BHmax=398 kJ/m^3.
    Typical resistivity 150 uOhm cm = 1.50e-6 Ohm m, density 7500 kg/m^3.
    N52 maximum working temperature guideline is 60 C (not generic Nxx 80 C).
    Recoil permeability is not given: 1.05 below is a LABELLED assumption.
    https://www.eclipsemagnetics.com/

Reading path / formulas
-----------------------
1. Geometry: a_k = 10 mm + t_Cu/2 + k*(t_Cu+t_Kapton).
   Square-loop perimeter = 8*a_k; R = rho*length/(width*thickness).
   Kapton is BETWEEN turns: 22 copper layers + 21 tape layers, not 22 of each.
   Sharp square corners match the user's two scripts. Real bend radii, joints,
   lead routing, plating clearance and the spiral transition are not specified.
   Thus lengths/fields are calculated for this geometry, not measured precision.
   All external wiring is represented ONLY by the specified 5 mOhm.
   Coil and magnet are assumed electrically insulated from each other. A real
   base insulating layer/clearance must be included in inner_clearance once
   known; zero here reproduces the supplied nominal bore dimensions, not
   permission to short copper to conductive Ni-Cu-Ni magnet plating.
2. Field: the exact finite-straight-wire Biot-Savart integral is integrated
   analytically across the foil's 5 mm width; Gauss quadrature averages its
   radial thickness. No fictitious "distance floor" or clipped color values.
   An independent center formula connects this to the user's short script.
3. Inductance: Neumann's double line integral, using its closed-form parallel-
   segment antiderivative. Averaging over conductor cross-sections removes the
   filament self-singularity. No long-solenoid formula or fitted K factor.
   F. W. Grover, Inductance Calculations (1946); A. E. Ruehli, "Inductance
   Calculations in a Complex Integrated Circuit Environment" (1972),
   https://doi.org/10.1147/rd.165.0470
4. Magnet eddies: a few nested square current ribbons, with R calculated from
   the same material volumes as the field/inductance integrals. These passive
   shorted turns solve L*dI/dt + R*I = voltage. They are a reduced-order
   approximation (currents uniform along z), NOT full 3-D diffusion. Increase
   eddy_rings to check radial convergence; use 0 for the transparent baseline.
5. Recoil: one reversible uniform-Mz mode, chi_eff=chi/(1+N_z*chi).
   Reciprocity gives d(lambda)/dM = mu0*volume*mean(H_per_amp), NOT mu0*N*A.
   This adds a positive rank-one inductance. Initial remanence produces static
   flux, not extra inductance; its time derivative is zero.
   The demagnetizing field is obtained from the same Biot-Savart kernel:
   a uniformly magnetized cuboid has side surface current K=M (A/m);
   H_self = B_bound/mu0 - M inside the magnet.
6. Circuit topology (one selected diagonal, other two SCRs untriggered):

       ideal C -- ESR -- ESL ---- P -- SCR -- 5mOhm -- coil -- SCR -- N
                                  |                              |
                                  +--- diode, anode at N --------+

   The diode is across the DC BUS, not the coil; it supports either selected
   H-bridge polarity. No actual diode was specified: default is an IDEAL,
   zero-drop diode, not an invented datasheet part. A real diode needs pulse
   characterization. BOTH SCRs remain in the recirculating coil-current path.
   During freewheel, I_coil = I_cap + I_diode. Capacitor ESR current does NOT
   suddenly become zero when the diode turns on. Its ESL is retained.
7. Copper heating is adiabatic for this isolated millisecond pulse:
   T=T0+Q_Cu/(mass*cp), R(T)=R20*[1+alpha*(T-20)].
   Foil current density is uniform across its width/thickness in this model.
   Thin-foil skin diffusion is assessed, not "corrected" with a single guessed
   frequency. End/fringing proximity crowding, electrode tabs and Ni plating
   eddies are omitted. They need a 3-D conductor model for improvement:
   https://www.femm.info/wiki/InductanceExample (energy vs flux linkage)
   https://getdp.info/ (general 3-D field formulations).
   The SCR's 43 pF typical junction capacitance [S] and distributed turn
   capacitances are omitted on the millisecond scale, not declared nonexistent.
   They and diode recovery can control the unmodelled fast commutation spikes.

Plots and definitions
---------------------
Exactly TWO matplotlib figures. The field map has a time slider, fixed color
scale, labelled isolines, hover readout and a "Peak" button. The second has three
unit-specific y axes and one independently selectable checkbox per series.
The field selector updates BOTH figures:
  Driven |H|: field from coil + induced eddies, excluding static remanence.
  Recoil-only |H|: includes the permanent magnet and reversible mean recoil;
                  a conditional extrapolation, not valid through reversal.
All four field statistics are over the SAME z=0, cell-centered slice as the
heatmap, NOT a claim about extrema of the whole 3-D volume. The center value is
also available. On this symmetry plane H is axial, so |H| = abs(H_z).
Capacitor voltage means terminal/DC-bus voltage; internal storage voltage is a
separate optional trace. Bridge voltage is the SUM across the selected diagonal
(including its blocking interval), not a single SCR voltage or the coil voltage.

High voltage / stored-energy warning: the nominal charge is 78.975 J at 450 V.
These figures do not establish capacitor pulse suitability, SCR safe operating
area, insulation integrity, repetition rate, mechanical restraint, or safe use.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from math import pi, sqrt
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, solve_ivp
from scipy.linalg import cho_factor, cho_solve


MU0 = 4e-7 * pi


@dataclass(frozen=True)
class Config:
    capacitance: float = 780e-6
    initial_voltage: float = 450.0  # Charge voltage assumption, distinct from rating.
    cap_esr: float = 37e-3         # [C], typical at 360 Hz, 60 C.
    cap_esl: float = 20e-9         # [C], capacitor's ESL; wiring L/C are neglected.
    scr_threshold: float = 0.82    # [S], per package, 150 C loss model.
    scr_slope: float = 5.2e-3      # [S], NOT the die-level 2.7 mOhm.
    gate_delay: float = 2e-6       # [S], maximum reference; ideal hard firing then.
    cutoff_current: float = 0.1   # [S], approximate final commutation below IH.
    wiring_resistance: float = 5e-3
    turns: int = 22
    copper_thickness: float = 0.2e-3
    strip_width: float = 5e-3
    kapton_thickness: float = 0.05e-3
    inner_clearance: float = 0.0   # Unspecified base insulation; nominal bore geometry.
    magnet_side: float = 20e-3
    magnet_height: float = 5e-3
    remanence: float = 1.430       # [M], grade minimum, not measured value.
    hc_b: float = 796e3           # [M], normal coercivity minimum.
    hc_j: float = 875e3           # [M], intrinsic coercivity minimum.
    magnet_resistivity: float = 1.50e-6  # [M], typical.
    magnet_density: float = 7500.0      # [M], typical, kg/m^3.
    magnet_heat_capacity: float = 0.12*4184  # [M], 0.12 kcal/(kg K).
    recoil_mu_r: float = 1.05     # ASSUMPTION, not specified by [M].
    copper_resistivity_20: float = 1.7241e-8  # Standard annealed copper, IACS.
    copper_alpha: float = 0.00393
    copper_density: float = 8960.0
    copper_heat_capacity: float = 385.0
    copper_initial_c: float = 20.0
    heat_copper: bool = True
    eddy_rings: int = 6
    radial_quadrature: int = 6
    inductance_quadrature: int = 80
    slice_pixels: int = 101       # Odd -> includes the center exactly.
    time_samples: int = 2201
    target_h: float = 3e6         # User's engineering comparison, NOT a grade spec.

    def validate(self):
        positive = (
            self.capacitance, self.initial_voltage, self.cap_esr, self.cap_esl,
            self.copper_thickness, self.strip_width, self.magnet_side,
            self.magnet_height, self.magnet_resistivity, self.cutoff_current,
        )
        if any(x <= 0 or not np.isfinite(x) for x in positive):
            raise ValueError("Capacitor, conductor, magnet and cutoff values must be finite and positive.")
        if self.turns < 1 or self.eddy_rings < 0 or self.radial_quadrature < 2:
            raise ValueError("Need >=1 turn, >=0 eddy rings, and >=2 radial quadrature points.")
        if self.slice_pixels < 11 or self.slice_pixels % 2 != 1 or self.time_samples < 101:
            raise ValueError("Use an odd slice resolution >=11 and >=101 time samples.")
        if self.kapton_thickness < 0 or self.inner_clearance < 0 or self.recoil_mu_r < 1:
            raise ValueError("Insulation/clearance must be nonnegative; this recoil model requires mu_r >=1.")
        if not np.isclose(self.strip_width, self.magnet_height, rtol=0, atol=1e-12):
            raise ValueError("The common-width analytical integrals require strip width = magnet height.")
        if self.initial_voltage > 450:
            raise ValueError("Charge voltage exceeds the selected capacitor's 450 V rating.")


def gauss_interval(order, low, high):
    """Nodes/weights for integral_low^high f(x) dx, not an unlabelled average."""
    x, w = leggauss(order)
    return low + (x + 1) * (high - low) / 2, w * (high - low) / 2


def square_sheet_field(points, half_side, width):
    """H per ampere [A/m/A] of one finite-width square turn, circulating +z.

    Start with the finite straight-line formula:
      H = (I/4pi) * (e x r_perp)/r_perp^2
          * [u1/sqrt(u1^2+r_perp^2) - u2/sqrt(u2^2+r_perp^2)].
    Distribute I uniformly along z' in [-width/2, width/2].
    With d the signed in-plane perpendicular distance and Z=z-z':
      integral d*u/[(d^2+Z^2)*sqrt(u^2+d^2+Z^2)] dZ
        = atan(u*Z/[d*sqrt(u^2+d^2+Z^2)])
      integral -Z*u/[(d^2+Z^2)*sqrt(u^2+d^2+Z^2)] dZ
        = asinh(u/sqrt(d^2+Z^2)).
    These give the axial and in-plane-normal components for each of four sides.
    Only the final division by width converts sheet current to TOTAL turn I.
    Points on a mathematical current sheet require a limiting prescription;
    our quadratures use interior cell centers and never sample that surface.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    h = np.zeros_like(p)
    corners = np.array([[-half_side, -half_side, 0], [half_side, -half_side, 0],
                        [half_side, half_side, 0], [-half_side, half_side, 0]])
    z_low, z_high = p[:, 2] - width / 2, p[:, 2] + width / 2
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        tangent = (end - start) / np.linalg.norm(end - start)
        normal = np.cross([0.0, 0.0, 1.0], tangent)
        r = p - start
        distance = r @ normal
        if np.any(np.abs(distance) < 1e-15):
            raise ValueError("Field query lies on a current sheet; change the quadrature/grid.")
        axial = np.zeros(len(p))
        transverse = np.zeros(len(p))
        for u, sign in ((r @ tangent, 1), ((p - end) @ tangent, -1)):
            for z, z_sign in ((z_high, 1), (z_low, -1)):
                radius = np.sqrt(u*u + distance*distance + z*z)
                axial += sign*z_sign*np.arctan(u*z / (distance*radius))
                transverse += sign*z_sign*np.arcsinh(u / np.hypot(distance, z))
        h[:, 2] += axial / (4*pi*width)
        h += transverse[:, None]*normal[None, :] / (4*pi*width)
    return h


def center_filament_h(half_side, axial_offset=0.0):
    """Independent square-filament result; at z=0 this is sqrt(2)/(pi*a)."""
    a, z = half_side, axial_offset
    return 2*a*a / (pi*(a*a + z*z)*sqrt(2*a*a + z*z))


def parallel_partial(half1, half2, distance):
    """Neumann partial inductance [H] of parallel centered segments.

    G(x,d)=x*asinh(x/d)-sqrt(x^2+d^2) is the second antiderivative of 1/r.
    Integrating from -a to a and -b to b gives 2*[G(a+b)-G(a-b)].
    Distances are real quadrature separations: there is no arbitrary GMD floor.
    """
    def g(x):
        return x*np.arcsinh(x/distance) - np.hypot(x, distance)
    return MU0/(2*pi) * (g(half1 + half2) - g(half1 - half2))


def square_mutual(half1, half2, dz):
    """Mutual inductance of two square filaments at axial separation dz.

    Perpendicular sides contribute zero because dl dot dl' = 0.
    Four same-side parallel pairs add; four opposite-side pairs subtract.
    """
    near = np.hypot(half1 - half2, dz)
    far = np.hypot(half1 + half2, dz)
    return 4*(parallel_partial(half1, half2, near)
              - parallel_partial(half1, half2, far))


@dataclass
class Model:
    cfg: Config
    turn_half_sides: np.ndarray
    length: float
    copper_mass: float
    r20: float
    eddy_resistances: np.ndarray
    source_half_sides: np.ndarray
    source_weights: np.ndarray
    inductance: np.ndarray
    air_inductance: float
    demag_n: float
    m_initial: float
    chi_effective: float
    h_mean_per_amp: np.ndarray
    slice_coordinates: np.ndarray
    drive_slice: np.ndarray
    recoil_slice: np.ndarray
    self_slice: np.ndarray

    @property
    def modes(self):
        return self.inductance.shape[0]

    def copper_r(self, q_copper=0.0):
        c = self.cfg
        temperature = c.copper_initial_c
        if c.heat_copper:
            temperature += q_copper / (self.copper_mass*c.copper_heat_capacity)
        return self.r20*(1 + c.copper_alpha*(temperature - 20.0))

    def fields(self, points):
        out = np.zeros((len(points), self.modes, 3))
        for a, weights in zip(self.source_half_sides, self.source_weights):
            h = square_sheet_field(points, a, self.cfg.strip_width)
            out += h[:, None, :]*weights[None, :, None]
        return out


def make_model(c):
    """Build conductor volumes, passive coupling matrix and the two field bases."""
    c.validate()
    offsets = c.copper_thickness/2 + np.arange(c.turns)*(
        c.copper_thickness + c.kapton_thickness)
    turn_a = c.magnet_side/2 + c.inner_clearance + offsets
    length = float(np.sum(8*turn_a))
    cross_section = c.copper_thickness*c.strip_width
    mass = length*cross_section*c.copper_density
    r20 = c.copper_resistivity_20*length/cross_section
    # Eddy mode k is a square annular ribbon of radial width dr and axial
    # width 5 mm. Volume=8*a*dr*w; R=rho*8*a/(dr*w), exactly the same volume.
    dr = c.magnet_side/(2*c.eddy_rings) if c.eddy_rings else 0.0
    eddy_a = (np.arange(c.eddy_rings) + 0.5)*dr
    eddy_r = c.magnet_resistivity*8*eddy_a/(dr*c.strip_width) if c.eddy_rings else np.array([])
    centers = np.r_[turn_a, eddy_a]
    thicknesses = np.r_[np.full(c.turns, c.copper_thickness), np.full(c.eddy_rings, dr)]
    rx, rw = gauss_interval(c.radial_quadrature, -0.5, 0.5)
    source_a = (centers[:, None] + thicknesses[:, None]*rx).ravel()
    # Map each radial integration ribbon into circuit modes. The winding is
    # clockwise (opposes the original +z magnetization); eddy signs are solved.
    weights = np.zeros((len(source_a), 1+c.eddy_rings))
    for k in range(len(centers)):
        rows = slice(k*c.radial_quadrature, (k+1)*c.radial_quadrature)
        weights[rows, 0 if k < c.turns else 1+k-c.turns] = -rw if k < c.turns else rw
    # Average over two uniform, equal-width strips:
    # <M> = 2/w^2 * integral_0^w (w-z)*M(z) dz.
    # z=w*u^2 makes the integrable logarithmic self-singularity gentler.
    uq, uw = gauss_interval(c.inductance_quadrature, 0, 1)
    mutual = np.zeros((len(source_a), len(source_a)))
    a1, a2 = source_a[:, None], source_a[None, :]
    for u, w in zip(uq, uw):
        mutual += (4*u*(1-u*u)*w)*square_mutual(a1, a2, c.strip_width*u*u)
    inductance = weights.T @ mutual @ weights
    inductance = (inductance + inductance.T)/2
    air_l = float(inductance[0, 0])

    # Mid-cell volume quadrature for the uniform reversible mode (not a local
    # permeability multiplier). The same reciprocity coefficient is used in L
    # and in the displayed internal-field increment.
    grid = [(np.arange(n)+0.5)/n*size-size/2
            for size, n in ((c.magnet_side, 15), (c.magnet_side, 15), (c.magnet_height, 9))]
    volume_points = np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1).reshape(-1, 3)
    model = Model(c, turn_a, length, mass, r20, eddy_r, source_a, weights,
                  inductance, air_l, 0, 0, 0, np.zeros(1+c.eddy_rings),
                  np.array([]), np.array([]), np.array([]), np.array([]))
    bound_h = c.magnet_height*square_sheet_field(volume_points, c.magnet_side/2, c.magnet_height)
    bound_h[:, 2] -= 1  # H = B/mu0 - M inside; Br is NOT an H field.
    nz = float(-np.mean(bound_h[:, 2]))
    chi = c.recoil_mu_r - 1
    chi_eff = chi/(1+nz*chi)
    mean_h = np.mean(model.fields(volume_points)[:, :, 2], axis=0)
    volume = c.magnet_side*c.magnet_side*c.magnet_height
    inductance += MU0*volume*chi_eff*np.outer(mean_h, mean_h)
    cho_factor(inductance)  # Fail explicitly if quadrature gives a nonpassive model.
    model.demag_n, model.chi_effective = nz, chi_eff
    model.m_initial = c.remanence/MU0/(1+nz*chi)
    model.h_mean_per_amp = mean_h
    coordinate = (np.arange(c.slice_pixels)+0.5)*c.magnet_side/c.slice_pixels-c.magnet_side/2
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    model.slice_coordinates = coordinate
    # Reflection symmetry about z=0 makes Hx=Hy=0 there.
    h = model.fields(points)
    if np.max(np.abs(h[:, :, :2])) > 1e-7*np.max(np.abs(h[:, :, 2])):
        raise ArithmeticError("Midplane axial symmetry was lost.")
    model.drive_slice = h[:, :, 2]
    self_h = c.magnet_height*square_sheet_field(points, c.magnet_side/2, c.magnet_height)[:, 2]-1
    model.self_slice = self_h*model.m_initial
    model.recoil_slice = model.drive_slice + self_h[:, None]*chi_eff*mean_h[None, :]
    return model


@dataclass
class Pulse:
    time: np.ndarray
    states: np.ndarray
    phase: np.ndarray
    bus_voltage: np.ndarray
    coil_voltage: np.ndarray
    bridge_voltage: np.ndarray
    stored_energy: np.ndarray
    energy_error: np.ndarray
    clamp_time: float | None
    cutoff_time: float
    cutoff_energy: float
    loss_start: int


def simulate(model):
    """Piecewise-smooth ODE with EXACT event states; no forced full discharge.

    State: [u_storage, i_cap, i_coil, i_eddy..., Q_Cu, Q_ESR, Q_wire,
            Q_bridge, Q_eddy, action_per_SCR].
    Phase 1, diode off: i_cap=i_coil and capacitor ESL adds to coil L.
    Phase 2, diode on: v_bus=0; capacitor has its own ESR/ESL decay,
                      coil + two SCRs + wiring recirculate through the diode.
    Phase 3, SCR current <IH: tiny commutation energy is explicitly booked,
                            eddy fluxes are continuous and their tails decay.
    No gate-turn-off command, instantaneous removal of capacitor ESR, or
    disappearance of appreciable inductor energy is hidden in these equations.
    """
    c, n = model.cfg, model.modes
    start_loss = 2+n
    qcu, qcap, qwire, qbridge, qeddy, action = range(start_loss, start_loss+6)
    matrix = model.inductance
    charged = matrix.copy()
    charged[0, 0] += c.cap_esl
    factor_on, factor_free = cho_factor(charged), cho_factor(matrix)
    factor_eddy = cho_factor(matrix[1:, 1:]) if n > 1 else None
    rb = 2*c.scr_slope
    vb = 2*c.scr_threshold

    def derivatives(t, y, phase):
        u, ic, currents = y[0], y[1], y[2:2+n]
        i = currents[0]
        rcoil = model.copper_r(y[qcu])
        dy = np.zeros_like(y)
        emf = -np.r_[rcoil+c.wiring_resistance+rb, model.eddy_resistances]*currents
        if phase == 1:
            emf[0] += u - c.cap_esr*i - vb
            dj = cho_solve(factor_on, emf)
            dic = dj[0]
            bus = u-c.cap_esr*ic-c.cap_esl*dic
        elif phase == 2:
            emf[0] -= vb
            dj = cho_solve(factor_free, emf)
            bus = 0.0  # Ideal diode across DC bus, conducting N -> P.
            dic = (u-c.cap_esr*ic)/c.cap_esl
        else:
            dj = np.zeros(n)
            if n > 1:
                dj[1:] = cho_solve(factor_eddy, -model.eddy_resistances*currents[1:])
            dic, bus = 0.0, u
        dy[0], dy[1], dy[2:2+n] = -ic/c.capacitance, dic, dj
        dy[qcu] = rcoil*i*i
        dy[qcap] = c.cap_esr*ic*ic
        dy[qwire] = c.wiring_resistance*i*i
        dy[qbridge] = vb*i+rb*i*i if phase in (1, 2) else 0
        dy[qeddy] = float(np.dot(model.eddy_resistances, currents[1:]**2))
        dy[action] = i*i
        coil_voltage = rcoil*i+matrix[0] @ dj
        bridge_voltage = bus-c.wiring_resistance*i-coil_voltage
        return dy, bus, coil_voltage, bridge_voltage

    def diode_on(t, y):
        return derivatives(t, y, 1)[1]
    diode_on.direction, diode_on.terminal = -1, True

    def finished(t, y):
        return y[2]-c.cutoff_current
    finished.direction, finished.terminal = -1, True

    def diode_off(t, y):
        return y[2]-y[1]
    diode_off.direction, diode_off.terminal = -1, True

    tau = max(c.capacitance*c.cap_esr, matrix[0, 0]/model.r20,
              sqrt((matrix[0, 0]+c.cap_esl)*c.capacitance))
    end_time = c.gate_delay+20*tau
    y = np.zeros(start_loss+6)
    y[0] = c.initial_voltage
    phase, t0, clamp_time = 1, c.gate_delay, None
    segments = []
    # A properly formed pulse needs at most one diode-on transition here.
    # A bounded hybrid loop also supports natural diode release, without
    # declaring two unequal branch currents identical at an arbitrary time.
    for _ in range(8):
        events = [diode_on, finished] if phase == 1 else [finished, diode_off]
        sol = solve_ivp(lambda t, y: derivatives(t, y, phase)[0], (t0, end_time), y,
                        method="Radau", rtol=2e-8, atol=1e-10, dense_output=True,
                        max_step=tau/12, events=events)
        if not sol.success:
            raise RuntimeError(f"Discharge phase {phase} failed: {sol.message}")
        segments.append((t0, sol.t[-1], phase, sol))
        if not any(len(event) for event in sol.t_events):
            raise RuntimeError("Pulse did not reach commutation; increase horizon or inspect model.")
        which = next(k for k, event in enumerate(sol.t_events) if len(event))
        t0, y = float(sol.t_events[which][0]), sol.y_events[which][0].copy()
        if events[which] is finished:
            break
        if phase == 1:
            clamp_time = t0 if clamp_time is None else clamp_time
            phase = 2
        else:
            phase = 1
    else:
        raise RuntimeError("Unexpected diode chattering; no pulse result was accepted.")

    # Approximate only the final sub-0.1-A commutation, and account for its
    # energy. Eddy flux linkage is continuous: Lee*dIe = Le0*I_cutoff.
    old = y[2:2+n].copy()
    if n > 1:
        y[3:2+n] += cho_solve(factor_eddy, matrix[1:, 0]*old[0])
    y[2] = 0
    cutoff_energy = float(0.5*(old @ matrix @ old-y[2:2+n] @ matrix @ y[2:2+n])
                          +0.5*c.cap_esl*y[1]**2)
    if abs(y[1]) > 1.01*c.cutoff_current or cutoff_energy < -1e-10:
        raise ArithmeticError("Final commutation is not negligible; a switch/snubber model is needed.")
    y[1] = 0
    cutoff_time = t0
    if cutoff_energy > 1e-5:
        raise ArithmeticError("Commutation stores >10 uJ: do not use the small-tail approximation.")
    # Continue far enough to include the slowest remaining eddy-mode decay,
    # instead of displaying a long, uninformative zero-current extension.
    tail_tau = 0.0
    if n > 1:
        root_r = np.sqrt(model.eddy_resistances)
        tail_tau = float(np.linalg.eigvalsh(matrix[1:, 1:]/root_r[:, None]/root_r[None, :])[-1])
    end_time = t0+max(12*tail_tau, 0.1*(t0-c.gate_delay), 1e-6)
    sol = solve_ivp(lambda t, y: derivatives(t, y, 3)[0], (t0, end_time), y,
                    method="Radau", rtol=2e-8, atol=1e-10, dense_output=True, max_step=tau)
    if not sol.success:
        raise RuntimeError(f"Eddy-current tail failed: {sol.message}")
    segments.append((t0, end_time, 3, sol))
    time = np.unique(np.r_[np.linspace(0, end_time, c.time_samples),
                           [segment[0] for segment in segments]])
    states = np.zeros((len(time), len(y)))
    states[:, 0] = c.initial_voltage
    phases = np.zeros(len(time), dtype=int)
    for begin, end, mode, sol in segments:
        indices = (time >= begin) & (time <= end)
        states[indices] = sol.sol(time[indices]).T
        phases[indices] = mode
    bus, vcoil, vbridge = np.empty((3, len(time)))
    for k, (t, y, mode) in enumerate(zip(time, states, phases)):
        _, bus[k], vcoil[k], vbridge[k] = derivatives(t, y, int(mode))
    currents = states[:, 2:2+n]
    energy = (0.5*c.capacitance*states[:, 0]**2 + 0.5*c.cap_esl*states[:, 1]**2
              +0.5*np.einsum("ti,ij,tj->t", currents, matrix, currents))
    losses = states[:, start_loss:start_loss+5].sum(axis=1)
    error = energy+losses+(time >= cutoff_time)*cutoff_energy-0.5*c.capacitance*c.initial_voltage**2
    return Pulse(time, states, phases, bus, vcoil, vbridge, energy, error,
                 clamp_time, cutoff_time, cutoff_energy, start_loss)


def field_statistics(model, pulse):
    """Stats of the displayed cell-centered z=0 plane, in A/m; bounded memory."""
    currents = pulse.states[:, 2:2+model.modes]
    output = {}
    for label, coefficients, static in (
        ("Driven |H|", model.drive_slice, np.zeros_like(model.self_slice)),
        ("Recoil-only |H|", model.recoil_slice, model.self_slice),
    ):
        result = np.empty((len(pulse.time), 5))
        for start in range(0, len(pulse.time), 96):
            stop = min(start+96, len(pulse.time))
            h = np.abs(coefficients @ currents[start:stop].T+static[:, None])
            result[start:stop] = np.stack([h.min(axis=0), h.max(axis=0),
                                          np.median(h, axis=0), h.mean(axis=0),
                                          h[len(h)//2]], axis=1)
        output[label] = result
    return output


def print_report(model, pulse, statistics):
    c = model.cfg
    i = pulse.states[:, 2]
    peak = int(np.argmax(i))
    driven = statistics["Driven |H|"]
    peak_h = int(np.argmax(driven[:, 3]))
    q = pulse.states[-1, pulse.loss_start:pulse.loss_start+6]
    final_c = c.copper_initial_c+q[0]/(model.copper_mass*c.copper_heat_capacity)
    f0 = 1/(2*pi*sqrt((model.inductance[0, 0]+c.cap_esl)*c.capacitance))
    delta_cu = sqrt(c.copper_resistivity_20/(pi*f0*MU0))
    m = model.inductance
    l_fast = m[0, 0]-(m[0, 1:] @ np.linalg.solve(m[1:, 1:], m[1:, 0]) if model.modes > 1 else 0)
    print(f"\nN52 / {c.turns}-turn foil discharge -- conditional linear-recoil reference")
    print(f"Conductor: {model.length:.6f} m; {model.copper_mass*1e3:.3f} g; "
          f"R20 = {model.r20*1e3:.3f} mOhm")
    build = c.turns*c.copper_thickness+(c.turns-1)*c.kapton_thickness
    print(f"Radial build = {build*1e3:.3f} mm; outer square side = "
          f"{(c.magnet_side+2*(c.inner_clearance+build))*1e3:.3f} mm")
    print(f"L air = {model.air_inductance*1e6:.4f} uH; "
          f"L with mean recoil = {m[0, 0]*1e6:.4f} uH")
    print(f"Shorted-eddy high-frequency L limit = {l_fast*1e6:.4f} uH "
          f"({c.eddy_rings} passive magnet modes)")
    print(f"Bridge = {2*c.scr_threshold:.2f} V + {2*c.scr_slope*1e3:g} mOhm * I; "
          f"external wire = {c.wiring_resistance*1e3:g} mOhm")
    print(f"Cap ESR = {c.cap_esr*1e3:g} mOhm, held constant; ESL = {c.cap_esl*1e9:g} nH")
    print(f"Stored energy = {0.5*c.capacitance*c.initial_voltage**2:.6f} J; "
          f"C tolerance: +/-20% (not propagated in this reference run)")
    print(f"Cold LC reference f0 = {f0:.1f} Hz; foil thickness / skin depth = "
          f"{c.copper_thickness/delta_cu:.4f}")
    print(f"Peak coil/SCR current = {i[peak]:.2f} A at {pulse.time[peak]*1e6:.2f} us")
    print(f"Peak capacitor current = {pulse.states[:, 1].max():.2f} A")
    print(f"DC-bus diode starts = {pulse.clamp_time*1e6:.2f} us" if pulse.clamp_time is not None
          else "DC-bus diode did not conduct in this run.")
    print(f"Final SCR commutation = {pulse.cutoff_time*1e6:.2f} us; "
          f"tail energy booked = {pulse.cutoff_energy:.3e} J")
    print(f"Final storage / terminal V = {pulse.states[-1, 0]:.6g} / {pulse.bus_voltage[-1]:.6g} V")
    print(f"Final copper temperature (adiabatic) = {final_c:.2f} C")
    magnet_mass = c.magnet_density*c.magnet_side**2*c.magnet_height
    print(f"Magnet eddy-heating-only temperature rise = "
          f"{q[4]/(magnet_mass*c.magnet_heat_capacity):.3f} K (hysteresis heating is unknown)")
    print("Energy audit [J]: copper / capacitor ESR / wiring / two SCRs / magnet eddies")
    print("                  "+" / ".join(f"{value:.6f}" for value in q[:5]))
    print(f"Max energy-balance residual = {np.max(np.abs(pulse.energy_error)):.3e} J")
    print("At peak SLICE AVERAGE driven field [MA/m]: min / max / median / mean / center")
    print("    "+" / ".join(f"{value/1e6:.4f}" for value in driven[peak_h]))
    center_reference = sum(quad(lambda z: center_filament_h(a, z), -c.strip_width/2,
                                c.strip_width/2, epsabs=1e-9)[0]/c.strip_width
                           for a in model.turn_half_sides)
    thin_reference = sum(center_filament_h(a) for a in model.turn_half_sides)
    print(f"Center H/I: thin-filament sum = {thin_reference:.4f}; "
          f"finite-width integral = {center_reference:.4f} A/m/A")
    print(f"User's 4500-A comparison, NOT a predicted current: "
          f"thin {4500*thin_reference/1e6:.4f}, finite-width {4500*center_reference/1e6:.4f} MA/m")
    reverse = -model.drive_slice @ pulse.states[peak_h, 2:2+model.modes]
    fraction = np.mean(reverse >= c.target_h)
    print(f"Slice area >= {c.target_h/1e6:g} MA/m reverse driven Hz at that instant: "
          f"{100*fraction:.2f}% (not switched magnet volume).")
    print(f"Calculated mean N_z = {model.demag_n:.5f}; "
          f"Br={c.remanence:g} T, Hcb_min={c.hc_b/1e3:g}, Hcj_min={c.hc_j/1e3:g} kA/m.")
    print(f"Initial center self-demag reference = "
          f"{model.self_slice[len(model.self_slice)//2]/1e3:.1f} kA/m.")


def make_figures(model, pulse, statistics):
    """Only this function creates figures. Widgets stay referenced on the figures."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

    # Extend the supplied matplotlib interface: white laboratory-plot background,
    # readable units, fixed scales, direct time control and explicit trace names.
    # No dashboard cards, extra charts, logarithmic clipping or hidden normalization.
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "semibold",
                         "axes.grid": False, "figure.facecolor": "white"})
    c = model.cfg
    mode = ["Driven |H|"]
    initial = int(np.argmax(statistics[mode[0]][:, 3]))
    time_ms = pulse.time*1e3
    currents = pulse.states[:, 2:2+model.modes]
    coord = model.slice_coordinates*1e3
    side_mm = c.magnet_side*1e3
    fig_map, ax_map = plt.subplots(figsize=(8.8, 8.2))
    fig_map.canvas.manager.set_window_title("1 / 2  |  Magnetic field slice")
    fig_map.subplots_adjust(left=0.11, right=0.84, bottom=0.26, top=0.86)
    maximum = float(statistics[mode[0]][:, 1].max()/1e6)
    image = ax_map.imshow(np.zeros((c.slice_pixels, c.slice_pixels)), origin="lower",
                          extent=(-side_mm/2, side_mm/2, -side_mm/2, side_mm/2),
                          cmap="cividis", vmin=0, vmax=maximum, interpolation="nearest")
    colorbar = fig_map.colorbar(image, ax=ax_map, pad=0.04)
    colorbar.set_label("|H| [MA/m]")
    ax_map.set(xlabel="x [mm]", ylabel="y [mm]", aspect="equal")
    fig_map.suptitle("Field inside the magnet  |  z = 0", y=0.975)
    subtitle = fig_map.text(0.11, 0.925, "", fontsize=9, va="top")
    slider = Slider(fig_map.add_axes((0.18, 0.15, 0.51, 0.035)), "Time [ms]",
                    time_ms[0], time_ms[-1], valinit=time_ms[initial], valfmt="%1.4f")
    peak_button = Button(fig_map.add_axes((0.84, 0.145, 0.10, 0.046)), "Peak")
    radio = RadioButtons(fig_map.add_axes((0.11, 0.025, 0.36, 0.09)),
                         tuple(statistics), active=0)
    radio.ax.set_facecolor("white")
    fig_map.text(0.49, 0.068, "Recoil-only is a conditional reference,\nnot a reversal prediction.",
                 fontsize=9, va="center", color="#7a3800")

    fig_ts, ax_i = plt.subplots(figsize=(14.8, 7.8))
    fig_ts.canvas.manager.set_window_title("2 / 2  |  Electrical and magnetic time series")
    fig_ts.subplots_adjust(left=0.07, right=0.62, top=0.82, bottom=0.12)
    ax_v, ax_h = ax_i.twinx(), ax_i.twinx()
    ax_h.spines["right"].set_position(("axes", 1.14))
    ax_i.set(xlabel="Time after gate command [ms]", ylabel="Current [kA]", xlim=(0, time_ms[-1]))
    ax_v.set_ylabel("Voltage [V]")
    ax_h.set_ylabel("|H| [MA/m]")
    ax_i.grid(alpha=0.18)
    fig_ts.suptitle("Discharge waveforms", x=0.07, ha="left", y=0.96)
    fig_ts.text(0.07, 0.90, "Click any checkbox to show/hide a trace. Currents overlap while the diode is off.\n"
                "Magnetic statistics cover the displayed z=0 slice; axes retain their own physical units.",
                fontsize=10, va="top")
    if pulse.clamp_time is not None:
        ax_i.axvspan(pulse.clamp_time*1e3, pulse.cutoff_time*1e3,
                      facecolor="#e8edf1", alpha=0.55)
    cursor = ax_i.axvline(slider.val, color="#777777", linewidth=0.8, linestyle=":")
    records = [
        ("Capacitor current", ax_i, pulse.states[:, 1]/1e3, "#245a81", "-", True),
        ("Coil current", ax_i, pulse.states[:, 2]/1e3, "#bb5921", "--", True),
        ("SCR / bridge current", ax_i, pulse.states[:, 2]/1e3, "#494949", ":", False),
        ("Capacitor terminal voltage", ax_v, pulse.bus_voltage, "#3c8270", "-", True),
        ("Coil voltage", ax_v, pulse.coil_voltage, "#b03047", "--", False),
        ("SCR / bridge voltage", ax_v, pulse.bridge_voltage, "#705689", ":", False),
        ("Storage voltage (inside ESR)", ax_v, pulse.states[:, 0], "#168da2", "-.", False),
    ]
    field_names = ("Minimum |H|", "Maximum |H|", "Median |H|", "Average |H|", "Center |H|")
    colors = ("#6d822b", "#b43b33", "#865892", "#bd790e", "#343c86")
    styles = (":", "--", "-.", "-", ":")
    for j, (name, color, style) in enumerate(zip(field_names, colors, styles)):
        records.append((name, ax_h, statistics[mode[0]][:, j]/1e6, color, style, j == 3))
    lines = {}
    for name, axes, values, color, style, visible in records:
        line, = axes.plot(time_ms, values, color=color, linestyle=style,
                          linewidth=1.6, visible=visible, label=name)
        lines[name] = line
    controls = CheckButtons(fig_ts.add_axes((0.785, 0.23, 0.207, 0.58)),
                            [item[0] for item in records], [item[5] for item in records])
    controls.ax.set_title("Visible series", fontsize=11, loc="left", pad=9)
    for label, item in zip(controls.labels, records):
        label.set_color(item[3])
        label.set_fontsize(9)
    for spine in controls.ax.spines.values():
        spine.set_visible(False)
    fig_ts.text(0.785, 0.11, "Bridge current = current in EACH\nconducting SCR, not their sum.\n\nShading: DC-bus freewheel interval.",
                fontsize=9, color="#444444")
    field_note = fig_ts.text(0.07, 0.025, "", fontsize=9, color="#7a3800")
    contours = [None]
    shown_field = [np.zeros((c.slice_pixels, c.slice_pixels))]
    shown_hz = [np.zeros((c.slice_pixels, c.slice_pixels))]

    def rescale():
        for axes in (ax_i, ax_v, ax_h):
            axes.relim(visible_only=True)
            axes.autoscale_view(scalex=False, scaley=True)

    def update_map(value):
        coefficients = model.drive_slice if mode[0] == "Driven |H|" else model.recoil_slice
        static = 0 if mode[0] == "Driven |H|" else model.self_slice
        current = np.array([np.interp(value, time_ms, currents[:, k]) for k in range(model.modes)])
        signed = (coefficients @ current+static).reshape(c.slice_pixels, c.slice_pixels)/1e6
        shown_hz[0] = signed
        field = np.abs(signed)
        shown_field[0] = field
        image.set_data(field)
        if contours[0] is not None:
            # ContourSet.remove() also removes the labels created by clabel().
            contours[0].remove()
        # Fixed levels in physical units, not percentile clipping; no contour
        # call for the zero-field frame or levels outside the current frame.
        levels = np.unique(np.r_[np.linspace(0, image.norm.vmax, 9)[1:-1], c.target_h/1e6])
        levels = levels[(levels > field.min()+1e-12) & (levels < field.max()-1e-12)]
        contours[0] = None
        if len(levels):
            contours[0] = ax_map.contour(coord, coord, field, levels=levels,
                                         colors="white", linewidths=0.8)
            outline = [path_effects.Stroke(linewidth=1.7, foreground="#303030"),
                       path_effects.Normal()]
            contours[0].set_path_effects(outline)
            labels = ax_map.clabel(contours[0], inline=True, fontsize=8, fmt="%.2f")
            for label in labels:
                label.set_path_effects(outline)
        label = "coil + magnet eddies; excludes static remanence" if mode[0] == "Driven |H|" else (
            "includes frozen remanence + mean recoil; NO irreversible reversal")
        subtitle.set_text(f"{mode[0]}: {label}\nTime = {value:.4f} ms   |   coil = "
                          f"{current[0]:.1f} A   |   slice mean = {field.mean():.3f} MA/m")
        ax_map.set_title(f"min {field.min():.3f}   max {field.max():.3f}   center "
                         f"{field[c.slice_pixels//2, c.slice_pixels//2]:.3f} MA/m", fontsize=10, pad=9)
        cursor.set_xdata([value, value])
        for figure in (fig_map, fig_ts):
            if plt.fignum_exists(figure.number):
                figure.canvas.draw_idle()

    def select_field(label):
        mode[0] = label
        data = statistics[label]
        image.set_clim(0, float(data[:, 1].max()/1e6))
        for k, name in enumerate(field_names):
            lines[name].set_ydata(data[:, k]/1e6)
        field_note.set_text("Magnetic field: "+label+". "+
                            ("Excludes static remanence; current is a linear-core reference."
                             if label == "Driven |H|" else
                             "Extrapolation only: the supplied sheet cannot determine an N52 reversal trajectory."))
        rescale()
        update_map(slider.val)

    def toggle(name):
        line = lines[name]
        line.set_visible(not line.get_visible())
        rescale()
        fig_ts.canvas.draw_idle()

    def format_hover(x, y):
        kx = int(np.clip(np.searchsorted(coord, x), 0, c.slice_pixels-1))
        ky = int(np.clip(np.searchsorted(coord, y), 0, c.slice_pixels-1))
        return (f"x={coord[kx]:.3f} mm, y={coord[ky]:.3f} mm, "
                f"|H|={shown_field[0][ky, kx]:.5f}, Hz={shown_hz[0][ky, kx]:+.5f} MA/m")

    ax_map.format_coord = format_hover
    slider.on_changed(update_map)
    radio.on_clicked(select_field)
    controls.on_clicked(toggle)
    peak_button.on_clicked(lambda _: slider.set_val(time_ms[np.argmax(statistics[mode[0]][:, 3])]))
    select_field(mode[0])
    # Keep widget objects alive, and expose only a small API for regression tests.
    handles = {"slider": slider, "field": radio, "series": controls, "lines": lines,
               "image": image, "peak": peak_button, "rescale": rescale}
    fig_map._pulse_controls = handles
    fig_ts._pulse_controls = handles
    return fig_map, fig_ts


def self_check():
    """Independent analytic checks, mesh refinement, energy/KVL and GUI callbacks.

    This is verification of the stated equations, NOT validation against hardware.
    --check uses Agg and closes both figures; it does not open extra plot windows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, width = 0.01, 0.005
    numerical = square_sheet_field(np.zeros((1, 3)), a, width)[0, 2]
    analytical = quad(lambda z: center_filament_h(a, z), -width/2, width/2, epsabs=1e-10)[0]/width
    np.testing.assert_allclose(numerical, analytical, rtol=2e-12)
    thin = square_sheet_field(np.zeros((1, 3)), a, 1e-7)[0, 2]
    np.testing.assert_allclose(thin, sqrt(2)/(pi*a), rtol=1e-10)
    # Independent two-dimensional integration of the 1/r Neumann kernel.
    x, wx = gauss_interval(64, -0.01, 0.01)
    y, wy = gauss_interval(64, -0.013, 0.013)
    direct = MU0/(4*pi)*np.sum(wx[:, None]*wy[None, :]
                                /np.sqrt((x[:, None]-y[None, :])**2+0.004**2))
    np.testing.assert_allclose(parallel_partial(0.01, 0.013, 0.004), direct, rtol=1e-10)
    print("PASS: finite-width center, thin-filament limit, independent Neumann integral.")

    c = Config(slice_pixels=51, time_samples=901)
    model = make_model(c)
    expected_length = 8*(c.turns*(c.magnet_side/2+c.inner_clearance+c.copper_thickness/2)
                         +(c.copper_thickness+c.kapton_thickness)*c.turns*(c.turns-1)/2)
    np.testing.assert_allclose(model.length, expected_length, rtol=1e-14)
    refined = make_model(replace(c, radial_quadrature=10, inductance_quadrature=140))
    change = abs(refined.inductance[0, 0]/model.inductance[0, 0]-1)
    if change > 0.005:
        raise AssertionError(f"Coil inductance is not converged: {change:.2%}")
    np.testing.assert_allclose(model.drive_slice.reshape(51, 51, -1),
                               model.drive_slice.reshape(51, 51, -1).transpose(1, 0, 2), rtol=1e-9, atol=1e-8)
    print(f"PASS: geometry, x/y symmetry, passive L; inductance refinement {change:.4%}.")

    pulse = simulate(model)
    E0 = 0.5*c.capacitance*c.initial_voltage**2
    error = np.max(np.abs(pulse.energy_error))
    if error > E0*2e-6:
        raise AssertionError(f"Energy does not balance: {error:g} J")
    np.testing.assert_allclose(pulse.bus_voltage, pulse.coil_voltage+pulse.bridge_voltage
                               +c.wiring_resistance*pulse.states[:, 2], atol=1e-8)
    charging = pulse.phase == 1
    np.testing.assert_allclose(pulse.states[charging, 1], pulse.states[charging, 2], atol=1e-7)
    assert pulse.clamp_time is not None
    assert pulse.bus_voltage.min() > -1e-6
    assert pulse.states[:, 2].min() >= -1e-8
    print(f"PASS: KVL, branch currents, ideal-diode polarity, energy residual {error:.3e} J.")

    # An independent damped series-RLC check BEFORE the diode turns on.
    simple = make_model(replace(c, eddy_rings=0, recoil_mu_r=1, heat_copper=False))
    reference = simulate(simple)
    L = simple.inductance[0, 0]+c.cap_esl
    R = simple.r20+c.cap_esr+c.wiring_resistance+2*c.scr_slope
    alpha = R/(2*L)
    omega = sqrt(1/(L*c.capacitance)-alpha*alpha)
    t = reference.time-c.gate_delay
    select = (reference.phase == 1) & (t >= 0)
    t = t[select]
    exact_i = (c.initial_voltage-2*c.scr_threshold)/(L*omega)*np.exp(-alpha*t)*np.sin(omega*t)
    np.testing.assert_allclose(reference.states[select, 2], exact_i, rtol=2e-6, atol=2e-5)
    print("PASS: diode-off waveform against independent analytical RLC solution.")

    more_rings = make_model(replace(c, eddy_rings=10))
    pulse_fine = simulate(more_rings)
    peak_change = abs(pulse_fine.states[:, 2].max()/pulse.states[:, 2].max()-1)
    if peak_change > 0.03:
        raise AssertionError(f"Reduced eddy model not radially converged: {peak_change:.2%}")
    print(f"PASS: 6 -> 10 magnet rings changes peak current by {peak_change:.3%} "
          "(not a check of the omitted 3-D modes).")

    statistics = field_statistics(model, pulse)
    figures = make_figures(model, pulse, statistics)
    assert len(plt.get_fignums()) == 2
    ui = figures[0]._pulse_controls
    for value in (0, pulse.time[np.argmax(pulse.states[:, 2])]*1e3, pulse.time[-1]*1e3):
        ui["slider"].set_val(value)
        for fig in figures:
            fig.canvas.draw()
        assert np.all(np.isfinite(ui["image"].get_array()))
    for k, name in enumerate(ui["lines"]):
        original = ui["lines"][name].get_visible()
        ui["series"].set_active(k)
        assert ui["lines"][name].get_visible() != original
        ui["series"].set_active(k)
    ui["field"].set_active(1)
    np.testing.assert_allclose(ui["lines"]["Average |H|"].get_ydata(),
                               statistics["Recoil-only |H|"][:, 3]/1e6)
    for fig in figures:
        fig.canvas.draw()
        plt.close(fig)
    print("PASS: exactly two figures, time slider, all selectable series and linked field mode.")
    print("All checks passed. Material reversal and component pulse ratings remain unvalidated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("Source ledger")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Run embedded numerical and widget checks.")
    parser.add_argument("--no-gui", action="store_true", help="Print reference results without opening figures.")
    args = parser.parse_args()
    if args.check:
        self_check()
        return
    model = make_model(Config())
    pulse = simulate(model)
    statistics = field_statistics(model, pulse)
    if np.max(np.abs(pulse.energy_error)) > 2e-6*0.5*model.cfg.capacitance*model.cfg.initial_voltage**2:
        raise ArithmeticError("Energy balance failed; refusing to show potentially misleading plots.")
    print_report(model, pulse, statistics)
    if not args.no_gui:
        import matplotlib
        import matplotlib.pyplot as plt
        if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template", "pgf"):
            raise RuntimeError("A GUI backend is required. Run on a desktop with Tk/Qt; "
                               "use --no-gui for calculations only.")
        make_figures(model, pulse, statistics)
        plt.show()


if __name__ == "__main__":
    main()
