import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.special import ellipk, ellipe

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================

# Simulation Control
SIM_TOTAL_TIME_US = 1000.0  # Total window 1ms
PULSE_WIDTH_US = 300.0      # Controller turns off at 300us
TIME_STEP_US = 1.0          # Time step for physics
ANIMATION_FRAMES = 50       # Frames for animation (kept moderate for speed)

# Magnet Parameters (Alnico 5)
MAG_DIAMETER_MM = 4.75
MAG_LENGTH_MM = 12.5
MAG_RADIUS = (MAG_DIAMETER_MM / 1000) / 2
MAG_LEN_M = MAG_LENGTH_MM / 1000

# Alnico 5 Hysteresis Properties
MAG_BR_T = 1.25
MAG_HC_A_M = 50000.0
MAG_SAT_M = (1.35 / (4 * np.pi * 1e-7))

# Coil Parameters
WIRE_DIAM_MM = 0.1
LAYERS = 3
TURNS_PER_LAYER = 125
TOTAL_TURNS = LAYERS * TURNS_PER_LAYER
COIL_RESISTANCE = 4.0

# Circuit Parameters
CAPACITANCE = 1e-3
VOLTAGE_INIT = 30.0
DIODE_DROP = 0.7

# Physical Constants
MU0 = 4 * np.pi * 1e-7

# ==========================================
# 2. PHYSICS MODELS (Vectorized)
# ==========================================

def calculate_inductance(N, R, L):
    """Estimate inductance using Wheeler/Nagaoka approx."""
    area = np.pi * R**2
    denom = L + 0.9 * R
    inductance = (MU0 * (N**2) * area) / denom
    return inductance

def loop_field(r, z, R, I):
    """Calculate B-field (Br, Bz) of a single current loop off-axis."""
    epsilon = 1e-9
    r = np.atleast_1d(r)
    z = np.atleast_1d(z)

    alpha = r / R
    beta = z / R

    Q = (1 + alpha)**2 + beta**2
    k2 = (4 * alpha) / Q
    k2 = np.clip(k2, 0, 0.999999)

    K = ellipk(k2)
    E = ellipe(k2)

    C = (MU0 * I) / (2 * np.pi * R * np.sqrt(Q))

    term1 = (1 - alpha**2 - beta**2) / ((1 - alpha)**2 + beta**2)
    Bz = C * (E * term1 + K)

    Br = np.zeros_like(r)
    mask = r > epsilon

    if np.any(mask):
        r_s = r[mask]
        z_s = z[mask]
        alpha_s = r_s / R
        beta_s = z_s / R
        Q_s = (1 + alpha_s)**2 + beta_s**2
        k2_s = (4 * alpha_s) / Q_s
        k2_s = np.clip(k2_s, 0, 0.999999)
        K_s = ellipk(k2_s)
        E_s = ellipe(k2_s)

        term_num = 2 * z_s * R
        term_den = np.sqrt((R + r_s)**2 + z_s**2)
        factor = (MU0 * I) / (2 * np.pi) * (z_s / (r_s * term_den))
        E_term = ((R**2 + r_s**2 + z_s**2) / ((R - r_s)**2 + z_s**2)) * E_s
        Br[mask] = factor * (-K_s + E_term)

    return Br, Bz

def solenoid_field_map(I, current_turns, coil_radius, coil_len, r_grid, z_grid):
    """Superposition of loops."""
    Bz_total = np.zeros_like(r_grid)
    Br_total = np.zeros_like(r_grid)

    slices = 40
    z_positions = np.linspace(-coil_len/2, coil_len/2, slices)
    dI = I / slices

    for z_pos in z_positions:
        dz = z_grid - z_pos
        br, bz = loop_field(r_grid, dz, coil_radius, dI)
        Bz_total += bz
        Br_total += br

    return Br_total, Bz_total

# --- Hysteresis Model ---
class MagnetHysteresis:
    def __init__(self, Ms, Hc):
        self.Ms = Ms
        self.Hc = Hc
        self.M = MAG_BR_T / MU0 # Start fully magnetized positive
        self.k = 5.0 / Hc

    def update(self, H_applied):
        # M_up and M_down for major loop envelope
        M_up = self.Ms * np.tanh(self.k * (H_applied + self.Hc))
        M_down = self.Ms * np.tanh(self.k * (H_applied - self.Hc))

        # Simple history dependent clamping
        if self.M > M_up:
            self.M = M_up
        elif self.M < M_down:
            self.M = M_down

        return self.M

# ==========================================
# 3. PRE-CALCULATION (Optimization)
# ==========================================
print("--- Pre-calculating Unit Field Maps ---")
# Define Grid
GRID_RES = 30 # Slightly lower res for smoother animation
z_vec = np.linspace(-0.02, 0.02, GRID_RES)
r_vec = np.linspace(0, 0.015, GRID_RES)
Z_grid, R_grid = np.meshgrid(z_vec, r_vec)

# 1. Coil Unit Map (1 Amp)
eff_coil_R = MAG_RADIUS + (WIRE_DIAM_MM/1000 * LAYERS)/2
Br_unit_coil, Bz_unit_coil = solenoid_field_map(1.0, TOTAL_TURNS, eff_coil_R, MAG_LEN_M, R_grid, Z_grid)

# 2. Magnet Unit Map (Equivalent Current for 1 Tesla Magnetization)
I_equiv_unit = (1.0 / MU0) * MAG_LEN_M
Br_unit_mag, Bz_unit_mag = solenoid_field_map(I_equiv_unit, 50, MAG_RADIUS, MAG_LEN_M, R_grid, Z_grid)

# ==========================================
# 4. SIMULATION LOOP
# ==========================================
print("--- Running Circuit Physics ---")

L_COIL = calculate_inductance(TOTAL_TURNS, eff_coil_R, MAG_LEN_M)
magnet = MagnetHysteresis(MAG_SAT_M, MAG_HC_A_M)

t_eval = np.arange(0, SIM_TOTAL_TIME_US, TIME_STEP_US) * 1e-6
results_I = []
results_V = []
results_M = []
results_H = []

Q = VOLTAGE_INIT * CAPACITANCE
I = 0.0

for t in t_eval:
    t_us = t * 1e6
    dt = TIME_STEP_US * 1e-6
    V_cap = Q / CAPACITANCE

    # Circuit Equations
    if t_us < PULSE_WIDTH_US:
        dIdt = (V_cap - I * COIL_RESISTANCE) / L_COIL
        dQdt = -I
    else:
        # Flyback / Dissipation
        if I > 0:
            dIdt = (-I * COIL_RESISTANCE - DIODE_DROP) / L_COIL
            dQdt = 0
        else:
            I = 0; dIdt = 0; dQdt = 0

    I += dIdt * dt
    Q += dQdt * dt

    # Magnet Physics
    # OPPOSITE POLARITY: Coil current generates H in negative Z direction
    H_coil_center = -1.0 * (TOTAL_TURNS * I) / MAG_LEN_M

    M_curr = magnet.update(H_coil_center)

    results_I.append(I)
    results_V.append(V_cap)
    results_M.append(M_curr * MU0) # Store as Tesla
    results_H.append(H_coil_center)

results_I = np.array(results_I)
results_V = np.array(results_V)
results_M = np.array(results_M)
results_H = np.array(results_H)
time_us = t_eval * 1e6

peak_I = np.max(results_I)
peak_time = time_us[np.argmax(results_I)]
energy_init = 0.5 * CAPACITANCE * VOLTAGE_INIT**2
energy_final = 0.5 * CAPACITANCE * results_V[-1]**2
energy_used = energy_init - energy_final

print(f"Peak Current: {peak_I:.2f} A")
print(f"Magnet Flip: {results_M[0]:.2f}T -> {results_M[-1]:.2f}T")

# ==========================================
# 5. ANIMATION SETUP
# ==========================================
print("--- Initializing Animation ---")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3)

# --- Top Row: Field Visualizations ---
ax_coil = fig.add_subplot(gs[0, 0])
ax_mag = fig.add_subplot(gs[0, 1])
ax_comb = fig.add_subplot(gs[0, 2])

# Setup Heatmaps (Magnitude)
# Using 'inferno' for high contrast magnitude map
kw = dict(shading='auto', cmap='inferno', vmin=0, vmax=1.5)

# Initial State
# Coil
factor_coil = -results_I[0]
Br_c = Br_unit_coil * factor_coil
Bz_c = Bz_unit_coil * factor_coil
Bmag_c = np.sqrt(Br_c**2 + Bz_c**2)
mesh_coil = ax_coil.pcolormesh(Z_grid*1000, R_grid*1000, Bmag_c, **kw)
ax_coil.set_title("Coil Field Magnitude")

# Magnet
Br_m = Br_unit_mag * results_M[0]
Bz_m = Bz_unit_mag * results_M[0]
Bmag_m = np.sqrt(Br_m**2 + Bz_m**2)
mesh_mag = ax_mag.pcolormesh(Z_grid*1000, R_grid*1000, Bmag_m, **kw)
ax_mag.set_title("Magnet Field Magnitude")

# Combined
Br_t = Br_c + Br_m
Bz_t = Bz_c + Bz_m
Bmag_t = np.sqrt(Br_t**2 + Bz_t**2)
mesh_comb = ax_comb.pcolormesh(Z_grid*1000, R_grid*1000, Bmag_t, **kw)
ax_comb.set_title("Combined Field Magnitude")

# Add colorbar for reference
fig.colorbar(mesh_comb, ax=ax_comb, label='Field Strength [T]')

# Geometry Overlays
for ax in [ax_coil, ax_mag, ax_comb]:
    ax.add_patch(plt.Rectangle((-MAG_LENGTH_MM/2, 0), MAG_LENGTH_MM, MAG_RADIUS*1000,
                 ec='cyan', fc='none', lw=1.5, ls='-'))
    ax.add_patch(plt.Rectangle((-MAG_LENGTH_MM/2, MAG_RADIUS*1000), MAG_LENGTH_MM, 2,
                 ec='orange', fc='none', ls='--', lw=1))
    ax.set_aspect('equal')
    ax.set_xlabel('Z [mm]')
    ax.set_ylabel('R [mm]')

# --- Bottom Row: Stats & Dynamics ---

# 1. Circuit Dynamics
ax_dyn = fig.add_subplot(gs[1, 0])
line_I, = ax_dyn.plot([], [], 'r-', label='Current (A)', linewidth=2)
ax_dyn.set_xlim(0, SIM_TOTAL_TIME_US)
ax_dyn.set_ylim(-1, peak_I * 1.2)
ax_dyn.set_xlabel('Time [$\mu$s]')
ax_dyn.set_ylabel('Current [A]', color='r')
ax_dyn.grid(True, alpha=0.3)

ax_volt = ax_dyn.twinx()
line_V, = ax_volt.plot([], [], 'b--', label='Voltage (V)')
ax_volt.set_ylim(0, VOLTAGE_INIT * 1.1)
ax_volt.set_ylabel('Voltage [V]', color='b')
ax_dyn.set_title("Circuit Dynamics")

# 2. Hysteresis Loop
ax_hys = fig.add_subplot(gs[1, 1])
# Static plot of the full trajectory
ax_hys.plot(results_H/1000, results_M, color='gray', linestyle='--', alpha=0.5)
# Moving dot
point_hys, = ax_hys.plot([], [], 'ro', markersize=8)
ax_hys.set_xlabel('Applied Field H [kA/m]')
ax_hys.set_ylabel('Magnetization M [T]')
ax_hys.set_title("Magnet Hysteresis Trajectory")
ax_hys.grid(True)
ax_hys.axvline(0, color='k', linewidth=0.5)
ax_hys.axhline(0, color='k', linewidth=0.5)

# 3. Stats Box
ax_stats = fig.add_subplot(gs[1, 2])
ax_stats.axis('off')
stats_template = (
    "SIMULATION STATISTICS\n"
    "---------------------\n"
    "Time: {time:.1f} $\mu$s\n\n"
    "Current (I): {curr:.2f} A\n"
    "Voltage (V): {volt:.1f} V\n"
    "Pulse Energy: {energy:.2f} J\n\n"
    "Magnet State:\n"
    "  H_applied: {happ:.1f} kA/m\n"
    "  Magnetization: {mag:.2f} T\n"
    "  Polarity: {pol}\n\n"
    "Status: {status}"
)
stats_text = ax_stats.text(0.05, 0.95, "", transform=ax_stats.transAxes,
                          fontsize=11, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

def update(frame):
    # Determine index
    idx = int((frame / ANIMATION_FRAMES) * len(t_eval))
    if idx >= len(t_eval): idx = len(t_eval) - 1

    t_now = time_us[idx]
    curr_I = results_I[idx]
    curr_V = results_V[idx]
    curr_M = results_M[idx]
    curr_H = results_H[idx]

    # 1. Update Heatmaps
    # Coil
    factor_coil = -curr_I
    Br_c = Br_unit_coil * factor_coil
    Bz_c = Bz_unit_coil * factor_coil
    Bmag_c = np.sqrt(Br_c**2 + Bz_c**2)
    mesh_coil.set_array(Bmag_c.ravel())

    # Magnet
    Br_m = Br_unit_mag * curr_M
    Bz_m = Bz_unit_mag * curr_M
    Bmag_m = np.sqrt(Br_m**2 + Bz_m**2)
    mesh_mag.set_array(Bmag_m.ravel())

    # Combined
    Br_t = Br_c + Br_m
    Bz_t = Bz_c + Bz_m
    Bmag_t = np.sqrt(Br_t**2 + Bz_t**2)
    mesh_comb.set_array(Bmag_t.ravel())

    # 2. Update Dynamics Lines
    line_I.set_data(time_us[:idx], results_I[:idx])
    line_V.set_data(time_us[:idx], results_V[:idx])

    # 3. Update Hysteresis Dot
    point_hys.set_data([curr_H/1000], [curr_M])

    # 4. Update Stats
    status = "PULSE ACTIVE" if t_now < PULSE_WIDTH_US else "FLYBACK/DECAY"
    pol = "NORTH UP" if curr_M > 0 else "SOUTH UP"

    txt = stats_template.format(
        time=t_now,
        curr=curr_I,
        volt=curr_V,
        energy=0.5 * CAPACITANCE * (VOLTAGE_INIT**2 - curr_V**2),
        happ=curr_H/1000,
        mag=curr_M,
        pol=pol,
        status=status
    )
    stats_text.set_text(txt)

    return mesh_coil, mesh_mag, mesh_comb, line_I, line_V, point_hys, stats_text

ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, interval=50, blit=False)

plt.tight_layout()
plt.suptitle(f"EPM Simulation: 300$\mu$s Pulse", fontsize=16)
# Adjust subplots to make room for suptitle
plt.subplots_adjust(top=0.92)

plt.show()

print("Animation initialized with full statistics and restored heatmaps.")
