import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from scipy.special import ellipk, ellipe

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================

# Simulation Control
SIM_TOTAL_TIME_US = 100.0  # Total window 0.1ms
PULSE_WIDTH_US = 30.0      # Controller turns off at 30us
TIME_STEP_US = 0.5         # Finer step for better physics
ANIMATION_FRAMES = 50      # Frames for animation

# Magnet Parameters (Alnico 5)
MAG_DIAMETER_MM = 4.75
MAG_LENGTH_MM = 12.5
MAG_RADIUS = (MAG_DIAMETER_MM / 1000) / 2
MAG_LEN_M = MAG_LENGTH_MM / 1000

# Alnico 5 Hysteresis Properties
MAG_BR_T = 1.35
MAG_HC_A_M = 50000.0
MAG_SAT_M = (1.35 / (4 * np.pi * 1e-7))

# Coil Parameters
WIRE_DIAM_MM = 0.1
LAYERS = 3
TURNS_PER_LAYER = 125
TOTAL_TURNS = LAYERS * TURNS_PER_LAYER
COIL_RESISTANCE = 4.0

# Circuit Parameters
CAPACITANCE = 1e-5         # 10 uF
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

        # Explicit Br calculation
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

    # Increase slices for smoother near-field resolution
    slices = 50
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
        self.M = MAG_BR_T / MU0
        self.k = 5.0 / Hc

    def update(self, H_applied):
        # Update Magnetization based on Hysteresis loop envelope
        M_up = self.Ms * np.tanh(self.k * (H_applied + self.Hc))
        M_down = self.Ms * np.tanh(self.k * (H_applied - self.Hc))

        if self.M > M_up:
            self.M = M_up
        elif self.M < M_down:
            self.M = M_down

        return self.M

# ==========================================
# 3. PRE-CALCULATION & SETUP
# ==========================================
print("--- Pre-calculating Unit Field Maps ---")

# Grid: Adjusted to show full side view (North and South)
# Z from -20mm to +20mm (Magnet is 12.5mm long, so this shows surroundings)
GRID_RES = 30
z_vec = np.linspace(-0.02, 0.02, GRID_RES)
r_vec = np.linspace(0, 0.015, GRID_RES) # Radial symmetry, showing top half
Z_grid, R_grid = np.meshgrid(z_vec, r_vec)

# Quiver Grid (Decimated for cleaner arrows)
Q_RES = 12
z_q = np.linspace(-0.02, 0.02, Q_RES)
r_q = np.linspace(0, 0.015, Q_RES)
Z_q, R_q = np.meshgrid(z_q, r_q)

# 1. Coil Unit Maps (1 Amp)
eff_coil_R = MAG_RADIUS + (WIRE_DIAM_MM/1000 * LAYERS)/2
# For Heatmap
Br_unit_coil, Bz_unit_coil = solenoid_field_map(1.0, TOTAL_TURNS, eff_coil_R, MAG_LEN_M, R_grid, Z_grid)
# For Arrows
Br_q_coil, Bz_q_coil = solenoid_field_map(1.0, TOTAL_TURNS, eff_coil_R, MAG_LEN_M, R_q, Z_q)

# 2. Magnet Unit Maps (Equivalent Current for 1 Tesla Magnetization)
I_equiv_unit = (1.0 / MU0) * MAG_LEN_M
# For Heatmap
Br_unit_mag, Bz_unit_mag = solenoid_field_map(I_equiv_unit, 50, MAG_RADIUS, MAG_LEN_M, R_grid, Z_grid)
# For Arrows
Br_q_mag, Bz_q_mag = solenoid_field_map(I_equiv_unit, 50, MAG_RADIUS, MAG_LEN_M, R_q, Z_q)

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

Q_cap = VOLTAGE_INIT * CAPACITANCE
I = 0.0

for t in t_eval:
    t_us = t * 1e6
    dt = TIME_STEP_US * 1e-6
    V_cap = Q_cap / CAPACITANCE

    # Circuit
    if t_us < PULSE_WIDTH_US:
        dIdt = (V_cap - I * COIL_RESISTANCE) / L_COIL
        dQdt = -I
    else:
        # Flyback
        if I > 0:
            dIdt = (-I * COIL_RESISTANCE - DIODE_DROP) / L_COIL
            dQdt = 0
        else:
            I = 0; dIdt = 0; dQdt = 0

    I += dIdt * dt
    Q_cap += dQdt * dt

    # Magnet Physics (Opposing field)
    # Coil Current is defined as opposing the initial magnet polarity
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

# ==========================================
# 5. VISUALIZATION SETUP
# ==========================================
print("--- Initializing Visualization ---")

# Custom Colormap: Blue (South) -> Black (Zero) -> Red (North)
colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]  # B -> K -> R
n_bins = 100
cmap_name = 'bk_bwr'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3)

# --- Top Row: Field Visualizations (Bz Component for Polarity) ---
ax_coil = fig.add_subplot(gs[0, 0])
ax_mag = fig.add_subplot(gs[0, 1])
ax_comb = fig.add_subplot(gs[0, 2])

# Plot Settings
# Coil needs a separate, tighter scale to be visible
# Peak field ~0.25T, so +/- 0.3T scale makes it bright
kw_mesh_coil = dict(shading='auto', cmap=cm, vmin=-0.3, vmax=0.3)
# Magnet needs full scale (+/- 1.5T)
kw_mesh_mag = dict(shading='auto', cmap=cm, vmin=-1.5, vmax=1.5)

# Initial Meshes (Bz Component)
# Coil (Opposing)
Bz_c_init = Bz_unit_coil * (-results_I[0])
mesh_coil = ax_coil.pcolormesh(Z_grid*1000, R_grid*1000, Bz_c_init, **kw_mesh_coil)
ax_coil.set_title("Coil Field (Breath w/ Pulse)")

# Magnet
Bz_m_init = Bz_unit_mag * results_M[0]
mesh_mag = ax_mag.pcolormesh(Z_grid*1000, R_grid*1000, Bz_m_init, **kw_mesh_mag)
ax_mag.set_title("Magnet Field")

# Combined
Bz_t_init = Bz_c_init + Bz_m_init
mesh_comb = ax_comb.pcolormesh(Z_grid*1000, R_grid*1000, Bz_t_init, **kw_mesh_mag)
ax_comb.set_title("Combined Interaction")

# Initial Quivers (Arrows)
q_kw = dict(color='white', pivot='mid', scale=25, width=0.005)
Q_coil = ax_coil.quiver(Z_q*1000, R_q*1000, np.zeros_like(Z_q), np.zeros_like(R_q), **q_kw)
Q_mag = ax_mag.quiver(Z_q*1000, R_q*1000, np.zeros_like(Z_q), np.zeros_like(R_q), **q_kw)
Q_comb = ax_comb.quiver(Z_q*1000, R_q*1000, np.zeros_like(Z_q), np.zeros_like(R_q), **q_kw)

# Geometry & Magnet Face Visualization
# We use two rectangles per magnet: North Half (Red) and South Half (Blue)
# This visually indicates the "Poles" on the faces, independent of the field map
magnet_patches = {}

def create_magnet_patches(ax):
    # Top Half (Positive Z)
    r_top = Rectangle((0, 0), MAG_LENGTH_MM/2, MAG_RADIUS*1000, ec='none', alpha=0.6)
    # Bottom Half (Negative Z)
    r_bot = Rectangle((-MAG_LENGTH_MM/2, 0), MAG_LENGTH_MM/2, MAG_RADIUS*1000, ec='none', alpha=0.6)

    # Border
    r_border = Rectangle((-MAG_LENGTH_MM/2, 0), MAG_LENGTH_MM, MAG_RADIUS*1000,
                         ec='black', fc='none', lw=2)

    ax.add_patch(r_top)
    ax.add_patch(r_bot)
    ax.add_patch(r_border)
    return r_top, r_bot

for name, ax in zip(['coil', 'mag', 'comb'], [ax_coil, ax_mag, ax_comb]):
    # Magnet Patches (Pole Indicators)
    magnet_patches[name] = create_magnet_patches(ax)

    # Coil Body
    ax.add_patch(Rectangle((-MAG_LENGTH_MM/2, MAG_RADIUS*1000), MAG_LENGTH_MM, 2,
                 ec='orange', fc='none', ls='--', lw=1))

    ax.set_aspect('equal')
    ax.set_xlabel('Z [mm]')
    ax.set_ylabel('R [mm]')
    ax.set_xlim(-20, 20)

# Colorbars
fig.colorbar(mesh_coil, ax=ax_coil, label='$B_z$ [T] (Low Scale)')
fig.colorbar(mesh_comb, ax=ax_comb, label='$B_z$ [T] (Full Scale)')

# --- Bottom Row: Stats & Dynamics ---

# 1. Circuit Dynamics
ax_dyn = fig.add_subplot(gs[1, 0])
line_I, = ax_dyn.plot([], [], 'r-', label='Current (A)', linewidth=2)
ax_dyn.set_xlim(0, SIM_TOTAL_TIME_US)
ax_dyn.set_ylim(-1, max(peak_I * 1.2, 1))
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
ax_hys.plot(results_H/1000, results_M, color='gray', linestyle='--', alpha=0.5)
point_hys, = ax_hys.plot([], [], 'ro', markersize=8)
ax_hys.set_xlabel('H [kA/m]')
ax_hys.set_ylabel('M [T]')
ax_hys.set_title("Hysteresis Trajectory")
ax_hys.grid(True)

# 3. Stats Box
ax_stats = fig.add_subplot(gs[1, 2])
ax_stats.axis('off')
stats_template = (
    "STATISTICS\n"
    "----------\n"
    "Time: {time:.1f} $\mu$s\n"
    "Current: {curr:.2f} A\n"
    "Cap Volts: {volt:.1f} V\n"
    "Energy Used: {energy:.4f} J\n\n"
    "Magnet State:\n"
    "  H_app: {happ:.1f} kA/m\n"
    "  Mag (M): {mag:.2f} T\n"
    "  Polarity: {pol}"
)
stats_text = ax_stats.text(0.05, 0.95, "", transform=ax_stats.transAxes,
                          fontsize=11, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# --- Animation Update Function ---
def update(frame):
    idx = int((frame / ANIMATION_FRAMES) * len(t_eval))
    if idx >= len(t_eval): idx = len(t_eval) - 1

    # Values
    curr_I = results_I[idx]
    curr_V = results_V[idx]
    curr_M = results_M[idx]

    # 1. Update Heatmaps
    factor_coil = -curr_I

    Bz_c = Bz_unit_coil * factor_coil
    Bz_m = Bz_unit_mag * curr_M
    Bz_t = Bz_c + Bz_m

    mesh_coil.set_array(Bz_c.ravel())
    mesh_mag.set_array(Bz_m.ravel())
    mesh_comb.set_array(Bz_t.ravel())

    # 2. Update Quivers
    br_c_q = Br_q_coil * factor_coil
    bz_c_q = Bz_q_coil * factor_coil
    Q_coil.set_UVC(bz_c_q, br_c_q)

    br_m_q = Br_q_mag * curr_M
    bz_m_q = Bz_q_mag * curr_M
    Q_mag.set_UVC(bz_m_q, br_m_q)

    Q_comb.set_UVC(bz_c_q + bz_m_q, br_c_q + br_m_q)

    # 3. Update Magnet Face Colors (Visual Poles)
    # If M > 0 (North is +Z/Right): Right Half Red, Left Half Blue
    # If M < 0 (South is +Z/Right): Right Half Blue, Left Half Red
    # Note: In our grid, +Z is right.
    if curr_M > 0:
        c_right = 'red'  # North Face
        c_left = 'blue'  # South Face
        pol_str = "NORTH (Right)"
    else:
        c_right = 'blue' # South Face
        c_left = 'red'   # North Face
        pol_str = "SOUTH (Right)"

    for name in ['coil', 'mag', 'comb']:
        r_top, r_bot = magnet_patches[name]
        # In our patches, r_top is at +Z (Right), r_bot is at -Z (Left)
        r_top.set_facecolor(c_right)
        r_bot.set_facecolor(c_left)

    # 4. Update Lines
    line_I.set_data(time_us[:idx], results_I[:idx])
    line_V.set_data(time_us[:idx], results_V[:idx])
    point_hys.set_data([results_H[idx]/1000], [curr_M])

    # 5. Stats
    e_consumed = 0.5 * CAPACITANCE * (VOLTAGE_INIT**2 - curr_V**2)

    txt = stats_template.format(
        time=time_us[idx],
        curr=curr_I,
        volt=curr_V,
        energy=e_consumed,
        happ=results_H[idx]/1000,
        mag=curr_M,
        pol=pol_str
    )
    stats_text.set_text(txt)

    return mesh_coil, mesh_mag, mesh_comb, Q_coil, Q_mag, Q_comb, line_I, line_V, stats_text

ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, interval=80, blit=False)

plt.tight_layout()
plt.suptitle("EPM Polarity Switch (Red/Blue Blocks = Poles)", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()

print("Animation initialized. Magnet faces now colored Red/Blue to indicate poles.")
