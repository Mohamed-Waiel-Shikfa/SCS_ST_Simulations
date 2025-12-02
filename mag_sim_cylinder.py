import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle
from scipy.special import ellipk, ellipe

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================

# Simulation Control
SIM_TOTAL_TIME_US = 45.0
PULSE_WIDTH_US = 30.0
TIME_STEP_US = 1
ANIMATION_FRAMES = 200      # Reduced slightly to ensure smooth framerate with heavy calc

# Assembly Parameters
NUM_CYLINDERS = 2
MAGS_PER_CYL = 8
TOTAL_MAGS = NUM_CYLINDERS * MAGS_PER_CYL
CYL_RADIUS_MM = 12.0
CYL_GAP_MM = 0.5

# Magnet Parameters (Alnico 5)
MAG_DIAMETER_MM = 4.75
MAG_LENGTH_MM = 12.5
MAG_LEN_M = MAG_LENGTH_MM / 1000.0
MAG_RADIUS_M = (MAG_DIAMETER_MM / 2.0) / 1000.0

# Hysteresis Properties
MAG_BR_T = 1.35
MAG_HC_A_M = 50000.0
MAG_SAT_M = (1.35 / (4 * np.pi * 1e-7))

# Circuit Parameters
LAYERS = 3
TURNS_PER_LAYER = 125
TOTAL_TURNS = LAYERS * TURNS_PER_LAYER
COIL_RESISTANCE = 4.0
CAPACITANCE = 1e-5
VOLTAGE_INIT = 30.0
DIODE_DROP = 0.7

MU0 = 4 * np.pi * 1e-7

# ==========================================
# 2. PHYSICS MODELS (Field Math)
# ==========================================

def loop_field(r, z, R, I):
    """Calculate B-field (Br, Bz) of a single current loop off-axis."""
    epsilon = 1e-9
    r = np.atleast_1d(r)
    z = np.atleast_1d(z)

    alpha = r / R
    beta = z / R
    gamma = z / (r + epsilon)

    Q = (1 + alpha)**2 + beta**2
    k2 = (4 * alpha) / Q
    k2 = np.clip(k2, 0, 0.999999)

    K = ellipk(k2)
    E = ellipe(k2)

    C = (MU0 * I) / (2 * np.pi * R * np.sqrt(Q))

    term1 = (1 - alpha**2 - beta**2) / ((1 - alpha)**2 + beta**2)
    Bz = C * (E * term1 + K)

    # Simple Br for now to save compute (we focus on Bz for heatmap)
    Br = np.zeros_like(r)
    # (Full Br calculation omitted for speed in pre-calc, as Bz is dominant for polarity vis)

    return Br, Bz

def get_magnet_field_on_grid(mag_idx, centers, angles, X_grid, Y_grid):
    """
    Compute the Bz field contribution of one magnet on the global 2D grid.
    Rotates the grid points into the magnet's local frame.
    """
    mx, my = centers[mag_idx]
    theta = angles[mag_idx]

    # 1. Shift Grid to Magnet Center
    dx = X_grid - mx
    dy = Y_grid - my

    # 2. Rotate Grid to Magnet Frame (Magnet Axis along X')
    # Local X' is axial, Local Y' is radial
    # Rotate by -theta
    c, s = np.cos(-theta), np.sin(-theta)
    x_local = dx * c - dy * s
    y_local = dx * s + dy * c

    # 3. Compute Field in Local Frame (Treating x_local as Z_axial, y_local as R_radial)
    # We approximate the magnet as a solenoid current sheet
    # I_equiv for 1 Tesla Magnetization = L/mu0
    I_equiv = (1.0 / MU0) * MAG_LEN_M

    # Discretize length
    slices = 10 # Lower resolution for matrix pre-calc speed
    dI = I_equiv / slices

    Bz_local_acc = np.zeros_like(X_grid)

    z_positions = np.linspace(-MAG_LEN_M/2, MAG_LEN_M/2, slices)

    # We take absolute value of y_local because calculating Br/Bz depends on distance from axis (R)
    r_local = np.abs(y_local)

    for z_pos in z_positions:
        dz = x_local - z_pos
        _, bz = loop_field(r_local, dz, MAG_RADIUS_M, dI)
        Bz_local_acc += bz

    return Bz_local_acc

# ==========================================
# 3. ASSEMBLY & PRE-CALCULATION
# ==========================================

class MagnetAssembly:
    def __init__(self):
        self.centers = np.zeros((TOTAL_MAGS, 2))
        self.angles = np.zeros(TOTAL_MAGS)
        self.polarity_sign = np.zeros(TOTAL_MAGS)

        R_center = CYL_RADIUS_MM + MAG_LENGTH_MM/2
        R_outer = CYL_RADIUS_MM + MAG_LENGTH_MM
        c1_x = -(R_outer + CYL_GAP_MM/2)
        c2_x = +(R_outer + CYL_GAP_MM/2)

        for i in range(TOTAL_MAGS):
            cyl_idx = i // MAGS_PER_CYL
            mag_idx = i % MAGS_PER_CYL
            theta_step = 2 * np.pi / MAGS_PER_CYL

            if cyl_idx == 0: # Left
                theta = mag_idx * theta_step
                cx, cy = c1_x, 0
                pat = 1 if mag_idx % 2 == 0 else -1
            else: # Right
                theta = np.pi + mag_idx * theta_step
                cx, cy = c2_x, 0
                pat = -1 if mag_idx % 2 == 0 else 1

            self.centers[i] = [cx + R_center * np.cos(theta), cy + R_center * np.sin(theta)]
            self.angles[i] = theta
            self.polarity_sign[i] = pat

assembly = MagnetAssembly()

# --- PRE-CALCULATION OF FIELD MAPS ---
print("--- Pre-calculating Global Field Maps (This might take a moment) ---")

# Define Global Grid (mm)
GRID_W, GRID_H = 60, 30
RES_X, RES_Y = 100, 60 # Resolution
x_vec = np.linspace(-GRID_W, GRID_W, RES_X)
y_vec = np.linspace(-GRID_H, GRID_H, RES_Y)
X_grid, Y_grid = np.meshgrid(x_vec, y_vec) # coords in mm

# Convert to meters for physics
X_grid_m = X_grid / 1000.0
Y_grid_m = Y_grid / 1000.0
centers_m = assembly.centers / 1000.0

# Store unit map for each magnet: Shape (16, RES_Y, RES_X)
UNIT_MAPS = np.zeros((TOTAL_MAGS, RES_Y, RES_X))

for i in range(TOTAL_MAGS):
    # Calculate map for Magnet i with M = 1T
    UNIT_MAPS[i] = get_magnet_field_on_grid(i, centers_m, assembly.angles, X_grid_m, Y_grid_m)

# Calculate COIL unit maps as well?
# We assume coils are wrapped around magnets.
# So Coil Field ~ Magnet Field shape (approx) but proportional to Current I.
# We can reuse UNIT_MAPS scaled by (mu0 * N * I / L) / (M_unit) logic.
# Scaling Factor: M_eff_coil = (N*I/L) * mu0.
# Since UNIT_MAPS is for M=1T (mu0*H=1),
# Coil Contribution = UNIT_MAPS * (N*I/L * mu0)

print("--- Computing Coupling Matrices ---")
# Re-using the simpler dipole/projection logic for the physics loop G-matrix
# to ensure stability, distinct from the heatmap visualization.

def get_field_at_point_physics(x, y, source_idx):
    # Simplified interaction for physics loop (dipole approx)
    sx, sy = centers_m[source_idx]
    stheta = assembly.angles[source_idx]
    dx, dy = x - sx, y - sy

    # Rotate to source frame
    c, s = np.cos(-stheta), np.sin(-stheta)
    dx_l, dy_l = dx*c - dy*s, dx*s + dy*c
    r = np.sqrt(dx_l**2 + dy_l**2)
    if r < 1e-6: return 0, 0

    vol = np.pi * MAG_RADIUS_M**2 * MAG_LEN_M
    m = 1.0 * vol # Unit M
    fac = (MU0 * m) / (4 * np.pi * r**3)
    bx = fac * (3*(dx_l/r)**2 - 1)
    by = fac * (3*(dx_l/r)*(dy_l/r))

    # Rotate back
    cb, sb = np.cos(stheta), np.sin(stheta)
    return bx*cb - by*sb, bx*sb + by*cb

G_matrix = np.zeros((TOTAL_MAGS, TOTAL_MAGS))
for i in range(TOTAL_MAGS):
    ix_dir = np.cos(assembly.angles[i])
    iy_dir = np.sin(assembly.angles[i])
    for j in range(TOTAL_MAGS):
        if i == j: continue
        bx, by = get_field_at_point_physics(centers_m[i,0], centers_m[i,1], j)
        G_matrix[i,j] = (bx * ix_dir + by * iy_dir) / MU0

# ==========================================
# 4. PHYSICS LOOP
# ==========================================

class SystemHysteresis:
    def __init__(self, count):
        self.count = count
        self.Ms = MAG_SAT_M
        self.Hc = MAG_HC_A_M
        self.M = np.zeros(count)
        self.k = 5.0 / self.Hc
        for k in range(count):
            self.M[k] = (MAG_BR_T / MU0) * assembly.polarity_sign[k]

    def update(self, H_coils):
        H_int = G_matrix @ self.M
        H_total = H_coils + H_int
        M_up = self.Ms * np.tanh(self.k * (H_total + self.Hc))
        M_down = self.Ms * np.tanh(self.k * (H_total - self.Hc))
        self.M = np.clip(self.M, M_down, M_up)
        return self.M, H_int

print("--- Running Simulation ---")

sys_model = SystemHysteresis(TOTAL_MAGS)
coil_polarity_vec = -1.0 * assembly.polarity_sign

t_eval = np.arange(0, SIM_TOTAL_TIME_US, TIME_STEP_US) * 1e-6
results_I = []
results_V = []
results_M_0 = []
results_M_1 = []
results_Hint_0 = []

Q_cap = VOLTAGE_INIT * CAPACITANCE
I = 0.0
L_sys = (MU0 * (TOTAL_TURNS**2) * (np.pi*(MAG_RADIUS_M*1.1)**2)) / MAG_LEN_M # Single coil approx

for t in t_eval:
    t_us = t * 1e6
    dt = TIME_STEP_US * 1e-6
    V_cap = Q_cap / CAPACITANCE

    # Circuit
    if t_us < PULSE_WIDTH_US:
        dIdt = (V_cap - I * COIL_RESISTANCE) / L_sys
        dQdt = -I
    else:
        if I > 0:
            dIdt = (-I * COIL_RESISTANCE - DIODE_DROP) / L_sys
            dQdt = 0
        else:
            I = 0; dIdt = 0; dQdt = 0

    I += dIdt * dt
    Q_cap += dQdt * dt

    H_mag = (TOTAL_TURNS * I) / MAG_LEN_M
    H_coils = H_mag * coil_polarity_vec

    M_vec, H_int_vec = sys_model.update(H_coils)

    results_I.append(I)
    results_V.append(V_cap)
    results_M_0.append(M_vec[0])
    results_M_1.append(M_vec[1])
    results_Hint_0.append(H_int_vec[0])

results_I = np.array(results_I)
results_V = np.array(results_V)
results_M_0 = np.array(results_M_0)
results_M_1 = np.array(results_M_1)
results_Hint_0 = np.array(results_Hint_0)
time_us = t_eval * 1e6

# ==========================================
# 5. VISUALIZATION (Old Format Restored)
# ==========================================

print("--- Initializing Visualization ---")

# Custom Colormap (Blue-Black-Red)
colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list('bk_bwr', colors, N=100)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3)

# --- Top Row: Massive Heatmap of 16-Magnet Assembly ---
ax_map = fig.add_subplot(gs[0, :])

# Initial Field Calculation (Summation)
# M_field = sum(UNIT_MAPS[i] * M[i])
# Coil_field = sum(UNIT_MAPS[i] * Coil_Factor[i])
# Total = M_field + Coil_field

# Note: Coil factor per magnet = (N*I/L * mu0) / 1T * polarity
coil_scale_factor = (MU0 * TOTAL_TURNS / MAG_LEN_M)

def compute_total_grid(idx):
    I_curr = results_I[idx]

    # Reconstruct M for all magnets based on symmetry assumption
    # (Mag 0 pattern or Mag 1 pattern)
    m0 = results_M_0[idx]
    m1 = results_M_1[idx]

    # Accumulator
    Total_Grid = np.zeros((RES_Y, RES_X))

    for i in range(TOTAL_MAGS):
        # 1. Magnet Contribution
        orig_sign = assembly.polarity_sign[i]
        M_val = m0 if orig_sign > 0 else m1

        # 2. Coil Contribution
        # Coil opposes original sign
        H_coil_val = (TOTAL_TURNS * I_curr / MAG_LEN_M) * (-orig_sign)
        M_coil_equiv = H_coil_val * MU0

        # Total "Source" strength (Magnet M + Equivalent Coil M)
        # Note: UNIT_MAPS are for M=1T. So we scale by Tesla.
        Source_T = (M_val * MU0) + M_coil_equiv

        Total_Grid += UNIT_MAPS[i] * Source_T

    return Total_Grid

B_init = compute_total_grid(0)
# Use Vmin/Vmax to keep Black at 0
mesh_map = ax_map.pcolormesh(X_grid, Y_grid, B_init, shading='auto', cmap=cm, vmin=-1.5, vmax=1.5)
fig.colorbar(mesh_map, ax=ax_map, label='Field Intensity [T]')

ax_map.set_title("16-EPM Magnetic Field Interactions (Real-time)", fontsize=14)
ax_map.set_aspect('equal')
ax_map.set_xlim(-60, 60)
ax_map.set_ylim(-30, 30)
ax_map.set_xlabel('X [mm]')
ax_map.set_ylabel('Y [mm]')

# Geometry Overlays
magnet_patches = []
for i in range(TOTAL_MAGS):
    cx, cy = assembly.centers[i]
    ang = assembly.angles[i]
    # Magnet Body
    rect = Rectangle((cx - MAG_LENGTH_MM/2, cy - MAG_DIAMETER_MM/2),
                     MAG_LENGTH_MM, MAG_DIAMETER_MM, angle=np.degrees(ang),
                     rotation_point='center', ec='white', lw=1, fc='gray', alpha=0.3)
    ax_map.add_patch(rect)
    # Pole Indicator (Split rectangle? Complex to rotate. Just colored border/fill)
    # We will update facecolor based on polarity
    magnet_patches.append(rect)

# --- Bottom Row: Stats & Dynamics ---

# 1. Circuit Dynamics
ax_dyn = fig.add_subplot(gs[1, 0])
l_I, = ax_dyn.plot([], [], 'r-', label='Current (A)', lw=2)
ax_dyn.set_xlim(0, SIM_TOTAL_TIME_US)
ax_dyn.set_ylim(-1, np.max(results_I)*1.2)
ax_dyn.set_xlabel('Time [us]')
ax_dyn.set_ylabel('Current [A]', color='r')
ax_dyn.grid(True, alpha=0.3)

ax_V = ax_dyn.twinx()
l_V, = ax_V.plot([], [], 'b--', label='Voltage (V)')
ax_V.set_ylim(0, VOLTAGE_INIT*1.1)
ax_V.set_ylabel('Voltage [V]', color='b')
ax_dyn.set_title("Circuit Dynamics")

# 2. Hysteresis Loop (Contact Magnet)
ax_hys = fig.add_subplot(gs[1, 1])
# Interaction Field vs Magnetization
ax_hys.plot(results_Hint_0/1000, results_M_0*MU0, 'k--', alpha=0.3, label='Trajectory')
pt_hys, = ax_hys.plot([], [], 'ro', markersize=8)
ax_hys.set_xlabel('Local Interaction H [kA/m]')
ax_hys.set_ylabel('Magnetization M [T]')
ax_hys.set_title("Hysteresis (Contact Magnet)")
ax_hys.grid(True)

# 3. Detailed Stats Box
ax_stats = fig.add_subplot(gs[1, 2])
ax_stats.axis('off')
stats_template = (
    "SYSTEM STATISTICS\n"
    "-----------------\n"
    "Time: {time:.1f} us\n"
    "Total Current: {curr:.1f} A\n"
    "Cap Voltage: {volt:.1f} V\n"
    "Energy Used: {en:.2f} J\n\n"
    "CONTACT MAGNET (#0):\n"
    "  Magnetization: {m0:.2f} T\n"
    "  Stray Field: {hint:.1f} kA/m\n\n"
    "NEIGHBOR MAGNET (#1):\n"
    "  Magnetization: {m1:.2f} T"
)
txt_stats = ax_stats.text(0.05, 0.95, "", transform=ax_stats.transAxes,
                          fontsize=11, family='monospace', verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

def update(frame):
    idx = int((frame / ANIMATION_FRAMES) * len(t_eval))
    if idx >= len(t_eval): idx = len(t_eval) - 1

    # 1. Update Heatmap
    B_grid = compute_total_grid(idx)
    mesh_map.set_array(B_grid.ravel())

    # 2. Update Magnet Colors (Red=North/+M, Blue=South/-M)
    m0 = results_M_0[idx]
    m1 = results_M_1[idx]

    for i in range(TOTAL_MAGS):
        orig_sign = assembly.polarity_sign[i]
        val = m0 if orig_sign > 0 else m1

        # Normalize -1.5T to 1.5T -> 0 to 1
        norm = (val * MU0 + 1.5) / 3.0
        c = plt.cm.bwr(np.clip(norm, 0, 1))
        magnet_patches[i].set_facecolor(c)
        magnet_patches[i].set_alpha(0.8) # Make solid

    # 3. Update Plots
    l_I.set_data(time_us[:idx], results_I[:idx])
    l_V.set_data(time_us[:idx], results_V[:idx])
    pt_hys.set_data([results_Hint_0[idx]/1000], [results_M_0[idx]*MU0])

    # 4. Stats
    en = 0.5 * CAPACITANCE * (VOLTAGE_INIT**2 - results_V[idx]**2)
    txt_stats.set_text(stats_template.format(
        time=time_us[idx], curr=results_I[idx], volt=results_V[idx], en=en,
        m0=results_M_0[idx]*MU0, hint=results_Hint_0[idx]/1000,
        m1=results_M_1[idx]*MU0
    ))

    return [mesh_map, l_I, l_V, pt_hys, txt_stats] + magnet_patches

ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, interval=80, blit=False)
plt.tight_layout()
plt.suptitle("16-EPM Multi-Body Simulation", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()
