import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle
from scipy.special import ellipk, ellipe

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================

# Time Settings (Microseconds for Pulse Physics)
FRAME_DURATION_US = 100.0   # Duration of one frame window
PULSE_WIDTH_US = 50.0       # Active pulse duration within frame
TIME_STEP_US = 1.0
ANIMATION_SKIP = 5          # Render every Nth frame

# Geometry
NUM_CYLINDERS = 2
MAGS_PER_CYL = 8
TOTAL_MAGS = NUM_CYLINDERS * MAGS_PER_CYL
CYL_RADIUS_MM = 8.0
CYL_GAP_MM = 0.5

# --- COIL CONTROL SEQUENCES ---
# Each frame defines the state of the 16 coils:
#  1 : Pulse North
# -1 : Pulse South
#  0 : OFF (Coil open circuit)
# Indices 0-7: Left Cylinder, 8-15: Right Cylinder

COIL_SEQUENCE = [
    np.array([
         0,  0,  0,  0,  0,  0,  0,  0,  # Left coils OFF
         0, 0,  0,  0,  0,  0,  0,  0   # Right: Only Mag 9 pulses South
    ]),

    # FRAME 0: INITIALIZATION
    # Fire ALL coils to set the robust "Attraction/Locked" pattern.
    np.array([
        # Left (0-7): N, S, N, S...
         1, -1,  1, -1,  1, -1,  1, -1,
        # Right (8-15): S, N, S, N... (Matches Left for attraction)
        -1,  1, -1,  1, -1,  1, -1,  1
    ]),

    np.array([
         0,  0,  0,  0,  0,  0,  0,  0,  # Left coils OFF
         1, -1,  0,  0,  0,  0,  0,  0   # Right: Only Mag 9 pulses South
    ]),

    np.array([
         0,  0,  0,  0,  0,  0,  0,  0,  # Left coils OFF
         0, 0,  0,  0,  0,  0,  0,  0   # Right: Only Mag 9 pulses South
    ]),

    np.array([
         0,  0,  0,  0,  0,  0,  0,  0,  # Left coils OFF
         -1, 1,  0,  0,  0,  0,  0,  0   # Right: Only Mag 9 pulses South
    ]),

    np.array([
         0,  0,  0,  0,  0,  0,  0,  0,  # Left coils OFF
         0, 0,  0,  0,  0,  0,  0,  0   # Right: Only Mag 9 pulses South
    ]),
]

# Total Simulation Time
SIM_TOTAL_TIME_US = len(COIL_SEQUENCE) * FRAME_DURATION_US
SIM_STEPS = int(SIM_TOTAL_TIME_US / TIME_STEP_US)

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
CAPACITANCE = 1e-3
VOLTAGE_INIT = 30.0
DIODE_DROP = 0.7

MU0 = 4 * np.pi * 1e-7

# ==========================================
# 2. PHYSICS MODELS (Field Math)
# ==========================================

def loop_field(r, z, R, I):
    """Calculate B-field (Bz) of a single current loop off-axis."""
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

    return Bz

def get_magnet_field_on_grid(mag_idx, centers, angles, X_grid, Y_grid):
    """Compute Bz field contribution of one magnet on global grid."""
    mx, my = centers[mag_idx]
    theta = angles[mag_idx]

    # Shift & Rotate to Magnet Frame
    dx = X_grid - mx
    dy = Y_grid - my
    c, s = np.cos(-theta), np.sin(-theta)
    x_local = dx * c - dy * s
    y_local = dx * s + dy * c

    # Compute Field
    I_equiv = (1.0 / MU0) * MAG_LEN_M
    slices = 8 # Optimized for speed
    dI = I_equiv / slices
    Bz_acc = np.zeros_like(X_grid)
    z_pos = np.linspace(-MAG_LEN_M/2, MAG_LEN_M/2, slices)
    r_local = np.abs(y_local)

    for z in z_pos:
        dz = x_local - z
        Bz_acc += loop_field(r_local, dz, MAG_RADIUS_M, dI)

    return Bz_acc

# ==========================================
# 3. ASSEMBLY & PRE-CALCULATION
# ==========================================

class MagnetAssembly:
    def __init__(self):
        self.centers = np.zeros((TOTAL_MAGS, 2))
        self.angles = np.zeros(TOTAL_MAGS)

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
            else: # Right
                theta = np.pi + mag_idx * theta_step
                cx, cy = c2_x, 0

            self.centers[i] = [cx + R_center * np.cos(theta), cy + R_center * np.sin(theta)]
            self.angles[i] = theta

assembly = MagnetAssembly()

# --- PRE-CALCULATION ---
print("--- Pre-calculating Global Field Maps ---")

GRID_W, GRID_H = 60, 30
RES_X, RES_Y = 100, 60
x_vec = np.linspace(-GRID_W, GRID_W, RES_X)
y_vec = np.linspace(-GRID_H, GRID_H, RES_Y)
X_grid, Y_grid = np.meshgrid(x_vec, y_vec)

X_grid_m = X_grid / 1000.0
Y_grid_m = Y_grid / 1000.0
centers_m = assembly.centers / 1000.0

UNIT_MAPS = np.zeros((TOTAL_MAGS, RES_Y, RES_X))
for i in range(TOTAL_MAGS):
    UNIT_MAPS[i] = get_magnet_field_on_grid(i, centers_m, assembly.angles, X_grid_m, Y_grid_m)

print("--- Computing Coupling Matrices ---")
def get_field_at_point_physics(x, y, source_idx):
    sx, sy = centers_m[source_idx]
    stheta = assembly.angles[source_idx]
    dx, dy = x - sx, y - sy
    c, s = np.cos(-stheta), np.sin(-stheta)
    dx_l, dy_l = dx*c - dy*s, dx*s + dy*c
    r = np.sqrt(dx_l**2 + dy_l**2)
    if r < 1e-6: return 0, 0

    vol = np.pi * MAG_RADIUS_M**2 * MAG_LEN_M
    m = 1.0 * vol
    fac = (MU0 * m) / (4 * np.pi * r**3)
    bx = fac * (3*(dx_l/r)**2 - 1)
    by = fac * (3*(dx_l/r)*(dy_l/r))
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
        # Initialize with Frame 0 pattern
        init_pattern = COIL_SEQUENCE[0]
        for k in range(count):
            self.M[k] = (MAG_BR_T / MU0) * init_pattern[k]

    def update(self, H_coils):
        H_int = G_matrix @ self.M
        H_total = H_coils + H_int
        M_up = self.Ms * np.tanh(self.k * (H_total + self.Hc))
        M_down = self.Ms * np.tanh(self.k * (H_total - self.Hc))
        self.M = np.clip(self.M, M_down, M_up)
        return self.M, H_int

print(f"--- Running Simulation ({SIM_TOTAL_TIME_US} us) ---")

sys_model = SystemHysteresis(TOTAL_MAGS)

# Output Containers
results = {
    't': [], 'I_all': [], 'V': [],
    'M_all': [],
    'H_total_all': [], # Store H_total (Coil + Int) for proper B-H loops
    'frame': []
}

Q_cap = VOLTAGE_INIT * CAPACITANCE
I_single_coil = 0.0
L_single = (MU0 * (TOTAL_TURNS**2) * (np.pi*(MAG_RADIUS_M*1.1)**2)) / MAG_LEN_M

current_frame_idx = -1

for step in range(SIM_STEPS):
    t_us = step * TIME_STEP_US

    # --- FRAME LOGIC ---
    frame_idx = int(t_us / FRAME_DURATION_US)
    if frame_idx >= len(COIL_SEQUENCE): frame_idx = len(COIL_SEQUENCE) - 1

    coil_states = COIL_SEQUENCE[frame_idx]
    num_active_coils = np.sum(np.abs(coil_states))

    # Trigger Pulse at start of frame
    if frame_idx != current_frame_idx:
        current_frame_idx = frame_idx
        # Capacitor NOT recharged (continuing discharge)
        I_single_coil = 0.0

    time_in_frame_us = t_us % FRAME_DURATION_US

    # --- CIRCUIT (RLC Discharge) ---
    V_cap = Q_cap / CAPACITANCE

    if time_in_frame_us < PULSE_WIDTH_US:
        dIdt = (V_cap - I_single_coil * COIL_RESISTANCE) / L_single
        total_current_draw = I_single_coil * num_active_coils
        dQdt = -total_current_draw
    else:
        if I_single_coil > 0:
            dIdt = (-I_single_coil * COIL_RESISTANCE - DIODE_DROP) / L_single
            dQdt = 0
        else:
            I_single_coil = 0; dIdt = 0; dQdt = 0

    I_single_coil += dIdt * (TIME_STEP_US * 1e-6)
    Q_cap += dQdt * (TIME_STEP_US * 1e-6)

    I_vec_all = I_single_coil * coil_states

    # --- COIL CONTROL ---
    H_mag_magnitude = (TOTAL_TURNS * I_single_coil) / MAG_LEN_M
    H_coils = H_mag_magnitude * coil_states

    # --- HYSTERESIS ---
    M_vec, H_int_vec = sys_model.update(H_coils)

    # Calculate Total Field for Visualization (Coil + Int)
    H_total_vec = H_coils + H_int_vec

    # --- STORAGE ---
    if step % ANIMATION_SKIP == 0:
        results['t'].append(t_us)
        results['I_all'].append(I_vec_all.copy())
        results['V'].append(V_cap)
        results['M_all'].append(M_vec.copy())
        results['H_total_all'].append(H_total_vec.copy()) # Store H_total
        results['frame'].append(frame_idx)

# Arrays
for k in results: results[k] = np.array(results[k])

# ==========================================
# 5. VISUALIZATION
# ==========================================
print("--- Initializing Visualization ---")

colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list('bk_bwr', colors, N=100)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3)

# Heatmap
ax_map = fig.add_subplot(gs[0, :])
mesh_map = ax_map.pcolormesh(X_grid, Y_grid, np.zeros_like(X_grid), shading='auto', cmap=cm, vmin=-1.5, vmax=1.5)
fig.colorbar(mesh_map, ax=ax_map, label='Field Intensity [T]')
ax_map.set_title("16-EPM Field (Coil Control)", fontsize=14)
ax_map.set_aspect('equal')
ax_map.set_xlim(-60, 60)
ax_map.set_ylim(-30, 30)

# Magnets
mag_patches = []
mag_labels = []
for i in range(TOTAL_MAGS):
    cx, cy = assembly.centers[i]
    ang = assembly.angles[i]
    rect = Rectangle((cx - MAG_LENGTH_MM/2, cy - MAG_DIAMETER_MM/2),
                     MAG_LENGTH_MM, MAG_DIAMETER_MM, angle=np.degrees(ang),
                     rotation_point='center', ec='white', lw=1, fc='gray')
    ax_map.add_patch(rect)
    mag_patches.append(rect)
    ax_map.text(cx, cy, str(i), color='yellow', fontsize=8, ha='center', va='center', alpha=0.7)

# --- 4x4 Grid of Dynamics Plots (Coil Currents) ---
gs_dyn = gs[1, 0].subgridspec(4, 4, wspace=0.1, hspace=0.1)

lines_I = []
text_I = []
axes_dyn = []

max_I = np.max(np.abs(results['I_all'])) + 0.5
cmap_L = plt.cm.Reds(np.linspace(0.4, 1.0, 8))
cmap_R = plt.cm.Blues(np.linspace(0.4, 1.0, 8))

for i in range(TOTAL_MAGS):
    row = i // 4
    col = i % 4
    ax_small = fig.add_subplot(gs_dyn[row, col])

    c = cmap_L[i] if i < 8 else cmap_R[i-8]
    l, = ax_small.plot([], [], lw=1.5, color=c)
    lines_I.append(l)
    axes_dyn.append(ax_small)

    txt = ax_small.text(0.5, 0.5, "", transform=ax_small.transAxes,
                        fontsize=8, color=c, fontweight='bold', ha='center', va='center')
    text_I.append(txt)

    ax_small.set_ylim(-max_I, max_I)
    ax_small.set_xlim(0, SIM_TOTAL_TIME_US)
    ax_small.set_xticks([])
    ax_small.set_yticks([])
    ax_small.text(0.05, 0.8, f"C#{i}", transform=ax_small.transAxes, fontsize=6, color='black', alpha=0.5)

    if i == 0:
        ax_v = ax_small.twinx()
        l_V_ref, = ax_v.plot([], [], 'g--', lw=1, alpha=0.6)
        ax_v.set_ylim(0, VOLTAGE_INIT*1.1)
        ax_v.set_yticks([])
        ax_small.text(0.5, 0.8, "V_cap", transform=ax_small.transAxes, fontsize=6, color='green')

# --- 4x4 Grid of Hysteresis Plots (16 Magnets) ---
gs_hys = gs[1, 1].subgridspec(4, 4, wspace=0.1, hspace=0.1)

lines_hys = []
points_hys = []
text_hys = []

# Hysteresis Bounds
# H_peak coil ~ 225 kA/m. With margin for interaction, scale to 300.
h_max = 150.0 # kA/m
m_max = 1.6

for i in range(TOTAL_MAGS):
    row = i // 4
    col = i % 4
    ax_small = fig.add_subplot(gs_hys[row, col])

    c = cmap_L[i] if i < 8 else cmap_R[i-8]
    l, = ax_small.plot([], [], lw=1, color='black', alpha=0.3)
    p, = ax_small.plot([], [], 'o', color=c, markersize=3)

    lines_hys.append(l)
    points_hys.append(p)

    txt = ax_small.text(0.5, 0.5, "", transform=ax_small.transAxes,
                        fontsize=8, color=c, fontweight='bold', ha='center', va='center')
    text_hys.append(txt)

    ax_small.set_ylim(-m_max, m_max)
    ax_small.set_xlim(-h_max, h_max)
    ax_small.set_xticks([])
    ax_small.set_yticks([])
    ax_small.grid(True, alpha=0.2)
    ax_small.text(0.05, 0.8, f"M#{i}", transform=ax_small.transAxes, fontsize=6, color='black', alpha=0.5)

# Stats
ax_stats = fig.add_subplot(gs[1, 2])
ax_stats.axis('off')
txt_stats = ax_stats.text(0.05, 0.95, "", transform=ax_stats.transAxes,
                          fontsize=11, family='monospace', verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

def update(frame):
    # Fix IndexError: slice inclusive of current frame
    idx = frame
    if idx >= len(results['t']): idx = len(results['t']) - 1

    # 1. Update Heatmap
    I_vec_now = results['I_all'][idx]
    curr_frame = int(results['frame'][idx])

    Total_Grid = np.zeros((RES_Y, RES_X))
    M_current_vec = results['M_all'][idx]

    for i in range(TOTAL_MAGS):
        M_val = M_current_vec[i]
        H_coil_val = (TOTAL_TURNS * I_vec_now[i] / MAG_LEN_M)
        Source_T = (M_val * MU0) + (H_coil_val * MU0)
        Total_Grid += UNIT_MAPS[i] * Source_T

        norm = (M_val * MU0 + 1.5) / 3.0
        mag_patches[i].set_facecolor(cm(np.clip(norm, 0, 1)))

    mesh_map.set_array(Total_Grid.ravel())

    # 2. Update Current Grid (4x4)
    # Use :idx+1 to ensure non-empty slice
    slice_idx = idx + 1

    for i in range(TOTAL_MAGS):
        lines_I[i].set_data(results['t'][:slice_idx], results['I_all'][:slice_idx, i])
        val = I_vec_now[i]
        text_I[i].set_text(f"{val:.1f}A")
        text_I[i].set_alpha(1.0 if abs(val) > 0.1 else 0.3)

    l_V_ref.set_data(results['t'][:slice_idx], results['V'][:slice_idx])

    # 3. Update Hysteresis Grid (4x4) - Now using Total Field
    for i in range(TOTAL_MAGS):
        h_traj = results['H_total_all'][:slice_idx, i] / 1000.0 # kA/m
        m_traj = results['M_all'][:slice_idx, i] * MU0          # T

        lines_hys[i].set_data(h_traj, m_traj)
        points_hys[i].set_data([h_traj[-1]], [m_traj[-1]])

        m_val = m_traj[-1]
        text_hys[i].set_text(f"{m_val:.2f}T")

    # 4. Stats - Updated fields
    curr_V = results['V'][idx]
    en = 0.5 * CAPACITANCE * (VOLTAGE_INIT**2 - curr_V**2)

    txt_stats.set_text(f"Time: {results['t'][idx]:.1f} us\n"
                       f"Frame: {curr_frame}\n"
                       f"Capacitor Voltage: {curr_V:.1f} V\n"
                       f"Energy Used: {en:.4f} J")

    return [mesh_map, txt_stats] + mag_patches + lines_I + text_I + lines_hys + points_hys + text_hys

ani = FuncAnimation(fig, update, frames=len(results['t']), interval=30, blit=False)
plt.tight_layout()
plt.show()
