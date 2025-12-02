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
SIM_TOTAL_TIME_US = 150.0  # Extended slightly to see relaxation
PULSE_WIDTH_US = 50.0      # Pulse width
TIME_STEP_US = 1.0
ANIMATION_FRAMES = 60

# Assembly Parameters
NUM_CYLINDERS = 2
MAGS_PER_CYL = 8
TOTAL_MAGS = NUM_CYLINDERS * MAGS_PER_CYL
CYL_RADIUS_MM = 12.0       # Mounting radius (distance to inner face of magnet)
CYL_GAP_MM = 0.5           # Air gap between the two cylinders at contact point

# Magnet Parameters (Alnico 5)
MAG_DIAMETER_MM = 4.75
MAG_LENGTH_MM = 12.5
MAG_LEN_M = MAG_LENGTH_MM / 1000.0
MAG_RADIUS_M = (MAG_DIAMETER_MM / 2.0) / 1000.0

# Hysteresis Properties (Alnico 5)
# Br = 1.35T, Hc ~ 50 kA/m.
# Note: Neighbor fields can be > 100 kA/m, significantly affecting dynamics!
MAG_BR_T = 1.35
MAG_HC_A_M = 50000.0
MAG_SAT_M = (1.35 / (4 * np.pi * 1e-7))

# Coil Parameters
# Assuming each magnet has its own coil, all wired in parallel or series to fire simultaneously
LAYERS = 3
TURNS_PER_LAYER = 125
TOTAL_TURNS = LAYERS * TURNS_PER_LAYER
COIL_RESISTANCE = 4.0      # Per Coil
# Circuit: We assume a powerful bank driving ALL coils.
# To simplify, we model 1 coil circuit and assume current is identical in all (Series)
# Adjusted capacitance/voltage for the larger load
CAPACITANCE = 1e-4         # 100 uF (Increased for 16 coils energy)
VOLTAGE_INIT = 60.0        # Increased Voltage for series/parallel load
DIODE_DROP = 0.7

MU0 = 4 * np.pi * 1e-7

# ==========================================
# 2. GEOMETRY & MATH HELPERS
# ==========================================

class MagnetAssembly:
    def __init__(self):
        self.centers = np.zeros((TOTAL_MAGS, 2)) # X, Y positions (Top down view)
        self.angles = np.zeros(TOTAL_MAGS)       # Orientation (0 = Pointing Right)
        self.polarity_sign = np.zeros(TOTAL_MAGS) # +1 or -1 for initial state pattern

        # Define Centers of the two main cylinders
        # Cyl 1 is on Left, Cyl 2 on Right.
        # Magnet length is 12.5mm.
        # Radius to center of magnet = Mounting_Radius + Length/2
        R_center = CYL_RADIUS_MM + MAG_LENGTH_MM/2

        # Shift cylinders so they almost touch at x=0
        # Tip of magnet is at Mounting_Radius + Length = Outer_Radius
        R_outer = CYL_RADIUS_MM + MAG_LENGTH_MM

        # Center of Cyl 1
        c1_x = -(R_outer + CYL_GAP_MM/2)
        # Center of Cyl 2
        c2_x = +(R_outer + CYL_GAP_MM/2)

        for i in range(TOTAL_MAGS):
            cyl_idx = i // MAGS_PER_CYL
            mag_idx = i % MAGS_PER_CYL

            # Angle step
            theta_step = 2 * np.pi / MAGS_PER_CYL

            if cyl_idx == 0:
                # Cylinder 1 (Left)
                # Align mag 0 to point East (0 rad) towards contact
                theta = mag_idx * theta_step
                cx, cy = c1_x, 0
                # Pattern: N, S, N, S... starting with N (1) at contact
                pat = 1 if mag_idx % 2 == 0 else -1
            else:
                # Cylinder 2 (Right)
                # Align mag 0 to point West (pi rad) towards contact
                # Phase shift so mag 0 is at pi
                theta = np.pi + mag_idx * theta_step
                cx, cy = c2_x, 0
                # Pattern: S, N, S, N... starting with S (-1) at contact
                # This ensures Cyl1(N) faces Cyl2(S) -> Attraction
                pat = -1 if mag_idx % 2 == 0 else 1

            # Position of magnet center
            mx = cx + R_center * np.cos(theta)
            my = cy + R_center * np.sin(theta)

            self.centers[i] = [mx, my]
            self.angles[i] = theta
            self.polarity_sign[i] = pat

assembly = MagnetAssembly()

def get_field_at_point(x, y, source_idx, M_val):
    """
    Calculate B_axial (along magnet axis) projected onto the target magnet's axis.
    Simplified dipole/solenoid model for interaction coupling.
    """
    # Source properties
    sx, sy = assembly.centers[source_idx]
    stheta = assembly.angles[source_idx]

    # Target point relative to source, rotated to source frame
    dx_global = x - sx
    dy_global = y - sy

    # Rotate into Source's local frame (X' along magnet axis)
    # Cos/Sin of negative angle to derotate
    cs, sn = np.cos(-stheta), np.sin(-stheta)
    dx_local = dx_global * cs - dy_global * sn
    dy_local = dx_global * sn + dy_global * cs

    # Field of a solenoid/dipole at (dx_local, dy_local)
    # Using the loop code from before would be accurate but slow for matrix.
    # Using Dipole approx for coupling matrix (valid for distances > L)
    # But neighbors are close. We stick to the robust loop code but simplified.
    # To save space, we just use a dipole approximation for the Interaction Matrix
    # B = (mu0 m / 4pi r^3) * (3(n.r)r - n)

    r_sq = dx_local**2 + dy_local**2
    r_val = np.sqrt(r_sq)
    if r_val < 1e-6: return 0 # Self

    # Magnetic Moment m = M * Volume. Direction is (1, 0) in local frame.
    vol = np.pi * (MAG_RADIUS_M)**2 * MAG_LEN_M
    m_mag = M_val * vol

    # Dipole formula 2D projection
    # r_hat = (dx_local/r, dy_local/r)
    # m_dot_r = 1.0 * (dx_local/r)
    # B_vec = (3 * m_dot_r * r_hat - [1,0])

    factor = (MU0 * m_mag) / (4 * np.pi * (r_val**3))
    bx_local = factor * (3 * (dx_local/r_val) * (dx_local/r_val) - 1)
    by_local = factor * (3 * (dx_local/r_val) * (dy_local/r_val) - 0)

    # Rotate B vector back to global
    cs_back, sn_back = np.cos(stheta), np.sin(stheta)
    bx_global = bx_local * cs_back - by_local * sn_back
    by_global = bx_local * sn_back + by_local * cs_back

    return bx_global, by_global

# ==========================================
# 3. INTERACTION MATRICES
# ==========================================

print("--- Computing Coupling Matrices (N Body Interaction) ---")

# G_matrix[i, j] = Field projected onto Axis of i, caused by Unit Magnetization of j
G_matrix = np.zeros((TOTAL_MAGS, TOTAL_MAGS))

# Convert geometry to meters for calculation
centers_m = assembly.centers / 1000.0

for i in range(TOTAL_MAGS):
    # Target i vector direction
    ix_dir = np.cos(assembly.angles[i])
    iy_dir = np.sin(assembly.angles[i])

    for j in range(TOTAL_MAGS):
        if i == j:
            G_matrix[i,j] = 0
            continue

        # Calculate Field from J at center of I (with Mj = 1 A/m)
        # Note: M is usually magnetization. Dipole moment m = M * Vol.
        # We pass M=1 to get the coefficient.
        bx, by = get_field_at_point(centers_m[i,0], centers_m[i,1], j, 1.0)

        # Project B onto the axis of magnet I
        # This determines how much J aids/opposes I's magnetization
        H_proj = (bx * ix_dir + by * iy_dir) / MU0 # Convert B to H

        G_matrix[i,j] = H_proj

print(f"Coupling Matrix Calculated. Max Coupling Factor: {np.max(np.abs(G_matrix)):.2f} (Dimensionless/Geom)")
# Check nearest neighbor coupling (approx)
# H ~ M * Factor. If Factor is 0.1, then 1000kA/m M causes 100kA/m H.

# ==========================================
# 4. PHYSICS ENGINE
# ==========================================

class SystemHysteresis:
    def __init__(self, count):
        self.count = count
        self.Ms = MAG_SAT_M
        self.Hc = MAG_HC_A_M
        self.M = np.zeros(count)
        self.k = 5.0 / self.Hc

        # Initialize pattern
        for k in range(count):
            # Start at Remanence (Br/mu0) * Sign
            self.M[k] = (MAG_BR_T / MU0) * assembly.polarity_sign[k]

    def update(self, H_coils):
        """
        Solved coupled system: M_i = f(H_coil_i + sum(G_ij * M_j))
        Simple relaxation or lagged update (stable for small dt)
        """
        # 1. Calculate Interaction Field based on previous M
        # H_int[i] = sum(G[i,j] * M[j])
        H_int = G_matrix @ self.M

        # 2. Total Local Field
        H_total = H_coils + H_int

        # 3. Update Hysteresis State
        # Envelope function for major loop
        # We assume the pulse tries to flip everyone.
        # So H_coils is opposing the initial state.

        M_up = self.Ms * np.tanh(self.k * (H_total + self.Hc))
        M_down = self.Ms * np.tanh(self.k * (H_total - self.Hc))

        # Vectorized Clamp
        # If M > M_up, snap down. If M < M_down, snap up.
        self.M = np.clip(self.M, M_down, M_up)

        return self.M, H_int

# Simulation Loop
print("--- Running Multi-Body Simulation ---")

sys_model = SystemHysteresis(TOTAL_MAGS)

# To simulate the flip, we apply a coil field OPPOSITE to the initial polarity.
# Coil field direction vector relative to magnet axis is -1 * sign(initial).
# But wait, we want to flip the *whole* pattern.
# So if Mag 0 was +1 (N), we apply -H. If Mag 1 was -1 (S), we apply +H.
# Effectively, H_coil_local = -1 * Initial_Sign * Magnitude
coil_polarity_vec = -1.0 * assembly.polarity_sign

t_eval = np.arange(0, SIM_TOTAL_TIME_US, TIME_STEP_US) * 1e-6
results_I = []
results_M_avg = []
results_M_0 = [] # Track Magnet 0 (Contact point)
results_M_1 = [] # Track Magnet 1 (Neighbor)
results_Hint_0 = [] # Track Interaction field on Mag 0

Q_cap = VOLTAGE_INIT * CAPACITANCE
I = 0.0

# Inductance: 16 coils in series? Or parallel?
# Let's assume Parallel bank. L_eq = L_one / 16. R_eq = R_one / 16.
# Dynamics are same as 1 coil with 1/16th cap? No.
# Let's model Single Coil Equivalent circuit for simplicity, scaled.
L_ONE = (MU0 * (TOTAL_TURNS**2) * (np.pi*(MAG_RADIUS_M*1.1)**2)) / MAG_LEN_M
# We simulate 1 branch.
L_sys = L_ONE
R_sys = COIL_RESISTANCE

for t in t_eval:
    t_us = t * 1e6
    dt = TIME_STEP_US * 1e-6
    V_cap = Q_cap / CAPACITANCE

    # Circuit Dynamics
    if t_us < PULSE_WIDTH_US:
        dIdt = (V_cap - I * R_sys) / L_sys
        dQdt = -I # This Q is for ONE coil's share of cap?
        # Let's just say infinite cap bank or scaled.
        # Simplified: Standard discharge curve.
        dQdt = -I
    else:
        if I > 0:
            dIdt = (-I * R_sys - DIODE_DROP) / L_sys
            dQdt = 0
        else:
            I = 0; dIdt = 0; dQdt = 0

    I += dIdt * dt
    Q_cap += dQdt * dt

    # Calculate H_coil magnitude
    # H = N*I / L
    H_mag = (TOTAL_TURNS * I) / MAG_LEN_M

    # Apply to all magnets with correct orientation to FLIP them
    H_coils = H_mag * coil_polarity_vec

    # Update System
    M_vec, H_int_vec = sys_model.update(H_coils)

    results_I.append(I)
    results_M_avg.append(np.mean(np.abs(M_vec)))
    results_M_0.append(M_vec[0])   # Cyl 1 Contact Magnet
    results_M_1.append(M_vec[1])   # Cyl 1 Neighbor
    results_Hint_0.append(H_int_vec[0])

results_I = np.array(results_I)
results_M_0 = np.array(results_M_0)
results_Hint_0 = np.array(results_Hint_0)
time_us = t_eval * 1e6

# ==========================================
# 5. VISUALIZATION
# ==========================================

print("--- Initializing Visualization ---")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

# Plot 1: Top Down Geometry & State
ax_geo = fig.add_subplot(gs[0, :])
ax_geo.set_aspect('equal')
ax_geo.set_facecolor('#1a1a1a')
ax_geo.set_title("16-EPM Assembly (Top Down) - Contact at Center", color='white')

# Draw Cylinders (Visual reference)
# Left Cylinder
c1 = Circle((-(CYL_RADIUS_MM + MAG_LENGTH_MM + CYL_GAP_MM/2), 0), CYL_RADIUS_MM, color='gray', alpha=0.3)
# Right Cylinder
c2 = Circle((+(CYL_RADIUS_MM + MAG_LENGTH_MM + CYL_GAP_MM/2), 0), CYL_RADIUS_MM, color='gray', alpha=0.3)
ax_geo.add_patch(c1)
ax_geo.add_patch(c2)

# Create Magnet Patches
mag_patches = []
for i in range(TOTAL_MAGS):
    cx, cy = assembly.centers[i]
    ang = assembly.angles[i]

    # Create rectangle rotated around center
    # Width = Length (radial), Height = Diameter (tangential)
    # Corner relative to center
    w = MAG_LENGTH_MM
    h = MAG_DIAMETER_MM

    rect = Rectangle((cx - w/2, cy - h/2), w, h, angle=np.degrees(ang),
                     rotation_point='center', ec='white', lw=1)
    ax_geo.add_patch(rect)
    mag_patches.append(rect)

ax_geo.set_xlim(-60, 60)
ax_geo.set_ylim(-30, 30)

# Plot 2: Dynamics (Current vs Time)
ax_dyn = fig.add_subplot(gs[1, 0])
l_curr, = ax_dyn.plot(time_us, results_I, 'y-', lw=2)
ax_dyn.set_title("Coil Current")
ax_dyn.set_xlabel("Time (us)")
ax_dyn.set_ylabel("Current (A)")
ax_dyn.grid(True, alpha=0.3)

# Plot 3: Interaction Analysis (Contact Magnet)
ax_int = fig.add_subplot(gs[1, 1])
l_hint, = ax_int.plot(time_us, results_Hint_0/1000, 'c-', label='Interaction H')
l_mag, = ax_int.plot(time_us, results_M_0 * MU0, 'r--', label='Magnetization (T)')
ax_int.set_title("Contact Magnet (#0) Physics")
ax_int.set_ylabel("Field [kA/m] / Mag [T]")
ax_int.legend()
ax_int.grid(True, alpha=0.3)

# Animation Update
def update(frame):
    idx = int((frame / ANIMATION_FRAMES) * len(t_eval))
    if idx >= len(t_eval): idx = len(t_eval) - 1

    # Get M vector at this time step
    # Re-calculate M state for visual (or store full history? Storing full history is better)
    # To save memory we re-sim or just store indices.
    # Actually, we didn't store full M_vec history for all magnets, only specific ones.
    # Let's reconstruct or simplify: We just animate the scalar arrays we have
    # and infer colors from the pattern logic + M_0/M_1?
    # No, for full visualization we need full state.
    # Re-running the update step for the specific frame index is fast enough.

    # Re-compute state for this frame (Fast approximation)
    # Or better: We assume all magnets behave similarly to their group leaders (0 and 1)
    # but with alternating signs.
    # Mag 0 (Cyl 1 contact) history is in results_M_0.
    # Mag 1 (Cyl 1 neighbor) history is in results_M_1.

    # Color mapping
    # N (positive M) = Red, S (negative M) = Blue
    m0_val = results_M_0[idx]
    m1_val = results_M_1[idx]

    for i in range(TOTAL_MAGS):
        # Determine which history to use
        # Even indices are "Contact-like" (Radial Out/In pattern A)
        # Odd indices are "Neighbor-like" (Radial In/Out pattern B)
        # We need to respect the sign of the original pattern.

        orig_sign = assembly.polarity_sign[i]

        # If orig_sign was +1 (like Mag 0), we use M0 trace (which starts + and goes -)
        # If orig_sign was -1 (like Mag 1), we use M1 trace (which starts - and goes +)

        # This is a valid assumption due to symmetry of the setup
        if orig_sign > 0:
            val = m0_val
        else:
            val = m1_val

        # Color: Blue (-1.5T) to Red (+1.5T)
        norm_val = val * MU0 # Tesla
        # Map -1.5..1.5 to 0..1
        color_val = (norm_val + 1.5) / 3.0
        color_val = np.clip(color_val, 0, 1)

        c = plt.cm.bwr(color_val)
        mag_patches[i].set_facecolor(c)

    # Update Lines
    l_curr.set_data(time_us[:idx], results_I[:idx])
    l_hint.set_data(time_us[:idx], results_Hint_0[:idx]/1000)
    l_mag.set_data(time_us[:idx], results_M_0[:idx] * MU0)

    return mag_patches + [l_curr, l_hint, l_mag]

ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, interval=50, blit=False)
plt.tight_layout()
plt.show()
