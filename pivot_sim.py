import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================

# Time Settings
DT = 0.001              # 1ms timestep for stability
TOTAL_TIME = 3.0        # Seconds
FLIP_TIME = 1.0         # Time when polarity flips
ANIMATION_SKIP = 20     # Render every Nth frame

# Geometry
NUM_MAGS = 8            # Per cylinder
CYL_RADIUS_MM = 12.0    # Inner mounting radius
MAG_LENGTH_MM = 12.5
MAG_DIAM_MM = 4.75
GAP_MM = 0.2            # Initial separation

# Physical Properties (Estimated)
MASS_CYLINDER = 0.05    # kg (50g)
INERTIA_CYLINDER = 0.5 * MASS_CYLINDER * ((CYL_RADIUS_MM+MAG_LENGTH_MM)/1000)**2

# Magnet Strength
# Dipole Moment m = M * Volume
# Alnico 5 Br ~ 1.35T. M = Br/mu0.
MU0 = 4 * np.pi * 1e-7
MAG_BR = 1.35
MAG_VOL = np.pi * ((MAG_DIAM_MM/2000)**2) * (MAG_LENGTH_MM/1000)
DIPOLE_MAG = (MAG_BR / MU0) * MAG_VOL

# Contact Model
K_SPRING = 10000.0      # N/m (Stiffness)
C_DAMP = 50.0           # Ns/m (Damping)
MU_FRICTION = 0.8       # Friction coefficient (High friction for gear-like grip)

# ==========================================
# 2. RIGID BODY CLASSES
# ==========================================

class RigidBody:
    def __init__(self, x, y, theta, is_fixed=False):
        self.pos = np.array([x, y], dtype=float)
        self.theta = float(theta)

        self.vel = np.array([0.0, 0.0])
        self.omega = 0.0

        self.is_fixed = is_fixed
        self.mass = MASS_CYLINDER
        self.inertia = INERTIA_CYLINDER

        # Magnet Configuration (Local Frame)
        # Store (r, phi) for each magnet relative to body center
        # Magnet 0 is at angle 0 relative to body axis
        self.mag_r = (CYL_RADIUS_MM + MAG_LENGTH_MM/2) / 1000.0
        self.mag_angles_local = np.linspace(0, 2*np.pi, NUM_MAGS, endpoint=False)
        self.polarities = np.ones(NUM_MAGS) # +1 or -1

    def get_magnet_states_global(self):
        """
        Returns (Positions, Dipole_Vectors) for all magnets in global frame.
        """
        global_angles = self.mag_angles_local + self.theta

        # Positions
        # x = cx + r * cos(angle)
        mx = self.pos[0] + self.mag_r * np.cos(global_angles)
        my = self.pos[1] + self.mag_r * np.sin(global_angles)
        positions = np.column_stack((mx, my))

        # Dipole Vectors
        # Direction is radial? Or tangential?
        # Standard configuration usually has dipoles Radial.
        # Let's assume Radial Outward is +1.
        dx = np.cos(global_angles) * self.polarities * DIPOLE_MAG
        dy = np.sin(global_angles) * self.polarities * DIPOLE_MAG
        moments = np.column_stack((dx, dy))

        return positions, moments

    def apply_force_torque(self, force, torque):
        if self.is_fixed: return

        # Linear
        acc = force / self.mass
        self.vel += acc * DT
        self.pos += self.vel * DT

        # Angular
        alpha = torque / self.inertia
        self.omega += alpha * DT
        self.theta += self.omega * DT

        # Simple Damping (Air resistance)
        self.vel *= 0.999
        self.omega *= 0.995

# ==========================================
# 3. PHYSICS ENGINE
# ==========================================

def compute_dipole_force(p1, m1, p2, m2):
    """
    Force exerted BY dipole 1 ON dipole 2.
    F = (3*mu0 / 4pi*r^5) * ...
    """
    r_vec = p2 - p1
    r_len = np.linalg.norm(r_vec)
    if r_len < 1e-4: return np.zeros(2) # Singularity check

    r_hat = r_vec / r_len

    # Dot products
    m1_dot_r = np.dot(m1, r_vec)
    m2_dot_r = np.dot(m2, r_vec)
    m1_dot_m2 = np.dot(m1, m2)

    prefactor = (3 * MU0) / (4 * np.pi * (r_len**5))

    term1 = m1_dot_r * m2
    term2 = m2_dot_r * m1
    term3 = m1_dot_m2 * r_vec
    term4 = -5 * m1_dot_r * m2_dot_r * r_vec / (r_len**2)

    force = prefactor * (term1 + term2 + term3 + term4)
    return force

def run_simulation():
    # Setup Geometry
    R_outer_m = (CYL_RADIUS_MM + MAG_LENGTH_MM) / 1000.0

    # Left Body (Fixed) at Origin
    # Initial Rotation: Mag 0 points East (0)
    left_body = RigidBody(0, 0, 0, is_fixed=True)
    # Pattern: NSNS... (Mag 0 is N=+1)
    # Mag 0 is at Angle 0 (Contact point)
    left_body.polarities = np.array([1, -1, 1, -1, 1, -1, 1, -1])

    # Right Body (Free)
    # Positioned to the right.
    # Gap distance
    dist = 2 * R_outer_m + (GAP_MM / 1000.0)

    # Orientation: Mag 0 points West (pi).
    # This aligns Left(0) and Right(0).
    # We want "Opposite Polarity" where they touch so they attract.
    # Left(0) is +1 (N). So Right(0) should be -1 (S).
    right_body = RigidBody(dist, 0, np.pi, is_fixed=False)
    # Pattern: SNSN... (Mag 0 is S=-1)
    right_body.polarities = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    # Data Storage
    history = {
        't': [],
        'rx': [], 'ry': [], 'rtheta': [],
        'force_mag': []
    }

    time_steps = int(TOTAL_TIME / DT)

    print(f"Simulating {TOTAL_TIME}s in {time_steps} steps...")

    for step in range(time_steps):
        t = step * DT

        # --- EVENT: POLARITY FLIP ---
        if abs(t - FLIP_TIME) < DT/2:
            print(f"EVENT: Flipping Magnet at t={t:.2f}s")
            # "Magnet on the left of the right cylinder touching magnet"
            # In Right Body local frame, Mag 0 is at 0 degrees local.
            # Right Body is rotated 180 (pi). So Mag 0 is at 180 (Global).
            # This is the contact magnet.
            # Old state: -1. New state: +1.
            right_body.polarities[0] = 1.0 # Flip to N

        # Get Global States
        p1, m1 = left_body.get_magnet_states_global()
        p2, m2 = right_body.get_magnet_states_global()

        # --- 1. MAGNETIC FORCES ---
        F_mag_total = np.zeros(2)
        Tau_mag_total = 0.0

        # N^2 Interactions (Optimize later if needed, 64 is fine)
        for i in range(NUM_MAGS):      # Left Body Magnets
            for j in range(NUM_MAGS):  # Right Body Magnets
                # Force on Right Body (j) from Left Body (i)
                f_pair = compute_dipole_force(p1[i], m1[i], p2[j], m2[j])

                F_mag_total += f_pair

                # Torque = r x F
                # Lever arm from Right Body Center to Magnet j
                lever = p2[j] - right_body.pos
                torque = lever[0]*f_pair[1] - lever[1]*f_pair[0]
                Tau_mag_total += torque

        # --- 2. CONTACT CONSTRAINT (Collision) ---
        # Simple Circle-Circle contact
        # Dist between centers
        delta_pos = right_body.pos - left_body.pos
        dist_sq = np.dot(delta_pos, delta_pos)
        dist_val = np.sqrt(dist_sq)

        collision_dist = 2 * R_outer_m
        F_contact = np.zeros(2)
        Tau_contact = 0.0

        if dist_val < collision_dist:
            # Penetration depth
            pen = collision_dist - dist_val
            normal = delta_pos / dist_val

            # Spring Force (Push apart)
            # Damping needs relative velocity along normal
            rel_vel = right_body.vel # Left is fixed 0
            v_normal = np.dot(rel_vel, normal)

            f_spring = K_SPRING * pen
            f_damp = -C_DAMP * v_normal
            f_normal_mag = max(0, f_spring + f_damp) # Can't pull

            F_normal = f_normal_mag * normal

            # --- 3. FRICTION (Rolling) ---
            # Velocity at contact point
            # V_contact = V_cm + omega x r
            contact_pt_rel = -normal * R_outer_m # Contact point relative to Right Center

            # Tangential velocity
            # Cross product 2D: omega * r_perp
            v_rot_x = -right_body.omega * contact_pt_rel[1]
            v_rot_y =  right_body.omega * contact_pt_rel[0]

            v_contact = right_body.vel + np.array([v_rot_x, v_rot_y])

            # Remove normal component to get tangent
            v_c_normal = np.dot(v_contact, normal) * normal
            v_tangent = v_contact - v_c_normal

            # Friction Force opposes tangential velocity
            if np.linalg.norm(v_tangent) > 1e-6:
                tangent_dir = v_tangent / np.linalg.norm(v_tangent)
                f_friction_mag = min(MU_FRICTION * f_normal_mag, C_DAMP * np.linalg.norm(v_tangent))
                F_friction = -f_friction_mag * tangent_dir
            else:
                F_friction = np.zeros(2)

            F_contact = F_normal + F_friction

            # Torque from friction
            # r x F_friction
            Tau_contact = contact_pt_rel[0]*F_friction[1] - contact_pt_rel[1]*F_friction[0]

        # Apply Forces
        right_body.apply_force_torque(F_mag_total + F_contact, Tau_mag_total + Tau_contact)

        # Store Data
        if step % ANIMATION_SKIP == 0:
            history['t'].append(t)
            history['rx'].append(right_body.pos[0])
            history['ry'].append(right_body.pos[1])
            history['rtheta'].append(right_body.theta)
            history['force_mag'].append(np.linalg.norm(F_mag_total))

    return history

# ==========================================
# 4. VISUALIZATION
# ==========================================

hist = run_simulation()

print("--- Initializing Animation ---")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')
ax.set_xlim(-0.05, 0.10)
ax.set_ylim(-0.05, 0.05)
ax.grid(True, alpha=0.3)
ax.set_title("EPM Rigid Body Dynamics: Polarity Flip at t=1.0s")

# Static Left Cylinder
left_circ = Circle((0, 0), CYL_RADIUS_MM/1000, color='gray', alpha=0.3)
ax.add_patch(left_circ)

# Dynamic Right Cylinder
right_circ = Circle((0, 0), CYL_RADIUS_MM/1000, color='gray', alpha=0.3)
ax.add_patch(right_circ)

# Magnet Patches
# We need to update 16 patches
mag_patches = []

# Left Magnets (Fixed)
# We can just draw them once, but for consistency we use update loop logic roughly
L_radius_m = (CYL_RADIUS_MM + MAG_LENGTH_MM/2)/1000
L_angles = np.linspace(0, 2*np.pi, NUM_MAGS, endpoint=False)
L_pols = np.array([1, -1, 1, -1, 1, -1, 1, -1])

for i in range(NUM_MAGS):
    # Left
    mx = L_radius_m * np.cos(L_angles[i])
    my = L_radius_m * np.sin(L_angles[i])
    col = 'red' if L_pols[i] > 0 else 'blue'
    rect = Rectangle((mx - 0.006, my - 0.002), 0.0125, 0.00475, angle=np.degrees(L_angles[i]),
                     rotation_point='center', color=col, ec='black')
    ax.add_patch(rect)

# Right Magnets (Dynamic)
R_patches = []
for i in range(NUM_MAGS):
    rect = Rectangle((0,0), 0.0125, 0.00475, color='gray', ec='black')
    ax.add_patch(rect)
    R_patches.append(rect)

# Trajectory Line
line_traj, = ax.plot([], [], 'k--', linewidth=1, alpha=0.5)
text_info = ax.text(0.05, 0.9, "", transform=ax.transAxes)

def update(frame):
    # Get State
    rx = hist['rx'][frame]
    ry = hist['ry'][frame]
    rtheta = hist['rtheta'][frame]
    t_val = hist['t'][frame]

    # Update Right Cylinder
    right_circ.center = (rx, ry)

    # Update Right Magnets
    mag_r = (CYL_RADIUS_MM + MAG_LENGTH_MM/2) / 1000.0
    local_angles = np.linspace(0, 2*np.pi, NUM_MAGS, endpoint=False)

    # Polarity Logic for Right Cylinder
    # Initial: [-1, 1, -1, 1...]
    # After 1s: [1, 1, -1, 1...] (Index 0 flips)
    current_pols = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    if t_val >= FLIP_TIME:
        current_pols[0] = 1 # The flip

    for i in range(NUM_MAGS):
        global_ang = local_angles[i] + rtheta
        mx = rx + mag_r * np.cos(global_ang)
        my = ry + mag_r * np.sin(global_ang)

        # Color
        col = 'red' if current_pols[i] > 0 else 'blue'

        R_patches[i].set_xy((mx - 0.00625, my - 0.002375)) # Centering roughly
        # Rotation is tricky with set_xy, simpler to recreate or set angle property
        R_patches[i].angle = np.degrees(global_ang)
        R_patches[i].rotation_point = 'center' # Matplotlib > 3.x
        # Actually rectangle rotation centers are tricky in updates, let's just set angle
        # A workaround for robustness:
        # We need to set the anchor point (bottom-left) correctly based on rotation
        # Or use a transform.
        # Simpler: remove and re-add? No, slow.
        # Efficient way:
        # Calculate corners manually? No.
        # Use set_angle and set_xy of the anchor (bottom left corner relative to rotation)
        # Unrotated anchor: (mx - w/2, my - h/2)
        # We need to manually calculate the rotated bottom-left corner position
        w, h = 0.0125, 0.00475

        # Center of magnet
        cx, cy = mx, my
        angle_rad = global_ang

        # Offset to corner (-w/2, -h/2) rotated
        dx = -w/2 * np.cos(angle_rad) - (-h/2) * np.sin(angle_rad)
        dy = -w/2 * np.sin(angle_rad) + (-h/2) * np.cos(angle_rad)

        R_patches[i].set_xy((cx + dx, cy + dy))
        R_patches[i].set_angle(np.degrees(angle_rad))
        R_patches[i].set_facecolor(col)

    # Line
    line_traj.set_data(hist['rx'][:frame], hist['ry'][:frame])

    status = "LOCKED" if t_val < FLIP_TIME else "FLIPPED (Repulsion)"
    text_info.set_text(f"Time: {t_val:.2f}s\nStatus: {status}")

    return R_patches + [right_circ, line_traj, text_info]

# Use number of frames from history
total_frames = len(hist['t'])
ani = FuncAnimation(fig, update, frames=total_frames, interval=20, blit=False)

plt.show()
