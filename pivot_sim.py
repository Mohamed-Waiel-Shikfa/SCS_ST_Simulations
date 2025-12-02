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
# Radius calc: 8 magnets of 4.75mm width forming a polygon.
# R = 4.75 / (2 * tan(pi/8)) ~ 5.74mm. Set to 5.9mm for tolerance.
CYL_RADIUS_MM = 5.9
MAG_LENGTH_MM = 12.5
MAG_DIAM_MM = 4.75
SHELL_THICKNESS_MM = 2.0 # Plastic shell thickness
GAP_MM = 0.5            # Initial separation (between shells)

# Physical Properties
MASS_CYLINDER = 0.05    # kg (50g)
# Inertia for hollow cylinder approx (radius increased by shell)
R_total_mm = CYL_RADIUS_MM + MAG_LENGTH_MM + SHELL_THICKNESS_MM
INERTIA_CYLINDER = 0.5 * MASS_CYLINDER * (R_total_mm/1000)**2

# Magnet Strength
# Dipole Moment m = M * Volume
# Alnico 5 Br ~ 1.35T. M = Br/mu0.
MU0 = 4 * np.pi * 1e-7
MAG_BR = 1.35
MAG_VOL = np.pi * ((MAG_DIAM_MM/2000)**2) * (MAG_LENGTH_MM/1000)
DIPOLE_MAG = (MAG_BR / MU0) * MAG_VOL

# Contact Model
K_SPRING = 15000.0      # N/m (Stiffer for smaller rigid bodies)
C_DAMP = 20.0           # Ns/m
MU_FRICTION = 0.9       # Friction coefficient

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
        mx = self.pos[0] + self.mag_r * np.cos(global_angles)
        my = self.pos[1] + self.mag_r * np.sin(global_angles)
        positions = np.column_stack((mx, my))

        # Dipole Vectors (Radial)
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

        # Damping
        self.vel *= 0.999
        self.omega *= 0.99

# ==========================================
# 3. PHYSICS ENGINE
# ==========================================

def compute_dipole_force(p1, m1, p2, m2):
    r_vec = p2 - p1
    r_len = np.linalg.norm(r_vec)
    if r_len < 1e-4: return np.zeros(2)

    # Simple Dipole-Dipole Force
    r_hat = r_vec / r_len

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
    R_mag_tip_m = (CYL_RADIUS_MM + MAG_LENGTH_MM) / 1000.0
    R_collision_m = R_mag_tip_m + (SHELL_THICKNESS_MM / 1000.0)

    # Left Body (Fixed)
    left_body = RigidBody(0, 0, 0, is_fixed=True)
    # Pattern: NSNS... (Mag 0 is N=+1)
    left_body.polarities = np.array([1, -1, 1, -1, 1, -1, 1, -1])

    # Right Body (Free)
    # Initial separation: Shells touching + GAP
    dist = 2 * R_collision_m + (GAP_MM / 1000.0)

    # Orientation: Mag 0 points West (pi) to face Left Body
    right_body = RigidBody(dist, 0, np.pi, is_fixed=False)
    # Pattern: SNSN... (Mag 0 is S=-1) -> Attraction at contact
    right_body.polarities = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    history = {
        't': [], 'rx': [], 'ry': [], 'rtheta': []
    }

    time_steps = int(TOTAL_TIME / DT)
    print(f"Simulating {TOTAL_TIME}s...")

    for step in range(time_steps):
        t = step * DT

        # --- EVENT: POLARITY FLIP ---
        if abs(t - FLIP_TIME) < DT/2:
            print(f"EVENT: Flipping Magnet at t={t:.2f}s")
            right_body.polarities[0] = 1.0 # Flip contact magnet to N (Repulsion)

        p1, m1 = left_body.get_magnet_states_global()
        p2, m2 = right_body.get_magnet_states_global()

        # 1. Magnetic Forces
        F_mag_total = np.zeros(2)
        Tau_mag_total = 0.0

        for i in range(NUM_MAGS):
            for j in range(NUM_MAGS):
                f_pair = compute_dipole_force(p1[i], m1[i], p2[j], m2[j])
                F_mag_total += f_pair

                # Torque = r x F (Lever from body center)
                lever = p2[j] - right_body.pos
                torque = lever[0]*f_pair[1] - lever[1]*f_pair[0]
                Tau_mag_total += torque

        # 2. Contact Constraint
        delta_pos = right_body.pos - left_body.pos
        dist_sq = np.dot(delta_pos, delta_pos)
        dist_val = np.sqrt(dist_sq)

        # Collision happens at Shell Radius
        collision_dist = 2 * R_collision_m
        F_contact = np.zeros(2)
        Tau_contact = 0.0

        if dist_val < collision_dist:
            pen = collision_dist - dist_val
            normal = delta_pos / dist_val

            # Spring-Damper
            rel_vel = right_body.vel
            v_normal = np.dot(rel_vel, normal)
            f_normal_mag = max(0, K_SPRING * pen - C_DAMP * v_normal)
            F_normal = f_normal_mag * normal

            # Friction (Tangential)
            # V_contact = V_cm + omega x r (where r is collision radius)
            contact_pt_rel = -normal * R_collision_m
            v_rot_x = -right_body.omega * contact_pt_rel[1]
            v_rot_y =  right_body.omega * contact_pt_rel[0]
            v_contact = right_body.vel + np.array([v_rot_x, v_rot_y])

            v_c_normal = np.dot(v_contact, normal) * normal
            v_tangent = v_contact - v_c_normal
            v_tan_mag = np.linalg.norm(v_tangent)

            if v_tan_mag > 1e-6:
                tangent_dir = v_tangent / v_tan_mag
                f_fric_mag = min(MU_FRICTION * f_normal_mag, C_DAMP * v_tan_mag)
                F_friction = -f_fric_mag * tangent_dir
            else:
                F_friction = np.zeros(2)

            F_contact = F_normal + F_friction
            Tau_contact = contact_pt_rel[0]*F_friction[1] - contact_pt_rel[1]*F_friction[0]

        right_body.apply_force_torque(F_mag_total + F_contact, Tau_mag_total + Tau_contact)

        if step % ANIMATION_SKIP == 0:
            history['t'].append(t)
            history['rx'].append(right_body.pos[0])
            history['ry'].append(right_body.pos[1])
            history['rtheta'].append(right_body.theta)

    return history

# ==========================================
# 4. VISUALIZATION
# ==========================================

hist = run_simulation()

print("--- Initializing Animation ---")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')
# Adjust view for geometry
ax.set_xlim(-0.04, 0.08)
ax.set_ylim(-0.05, 0.05)
ax.grid(True, alpha=0.3)
ax.set_title("EPM Pivot Dynamics (2mm Plastic Shells)")

# Chassis (Core)
left_core = Circle((0, 0), CYL_RADIUS_MM/1000, color='#888888', alpha=0.9, ec='black')
ax.add_patch(left_core)
right_core = Circle((0, 0), CYL_RADIUS_MM/1000, color='#888888', alpha=0.9, ec='black')
ax.add_patch(right_core)

# Shells (Plastic)
R_shell_m = (CYL_RADIUS_MM + MAG_LENGTH_MM + SHELL_THICKNESS_MM) / 1000.0
left_shell = Circle((0, 0), R_shell_m, color='orange', alpha=0.2, ec='orange', linestyle='--')
ax.add_patch(left_shell)
right_shell = Circle((0, 0), R_shell_m, color='orange', alpha=0.2, ec='orange', linestyle='--')
ax.add_patch(right_shell)

# Magnets
mag_patches_L = []
mag_patches_R = []

# Initialize Left Magnets
L_r = (CYL_RADIUS_MM + MAG_LENGTH_MM/2)/1000
L_angs = np.linspace(0, 2*np.pi, NUM_MAGS, endpoint=False)
L_pols = np.array([1, -1, 1, -1, 1, -1, 1, -1])

for i in range(NUM_MAGS):
    mx = L_r * np.cos(L_angs[i])
    my = L_r * np.sin(L_angs[i])
    c = 'red' if L_pols[i] > 0 else 'blue'
    rect = Rectangle((mx - 0.00625, my - 0.002375), 0.0125, 0.00475,
                     angle=np.degrees(L_angs[i]), rotation_point='center',
                     color=c, ec='black', lw=0.5)
    ax.add_patch(rect)
    mag_patches_L.append(rect)

# Initialize Right Magnets
for i in range(NUM_MAGS):
    rect = Rectangle((0,0), 0.0125, 0.00475, color='gray', ec='black', lw=0.5)
    ax.add_patch(rect)
    mag_patches_R.append(rect)

status_text = ax.text(0.05, 0.92, "", transform=ax.transAxes, fontsize=11,
                      bbox=dict(facecolor='white', alpha=0.8))

def update(frame):
    rx = hist['rx'][frame]
    ry = hist['ry'][frame]
    rtheta = hist['rtheta'][frame]
    t = hist['t'][frame]

    # Update Right Body
    right_core.set_center((rx, ry))
    right_shell.set_center((rx, ry))

    # Update Right Magnets
    mag_r = (CYL_RADIUS_MM + MAG_LENGTH_MM/2) / 1000.0
    local_angs = np.linspace(0, 2*np.pi, NUM_MAGS, endpoint=False)

    # Polarity Logic
    pols = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    if t >= FLIP_TIME: pols[0] = 1 # Flip contact magnet

    for i in range(NUM_MAGS):
        glob_ang = local_angs[i] + rtheta
        mx = rx + mag_r * np.cos(glob_ang)
        my = ry + mag_r * np.sin(glob_ang)

        c = 'red' if pols[i] > 0 else 'blue'

        # Robust rotation
        w, h = 0.0125, 0.00475
        dx = -w/2 * np.cos(glob_ang) - (-h/2) * np.sin(glob_ang)
        dy = -w/2 * np.sin(glob_ang) + (-h/2) * np.cos(glob_ang)

        mag_patches_R[i].set_xy((mx + dx, my + dy))
        mag_patches_R[i].set_angle(np.degrees(glob_ang))
        mag_patches_R[i].set_facecolor(c)

    s = "LOCKED (Attraction)" if t < FLIP_TIME else "FLIPPED (Repulsion)"
    status_text.set_text(f"Time: {t:.3f}s\nMode: {s}")

    return mag_patches_R + [right_core, right_shell, status_text]

ani = FuncAnimation(fig, update, frames=len(hist['t']), interval=20, blit=False)
plt.show()
