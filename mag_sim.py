import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import ellipk, ellipe

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================

# Simulation Control
SIM_TOTAL_TIME_US = 1000.0  # Total window 1ms
PULSE_WIDTH_US = 300.0      # Controller turns off at 300us
TIME_STEP_US = 0.5          # Resolution for physics loop
GRID_RES = 40               # Resolution of the spatial grid (kept moderate for speed)

# Magnet Parameters (Alnico 5)
MAG_DIAMETER_MM = 4.75
MAG_LENGTH_MM = 12.5
MAG_RADIUS = (MAG_DIAMETER_MM / 1000) / 2
MAG_LEN_M = MAG_LENGTH_MM / 1000

# Alnico 5 Hysteresis Properties (Approximate)
# Br ~ 1.25T, Hc ~ 50 kA/m (~640 Oe)
MAG_BR_T = 1.25
MAG_HC_A_M = 50000.0
MAG_SAT_M = (1.35 / (4 * np.pi * 1e-7)) # Saturation magnetization slightly above Br

# Coil Parameters
WIRE_DIAM_MM = 0.1
LAYERS = 3
TURNS_PER_LAYER = 125
TOTAL_TURNS = LAYERS * TURNS_PER_LAYER
COIL_RESISTANCE = 4.0       # Ohms

# Circuit Parameters
CAPACITANCE = 1e-3          # 1 mF
VOLTAGE_INIT = 30.0         # 30 V
DIODE_DROP = 0.7            # Flyback diode voltage drop

# Physical Constants
MU0 = 4 * np.pi * 1e-7

# ==========================================
# 2. PHYSICS MODELS
# ==========================================

def calculate_inductance(N, R, L):
    """Estimate inductance (Wheeler/Nagaoka)."""
    area = np.pi * R**2
    # Finite solenoid approximation
    # L = (mu0 * N^2 * A) / (l + 0.9r)
    denom = L + 0.9 * R
    inductance = (MU0 * (N**2) * area) / denom
    return inductance

def loop_field(r, z, R, I):
    """Calculate B-field (Br, Bz) of a single current loop off-axis."""
    epsilon = 1e-9

    # Ensure inputs are arrays
    r = np.atleast_1d(r)
    z = np.atleast_1d(z)

    alpha = r / R
    beta = z / R
    gamma = z / (r + epsilon)

    Q = (1 + alpha)**2 + beta**2
    k2 = (4 * alpha) / Q

    # Clamp k2 to avoid singularity at r=R, z=0
    k2 = np.clip(k2, 0, 0.999999)

    K = ellipk(k2)
    E = ellipe(k2)

    C = (MU0 * I) / (2 * np.pi * R * np.sqrt(Q))

    term1 = (1 - alpha**2 - beta**2) / ((1 - alpha)**2 + beta**2)
    Bz = C * (E * term1 + K)

    Br = np.zeros_like(r)
    # Mask for r > 0 to avoid division by zero
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

        # Derived from standard elliptic integral forms for Br
        term_num = 2 * z_s * R
        term_den = np.sqrt((R + r_s)**2 + z_s**2)
        factor = (MU0 * I) / (2 * np.pi) * (z_s / (r_s * term_den))

        E_term = ((R**2 + r_s**2 + z_s**2) / ((R - r_s)**2 + z_s**2)) * E_s
        Br[mask] = factor * (-K_s + E_term)

    return Br, Bz

def solenoid_field_map(I, current_turns, coil_radius, coil_len, r_grid, z_grid):
    """Superposition of loops to model solenoid/magnet."""
    Bz_total = np.zeros_like(r_grid)
    Br_total = np.zeros_like(r_grid)

    # 50 slices for numerical integration of the cylinder
    slices = 50
    z_positions = np.linspace(-coil_len/2, coil_len/2, slices)
    dI = I / slices # Current per slice

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

        # Steepness of the transition
        self.k = 5.0 / Hc

    def update(self, H_applied):
        """
        Update magnetization M based on applied H and history.
        Using a simple rate-independent envelope model.
        """
        # Upper and Lower branches of the major loop
        # M_up = Ms * tanh(k(H + Hc))
        # M_down = Ms * tanh(k(H - Hc))

        M_up = self.Ms * np.tanh(self.k * (H_applied + self.Hc))
        M_down = self.Ms * np.tanh(self.k * (H_applied - self.Hc))

        # Logic: If H is increasing, we tend towards M_down (lower curve).
        # If H is decreasing, we tend towards M_up (upper curve)?
        # Wait, standard hysteresis loops counter-clockwise.
        # Increasing H -> Follows Lower curve (rising) to Saturation?
        # Decreasing H -> Follows Upper curve (falling) from Saturation?

        # Let's verify:
        # At large -H, we are at -Ms. Increasing H, we stay low until +Hc. (Lower Branch)
        # At large +H, we are at +Ms. Decreasing H, we stay high until -Hc. (Upper Branch)

        # We need to know dH/dt direction, but here we just clamp to the envelope
        # If current M is above M_up, snap to M_up.
        # If current M is below M_down, snap to M_down.
        # Otherwise stay constant (minor loop approximation)

        if self.M > M_up:
            self.M = M_up
        elif self.M < M_down:
            self.M = M_down

        return self.M

# ==========================================
# 3. SIMULATION RUNNER
# ==========================================

print("--- Initializing Simulation ---")

effective_coil_radius = MAG_RADIUS + (WIRE_DIAM_MM/1000 * LAYERS)/2
L_COIL = calculate_inductance(TOTAL_TURNS, effective_coil_radius, MAG_LEN_M)
magnet = MagnetHysteresis(MAG_SAT_M, MAG_HC_A_M)

# Time Arrays
t_eval = np.arange(0, SIM_TOTAL_TIME_US, TIME_STEP_US) * 1e-6
results_I = []
results_V = []
results_M = []
results_H_applied = []

# Initial State
Q = VOLTAGE_INIT * CAPACITANCE
I = 0.0

print(f"Running time steps: {len(t_eval)} steps...")

for t in t_eval:
    t_us = t * 1e6

    # 1. Circuit Update (Euler integration for explicit control over switch)
    dt = TIME_STEP_US * 1e-6

    V_cap = Q / CAPACITANCE

    if t_us < PULSE_WIDTH_US:
        # Switch Closed: RLC Discharge
        # V_L = V_cap - I*R
        dIdt = (V_cap - I * COIL_RESISTANCE) / L_COIL
        dQdt = -I
    else:
        # Switch Open: Flyback Mode
        # Capacitor disconnected (or switch high impedance).
        # Current loops through diode/coil resistance.
        # V_L = -I*R - V_diode
        # If I is 0, it stays 0
        if I > 0:
            dIdt = (-I * COIL_RESISTANCE - DIODE_DROP) / L_COIL
            dQdt = 0 # Capacitor preserved
        else:
            I = 0
            dIdt = 0
            dQdt = 0

    I += dIdt * dt
    Q += dQdt * dt

    # 2. Magnet Physics
    # H field applied to magnet center by solenoid
    # H = N*I / L (approx for long solenoid, more accurate for center)
    H_coil = (TOTAL_TURNS * I) / MAG_LEN_M

    # Update Magnet State
    M_curr = magnet.update(H_coil)

    # Store
    results_I.append(I)
    results_V.append(V_cap)
    results_M.append(M_curr * MU0) # Store as Tesla (J approx)
    results_H_applied.append(H_coil)

# Convert to arrays
results_I = np.array(results_I)
results_V = np.array(results_V)
results_M = np.array(results_M)
results_H = np.array(results_H_applied)
time_us = t_eval * 1e6

# Stats
peak_current = np.max(results_I)
final_mag_state = results_M[-1]
initial_mag_state = results_M[0]

print(f"Peak Current: {peak_current:.2f} A")
print(f"Magnet State: {initial_mag_state:.2f}T -> {final_mag_state:.2f}T")

# ==========================================
# 4. FINAL FIELD MAP GENERATION
# ==========================================
print("--- Generating Field Map (Final State) ---")

# Fix: Meshgrid order for streamplot
# Z is horizontal (x-axis), R is vertical (y-axis)
z_vec = np.linspace(-0.02, 0.02, GRID_RES)
r_vec = np.linspace(0, 0.015, GRID_RES)
Z_grid, R_grid = np.meshgrid(z_vec, r_vec) # shape (len(r), len(z))

# 1. Coil Field (Use final current, likely 0, but let's check)
I_final = results_I[-1]
Br_coil, Bz_coil = solenoid_field_map(I_final, TOTAL_TURNS, effective_coil_radius, MAG_LEN_M, R_grid, Z_grid)

# 2. Magnet Field (Use final Magnetization)
# Convert M (Tesla) back to Equivalent Current
# I_equiv = (M/mu0) * L / N_turns_model
M_final_T = results_M[-1]
equiv_total_current = (M_final_T / MU0) * MAG_LEN_M
Br_pm, Bz_pm = solenoid_field_map(equiv_total_current, 50, MAG_RADIUS, MAG_LEN_M, R_grid, Z_grid)

# 3. Total
Br_total = Br_coil + Br_pm
Bz_total = Bz_coil + Bz_pm
B_mag_total = np.sqrt(Br_total**2 + Bz_total**2)

# ==========================================
# 5. VISUALIZATION
# ==========================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3)

# Plot A: Circuit Current & Voltage
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_us, results_I, 'r-', label='Current (A)')
ax1.set_ylabel('Current [A]', color='r')
ax1.set_xlabel('Time [$\mu$s]')
ax1.grid(True, alpha=0.3)
ax1.axvline(PULSE_WIDTH_US, color='k', linestyle='--', label='Cutoff')

ax1b = ax1.twinx()
ax1b.plot(time_us, results_V, 'b--', label='Cap Voltage (V)')
ax1b.set_ylabel('Voltage [V]', color='b')
ax1.set_title(f"Circuit Dynamics (Pulse {PULSE_WIDTH_US}$\mu$s)")

# Plot B: Magnetization & Applied H
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_us, results_M, 'g-', linewidth=2, label='Magnetization M (T)')
ax2.set_ylabel('Magnetization $\mu_0 M$ [T]', color='g')
ax2.set_xlabel('Time [$\mu$s]')
ax2.grid(True)
ax2.set_title("Magnetization Evolution")

ax2b = ax2.twinx()
ax2b.plot(time_us, results_H/1000, 'k:', alpha=0.5, label='Applied H (kA/m)')
ax2b.set_ylabel('Applied Coil Field H [kA/m]', color='k')

# Plot C: Hysteresis Loop (M vs H)
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(results_H/1000, results_M, color='purple', linewidth=2)
ax3.set_xlabel('Applied Field H [kA/m]')
ax3.set_ylabel('Magnetization [T]')
ax3.set_title("In-Simulation B-H Trajectory")
ax3.grid(True)
ax3.axvline(0, color='k', linewidth=0.5)
ax3.axhline(0, color='k', linewidth=0.5)

# Plot D: Final Magnetic Field Map
ax4 = fig.add_subplot(gs[1, :]) # Spans bottom row
pcm = ax4.pcolormesh(Z_grid*1000, R_grid*1000, B_mag_total, shading='auto', cmap='inferno', vmin=0, vmax=2.0)
fig.colorbar(pcm, ax=ax4, label='B Magnitude [T]')

# Streamplot Fix: Ensure 1D arrays match the grid dimensions relative to meshgrid
# Z_grid (x coords), R_grid (y coords)
# streamplot(x, y, u, v) where x, y can be 2D grids of same shape as u, v
ax4.streamplot(Z_grid*1000, R_grid*1000, Bz_total, Br_total,
              color='white', linewidth=0.8, density=1.2, arrowstyle='->')

# Magnet Outline
rect = plt.Rectangle((-MAG_LENGTH_MM/2, 0), MAG_LENGTH_MM, MAG_RADIUS*1000,
                     linewidth=2, edgecolor='cyan', facecolor='none', linestyle='-')
ax4.add_patch(rect)
# Coil Outline
rect_c = plt.Rectangle((-MAG_LENGTH_MM/2, MAG_RADIUS*1000), MAG_LENGTH_MM, (LAYERS*WIRE_DIAM_MM),
                     linewidth=1, edgecolor='orange', facecolor='none', linestyle='--')
ax4.add_patch(rect_c)

ax4.set_title(f"Final Field Map at {SIM_TOTAL_TIME_US:.0f}$\mu$s (State: {results_M[-1]:.2f}T)", fontsize=14)
ax4.set_xlabel('Axial Z [mm]')
ax4.set_ylabel('Radial R [mm]')
ax4.set_aspect('equal')
ax4.set_ylim(0, 15)

# Stats Text
txt = (f"STATS:\n"
       f"Pulse Width: {PULSE_WIDTH_US} us\n"
       f"Peak I: {peak_current:.1f} A\n"
       f"Initial M: {initial_mag_state:.2f} T\n"
       f"Final M: {final_mag_state:.2f} T")
ax4.text(0.02, 0.85, txt, transform=ax4.transAxes, color='white',
         bbox=dict(facecolor='black', alpha=0.5))

plt.tight_layout()
plt.show()
