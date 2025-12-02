import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import ellipk, ellipe

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================

# Simulation Control
SIM_LENGTH_US = 300.0       # Simulation duration in microseconds
TIME_STEP_US = 0.1          # Time step for numerical approximation
GRID_RES = 60               # Resolution of the spatial grid (higher = slower)

# Magnet Parameters (Alnico 5)
MAG_DIAMETER_MM = 4.75
MAG_LENGTH_MM = 12.5
MAG_BR = 1.25               # Remanence of Alnico 5 (Tesla) ~1.2-1.3T
MAG_RADIUS = (MAG_DIAMETER_MM / 1000) / 2
MAG_LEN_M = MAG_LENGTH_MM / 1000

# Coil Parameters
WIRE_DIAM_MM = 0.1
WIRE_LEN_M = 11.0
LAYERS = 3
TURNS_PER_LAYER = 125
TOTAL_TURNS = LAYERS * TURNS_PER_LAYER
COIL_RESISTANCE = 4.0       # Ohms

# Circuit Parameters
CAPACITANCE = 1e-3          # 1 mF
VOLTAGE_INIT = 30.0         # 30 V

# Physical Constants
MU0 = 4 * np.pi * 1e-7      # Vacuum permeability

# ==========================================
# 2. PHYSICS ENGINE
# ==========================================

def calculate_inductance(N, R, L):
    """
    Estimate inductance of a finite solenoid (Wheeler's formula approximation).
    L (Henries) = (u0 * N^2 * A) / (l + 0.9r) roughly, or standard long solenoid.
    Using Nagaoka coefficient approximation for short solenoids.
    """
    area = np.pi * R**2
    # Simple solenoid approx for L >> R, adapted for finite geometry
    # L = (mu0 * N^2 * A) / length * Correction
    # This is a basic estimation to drive the circuit dynamic
    inductance = (MU0 * (N**2) * area) / L
    return inductance

def loop_field(r, z, R, I):
    """
    Calculate B-field (Br, Bz) of a single current loop using Elliptic Integrals.
    r, z: Coordinates where field is measured
    R: Radius of the loop
    I: Current
    """
    # Avoid division by zero at the wire location
    epsilon = 1e-9

    alpha = r / R
    beta = z / R
    gamma = z / r

    Q = (1 + alpha)**2 + beta**2
    k2 = (4 * alpha) / Q

    # Elliptic integrals
    K = ellipk(k2)
    E = ellipe(k2)

    # Common factor
    C = (MU0 * I) / (2 * np.pi * R * np.sqrt(Q))

    # Bz component
    term1 = (1 - alpha**2 - beta**2) / ((1 - alpha)**2 + beta**2)
    Bz = C * (E * term1 + K)

    # Br component (Radial)
    # Handle on-axis singularity (r=0)
    Br = np.zeros_like(r)
    mask = r > epsilon

    if np.any(mask):
        r_safe = r[mask]
        alpha_s = r_safe / R
        beta_s = z[mask] / R
        Q_s = (1 + alpha_s)**2 + beta_s**2
        k2_s = (4 * alpha_s) / Q_s
        K_s = ellipk(k2_s)
        E_s = ellipe(k2_s)
        C_s = (MU0 * I) / (2 * np.pi * R * np.sqrt(Q_s))

        term2 = (1 + alpha_s**2 + beta_s**2) / ((1 - alpha_s)**2 + beta_s**2)
        Br[mask] = C_s * (gamma[mask] * (E_s * term2 - K_s)) # Note: Using calculated gamma

        # Explicit formula correction for Br to avoid gamma ambiguity
        # Br = (mu0 I z) / (2 pi r sqrt(Q)) * [-K + ((R^2 + r^2 + z^2)/((R-r)^2 + z^2)) * E]
        # Re-implementing strictly:
        numerator = 2 * z[mask] * R
        den_k = np.sqrt((R + r_safe)**2 + z[mask]**2)
        term_e_num = R**2 + r_safe**2 + z[mask]**2
        term_e_den = (R - r_safe)**2 + z[mask]**2

        Br[mask] = (MU0 * I / (2 * np.pi)) * (z[mask] / (r_safe * den_k)) * (-K_s + (term_e_num / term_e_den) * E_s)

    return Br, Bz

def solenoid_field_map(I, current_turns, coil_radius, coil_len, r_grid, z_grid):
    """
    Sum the fields of N loops distributed along the length.
    """
    Bz_total = np.zeros_like(r_grid)
    Br_total = np.zeros_like(r_grid)

    # Discretize the solenoid into loops
    z_positions = np.linspace(-coil_len/2, coil_len/2, int(current_turns))

    # Current per loop (Total Current I is passed through each loop in series)
    # The 'I' parameter here is the wire current.

    for z_pos in z_positions:
        # Shift z_grid relative to this loop
        dz = z_grid - z_pos
        br, bz = loop_field(r_grid, dz, coil_radius, I)
        Bz_total += bz
        Br_total += br

    return Br_total, Bz_total

# ==========================================
# 3. CIRCUIT SIMULATION (RLC Discharge)
# ==========================================

# Calculate Inductance
# Effective radius is slightly larger than magnet due to wire thickness
effective_coil_radius = MAG_RADIUS + (WIRE_DIAM_MM/1000 * LAYERS)/2
L_COIL = calculate_inductance(TOTAL_TURNS, effective_coil_radius, MAG_LEN_M)

def circuit_dynamics(t, y):
    """
    Differential equation for RLC circuit.
    y[0] = Charge (Q)
    y[1] = Current (I) = dQ/dt
    """
    Q, I = y

    # Kirchhoff's Voltage Law: Vc - Vr - Vl = 0
    # Q/C - I*R - L*(dI/dt) = 0
    # dI/dt = (Q/C - I*R) / L

    dQdt = I
    dIdt = (Q/CAPACITANCE - I*COIL_RESISTANCE) / L_COIL

    return [dQdt, dIdt]

print("--- Running Circuit Simulation ---")
t_span = (0, SIM_LENGTH_US * 1e-6)
t_eval = np.linspace(0, SIM_LENGTH_US * 1e-6, int(SIM_LENGTH_US/TIME_STEP_US))
y0 = [VOLTAGE_INIT * CAPACITANCE, 0.0] # Initial Charge, Initial Current

sol = solve_ivp(circuit_dynamics, t_span, y0, t_eval=t_eval)

time_us = sol.t * 1e6
current_A = sol.y[1]
voltage_C = sol.y[0] / CAPACITANCE

# Statistics
peak_current = np.max(current_A)
peak_time = time_us[np.argmax(current_A)]
final_current = current_A[-1]
avg_current = np.mean(current_A)

print(f"Peak Current: {peak_current:.2f} A at {peak_time:.1f} us")
print(f"Voltage dropped from {VOLTAGE_INIT}V to {voltage_C[-1]:.2f}V")
print(f"Inductance Estimated: {L_COIL*1e6:.2f} uH")

# ==========================================
# 4. FIELD CALCULATION
# ==========================================

print("--- Calculating Magnetic Fields (This may take a moment) ---")

# Define spatial grid (Cross section in r-z plane)
# r from 0 to 10mm, z from -15mm to 15mm
r_vec = np.linspace(0, 0.01, GRID_RES)
z_vec = np.linspace(-0.015, 0.015, GRID_RES)
R_grid, Z_grid = np.meshgrid(r_vec, z_vec)

# 1. COIL FIELD (At Peak Current)
# To save compute, we simulate the equivalent of 3 layers stacked as one effective solenoid
# Since layers are tight, we assume one radius (average radius).
Br_coil, Bz_coil = solenoid_field_map(peak_current, TOTAL_TURNS, effective_coil_radius, MAG_LEN_M, R_grid, Z_grid)
B_mag_coil = np.sqrt(Br_coil**2 + Bz_coil**2)

# 2. MAGNET FIELD (Alnico 5)
# Modeled as a solenoid with equivalent surface current density
# B_center approx = mu0 * (N/L) * I
# We know Br (Remanence). So Equivalent Current Total (I_total) = (Br * L) / mu0
# Because it's a solid block, we model it as 'N' loops of current spread out
equiv_total_current = (MAG_BR * MAG_LEN_M) / MU0
# We distribute this current over, say, 50 loops to make it smooth
mag_model_turns = 50
I_per_loop_mag = equiv_total_current / mag_model_turns

Br_pm, Bz_pm = solenoid_field_map(I_per_loop_mag, mag_model_turns, MAG_RADIUS, MAG_LEN_M, R_grid, Z_grid)
B_mag_pm = np.sqrt(Br_pm**2 + Bz_pm**2)

# 3. COMBINED FIELD
Br_total = Br_coil + Br_pm
Bz_total = Bz_coil + Bz_pm
B_mag_total = np.sqrt(Br_total**2 + Bz_total**2)

# ==========================================
# 5. VISUALIZATION
# ==========================================

fig = plt.figure(figsize=(18, 10))
plt.suptitle(f"Electropermanent Magnet Simulation\nTime: {peak_time:.1f} $\mu$s (Peak Pulse) | I_peak: {peak_current:.2f} A", fontsize=16)

# Helper for plotting
def plot_field(ax, Br, Bz, Bmag, title):
    # Field Magnitude (Heatmap)
    # Use log scale for color to see details inside and outside magnet
    pcm = ax.pcolormesh(Z_grid*1000, R_grid*1000, Bmag, shading='auto', cmap='inferno', vmin=0, vmax=2.0)

    # Streamlines (Field Lines)
    # Density controls line spacing
    ax.streamplot(Z_grid*1000, R_grid*1000, Bz, Br, color='white', linewidth=0.8, density=1.0, arrowstyle='->')

    # Draw Magnet/Coil Geometry
    rect = plt.Rectangle((-MAG_LENGTH_MM/2, 0), MAG_LENGTH_MM, MAG_RADIUS*1000,
                         linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Z (axial) [mm]')
    ax.set_ylabel('R (radial) [mm]')
    ax.set_aspect('equal')
    return pcm

# Plot 1: Coil Only
ax1 = fig.add_subplot(2, 3, 1)
pcm1 = plot_field(ax1, Br_coil, Bz_coil, B_mag_coil, "Coil Field Only (Pulse Peak)")
fig.colorbar(pcm1, ax=ax1, label='B Field [Tesla]')

# Plot 2: Magnet Only
ax2 = fig.add_subplot(2, 3, 2)
pcm2 = plot_field(ax2, Br_pm, Bz_pm, B_mag_pm, "Magnet Field Only (Alnico 5)")
fig.colorbar(pcm2, ax=ax2, label='B Field [Tesla]')

# Plot 3: Addition
ax3 = fig.add_subplot(2, 3, 3)
pcm3 = plot_field(ax3, Br_total, Bz_total, B_mag_total, "Combined Field (Interaction)")
fig.colorbar(pcm3, ax=ax3, label='B Field [Tesla]')

# Plot 4: Circuit Dynamics (Current vs Time)
ax4 = fig.add_subplot(2, 1, 2)
ax4.plot(time_us, current_A, label='Current (A)', color='orange', linewidth=2)
ax4.plot(time_us, voltage_C, label='Capacitor Voltage (V)', color='blue', linestyle='--')
ax4.set_title("Circuit Discharge Dynamics (0 - 300 $\mu$s)", fontsize=14)
ax4.set_xlabel("Time ($\mu$s)")
ax4.set_ylabel("Amplitude")
ax4.grid(True, alpha=0.3)
ax4.legend()

# Stats Box
stats_text = (
    f"SIMULATION STATS:\n"
    f"----------------\n"
    f"Max Current: {peak_current:.2f} A\n"
    f"Avg Current: {avg_current:.2f} A\n"
    f"Pulse Energy: {0.5 * CAPACITANCE * (VOLTAGE_INIT**2 - voltage_C[-1]**2):.2f} J\n"
    f"Max Coil B-Field (Center): {np.max(B_mag_coil):.2f} T\n"
    f"Max Magnet B-Field (Remanence): {np.max(B_mag_pm):.2f} T\n"
    f"Total B-Field Peak: {np.max(B_mag_total):.2f} T\n"
)
ax4.text(0.02, 0.5, stats_text, transform=ax4.transAxes,
         bbox=dict(facecolor='white', alpha=0.8), fontsize=11, family='monospace')

plt.tight_layout()
plt.show()

# Print detailed stats to console as well
print("\n=== FINAL STATISTICS ===")
print(f"Max Current in Coil: {peak_current:.4f} A")
print(f"Average Current over {SIM_LENGTH_US}us: {avg_current:.4f} A")
print(f"Magnetic Field (Coil Only) Max: {np.max(B_mag_coil):.4f} T")
print(f"Magnetic Field (Magnet Only) Max: {np.max(B_mag_pm):.4f} T")
print(f"The simulation assumes the magnet is fully magnetized (B_r = {MAG_BR}T).")
print(f"The visualization shows the state at t = {peak_time:.1f} microseconds.")
