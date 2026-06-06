import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

# --- 1. System Dimensions and Constants ---
mu0 = 4 * np.pi * 1e-7
g = 9.81

wim = 0.010      # width in meters (10mm)
hem = 0.010      # height in meters (10mm)
am = hem * wim  # Physical cross-sectional area of the magnet
# am = np.pi * (0.00475 / 2)**2  # Physical cross-sectional area of the magnet

lm = 0.001      # length of one magnet (1.0mm)
# lm = 0.0125      # length of one magnet (12.5mm)
lg = 0.00015     # length of the gap (1mm)

# Alnico 5-7 Material Properties
Br = 1.35       # Remanence (Tesla)
Hc = -59000     # Coercivity (Amperes/meter)
alnico_density = 7300 # kg/m^3

print("--- Dimensions ---")
print(f"am = {am*1000000:.4f} mm²")
print(f"lm = {lm*1000:.4f} mm")
print(f"lg = {lg*1000:.4f} mm")

# --- 2. Calculate Total Permeance (Pt) with Fringing ---
# Using the effective area method for the permeance calculation
wim_eff = wim + lg
hem_eff = hem + lg
ag_eff = wim_eff * hem_eff
# ag_eff = am + lg**2

# Total permeance of the gap including fringing
Pt = mu0 * (ag_eff / lg)

# Mass and Weight of one magnet
mass_m = alnico_density * am * lm  # in kg
weight_g = mass_m * 1000           # in grams
force_gravity = mass_m * g         # in Newtons

# --- 3. Non-Linear Alnico 5-7 Demagnetization Curve ---
# These are realistic data points reflecting the "knee" of an Alnico 5-7 curve
# H is in A/m (negative in the 2nd quadrant), B is in Tesla
H_data = np.array([-59000, -56000, -52000, -45000, -30000, -15000, 0])
B_data = np.array([0.0,    0.50,   0.90,   1.15,   1.28,   1.33,   1.35])

# Smooth cubic spline interpolation
demag_curve = CubicSpline(H_data, B_data)

# --- 4. The Loadline ---
# Bm = m_load * Hm
m_load = (-2 * lm * Pt) / am

def loadline(Hm):
    return m_load * Hm

# --- 5. Numerical Solver for the Intersection ---
def intersection_equation(Hm):
    return loadline(Hm) - demag_curve(Hm)

Hm_guess = -30000
Hm_intersect = fsolve(intersection_equation, Hm_guess)[0]

# Operating point coordinates inside the magnet
Bm_intersect = loadline(Hm_intersect)

print("\n--- Accurate Non-Linear Operating Point ---")
print(f"Hm = {Hm_intersect:.2f} A/m")
print(f"Bm = {Bm_intersect:.4f} T")

# --- 7. Force Compute ---
# 1. Use Flux Conservation to find the flux density in the gap
# Bm * Am = Bg * Ag_eff  --> Bg = Bm * (Am / Ag_eff)
bg = Bm_intersect * (am / ag_eff)

# 2. Maxwell Stress Tensor Force equation for a single gap interface
# F = (Ag_eff * Bg^2) / (2 * mu0)
force_attraction = (ag_eff * (bg ** 2)) / (2 * mu0)

print("\n--- Force Balance ---")
print(f"Magnet Mass     = {weight_g:.3f} g")
print(f"Gravity Force   = {force_gravity:.5f} N")
print(f"Attraction Force= {force_attraction:.3f} N")
print(f"Attraction Force / Gravity Force ratio = {force_attraction/force_gravity:.3f} times")

# print(f"42 Magnet Mass     = {42 * weight_g:.3f} g")
# print(f"42 Gravity Force   = {42 * force_gravity:.5f} N")
# print(f"Attraction Force / Gravity Force ratio = {force_attraction/(42* force_gravity):.3f} times")

# if force_attraction > force_gravity:
#     print("Status          = Magnets will snap together!")
# else:
#     print("Status          = Gravity wins, or system is floating/weak.")

# --- 8. Module Force Compute ---

# Static coefficient of friction between rubber/latex balloon and the magnet face
# (Typical values for rubber-on-metal or rubber-on-plastic range from 0.5 to 0.9)
mu_s = 1

# The horizontal magnetic attraction acts as the Normal Force squeezing the rubber
normal_force = force_attraction

# Maximum static friction force supporting the weight vertically
max_friction_force = mu_s * normal_force

# Weight of the load trying to slide down
# Setup 1: Just a single magnet falling
weight_to_support_single = force_gravity

# Setup 2: A module of 42 magnets/mass equivalents falling
weight_to_support_module = 42 * force_gravity


# print("\n--- Force Balance & Friction ---")
# print(f"Magnet Mass              = {weight_g:.3f} g")
# print(f"Horizontal Normal Force  = {normal_force:.3f} N (Magnetic Attraction)")
# print(f"Max Vertical Friction    = {max_friction_force:.3f} N (at mu_s = {mu_s})")

# print(f"\n--- Scenario A: Single Magnet Face-to-Face ---")
# print(f"Gravity Pulling Down     = {weight_to_support_single:.5f} N")
# print(f"Friction Safety Factor   = {max_friction_force / weight_to_support_single:.2f}x")
# if max_friction_force > weight_to_support_single:
#     print("Status                   = STICKS! Friction prevents it from sliding down.")
# else:
#     print("Status                   = SLIDES DOWN! Magnetic force is strong, but not enough friction.")

# print(f"\n--- Scenario B: Entire Module (42x Mass) ---")
# print(f"Gravity Pulling Down     = {weight_to_support_module:.5f} N")
# print(f"Friction Safety Factor   = {max_friction_force / weight_to_support_module:.2f}x")
# if max_friction_force > weight_to_support_module:
#     print("Status                   = STICKS! One pair can hold all 42 masses.")
# else:
#     print("Status                   = SLIDES DOWN! The weight of 42 magnets overcomes the friction of 1 pair.")

# --- 6. Visualization ---
Hm_plot = np.linspace(-60000, 0, 500)
Bm_demag_plot = demag_curve(Hm_plot)
Bm_load_plot = loadline(Hm_plot)

plt.figure(figsize=(9, 6))
plt.plot(Hm_plot, Bm_demag_plot, label='Alnico 5-7 Curve', color='blue', linewidth=2)
plt.scatter(H_data, B_data, color='blue', marker='x', label='Data Points')
plt.plot(Hm_plot, Bm_load_plot, label='Circuit Loadline', color='red', linestyle='--')
plt.scatter(Hm_intersect, Bm_intersect, color='green', s=100, zorder=5,
            label=f'Operating Point\n({Hm_intersect:.0f} A/m, {Bm_intersect:.2f} T)')

plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle=':', alpha=0.7)
plt.title('Magnetic Circuit Operating Point & Force Calculation')
plt.xlabel('Magnetic Field Intensity, Hm (A/m)')
plt.ylabel('Magnetic Flux Density, Bm (T)')
plt.xlim(-65000, 2000)
plt.ylim(-0.1, 1.5)
plt.legend(loc='lower right')
plt.show()
