import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

# --- 1. System Dimensions and Constants ---
mu0 = 4 * np.pi * 1e-7 # Better precision for mu0 (approx 1.2566e-6)
g = 9.81

wim = 0.01      # width in meters (10mm)
hem = 0.01      # height in meters (10mm)
am = hem * wim  # Area of the magnet

lm = 0.001     # length of one magnet (0.3mm)
lg = 0.0001     # length of the gap (0.1mm)

# Alnico 5-7 Material Properties
Br = 1.35       # Remanence (Tesla)
Hc = -59000     # Coercivity (Amperes/meter) - Negative for the 2nd quadrant
alnico_density = 7300

# --- 2. Calculate Total Permeance (Pt) with Fringing ---
# Using the effective area method
wim_eff = wim + lg
hem_eff = hem + lg
ag = wim_eff * hem_eff

# Total permeance of the gap including fringing
Pt = mu0 * (ag / lg)

# Weight of one magnet
weightm = alnico_density * am * lm

# --- 3. Non-Linear Alnico 5-7 Demagnetization Curve ---
# These are realistic data points reflecting the "knee" of an Alnico 5-7 curve
# H is in A/m (negative in the 2nd quadrant), B is in Tesla
H_data = np.array([-59000, -56000, -52000, -45000, -30000, -15000, 0])
B_data = np.array([0.0,    0.50,   0.90,   1.15,   1.28,   1.33,   1.35])

# Create a smooth cubic interpolation from the data points
# This allows us to get a B value for ANY given H value along the curve
demag_curve = CubicSpline(H_data, B_data)

# --- 4. The Loadline ---
# Bm = m_load * Hm
m_load = (-2 * lm * Pt) / am

def loadline(Hm):
    return m_load * Hm

# --- 5. Numerical Solver for the Intersection ---
# We want to find where Loadline(Hm) = DemagCurve(Hm)
# Meaning: Loadline(Hm) - DemagCurve(Hm) = 0

def intersection_equation(Hm):
    return loadline(Hm) - demag_curve(Hm)

# We use fsolve to find the root. We give it an initial guess, like Hc/2 (-30000)
Hm_guess = -30000
Hm_intersect = fsolve(intersection_equation, Hm_guess)[0]

# Plug the found Hm back into either equation to get Bm
Bm_intersect = loadline(Hm_intersect)

print("--- Accurate Non-Linear Operating Point ---")
print(f"Hm = {Hm_intersect:.2f} A/m")
print(f"Bm = {Bm_intersect:.4f} T")

# --- 7. Force compute ---
hg = -2 * Hm_intersect * lm / lg

bg1 = mu0 * hg
bg2 = Bm_intersect * am / ag

force = -ag * bg2**2 / (2 * mu0)

force_gravity = g * weightm

print("--- Force ---")
print(f"bg1 = {bg1:.10f} T")
print(f"bg2 = {bg2:.10f} T")
print(f"weight = {weightm*1000:.2f} g")
print(f"Fattrac = {force:.2f} N")
print(f"Fgravity = {force_gravity:.2f} N")

# --- 6. Visualization ---
# Generate high-resolution points for plotting the smooth curve
Hm_plot = np.linspace(-59000, 0, 500)
Bm_demag_plot = demag_curve(Hm_plot)
Bm_load_plot = loadline(Hm_plot)

plt.figure(figsize=(9, 6))

# Plot the smooth non-linear demagnetization curve
plt.plot(Hm_plot, Bm_demag_plot, label='Alnico 5-7 (Cubic Spline Fit)', color='blue', linewidth=2)

# Plot the original data points to show what the spline is fitting
plt.scatter(H_data, B_data, color='blue', marker='x', label='Manufacturer Data Points')

# Plot the Loadline
plt.plot(Hm_plot, Bm_load_plot, label='Magnetic Circuit Loadline', color='red', linestyle='--')

# Plot the Intersection Point
plt.scatter(Hm_intersect, Bm_intersect, color='green', s=100, zorder=5,
            label=f'Operating Point\n({Hm_intersect:.0f} A/m, {Bm_intersect:.2f} T)')

plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle=':', alpha=0.7)
plt.title('Accurate Operating Point with Non-Linear Alnico Curve')
plt.xlabel('Magnetic Field Intensity, Hm (A/m)')
plt.ylabel('Magnetic Flux Density, Bm (T)')
plt.xlim(-65000, 2000)
plt.ylim(-0.1, 1.5)
plt.legend(loc='lower right')

plt.show()

