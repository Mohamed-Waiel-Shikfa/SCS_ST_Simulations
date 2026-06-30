import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# --- Constants ---
mu0 = 4 * np.pi * 1e-7
Hc_alnico = -65000

# # =====================================================================
# 1. DATASETS (Optimized for Single Global Splines)
# =====================================================================

# Alnico 5-7: 13-point smooth dataset
H_alnico_q2 = np.array([
    -59500, -59000, -57500, -56000, -54000, -52000, 
    -49000, -45000, -38000, -30000, -20000, -10000, 0
])
J_alnico_q2 = np.array([
    0.0,    0.074,  0.305,  0.570,  0.800,  0.965,  
    1.100,  1.207,  1.285,  1.318,  1.338,  1.346,  1.35
])
# =====================================================================
# 2. SINGLE-SPLINE INTERPOLATION
# =====================================================================
# Each material utilizes exactly ONE global smooth spline function
demag_curve = CubicSpline(H_alnico_q2, J_alnico_q2)
# High-resolution evaluation arrays
H_al_fine  = np.linspace(-59500, 0, 500)
J_al_fine  = demag_curve(H_al_fine)

def calculate_clamping_force(shape, lm, g_min, g_max, w=None, h=None, diameter=None):
    """
    Calculates clamping force using the Roters-Log-Mean Hybrid method.
    All dimension are in mm and will be converted to m.
    
    Parameters:
    - shape: 'rectangular' or 'cylindrical'
    - lm: magnet length
    - g_min: closest gap
    - g_max: furthest gap
    - w, h: width and height (required if shape is 'rectangular')
    - diameter: face diameter (required if shape is 'cylindrical')
    """
    
    # --- Unit conversion ---
    lm/=1000
    g_min/=1000
    g_max/=1000

    # --- Geometry Calculations based on Shape ---
    if shape == 'rec':
        if w is None or h is None:
            raise ValueError("Both 'w' and 'h' must be provided for a rectangular shape.")
        print("--- Rectangular Magnet ---")
        w/=1000
        h/=1000
        area = w * h
        perimeter = 2 * w + 2 * h
        # Roters: 2 corners at g_min, 2 corners at g_max
        P_corner = (2 * 0.077 * mu0 * g_min) + (2 * 0.077 * mu0 * g_max)
    elif shape == 'cyl':
        if diameter is None:
            raise ValueError(" 'diameter' must be provided for a cylindrical shape.")
        print("--- Cylindrical Magnet ---")
        diameter/=1000
        radius = diameter / 2
        area = np.pi * (radius ** 2)
        perimeter = 2 * np.pi * radius
        # Cylindrical shapes have no sharp corners
        P_corner = 0.0   
    else:
        raise ValueError("Invalid shape. Choose 'rec' or 'cyl'.")

    print(f"Am: {area*100**2:.4f} cm²")    
    print(f"perimeter: {perimeter*100:.4f} cm")    

    # Calculate the equivalent average physical gap distance
    g_avg = g_min if np.isclose(g_min, g_max) else (g_max - g_min) / np.log(g_max / g_min)
    print(f"Lgeq: {g_avg*1000:.4f} mm")    

    # --- 1. Main Gap Permeance (Log-Mean) ---
    P_main = (mu0 * area) / g_avg
    print(f"P_main: {P_main:.4e} H")    

    # --- 2. Edge Fringing Permeance (Roters' Half-Cylinders) ---
    P_edge = mu0 * 0.26 * perimeter
    print(f"P_edge: {P_edge:.4e} H")    
    print(f"P_corner: {P_corner:.4e} H")    

    # --- 3. Total Adjusted Circuit Permeance ---
    Pt = P_main + P_edge + P_corner
    print(f"Pt: {Pt:.4e} H")    

    # --- 4. Magnetic Circuit Solver ---
    # Loadline slope (Bm = m_load * Hm)
    m_load = (-2 * lm * Pt) / area

    def intersection_eq(Hm):
        return (m_load * Hm) - demag_curve(Hm)

    try:
        res = root_scalar(intersection_eq, bracket=[Hc_alnico, 0], method='brentq')
        Hm_intersect = res.root
    except ValueError:
        Hm_intersect = 0

    Bm_intersect = m_load * Hm_intersect
    print(f"Ho: {Hm_intersect:.0f} At/m")    
    print(f"Bo: {Bm_intersect:.4f} T")    
    

    # --- 5. Force Calculation ---
    # Back-calculate the Roters equivalent area
    A_roters = (Pt * g_avg) / mu0
    print(f"Ag: {A_roters*100**2:.4f} cm²")    

    # Calculate Flux Density in the gap
    Bg = Bm_intersect * (area / A_roters)
    print(f"Bg: {Bg:.4f} T")    

    # Maxwell Stress Tensor
    force = (A_roters * (Bg ** 2)) / (2 * mu0)
    
    print(f"Clamping Force: {force:.4f} N")

    return force, m_load, Hm_intersect, Bm_intersect

# 1. Unpack your updated function return values
# cylinder face to face
force, m_load, Hm_intersect, Bm_intersect = calculate_clamping_force(
    shape='cyl', 
    lm=12.5, 
    g_min=0.84, 
    g_max=0.84, 
    diameter=4.75
)

# cylinder angled
# force, m_load, Hm_intersect, Bm_intersect = calculate_clamping_force(
#     shape='cyl', 
#     lm=12.5, 
#     g_min=8.42, 
#     g_max=15.14, 
#     diameter=4.75
# )

# square face to face
# force, m_load, Hm_intersect, Bm_intersect = calculate_clamping_force(
#     shape='rec', 
#     lm=5, 
#     g_min=2.8, 
#     g_max=2.8, 
#     w=10,
#     h=20
# )

# 2. Compute the loadline across the global fine H array
B_loadline = m_load * H_al_fine

# 3. Create the single plot
plt.figure(figsize=(8, 6))

# Plot the demagnetization spline curve and data points
plt.plot(H_al_fine, J_al_fine, color='royalblue', linewidth=2.5, label='Demag Curve $J(H)$')
plt.scatter(H_alnico_q2, J_alnico_q2, color='darkblue', s=35, zorder=3, label='Dataset Points')

# Plot the circuit loadline
plt.plot(H_al_fine, B_loadline, color='crimson', linestyle='--', linewidth=2, label='Circuit Loadline')

# Highlight the operating/intersection point
plt.scatter(Hm_intersect, Bm_intersect, color='black', s=90, zorder=5, 
            label=f"Operating Point\n($H_0$: {Hm_intersect:.0f} A/m, $B_0$: {Bm_intersect:.2f} T)")

# Axis and labels styling
plt.xlim(-65000, 2000)
plt.ylim(-0.1, 1.5)
plt.xlabel('Magnetic Field Strength H (A/m)', fontsize=12)
plt.ylabel('Magnetic Flux Density / Polarization (T)', fontsize=12)
plt.title('Alnico 5-7 Intrinsic Demagnetization & Operating Point', fontsize=14, pad=15)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()