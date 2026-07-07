import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# --- Constants ---
mu0 = 4 * np.pi * 1e-7
Hc_alnico = -60000

# =====================================================================
# 1. ALNICO DATASET
# =====================================================================
H_alnico_q2 = np.array([
    -60000, -59000, -57500, -56000, -54000, -52000, 
    -49000, -45000, -38000, -30000, -20000, -10000, 0
])
J_alnico_q2 = np.array([
    0.0,    0.074,  0.305,  0.570,  0.800,  0.965,  
    1.100,  1.207,  1.285,  1.318,  1.338,  1.346,  1.35
])

# Conversion to pure Flux Density (B)
B_alnico_q2 = J_alnico_q2 + (mu0 * H_alnico_q2)
base_demag_curve = CubicSpline(H_alnico_q2, B_alnico_q2)

def calc_Pc(length, diam):
    """Empirical formula for the Permeance Coefficient of a cylinder in free space."""
    return 1.0 + 2.8 * (length / diam)

def calculate_clamping_force(shape, lm, g_min, g_max, diameter):
    if shape != 'cyl':
        raise ValueError("This function is strictly for cylindrical magnets.")
        
    lm /= 1000
    g_min /= 1000
    g_max /= 1000
    diameter /= 1000
    
    radius = diameter / 2
    area = np.pi * (radius ** 2)
    g_avg = g_min if np.isclose(g_min, g_max) else (g_max - g_min) / np.log(g_max / g_min)

    # =================================================================
    # PHASE 1: OPEN-CIRCUIT SELF-DEMAGNETIZATION 
    # =================================================================
    Pc_open = calc_Pc(lm, diameter)
    m_open = -Pc_open * mu0
    
    def open_circuit_eq(Hm):
        return (m_open * Hm) - base_demag_curve(Hm)
        
    res_open = root_scalar(open_circuit_eq, bracket=[Hc_alnico, 0], method='brentq')
    H_open = res_open.root
    B_open = m_open * H_open
    
    # Recoil Line (mu_rec for Alnico is ~1.9 * mu0)
    mu_rec = 1.9 * mu0
    def recoil_curve(Hm):
        B_rec = B_open + mu_rec * (Hm - H_open)
        return np.minimum(B_rec, base_demag_curve(Hm))

    # =================================================================
    # PHASE 2: DYNAMIC CLAMPING SOLVER 
    # =================================================================
    # At g=0, they act as a cylinder of length 2*L. As g increases, they isolate.
    Pc_clamped = calc_Pc(2 * lm, diameter)
    Pc_g = Pc_open + (Pc_clamped - Pc_open) * np.exp(-g_avg / radius)
    
    m_load = -Pc_g * mu0
    def intersection_eq(Hm):
        return (m_load * Hm) - recoil_curve(Hm)

    res = root_scalar(intersection_eq, bracket=[Hc_alnico, 0], method='brentq')
    Hm_intersect = res.root
    Bm_intersect = m_load * Hm_intersect

    # =================================================================
    # PHASE 3: MAXWELL-COULOMB HYBRID FORCE MODEL
    # =================================================================
    
    # --- Component A: Perfect 1D Maxwell Contact Force (Controls g = 0 mm) ---
    F_maxwell = (Bm_intersect**2 * area) / (2 * mu0)

    # --- Component B: 3D Coulomb Mesh Force (Controls g > 0.3 mm fringing) ---
    N = 40 
    x = np.linspace(-radius, radius, N)
    y = np.linspace(-radius, radius, N)
    xv, yv = np.meshgrid(x, y)
    
    mask = (xv**2 + yv**2) <= radius**2
    x_flat = xv[mask]
    y_flat = yv[mask]
    
    dx = diameter / N
    dA = dx**2
    q_patch = Bm_intersect * dA
    
    def calc_force_between_faces(z1, z2, q1_sign, q2_sign):
        dx_matrix = x_flat[:, np.newaxis] - x_flat[np.newaxis, :]
        dy_matrix = y_flat[:, np.newaxis] - y_flat[np.newaxis, :]
        dz = z2 - z1
        r_point = np.sqrt(dx_matrix**2 + dy_matrix**2 + dz**2)
        
        r_min_floor = np.sqrt(dA / np.pi) 
        r_stabilized = np.maximum(r_point, r_min_floor)
        
        force_matrix = (q_patch**2 / (4 * np.pi * mu0)) * (q1_sign * q2_sign) * (dz / r_stabilized**3)
        return np.sum(force_matrix)

    F_near = calc_force_between_faces(0, g_avg, 1, -1)
    F_top = calc_force_between_faces(0, g_avg + lm, 1, 1)
    F_bottom = calc_force_between_faces(-lm, g_avg, -1, -1)
    F_far = calc_force_between_faces(-lm, g_avg + lm, -1, 1)
    
    # Mesh Output with your required shared-field "Divide by 2" logic
    F_coulomb = abs(F_near + F_top + F_bottom + F_far) / 2.0

    # --- Component C: Physics Blending ---
    # As the gap drops below the mesh grid resolution (dx), we mathematically transition 
    # from the 3D fringing dipole to the 1D parallel Maxwell limit.
    transition_weight = np.exp(-g_avg / (1.5 * dx))
    
    hybrid_force = (F_maxwell * transition_weight) + (F_coulomb * (1 - transition_weight))

    # print(f"Gap: {g_avg*1000:.2f} mm | Op Point -> B0: {Bm_intersect:.4f} T")
    # print(f"---> HYBRID CLAMPING FORCE: {hybrid_force:.4f} N <--- \n")
    print(hybrid_force)

    return hybrid_force

# Execute the loop across your gap data table
gaps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

for gap in gaps:
    calculate_clamping_force(
        shape='cyl', 
        lm=12.5, 
        g_min=gap, 
        g_max=gap, 
        diameter=4.75
    )