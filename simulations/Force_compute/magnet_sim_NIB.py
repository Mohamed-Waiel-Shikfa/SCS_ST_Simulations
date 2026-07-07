import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# --- Constants ---
mu0 = 4 * np.pi * 1e-7
Hc_n40 = -955000

# =====================================================================
# 1. DATASETS 
# =====================================================================
H_n40_q2 = np.array([
    -955000, -953000, -950000, -945000, -938000, -928000, 
    -915000, -895000, -860000, -800000, -700000, -500000, 
    -300000, -150000, -50000,  0
])
J_n40_q2 = np.array([
    0.0,      0.08,     0.20,     0.40,     0.60,     0.82,     
    1.02,     1.15,     1.22,     1.25,     1.26,     1.266,    
    1.268,    1.269,    1.27,     1.27
])

demag_curve = CubicSpline(H_n40_q2, J_n40_q2)
H_n40_fine = np.linspace(-955000, 0, 1000)
J_n40_fine = demag_curve(H_n40_fine)

def calculate_clamping_force(lm, g_min, g_max, w, h):
    print(f"--- 1. Reluctance Circuit (Finding the Operating Point) ---")
    
    lm /= 1000
    g_min /= 1000
    g_max /= 1000
    w /= 1000
    h /= 1000
    
    area = w * h
    perimeter = 2 * w + 2 * h
    g_avg = g_min if np.isclose(g_min, g_max) else (g_max - g_min) / np.log(g_max / g_min)
    
    # 1. GAP PERMEANCE
    x_gap = lm 
    P_main = (mu0 * area) / (g_avg + 1e-9) # Prevent absolute zero div in permeance
    P_edge = (mu0 * perimeter / np.pi) * np.log(1 + (np.pi * x_gap) / (g_avg + 1e-9))
    P_corner = 2 * mu0 * (x_gap - ((g_avg + 1e-9) / np.pi) * np.log(1 + (np.pi * x_gap) / (g_avg + 1e-9)))
    P_gap_total = P_main + P_edge + P_corner

    # 2. RETURN PATH PERMEANCE 
    L_ret = (2 * lm) + g_avg
    x_ret = np.sqrt(w * h)
    P_ret_main = (mu0 * area) / L_ret
    P_ret_edge = (mu0 * perimeter / np.pi) * np.log(1 + (np.pi * x_ret) / L_ret)
    P_ret_corner = 2 * mu0 * (x_ret - (L_ret / np.pi) * np.log(1 + (np.pi * x_ret) / L_ret))
    P_return = P_ret_main + P_ret_edge + P_ret_corner

    P_gap_series = 1 / ((1 / P_gap_total) + (1 / P_return))

    # 3. PARALLEL SELF-LEAKAGE
    x_leak = np.sqrt(w * h)
    P_leak_edge = (mu0 * perimeter / np.pi) * np.log(1 + (np.pi * x_leak) / lm)
    P_leak_corner = 2 * mu0 * (x_leak - (lm / np.pi) * np.log(1 + (np.pi * x_leak) / lm))
    P_leak_total = P_leak_edge + P_leak_corner

    # 4. SOLVER
    P_eff_per_magnet = (2 * P_gap_series) + P_leak_total
    m_load = (-lm * P_eff_per_magnet) / area

    def intersection_eq(Hm):
        return (m_load * Hm) - demag_curve(Hm)

    try:
        res = root_scalar(intersection_eq, bracket=[Hc_n40, 0], method='brentq')
        Hm_intersect = res.root
    except ValueError:
        Hm_intersect = 0

    Bm_intersect = m_load * Hm_intersect
    print(f"Gap: {g_avg*1000:.2f} mm | Op Point -> H0: {Hm_intersect:.0f} A/m, B0: {Bm_intersect:.4f} T")    

    # =================================================================
    # 5. STABILIZED COULOMBIAN MESH MODEL (With Shared-Energy Factor)
    # =================================================================
    print(f"--- 2. Coulombian Charge Mesh (Force Calculation) ---")
    
    Nx, Ny = 20, 40 
    x = np.linspace(-w/2, w/2, Nx)
    y = np.linspace(-h/2, h/2, Ny)
    xv, yv = np.meshgrid(x, y)
    
    x_flat = xv.flatten()
    y_flat = yv.flatten()
    
    dA = (w / Nx) * (h / Ny)
    J = Bm_intersect  
    q_patch = J * dA
    
    def calc_force_between_faces(z1, z2, q1_sign, q2_sign):
        dx = x_flat[:, np.newaxis] - x_flat[np.newaxis, :]
        dy = y_flat[:, np.newaxis] - y_flat[np.newaxis, :]
        dz = z2 - z1
        
        # 1. Standard Point-Charge Distance
        r_point = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 2. Prevent Singularity: Create an artificial mathematical floor for distance
        # based on the patch size to simulate flat surface area instead of a point.
        # This forces the math to max out gracefully at the theoretical limit.
        r_min_floor = np.sqrt(dA / (2 * np.pi))
        r_stabilized = np.maximum(r_point, r_min_floor)
        
        force_matrix = (q_patch**2 / (4 * np.pi * mu0)) * (q1_sign * q2_sign) * (dz / r_stabilized**3)
        return np.sum(force_matrix)

    z_m1_south = -lm          
    z_m1_north = 0            
    z_m2_south = g_avg        
    z_m2_north = g_avg + lm   
    
    F_attract_near = calc_force_between_faces(z_m1_north, z_m2_south, 1, -1)
    F_repel_top = calc_force_between_faces(z_m1_north, z_m2_north, 1, 1)
    F_repel_bottom = calc_force_between_faces(z_m1_south, z_m2_south, -1, -1)
    F_attract_far = calc_force_between_faces(z_m1_south, z_m2_north, -1, 1)
    
    # Sum the mesh forces
    rigid_dipole_force = abs(F_attract_near + F_repel_top + F_repel_bottom + F_attract_far)
    
    # 3. Apply the Shared Gap Energy rule (Divide by 2)
    # Transitions calculation from "Rigid Free-Space Dipoles" to "Shared Field Gap Tension"
    true_clamping_force = rigid_dipole_force / 2.0
    
    print(f"\n---> TRUE CLAMPING FORCE (Stabilized): {true_clamping_force:.4f} N <--- \n")

    return true_clamping_force, m_load, Hm_intersect, Bm_intersect
    
# Execute
force, m_load, Hm_intersect, Bm_intersect = calculate_clamping_force(
    lm=5, 
    g_min=0,  # 10 mm gap
    g_max=0, 
    w=10,
    h=20
)

# Plotting
B_loadline = m_load * H_n40_fine
plt.figure(figsize=(8, 6))
plt.plot(H_n40_fine, J_n40_fine, color='royalblue', linewidth=2.5, label='Demag Curve $J(H)$')
plt.scatter(H_n40_q2, J_n40_q2, color='darkblue', s=35, zorder=3, label='Dataset Points')
plt.plot(H_n40_fine, B_loadline, color='crimson', linestyle='--', linewidth=2, label='Reluctance Model')
plt.scatter(Hm_intersect, Bm_intersect, color='black', s=90, zorder=5, 
            label=f"Operating Point\n($H_0$: {Hm_intersect:.0f} A/m, $B_0$: {Bm_intersect:.2f} T)")

plt.xlim(-1_000_000, 5000)
plt.ylim(-0.1, 1.5)
plt.xlabel('Magnetic Field Strength H (A/m)', fontsize=12)
plt.ylabel('Magnetic Flux Density / Polarization (T)', fontsize=12)
plt.title('NIB N40 Complete Model (Coulombian Mesh Force)', fontsize=14, pad=15)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.show()