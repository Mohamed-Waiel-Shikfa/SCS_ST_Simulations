import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# =====================================================================
# 1. DATASETS (Optimized for Single Global Splines)
# =====================================================================

# Alnico 5-7: 13-point smooth dataset
H_alnico_q2 = np.array([
    -60000, -59000, -57500, -56000, -54000, -52000, 
    -49000, -45000, -38000, -30000, -20000, -10000, 0
])
J_alnico_q2 = np.array([
    0.0,    0.074,  0.305,  0.570,  0.800,  0.965,  
    1.100,  1.207,  1.285,  1.318,  1.338,  1.346,  1.35
])

# N40 Neodymium: 16-point dataset 
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

# =====================================================================
# 2. SINGLE-SPLINE INTERPOLATION
# =====================================================================
# Each material utilizes exactly ONE global smooth spline function
spline_alnico = CubicSpline(H_alnico_q2, J_alnico_q2)
spline_n40    = CubicSpline(H_n40_q2, J_n40_q2)

# High-resolution evaluation arrays
H_al_fine  = np.linspace(-60000, 0, 500)
J_al_fine  = spline_alnico(H_al_fine)

H_n40_fine = np.linspace(-955000, 0, 1000)
J_n40_fine = spline_n40(H_n40_fine)

# =====================================================================
# 3. VISUALIZATION
# =====================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

# First Panel: Alnico 5-7
ax1.plot(H_al_fine, J_al_fine, color='royalblue', linewidth=2.5, label='Demag Curve')
ax1.scatter(H_alnico_q2, J_alnico_q2, color='darkblue', s=35, zorder=3, label='Points')
ax1.set_xlim(-65000, 2000)
ax1.set_xlabel('Magnetic Field Strength H (A/m)', fontsize=12)
ax1.set_ylabel('Intrinsic Polarization J (Tesla)', fontsize=12)
ax1.set_title('Alnico 5-7 Intrinsic Demagnetization', fontsize=13)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right')

# Second Panel: N40 Neodymium
ax2.plot(H_n40_fine, J_n40_fine, color='crimson', linewidth=2.5, label='Demag Curve')
ax2.scatter(H_n40_q2, J_n40_q2, color='darkred', s=35, zorder=3, label='Points')
ax2.set_xlim(-1000000, 30000)
ax2.set_xlabel('Magnetic Field Strength H (A/m)', fontsize=12)
# ax2.set_ylabel('Intrinsic Polarization J (Tesla)', fontsize=12)
ax2.set_title('N40 Neodymium Intrinsic Demagnetization', fontsize=13)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower right')

# Third Panel: Alnico 5-7 and N40 Neodymium
ax3.plot(H_al_fine, J_al_fine, color='royalblue', linewidth=2.5, label='Alnico')
ax3.scatter(H_alnico_q2, J_alnico_q2, color='darkblue', s=35, zorder=3)
ax3.plot(H_n40_fine, J_n40_fine, color='crimson', linewidth=2.5, label='NIB')
ax3.scatter(H_n40_q2, J_n40_q2, color='darkred', s=35, zorder=3)
ax3.set_xlim(-1000000, 30000)
ax3.set_xlabel('Magnetic Field Strength H (A/m)', fontsize=12)
# ax3.set_ylabel('Intrinsic Polarization J (Tesla)', fontsize=12)
ax3.set_title('N40 Neodymium and Alnico 5-7 Intrinsic Demagnetization', fontsize=13)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='lower center')

plt.suptitle('Quadrant 2 Intrinsic Demagnetization Curves (Unified Smooth Splines)', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()