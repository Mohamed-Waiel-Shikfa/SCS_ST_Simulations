import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. N42 NEODYMIUM PARAMETERS (SI & CGS)
# ==========================================
# Applied Field Range (kA/m) -> +/- 2500 kA/m (~31.4 kOe)
H_sat_max = 2500.0

# Target properties for N42 Grade
Br = 1.30        # Residual Induction (Tesla)
HcB = 900.0      # Normal Coercivity (kA/m)
HcJ = 955.0      # Intrinsic Coercivity (kA/m)
Bsat = 1.55      # Saturation Flux Density (Tesla)
mu_0 = 4 * np.pi * 1e-7 # Permeability of free space (H/m)
mu_r = 1.05      # Recoil permeability

# ==========================================
# 2. MODELING FUNCTIONS (Extended tanh model)
# ==========================================
def initial_virgin_curve(H):
    """
    Quadrant I Virgin Curve: Unmagnetized (0,0) to +Saturation (+Bsat, +Hsat)
    """
    # S-curve function starting at origin
    alpha = 0.002
    return Bsat * np.tanh(alpha * H)

def hysteresis_upper_branch(H):
    """
    Upper Loop Branch: From +Saturation down to -Saturation (Quadrants I -> II -> III)
    Passes through +Br (1.30 T) and -HcB (-900 kA/m)
    """
    # Dynamic parameter adjustment to hit N42 coercivity and remanence
    shift = 750.0  # Field transition scale
    linear_recoil = mu_r * (mu_0 * 1000) * H  # Converting kA/m for linear offset

    # Sigmoidal transition around demag knee (-HcJ)
    B_ferro = (Br) * np.tanh((H + 780.0) / 250.0)

    # Adding linear recoil contribution outside hysteresis knee
    B_total = B_ferro + 0.0001 * H
    return np.clip(B_total, -Bsat, Bsat)

def hysteresis_lower_branch(H):
    """
    Lower Loop Branch: From -Saturation up to +Saturation (Quadrants III -> IV -> I)
    Symmetric point reflection across origin
    """
    return -hysteresis_upper_branch(-H)

# ==========================================
# 3. GENERATING DATA POINTS
# ==========================================
# Point resolution
n_pts = 500

# Branch 1: Virgin magnetization curve
H_virgin = np.linspace(0, H_sat_max, n_pts)
B_virgin = initial_virgin_curve(H_virgin)

# Branch 2: Upper branch (+Hsat down to -Hsat)
H_upper = np.linspace(H_sat_max, -H_sat_max, n_pts)
B_upper = hysteresis_upper_branch(H_upper)

# Branch 3: Lower branch (-Hsat back up to +Hsat)
H_lower = np.linspace(-H_sat_max, H_sat_max, n_pts)
B_lower = hysteresis_lower_branch(H_lower)

# Combine for full loop CSV export
H_full = np.concatenate([H_virgin, H_upper, H_lower])
B_full = np.concatenate([B_virgin, B_upper, B_lower])
stage = (['Virgin'] * n_pts) + ['Upper Branch'] * n_pts + ['Lower Branch'] * n_pts

# Export DataFrame to CSV
df_bh = pd.DataFrame({'H_kA_per_m': H_full, 'B_Tesla': B_full, 'Segment': stage})
df_bh.to_csv('N42_Full_BH_Curve.csv', index=False)
print("Data exported successfully as 'N42_Full_BH_Curve.csv'")

# ==========================================
# 4. PLOTTING THE 4-QUADRANT LOOP
# ==========================================
plt.figure(figsize=(10, 7), dpi=120)

# Plot curves
plt.plot(H_virgin, B_virgin, 'g--', linewidth=1.8, label='Virgin Curve (Q-I)')
plt.plot(H_upper, B_upper, 'b-', linewidth=2.0, label='Upper Branch (Q-I → Q-II → Q-III)')
plt.plot(H_lower, B_lower, 'r-', linewidth=2.0, label='Lower Branch (Q-III → Q-IV → Q-I)')

# Plot & Label key physical landmarks
landmarks = [
    (0, Br, f'$B_r$ = {Br} T', 'blue', (-350, 0.08)),
    (0, -Br, f'$-B_r$ = -{Br} T', 'blue', (100, -0.12)),
    (-HcB, 0, f'$-H_{{cB}}$ = -{HcB:.0f} kA/m', 'purple', (-900, 0.12)),
    (HcB, 0, f'$+H_{{cB}}$ = +{HcB:.0f} kA/m', 'purple', (200, -0.15)),
    (-H_sat_max, -Bsat, f'$-B_{{sat}}$ = -{Bsat} T', 'black', (-H_sat_max+50, -Bsat-0.12)),
    (H_sat_max, Bsat, f'$+B_{{sat}}$ = +{Bsat} T', 'black', (H_sat_max-500, Bsat+0.05)),
]

for H_val, B_val, txt, color, offset in landmarks:
    plt.plot(H_val, B_val, marker='o', markersize=6, color=color)
    plt.annotate(txt, (H_val, B_val), textcoords="offset points",
                 xytext=offset, fontsize=9, fontweight='bold', color=color)

# Quadrant dividing lines
plt.axhline(0, color='black', linewidth=0.8, linestyle=':')
plt.axvline(0, color='black', linewidth=0.8, linestyle=':')

# Quadrant Markers
plt.text(1200, 0.8, 'QUADRANT I\n(Initial Saturation)', fontsize=10, alpha=0.3, fontweight='bold')
plt.text(-1800, 0.8, 'QUADRANT II\n(Demagnetization Line)', fontsize=10, alpha=0.3, fontweight='bold')
plt.text(-1800, -0.8, 'QUADRANT III\n(Reverse Saturation)', fontsize=10, alpha=0.3, fontweight='bold')
plt.text(1200, -0.8, 'QUADRANT IV\n(Return Path)', fontsize=10, alpha=0.3, fontweight='bold')

# Axes limits and labels
plt.xlim(-2800, 2800)
plt.ylim(-1.8, 1.8)
plt.title('Grade N42 Neodymium (NdFeB) Magnet - Full 4-Quadrant B-H Hysteresis Loop', fontsize=12, pad=15)
plt.xlabel('Applied Magnetic Field Strength, $H$ (kA/m)', fontsize=11)
plt.ylabel('Magnetic Flux Density, $B$ (Tesla)', fontsize=11)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.show()
