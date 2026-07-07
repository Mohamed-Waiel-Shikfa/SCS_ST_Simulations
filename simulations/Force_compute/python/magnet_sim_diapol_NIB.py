import numpy as np

# --- Constants ---
mu0 = 4 * np.pi * 1e-7

# --- N40 NdFeB Material Properties ---
# NdFeB has a linear demagnetization curve in the 2nd quadrant.
Br_n40 = 1.28           # Remanence in Tesla
Hc_n40 = -995000        # Coercivity in A/m
mu_rec = 1.05 * mu0     # Recoil permeability (approx 1.05 for NdFeB)

def calculate_n40_block_force(w_mm, h_mm, l_mm, gap_mm):
    """
    Calculates the clamping force between two identical rectangular N40 magnets 
    in open air using the Equivalent-Diameter Dipole Model.
    """
    print(f"--- N40 Block Magnet ({w_mm}x{h_mm}x{l_mm} mm) ---")
    
    # Unit conversions to meters
    w = w_mm / 1000
    h = h_mm / 1000
    L = l_mm / 1000
    g = gap_mm / 1000
    
    # --- 1. Geometry & Effective Diameter ---
    area = w * h
    volume = area * L
    
    # Convert rectangular area to an equivalent circular diameter for the dipole equations
    D_eff = 2 * np.sqrt(area / np.pi)
    
    # --- 2. Open-Circuit Demagnetization Factor (Nd) ---
    # Using the effective diameter in the cylinder approximation formula
    Nd = D_eff / (2 * L + D_eff)
    
    # Calculate Open-Circuit Permeance Coefficient (Pc)
    Pc = (1 - Nd) / Nd
    
    # Loadline slope for the isolated magnet in open air
    m_load = mu0 * Pc

    # --- 3. Solve for Operating Point (B0, H0) ---
    # Because NdFeB is a straight line (B = Br + mu_rec * H), 
    # and the loadline is a straight line (B = -m_load * H),
    # we can solve exactly with pure algebra instead of root_scalar.
    
    H_intersect = -Br_n40 / (m_load + mu_rec)
    B0 = m_load * abs(H_intersect)
    
    # --- 4. Magnetic Moment (m) ---
    # m = (B / mu0) * Volume
    moment = (B0 / mu0) * volume
    
    # --- 5. Dipole-Dipole Force Calculation ---
    # Distance between the exact geometric centers of the two magnets
    # Center of M1 is L/2 away from the gap, center of M2 is L/2 away from the gap.
    z = L + g 
    
    # Force formula for two coaxial dipoles
    force = (3 * mu0 * (moment ** 2)) / (2 * np.pi * (z ** 4))
    
    print(f"Effective Dia (Deff): {D_eff * 1000:.2f} mm")
    print(f"Demag Factor (Nd): {Nd:.3f}")
    print(f"Permeance Coef (Pc): {Pc:.3f}")
    print(f"Operating Flux (B0): {B0:.4f} T")
    print(f"Magnetic Moment (m): {moment:.4f} A·m²")
    print(f"Center-to-Center (z): {z * 1000:.1f} mm")
    print(f"Predicted Force: {force:.2f} N  (~{force/9.81:.2f} kg)")
    print("-" * 40)
    
    return force

# Run the simulation for your 10x20x5 mm N40 magnets at a 2 mm gap
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=0.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=1.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=2.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=3.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=4.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=5.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=6.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=7.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=8.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=9.0)
force = calculate_n40_block_force(w_mm=10, h_mm=20, l_mm=5, gap_mm=10.0)
