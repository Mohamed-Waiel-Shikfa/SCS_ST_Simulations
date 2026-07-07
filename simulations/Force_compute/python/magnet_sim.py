import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

# --- Constants ---
mu0 = 4 * np.pi * 1e-7
Hc_alnico = -65000

# Alnico 5-7 Demagnetization Curve
H_data = np.array([-59000, -56000, -52000, -45000, -30000, -15000, 0])
B_data = np.array([0.0,    0.50,   0.90,   1.15,   1.28,   1.33,   1.35])
demag_curve = CubicSpline(H_data, B_data)

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

    return force

# --- Run the Hybrid Solver Example ---
calculate_clamping_force(
    shape='cyl', 
    lm=12.5, 
    g_min=0.8, 
    g_max=0.8, 
    diameter=4.75
)