from math import pi, sqrt

# =====================================================================
# PARAMETERS - edit these to design your electromagnet
# =====================================================================

# ---- Copper strip cross-section ----
STRIP_WIDTH_MM     = 5.0    # strip size along the coil axis (z)     [mm]
STRIP_THICKNESS_MM = 0.1    # strip size in the radial direction     [mm]
INS_THICKNESS_MM = 0.05    # strip size in the radial direction     [mm]

# ---- Winding ----
TURNS_PER_LAYER = 1        # turns stacked along the axis, per layer
NUM_LAYERS      = 22         # layers stacked radially outward
CURRENT_A       = 4500.0       # target current through the strip       [A]

# ---- Volume enclosed by the electromagnet (the bore) ----
BORE_WIDTH_X_MM  = 20.0     # inner free space, x direction          [mm]
BORE_WIDTH_Y_MM  = 20.0     # inner free space, y direction          [mm]
BORE_LENGTH_Z_MM = 5.0    # inner free space, along the coil axis  [mm]

# ---- Material ----
COPPER_RESISTIVITY = 1.68e-8   # copper resistivity at ~20 C        [Ohm*m]

# ---- Simulation grid resolution (points spanning the bore) ----
N_GRID_X = 40
N_GRID_Y = 40
N_GRID_Z = 10
N_SLICE  = 200    # resolution of the smooth xy heatmap (independent of the grid above)

# =====================================================================
# 1. GEOMETRY IN SI UNITS
# =====================================================================

strip_width     = STRIP_WIDTH_MM * 1e-3
strip_thickness = STRIP_THICKNESS_MM * 1e-3
ins_thickness = INS_THICKNESS_MM * 1e-3
bore_x = BORE_WIDTH_X_MM * 1e-3
bore_y = BORE_WIDTH_Y_MM * 1e-3
bore_z = BORE_LENGTH_Z_MM * 1e-3

coil_length_z = TURNS_PER_LAYER * strip_width     # actual axial length of the winding
total_turns   = TURNS_PER_LAYER * NUM_LAYERS

# =====================================================================
# 2. CLOSED FORM SOLVER
# =====================================================================

h=0
for layer in range(NUM_LAYERS):
    a=bore_x + (2 * layer + 1) * strip_thickness + (2 * layer) * ins_thickness
    print(f"{a*1000:.1f} mm")
    h+=sqrt(2)*CURRENT_A/(pi*(a/2))


# =====================================================================
# 3. RESULTS
# =====================================================================

print("=" * 60)
print("MAGNETIC FIELD INTENSITY AT THE CENTER")
print(f"|H| : {h/1_000_000:.3f} MA/m")
print("=" * 60)
print()
