"""
=====================================================================
VOLUMETRIC ELECTROMAGNET SIMULATION (BIOT-SAVART TRIPLE INTEGRAL)
=====================================================================
Models an electromagnet built by winding rectangular copper strip
around a rectangular bore, layer by layer.

Conductor Cross-Section Consideration (Method 1)
-----------------------------------------------
- Instead of thin filaments, each rectangular current turn is treated
  as a 3D conductor with finite width (along z) and thickness (along x/y).
- The H-field is calculated via a volumetric Biot-Savart integral over
  the 3D rectangular cross-section and turn perimeter:
      H(r) = (1 / 4*pi) * ///_V [ J(r') x (r - r') / |r - r'|^3 ] dV'
- Replaces unphysical infinity singularities near conductor edges with
  physically accurate finite magnetic field distributions.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider

# =====================================================================
# PARAMETERS - edit these to design your electromagnet
# =====================================================================

# ---- Copper strip cross-section ----
STRIP_WIDTH_MM     = 5.0    # strip size along the coil axis (z)     [mm]
STRIP_THICKNESS_MM = 0.2    # strip size in the radial direction     [mm]
INS_THICKNESS_MM   = 0.05   # insulation between layers               [mm]

# ---- Winding ----
TURNS_PER_LAYER = 1        # turns stacked along the axis, per layer
NUM_LAYERS      = 22       # layers stacked radially outward
CURRENT_A       = 4500.0   # target current through the strip       [A]

# ---- Volume enclosed by the electromagnet (the bore) ----
BORE_WIDTH_X_MM  = 20.0     # inner free space, x direction          [mm]
BORE_WIDTH_Y_MM  = 20.0     # inner free space, y direction          [mm]
BORE_LENGTH_Z_MM = 5.0      # inner free space, along the coil axis  [mm]

# ---- Material ----
COPPER_RESISTIVITY = 1.68e-8   # copper resistivity at ~20 C        [Ohm*m]

# ---- Simulation grid resolution (points spanning the bore) ----
N_GRID_X = 20
N_GRID_Y = 20
N_GRID_Z = 5
N_SLICE  = 100    # resolution of the smooth xy heatmap

# ---- Volumetric Integration Discretization (per segment) ----
# Higher values increase spatial precision inside and near the conductor
N_DISC_LENGTH    = 20    # integration steps along segment length
N_DISC_WIDTH     = 3     # integration steps across strip width (axial, z)
N_DISC_THICKNESS = 3     # integration steps across strip thickness (radial)

# ---- Specific User Target Points [mm] for field evaluation ----
USER_EVALUATION_POINTS_MM = [
    (0.0, 0.0, 0.0),       # Bore Center
    (5.0, 5.0, 0.0),       # Off-axis mid-plane point
    (0.0, 0.0, 2.5),       # Axis endpoint
    (8.0, 8.0, 1.0)        # Near-wall internal point
]

# ---- Plot appearance ----
COLOR_CLIP_PERCENTILE = 0.01

# =====================================================================
# 1. GEOMETRY IN SI UNITS
# =====================================================================

strip_width     = STRIP_WIDTH_MM * 1e-3
strip_thickness = STRIP_THICKNESS_MM * 1e-3
ins_thickness   = INS_THICKNESS_MM * 1e-3
bore_x          = BORE_WIDTH_X_MM * 1e-3
bore_y          = BORE_WIDTH_Y_MM * 1e-3
bore_z          = BORE_LENGTH_Z_MM * 1e-3

coil_length_z   = TURNS_PER_LAYER * strip_width
total_turns     = TURNS_PER_LAYER * NUM_LAYERS

# =====================================================================
# 2. RESISTANCE AND POWER COMPUTATION
# =====================================================================

cross_section_area = strip_width * strip_thickness     # m^2
total_length = 0.0

for layer in range(NUM_LAYERS):
    width_x = bore_x + (2 * layer + 1) * strip_thickness + (2 * layer) * ins_thickness
    width_y = bore_y + (2 * layer + 1) * strip_thickness + (2 * layer) * ins_thickness
    perimeter = 2 * (width_x + width_y)
    total_length += TURNS_PER_LAYER * perimeter

resistance       = COPPER_RESISTIVITY * total_length / cross_section_area
voltage_needed   = CURRENT_A * resistance
power_dissipated = CURRENT_A**2 * resistance

# print("=" * 60)
# print("COIL SUMMARY")
# print("=" * 60)
# print(f"Total turns          : {total_turns}")
# print(f"Coil winding length  : {coil_length_z*1000:.1f} mm")
# print(f"Total strip length   : {total_length:.2f} m")
# print(f"Strip cross-section  : {cross_section_area*1e6:.3f} mm^2")
# print(f"Coil resistance      : {resistance:.4f} Ohm ({resistance*1000:.2f} mOhm)")
# print(f"--> at {CURRENT_A:.2f} A: {voltage_needed:.3f} V, {power_dissipated:.2f} W")
# print("=" * 60)
# print()

# =====================================================================
# 3. BUILD 3D VOLUMETRIC CURRENT ELEMENTS (METHOD 1)
# =====================================================================
# Instead of 1D line segments, each straight section of a rectangular turn
# is decomposed into volumetric differential elements with current density J.

volume_elements_r = []   # Position vectors of volume elements (N, 3)
volume_elements_J = []   # Current density vectors J * dV for each element (N, 3)

J_magnitude = CURRENT_A / cross_section_area  # Current density A/m^2

for layer in range(NUM_LAYERS):
    # Radial dimensions for this layer
    layer_inner_x = bore_x / 2 + layer * (strip_thickness + ins_thickness)
    layer_inner_y = bore_y / 2 + layer * (strip_thickness + ins_thickness)

    for turn in range(TURNS_PER_LAYER):
        turn_z_center = -coil_length_z / 2 + strip_width * (turn + 0.5)

        # Grid inside the conductor cross-section (Thickness x Width)
        t_offsets = np.linspace(0, strip_thickness, N_DISC_THICKNESS)
        w_offsets = np.linspace(-strip_width/2, strip_width/2, N_DISC_WIDTH)

        dv = (strip_thickness / N_DISC_THICKNESS) * (strip_width / N_DISC_WIDTH)

        # Build 4 straight volumetric bar segments for the rectangular loop
        for t_val in t_offsets:
            hx = layer_inner_x + t_val
            hy = layer_inner_y + t_val

            for w_val in w_offsets:
                zc = turn_z_center + w_val

                # Segment 1: Bottom (+x direction)
                s1_x = np.linspace(-hx, hx, N_DISC_LENGTH)
                dl1 = 2 * hx / N_DISC_LENGTH
                for x_p in s1_x:
                    volume_elements_r.append([x_p, -hy, zc])
                    volume_elements_J.append([J_magnitude * dv * dl1, 0, 0])

                # Segment 2: Right (+y direction)
                s2_y = np.linspace(-hy, hy, N_DISC_LENGTH)
                dl2 = 2 * hy / N_DISC_LENGTH
                for y_p in s2_y:
                    volume_elements_r.append([hx, y_p, zc])
                    volume_elements_J.append([0, J_magnitude * dv * dl2, 0])

                # Segment 3: Top (-x direction)
                s3_x = np.linspace(hx, -hx, N_DISC_LENGTH)
                dl3 = 2 * hx / N_DISC_LENGTH
                for x_p in s3_x:
                    volume_elements_r.append([x_p, hy, zc])
                    volume_elements_J.append([-J_magnitude * dv * dl3, 0, 0])

                # Segment 4: Left (-y direction)
                s4_y = np.linspace(hy, -hy, N_DISC_LENGTH)
                dl4 = 2 * hy / N_DISC_LENGTH
                for y_p in s4_y:
                    volume_elements_r.append([-hx, y_p, zc])
                    volume_elements_J.append([0, -J_magnitude * dv * dl4, 0])

vol_r = np.array(volume_elements_r)  # (N_elements, 3)
vol_J = np.array(volume_elements_J)  # (N_elements, 3) -> represents J * dV

print(f"Volumetric Biot-Savart discretization generated {len(vol_r)} source elements.")
print()

# =====================================================================
# 4. VOLUMETRIC BIOT-SAVART CALCULATOR
# =====================================================================

def compute_H_field_volumetric(points):
    """
    Computes generalized 3D H-field (A/m) via Method 1 (Volumetric Biot-Savart).
    Accepts arbitrary array of target points with shape (..., 3).
    """
    orig_shape = points.shape
    pts_flat = points.reshape(-1, 3)  # Flatten spatial dimensions
    H_flat = np.zeros_like(pts_flat)

    # Chunking to manage memory efficiently during vectorized evaluation
    chunk_size = 500
    for i in range(0, len(pts_flat), chunk_size):
        pts_chunk = pts_flat[i:i+chunk_size]  # (M, 3)

        # Displacement vectors: r - r'
        disp = pts_chunk[:, None, :] - vol_r[None, :, :]  # (M, N_elem, 3)
        dist_sq = np.sum(disp**2, axis=-1)
        dist_sq = np.maximum(dist_sq, 1e-12)               # Regularize singularity
        dist_cube = dist_sq * np.sqrt(dist_sq)             # |r - r'|^3

        # Vector cross product: (J * dV) x (r - r')
        cross = np.cross(vol_J[None, :, :], disp)          # (M, N_elem, 3)

        # Integrate over all volume elements
        dH = cross / (4 * np.pi * dist_cube[..., None])
        H_flat[i:i+chunk_size] = np.sum(dH, axis=1)

    return H_flat.reshape(orig_shape)

# Compute 3D Bore Volume Grid
x = np.linspace(-bore_x / 2, bore_x / 2, N_GRID_X)
y = np.linspace(-bore_y / 2, bore_y / 2, N_GRID_Y)
z = np.linspace(-bore_z / 2, bore_z / 2, N_GRID_Z)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
grid = np.stack([X, Y, Z], axis=-1)

print("Calculating magnetic field over 3D bore volume...")
H_field = compute_H_field_volumetric(grid)
H_mag = np.linalg.norm(H_field, axis=-1)

# =====================================================================
# 5. FIELD EVALUATION AT CUSTOM USER INPUT POINTS
# =====================================================================

user_pts = np.array(USER_EVALUATION_POINTS_MM) * 1e-3
user_H_field = compute_H_field_volumetric(user_pts)
user_H_mag = np.linalg.norm(user_H_field, axis=-1)

# =====================================================================
# 6. RESULTS & STATISTICAL ANALYSIS
# =====================================================================

print("=" * 60)
print("MAGNETIC FIELD INTENSITY (|H|) STATISTICAL ANALYSIS")
print("=" * 60)
print(f"BORE VOLUME GRID STATISTICS ({N_GRID_X}x{N_GRID_Y}x{N_GRID_Z} points):")
print(f"  - Average |H|   : {np.mean(H_mag):.3e} A/m")
print(f"  - Median |H|    : {np.median(H_mag):.3e} A/m")
print(f"  - Min |H|       : {np.min(H_mag):.3e} A/m")
print(f"  - Max |H|       : {np.max(H_mag):.3e} A/m")
print(f"  - Std Dev |H|   : {np.std(H_mag):.3e} A/m")
print("-" * 60)
print("SPECIFIC INPUT TARGET POINTS EVALUATION:")
for pt_mm, h_vec, h_m in zip(USER_EVALUATION_POINTS_MM, user_H_field, user_H_mag):
    print(f"  - Point (x={pt_mm[0]:5.1f}, y={pt_mm[1]:5.1f}, z={pt_mm[2]:5.1f}) mm:")
    print(f"      |H| = {h_m:.4e} A/m  | H_vec = [{h_vec[0]:.2e}, {h_vec[1]:.2e}, {h_vec[2]:.2e}] A/m")

print("-" * 60)
print(f"USER POINTS SUMMARY STATISTICS ({len(user_pts)} points):")
print(f"  - Average |H|   : {np.mean(user_H_mag):.3e} A/m")
print(f"  - Median |H|    : {np.median(user_H_mag):.3e} A/m")
print(f"  - Min |H|       : {np.min(user_H_mag):.3e} A/m")
print(f"  - Max |H|       : {np.max(user_H_mag):.3e} A/m")
print(f"  - Std Dev |H|   : {np.std(user_H_mag):.3e} A/m")
print("=" * 60)
print()
