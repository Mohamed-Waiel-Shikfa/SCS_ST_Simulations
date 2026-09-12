"""
=====================================================================
SIMPLE ELECTROMAGNET SIMULATION
=====================================================================
Models an electromagnet built by winding rectangular copper strip
around a rectangular bore, layer by layer.

Assumed winding pattern
------------------------
- The coil axis is z.
- TURNS_PER_LAYER turns are stacked along z, edge to edge, each
  taking up STRIP_WIDTH_MM of axial length.
- NUM_LAYERS layers are stacked radially outward in x/y, each layer
  adding STRIP_THICKNESS_MM to the coil's half-width.
- Every turn circulates in the same direction, so every turn's field
  adds constructively on-axis - exactly like a real coil.

Each turn is treated as a flat rectangular loop (4 straight
segments), i.e. the helix is approximated as a stack of closed
loops centred at their own z. This is a standard, very accurate
approximation once you have more than a handful of turns, and it
keeps the field calculation exact (no curve-discretisation error).

The magnetic field intensity H (in A/m) is found by direct
numerical evaluation of the Biot-Savart law for a finite straight
wire, summed over every segment of every turn, at every point of a
3D grid filling the bore. No external magnetics/FEM library is used
- just numpy for the maths and matplotlib for the plot.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
# 2. RESISTANCE OF THE COIL
# =====================================================================
# resistance = resistivity * (total wire length) / (cross-section area)

cross_section_area = strip_width * strip_thickness     # m^2

total_length = 0.0
for layer in range(NUM_LAYERS):
    # each layer sits one strip-thickness further out than the last
    width_x = bore_x + (2 * layer + 1) * strip_thickness
    width_y = bore_y + (2 * layer + 1) * strip_thickness
    perimeter = 2 * (width_x + width_y)
    total_length += TURNS_PER_LAYER * perimeter

resistance        = COPPER_RESISTIVITY * total_length / cross_section_area
voltage_needed    = CURRENT_A * resistance
power_dissipated  = CURRENT_A**2 * resistance

print("=" * 60)
print("COIL SUMMARY")
print("=" * 60)
print(f"Total turns          : {total_turns}")
print(f"Coil winding length  : {coil_length_z*1000:.1f} mm  "
      f"(bore length parameter: {BORE_LENGTH_Z_MM:.1f} mm)")
print(f"Total strip length   : {total_length:.2f} m")
print(f"Strip cross-section  : {cross_section_area*1e6:.3f} mm^2")
print(f"Coil resistance      : {resistance:.4f} Ohm  ({resistance*1000:.2f} mOhm)")
print(f"--> at {CURRENT_A:.2f} A this needs {voltage_needed:.3f} V, "
      f"dissipating {power_dissipated:.2f} W")
print("=" * 60)
print()

# =====================================================================
# 3. BUILD THE COIL AS A LIST OF STRAIGHT CURRENT SEGMENTS
# =====================================================================
# Every turn is 4 straight segments forming a rectangle, all turns
# using the same corner order so every turn's field adds up.

def rectangle_corners(half_x, half_y, z):
    return np.array([
        [ half_x, -half_y, z],
        [ half_x,  half_y, z],
        [-half_x,  half_y, z],
        [-half_x, -half_y, z],
    ])

segment_starts = []
segment_ends   = []

for layer in range(NUM_LAYERS):
    width_x = bore_x + (2 * layer + 1) * strip_thickness + (2 * layer) * ins_thickness
    width_y = bore_y + (2 * layer + 1) * strip_thickness + (2 * layer) * ins_thickness
    half_x, half_y = width_x / 2, width_y / 2

    for turn in range(TURNS_PER_LAYER):
        turn_z = -coil_length_z / 2 + strip_width * (turn + 0.5)
        corners = rectangle_corners(half_x, half_y, turn_z)
        for i in range(4):
            segment_starts.append(corners[i])
            segment_ends.append(corners[(i + 1) % 4])

segment_starts = np.array(segment_starts)   # (n_segments, 3)
segment_ends   = np.array(segment_ends)     # (n_segments, 3)

print(f"Coil modeled as {len(segment_starts)} straight current segments "
      f"({total_turns} turns x 4 sides)")
print()

# =====================================================================
# 4. MAGNETIC FIELD INTENSITY - BRUTE FORCE BIOT-SAVART SUMMATION
# =====================================================================
# For a straight segment from p1 to p2 carrying current I, the exact
# field it produces at a point P (the closed-form Biot-Savart
# integral for a finite straight wire) is:
#
#   H(P) = (I / 4*pi) * (Lhat x r1) / d^2 * (cos(a1) - cos(a2))
#
# where Lhat is the unit vector p1->p2, r1 = P - p1, d is the
# perpendicular distance from P to the infinite line through the
# segment, and a1, a2 are the angles between the segment direction
# and the lines from p1 and p2 to P. This already gives the field
# INTENSITY H in A/m directly (H = B/mu0 in air, so mu0 never has to
# appear). No discretisation, no library - this is the exact
# solution for one straight wire, summed here over every segment.

def compute_H_field(points):
    """H field (A/m) at an array of points, shape (...,3) -> (...,3)."""
    H = np.zeros_like(points)
    for p1, p2 in zip(segment_starts, segment_ends):
        L = p2 - p1
        L_hat = L / np.linalg.norm(L)

        r1 = points - p1        # vector from segment start to each point
        r2 = points - p2        # vector from segment end to each point

        cross = np.cross(L_hat, r1)
        d2 = np.maximum(np.sum(cross ** 2, axis=-1), 1e-12)

        r1_len = np.maximum(np.linalg.norm(r1, axis=-1), 1e-12)
        r2_len = np.maximum(np.linalg.norm(r2, axis=-1), 1e-12)
        cos_a1 = np.sum(r1 * L_hat, axis=-1) / r1_len
        cos_a2 = np.sum(r2 * L_hat, axis=-1) / r2_len

        factor = (CURRENT_A / (4 * np.pi)) * (cos_a1 - cos_a2) / d2
        H += factor[..., None] * cross
    return H

x = np.linspace(-bore_x / 2, bore_x / 2, N_GRID_X)
y = np.linspace(-bore_y / 2, bore_y / 2, N_GRID_Y)
z = np.linspace(-bore_z / 2, bore_z / 2, N_GRID_Z)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
grid = np.stack([X, Y, Z], axis=-1)          # (Nx, Ny, Nz, 3)

H_field = compute_H_field(grid)
H_magnitude = np.linalg.norm(H_field, axis=-1)   # (Nx, Ny, Nz), A/m

# =====================================================================
# 5. RESULTS
# =====================================================================

print("=" * 60)
print("MAGNETIC FIELD INTENSITY IN THE BORE VOLUME")
print("=" * 60)
print(f"Average |H|  : {H_magnitude.mean():.3f} A/m")
print(f"Min |H|      : {H_magnitude.min():.3f} A/m")
print(f"Max |H|      : {H_magnitude.max():.3f} A/m")
print("=" * 60)
print("(Note: each turn is modeled as an infinitely thin filament, so")
print(" values right next to the windings can look artificially high -")
print(" this matters less as you move away from the coil surface.)")
print()

# ---- smooth heatmap: xy plane through the bore centre (z = 0) ----
# Its own fine linspace grid, independent of N_GRID_X/Y/Z above, so
# the image isn't limited by the (coarser) grid used for the average.
x_slice = np.linspace(-bore_x / 2, bore_x / 2, N_SLICE)
y_slice = np.linspace(-bore_y / 2, bore_y / 2, N_SLICE)
Xs, Ys = np.meshgrid(x_slice, y_slice, indexing='ij')
slice_points = np.stack([Xs, Ys, np.zeros_like(Xs)], axis=-1)   # z = 0

H_slice_magnitude = np.linalg.norm(compute_H_field(slice_points), axis=-1)  # A/m

plt.figure(figsize=(7, 6))
extent = [x_slice.min() * 1000, x_slice.max() * 1000,
          y_slice.min() * 1000, y_slice.max() * 1000]
im = plt.imshow(H_slice_magnitude.T, extent=extent, origin='lower',
                 aspect='auto', cmap='inferno', norm=LogNorm())
plt.colorbar(im, label='|H| (A/m)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Magnetic field intensity - middle slice (z = 0 plane)')
plt.tight_layout()
plt.savefig('electromagnet_field.png', dpi=150)
plt.show()
