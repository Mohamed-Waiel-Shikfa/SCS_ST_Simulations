import numpy as np
import math
from netgen.occ import *
from ngsolve import *
import pyvista as pv

# ==========================================
# 1. Setup Parameters
# ==========================================
mu0 = 4 * np.pi * 1e-7
Br = 1.35       # Remanence of Alnico 5-7 (Tesla)
M_mag = Br / mu0 # Magnetization magnitude (A/m)

# Dimensions (in meters)
w, d, h = 0.01, 0.01, 0.002  # 10x10x2 mm
gap = 0.001                  # 1 mm gap at the closest edge
angle_deg = 15               # Tilt angle of the top magnet

# ==========================================
# 2. 3D OpenCascade Geometry definition
# ==========================================
print("Generating 3D Geometry...")

# Magnet 1: Flat on the bottom
mag1 = Box((-w/2, -d/2, -h), (w/2, d/2, 0))
mag1.mat("magnet1")

# Magnet 2: Shifted up by the gap, then rotated around the X-axis
mag2 = Box((-w/2, -d/2, 0), (w/2, d/2, h))
mag2 = mag2.Move((0, 0, gap))
mag2 = mag2.Rotate(Axis((0,0,gap), (1,0,0)), angle_deg)
mag2.mat("magnet2")

# To calculate the force via Maxwell Stress Tensor, we need to tag the boundary faces of Mag 2
for f in mag2.faces:
    f.name = "mag2_bnd"

# Air Domain: A larger box surrounding the magnets
air = Box((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02))
air.mat("air")
# Cut the magnets out of the air box so they fit perfectly together like puzzle pieces
air = air - mag1 - mag2

# FIX: Safely catch unassigned NoneType outer faces
for f in air.faces:
    if f.name is None:
        f.name = "outer_bnd"

# Glue the geometry together to create shared continuous interfaces
geo = Glue([air, mag1, mag2])

# ==========================================
# 3. Meshing
# ==========================================
print("Meshing Domain (This may take a moment)...")
mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.0015))

# ==========================================
# 4. FEM Formulation (Magnetic Scalar Potential)
# ==========================================
print("Setting up Maxwell's Equations...")

fes = H1(mesh, order=2, dirichlet="outer_bnd")
u = fes.TrialFunction()
v = fes.TestFunction()

theta = math.radians(angle_deg)
M2_y = -M_mag * math.sin(theta)
M2_z = -M_mag * math.cos(theta)

M_dict = {
    "magnet1": (0, 0, M_mag),
    "magnet2": (0, M2_y, M2_z),
    "air":     (0, 0, 0)
}
M = CoefficientFunction([M_dict[mat] for mat in mesh.GetMaterials()])

a = BilinearForm(fes, symmetric=True)
a += mu0 * grad(u) * grad(v) * dx

f = LinearForm(fes)
f += mu0 * InnerProduct(M, grad(v)) * dx

# ==========================================
# 5. Solve the Linear System
# ==========================================
print("Solving Matrix System...")
a.Assemble()
f.Assemble()

u_sol = GridFunction(fes)
u_sol.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# ==========================================
# 6. Post-Processing & Force Calculation
# ==========================================
print("Calculating Maxwell Stress Tensor...")

H = -grad(u_sol)
B = mu0 * (H + M)

n = specialcf.normal(3)
T_n = 1/mu0 * ( InnerProduct(B, n)*B - 0.5*InnerProduct(B, B)*n )

force_x = Integrate(T_n[0], mesh, BND, definedon=mesh.Boundaries("mag2_bnd"))
force_y = Integrate(T_n[1], mesh, BND, definedon=mesh.Boundaries("mag2_bnd"))
force_z = Integrate(T_n[2], mesh, BND, definedon=mesh.Boundaries("mag2_bnd"))

total_force_mag = math.sqrt(force_x**2 + force_y**2 + force_z**2)

print("\n==================================")
print("     3D FEM RESULTS (NGSolve)     ")
print("==================================")
print(f"Angle     : {angle_deg} degrees")
print(f"Gap       : {gap*1000} mm")
print("----------------------------------")
print(f"Force (X) : {force_x:.4f} N (Lateral Shear)")
print(f"Force (Y) : {force_y:.4f} N (Angular Slip)")
print(f"Force (Z) : {force_z:.4f} N (Normal Clamp)")
print("----------------------------------")
print(f"TOTAL MAGNITUDE: {total_force_mag:.4f} N")
print("==================================\n")

# ==========================================
# 7. Export to VTK & 3D PyVista Visualization
# ==========================================
print("Exporting data to VTK for 3D visualization...")
# Exporting with legacy=True avoids formatting conflicts with specific python environments
vtk = VTKOutput(mesh, coefs=[u_sol, B], names=["potential", "Bfield"], filename="magnet_sim", legacy=True, subdivision=1)
vtk.Do()

print("Launching 3D PyVista Visualizer...")
mesh_pv = pv.read("magnet_sim.vtk")

# Compute the absolute 3D vector length for flux density visualization
b_vectors = mesh_pv.point_data["Bfield"]
b_magnitude = np.linalg.norm(b_vectors, axis=1)
mesh_pv.point_data["B_mag"] = b_magnitude

# Visualizer Setup
plotter = pv.Plotter()
plotter.set_background("#141419")

# Add the air/space matrix with high transparency to show field contours
plotter.add_mesh(
    mesh_pv,
    scalars="potential",
    cmap="coolwarm",
    opacity=0.12,
    show_edges=False,
    scalar_bar_args={"title": "Scalar Potential (V)", "color": "white", "position_x": 0.05}
)

# Compute beautiful 3D streamlines tracing directly through the gap center
streamlines = mesh_pv.streamlines(
    vectors="Bfield",
    source_center=(0, 0, gap/2),
    source_radius=0.012,
    n_points=150
)

# Render the flux lines as solid physical tubes
plotter.add_mesh(
    streamlines.tube(radius=0.00025),
    scalars="B_mag",
    cmap="turbo",
    scalar_bar_args={"title": "Flux Density B (Tesla)", "color": "white", "position_x": 0.85}
)

# Draw precise bounding box outlines to clearly show where the physical 10x10x2 mm blocks sit
# Create the box meshes first, then add them to the plotter
box1 = pv.Box(bounds=[-w/2, w/2, -d/2, d/2, -h, 0])
plotter.add_mesh(box1, color="#00f5d4", style="wireframe", line_width=2.5)

# Note: Because the top magnet is angled, this axis-aligned box just shows
# the un-rotated bounding area to help you visualize the initial gap separation.
box2 = pv.Box(bounds=[-w/2, w/2, -d/2, d/2, gap, gap+h])
plotter.add_mesh(box2, color="#ff0055", style="wireframe", line_width=2.5)

print("\nControl layout: Left-click to rotate, Scroll to zoom, Right-click to pan.")
plotter.show()
