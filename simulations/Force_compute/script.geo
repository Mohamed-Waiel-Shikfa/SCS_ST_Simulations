SetFactory("OpenCASCADE");

// ==========================================
// MESH FINENESS CONTROL
// ==========================================
// Lower values = finer mesh (more elements). Higher values = coarser mesh.
mesh_density = 0.5; 
//mesh_density = 1; 
Mesh.MeshSizeFromCurvature = 32; 

Mesh.MeshSizeMin = mesh_density;
Mesh.MeshSizeMax = mesh_density * 1000;

// ==========================================

// 1. Define your real-world dimensions (Adjust these as needed)
R_outer = 10;      // Radius of outer boundary
H_outer = 40;      // Height of outer boundary

R_mag = 2.375;        // Radius of your magnets
H_mag = 12.5;        // Height of each magnet
gap = 0.1;         // Half-distance between the two magnets     

// 2. Build the solid primitives
Cylinder(1) = {0, 0, -H_outer/2, 0, 0, H_outer, R_outer}; 
Cylinder(2) = {0, 0, gap,        0, 0, H_mag,   R_mag};   
Cylinder(3) = {0, 0, -gap-H_mag, 0, 0, H_mag,   R_mag};   

// 3. Carve out the cavities perfectly
BooleanDifference(4) = { Volume{1}; Delete; }{ Volume{2, 3}; };

// 4. Define Volumes for FreeFEM Physics
Physical Volume("magtop", 102) = {2};
Physical Volume("magbtm", 101) = {3};
Physical Volume("outerboundary", 103) = {4};

// 5. Select ONLY the true exterior skin of the entire domain!
// This prevents short-circuiting the internal magnet-air interfaces.
Physical Surface("mesh_skin", 201) = Boundary{ Volume{2, 3, 4}; };