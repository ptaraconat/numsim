import gmsh 
import meshio

# Parameters 
extrude_distance = 2
edge_len = 1 
mesh_size = 0.1

gmsh.initialize()
# Define vertices
v1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
v2 = gmsh.model.geo.addPoint(edge_len, 0, 0, mesh_size)
v3 = gmsh.model.geo.addPoint(edge_len, edge_len, 0, mesh_size)
v4 = gmsh.model.geo.addPoint(0, edge_len, 0, mesh_size)
# Define lines
l1 = gmsh.model.geo.addLine(v1, v2)
l2 = gmsh.model.geo.addLine(v2, v3)
l3 = gmsh.model.geo.addLine(v3, v4)
l4 = gmsh.model.geo.addLine(v4, v1)
# Define curves
c1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
# Define surfaces
s1 = gmsh.model.geo.addPlaneSurface([c1])
# Syncrhonize 
gmsh.model.geo.synchronize()
## Set mesh algorithm to transfinite for quadrilateral elements
gmsh.option.setNumber('Mesh.Algorithm', 8)
# Set recombination algo for geting quad (recombined from tri element)
gmsh.model.mesh.setRecombine(2, s1)
# Generate the recombined mesh
gmsh.model.mesh.generate(2)


# Write the mesh to a file
gmsh.write("mesh.vtk")
gmsh.finalize()


##############
mesh = meshio.read("mesh.vtk")
for i in range(len(mesh.cells)):
    cell = mesh.cells[i]
    print(cell.type)
    if cell.type == 'quad':
        print(cell.data.shape)
    #print(cell.data)