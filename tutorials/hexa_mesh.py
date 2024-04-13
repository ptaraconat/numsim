import gmsh 
import meshio

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

gmsh.model.geo.synchronize()

# Specify mesh options for the square surface
gmsh.model.geo.mesh.setTransfiniteSurface(s1)

# Set mesh algorithm to transfinite for quadrilateral elements
gmsh.option.setNumber('Mesh.Algorithm', 8)

# Generate mesh
gmsh.model.mesh.generate(2)

gmsh.write('test.msh', 'w', format='gmsh22')

gmsh.finalize()

##############
mesh = meshio.read("test.msh")
for i in range(len(mesh.cells)):
    cell = mesh.cells[i]
    print(cell.type)
    #print(cell.data)