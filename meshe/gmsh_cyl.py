import gmsh 
import meshio
import numpy as np 

gmsh.initialize()

gmsh.model.add('test_model')

radius = 0.5
height = 1 
mesh_size = 0.25

factory = gmsh.model.geo
factory.addPoint(0, 0, 0,  tag = 1)
factory.addPoint(0, radius, 0,  tag = 2)
factory.addPoint(-radius, 0, 0, tag = 3)
factory.addPoint(0, -radius, 0, tag = 4)
factory.addPoint(radius, 0, 0, tag = 5)
circle_curve1 = factory.addCircleArc(2,1,3, tag = 1)
circle_curve2 = factory.addCircleArc(3,1,4, tag = 2)
circle_curve2 = factory.addCircleArc(4,1,5, tag = 3)
circle_curve2 = factory.addCircleArc(5,1,2, tag = 4)
circle = factory.addCurveLoop([1,2,3,4], tag = 3)
print(circle)
# Create a surface from the circle
surface = factory.addPlaneSurface([circle],tag = 1)
print(surface)
# Extrude the surface
extruded_volume = factory.extrude([(2,surface)],0,0,height)
print(extruded_volume)
factory.synchronize()
volume = gmsh.model.addPhysicalGroup(3, [extruded_volume[1][1]], name="fluid")
inlet = gmsh.model.addPhysicalGroup(2, [1], tag = 101, name="inlet")
outlet = gmsh.model.addPhysicalGroup(2, [26], tag = 102, name="outlet")
surface3 = gmsh.model.addPhysicalGroup(2, [13,17,21,25], tag = 103, name="wall")

# Generate the mesh
factory.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write('test.msh')
gmsh.finalize()

# Load mesh from file
mesh = meshio.read("test.msh")
points = mesh.points
cells = mesh.cells
print('Number of nodes :', np.shape(points))
print(cells)
print('Number of cells block :', len(cells))

# Separate triangles and tetrahedra
triangles = [cell.data for cell in mesh.cells if cell.type == "triangle"]
tetrahedra = [cell.data for cell in mesh.cells if cell.type == "tetra"]

print(len(mesh.cell_data))
#print(mesh.cell_data_dict['gmsh:physical']['triangle'])
print(len(mesh.cell_data_dict['gmsh:physical']['triangle']))
#print(mesh.cell_data_dict['gmsh:physical']['tetra'])
print(len(mesh.cell_data_dict['gmsh:physical']['tetra']))
#print(mesh.cell_data['gmsh:physical'])
print(len(mesh.cell_data['gmsh:physical']))

for i in range(len(mesh.cells)):
    elements = mesh.cells[i].data
    print('element shape :', np.shape(elements))
    print('elements type : ', mesh.cells[i].type)
    elements_data = mesh.cell_data['gmsh:physical'][i]
    print('element data shape : ', np.shape(elements_data))



