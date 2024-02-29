import gmsh 
import meshio
import numpy as np 
from mesh import * 

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
triangles_labels = []
tetrahedra = [cell.data for cell in mesh.cells if cell.type == "tetra"]

print(len(mesh.cell_data))
#print(mesh.cell_data_dict['gmsh:physical']['triangle'])
print(len(mesh.cell_data_dict['gmsh:physical']['triangle']))
#print(mesh.cell_data_dict['gmsh:physical']['tetra'])
print(len(mesh.cell_data_dict['gmsh:physical']['tetra']))
#print(mesh.cell_data['gmsh:physical'])
print(len(mesh.cell_data['gmsh:physical']))

surf_elements = []
vol_elements = []
surf_tags = []
for i in range(len(mesh.cells)):
    cell = mesh.cells[i]
    elements_data = mesh.cell_data['gmsh:physical'][i]
    if cell.type == 'triangle' : 
        surf_elements.append(cell.data)
        surf_tags.append(elements_data)
    if cell.type == 'tetra':
        vol_elements.append(cell.data)
    elements = cell.data
    print('##########',i,'###########')
    print('element shape :', np.shape(elements))
    print('element data shape : ', np.shape(elements_data))
    print('elements type : ', mesh.cells[i].type)
vol_elements = np.concatenate(vol_elements)
print(np.shape(vol_elements))
surf_elements = np.concatenate(surf_elements)
surf_tags = np.concatenate(surf_tags)
print(np.shape(surf_elements))
print(np.shape(surf_tags))

#print(mesh)
#print(mesh.cell_sets['wall'])
#print(mesh.cell_sets['gmsh:bounding_entities'])

mesh = TetraMesh()
mesh.gmsh_reader('test.msh')
print(np.shape(mesh.nodes))
print(np.shape(mesh.elements))
print(np.shape(mesh.boundary_elements))
print(np.shape(mesh.boundary_tags))





