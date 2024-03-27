import gmsh 
import numpy as np 
import sys as sys 
sys.path.append('../')
from meshe.mesh import * 
from fvm.gradient import *
from fvm.diffusion import * 
from fvm.convection import * 

gmsh.initialize()
gmsh.model.add('test_model')

diffusion_coeff = 1.
velocity = 0.1
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
# load mesh 
mesh = TetraMesh()
mesh.gmsh_reader('test.msh')
mesh.set_internal_faces()
mesh.set_elements_intfaces_connectivity()
mesh.set_boundary_faces()
mesh.set_elements_centroids()
print('Number of nodes : ', np.shape(mesh.nodes))
print('Number of elements :',np.shape(mesh.elements))
print('Number of boundary faces :',np.shape(mesh.bndfaces))
print('Number of internal faces :', np.shape(mesh.intfaces))
# set data 
arr_tmp = np.zeros([np.size(mesh.elements,0),3])
arr_tmp[:,0] = velocity
mesh.elements_data['velocity'] = arr_tmp
arr_tmp = np.zeros([np.size(mesh.bndfaces,0),3])
arr_tmp[:,0] = velocity
mesh.bndfaces_data['velocity'] = arr_tmp

boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : 10},
                       'outlet' : {'type' : 'dirichlet',
                                   'value' : 0},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])}}

diffusion_op = OrthogonalDiffusion()
convection_op = CentralDiffConvection(velocity_data = 'velocity')
mat, rhs_vec = diffusion_op(mesh,boundary_conditions,diffusion_coeff=diffusion_coeff)
mat_c, rhs_c = convection_op(mesh,boundary_conditions)
mat += mat_c
rhs_vec += rhs_c 
solution = np.linalg.solve(mat,rhs_vec)
mesh.elements_data['convdiff_solution'] = solution

print(mesh.elements_data['convdiff_solution'] )
mesh.save_vtk(output_file = '3Dcyl_convdiff_dump.vtk')