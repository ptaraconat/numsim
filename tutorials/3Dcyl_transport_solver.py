import sys as sys 
sys.path.append('../')
from solver.transport import TransportSolver
from meshe.mesh import TetraMesh
import gmsh
import numpy as np
import os as os  

# Parameters 
savedir = '3D_cyl_transport_tuto/'

n_ite = 300
dump_ite = 20
fourier = 0.49
cfl = 0.6

diffusion_coeff = 1.
velocity = 5
radius = 0.5
height = 1 
mesh_size = 0.25

boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : 10},
                       'outlet' : {'type' : 'dirichlet',
                                   'value' : 0},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])}}
# Create savedir 
if os.path.exists(savedir):
    print(savedir, ' already exists')
else : 
    os.mkdir(savedir)
# Create mesh 
gmsh.initialize()
gmsh.model.add('test_model')
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
mesh.set_elements_volumes()
print('Number of nodes : ', np.shape(mesh.nodes))
print('Number of elements :',np.shape(mesh.elements))
print('Number of boundary faces :',np.shape(mesh.bndfaces))
print('Number of internal faces :', np.shape(mesh.intfaces))
# set data 
mesh.elements_data['temp'] = np.zeros((np.size(mesh.elements,0),1))
arr_tmp = np.zeros([np.size(mesh.elements,0),3])
arr_tmp[:,2] = velocity
mesh.elements_data['velocity'] = arr_tmp
arr_tmp = np.zeros([np.size(mesh.bndfaces,0),3])
arr_tmp[:,2] = velocity
mesh.bndfaces_data['velocity'] = arr_tmp
#
arr_tmp = np.ones((np.size(mesh.elements,0),1))
mesh.elements_data['diffusion'] = diffusion_coeff * arr_tmp
arr_tmp = np.ones((np.size(mesh.bndfaces,0),1))
mesh.bndfaces_data['diffusion'] = diffusion_coeff * arr_tmp
#
solver = TransportSolver('temp',
                         velocity = 'velocity',
                         diffusivity ='diffusion',
                         fourier = fourier,
                         cfl = cfl)
# not required atm 
solver._set_operators(mesh, boundary_conditions)
# steady solution 
mat = solver.mat_diff + solver.mat_conv
rhs = solver.rhs_diff + solver.rhs_conv
static_sol = np.linalg.solve(mat,rhs)
# Temporal Loop 
for i in range(n_ite):
    if i % dump_ite == 0 :
        save_path = savedir + f"output_{i:04d}.vtk"
        print('dump solution : ', save_path)
        mesh.save_vtk(output_file = save_path)
    solver.step(mesh, boundary_conditions)

print(np.mean(static_sol ))
print(np.mean(mesh.elements_data['temp'] ))