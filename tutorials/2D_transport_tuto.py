import gmsh 
import meshio
import numpy as np 
import sys as sys 
sys.path.append('../')
from meshe.mesh import Mesh2D
from solver.transport import TransportSolver
import os as os 

# Parameters
savedir = '2D_transport_tuto/'
 
fourier = 0.5
cfl = 0.6

n_ite = 100
dump_ite = 20

velocity = 0.6
diffusion_coeff = 1.

edge_len = 1 
mesh_size = 0.1

boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : 10},
                       'outlet' : {'type' : 'dirichlet',
                                   'value' : 0},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])},
                       'FrontBack' : {'type' : 'neumann',
                                      'value' : np.array([0,0,0])}}

# Create savedir 
if os.path.exists(savedir):
    print(savedir, ' already exists')
else : 
    os.mkdir(savedir)

################
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
#
volume = gmsh.model.addPhysicalGroup(2, [s1], name="fluid")
inlet = gmsh.model.addPhysicalGroup(1, [l1], tag = 101, name="inlet")
outlet = gmsh.model.addPhysicalGroup(1, [l3], tag = 102, name="outlet")
surface3 = gmsh.model.addPhysicalGroup(1, [l2,l4], tag = 103, name="wall")
## Set mesh algorithm to transfinite for quadrilateral elements
gmsh.option.setNumber('Mesh.Algorithm', 8)
# Set recombination algo for geting quad (recombined from tri element)
gmsh.model.mesh.setRecombine(2, s1)
# Generate the recombined mesh
gmsh.model.mesh.generate(2)
# Write the mesh to a file
gmsh.write("mesh.msh")
gmsh.finalize()

##############
mesh = Mesh2D()
mesh.set_mesh(meshio.read("mesh.msh"))
indice = 1
mesh.set_internal_faces()
mesh.set_elements_intfaces_connectivity()
mesh.set_boundary_faces()
mesh.set_elements_centroids()
mesh.set_elements_volumes()
print('Number of nodes : ', np.shape(mesh.nodes))
print('Number of elements :',np.shape(mesh.elements))
print('Number of boundary faces :',np.shape(mesh.bndfaces))
print('Number of internal faces :', np.shape(mesh.intfaces))
print(mesh.physical_entities)
###############
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
###############
mesh.save_vtk(output_file = '2D_dump.vtk')