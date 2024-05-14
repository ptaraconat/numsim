import gmsh 
import meshio
import numpy as np 
import sys as sys 
sys.path.append('../')
from meshe.mesh import Mesh2D
from solver.incompressible import IncompressibleSolver
import os as os 

# Parameters
savedir = '2D_ics_tuto/'
 
fourier = 0.5
cfl = 0.6

n_ite = 300
dump_ite = 20

velocity = 0.0001
dynamic_viscosity = 1e-3
rho = 1000 

edge_len = 10
mesh_size = 1


boundary_conditions = {'inlet' : {'type' : 'inlet',
                                  'value' : np.array([0,velocity,0])},
                        'outlet' : {'type' : 'outlet',
                                    'value' : 0},
                        'wall' : {'type' : 'wall',
                                  'value' : None},
                        'FrontBack' : {'type' : 'FrontBack',
                                      'value' : None}}

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

solver = IncompressibleSolver()
#
solver.set_boundary_conditions(boundary_conditions)
solver.init_data(mesh)
solver.set_constant_density(mesh, rho)
solver.set_constant_kinematic_viscosity(mesh, rho, dynamic_viscosity)
# Solve Pressure 
solver.update_boundary_velocity(mesh, boundary_conditions)
deltat = 1.
solver._update_velocity_div(mesh, deltat = deltat )
solver._set_pl_operators(mesh)
rhs = mesh.elements_data['div_velocity']
mat = solver.mat_pressure
expl = solver.rhs_pressure
mesh.elements_data['pressure'] = np.linalg.solve(mat,rhs+expl)
print(mat)
print(expl)
print(rhs)
print(mesh.elements_data['pressure'])
# Velocity correction 
solver.update_boundary_pressure(mesh, boundary_conditions)
solver.gradop(mesh)
oo_rho = 1./mesh.elements_data['rho']
corrector = -deltat*np.multiply(oo_rho,mesh.elements_data['grad_pressure'])
mesh.elements_data['velocity'] = mesh.elements_data['velocity'] + corrector
#issue_arg = np.where(mesh.elements_data['grad_pressure'][:,1]>0)[0]
#print(issue_arg)
#print(mesh.elements_data['grad_pressure'][issue_arg])
print(mesh.elements_data['grad_pressure'])
print(mesh.elements_data['velocity'])
   
i = 0
save_path = savedir + f"output_{i:04d}.vtk"
print('dump solution : ', save_path)
mesh.save_vtk(output_file = save_path)