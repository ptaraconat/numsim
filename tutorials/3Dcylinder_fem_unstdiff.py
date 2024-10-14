import gmsh 
import numpy as np 
import sys as sys 
sys.path.append('../')
from meshe.mesh import * 
from fem.elements import *
from solver.fem_scalar_diffusion import * 
import os as os 

# Parameters 
savedir = '3Dcyl_fem_unsteadydiff_tuto/'

boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : 10},
                       'outlet' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])}}

boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : 10},
                       'outlet' : {'type' : 'dirichlet',
                                   'value' : 0},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])}}



gmsh.initialize()
gmsh.model.add('test_model')

radius = 0.5
height = 1 
mesh_size = 0.25

time_step = 0.000001
alpha = 1
diffusion_coeff = 100.
rho_coeff = 1
n_ite = 4000
dump_ite = 500

# Create savedir 
if os.path.exists(savedir):
    print(savedir, ' already exists')
else : 
    os.mkdir(savedir)

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
#
mesh = TetraMesh()
mesh.gmsh_reader('test.msh')
# Init Solver
solver = FemUnsteadyDiffusion(alpha = alpha, deltat = time_step)
# init conductivity matrix
ndim = 3
identity = diffusion_coeff*np.identity(3)
solver.set_constant_diffusion(mesh,identity)
# init rho value 
solver.set_constant_rho_data(mesh,rho_coeff)
# init solved data 
solver.initialize_solved_data(mesh)
# CSet operators 
solver.build_discrete_operators(mesh,boundary_conditions)
# Temporal Loop  
for i in range(n_ite):
    if i % dump_ite == 0 :
        print(np.mean(mesh.nodes_data[solver.solved_data] ))
        save_path = savedir + f"output_{i:04d}.vtk"
        print('dump solution : ', save_path)
        mesh.save_vtk(output_file = save_path)
    solver.step2(mesh,boundary_conditions) 
