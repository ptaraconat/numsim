import gmsh 
import numpy as np 
import sys as sys 
sys.path.append('../')
from meshe.mesh import * 
from fem.elements import *
from solver.fem_linear_elasticity import * 

boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : [0, 0, 0]},
                       'outlet' : {'type' : 'dirichlet',
                                   'value' : [None, None, 0.1]},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])}}

param_dict = {'STATE_LAW' : 'HOM_ISO',
              'HOM_ISO_POISSON' : 0.3,
              'HOM_ISO_YOUNG' : 200e9,
              'EL_TYPE' : 'TET4',
              'DUMP_DIR' : '3Dcyl_fem_linel/',
              'DUMP_DISPLACEMENT_SCALING' : 1.}

radius = 0.5
height = 1 
mesh_size = 0.25

#################################
#################################
gmsh.initialize()
gmsh.model.add('test_model')
#
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
#################################
#################################

###############
# Init Solver##
solver = FemLinearElasticity(boundary_conditions, param_dict = param_dict)
# Solve EDP
solver.solve(mesh)
#


