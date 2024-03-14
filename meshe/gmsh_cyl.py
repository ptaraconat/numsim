import gmsh 
import meshio
import numpy as np 
import sys as sys 
sys.path.append('../')
from meshe.mesh import * 
from fvm.gradient import *
from fvm.diffusion import * 

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

mesh = TetraMesh()
mesh.gmsh_reader('test.msh')
mesh.set_internal_faces()
mesh.set_elements_intfaces_connectivity()
print('NUmber of nodes : ', np.shape(mesh.nodes))
print('Number of elements :',np.shape(mesh.elements))
print('Number of boundary faces :',np.shape(mesh.bndfaces))
print('Number of internal faces :', np.shape(mesh.intfaces))

# set data 
def function(x,y,z):
    return 4*x + 0*y + 0*z
mesh.set_elements_data('T', function)
# calc data gradient 
grad_computer = LSGradient('T','gradT', mesh,weighting=True)
#count = 0 
#for i in range(len(mesh.elements)):
#    grad = grad_computer.calc_element_gradient(i)
#    gradx = grad[0]
#    err = np.abs((4 - gradx)/4) * 100
#    if err > 1 : 
#        count += 1
#        print(i,grad)
#        print(mesh.elements_intf_conn[i]) 
#print(count)
grad_computer.calculate_gradients()
# 
mesh.set_boundary_faces()
boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : 10},
                       'outlet' : {'type' : 'dirichlet',
                                   'value' : 0},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])}}

diffusion_op = OrthogonalDiffusion()
mat, rhs_vec = diffusion_op(mesh,boundary_conditions)
solution1 = np.linalg.solve(mat,rhs_vec)
#print(solution)
mesh.elements_data['orthodiff_solution'] = solution1
mesh.save_vtk()

# Init data 
def function(x,y,z):
    return 0*x + 0*y + 0*z
mesh.set_elements_data('temp', function)
mesh.set_elements_data('grad_temp', function)
gradient_computer = LSGradient('temp', 'grad_temp', mesh, weighting = False)
diff_op = NonOthogonalDiffusion(data_name = 'temp', 
                                grad_data_name = 'grad_temp',
                                method = 'over_relaxed')
for i in range(1): 
        mat, rhs = diff_op(mesh, 
                           boundary_conditions, 
                           diffusion_coeff=1.)
        #print(mat)
        #print(rhs)
        solution = np.linalg.solve(mat,rhs)
        mesh.elements_data['temp'] = solution
        gradient_computer.calculate_gradients()
#print(np.abs(mesh.elements_data['temp']-solution1))