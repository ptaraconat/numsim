import sys as sys
import numpy as np  
sys.path.append('../')
from fvm.convection import CentralDiffConvection
from fvm.diffusion import OrthogonalDiffusion
from meshe.mesh import Mesh1D

velocity = 0.2
diffusion_coeff = 1.

dx = 1. 
n_elem = 20
#
mesh = Mesh1D(dx,n_elem)
mesh.physical_entities = {'inlet': np.array([1,   2]), 
                          'outlet': np.array([3,   2]), 
                          'wall': np.array([2,   2])}
# set data 
arr_tmp = np.zeros((n_elem,3))
arr_tmp[:,0] = 1. 
mesh.elements_data['velocity'] = velocity * arr_tmp
mesh.elements_data['temp'] = np.zeros((n_elem,1))
n_bndf = np.size(mesh.bndfaces,0)
arr_tmp = np.zeros((n_bndf,3))
arr_tmp[:,0] = 1. 
mesh.bndfaces_data['velocity'] =   velocity * arr_tmp 

boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : 3},
                       'outlet' : {'type' : 'dirichlet',
                                   'value' : 0},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])}}

operator = CentralDiffConvection(velocity_data= 'velocity',convected_data = 'temp')
diff_op = OrthogonalDiffusion()

mat, rhs = operator(mesh,boundary_conditions)
mat_, rhs_ = diff_op(mesh, 
                     boundary_conditions, 
                     diffusion_coeff=diffusion_coeff)
mat += mat_
rhs += rhs_
solution = np.dot(np.linalg.pinv(mat),rhs)

print(solution)