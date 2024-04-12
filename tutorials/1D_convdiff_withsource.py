import sys as sys
import numpy as np  
sys.path.append('../')
from fvm.convection import CentralDiffConvection
from fvm.diffusion import OrthogonalDiffusion
from fvm.source_term import SourceTerm
from meshe.mesh import Mesh1D

velocity = 0.2#0.2
diffusion_coeff = 0.5#1.
source_val = 1

dx = 1. 
n_elem = 20
source_loc = 10
#
mesh = Mesh1D(dx,n_elem)
mesh.physical_entities = {'inlet': np.array([1,   2]), 
                          'outlet': np.array([3,   2]), 
                          'wall': np.array([2,   2])}
# set data 
n_bndf = np.size(mesh.bndfaces,0)
#
arr_tmp = np.zeros((n_elem,3))
arr_tmp[:,0] = 1. 
mesh.elements_data['velocity'] = velocity * arr_tmp
mesh.elements_data['temp'] = np.zeros((n_elem,1))
arr_tmp = np.zeros((n_bndf,3))
arr_tmp[:,0] = 1. 
mesh.bndfaces_data['velocity'] =   velocity * arr_tmp 
#
arr_tmp = np.zeros((n_elem,))
arr_tmp[source_loc] = source_val
mesh.elements_data['source'] = arr_tmp
#
arr_tmp = np.ones((n_elem,1))
mesh.elements_data['diffusion'] = diffusion_coeff * arr_tmp
arr_tmp = np.ones((n_bndf,1))
mesh.bndfaces_data['diffusion'] = diffusion_coeff * arr_tmp

boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : 0},
                       'outlet' : {'type' : 'dirichlet',
                                   'value' : 0},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])}}

operator = CentralDiffConvection(velocity_data= 'velocity')
diff_op = OrthogonalDiffusion(diffusion_data= 'diffusion')
source_op = SourceTerm(data_name = 'source')

mat_d, rhs_d = operator(mesh,boundary_conditions)
mat_c, rhs_c = diff_op(mesh, 
                     boundary_conditions)
rhs_source = source_op(mesh)
mat = mat_d + mat_c
rhs = rhs_d + rhs_c + rhs_source
solution = np.dot(np.linalg.pinv(mat),rhs)

print(solution)