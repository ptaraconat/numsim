import sys as sys
import numpy as np  
sys.path.append('../')
from fvm.convection import CentralDiffConvection
from fvm.diffusion import OrthogonalDiffusion
from tstep.fdts import ForwardEulerScheme
from meshe.mesh import Mesh1D

fourier = 0.49
cfl = 0.6

velocity = 0.2
diffusion_coeff = 1.

dx = 1. 
n_elem = 20
#
mesh = Mesh1D(dx,n_elem)
mesh.physical_entities = {'inlet': np.array([1,   2]), 
                          'outlet': np.array([3,   2]), 
                          'wall': np.array([2,   2])}
mesh.set_elements_volumes()
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
arr_tmp = np.ones((n_elem,1))
mesh.elements_data['diffusion'] = diffusion_coeff * arr_tmp
arr_tmp = np.ones((n_bndf,1))
mesh.bndfaces_data['diffusion'] = diffusion_coeff * arr_tmp

boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                  'value' : 3},
                       'outlet' : {'type' : 'dirichlet',
                                   'value' : 0},
                       'wall' : {'type' : 'neumann',
                                 'value' : np.array([0,0,0])}}

conv_op = CentralDiffConvection(velocity_data= 'velocity')
diff_op = OrthogonalDiffusion(diffusion_data= 'diffusion')
tstepper = ForwardEulerScheme()
#
meshsize = mesh.elements_volumes**(1/3)
velocity_array = mesh.elements_data['velocity']
diffusion_array = mesh.elements_data['diffusion']
dt_diff = tstepper._calc_dt_diff(fourier,diffusion_array,1,meshsize)
dt_conv = tstepper._calc_dt_conv(cfl,velocity_array, meshsize)
dt = np.min([dt_diff, dt_conv])
tstepper.set_timestep(dt)
#
mat_c, rhs_c = conv_op(mesh,boundary_conditions)
mat_d, rhs_d = diff_op(mesh, 
                     boundary_conditions)
mat = mat_c + mat_d
rhs = rhs_c + rhs_d
static_solution = np.dot(np.linalg.pinv(mat),rhs)
print(static_solution)
# Temporal loop
current_array = mesh.elements_data['temp']
n_ite = 1000 
for i in range(n_ite):
    implicit_contribution = mat
    explicit_contribution = rhs
    current_array = tstepper.step(current_array, mesh, implicit_contribution, explicit_contribution)
print(current_array)
print(dt)