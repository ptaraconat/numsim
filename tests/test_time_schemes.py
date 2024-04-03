import pytest
import sys as sys 
sys.path.append('.')
from tstep.fdts import * 
from fvm.diffusion import OrthogonalDiffusion
from meshe.mesh import *

@pytest.fixture()
def mesh_fixture():
    dx = 1
    n_elem = 10
    mesh = Mesh1D(dx,n_elem)
    #
    mesh = Mesh1D(dx,n_elem)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    mesh.set_elements_volumes()
    # set data 
    mesh.elements_data['temp'] = np.zeros((n_elem,1))
    #
    #arr_tmp = np.zeros((n_elem,3))
    #arr_tmp[:,0] = 1. 
    #mesh.elements_data['velocity'] = velocity * arr_tmp
    #n_bndf = np.size(mesh.bndfaces,0)
    #arr_tmp = np.zeros((n_bndf,3))
    #arr_tmp[:,0] = 1. 
    #mesh.bndfaces_data['velocity'] =   velocity * arr_tmp 
    return mesh

def test_euler_diffusion(mesh_fixture):
    diffusion_coeff = 1.
    dt = 0.1
    tstepper = EulerScheme()
    tstepper.set_timestep(dt)
    diffop = OrthogonalDiffusion()
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 3},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 0},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    #
    mat_d, rhs_d = diffop(mesh_fixture,
                          boundary_conditions, 
                          diffusion_coeff = diffusion_coeff)
    static_sol = np.linalg.solve(mat_d,rhs_d)
    print(mat_d)
    print(rhs_d)
    print(static_sol)
    #
    current_array = mesh_fixture.elements_data['temp']
    n_ite = 1000 
    for i in range(n_ite):
        implicit_contribution = mat_d
        explicit_contribution = -rhs_d
        current_array = tstepper.step(current_array, mesh_fixture, implicit_contribution, explicit_contribution)
    print(current_array)
    print(np.abs(static_sol - current_array))
    #
    assertion = np.all(np.abs(static_sol - current_array) < 1e-3)
    assert assertion
    