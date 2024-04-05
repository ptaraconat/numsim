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

def test_calc_diff_dt():
    tstepper = TimeSteping()
    fourier = 0.5
    n_elem = 5
    diffusivity_array = 5*np.ones((n_elem,1))
    rho_array = 1000*np.ones((n_elem,1))
    meshsize_array = 0.01*np.ones((n_elem,1))
    dt = tstepper._calc_dt_diff(fourier,
                                diffusivity_array,
                                rho_array,
                                meshsize_array)
    print(dt)
    assertion = np.abs(dt - 0.01) < 1e-16
    assert assertion 

def test_forward_euler_diffusion(mesh_fixture):
    diffusion_coeff = 1.
    fourier = 0.49
    tstepper = ForwardEulerScheme()
    meshsize = mesh_fixture.elements_volumes**(1/3)
    dt = tstepper._calc_dt_diff(fourier,diffusion_coeff,1,meshsize)
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
    #
    current_array = mesh_fixture.elements_data['temp']
    n_ite = 1000 
    for i in range(n_ite):
        implicit_contribution = mat_d
        explicit_contribution = rhs_d
        current_array = tstepper.step(current_array, mesh_fixture, implicit_contribution, explicit_contribution)
    print(current_array)
    print(np.abs(static_sol - current_array))
    print(dt)
    #
    assertion = np.all(np.abs(static_sol - current_array) < 1e-3)
    assert assertion


def test_backward_euler_diffusion(mesh_fixture):
    diffusion_coeff = 1.
    dt = 1
    tstepper = BackwardEulerScheme()
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
    #
    current_array = mesh_fixture.elements_data['temp']
    implicit_contribution = mat_d
    explicit_contribution =  rhs_d
    n_ite = 100
    for _ in range(n_ite):
        current_array = tstepper.step(current_array, mesh_fixture, implicit_contribution, explicit_contribution)
    print(current_array)
    print(np.abs(static_sol - current_array))
    #
    assertion = np.all(np.abs(static_sol - current_array) < 1e-3)
    assert assertion
    

def test_cn_diffusion(mesh_fixture):
    diffusion_coeff = 1.
    dt = 2
    tstepper = CNScheme()
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
    #
    current_array = mesh_fixture.elements_data['temp']
    implicit_contribution = mat_d
    explicit_contribution =  rhs_d
    n_ite = 50
    for _ in range(n_ite):
        current_array = tstepper.step(current_array, mesh_fixture, implicit_contribution, explicit_contribution)
    print(current_array)
    print(np.abs(static_sol - current_array))
    #
    assertion = np.all(np.abs(static_sol - current_array) < 1e-3)
    assert assertion