import pytest
import sys as sys 
sys.path.append('.')
from solver.transport import TransportSolver
from meshe.mesh import *

@pytest.fixture()
def mesh_fixture():
    dx = 1
    n_elem = 21
    mesh = Mesh1D(dx,n_elem)
    #
    mesh = Mesh1D(dx,n_elem)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    mesh.set_elements_volumes()
    return mesh

@pytest.fixture()
def solver_fixture(): 
    solver = TransportSolver('temp',
                             'velocity',
                             'diffusivity')
    return solver

@pytest.fixture()
def solver_fixture2(): 
    solver = TransportSolver('temp',
                             'velocity',
                             'diffusivity',
                             'source')
    return solver
    

def test_data_init(mesh_fixture, solver_fixture):
    solver_fixture.initialize_data(mesh_fixture)
    n_elem = np.size(mesh_fixture.elements,0)
    n_bndf = np.size(mesh_fixture.bndfaces,0)
    exp_el_velocity = np.zeros((n_elem,3))
    exp_bf_velocity = np.zeros((n_bndf,3))
    exp_el_diff = np.zeros((n_elem,1))
    exp_bf_diff = np.zeros((n_bndf,1))
    exp_el_data = np.zeros((n_elem,1))
    exp_bf_data = np.zeros((n_bndf,1))
    assertion = [np.all(mesh_fixture.elements_data['velocity'] == exp_el_velocity), 
                 np.all(mesh_fixture.bndfaces_data['velocity'] == exp_bf_velocity),
                 np.all(mesh_fixture.elements_data['diffusivity'] == exp_el_diff), 
                 np.all(mesh_fixture.bndfaces_data['diffusivity'] == exp_bf_diff),
                 np.all(mesh_fixture.elements_data['temp'] == exp_el_data), 
                 np.all(mesh_fixture.bndfaces_data['temp'] == exp_bf_data)]
    assert assertion 

def test_set_constant_velocity(mesh_fixture, solver_fixture):
    solver_fixture.initialize_data(mesh_fixture)
    n_elem = np.size(mesh_fixture.elements,0)
    n_bndf = np.size(mesh_fixture.bndfaces,0)
    velocity = np.array([3,2,1])
    exp_elem_velocity = np.tile(velocity,(n_elem,1))
    exp_bf_velocity = np.tile(velocity, (n_bndf,1))
    solver_fixture.set_constant_velocity(mesh_fixture,velocity)
    assertion = [np.all(mesh_fixture.elements_data['velocity'] == exp_elem_velocity),
                 np.all(mesh_fixture.bndfaces_data['velocity'] == exp_bf_velocity)] 
    assert assertion  

def test_set_constant_diffusivity(mesh_fixture, solver_fixture):
    solver_fixture.initialize_data(mesh_fixture)
    n_elem = np.size(mesh_fixture.elements,0)
    n_bndf = np.size(mesh_fixture.bndfaces,0)
    diff_coeff = 3
    exp_elem_diff = diff_coeff * np.ones((n_elem,1))
    exp_bf_diff = diff_coeff * np.ones((n_bndf,1))
    solver_fixture.set_constant_diffusivity(mesh_fixture,diff_coeff)
    assertion = [np.all(mesh_fixture.elements_data['diffusivity'] == exp_elem_diff),
                 np.all(mesh_fixture.bndfaces_data['diffusivity'] == exp_bf_diff)] 
    assert assertion  
    
def test_set_constant_source(mesh_fixture, solver_fixture2):
    solver_fixture2.initialize_data(mesh_fixture)
    #
    n_elem = np.size(mesh_fixture.elements,0)
    n_bndf = np.size(mesh_fixture.bndfaces,0)
    #
    source_term = 3.5
    exp_elem_diff = source_term * np.ones((n_elem,1))
    exp_bf_diff = source_term * np.ones((n_bndf,1))
    #
    solver_fixture2.set_constant_source(mesh_fixture,source_term)
    #
    assertion = [np.all(mesh_fixture.elements_data['source'] == exp_elem_diff),
                 np.all(mesh_fixture.bndfaces_data['source'] == exp_bf_diff)] 
    assert assertion  
    
def test_convdiff_solver1(mesh_fixture,solver_fixture):
    velocity = np.array([0.8,0,0])
    diff_coeff = 1
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 3},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 0},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    #
    solver_fixture.initialize_data(mesh_fixture)
    solver_fixture.set_constant_velocity(mesh_fixture,velocity)
    # not required atm 
    solver_fixture.set_constant_diffusivity(mesh_fixture,diff_coeff)
    solver_fixture.diffusion_coeff = diff_coeff
    solver_fixture._set_operators(mesh_fixture, boundary_conditions)
    # steady solution 
    mat = solver_fixture.mat_diff + solver_fixture.mat_conv
    rhs = solver_fixture.rhs_diff + solver_fixture.rhs_conv
    static_sol = np.linalg.solve(mat,rhs)
    # Temporal Loop 
    n_ite = 300
    for i in range(n_ite):
        solver_fixture.step(mesh_fixture, boundary_conditions)
    print(mesh_fixture.elements_data['temp'])
    print(static_sol)
    #
    assertion = np.all(np.abs(static_sol - mesh_fixture.elements_data['temp']) < 1e-3)
    assert assertion 

def test_convdiff_solver_wsource(mesh_fixture,solver_fixture2):
    source_val = 1
    source_loc = 10
    velocity = np.array([0.5,0,0])
    diff_coeff = 2
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 0},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 0},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    #
    solver_fixture2.initialize_data(mesh_fixture)
    solver_fixture2.set_constant_velocity(mesh_fixture,velocity)
    # source term 
    n_elem = np.size(mesh_fixture.elements,0)
    arr_tmp = np.zeros((n_elem,1))
    arr_tmp[source_loc] = source_val
    mesh_fixture.elements_data['source'] = arr_tmp
    del arr_tmp
    # not required atm 
    solver_fixture2.set_constant_diffusivity(mesh_fixture,diff_coeff)
    solver_fixture2.diffusion_coeff = diff_coeff
    solver_fixture2._set_operators(mesh_fixture, boundary_conditions)
    # steady solution 
    mat = solver_fixture2.mat_diff + solver_fixture2.mat_conv + solver_fixture2.mat_source
    rhs = solver_fixture2.rhs_diff + solver_fixture2.rhs_conv + solver_fixture2.rhs_source
    static_sol = np.linalg.solve(mat,rhs)
    # Temporal Loop 
    n_ite = 500
    for i in range(n_ite):
        solver_fixture2.step(mesh_fixture, boundary_conditions)
    print(mesh_fixture.elements_data['temp'])
    print(static_sol)
    print(100*np.abs(static_sol - mesh_fixture.elements_data['temp'])/static_sol)
    # consider relative error in % 
    assertion = np.all(100*np.abs(static_sol - mesh_fixture.elements_data['temp'])/static_sol < 1)
    assert assertion 