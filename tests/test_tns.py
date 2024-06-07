import pytest
import sys as sys 
sys.path.append('.')
from solver.ns_transient import TNSSolver
from meshe.mesh import *

EPSILON = 1e-5

@pytest.fixture()
def solver_fixture(): 
    solver = TNSSolver(time_scheme = 'forward_euler')#, convection_scheme = 'upwind')
    return solver

@pytest.fixture()
def simple_mesh_fixture():
    dx = 1
    n_elem = 3
    mesh = Mesh1D(dx,n_elem)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    mesh.set_elements_volumes()
    return mesh 

@pytest.fixture()
def mesh_fixture():
    dx = 2
    n_elem = 21
    mesh = Mesh1D(dx,n_elem)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    mesh.set_elements_volumes()
    #
    velocity = 1. 
    arr_tmp = np.zeros((n_elem,3))
    arr_tmp[:,0] = velocity * 1. 
    mesh.elements_data['velocity'] =  arr_tmp
    n_bndf = np.size(mesh.bndfaces,0)
    arr_tmp = np.zeros((n_bndf,3))
    arr_tmp[:,0] = velocity * 1. 
    arr_tmp[0,0] = 10
    mesh.bndfaces_data['velocity'] =    arr_tmp 
    return mesh 

def test_step(solver_fixture, mesh_fixture):
    density = 1000 
    dyna_visco = 1e-0
    velocity = 0.001
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([velocity,0,0])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.initialize_data(mesh_fixture)
    solver_fixture.set_constant_kinematic_viscosity(mesh_fixture, density, dyna_visco)
    solver_fixture.set_boundary_conditions(boundary_conditions= boundary_conditions)
    #
    #solver_fixture.set_time_step(mesh_fixture)
    for _ in range(5):
        solver_fixture.timeop.dt = 1
        solver_fixture.projection_step(mesh_fixture)
        solver_fixture.poisson_step(mesh_fixture)
        solver_fixture.correction_step(mesh_fixture)
        solver_fixture.calc_time_step(mesh_fixture,0.001)
    print(mesh_fixture.elements_data[solver_fixture.pressure_data])
    print(mesh_fixture.elements_data[solver_fixture.velocity_data])
    n_elem = np.size(mesh_fixture.elements_data[solver_fixture.velocity_data],0)
    expected_velocity = np.zeros((n_elem,3))
    expected_velocity[:,0] = velocity
    diff = np.abs(expected_velocity - mesh_fixture.elements_data[solver_fixture.velocity_data])
    print(diff)
    assertion = np.all(diff < EPSILON)
    
    assert assertion 