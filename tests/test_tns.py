import pytest
import sys as sys 
sys.path.append('.')
from solver.ns_transient import TNSSolver
from meshe.mesh import *

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
    dx = 1
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

def test_sourceop(solver_fixture, mesh_fixture):
    solver_fixture.initialize_data(mesh_fixture)
    solver_fixture.set_constant_kinematic_viscosity(mesh_fixture, 1000, 1e-3)
    solver_fixture.set_source_operators(mesh_fixture)
    assertion = False 
    assert assertion 
    

def test_step(solver_fixture, mesh_fixture):
    density = 1000 
    dyna_visco = 1e-0
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([0.001,0,0])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.initialize_data(mesh_fixture)
    solver_fixture.set_constant_kinematic_viscosity(mesh_fixture, density, dyna_visco)
    solver_fixture.set_boundary_conditions(boundary_conditions= boundary_conditions)
    #
    #solver_fixture.set_time_step(mesh_fixture)
    for _ in range(100):
        solver_fixture.timeop.dt = 1
        solver_fixture.projection_step(mesh_fixture)
        solver_fixture.poisson_step(mesh_fixture)
        solver_fixture.correction_step(mesh_fixture)
        solver_fixture.calc_time_step(mesh_fixture,0.001)
    #print(mesh_fixture.elements_data[solver_fixture.velocity_data])
    
    
    assertion = False 
    
    assert assertion 