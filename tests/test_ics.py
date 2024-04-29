import pytest
import sys as sys 
sys.path.append('.')
from solver.incompressible import IncompressibleSolver
from meshe.mesh import *

@pytest.fixture()
def solver_fixture(): 
    solver = IncompressibleSolver()
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
    return mesh 

def test_set_bnd_velocity(simple_mesh_fixture, solver_fixture):
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([1,0,0])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.set_boundary_conditions(boundary_conditions)
    solver_fixture.init_data(simple_mesh_fixture)
    solver_fixture.update_boundary_velocity(simple_mesh_fixture, 
                                            boundary_conditions)
    print(simple_mesh_fixture.bndfaces_data[solver_fixture.velocity_data])
    bnd_velocity = simple_mesh_fixture.bndfaces_data[solver_fixture.velocity_data]
    expected = np.array([[1, 0, 0],
                         [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0],
                         [0, 0, 0]])
    assertion = np.all(bnd_velocity == expected) 
    assert assertion 
    
def test_setbc(solver_fixture):
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([1,2,3])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.set_boundary_conditions(boundary_conditions)
    assertion = False
    assert assertion 

def test_advance_velocity(mesh_fixture, solver_fixture):
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([1,0,0])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.set_boundary_conditions(boundary_conditions)
    solver_fixture.init_data(mesh_fixture)
    density_value = 1000.
    dyna_visco = 1e-3
    solver_fixture.set_constant_kinematic_viscosity(mesh_fixture, density_value, dyna_visco)
    n_ite = 1
    for _ in range(n_ite):
        print(_)
        solver_fixture.advance_velocity(mesh_fixture)
        solver_fixture.update_boundary_velocity(mesh_fixture, 
                                                boundary_conditions)
    print(mesh_fixture.elements_data[solver_fixture.velocity_data])
    print(np.mean(mesh_fixture.elements_data[solver_fixture.velocity_data],axis = 0))
    assertion = False
    assert assertion 
    
def test_pressure_lapl(mesh_fixture, solver_fixture):
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([1,0,0])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : 1000},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.set_boundary_conditions(boundary_conditions)
    solver_fixture.init_data(mesh_fixture)
    solver_fixture._set_pl_operators(mesh_fixture)
    print(solver_fixture.mat_pressure)
    print(solver_fixture.rhs_pressure)
    assertion = False
    assert assertion 
