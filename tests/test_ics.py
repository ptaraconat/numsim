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
def mesh_fixture():
    dx = 1
    n_elem = 21
    mesh = Mesh1D(dx,n_elem)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    mesh.set_elements_volumes()
    return mesh 

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
    solver_fixture.advance_velocity(mesh_fixture)
    assertion = False 
    assert assertion 