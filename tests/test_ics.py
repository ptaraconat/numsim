import pytest
import sys as sys 
sys.path.append('.')
from solver.incompressible import IncompressibleSolver
from meshe.mesh import *

@pytest.fixture()
def solver_fixture(): 
    solver = IncompressibleSolver()
    return solver


def test_setbc(solver_fixture):
    boundary_conditions = {'inlet' : {'type' : 'inlet',
                                      'value' : np.array([1,2,3])},
                           'outlet' : {'type' : 'outlet',
                                       'value' : None},
                           'wall' : {'type' : 'wall',
                                     'value' : None}}
    solver_fixture.set_boundary_conditions(boundary_conditions)
    assertion = False 
    assert assertion 