import pytest
import sys as sys 
sys.path.append('../')
from meshe.mesh import * 
from fvm.gradient import * 

@pytest.fixture 
def lsgrad_fixture():
    mesh = TetraMesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1],
                           [-1, 0, 0]])
    mesh.elements = np.array([[0, 1, 2, 3],
                              [0,4,1,3]])
    mesh.set_internal_faces()
    def function(x,y,z):
        return 4*x + 0*y + 0*z
    mesh.set_elements_data('T', function)
    grad_computer = LSGradient('gradT', mesh)
    return grad_computer
    
def test_ls_gradient(lsgrad_fixture): 
    lsgrad_fixture.calc_element_gradient(0)
    lsgrad_fixture.calc_element_gradient(1)
    assertion = False 
    assert assertion 