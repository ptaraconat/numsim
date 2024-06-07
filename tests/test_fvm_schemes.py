import pytest
import sys as sys 
sys.path.append('.')
from fvm.gradient import * 
from meshe.mesh import TetraMesh
from fvm.interpolation import * 

EPSILON = 1e-10

@pytest.fixture
def mesh_fixture():
    mesh = TetraMesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1],
                           [-1, 0, 0]])
    mesh.elements = np.array([[0, 1, 2, 3],
                              [0,4,1,3]])
    mesh.bndfaces = np.array([[0,1,2],
                              [0,2,3],
                              [1,2,3],
                              [0,4,1],
                              [0,4,3],
                              [4,1,3]])
    
    mesh.set_internal_faces()
    mesh.set_elements_intfaces_connectivity()
    mesh.set_boundary_faces()
    def function(x,y,z):
        return 4*x + 0*y + 0*z
    mesh.set_elements_data('T', function)
    mesh.set_bndfaces_data('T', function)
    return mesh

@pytest.fixture 
def lsgrad_fixture():
    grad_computer = LSGradient('T','gradT')
    return grad_computer


def test_ls_gradient(lsgrad_fixture, mesh_fixture): 
    gradient0 = lsgrad_fixture.calc_element_gradient(0, mesh_fixture)
    gradient1 = lsgrad_fixture.calc_element_gradient(1, mesh_fixture)
    expected_grad = np.array([4, 0, 0])
    assertion = np.all(gradient0 == expected_grad) and np.all(gradient1 == expected_grad)
    assert assertion 

def test_ls_gradient2(lsgrad_fixture, mesh_fixture): 
    lsgrad_fixture(mesh_fixture)
    gradients = mesh_fixture.elements_data['gradT']
    expected_grad = np.array([[4., 0, 0],
                              [4., 0, 0]])
    print(gradients)
    assertion = np.all(gradients == expected_grad)
    assert assertion 
      
def test_ls_gradient_withweights(lsgrad_fixture, mesh_fixture):
    lsgrad_fixture.weighting = True 
    gradient0 = lsgrad_fixture.calc_element_gradient(0, mesh_fixture)
    gradient1 = lsgrad_fixture.calc_element_gradient(1, mesh_fixture)
    expected_grad = np.array([4, 0, 0])
    print(gradient0,gradient1)
    assertion = assertion = np.all(gradient0 == expected_grad) and np.all(gradient1 == expected_grad)
    assert assertion 
    
def test_face_interpolation():
    surface_element = np.array([[0, 0, 0.],
                                [0, 1., 0.],
                                [1., 1., 0.],
                                [1., 0, 0.]])
    node1 = np.array([0.25, 0.25, -0.25])
    node2 = np.array([0.25, 0.25, 0.5])
    value1 = 0
    value2 = 9
    pair_face_intersc = Mesh()._calc_face_pairnode_intersection(surface_element,
                                                                node1,
                                                                node2)
    print(pair_face_intersc)
    interpolator = FaceInterpolattion()
    interpolated_val = interpolator.face_computation(node1,
                                                     node2,
                                                     value1,
                                                     value2,
                                                     pair_face_intersc)
    print(interpolated_val)
    expected_value = 3.
    print(expected_value)
    assertion =  interpolated_val == expected_value
    assert assertion 
    
def test_face_gradient_interpolation():
    surface_element = np.array([[0, 0, 0.],
                                [0, 1., 0.],
                                [1., 1., 0.],
                                [1., 0, 0.]])
    node1 = np.array([0.25, 0.25, -0.25])
    node2 = np.array([0.25, 0.25, 0.5])
    value1 = 0
    grad_val1 = np.array([0,1,0])
    value2 = 9
    grad_val2 = np.array([0,4,3])
    pair_face_intersc = Mesh()._calc_face_pairnode_intersection(surface_element,
                                                                node1,
                                                                node2)
    print(pair_face_intersc)
    interpolator = FaceGradientInterpolation()
    interpolated_val = interpolator.face_computation(node1,
                                                     node2,
                                                     value1,
                                                     value2,
                                                     grad_val1,
                                                     grad_val2,
                                                     pair_face_intersc)
    print(interpolated_val)
    expected_value = np.array([0,2,12])
    
    assertion = np.all(interpolated_val == expected_value)
    assert assertion 
    
def test_ls_gradient_with_bndval(lsgrad_fixture, mesh_fixture): 
    lsgrad_fixture.use_boundaries = True
    gradient0 = lsgrad_fixture.calc_element_gradient(0, mesh_fixture)
    gradient1 = lsgrad_fixture.calc_element_gradient(1, mesh_fixture)
    expected_grad = np.array([4, 0, 0])
    print(gradient0)
    print(gradient1)
    assertion = np.all(np.abs(gradient0 - expected_grad) < EPSILON) and np.all(np.abs(gradient1 - expected_grad) < EPSILON)
    assert assertion  