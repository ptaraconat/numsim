import pytest
import sys as sys 
sys.path.append('.')
from fem.elements import * 
from meshe.mesh import * 

EPSILON = 1e-8

@pytest.fixture()
def tri_fixture(): 
    constructor = Tri3()
    return constructor
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
    return mesh 

def test_mapping(tri_fixture): 
    local_coords = np.array([0.5,0.5,0])
    element_coords = np.array([[0, 0, 0],
                               [2, 0, 0],
                               [0, 2, 0]])
    tri_fixture.set_element(element_coords)
    ret_arr = tri_fixture.mapping(local_coords)
    print(ret_arr)
    assertion = np.all(ret_arr == np.array([1, 1., 0]))
    assert assertion 

def test_mapping2(tri_fixture): 
    local_coords = np.array([0.5,0.5,0])
    element_coords = np.array([[0, 0, 0],
                               [0, 2, 0],
                               [0, 0, 2]])
    tri_fixture.set_element(element_coords)
    ret_arr = tri_fixture.mapping(local_coords)
    print(ret_arr)
    assertion = np.all(ret_arr == np.array([0, 1., 1.]))
    assert assertion 

def test_mapping_translated(tri_fixture): 
    local_coords = np.array([0.5,0.5,0])
    element_coords = np.array([[0, 0, 0],
                               [2, 0, 0],
                               [0, 2, 0]])
    translation = np.array([1,1,1])
    element_coords += translation
    tri_fixture.set_element(element_coords)
    ret_arr = tri_fixture.mapping(local_coords)
    print(ret_arr)
    assertion = np.all(ret_arr == np.array([2, 2., 1]))
    assert assertion 

def test_get_dbf(tri_fixture):
    coords = np.array([0,1,0])
    ret_arr = tri_fixture.get_dbf_array(coords) 
    print(ret_arr)
    print(np.shape(ret_arr))
    expected_arr = np.array([[-1, -1],
                             [ 1,  0],
                             [ 0,  1]])
    assertion = np.all(ret_arr == expected_arr)
    assert assertion

def test_calc_jacobian(tri_fixture):
    coords = np.array([0.5,0.5,0])
    element_coords = np.array([[0, 0, 0.0],
                               [2, 0, 0],
                               [0, 2, 0]])
    tri_fixture.set_element(element_coords)
    ret_arr, scalar, ret_arr2 = tri_fixture.calc_jacobian(coords)
    print(ret_arr)
    print(ret_arr2)
    print(scalar)
    expected_arr = np.array([[2., 0.],
                             [0., 2.],
                             [0., 0.]])
    assertion = np.all(ret_arr == expected_arr)
    expected_arr = np.array([[0.5,0],
                             [0,0.5]])
    assertion = assertion and np.all(ret_arr2 == expected_arr)
    assertion = assertion and (scalar == 4.)
    assert assertion 