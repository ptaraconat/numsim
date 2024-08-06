import pytest
import sys as sys 
sys.path.append('.')
from fem.elements import * 

@pytest.fixture()
def tet_fixture(): 
    constructor = Tet4()
    return constructor

def test_get_bf(tet_fixture):
    coords = np.array([0,1,0])
    ret_arr = tet_fixture.get_bf_array(coords) 
    assertion = np.all(ret_arr == np.array([1, 0, 0, 0]))
    coords = np.array([0,0,1])
    ret_arr = tet_fixture.get_bf_array(coords) 
    assertion = np.all(ret_arr == np.array([0, 1, 0, 0]))
    coords = np.array([0,0,0])
    ret_arr = tet_fixture.get_bf_array(coords) 
    assertion = np.all(ret_arr == np.array([0, 0, 1, 0]))
    coords = np.array([1,0,0])
    ret_arr = tet_fixture.get_bf_array(coords) 
    assertion = np.all(ret_arr == np.array([0, 0, 0, 1]))
    assert assertion

def test_mapping(tet_fixture): 
    local_coords = np.array([0.5,0.5,0.5])
    element_coords = np.array([[0, 2, 0],
                               [0, 0, 2],
                               [0, 0, 0],
                               [2, 0, 0]])
    tet_fixture.set_element(element_coords)
    ret_arr = tet_fixture.mapping(local_coords)
    assertion = np.all(ret_arr == np.array([1., 1., 1.]))
    assert assertion 

def test_mapping_translated(tet_fixture): 
    local_coords = np.array([0.5,0.5,0.5])
    element_coords = np.array([[0, 1, 0],
                               [0, 0, 1],
                               [0, 0, 0],
                               [1, 0, 0]])
    translation = np.array([1,1,1])
    element_coords += translation
    print(element_coords)
    tet_fixture.set_element(element_coords)
    ret_arr = tet_fixture.mapping(local_coords)
    print(ret_arr)
    assertion = np.all(ret_arr == np.array([1.5, 1.5, 1.5]))
    assert assertion
    
def test_get_dbf(tet_fixture):
    coords = np.array([0,1,0])
    ret_arr = tet_fixture.get_dbf_array(coords) 
    print(ret_arr)
    print(np.shape(ret_arr))
    assertion = False 
    assert assertion 
    
def test_calc_jacobian(tet_fixture):
    coords = np.array([0,1,0])
    element_coords = np.array([[0, 2, 0],
                               [0, 0, 2],
                               [0, 0, 0],
                               [2, 0, 0]])
    tet_fixture.set_element(element_coords)
    ret_arr, scalar, ret_arr2 = tet_fixture.calc_jacobian(coords)
    print(ret_arr)
    print(scalar)
    print(ret_arr2)
    assertion = False 
    assert assertion 