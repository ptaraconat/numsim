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