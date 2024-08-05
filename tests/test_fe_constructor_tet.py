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
    assertion = False 
    assert assertion