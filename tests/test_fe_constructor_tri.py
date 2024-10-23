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
@pytest.fixture()
def tri_fixture2(): 
    constructor = Tri3(variable_dimension=3)
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

def test_calc_bndflux(tri_fixture):
    coordinates = np.array([0.5,0.5,0])
    element_coords = np.array([[0, 0, 0.0],
                               [2, 0, 0],
                               [0, 2, 0]])
    tri_fixture.set_element(element_coords)
    flux = 10
    ret_arr = tri_fixture.calc_bndflux_integrand(coordinates, flux)
    print(ret_arr)
    expected_arr = np.array([[0],[20],[20]])
    assertion = np.all(ret_arr == expected_arr) 
    assert assertion 

def test_calc_bndflux_integrand(tri_fixture2):
    coordinates = np.array([0.5,0.5,0])
    element_coords = np.array([[0, 0, 0.0],
                               [2, 0, 0],
                               [0, 2, 0]])
    tri_fixture2.set_element(element_coords)
    flux = np.array([10,20,30])
    ret_arr = tri_fixture2.calc_bndflux_integrand(coordinates, flux)
    print(ret_arr)
    expected_arr = np.array([[0],[0],[0],[20],[40],[60],[20],[40],[60]])
    assertion = np.all(ret_arr == expected_arr) 
    assert assertion 

def test_fluxes_interpolation(tri_fixture2):
    fluxes = np.array([[10,10,10],
                       [20,20,20],
                       [30,30,30]])
    ret_arr = tri_fixture2.interpolate_fluxes(fluxes)
    print(ret_arr)
    expected_arr = np.array([[15., 15., 15.],
                             [20., 20., 20.],
                             [25., 25., 25.]])
    assertion = np.all(ret_arr == expected_arr)
    assert assertion 

def test_fluxes_interpolation2(tri_fixture):
    fluxes = np.array([[10],
                       [20],
                       [30]])
    ret_arr = tri_fixture.interpolate_fluxes(fluxes)
    print(ret_arr)
    expected_arr = np.array([[15],
                             [20],
                             [25]])
    assertion = np.all(ret_arr == expected_arr)
    assert assertion 

def test_set_fluxes(tri_fixture2):
    fluxes = np.array([[10,10,10],
                       [20,20,20],
                       [30,30,30]])
    tri_fixture2.set_fluxes(fluxes)
    ret_arr = tri_fixture2.gauss_point_fluxes
    print(ret_arr)
    expected_arr = np.array([[15., 15., 15.],
                             [20., 20., 20.],
                             [25., 25., 25.]])
    assertion = np.all(ret_arr == expected_arr)
    assert assertion

def test_set_fluxes2(tri_fixture2):
    fluxes = np.array([10,10,10])
    tri_fixture2.set_fluxes(fluxes)
    ret_arr = tri_fixture2.gauss_point_fluxes
    print(ret_arr)
    expected_arr = np.array([[10., 10., 10.],
                             [10., 10., 10.],
                             [10., 10., 10.]])
    assertion = np.all(ret_arr == expected_arr)
    assert assertion

def test_set_fluxes3(tri_fixture):
    fluxes = np.array([[10],
                       [20],
                       [30]])
    tri_fixture.set_fluxes(fluxes)
    ret_arr = tri_fixture.gauss_point_fluxes
    print(ret_arr)
    expected_arr = np.array([[15],
                             [20],
                             [25]])
    assertion = np.all(ret_arr == expected_arr)
    assert assertion

def test_set_fluxes4(tri_fixture):
    fluxes = 10
    tri_fixture.set_fluxes(fluxes)
    ret_arr = tri_fixture.gauss_point_fluxes
    print(ret_arr)
    expected_arr = np.array([[10],
                             [10],
                             [10]])
    assertion = np.all(ret_arr == expected_arr)
    assert assertion

def test_calc_bndfluxes(tri_fixture2):
    # setup test case 
    element_coords = np.array([[0, 0, 0.0],
                               [1, 0, 0],
                               [0, 1, 0]])
    fluxes = 6*np.array([[10,10,10],
                       [10,10,10],
                       [10,10,10]])
    tri_fixture2.set_fluxes(fluxes)
    tri_fixture2.set_element(element_coords)
    # 
    ret_arr = tri_fixture2.calc_bndflux()
    print(ret_arr)
    expected_arr = np.array([[10],[10],[10],
                             [10],[10],[10],
                             [10],[10],[10]])
    assertion = np.all((ret_arr - expected_arr) < EPSILON )
    assert assertion 

def test_calc_bndfluxes2(tri_fixture):
    # setup test case 
    element_coords = np.array([[0, 0, 0.0],
                               [2, 0, 0],
                               [0, 2, 0]])
    fluxes = 6*np.array([[10],
                       [10],
                       [10]])
    tri_fixture.set_fluxes(fluxes)
    tri_fixture.set_element(element_coords)
    # 
    ret_arr = tri_fixture.calc_bndflux()
    print(ret_arr)
    expected_arr = np.array([[40],
                             [40],
                             [40]])
    assertion = np.all((ret_arr - expected_arr) < EPSILON )
    assert assertion 

