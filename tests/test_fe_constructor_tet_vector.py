import pytest
import sys as sys 
sys.path.append('.')
from fem.elements import * 
from meshe.mesh import * 

EPSILON = 1e-8

@pytest.fixture()
def tet_fixture(): 
    constructor = Tet4Vector()
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

def test_calc_global_bf_derivatives(tet_fixture):
    # test case setup 
    element_coords = np.array([[0, 1, 0],
                               [0, 0, 1],
                               [0., 0., 0.],
                               [1, 0, 0]])
    tet_fixture.set_element(element_coords)
    #
    coords = np.array([1,1,0])
    _, _, inv_jacobian = tet_fixture.calc_jacobian(coords)
    ret_arr = tet_fixture.calc_global_dbf_array_symgrad(coords,inv_jacobian)
    print(ret_arr)
    expected_arr = np.array([[ 0.,  0.,  0.,  1.,  0.,  0.],
                             [ 0.,  1.,  0.,  0.,  0.,  0.],
                             [ 0.,  0.,  0.,  0.,  0.,  1.],
                             [ 0.,  0.,  0.,  0.,  1.,  0.],
                             [ 0.,  0.,  0.,  0.,  0.,  1.],
                             [ 0.,  0.,  1.,  0.,  0.,  0.],
                             [-1.,  0.,  0., -1., -1.,  0.],
                             [ 0., -1.,  0., -1.,  0., -1.],
                             [ 0.,  0., -1.,  0., -1., -1.],
                             [ 1.,  0.,  0.,  0.,  0.,  0.],
                             [ 0.,  0.,  0.,  1.,  0.,  0.],
                             [ 0.,  0.,  0.,  0.,  1.,  0.]])
    assertion = np.all(ret_arr == expected_arr)
    assert assertion 

def test_calc_stiff_integrand(tet_fixture):
    # test case setup 
    element_coords = np.array([[0, 1, 0],
                               [0, 0, 1],
                               [0., 0., 0.],
                               [1, 0, 0]])
    tet_fixture.set_element(element_coords)
    #
    coords = np.array([0,0,0])
    state_mat = np.array([[1,0,0,0,0,0],
                          [0,1,0,0,0,0],
                          [0,0,1,0,0,0],
                          [0,0,0,1,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,0,0,1]])
    ret_arr = tet_fixture.calc_stifness_integrand(coords, state_mat)
    expected_arr = np.array([[ 1.,  0.,  0.,  0.,  0.,  0., -1., -1.,  0.,  0.,  1.,  0.],
                             [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
                             [ 0.,  0.,  1.,  0.,  1.,  0.,  0., -1., -1.,  0.,  0.,  0.],
                             [ 0.,  0.,  0.,  1.,  0.,  0., -1.,  0., -1.,  0.,  0.,  1.],
                             [ 0.,  0.,  1.,  0.,  1.,  0.,  0., -1., -1.,  0.,  0.,  0.],
                             [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,  0.],
                             [-1.,  0.,  0., -1.,  0.,  0.,  3.,  1.,  1., -1., -1., -1.],
                             [-1., -1., -1.,  0., -1.,  0.,  1.,  3.,  1.,  0., -1.,  0.],
                             [ 0.,  0., -1., -1., -1., -1.,  1.,  1.,  3.,  0.,  0., -1.],
                             [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.],
                             [ 1.,  0.,  0.,  0.,  0.,  0., -1., -1.,  0.,  0.,  1.,  0.],
                             [ 0.,  0.,  0.,  1.,  0.,  0., -1.,  0., -1.,  0.,  0.,  1.]])
    print(ret_arr)
    assertion = np.all(ret_arr == expected_arr)
    assert assertion 

def test_set_state_mat(tet_fixture): 
    state_arr = np.array([[1,0,0,0,0,0],
                          [0,1,0,0,0,0],
                          [0,0,1,0,0,0],
                          [0,0,0,1,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,0,0,1]])
    tet_fixture.set_state_matrices(state_arr)
    ret_arr = tet_fixture.state_matrices
    expected_arr = np.zeros((4,6,6))
    for i in range(4): 
        expected_arr[i,:,:] = state_arr
    print(ret_arr)
    assertion = np.all(ret_arr == expected_arr)
    assert assertion
    
def test_stiffness_mat(tet_fixture):
    state_arr = np.array([[1,0,0,0,0,0],
                          [0,1,0,0,0,0],
                          [0,0,1,0,0,0],
                          [0,0,0,1,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,0,0,1]])
    tet_fixture.set_state_matrices(state_arr)
    ret_arr = tet_fixture.calc_stifness_matrix()
    expected_arr = np.array([[ 0.16666667,  0.,          0.,          0.,          0.,          0., -0.16666667, -0.16666667,  0.,          0.,          0.16666667,  0.        ],
                             [ 0.,          0.16666667,  0.,          0.,          0.,          0., 0.,         -0.16666667,  0.,          0.,          0.,          0.        ],
                             [ 0.,          0.,          0.16666667,  0.,          0.16666667,  0., 0.,         -0.16666667, -0.16666667,  0.,          0.,          0.        ],
                             [ 0.,          0.,          0.,          0.16666667,  0.,          0., -0.16666667,  0.,         -0.16666667,  0.,          0.,          0.16666667],
                             [ 0.,          0.,          0.16666667,  0.,          0.16666667,  0., 0.,         -0.16666667, -0.16666667,  0.,          0.,          0.        ],
                             [ 0.,          0.,          0.,          0.,          0.,          0.16666667, 0.,          0.,         -0.16666667,  0.,          0.,          0.        ],
                             [-0.16666667,  0.,          0.,         -0.16666667,  0.,          0., 0.5,         0.16666667,  0.16666667, -0.16666667, -0.16666667, -0.16666667],
                             [-0.16666667, -0.16666667, -0.16666667,  0.,         -0.16666667,  0., 0.16666667,  0.5,         0.16666667,  0.,         -0.16666667,  0.        ],
                             [ 0.,          0.,         -0.16666667, -0.16666667, -0.16666667, -0.16666667, 0.16666667,  0.16666667,  0.5,         0.,          0.,         -0.16666667],
                             [ 0.,          0.,          0.,          0.,          0.,          0., -0.16666667,  0.,          0.,          0.16666667,  0.,          0.        ],
                             [ 0.16666667,  0.,          0.,          0.,          0.,          0., -0.16666667, -0.16666667,  0.,          0.,          0.16666667,  0.        ],
                             [ 0.,          0.,          0.,          0.16666667,  0.,          0., -0.16666667,  0.,         -0.16666667,  0.,          0.,          0.16666667]])
    print(ret_arr)
    print(np.shape(ret_arr))
    print(np.abs(ret_arr-expected_arr))
    assertion = np.all(np.abs(ret_arr-expected_arr)<EPSILON) 
    assert assertion 

def test_global_stiffness(tet_fixture,mesh_fixture): 
    # set state data 
    state_data = 'conductivity'
    nnodes = np.size(mesh_fixture.nodes,0)
    ndim = 6
    state = np.zeros((nnodes,ndim,ndim))
    identity = np.identity(ndim)
    for i in range(nnodes): 
        state[i,:,:] = identity
    mesh_fixture.nodes_data[state_data] = state
    #
    ret_arr = tet_fixture.calc_global_stiffness_matrix( mesh_fixture, state_data)
    expected_arr = np.array([])
    #print(ret_arr)
    #print(np.abs(ret_arr-expected_arr))
    assertion = False #np.all(np.abs(ret_arr-expected_arr)<EPSILON) 
    assert assertion 