import pytest
import sys as sys 
sys.path.append('.')
from fvm.convection import * 
from fvm.diffusion import * 
from meshe.mesh import Mesh, TetraMesh

EPSILON = 1e-10

@pytest.fixture 
def mesh_fixture():
    mesh = TetraMesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1],
                           [0, 0, -1]])
    mesh.elements = np.array([[0, 1, 2, 3],
                              [0,4,1,3]])
    mesh.set_internal_faces()
    mesh.set_elements_intfaces_connectivity()
    velocity_arr = np.array([[]])
    mesh.elements_data['velocity'] = velocity_arr
    return mesh 

@pytest.fixture
def mesh_fixture1d():
    dx = 1. 
    n_elem = 20
    velocity = 0.2
    #
    mesh = Mesh1D(dx,n_elem)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    # set data 
    arr_tmp = np.zeros((n_elem,3))
    arr_tmp[:,0] = 1. 
    mesh.elements_data['velocity'] = velocity * arr_tmp
    mesh.elements_data['temp'] = np.zeros((n_elem,1))
    n_bndf = np.size(mesh.bndfaces,0)
    arr_tmp = np.zeros((n_bndf,3))
    arr_tmp[:,0] = 1. 
    mesh.bndfaces_data['velocity'] =   velocity * arr_tmp 

    return mesh
    
@pytest.fixture 
def mesh_fixture4():
    mesh = Mesh()
    dx = 1. 
    mesh.nodes = np.array([[0, 0 ,0],
                           [0, dx, 0],
                           [0, dx, dx],
                           [0, 0, dx],
                           [dx, 0 ,0],
                           [dx, dx, 0],
                           [dx, dx, dx],
                           [dx, 0, dx],
                           [2*dx, 0 ,0],
                           [2*dx, dx, 0],
                           [2*dx, dx, dx],
                           [2*dx, 0, dx],
                           [3*dx, 0 ,0],
                           [3*dx, dx, 0],
                           [3*dx, dx, dx],
                           [3*dx, 0, dx]])
    mesh.elements = np.array([[1,2,3,4,5,6,7,8],
                              [5,6,7,8,9,10,11,12],
                              [9,10,11,12,13,14,15,16]]) - 1
    mesh.bndfaces = np.array([[1,2,3,4],
                              [1,5,8,4],
                              [2,6,7,3],
                              [5,9,12,8],
                              [6,10,11,7],
                              [9,13,16,12],
                              [10,14,15,11],
                              [13,14,15,16]]) -1
    mesh.intfaces = np.array([[5,6,7,8],
                              [9,10,11,12]]) -1
    mesh.intfaces_elem_conn = np.array([[1,2],
                                        [2,3]]) - 1
    mesh.elements_bndf_conn = [[0,1,2],
                               [3,4],
                               [5,6,7]]
    mesh.bndfaces_elem_conn = np.expand_dims(np.array([1,1,1,2,2,3,3,3]),axis = 1)-1
    mesh.bndfaces_tags = np.array([1,2,2,2,2,2,2,3])
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}

    mesh.set_elements_intfaces_connectivity()
    mesh.set_elements_centroids()
    #mesh.set_boundary_faces()
    # set data 
    arr_tmp = np.array([[1,0,0],
                         [1,0,0],
                         [1,0,0]])
    velocity = -0.1
    mesh.elements_data['velocity'] = velocity * arr_tmp
    mesh.elements_data['temp'] = np.zeros((3,1))
    arr_tmp = np.array([[1,0,0],
                         [1,0,0],
                         [1,0,0],
                         [1,0,0],
                         [1,0,0],
                         [1,0,0],
                         [1,0,0],
                         [1,0,0]])
    mesh.bndfaces_data['velocity'] =   velocity * arr_tmp 
    
    return mesh

def test_conv_central_differencing():
    convection_operator = CentralDiffConvection()
    centroid1 = np.array([0,0,-1])
    centroid2 = np.array([0,0,1])
    surface_centroid = np.array([0,0,0])
    surface_area = 0.5
    surface_vector = np.array([0,0,1])
    velocity1 = np.array([0,0,2])
    velocity2 = np.array([0,0,1])
    face_coeff, w1, w2 = convection_operator.calc_surface_coef(centroid1, centroid2, 
                                                               surface_area, surface_vector, 
                                                               surface_centroid,
                                                               velocity1, velocity2)
    print(face_coeff,w1,w2)
    assertion = face_coeff == 0.75 and w1 == 0.5 and w2 == 0.5 
    assert assertion 
    
def test_conv_central_differencing2():
    convection_operator = CentralDiffConvection()
    centroid1 = np.array([0,0,-1])
    centroid2 = np.array([0,0,3])
    surface_centroid = np.array([0,0,0])
    surface_area = 0.5
    surface_vector = np.array([0,0,1])
    velocity1 = np.array([0,0,4])
    velocity2 = np.array([0,0,1])
    face_coeff, w1, w2 = convection_operator.calc_surface_coef(centroid1, centroid2, 
                                                               surface_area, surface_vector, 
                                                               surface_centroid,
                                                               velocity1, velocity2)
    print(face_coeff,w1,w2)
    assertion = face_coeff == 1.625 and w1 == 0.75 and w2 == 0.25 
    assert assertion 
    
def test_conv_upwind():
    convection_operator = UpwindConvection()
    centroid1 = np.array([0,0,-1])
    centroid2 = np.array([0,0,3])
    surface_centroid = np.array([0,0,0])
    surface_area = 0.5
    surface_vector = np.array([0,0,1])
    velocity1 = np.array([0,0,4])
    velocity2 = np.array([0,0,1])
    
    face_coeff, w1, w2 = convection_operator.calc_surface_coef(centroid1, centroid2, 
                                                               surface_area, surface_vector, 
                                                               surface_centroid,
                                                               velocity1, velocity2)
    print(face_coeff,w1,w2)
    
    assertion = face_coeff == 1.625 and w1 == 1 and w2 == 0 
    assert assertion

def test_conv_upwind2():
    convection_operator = UpwindConvection()
    centroid1 = np.array([0,0,-1])
    centroid2 = np.array([0,0,3])
    surface_centroid = np.array([0,0,0])
    surface_area = 0.5
    surface_vector = np.array([0,0,1])
    velocity1 = np.array([0,0,-5])
    velocity2 = np.array([0,0,-1])
    
    face_coeff, w1, w2 = convection_operator.calc_surface_coef(centroid1, centroid2, 
                                                               surface_area, surface_vector, 
                                                               surface_centroid,
                                                               velocity1, velocity2)
    print(face_coeff,w1,w2)
    
    assertion = face_coeff == 2 and w1 == 0 and w2 == 1 
    assert assertion
    
def test_conv_dir_central_diff():
    convection_operator = CentralDiffConvection()
    surface_area = 0.5
    surface_normal = np.array([0,0,-1])
    surface_velocity = np.array([0,0,2])
    centroid = np.array([0,0,1])
    surface_centroid = np.array([0,0,0])
    surf_coeff = convection_operator.calc_dirichlet_surface_coeff(surface_area, surface_normal, surface_velocity,centroid,surface_centroid)
    print('surface coeff : ', surf_coeff)
    assertion = surf_coeff == -1.
    assert assertion
    
def test_conv_neum_central_diff():
    convection_operator = CentralDiffConvection()
    centroid = np.array([0,0,1])
    surface_centroid = np.array([0,0,0])
    surface_area = 0.5 
    surface_normal = np.array([0,0,1])
    surface_velocity = np.array([0,0,2])
    centroid_value = 4
    surface_gradient = np.array([0,0,1])
    face_coeff, face_value = convection_operator.calc_neumann_surface_coeff(centroid,
                                                                            surface_centroid,
                                                                            surface_area,
                                                                            surface_normal,
                                                                            surface_velocity,
                                                                            surface_gradient)
    
    print(face_coeff, face_value)
    assertion = face_coeff == 1. and face_value == 1
    assert assertion 
    
def test_central_diff_operator(mesh_fixture4):
    operator = CentralDiffConvection(velocity_data= 'velocity')
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 3},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 0},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    mat, rhs = operator(mesh_fixture4,boundary_conditions)
    expected_mat = -np.array([[-0.05, -0.05,  0.  ],
                             [0.05,  0.,   -0.05],
                             [0.,    0.05,  0.05]])
    expected_rhs = -np.array([[-0.3],[0.],[0.]])
    #solution = np.dot(np.linalg.pinv(mat),rhs)
    assertion = np.all(np.abs(expected_mat - mat) < EPSILON) and np.all(np.abs(expected_rhs-rhs) < EPSILON)
    assert assertion 
    
