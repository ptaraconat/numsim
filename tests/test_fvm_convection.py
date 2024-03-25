import pytest
import sys as sys 
sys.path.append('.')
from fvm.convection import * 
from fvm.diffusion import * 
from meshe.mesh import Mesh, TetraMesh

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
    mesh = Mesh()
    dx = 1. 
    n_elem = 100
    # Init first element 
    nodes_list = [np.array([0, 0 ,0]),
                  np.array([0, dx, 0]),
                  np.array([0, dx, dx]),
                  np.array([0, 0, dx]),
                  np.array([dx, 0 ,0]),
                  np.array([dx, dx, 0]),
                  np.array([dx, dx, dx]),
                  np.array([dx, 0, dx])]
    elements_list = [np.array([0,1,2,3,4,5,6,7])]
    intfaces_list = []
    bndfaces_list = [np.array([0,1,2,3]),
                     np.array([0,4,7,3]),
                     np.array([1,5,6,2]),
                     np.array([3,7,6,2]),
                     np.array([0,4,5,1])]
    bndfaces_tags_list = [1,2,2,2,2]    
    elements_bndf_conn_list = [[0,1,2,3,4]]
    intfaces_elem_conn_list = []
    bndfaces_elem_conn_list = [0,0,0,0,0]
    for i in range(n_elem-1): 
        print(i)
        print('add element', i+2)
        # add dx (along x axis) to previous 4 nodes 
        delta = np.array([dx,0,0])
        last_nodes = nodes_list[-4:]
        add_nodes = [node + delta for node in last_nodes]
        nodes_list += add_nodes
        # add new element 
        last_el = elements_list[-1]
        add_el = [last_el+4]
        elements_list +=add_el
        # add new internal face 
        add_intf = last_el[-4:]
        intfaces_list += [add_intf]
        add_intf_el_conn = [[len(elements_list)-2,len(elements_list)-1]]
        intfaces_elem_conn_list += add_intf_el_conn
        # add new bndfaces 
        last_bndfaces = bndfaces_list[-4:]
        add_bndfaces = [bndface + 4 for bndface in last_bndfaces]
        bndfaces_list +=add_bndfaces
        bndfaces_tags_list += [2,2,2,2]
        elem_id = len(elements_list)-1
        add_bndf_el_conn = [elem_id,elem_id,elem_id,elem_id]
        bndfaces_elem_conn_list += add_bndf_el_conn
        # elem_bndf_conn 
        last_elem_bndf_conn = elements_bndf_conn_list[-1][-4:]
        print('last elem bnd conn : ',last_elem_bndf_conn)
        add_elem_bndf_conn = np.asarray(last_elem_bndf_conn)+4
        elements_bndf_conn_list += [add_elem_bndf_conn.tolist()]
    # Add outlet to bndfaces_list
    outlet_bndface = [elements_list[-1][-4:]]
    bndfaces_list += outlet_bndface
    bndfaces_tags_list += [3]
    bndfaces_elem_conn_list += [elem_id]
    #
    mesh.nodes = np.asarray(nodes_list)
    mesh.elements = np.asarray(elements_list)
    mesh.bndfaces = np.asarray(bndfaces_list)
    mesh.intfaces = np.asarray(intfaces_list)
    mesh.intfaces_elem_conn = np.asarray(intfaces_elem_conn_list)
    mesh.elements_bndf_conn = elements_bndf_conn_list
    mesh.bndfaces_elem_conn = np.expand_dims(np.asarray(bndfaces_elem_conn_list),axis = 1)
    mesh.bndfaces_tags = np.asarray(bndfaces_tags_list)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    #
    mesh.set_elements_intfaces_connectivity()
    mesh.set_elements_centroids()
    #
    # set data 
    arr_tmp = np.zeros((n_elem,3))
    arr_tmp[:,0] = 1. 
    velocity = -0.1
    mesh.elements_data['velocity'] = velocity * arr_tmp
    print(mesh.elements_data['velocity'])
    mesh.elements_data['temp'] = np.zeros((n_elem,1))
    n_bndf = np.size(mesh.bndfaces,0)
    arr_tmp = np.zeros((n_bndf,3))
    arr_tmp[:,0] = 1. 
    mesh.bndfaces_data['velocity'] =   velocity * arr_tmp 
    
    print(np.asarray(nodes_list))
    print(np.asarray(elements_list))
    print(np.asarray(intfaces_list))
    print(elements_bndf_conn_list)
    print(np.asarray(intfaces_elem_conn_list))
    
    print(np.asarray(bndfaces_list))
    print(np.asarray(bndfaces_tags_list))
    print(np.asarray(bndfaces_elem_conn_list))
    print(mesh.bndfaces_data['velocity'] )
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
    operator = CentralDiffConvection(velocity_data= 'velocity',convected_data = 'temp')
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 3},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 0},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    print(mesh_fixture4.bndfaces)
    mat, rhs = operator(mesh_fixture4,boundary_conditions)
    print(mat)
    print(rhs)
    solution = np.dot(np.linalg.pinv(mat),rhs)
    print(solution)
    assertion = True
    assert assertion 
    
def test_central_diff_operator__(mesh_fixture4):
    operator = CentralDiffConvection(velocity_data= 'velocity',convected_data = 'temp')
    diff_op = OrthogonalDiffusion()
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 3},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 0},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    mat, rhs = operator(mesh_fixture4,boundary_conditions)
    mat_, rhs_ = diff_op(mesh_fixture4, 
                       boundary_conditions, 
                       diffusion_coeff=1.)
    print(mat)
    print(mat_)
    mat += mat_
    rhs += rhs_
    print(mat)
    print(rhs)
    solution = np.dot(np.linalg.pinv(mat),rhs)
    print(solution)
    assertion = True 
    assert assertion 
    
def test_1d_conv_diff(mesh_fixture1d):
    operator = CentralDiffConvection(velocity_data= 'velocity',convected_data = 'temp')
    diff_op = OrthogonalDiffusion()
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 3},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 0},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    mat, rhs = operator(mesh_fixture1d,boundary_conditions)
    mat_, rhs_ = diff_op(mesh_fixture1d, 
                       boundary_conditions, 
                       diffusion_coeff=1.)
    print(mat)
    print(mat_)
    mat += mat_
    rhs += rhs_
    print(mat)
    print(rhs)
    solution = np.dot(np.linalg.pinv(mat),rhs)
    print(solution)
    
    assertion = False 
    assert assertion 