import pytest 
import sys as sys 
sys.path.append('.')
from meshe.mesh import * 

@pytest.fixture 
def mesh_fixture():
    mesh = Mesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])
    mesh.elements = np.array([[0, 1, 2, 3]])
    return mesh 

@pytest.fixture 
def mesh_fixture2():
    mesh = TetraMesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1],
                           [-1, 0, 0]])
    mesh.elements = np.array([[0, 1, 2, 3],
                              [0,4,1,3]])
    return mesh 

@pytest.fixture 
def mesh_fixture3():
    mesh = TetraMesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1],
                           [-1, 0, 0]])
    mesh.elements = np.array([[0, 1, 2, 3],
                              [0,4,1,3]])
    mesh.bndfaces = np.array([[0, 4, 1]])
    #mesh.bndfaces_tags = np.array
    return mesh 

@pytest.fixture 
def mesh_fixture4():
    mesh = TetraMesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1],
                           [-0.5, 0, 0]])
    mesh.elements = np.array([[0, 1, 2, 3],
                              [0,4,1,3]])
    return mesh 

def test_centroid_calculation(mesh_fixture):
    element = mesh_fixture.nodes[mesh_fixture.elements[0]]
    centroid = mesh_fixture._calc_centroid(element)
    expected = np.array([0.25, 0.25, 0.25])
    print(element)
    print(np.shape(element))
    print(centroid)
    print(np.shape(centroid))
    print(expected)
    print(np.shape(expected))
    assertion = np.all(centroid == expected)
    assert assertion

def test_calc_surface(mesh_fixture):
    surface_element = np.array([[0, 0, 0],
                                [0, 1.5, 0],
                                [1.5, 0, 0]])
    surf = mesh_fixture._calc_surface_area(surface_element)
    expected = 0.5*(1.5*1.5)
    print(surf)
    print(expected)
    assertion = surf == expected
    assert assertion 

def test_calc_surface2(mesh_fixture):
    surface_element = np.array([[0, 0, 0],
                                [0, 1.5, 0],
                                [1.5, 1.5, 0],
                                [1.5, 0, 0]])
    surf = mesh_fixture._calc_surface_area(surface_element)
    expected = (1.5*1.5)
    print(surf)
    print(expected)
    assertion = surf == expected
    assert assertion 

def test_calc_surface_normal(mesh_fixture) :
    surface_element = np.array([[0, 0, 0],
                                [0, 1.5, 0],
                                [1.5, 0, 0]])
    normal = mesh_fixture._calc_surface_normal(surface_element)
    expected = np.array([0, 0, 1])
    print(normal)
    print(expected)
    assertion = np.all(normal == expected)
    assert assertion  

def test_calc_surface_normal2(mesh_fixture) :
    surface_element = np.array([[0, 0, 0],
                                [0, 1.5, 0],
                                [1.5, 1.5, 0],
                                [1.5, 0, 0]])
    normal = mesh_fixture._calc_surface_normal(surface_element)
    expected = np.array([0, 0, 1])
    print(normal)
    print(expected)
    assertion = np.all(normal == expected)
    assert assertion  

def test_calc_surface_normal3(mesh_fixture) :
    surface_element = np.array([[0, 1.5, 0],
                                [0, 0, 0],
                                [1.5, 0, 0]])
    normal = mesh_fixture._calc_surface_normal(surface_element)
    expected = np.array([0, 0, -1])
    print(normal)
    print(expected)
    assertion = np.all(normal == expected)
    assert assertion 

def test_calc_surf_volflux(mesh_fixture) : 
    surface_element = np.array([[-1.5,-1.5,1],
                                [-1.5,1.5,1],
                                [1.5,1.5,1],
                                [1.5,-1.5,1]])
    centroid = np.array([0,0,0])
    volflux = mesh_fixture._calc_surface_volflux(surface_element,centroid)
    print(volflux)
    expected = 3
    assertion = volflux == expected
    assert assertion   
    
def test_calc_element_volume(mesh_fixture) : 
    element_faces = [] 
    element_faces.append(mesh_fixture.nodes[[0,1,2]])        
    element_faces.append(mesh_fixture.nodes[[0,2,3]])    
    element_faces.append(mesh_fixture.nodes[[0,1,3]])             
    element_faces.append(mesh_fixture.nodes[[1,2,3]])  
    centroid = mesh_fixture._calc_centroid(mesh_fixture.nodes[mesh_fixture.elements[0]])
    volume = mesh_fixture._calc_element_volume(element_faces,centroid)
    expected = 1/6
    print(element_faces)
    print(volume)
    assertion = volume == expected
    assert assertion 

def test_tetra_faces(mesh_fixture2):
    faces, connectivity = mesh_fixture2._get_elements_faces()
    expected_faces = np.array([[0, 1, 2],
                               [0, 2, 3],
                               [0, 3, 1],
                               [1, 2, 3],
                               [0, 4, 1],
                               [0, 3, 4],
                               [4, 1, 3]])
    expected_connectivity = [[0], [0], [0, 1], [0], [1], [1], [1]]
    expected_el_face_conn = np.array([[0,1,2,3],
                                     [4,2,5,6]])
    assertion = np.all(faces == expected_faces) and len(faces) == 7
    assertion = assertion and (connectivity == expected_connectivity)
    #assertion = False
    assert assertion

def test_set_internal_faces(mesh_fixture2): 
    mesh_fixture2.set_internal_faces()
    expected_int_faces = np.array([[0,3,1]])
    expected_int_connectivity = np.array([[0,1]])
    expected_el_face_connectivity = np.array([[-1, -1, 2, -1],
                                              [-1, 2, -1, -1]])
    assertion = np.all(mesh_fixture2.intfaces == expected_int_faces)
    assertion = assertion and np.all(mesh_fixture2.intfaces_elem_conn == expected_int_connectivity)
    assert assertion 

def test_set_centroids(mesh_fixture2):
    mesh_fixture2.set_elements_centroids()
    expected_centroids = np.array([[0.25, 0.25, 0.25],
                                   [-0.25, 0.25, 0.25]])
    assertion = np.all(mesh_fixture2.elements_centroids == expected_centroids)
    assert assertion 

def test_set_elem_data(mesh_fixture2):
    def function(x,y,z):
        return 4*x + 0*y + 0*z
    mesh_fixture2.set_elements_data('temp',function)
    print(mesh_fixture2.elements_data)
    expected_data_array = np.array([1,-1])
    assertion = np.all(mesh_fixture2.elements_data['temp'] == expected_data_array)
    assert assertion

def test_set_elem_intf_conn(mesh_fixture2) : 
    mesh_fixture2.set_internal_faces()
    mesh_fixture2.set_elements_intfaces_connectivity()
    print(mesh_fixture2.elements_intf_conn)
    print(mesh_fixture2.intfaces)
    expected = [[0],[0]]
    assertion = mesh_fixture2.elements_intf_conn == expected
    assert assertion 
    
def test_bnd_element_conn(mesh_fixture3):
    mesh_fixture3.set_internal_faces()
    mesh_fixture3.set_elements_intfaces_connectivity()
    mesh_fixture3.set_boundary_faces()
    print(mesh_fixture3.elements_bndf_conn)
    print(mesh_fixture3.bndfaces_elem_conn)
    print(len(mesh_fixture3.elements_bndf_conn))
    print(np.shape(mesh_fixture3.bndfaces_elem_conn))
    expected_el_bndf_conn = [[],[0]]
    expected_bndf_el_conn = np.array([[1]])
    assertion = expected_el_bndf_conn == mesh_fixture3.elements_bndf_conn
    assertion = assertion and np.all(expected_bndf_el_conn == mesh_fixture3.bndfaces_elem_conn) 
    assert assertion 

def test_surface_vertex_distance(mesh_fixture4): 
    mesh_fixture4.set_elements_centroids()
    mesh_fixture4.set_internal_faces()
    face = mesh_fixture4.nodes[mesh_fixture4.intfaces[0]]
    print(face)
    vertex = mesh_fixture4.elements_centroids[0]
    distance1 = mesh_fixture4._calc_vertex_face_distance(vertex, face)
    print(vertex)
    print(distance1)
    vertex = mesh_fixture4.elements_centroids[1]
    distance2 = mesh_fixture4._calc_vertex_face_distance(vertex, face)
    print(vertex)
    print(distance2)
    assertion = (distance1 == 0.25) and (distance2 == 0.125)
    assert assertion  
    
def test_face_pairnod_angle(mesh_fixture):
    surface_element = np.array([[0, 0, 0],
                                [0, 1., 0],
                                [1., 0, 0]])
    node1 = np.array([0.25, 0.25, -1])
    node2 = np.array([0.25, 0.25, 1])
    theta = mesh_fixture._calc_face_pairnode_theta(surface_element,node1, node2)
    print(np.degrees(theta))
    assertion = False 
    assert assertion                   