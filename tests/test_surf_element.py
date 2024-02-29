import pytest 
import sys as sys 
sys.path.append('../')
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
    faces, connectivity, el_face_conn = mesh_fixture2._get_elements_faces()
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
    assertion = assertion and np.all(el_face_conn == expected_el_face_conn)
    #assertion = False
    assert assertion

def test_set_internal_faces(mesh_fixture2): 
    mesh_fixture2.set_internal_faces()
    expected_int_faces = np.array([[0,3,1]])
    expected_int_connectivity = np.array([[0,1]])
    expected_el_face_connectivity = np.array([[-1, -1, 2, -1],
                                              [-1, 2, -1, -1]])
    assertion = np.all(mesh_fixture2.internal_faces == expected_int_faces)
    assertion = assertion and np.all(mesh_fixture2.internal_connectivity == expected_int_connectivity)
    assertion = assertion and np.all(mesh_fixture2.elements_face_connectivity == expected_el_face_connectivity)
    assert assertion 
                        