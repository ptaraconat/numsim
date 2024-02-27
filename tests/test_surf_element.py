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
    surf = mesh_fixture._calc_surface(surface_element)
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
    surf = mesh_fixture._calc_surface(surface_element)
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
                        