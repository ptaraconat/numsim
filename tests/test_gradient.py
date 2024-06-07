import pytest
import sys as sys 
sys.path.append('.')
from fvm.gradient import CellBasedGradient
from meshe.mesh import *


EPSILON = 1e-10

@pytest.fixture()
def mesh_fixture():
    dx = 1
    n_elem = 10
    mesh = Mesh1D(dx,n_elem)
    mesh.physical_entities = {'inlet': np.array([1,   2]), 
                              'outlet': np.array([3,   2]), 
                              'wall': np.array([2,   2])}
    mesh.set_elements_volumes()
    #
    velocity = 1. 
    arr_tmp = np.zeros((n_elem,3))
    arr_tmp[:,0] = velocity * 1. 
    mesh.elements_data['velocity'] =  arr_tmp
    n_bndf = np.size(mesh.bndfaces,0)
    arr_tmp = np.zeros((n_bndf,3))
    arr_tmp[:,0] = velocity * 1. 
    arr_tmp[0,0] = 10
    mesh.bndfaces_data['velocity'] =    arr_tmp 
    #
    def function(x,y,z):
        return 4*x + 0*y + 0*z
    mesh.set_elements_data('data', function)
    mesh.set_bndfaces_data('data', function)
    return mesh 

@pytest.fixture()
def gradop_fixture():
    operator = CellBasedGradient('data','grad_data')
    return operator

def test_surface_component(gradop_fixture):
    centroid1 = np.array([0,0,-1])
    centroid2 = np.array([0,0,1])
    surface_centroid = np.array([0,0,0])
    surface_area = 1.
    surface_normal = np.array([0,0,1])
    data1 = np.array([0,1,5])
    data2 = np.array([0,9,1])
    face_component = gradop_fixture.calc_surface_component(centroid1,
                                                           centroid2,
                                                           surface_area,
                                                           surface_normal,
                                                           surface_centroid,
                                                           data1,
                                                           data2)
    print(face_component)
    expected = np.array([[0., 0., 0.],
                         [0., 0., 5.],
                         [0., 0., 3.]])
    assertion = np.all(face_component == expected)
    assert assertion
    
def test_surface_component2(gradop_fixture):
    centroid1 = np.array([0,0,-1])
    centroid2 = np.array([0,0,1])
    surface_centroid = np.array([0,0,0])
    surface_area = 0.5
    surface_normal = np.array([0,0,1])
    data1 = 5
    data2 = 1
    face_component = gradop_fixture.calc_surface_component(centroid1,
                                                           centroid2,
                                                           surface_area,
                                                           surface_normal,
                                                           surface_centroid,
                                                           data1,
                                                           data2)
    print(face_component)
    assertion = np.all(face_component == np.array([0, 0, 1.5]))
    assert assertion

def test_calc_bndface_flux(gradop_fixture):
    el_centroid = np.array([0, 0, -0.5])
    face_centroid = np.array([0, 0, 0])
    face_normal = np.array([0, 0, 1])
    face_area = 0.5
    face_data = np.array([0,5,3])
    face_component = gradop_fixture.calc_bndface_component(el_centroid, 
                                                           face_data,
                                                           face_centroid, 
                                                           face_area, 
                                                           face_normal)
    print(face_component)
    expected = np.array([[0., 0., 0.],
                         [0., 0., 2.5],
                         [0., 0., 1.5]])
    assertion = np.all(face_component == expected)
    assert assertion 

def test_gradop(gradop_fixture, mesh_fixture):
    gradop_fixture(mesh_fixture)
    gradient = mesh_fixture.elements_data['grad_data']
    n_elem = np.size(mesh_fixture.elements,0)
    expected = np.zeros((n_elem,3))
    expected[:,0] = 4.
    print(np.abs(gradient - expected))
    assertion = np.all(np.abs(gradient - expected) < EPSILON)
    assert assertion 