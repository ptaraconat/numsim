import pytest
import sys as sys 
sys.path.append('.')
from fvm.divergence import DivergenceComputer
from meshe.mesh import *

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
    return mesh 

@pytest.fixture()
def divop_fixture():
    operator = DivergenceComputer('velocity','divergence')
    return operator

def test_surface_flowrate(divop_fixture):
    centroid1 = np.array([0,0,-1])
    centroid2 = np.array([0,0,1])
    surface_centroid = np.array([0,0,0])
    surface_area = 0.5
    surface_normal = np.array([0,0,1])
    data1 = np.array([0,0,5])
    data2 = np.array([0,0,1])
    flow_rate = divop_fixture.calc_surface_flowrate(centroid1,
                                                    centroid2,
                                                    surface_area,
                                                    surface_normal,
                                                    surface_centroid,
                                                    data1,
                                                    data2)
    print(flow_rate)
    assertion = flow_rate == 1.5
    assert assertion

def test_surface_flowrate2(divop_fixture):
    centroid1 = np.array([0,0,-1])
    centroid2 = np.array([0,0,3])
    surface_centroid = np.array([0,0,0])
    surface_area = 0.5
    surface_normal = np.array([0,0,1])
    data1 = np.array([0,0,1])
    data2 = np.array([0,0,8])
    flow_rate = divop_fixture.calc_surface_flowrate(centroid1,
                                                    centroid2,
                                                    surface_area,
                                                    surface_normal,
                                                    surface_centroid,
                                                    data1,
                                                    data2)
    print(flow_rate)
    assertion = flow_rate == 1.375
    assert assertion

def test_surface_dataoutflux(divop_fixture):
    centroid1 = np.array([0,0,-1])
    centroid2 = np.array([0,0,1])
    surface_centroid = np.array([0,0,0])
    surface_area = 0.5
    surface_normal = np.array([0,0,1])
    data1 = 5
    data2 = 1
    flux = divop_fixture.calc_surface_flowrate(centroid1,
                                                    centroid2,
                                                    surface_area,
                                                    surface_normal,
                                                    surface_centroid,
                                                    data1,
                                                    data2)
    print(flux)
    assertion = np.all(flux == np.array([0, 0, 1.5])) 
    assert assertion 
    
def test_calc_bndface_flux(divop_fixture):
    el_centroid = np.array([0, 0, -0.5])
    face_centroid = np.array([0, 0, 0])
    face_normal = np.array([0, 0, 1])
    face_area = 0.5
    face_data = np.array([1,2,3])
    flux = divop_fixture.calc_bndface_flowrate(el_centroid, 
                                               face_data,
                                               face_centroid, 
                                               face_area, 
                                               face_normal)
    print(flux)
    assertion = flux == 1.5
    assert assertion 
    
def test_calc_bndface_flux2(divop_fixture):
    el_centroid = np.array([0, 0, 0.5])
    face_centroid = np.array([0, 0, 0])
    face_normal = np.array([0, 0, -1])
    face_area = 0.5
    face_data = np.array([1,2,3])
    flux = divop_fixture.calc_bndface_flowrate(el_centroid, 
                                               face_data,
                                               face_centroid, 
                                               face_area, 
                                               face_normal)
    print(flux)
    assertion = flux == - 1.5
    assert assertion 
    
def test_divop(divop_fixture, mesh_fixture):
    divop_fixture(mesh_fixture)
    print(mesh_fixture.elements_data['divergence'])
    divergence = mesh_fixture.elements_data['divergence']
    expected = np.array([[-9], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    assertion = np.all(expected == divergence)
    assert assertion 