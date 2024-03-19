import pytest
import sys as sys 
sys.path.append('.')
from fvm.convection import * 
from meshe.mesh import TetraMesh

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
                                                                            centroid_value,
                                                                            surface_gradient)
    
    print(face_coeff, face_value)
    assertion = face_coeff == 1. and face_value == 3
    assert assertion 