import pytest
import sys as sys 
sys.path.append('.')
from fvm.gradient import * 
from meshe.mesh import TetraMesh
from fvm.diffusion import * 

@pytest.fixture 
def lsgrad_fixture():
    mesh = TetraMesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1],
                           [-1, 0, 0]])
    mesh.elements = np.array([[0, 1, 2, 3],
                              [0,4,1,3]])
    mesh.set_internal_faces()
    mesh.set_elements_intfaces_connectivity()
    def function(x,y,z):
        return 4*x + 0*y + 0*z
    mesh.set_elements_data('T', function)
    grad_computer = LSGradient('T','gradT', mesh)
    return grad_computer

@pytest.fixture 
def mesh_fixture():
    mesh = TetraMesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1],
                           [-0.25, 0, 0]])
    mesh.elements = np.array([[0, 1, 2, 3],
                              [0,4,1,3]])
    mesh.set_internal_faces()
    mesh.set_elements_intfaces_connectivity()
    def function(x,y,z):
        return 4*x + 0*y + 0*z
    mesh.set_elements_data('T', function)
    return mesh 
    
    
def test_ls_gradient(lsgrad_fixture): 
    gradient0 = lsgrad_fixture.calc_element_gradient(0)
    gradient1 = lsgrad_fixture.calc_element_gradient(1)
    expected_grad = np.array([4, 0, 0])
    assertion = np.all(gradient0 == expected_grad) and np.all(gradient1 == expected_grad)
    assert assertion 

def test_ls_gradient2(lsgrad_fixture): 
    lsgrad_fixture.calculate_gradients()
    gradients = lsgrad_fixture.mesh.elements_data['gradT']
    expected_grad = np.array([[4., 0, 0],
                              [4., 0, 0]])
    print(gradients)
    assertion = np.all(gradients == expected_grad)
    assert assertion 
    
def test_orto_diff_face(mesh_fixture):
    diff_op = OrthogonalDiffusion()
    face = mesh_fixture.intfaces[0]
    ind_cent1 = mesh_fixture.intfaces_elem_conn[0][0]
    ind_cent2 = mesh_fixture.intfaces_elem_conn[0][1]
    
    centroid1 = mesh_fixture.elements_centroids[ind_cent1]
    centroid2 = mesh_fixture.elements_centroids[ind_cent2]
    coord_face = mesh_fixture.nodes[face]
    face_area = mesh_fixture._calc_surface_area(coord_face)
    face_normal = mesh_fixture._calc_surface_normal(coord_face)
    
    print('centroid1 : ', centroid1)
    print('centroid2 : ',centroid2)
    print('centroids distance :', np.sqrt(np.sum((centroid1-centroid2)**2)))
    print('face area : ', face_area)
    print('face normal :', face_normal)
    
    # expect diffusion_coeff*face_area*(1/centroids_distance)
    surf_flux = diff_op(centroid1, centroid2, face_area, face_normal, diffusion_coeff=1.)
    print(surf_flux)
    
    assertion = surf_flux == 1.6
    assert assertion 