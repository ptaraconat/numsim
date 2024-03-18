import pytest
import sys as sys 
sys.path.append('.')
from meshe.mesh import TetraMesh, Mesh
from fvm.diffusion import * 
from fvm.gradient import LSGradient

EPSILON = 1e-10

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
    mesh.bndfaces = np.array([[0, 1, 2],
                              [0, 2, 3],
                              [1, 2, 3]])
    
    mesh.set_internal_faces()
    mesh.set_elements_intfaces_connectivity()
    def function(x,y,z):
        return 4*x + 0*y + 0*z
    mesh.set_elements_data('T', function)
    return mesh 

@pytest.fixture 
def mesh_fixture2():
    mesh = Mesh()
    mesh.nodes = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1],
                           [1, 0, 1]])
    mesh.elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mesh.bndfaces = np.array([[0, 1, 2, 3]])
    mesh.elements_intf_conn = []
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
    def function(x,y,z):
        return 4*x + 0*y + 0*z
    mesh.set_elements_data('t', function)
    #def function(x,y,z):
    #    return 4, 0, 0
    #mesh.set_elements_data('gradt', function)
    grad_tmp = np.array([[4,0,0],
                         [4,0,0],
                         [4,0,0]])
    mesh.elements_data['gradt'] = grad_tmp
    
    return mesh
  
@pytest.fixture 
def mesh_fixture5():
    mesh = Mesh()
    dx = 1. 
    delta = dx*(50/100)
    delta_p = dx * (50/100)
    delta2 = dx*(0/100)
    mesh.nodes = np.array([[0, 0 ,0],
                           [0, dx, 0],
                           [0, dx, dx],
                           [0, 0, dx],
                           [dx+delta, 0+delta2 ,0],
                           [dx+delta, dx+delta2, 0],
                           [dx-delta, dx+delta2, dx],
                           [dx-delta, 0+delta2, dx],
                           [2*dx+delta_p, 0-delta2 ,0],
                           [2*dx+delta_p, dx-delta2, 0],
                           [2*dx-delta_p, dx-delta2, dx],
                           [2*dx-delta_p, 0-delta2, dx],
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
    # Init data 
    def function(x,y,z):
        return 0*x + 0*y + 0*z
    mesh.set_elements_data('temp', function)
    mesh.set_elements_data('grad_temp', function)

    return mesh
 
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
    surf_flux = diff_op.calc_surface_coef(centroid1, centroid2, face_area, face_normal, diffusion_coeff=1.)
    print(surf_flux)
    
    assertion = surf_flux == 1.6
    assert assertion 

def test_ortho_diff_drbnd(mesh_fixture2):
    diff_op = OrthogonalDiffusion()
    mesh_fixture2.set_boundary_faces()
    mesh_fixture2.set_elements_centroids()
    #
    face_ind = 0 
    bnd_face = mesh_fixture2.bndfaces[face_ind]
    elem_ind = int(mesh_fixture2.bndfaces_elem_conn[face_ind][0])
    centroid = mesh_fixture2.elements_centroids[elem_ind]
    face_nodes = mesh_fixture2.nodes[bnd_face]
    face_centroid = mesh_fixture2._calc_centroid(face_nodes)
    face_normal = mesh_fixture2._calc_surface_normal(face_nodes)
    face_area = mesh_fixture2._calc_surface_area(face_nodes)
    #
    print(face_nodes)
    print('centroid : ', centroid)
    print('face centroid : ', face_centroid)
    print('face area :', face_area)
    print('face normal : ', face_normal)
    #
    surf_flux = diff_op.calc_dirchlet_bnd_surface_coef(centroid, 
                                                       face_centroid, 
                                                       face_area, 
                                                       face_normal, 
                                                       diffusion_coeff = 1)
    print('surface flux : ', surf_flux)
    
    assertion = surf_flux == 2.
    assert assertion 
    
 

def test_ortho_diff_neumbnd(mesh_fixture2):
    diff_op = OrthogonalDiffusion()
    mesh_fixture2.set_boundary_faces()
    mesh_fixture2.set_elements_centroids()
    bnd_flux = np.array([0,0,2])
    #
    face_ind = 0 
    bnd_face = mesh_fixture2.bndfaces[face_ind]
    face_nodes = mesh_fixture2.nodes[bnd_face]
    face_normal = mesh_fixture2._calc_surface_normal(face_nodes)
    face_area = mesh_fixture2._calc_surface_area(face_nodes)
    #
    print(face_nodes)
    print('face area :', face_area)
    print('face normal : ', face_normal)
    print('bnd flux : ', bnd_flux)
    
    surf_flux = diff_op.calc_neumann_bnd_surface_coef(face_area, face_normal, bnd_flux)
    print('surface flux : ', surf_flux)
    assertion = surf_flux == 2.
    assert assertion

def test_nonortho_diff_intface():
    face_gradient = np.array([4,0,1])
    face_area = 2.
    face_vector = np.array([3.,4,0])
    diffusion = 1. 
    non_orthodiff = NonOthogonalDiffusion()
    face_coeff = non_orthodiff.cal_surface_coef(face_gradient,
                                                face_area,
                                                face_vector,
                                                diffusion_coeff=diffusion)
    print(face_coeff)
    assertion = face_coeff == 24
    assert assertion

def test_over_relaxed_decomp():
    diff_op = NonOthogonalDiffusion(method = 'over_relaxed')
    face = np.array([[0, 0, 0.],
                     [0, 1., 0.],
                     [1., 1., 0.],
                     [1., 0, 0.]])
    node1 = np.array([0.25, 0.25, -0.25])
    node2 = np.array([0.25, 0.25, 0.5])
    ortho, nonortho = diff_op._decompose_normal(face,node2, node1)
    print(ortho)
    print(nonortho)
    assertion = np.all(ortho == np.array([0,0,-1])) and np.all(nonortho == np.array([0,0,0]))
    assert assertion 

def test_ortho_diff_operator(mesh_fixture4):
    diff_op = OrthogonalDiffusion()
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 0},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 3},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    
    print(mesh_fixture4.bndfaces_tags)
    mat, rhs = diff_op(mesh_fixture4, 
                       boundary_conditions, 
                       diffusion_coeff=1.)
    print(mat)
    print(rhs)
    solution = np.linalg.solve(mat,rhs)
    print(solution)
    
    expected_mat = np.array([[-3, 1, 0],
                              [1, -2, 1],
                              [0, 1, -3]])
    expected_rhs = np.array([[0],[0],[-6]])
    assertion = np.all(mat == expected_mat) and np.all(rhs == expected_rhs)
    
    assert assertion 

def test_nonortho_diff_operator(mesh_fixture4):
    diff_op = NonOthogonalDiffusion(data_name = 't', 
                                    grad_data_name = 'gradt',
                                    method = 'over_relaxed')
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 0},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 3},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    
    mat, rhs = diff_op(mesh_fixture4, 
                       boundary_conditions, 
                       diffusion_coeff=1.)
    print(mat)
    print(rhs)
    solution = np.linalg.solve(mat,rhs)
    print(solution)
    
    expected_mat = np.array([[-3, 1, 0],
                              [1, -2, 1],
                              [0, 1, -3]])
    expected_rhs = np.array([[0],[0],[-6]])
    assertion = np.all(mat == expected_mat) and np.all(rhs == expected_rhs)
    
    assert assertion 
   
def test_ortho_diff_operator_on_nonorthomesh(mesh_fixture5):
    diff_op = OrthogonalDiffusion()
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 0},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 3},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    
    print(mesh_fixture5.bndfaces_tags)
    mat, rhs = diff_op(mesh_fixture5, 
                       boundary_conditions, 
                       diffusion_coeff=1.)
    print(mat)
    print(rhs)
    solution = np.linalg.solve(mat,rhs)
    print(solution)
    
    expected_mat = np.array([[-3, 1, 0],
                              [1, -2, 1],
                              [0, 1, -3]])
    expected_rhs = np.array([[0],[0],[-6]])
    assertion = np.all(mat == expected_mat) and np.all(rhs == expected_rhs)
    
    assert assertion 

def test_nonortho_diff_operator2(mesh_fixture5):
    gradient_computer = LSGradient('temp', 'grad_temp', 
                                   mesh_fixture5, 
                                   weighting = False,
                                   use_boundaries = True)
    diff_op = NonOthogonalDiffusion(data_name = 'temp', 
                                    grad_data_name = 'grad_temp',
                                    method = 'over_relaxed')
    boundary_conditions = {'inlet' : {'type' : 'dirichlet',
                                      'value' : 0},
                           'outlet' : {'type' : 'dirichlet',
                                       'value' : 3},
                           'wall' : {'type' : 'neumann',
                                     'value' : np.array([0,0,0])}}
    
    for i in range(30): 
        mat, rhs = diff_op(mesh_fixture5, 
                        boundary_conditions, 
                        diffusion_coeff=1.)
        print('########### ', i)
        print(mat)
        print(rhs)
        solution = np.linalg.solve(mat,rhs)
        mesh_fixture5.elements_data['temp'] = solution
        mesh_fixture5.set_bndfaces_data_from_bc('temp', boundary_conditions)
        gradient_computer.calculate_gradients()
        print(mesh_fixture5.elements_data['grad_temp'])
        print(mesh_fixture5.elements_data['temp'])
    
    expected_solution = np.expand_dims(np.array([0.5,1.5,2.5]),axis = 1)
    
    diff = np.abs(expected_solution-mesh_fixture5.elements_data['temp'])
    print(diff)
    assertion = np.all( diff < EPSILON)
    assert assertion