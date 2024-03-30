import pytest
import sys as sys 
sys.path.append('.')
from fvm.source_term import * 
from meshe.mesh import *

@pytest.fixture
def mesh_fixture():
    mesh = Mesh1D(3,3)
    return mesh

@pytest.fixture
def mesh_fixture2():
    n_elem = 5
    dx = 3
    mesh = Mesh1D(dx,n_elem)
    # set data 
    arr_tmp = np.zeros((n_elem,))
    arr_tmp[2] = 3.5
    mesh.elements_data['source'] = arr_tmp
    return mesh

def test_source_term(mesh_fixture):
    el_ind = 0
    #
    element_surfaces = mesh_fixture._get_element_bounding_faces(el_ind)
    #
    source_value = 3.5
    element = mesh_fixture.nodes[mesh_fixture.elements[el_ind]]
    element_centroid = mesh_fixture._calc_centroid(element)
    source_operator = SourceTerm()
    source_coeff = source_operator.calc_element_coeff(element_surfaces,
                                                      element_centroid,
                                                      source_value)
    print(source_coeff)
    assertion = source_coeff == 94.5
    assert assertion 
    
def test_source_operator(mesh_fixture2): 
    source_operator = SourceTerm(data_name = 'source')
    rhs_vec = source_operator(mesh_fixture2)
    expected_rhs = np.array([[0],[0],[-94.5],[0],[0]])
    assertion = np.all(rhs_vec == expected_rhs)
    assert assertion 