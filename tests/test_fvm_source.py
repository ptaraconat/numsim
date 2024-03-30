import pytest
import sys as sys 
sys.path.append('.')
from fvm.source_term import * 
from meshe.mesh import *

@pytest.fixture
def mesh_fixture():
    mesh = Mesh1D(3,3)
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