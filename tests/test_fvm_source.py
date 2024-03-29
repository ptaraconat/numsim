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
    print(mesh_fixture.elements)
    print(mesh_fixture.elements_bndf_conn)
    print(mesh_fixture.elements_intf_conn)
    el_ind = 0
    intf_ind = mesh_fixture.elements_intf_conn[el_ind]
    intfaces = mesh_fixture.intfaces[intf_ind]
    bndf_ind = mesh_fixture.elements_bndf_conn[el_ind]
    bndfaces = mesh_fixture.bndfaces[bndf_ind]
    element_faces_ind = np.concatenate((intfaces , bndfaces),axis = 0)
    print(bndfaces)
    print(intfaces)
    print(element_faces_ind)
    element_surfaces = [ ]
    for i in range(np.size(element_faces_ind,0)) : 
        ind = element_faces_ind[i,:]
        print(ind)
        element_surfaces.append(mesh_fixture.nodes[ind])
    
    source_value = 3.5
    source_operator = SourceTerm()
    source_coeff = source_operator.calc_element_coeff(element_surfaces,source_value)
    print(source_coeff)
    assertion = False 
    assert assertion 