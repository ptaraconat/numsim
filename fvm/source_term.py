import numpy as np 
from meshe.mesh import Mesh

class SourceTerm():
    
    def __init__(self,data_name = ''):
        '''
        '''
        self.data_name = data_name
    
    def calc_element_coeff(self,element_faces,element_centroid,source_value):
        '''
        element_faces ::: list of np.array(n_nodes,3) ::: list of 
        arrays of faces coordinates dounding a given element
        element_centroid ::: np.array(3,) ::: centroid of the element
        source_value ::: float ::: source term at element centroid 
        returns 
        element_coeff ::: float ::: 
        '''
        element_volume = Mesh()._calc_element_volume(element_faces,element_centroid)
        return element_volume*source_value 
    
    def __call__(self, mesh) : 
        '''
        arguments 
        mesh ::: numsim.meshe.mesh.Mesh ::: mesh on which the diffusion operator is calculated
        returns 
        rhs_vec ::: np.array(n_elem,) :::
        '''
        n_elem = np.size(mesh.elements,0)
        rhs_vec = np.zeros((n_elem,1))
        for ind_elem, element in enumerate(mesh.elements):
            element_faces = mesh._get_element_bounding_faces(ind_elem)
            element_centroid = mesh._calc_centroid(mesh.nodes[element])
            source_value = mesh.elements_data[self.data_name][ind_elem]
            el_coeff = self.calc_element_coeff(element_faces,
                                               element_centroid,
                                               source_value)
            rhs_vec[ind_elem] += el_coeff  
        return rhs_vec