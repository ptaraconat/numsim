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