import numpy as np 
from meshe.mesh import Mesh

class SourceTerm():
    
    def __init__(self,data_name = ''):
        '''
        '''
        self.data_name = data_name
    
    def calc_element_coeff(self,element,source_value):
        '''
        element ::: np.array(n_nodes,3) ::: coordinates of element nodes 
        source_value ::: float ::: source term at element centroid 
        returns 
        element_coeff ::: float ::: 
        '''
        centroid = Mesh()._calc_centroid(element)
        element_volume = Mesh()._calc_element_volume(element,centroid)
        return element_volume*source_value 