import numpy as np
from meshe.mesh import * 

class ElementsGradientComputer : 

    def __init__(self,dataname , mesh):
        '''
        arguments 
        dataname ::: string ::: name of the gradient data 
        mesh ::: numsim.meshe.mesh.Mesh object ::: mesh on which the 
        gradient would be computed 
        '''
        self.dataname = dataname 
        self.mesh = mesh

class LSGradient(ElementsGradientComputer): 
    
    def __init__(self,dataname, mesh): 
        '''
        arguments 
        dataname ::: string ::: name of the gradient data 
        mesh ::: numsim.meshe.mesh.Mesh object ::: mesh on which the 
        gradient would be computed 
        '''
        super().__init__(dataname, mesh)

    def calc_element_gradient(self,elem_indice):
        '''
        arguments 
        elem_indice ::: int ::: index of the element 
        '''
        # loop over neighbors 
        el_centroid = self.mesh.elements_centroids[elem_indice]
        el_face_conn = self.mesh.elements_face_connectivity[elem_indice]
        el_face_conn = el_face_conn[el_face_conn != -1]
        print(el_face_conn)
        print(self.mesh.internal_faces)
        el_bnd_faces = self.mesh.internal_faces[el_face_conn]
        for face in el_bnd_faces : 
            print(face)

