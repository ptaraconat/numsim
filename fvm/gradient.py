import numpy as np
from meshe.mesh import Mesh 

class ElementsGradientComputer : 

    def __init__(self,dataname, gdataname, mesh):
        '''
        arguments 
        dataname ::: string ::: name of the data for which 
        the gradient will be calculated  
        gdataname ::: string ::: name of the gradient data 
        mesh ::: numsim.meshe.mesh.Mesh object ::: mesh on which the 
        gradient would be computed 
        '''
        self.dataname = dataname 
        self.gdataname = gdataname
        self.mesh = mesh
    
    def calculate_gradients(self):
        '''
        Calculate data gradient at elements centroids and update 
        the mesh elements_data attribute
        '''
        gradients = []
        for i in range(len(self.mesh.elements)):
            grad = self.calc_element_gradient(i)
            gradients.append(grad)
        gradients = np.asarray(gradients)
        self.mesh.elements_data[self.gdataname] = gradients

class LSGradient(ElementsGradientComputer): 
    
    def __init__(self,dataname, gdataname, mesh, weighting = False): 
        '''
        arguments 
        dataname ::: string ::: name of the data for which 
        the gradient will be calculated  
        gdataname ::: string ::: name of the gradient data 
        mesh ::: numsim.meshe.mesh.Mesh object ::: mesh on which the 
        gradient would be computed 
        weighting ::: bool ::: wether or not use distance weights 
        for calculating gradients. 
        '''
        super().__init__(dataname, gdataname, mesh)
        self.weighting = weighting

    def calc_element_gradient(self,elem_indice):
        '''
        When considering fixed mesh, this routine may be opptimized by precomputing 
        the element/neighbors distance, 
        or even by precomputing the G^-1 matrix (G = distance_matrix.T distance_matrix). 
        arguments 
        elem_indice ::: int ::: index of the element 
        '''
        # 
        el_centroid = self.mesh.elements_centroids[elem_indice]
        el_data = self.mesh.elements_data[self.dataname][elem_indice]
        # loop over neighbors
        el_face_conn = self.mesh.elements_intf_conn[elem_indice]
        distance_matrix = []
        delta_data_vector = []
        weights_vector = []
        for face_index in el_face_conn :
            # Find neighboor element 
            index_el1, index_el2 = self.mesh.intfaces_elem_conn[face_index]
            if index_el1 == elem_indice : 
                neigh_indice = index_el2
            elif index_el2 == elem_indice : 
                neigh_indice = index_el1
            # get neighbor centroid and data value 
            neigh_centroid = self.mesh.elements_centroids[neigh_indice]
            neigh_data = self.mesh.elements_data[self.dataname][neigh_indice]
            # Update distance matrix and delta vector (delta of data)
            distance = neigh_centroid - el_centroid
            delta_data = neigh_data - el_data
            absolute_distance = np.sqrt(np.sum(distance**2.))
            weight = 1/absolute_distance
            distance_matrix.append(distance)
            delta_data_vector.append(delta_data)
            weights_vector.append(weight)
        # Uses Pseudo inverse to solve the gradients
        delta_data_vector = np.asarray(delta_data_vector)
        distance_matrix = np.asarray(distance_matrix)
        weights_vector = np.asarray(weights_vector)
        if self.weighting : 
            delta_data_vector = np.multiply(weights_vector,delta_data_vector)
            distance_matrix = distance_matrix*weights_vector[:, np.newaxis]
        #print(np.linalg.pinv(distance_matrix),np.shape(np.linalg.pinv(distance_matrix)))
        #print(np.transpose(delta_data_vector),np.shape(np.transpose(delta_data_vector)))
        #el_grad = np.dot(np.linalg.pinv(distance_matrix), np.transpose(delta_data_vector))
        el_grad = np.dot(np.linalg.pinv(distance_matrix), delta_data_vector)
        #print(el_grad, np.shape(el_grad))
        return np.squeeze(el_grad)
        
