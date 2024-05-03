import numpy as np
from meshe.mesh import Mesh 
from .interpolation import FaceInterpolattion

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

class CellBasedGradient(ElementsGradientComputer):
    '''
    '''
    def __init__(self,dataname, grad_dataname):
        '''
        arguments 
        dataname ::: str ::: name of the data 
        grad_dataname ::: str ::: name of the gradient data 
        '''
        self.dataname = dataname
        self.grad_dataname = grad_dataname
    
    def calc_surface_component(self, centroid1, centroid2, 
                               surface_area, surface_vector, 
                               face_vertex,
                               data1, data2):
        '''
        arguments 
        centroid1 ::: np.array(3,) ::: coordinates of first node
        centroid2 ::: np.array(3,) ::: coordinates of the second node 
        surface_area ::: float ::: Area of face associated with the pair of nodes 
        centroid1/centroid2
        surface_vector ::: np.array(3,) ::: vector associated with that face
        face_vertex ::: np.array(3,) ::: Point that belong to the face. Typically, its centroid 
        data1 ::: float or np.array(data_dim,) ::: Convective velocity at centroid1
        data2 ::: float or np.array(data_dim,) ::: Convective velocity at centroid2
        returns 
        face_component ::: float ::: 
        '''
        face_data = FaceInterpolattion().face_computation(centroid1, centroid2, 
                                                              data1, data2, 
                                                              face_vertex)
        #
        sign = np.sign(np.dot(surface_vector,centroid2-centroid1))
        if sign < 0 : 
            surface_vector = - surface_vector
        #
        #face_component = surface_area*np.dot(face_data,surface_vector)
        face_data = np.transpose(np.expand_dims(face_data,axis = 0))
        face_component = surface_area*face_data*surface_vector
        return face_component
    
    def calc_bndface_component(self,centroid, face_data,face_centroid, face_area, face_normal):
        '''
        arguments
        centroid ::: np.array(3,) ::: coordinates of the element to which the boundary 
        face belongs to 
        face_data ::: float or np.array(data_dim,) ::: data for which the face flux 
        is computed 
        face_centroid ::: np.array(3,) ::: coordinate of the boundary face centroid
        face_area ::: float ::: area of the boundary face  
        face_normal ::: np.array(3,) ::: face normal vector
        returns 
        face_component ::: np.array() :::
        '''
        #
        sign = np.sign(np.dot(face_normal,centroid-face_centroid))
        if sign > 0 : 
            face_normal = - face_normal
        #flux = face_area * np.dot(face_normal,face_data)
        face_data = np.transpose(np.expand_dims(face_data,axis = 0))
        face_component = face_area*face_data*face_normal
        return face_component

class LSGradient(ElementsGradientComputer): 
    
    def __init__(self,dataname, gdataname, mesh, weighting = False, use_boundaries = False): 
        '''
        arguments 
        dataname ::: string ::: name of the data for which 
        the gradient will be calculated  
        gdataname ::: string ::: name of the gradient data 
        mesh ::: numsim.meshe.mesh.Mesh object ::: mesh on which the 
        gradient would be computed 
        weighting ::: bool ::: wether or not use distance weights 
        for calculating gradients. 
        use_boundaries ::: bool ::: 
        '''
        super().__init__(dataname, gdataname, mesh)
        self.weighting = weighting
        self.use_boundaries = use_boundaries

    def calc_element_gradient(self,elem_indice ):
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
        if self.use_boundaries : 
            el_bndface_conn = self.mesh.elements_bndf_conn[elem_indice]
            for face_index in el_bndface_conn : 
                neigh_bndf_indices = self.mesh.bndfaces[face_index]
                neigh_bndf = self.mesh.nodes[neigh_bndf_indices]
                # get neighbor centroid and data value
                neigh_centroid = self.mesh._calc_centroid(neigh_bndf)
                neigh_data = self.mesh.bndfaces_data[self.dataname][face_index]
                if neigh_data != None : 
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
        
