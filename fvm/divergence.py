import numpy as np
from meshe.mesh import Mesh 
from .interpolation import FaceInterpolattion

class DivergenceComputer : 
    
    def __init__(self, dataname, divdataname):
        '''
        argument
        dataname ::: str ::: name of the data for which divergence will be computed 
        divdataname ::: str ::: name of the divergence data 
        '''
        self.dataname = dataname
        self.divdataname = divdataname
    
    def calc_surface_flowrate(self, centroid1, centroid2, 
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
        surf_flux ::: float ::: surface flux through the surface induced associated with 
        the convection 
        weight1 ::: float ::: 
        weight2 ::: float ::: 
        '''
        face_velocity = FaceInterpolattion().face_computation(centroid1, centroid2, 
                                                              data1, data2, 
                                                              face_vertex)
        #
        sign = np.sign(np.dot(surface_vector,centroid2-centroid1))
        if sign < 0 : 
            surface_vector = - surface_vector
        #
        flow_rate = surface_area*np.dot(surface_vector,face_velocity)
        return flow_rate
    
    def calc_bndface_flowrate(self,centroid, face_data,face_centroid, face_area, face_normal):
        '''
        arguments
        centroid ::: np.array(3,) ::: coordinates of the element to which the boundary 
        face belongs to 
        face_data ::: float or np.array(data_dim,) ::: data for which the face flux 
        is computed 
        face_centroid ::: np.array(3,) ::: coordinate of the boundary face centroid
        face_area ::: float ::: area of the boundary face  
        face_normal ::: np.array(3,) ::: face normal vector
        '''
        flux = face_area * np.dot(face_normal,face_data)
        #
        sign = np.sign(np.dot(face_normal,centroid-face_centroid))
        if sign > 0 : 
            flux = - flux
        return flux 
    