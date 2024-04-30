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
                              velocity1, velocity2):
        '''
        arguments 
        centroid1 ::: np.array(3,) ::: coordinates of first node
        centroid2 ::: np.array(3,) ::: coordinates of the second node 
        surface_area ::: float ::: Area of face associated with the pair of nodes 
        centroid1/centroid2
        surface_vector ::: np.array(3,) ::: vector associated with that face
        face_vertex ::: np.array(3,) ::: Point that belong to the face. Typically, its centroid 
        velocity1 ::: np.array(3,) ::: Convective velocity at centroid1
        velocity2 ::: np.array(3,) ::: Convective velocity at centroid2
        returns 
        surf_flux ::: float ::: surface flux through the surface induced associated with 
        the convection 
        weight1 ::: float ::: 
        weight2 ::: float ::: 
        '''
        face_velocity = FaceInterpolattion().face_computation(centroid1, centroid2, 
                                                              velocity1, velocity2, 
                                                              face_vertex)
        #
        # Calculate distances between centroids and pair_nodes/face intersection 
        distance1 = np.sqrt(np.sum( (face_vertex-centroid1)**2. ))
        distance2 = np.sqrt(np.sum( (face_vertex-centroid2)**2. ))
        # Calculate distances between centroid sharing the face 
        centroids_distance = distance1 + distance2 #np.sqrt(np.sum( (centroid1-centroid2)**2. ))
        # Calculate centroids weights 
        weight1 = distance2/centroids_distance
        weight2 = distance1/centroids_distance
        #
        sign = np.sign(np.dot(surface_vector,centroid2-centroid1))
        if sign < 0 : 
            surface_vector = - surface_vector
        #
        flow_rate = surface_area*np.dot(surface_vector,face_velocity)
        return flow_rate
    