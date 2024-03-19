import numpy as np 

from .diffusion import FaceComputer
from .interpolation import FaceInterpolattion

class CentralDiffConvection(FaceComputer):

    def __init__(self,velocity_data = ''):
        '''
        '''
        self.velocity_data = velocity_data
    
    def calc_surface_coef(self, centroid1, centroid2, surface_area, surface_vector, face_vertex,velocity1, velocity2):
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
        surf_coeff = surface_area*np.abs(np.dot(surface_vector,face_velocity))
        return surf_coeff, weight1, weight2
    
    def calc_dirichlet_surface_coeff(self, surface_area, surface_vector, surface_velocity):
        '''
        arguments : 
        surface_area ::: float ::: Area of face associated with the pair of nodes 
        centroid1/centroid2
        surface_vector ::: np.array(3,) ::: vector associated with that face
        surface_velocity ::: np.array(3,) ::: Convective velocity at
        the boundary face
        returns 
        surf_coeff ::: float ::: convective coefficient associated with the boundary surface 
        '''
        surf_coeff = surface_area*np.abs(np.dot(surface_vector,surface_velocity))
        return surf_coeff 
    
    def calc_neumann_surface_coeff(self,centroid, surface_centroid,surface_area, surface_vector, surface_velocity, centroid_value, surface_gradient):
        '''
        arguments : 
        centroid ::: np.array(3,) ::: 
        surface_centroid ::: np.array(3,) ::: 
        surface_area ::: float ::: Area of face associated with the pair of nodes 
        centroid1/centroid2
        surface_vector ::: np.array(3,) ::: vector associated with that face
        surface_velocity ::: np.array(3,) ::: Convective velocity at
        the boundary face
        centroid_value ::: float ::: value at the element centroid 
        surface_gradient ::: np.array(3,) ::: gradient value at the boundary face 
        returns 
        surf_coeff ::: float ::: convective coefficient associated with the boundary surface
        surface_value ::: float ::: data value at the boundary face  
        '''
        centroids_distance = centroid - surface_centroid
        surface_value = centroid_value - np.dot(surface_gradient,centroids_distance)
        surf_coeff = surface_area*np.abs(np.dot(surface_vector,surface_velocity))
        return surf_coeff, surface_value
    
class UpwindConvection(FaceComputer): 
    
    def __init__(self,velocity_data = ''):
        '''
        '''
        self.velocity_data = velocity_data 
    
    def calc_surface_coef(self,centroid1, centroid2, surface_area, surface_vector, face_vertex, velocity1, velocity2):
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
        print(face_velocity)
        #
        centroids_vector = centroid2 - centroid1 
        sign = np.sign(np.dot(centroids_vector,face_velocity))
        if sign > 0 :
            # the flow goes from centroid1 to centroid2 
            # it goes out from the control volume arround centroid1 
            weight1 = 1 
            weight2 = 0 
        else :
            # the flow goes in the control volume arround centroid1 
            weight2 = 1 
            weight1 = 0 
        #
        surf_coeff = surface_area*np.abs(np.dot(surface_vector,face_velocity))
        return surf_coeff, weight1, weight2
        
        
        