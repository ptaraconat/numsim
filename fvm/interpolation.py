import numpy as np 
from fvm.diffusion import FaceComputer

class FaceInterpolattion(FaceComputer):
    
    def __init__(self):
        '''
        '''
        super().__init__('Face Interpolator', None)
    
    def face_computation(self,centroid1, centroid2, value1, value2, face_intrsc_vertex):
        '''
        arguments 
        centroid1 ::: np.array(3,) ::: 
        centroid2 ::: np.array(3,) :::
        value1 ::: float ::: 
        value 2 ::: float :::
        face_intrsc_vertex ::: np.array(3,) :::
        returns 
        face_value ::: float ::: interpolated value at the intersection of the 
        face and the pair of centroids 
        '''
        # Calculate distances between centroids and pair_nodes/face intersection 
        distance1 = np.sqrt(np.sum( (face_intrsc_vertex-centroid1)**2. ))
        distance2 = np.sqrt(np.sum( (face_intrsc_vertex-centroid2)**2. ))
        # Calculate distances between centroid sharing the face 
        centroids_distance = np.sqrt(np.sum( (centroid1-centroid2)**2. ))
        # Calculate centroids weights 
        weight1 = distance2/centroids_distance
        weight2 = distance1/centroids_distance
        face_value = weight1*value1 + weight2*value2
        return face_value 

class FaceGradientInterpolation(FaceComputer):
    
    def __init__(self):
        '''
        '''
        super().__init__('Face Gradient Interpolator', None)
        self.classic_interpolator = FaceInterpolattion()
        
    def face_computation(self,
                         centroid1, centroid2, 
                         value1, value2, 
                         grad_value1, grad_value2, 
                         face_intrsc_vertex):
        '''
        arguments 
        centroid1 ::: np.array(3,) ::: 
        centroid2 ::: np.array(3,) :::
        value1 ::: float ::: 
        value2 ::: float :::
        grad_value1 ::: float :::
        grad_value2 ::: float ::: 
        face_intrsc_vertex ::: np.array(3,) :::
        returns 
        interpolated_gradient ::: np.array(3,) :::  
        '''
        # Calc interpolated gradient 
        interpolated_gradient = self.classic_interpolator.face_computation(centroid1, centroid2, 
                                                                           grad_value1, grad_value2, 
                                                                           face_intrsc_vertex)
        #print('interp grad : ', interpolated_gradient)
        # calc unit vector defining centroid1/centroid2 direction 
        pair_unit_vector = centroid2 - centroid1
        centroids_distance = np.sqrt(np.sum( pair_unit_vector**2 ))
        pair_unit_vector = pair_unit_vector/centroids_distance
        # Calc interpolation corrector 
        corrector = ((value2- value1)/centroids_distance) - np.dot(interpolated_gradient,
                                                                   pair_unit_vector)
        interpolated_gradient += corrector*pair_unit_vector
        return interpolated_gradient
        