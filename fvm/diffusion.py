import numpy as np 
from meshe.mesh import * 

class FaceComputer : 
    
    def __init__(self, name) : 
        self.operator_name = name

class OrthogonalDiffusion(FaceComputer):
    '''
    '''
    
    def __init__(self):
        super().__init__('Orto_diffusion')
        
    def __call__(self, centroid1, centroid2, surface_area, surface_vector,diffusion_coeff = 1):
        '''
        arguments 
        centroid1 ::: np.array(3,) ::: coordinates of first node
        centroid2 ::: np.array(3,) ::: coordinates of the second node 
        surface_area ::: float ::: Area of face associated with the pair of nodes 
        centroid1/centroid2
        surface_vector ::: np.array(3,) ::: vector associated with that face
        diffusion_coef ::: float ::: material parameter. Diffusion coefficient 
        returns 
        surf_flux ::: float ::: surface flux through the surface induced associated with 
        the diffusion
        '''
        centroid_distance = np.sqrt(np.sum( (centroid1-centroid2)**2 ))
        gradf = (centroid1 - centroid2)/centroid_distance**2
        surf_flux = diffusion_coeff*surface_area*np.abs(np.dot(gradf,surface_vector))
        return surf_flux
        
    
    