import numpy as np 
from meshe.mesh import * 

class FaceComputer : 
    
    def __init__(self, name, type) : 
        self.operator_name = name
        self.operator_type = type

class OrthogonalDiffusion(FaceComputer):
    '''
    '''
    
    def __init__(self):
        super().__init__('Orto_diffusion', 'implicit')
        
    def calc_surface_flux(self, centroid1, centroid2, surface_area, surface_vector,diffusion_coeff = 1):
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
    
    def calc_dirchlet_bnd_surface_flux(self,centroid,surface_centroid,surface_area, surface_vector, diffusion_coeff = 1):
        '''
        arguments 
        centroid ::: np.array(3,) ::: coordinates of first node
        surface_centroid ::: np.array(3,) ::: coordinates of the second node 
        surface_area ::: float ::: Area of the boundary face
        surface_vector ::: np.array(3,) ::: vector associated with that face
        diffusion_coef ::: float ::: material parameter. Diffusion coefficient 
        returns 
        surf_flux ::: float ::: surface flux through the boundary surface
        '''
        centroids_distance = np.sqrt(np.sum( (centroid-surface_centroid)**2 ) )
        gradf = (centroid-surface_centroid)/centroids_distance**2.
        #print('centroids distance :', centroids_distance)
        #print('grad f : ', gradf)
        surf_flux = diffusion_coeff*surface_area*np.abs(np.dot(gradf,surface_vector))
        return surf_flux
    
    def calc_neumann_bnd_surface_flux(self, surface_area, surface_vector, surface_flux):
        '''
        arguments 
        surface_area ::: float ::: Area of the boundary face
        surface_vector ::: np.array(3,) ::: vector associated with that face
        surface_flux ::: np.array(3,) ::: vector deffining the flux through the 
        boudary face (viz. Von Neumann Boundary condition)
        returns 
        surf_flux ::: float ::: surface flux through the boundary surface
        '''
        surf_flux = surface_area*np.dot(surface_vector,surface_flux)
        return surf_flux
    
    def __call__(self,mesh,diffusion_coeff = 1):
        '''
        arguments 
        mesh ::: numsim.meshe.mesh.Mesh ::: mesh on which the diffusion operator is calculated 
        diffusion_coeff ::: float ::: 
        returns 
        matrix ::: np.array (n_elem,n_elem) :::
        '''
        n_elem = np.size(mesh.elements,0)
        matrix = np.zeros((n_elem,n_elem))
        for ind_face,face in enumerate(mesh.intfaces) : 
            ind_cent1 = mesh.intfaces_elem_conn[ind_face][0]
            ind_cent2 = mesh.intfaces_elem_conn[ind_face][1]
            centroid1 = mesh.elements_centroids[ind_cent1]
            centroid2 = mesh.elements_centroids[ind_cent2]
            #
            coord_face = mesh.nodes[face]
            face_area = mesh._calc_surface_area(coord_face)
            face_normal = mesh._calc_surface_normal(coord_face)
            #
            surf_flux = self.calc_surface_flux(centroid1, 
                                               centroid2, 
                                               face_area, 
                                               face_normal, 
                                               diffusion_coeff=diffusion_coeff)
            # diagonal terms
            matrix[ind_cent1,ind_cent1] += -surf_flux
            matrix[ind_cent2,ind_cent2] += -surf_flux
            # off diagonal terms
            matrix[ind_cent1,ind_cent2] += surf_flux
            matrix[ind_cent2,ind_cent1] += surf_flux
        # Treat boundaries 
        return matrix 
            
        
    
    
    