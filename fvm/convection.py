import numpy as np 

from .diffusion import FaceComputer
from .interpolation import FaceInterpolattion


class ConvectionOperator(FaceComputer):
    def __init__(self,velocity_data = ''):
        '''
        '''
        self.velocity_data = velocity_data
    
    def calc_dirichlet_surface_coeff(self, surface_area, surface_vector, surface_velocity,centroid,surface_centroid):
        '''
        arguments : 
        surface_area ::: float ::: Area of face associated with the pair of nodes 
        centroid1/centroid2
        surface_vector ::: np.array(3,) ::: vector associated with that face
        surface_velocity ::: np.array(3,) ::: Convective velocity at
        the boundary face
        centroid ::: 
        surface_centroid :::
        returns 
        surf_coeff ::: float ::: convective coefficient associated with the boundary surface 
        '''
        sign = np.sign(np.dot(surface_vector,surface_centroid-centroid))
        if sign < 0 : 
            surface_vector = - surface_vector
        surf_coeff = surface_area*np.dot(surface_vector,surface_velocity)
        return surf_coeff 
    
    def calc_neumann_surface_coeff(self,centroid, surface_centroid,surface_area, surface_vector, surface_velocity, surface_gradient):
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
        sign = np.sign(np.dot(surface_vector,surface_centroid-centroid))
        if sign < 0 : 
            surface_vector = - surface_vector
        centroids_distance = centroid - surface_centroid
        #surface_value = centroid_value - np.dot(surface_gradient,centroids_distance)
        rhs_comp =  np.dot(surface_gradient,centroids_distance)
        surf_coeff = surface_area*np.abs(np.dot(surface_vector,surface_velocity))
        return surf_coeff, rhs_comp
    
    def __call__(self,mesh, boundary_conditions):
        '''
        arguments 
        mesh ::: numsim.meshe.mesh.Mesh ::: mesh on which the diffusion operator is calculated
        boundary_conditions ::: dictionnary ::: dictionnary that specifies the boundary conditions 
        returns 
        matrix ::: np.array (n_elem,n_elem) :::
        '''
        n_elem = np.size(mesh.elements,0)
        matrix = np.zeros((n_elem,n_elem))
        rhs_vec = np.zeros((n_elem,1))
        for ind_face,face in enumerate(mesh.intfaces) : 
            ind_cent1 = mesh.intfaces_elem_conn[ind_face][0]
            ind_cent2 = mesh.intfaces_elem_conn[ind_face][1]
            centroid1 = mesh.elements_centroids[ind_cent1]
            centroid2 = mesh.elements_centroids[ind_cent2]
            #
            coord_face = mesh.nodes[face]
            face_area = mesh._calc_surface_area(coord_face)
            face_normal = mesh._calc_surface_normal(coord_face)
            face_centroid = mesh._calc_centroid(coord_face)
            #
            velocity1 = mesh.elements_data[self.velocity_data][ind_cent1]
            velocity2 = mesh.elements_data[self.velocity_data][ind_cent2]
            #
            surf_flux, w1, w2 = self.calc_surface_coef(centroid1, centroid2, 
                                                       face_area, 
                                                       face_normal, 
                                                       face_centroid,
                                                       velocity1, 
                                                       velocity2)
            # Fill Matrix 
            #
            matrix[ind_cent1,ind_cent1] += -w1*surf_flux
            matrix[ind_cent1,ind_cent2] += -w2*surf_flux
            #
            matrix[ind_cent2,ind_cent2] += +w2*surf_flux
            matrix[ind_cent2,ind_cent1] += +w1*surf_flux
            #
            del centroid1, centroid2, velocity1, velocity2, w1, w2, ind_cent1, ind_cent2
            del coord_face, face_area, face_normal, face_centroid, surf_flux
        # Treat boundaries 
        # Loop over different boundary conditions 
        for bc_key,val in boundary_conditions.items():
            bc_index = mesh._get_bc_index(bc_key)
            type = val['type']
            bc_val = val['value']
            # get index associated with the current bondary condition 
            surfaces_indices =np.squeeze(np.argwhere(mesh.bndfaces_tags == bc_index))
            if surfaces_indices.shape == () : 
                surfaces_indices = [surfaces_indices]
            # Loop over the later boundary surfaces 
            for i in surfaces_indices :
                # get face nodes 
                bnd_face = mesh.bndfaces[i]
                elem_ind = int(mesh.bndfaces_elem_conn[i][0])
                face_nodes = mesh.nodes[bnd_face]
                if type == 'dirichlet':
                    # Treat a dirichlet bc
                    centroid = mesh.elements_centroids[elem_ind]
                    face_centroid = mesh._calc_centroid(face_nodes)
                    surface_area = mesh._calc_surface_area(face_nodes) 
                    surface_normal = mesh._calc_surface_normal(face_nodes)
                    surface_velocity = mesh.bndfaces_data[self.velocity_data][i]
                    face_coeff = self.calc_dirichlet_surface_coeff(surface_area, surface_normal, surface_velocity, centroid, face_centroid)
                    bc_dir_value = bc_val
                    # rework ::: check sign
                    rhs_vec[elem_ind] += +bc_dir_value*face_coeff 
                    del surface_area, surface_normal, surface_velocity, face_coeff, bc_dir_value
                if type == 'neumann' : 
                    # Treat a Von Neumann bc 
                    centroid = mesh.elements_centroids[elem_ind]
                    surface_centroid = mesh._calc_centroid(face_nodes)
                    surface_area = mesh._calc_surface_area(face_nodes)
                    surface_normal = mesh._calc_surface_normal(face_nodes)
                    surface_velocity =mesh.bndfaces_data[self.velocity_data][i]
                    #centroid_value = mesh.elements_data[self.convected_data][elem_ind]
                    bc_neu_value =  bc_val
                    face_coeff, face_value = self.calc_neumann_surface_coeff(centroid, 
                                                                 surface_centroid,
                                                                 surface_area, 
                                                                 surface_normal, 
                                                                 surface_velocity,  
                                                                 bc_neu_value)
                    # rework ::: check sign
                    matrix[elem_ind,elem_ind] += -face_coeff
                    rhs_vec[elem_ind] += -face_coeff*face_value
        return matrix, rhs_vec

class CentralDiffConvection(ConvectionOperator):

    def __init__(self,velocity_data = ''):
        '''
        '''
        super().__init__(velocity_data=velocity_data)

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
        sign = np.sign(np.dot(surface_vector,centroid2-centroid1))
        if sign < 0 : 
            surface_vector = - surface_vector
        #
        surf_coeff = surface_area*np.dot(surface_vector,face_velocity)
        return surf_coeff, weight1, weight2
  
class UpwindConvection(ConvectionOperator): 
    
    def __init__(self,velocity_data = ''):
        '''
        '''
        super().__init__(velocity_data=velocity_data)
    
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
        
        
        