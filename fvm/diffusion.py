import numpy as np 
from meshe.mesh import * 

class FaceComputer : 
    
    def __init__(self, name, type) : 
        self.operator_name = name
        self.operator_type = type

class NonOthogonalDiffusion(FaceComputer):
    '''
    '''
    
    def __init__(self, data_name = '', grad_data_name = '',method = 'over_relaxed'):
        super().__init__('Non-Orto_diffusion', 'explicit')
        self.method = method
        self.ortho_calculator = OrthogonalDiffusion()
        self.data_name = data_name
        self.grad_data_name = grad_data_name
    
    def cal_surface_coef(self,face_gradient,surface_area, surface_vector,diffusion_coeff = 1):
        '''
        arguments 
        face_gradient ::: np.array (3,) ::: gradient at the face 
        surface_area ::: float ::: area of the face
        surface_vector ::: np.array (3,) ::: vector associated with that face
        diffusion_coef ::: float ::: material parameter. Diffusion coefficient 
        returns 
        face_coeff ::: float ::: explicit non-orthogonal 
        '''
        face_coeff = diffusion_coeff*surface_area*np.abs(np.dot(face_gradient,surface_vector))
        return face_coeff
    
    def _decompose_normal(self,face, centroid1, centroid2):
        '''
        arguments 
        face ::: np.array (n_nodes,3) :::
        centroid1 ::: np.array (3,) :::
        centroid2 ::: np.array (3,) :::
        method ::: string :::
        returns 
        ortho_component ::: np.array (3,) :::
        nonortho_component ::: np.array (3,) :::
        '''
        theta = Mesh()._calc_face_pairnode_theta(face,centroid1, centroid2)
        centroids_vector = centroid2 - centroid1
        centroids_distance = np.sqrt(np.sum(centroids_vector**2.))
        centroids_unit_vector = centroids_vector/centroids_distance
        face_normal = Mesh()._calc_surface_normal(face)
        sign_tmp = np.sign(np.dot(face_normal,centroids_unit_vector))
        if sign_tmp < 0 :
            face_normal = -face_normal
        face_area = Mesh()._calc_surface_area(face)
        if self.method == 'over_relaxed' : 
            #print('over relaxed non orthogonal correction')
            ortho_component = (face_area/np.cos(theta))*centroids_unit_vector
            nonortho_component = face_area*face_normal - ortho_component
        return ortho_component, nonortho_component 
    
    def __call__(self,mesh,boundary_conditions,diffusion_coeff = 1):
        '''
        arguments 
        mesh ::: numsim.meshe.mesh.Mesh ::: mesh on which the diffusion operator is calculated
        boundary_conditions ::: dictionnary ::: dictionnary that specifies the boundary conditions 
        diffusion_coeff ::: float ::: 
        returns 
        matrix ::: np.array (n_elem,n_elem) :::
        '''
        from .interpolation import FaceGradientInterpolation
        grad_interpolator = FaceGradientInterpolation()
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
            face_normal_ortho_comp, face_normal_nonortho_comp = self._decompose_normal(coord_face, 
                                                                                       centroid1, 
                                                                                       centroid2)
            #face_normal = mesh._calc_surface_normal(coord_face)
            #face_area = mesh._calc_surface_area(coord_face)
            face_ortho_comp_area = np.sqrt(np.sum(face_normal_ortho_comp**2.))
            face_nonortho_comp_area = np.sqrt(np.sum(face_normal_nonortho_comp**2.))
            #
            face_normal_ortho_comp /= face_ortho_comp_area
            if face_nonortho_comp_area != 0. :
                face_normal_nonortho_comp /= face_nonortho_comp_area
            # Treat orthogonal component 
            surf_flux = self.ortho_calculator.calc_surface_coef(centroid1, 
                                                                centroid2, 
                                                                face_ortho_comp_area, 
                                                                face_normal_ortho_comp, 
                                                                diffusion_coeff=diffusion_coeff)
            # diagonal terms
            matrix[ind_cent1,ind_cent1] += -surf_flux
            matrix[ind_cent2,ind_cent2] += -surf_flux
            # off diagonal terms
            matrix[ind_cent1,ind_cent2] += surf_flux
            matrix[ind_cent2,ind_cent1] += surf_flux
            # Treat Non-orthogonal component 
            value1 = mesh.elements_data[self.data_name][ind_cent1]
            value2 = mesh.elements_data[self.data_name][ind_cent2]
            grad_value1 = mesh.elements_data[self.grad_data_name][ind_cent1]
            grad_value2 = mesh.elements_data[self.grad_data_name][ind_cent2]
            face_centroid = Mesh()._calc_centroid(coord_face)
            face_gradient = grad_interpolator.face_computation(centroid1, centroid2, 
                                                               value1, value2, 
                                                               grad_value1, grad_value2, 
                                                               face_centroid)
            surf_flux = self.cal_surface_coef(face_gradient,
                                              face_nonortho_comp_area, 
                                              face_normal_nonortho_comp,
                                              diffusion_coeff = diffusion_coeff)
            #print('orthogonal vector')
            #print('vec : ', face_normal_ortho_comp)
            #print('mag : ', face_ortho_comp_area)
            #print('non orthogonal vector')
            #print('vec : ', face_normal_nonortho_comp)
            #print('mag : ', face_nonortho_comp_area)
            #print('value 1 : ', value1)
            #print('value 2 : ',value2)
            #print('grad val 1 : ', grad_value1)
            #print('grad val 2 : ', grad_value2)
            #print('face_gradient : ',face_gradient)
            # Check sign ??
            rhs_vec[ind_cent1] += -surf_flux
            rhs_vec[ind_cent2] += +surf_flux
            del ind_cent1, ind_cent2, centroid1, centroid2, coord_face 
            del face_normal_ortho_comp, face_normal_nonortho_comp, face_ortho_comp_area, face_nonortho_comp_area
            del surf_flux, value1, value2, grad_value1, grad_value2
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
                    #surface_area = mesh._calc_surface_area(face_nodes) 
                    #surface_normal = mesh._calc_surface_normal(face_nodes)
                    face_normal_ortho_comp, face_normal_nonortho_comp = self._decompose_normal(face_nodes, 
                                                                                               centroid, 
                                                                                               face_centroid)
                    face_ortho_comp_area = np.sqrt(np.sum(face_normal_ortho_comp**2.))
                    face_nonortho_comp_area = np.sqrt(np.sum(face_normal_nonortho_comp**2.))
                    face_normal_ortho_comp /= face_ortho_comp_area
                    if face_nonortho_comp_area != 0. :
                        face_normal_nonortho_comp /= face_nonortho_comp_area
                    # Treat orthogonal component 
                    face_coeff = self.ortho_calculator.calc_dirchlet_bnd_surface_coef(centroid, 
                                                                                      face_centroid, 
                                                                                      face_ortho_comp_area, 
                                                                                      face_normal_ortho_comp, 
                                                                                      diffusion_coeff = diffusion_coeff)
                    bc_dir_value = bc_val
                    # rework ::: check sign
                    rhs_vec[elem_ind] += -bc_dir_value*face_coeff 
                    matrix[elem_ind,elem_ind] += -face_coeff
                    # Treat non orthogonal component
                    prev_centroid_val = mesh.elements_data[self.data_name][elem_ind]
                    face_coeff = self.ortho_calculator.calc_dirchlet_bnd_surface_coef(centroid, 
                                                                                      face_centroid, 
                                                                                      face_nonortho_comp_area, 
                                                                                      face_normal_nonortho_comp, 
                                                                                      diffusion_coeff = diffusion_coeff)
                    rhs_vec[elem_ind] += - face_coeff*(bc_dir_value-prev_centroid_val) ###### check sign
                if type == 'neumann' : 
                    # For a Non-orthogonal treatement we do exactely the same as for 
                    # the orthogonal diffusion 
                    # Treat a Von Neumann bc 
                    surface_area = mesh._calc_surface_area(face_nodes)
                    surface_normal = mesh._calc_surface_normal(face_nodes)
                    bc_neu_value =  bc_val
                    face_coeff = self.ortho_calculator.calc_neumann_bnd_surface_coef(surface_area, 
                                                                                     surface_normal, 
                                                                                     bc_neu_value)
                    # rework ::: check sign
                    rhs_vec[elem_ind] += -face_coeff
        return matrix, rhs_vec

class OrthogonalDiffusion(FaceComputer):
    '''
    '''
    
    def __init__(self):
        super().__init__('Orto_diffusion', 'implicit')
        
    def calc_surface_coef(self, centroid1, centroid2, surface_area, surface_vector,diffusion_coeff = 1):
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
    
    def calc_dirchlet_bnd_surface_coef(self,centroid,surface_centroid,surface_area, surface_vector, diffusion_coeff = 1):
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
    
    def calc_neumann_bnd_surface_coef(self, surface_area, surface_vector, surface_flux):
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
    
    def __call__(self,mesh,boundary_conditions,diffusion_coeff = 1):
        '''
        arguments 
        mesh ::: numsim.meshe.mesh.Mesh ::: mesh on which the diffusion operator is calculated
        boundary_conditions ::: dictionnary ::: dictionnary that specifies the boundary conditions 
        diffusion_coeff ::: float ::: 
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
            #
            surf_flux = self.calc_surface_coef(centroid1, 
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
                    face_coeff = self.calc_dirchlet_bnd_surface_coef(centroid, 
                                                                     face_centroid, 
                                                                     surface_area, 
                                                                     surface_normal, 
                                                                     diffusion_coeff = diffusion_coeff)
                    bc_dir_value = bc_val
                    # rework ::: check sign
                    rhs_vec[elem_ind] += -bc_dir_value*face_coeff 
                    matrix[elem_ind,elem_ind] += -face_coeff 
                if type == 'neumann' : 
                    # Treat a Von Neumann bc 
                    surface_area = mesh._calc_surface_area(face_nodes)
                    surface_normal = mesh._calc_surface_normal(face_nodes)
                    bc_neu_value =  bc_val
                    face_coeff = self.calc_neumann_bnd_surface_coef(surface_area, 
                                                                    surface_normal, 
                                                                    bc_neu_value)
                    # rework ::: check sign
                    rhs_vec[elem_ind] += -face_coeff
            #print(np.shape(mesh.bndfaces_tags))
            #print(np.shape(surfaces_indices))
            
        return matrix, rhs_vec
            
        
    
    
    