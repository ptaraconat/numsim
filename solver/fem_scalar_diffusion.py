from meshe.mesh import *
from fem.elements import * 

class FemDiffusion():
    
    def __init__(self,element_type = 'TET4'):
        '''
        arguments 
        element_type ::: str ::: ID of the elements used
        '''
        if element_type == 'TET4' : 
            self.constructor = Tet4Scalar()
            self.ndim = 3
        self.diffusion_data = 'diffusion_mat'
    
    def set_constant_diffusion(self,mesh,diffusion_matrix) : 
        '''
        arguments 
        mesh ::: meshe.mesh ::: domain grid 
        diffusion_matrix ::: np.array (3,3) ::: diffusion matrix 
        '''
        nnodes = np.size(mesh.nodes,0)
        ndim = self.ndim
        state = np.zeros((nnodes,ndim,ndim))
        for i in range(nnodes): 
            state[i,:,:] = diffusion_matrix
        mesh.nodes_data[self.diffusion_data] = state
        
    def build_discrete_operators(self,mesh,boundary_conditions):
        '''
        arguments 
        mesh ::: meshe.mesh ::: domain grid 
        boundary_conditions ::: dict ::: 
        '''
        nnodes = np.size(mesh.nodes,0)
        stiffness_matrix = self.constructor.calc_global_stiffness_matrix(mesh, self.diffusion_data)
        # Apply dirichlet Boundary condition 
        rhs_vector = np.zeros((nnodes,1))
        bc = 'inlet'
        for bc in boundary_conditions.keys() : 
            bc_val = boundary_conditions[bc]['value']
            bc_type =  boundary_conditions[bc]['type']
            bc_tag = mesh.physical_entities[bc][0]
            if bc_type == 'dirichlet' : 
                boundary_elements = mesh.bndfaces[np.where(mesh.bndfaces_tags == bc_tag)[0]]
                boundary_node_indices = np.unique(boundary_elements.flatten())
                rhs_vector[boundary_node_indices] = bc_val
                stiffness_matrix[boundary_node_indices,:] = 0
                stiffness_matrix[boundary_node_indices,boundary_node_indices] = 1
        #
        self.rhs_vector = rhs_vector
        self.stiffness_matrix = stiffness_matrix

class FemUnsteadyDiffusion(FemDiffusion):
    '''
    '''
    def __init__(self,element_type = 'TET4',alpha = 0.5, deltat = 0.001): 
        '''
        '''
        super().__init__(element_type = element_type)  
        self.rho_data = 'rho' 
        self.solved_data = 'temp'
        self.predictor_data = 'predictor'
        self.alpha = alpha
        self.deltat = deltat

    def set_constant_rho_data(self,mesh,rho_value) : 
        '''
        arguments 
        mesh ::: meshe.mesh ::: domain grid 
        rho_value ::: float ::: rho value (constant related to the mass Matrix)
        '''
        nnodes = np.size(mesh.nodes,0)
        rho_arr = rho_value*np.ones((nnodes))
        mesh.nodes_data[self.rho_data] = rho_arr
    
    def initialize_solved_data(self, mesh, init_value = 0):
        '''
        mesh ::: meshe.mesh ::: domain grid 
        init_value ::: float ::: constant intial value for the data we are solving 
        '''
        nnodes = np.size(mesh.nodes,0)
        mesh.nodes_data[self.solved_data] = init_value*np.ones((nnodes))
        mesh.nodes_data[self.predictor_data] = np.zeros((nnodes))
    
    def build_discrete_operators(self,mesh,boundary_conditions={}):
        '''
        arguments 
        mesh ::: meshe.mesh ::: domain grid 
        boundary_conditions ::: dict ::: 
        '''
        stiffness_matrix = self.constructor.calc_global_stiffness_matrix(mesh, self.diffusion_data)
        mass_matrix = self.constructor.calc_global_mass_matrix(mesh,self.rho_data)
        #
        self.mass_matrix = mass_matrix
        self.stiffness_matrix = stiffness_matrix
        self.forcing_term = np.zeros((np.size(mesh.nodes,0)))
        
    def time_stepping(self,mesh, boundary_conditions) : 
        '''
        arguments 
        mesh ::: meshe.mesh ::: domain grid 
        '''
        prev_data = mesh.nodes_data[self.solved_data]
        prev_velocity = mesh.nodes_data[self.predictor_data]
        prediction_arr = prev_data + (1-self.alpha)*self.deltat*prev_velocity
        mat = self.mass_matrix + (self.alpha*self.deltat*self.stiffness_matrix)
        rhs = (self.alpha*self.deltat*self.forcing_term) - np.dot(self.mass_matrix,prediction_arr)
        #Apply dirichlet Boundary condition 
        bc = 'inlet'
        for bc in boundary_conditions.keys() : 
            bc_val = boundary_conditions[bc]['value']
            bc_type =  boundary_conditions[bc]['type']
            bc_tag = mesh.physical_entities[bc][0]
            if bc_type == 'dirichlet' : 
                boundary_elements = mesh.bndfaces[np.where(mesh.bndfaces_tags == bc_tag)[0]]
                boundary_node_indices = np.unique(boundary_elements.flatten())
                rhs[boundary_node_indices] = bc_val
                mat[boundary_node_indices,:] = 0
                mat[boundary_node_indices,boundary_node_indices] = 1
        # Solve and update data 
        current_data = np.linalg.solve(mat,rhs)
        prev_velocity = (current_data-prediction_arr)/(self.alpha*self.deltat)
        mesh.nodes_data[self.predictor_data] = prev_velocity
        mesh.nodes_data[self.solved_data] = current_data