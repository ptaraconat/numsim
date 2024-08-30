from meshe.mesh import *
from fem.elements import * 

class FemDiffusion():
    
    def __init__(self,element_type = 'TET4'):
        '''
        arguments 
        element_type ::: str ::: ID of the elements used
        '''
        if element_type == 'TET4' : 
            self.constructor = Tet4()
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
        
        