from meshe.mesh import *
from fem.elements import * 

class FemLinearElasticity():
    
    def __init__(self,element_type = 'TET4'):
        '''
        arguments 
        element_type ::: str ::: ID of the elements used
        '''
        if element_type == 'TET4' : 
            self.constructor = Tet4Vector()
            self.ndim = 3
        self.state_data = 'state_mat'
    
    def set_constant_state_matrix(self,mesh,state_matrix) : 
        '''
        arguments 
        mesh ::: meshe.mesh ::: domain grid 
        state_matrix ::: np.array (6,6) ::: diffusion matrix 
        '''
        nnodes = np.size(mesh.nodes,0)
        ndim = 6
        state = np.zeros((nnodes,ndim,ndim))
        for i in range(nnodes): 
            state[i,:,:] = state_matrix
        mesh.nodes_data[self.state_data] = state
        
    def build_discrete_operators(self,mesh,boundary_conditions):
        '''
        arguments 
        mesh ::: meshe.mesh ::: domain grid 
        boundary_conditions ::: dict ::: 
        '''
        nnodes = np.size(mesh.nodes,0)
        stiffness_matrix = self.constructor.calc_global_stiffness_matrix(mesh, self.state_data)
        # Apply dirichlet Boundary condition 
        rhs_vector = np.zeros((nnodes*3,1))
        bc = 'inlet'
        for bc in boundary_conditions.keys() : 
            bc_val = boundary_conditions[bc]['value']
            bc_type =  boundary_conditions[bc]['type']
            bc_tag = mesh.physical_entities[bc][0]
            if bc_type == 'dirichlet' : 
                boundary_elements = mesh.bndfaces[np.where(mesh.bndfaces_tags == bc_tag)[0]]
                boundary_node_indices = np.unique(boundary_elements.flatten())
                for i,val in enumerate(bc_val) : 
                    print(i,val)
                    comp_conn = 3*boundary_node_indices + i
                    if val != None : 
                        rhs_vector[comp_conn] = val
                        stiffness_matrix[comp_conn,:] = 0 
                        stiffness_matrix[comp_conn,comp_conn] = 1
        #
        self.rhs_vector = rhs_vector
        self.stiffness_matrix = stiffness_matrix
        
        