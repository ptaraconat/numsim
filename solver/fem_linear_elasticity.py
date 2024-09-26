from meshe.mesh import *
from fem.elements import * 
import os as os 

default_dict = {'STATE_LAW' : 'HOM_ISO',
                'HOM_ISO_POISSON' : 0.3,
                'HOM_ISO_YOUNG' : 2e6,
                'EL_TYPE' : 'TET4',
                'DUMP_DIR' : '3Dcyl_fem_linel/',
                'DUMP_DISPLACEMENT_SCALING' : 1.}

class FemLinearElasticity():
    
    def __init__(self, boundary_conditions, param_dict = default_dict):
        '''
        arguments 
        boundary_conditions ::: dict ::: dict that specifies the boundary condition 
        param_dict ::: dict ::: simulation parameters 
        '''
        self.param_dict = param_dict
        self.boundary_conditions = boundary_conditions
        self.ndim = 3
        self.state_data = 'state_mat'
        self.displacement_data = 'displacement'
        #
        if self.param_dict['EL_TYPE']== 'TET4' : 
            self.constructor = Tet4Vector()
    
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
    
    def set_homogeneous_isotropic(self, mesh, young_coeff, poisson_coeff) : 
        '''
        arguments 
        mesh ::: meshe.mesh ::: 
        young_coeff ::: float ::: Young modulus 
        poisson_coeff ::: float ::: Poisson modulus
        '''
        a = 1 - poisson_coeff
        b = poisson_coeff
        c = (1-2*poisson_coeff)*0.5
        coeff = young_coeff/((1+poisson_coeff)*(1-2*poisson_coeff))
        state_mat = coeff * np.array([[a, b, b, 0, 0, 0],
                                      [b, a, b, 0, 0, 0],
                                      [b, b, a, 0, 0, 0],
                                      [0, 0, 0, c, 0, 0],
                                      [0, 0, 0, 0, c, 0],
                                      [0, 0, 0, 0, 0, c]])
        self.set_constant_state_matrix(mesh,state_mat)
        
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
    
    def solve(self,mesh):
        '''
        solve linear elasticity 
        argument 
        mesh ::: meshe.mesh :::
        '''
        #
        savedir = self.param_dict['DUMP_DIR']
        if os.path.exists(savedir):
            print(savedir, ' already exists')
        else : 
            os.mkdir(savedir)
        # 
        if self.param_dict['STATE_LAW'] == 'HOM_ISO' : 
            young_coeff = self.param_dict['HOM_ISO_YOUNG']
            poisson_coeff = self.param_dict['HOM_ISO_POISSON']
            self.set_homogeneous_isotropic(mesh,young_coeff, poisson_coeff)
        # Calc stiffness matrix 
        self.build_discrete_operators(mesh,self.boundary_conditions)
        # Solve EDP
        solution = np.linalg.solve(self.stiffness_matrix,self.rhs_vector)
        nnodes = np.size(mesh.nodes,0)
        solution = solution.reshape((nnodes,3))
        mesh.nodes_data[self.displacement_data] = solution
        savepath = savedir + f"output_{0:04d}.vtk"
        mesh.save_vtk(output_file = savepath)
        #
        mesh.nodes_data['reference_position'] = np.copy(mesh.nodes)
        scaling = self.param_dict['DUMP_DISPLACEMENT_SCALING']
        mesh.nodes += scaling*mesh.nodes_data[self.displacement_data]
        savepath = savedir + f"output_{1:04d}.vtk"
        mesh.save_vtk(output_file = savepath)
        mesh.nodes = mesh.nodes_data['reference_position']

        