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
    
    def calc_stress(self,mesh):
        '''
        argument 
        mesh ::: meshe.mesh :::
        '''
        stress_arr = []
        for i in range(np.size(mesh.elements,0)):
            element = mesh.elements[i,:]
            element_coords = mesh.nodes[element]
            state_arr = mesh.nodes_data[self.state_data][element]
            disp_arr = mesh.nodes_data[self.displacement_data][element]
            self.constructor.set_element(element_coords)
            stress = self.constructor.calc_stress_tensor(disp_arr,state_arr)
            stress_arr.append(stress)
        stress_arr = np.asarray(stress_arr)
        print(np.shape(stress_arr))
        mesh.elements_data['ssigma_xx'] = stress_arr[:,0]
        mesh.elements_data['ssigma_yy'] = stress_arr[:,1]
        mesh.elements_data['ssigma_zz'] = stress_arr[:,2]
        mesh.elements_data['ssigma_xy'] = stress_arr[:,3]
        mesh.elements_data['ssigma_xz'] = stress_arr[:,4]
        mesh.elements_data['ssigma_yz'] = stress_arr[:,5]
    
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
        self.calc_stress(mesh)
        savepath = savedir + f"output_{0:04d}.vtk"
        mesh.save_vtk(output_file = savepath)
        #
        mesh.nodes_data['reference_position'] = np.copy(mesh.nodes)
        scaling = self.param_dict['DUMP_DISPLACEMENT_SCALING']
        mesh.nodes += scaling*mesh.nodes_data[self.displacement_data]
        savepath = savedir + f"output_{1:04d}.vtk"
        mesh.save_vtk(output_file = savepath)
        mesh.nodes = mesh.nodes_data['reference_position']

class FemElastodyn(FemLinearElasticity): 
    '''
    '''
    def __init__(self,boundary_conditions, param_dict = default_dict) : 
        '''
        '''
        super().__init__(boundary_conditions=boundary_conditions, param_dict=param_dict)
        self.rho_data = 'rho'
        self.ddot_data = 'u_ddot'
        self.dot_data = 'u_dot'
        self.gamma = 1/2
        self.beta = 1/4
        self.deltat = param_dict['DT']
    
    def set_constant_rho_data(self,mesh,rho_value) : 
        '''
        arguments 
        mesh ::: meshe.mesh ::: domain grid 
        rho_value ::: float ::: rho value (constant related to the mass Matrix)
        '''
        nnodes = np.size(mesh.nodes,0)
        rho_arr = rho_value*np.ones((nnodes))
        mesh.nodes_data[self.rho_data] = rho_arr

    def build_discrete_operators(self,mesh,boundary_conditions={}):
        '''
        arguments 
        mesh ::: meshe.mesh ::: domain grid 
        boundary_conditions ::: dict ::: 
        '''
        stiffness_matrix = self.constructor.calc_global_stiffness_matrix(mesh, self.state_data)
        mass_matrix = self.constructor.calc_global_mass_matrix(mesh,self.rho_data)
        # treat dirichlet BC 
        dirichlet_values = np.zeros((np.size(mesh.nodes,0)*3))
        ddot_dirichlet_values = np.zeros((np.size(mesh.nodes,0)*3))
        dirichlet_indices = []
        for bc in boundary_conditions.keys() : 
            bc_val = boundary_conditions[bc]['value']
            bc_type =  boundary_conditions[bc]['type']
            bc_tag = mesh.physical_entities[bc][0]
            ####
            if bc_type == 'dirichlet' : 
                # Find boundary node indexes 
                boundary_elements = mesh.bndfaces[np.where(mesh.bndfaces_tags == bc_tag)[0]]
                boundary_node_indices = np.unique(boundary_elements.flatten())
                # Loop over the three components of the boundary condition 
                for i,val in enumerate(bc_val) : 
                    print(i,val)
                    # calcute the indexes in the Matrix-Vector form 
                    # associated with the boundary condition, 
                    # given the component  
                    comp_conn = 3*boundary_node_indices + i
                    if val != None : 
                        # If the component is specified 
                        # update the Dirichlet BC index list 
                        # and the associated values array
                        comp_conn = comp_conn.tolist()
                        dirichlet_values[comp_conn] = bc_val[i]
                        dirichlet_indices = dirichlet_indices + comp_conn
        # Set dirichlet indices/values attributes 
        all_indices = list(range(0,np.size(mesh.nodes,0)*3))
        not_dirichlet_indices = all_indices.copy() 
        for i in dirichlet_indices : 
            not_dirichlet_indices.remove(i)
        self.dirichlet_indices = dirichlet_indices
        self.not_dirichlet_indices = not_dirichlet_indices
        self.dirichlet_values = dirichlet_values
        # Update mass and stiffness matrices so that they account for Dirichlet BC
        # and update RHS consequently 
        reduced_stiffness = np.delete(stiffness_matrix,self.dirichlet_indices,axis = 0)
        reduced_mass = np.delete(mass_matrix, self.dirichlet_indices, axis = 0 )
        rhs_dirbc = np.dot(reduced_stiffness,dirichlet_values) + np.dot(reduced_mass, ddot_dirichlet_values)
        reduced_stiffness = np.delete(reduced_stiffness,self.dirichlet_indices,axis = 1)
        reduced_mass = np.delete(reduced_mass, self.dirichlet_indices, axis = 1 )
        # 
        self.mass_matrix = reduced_mass
        self.stiffness_matrix = reduced_stiffness
        self.forcing_term = np.zeros((np.size(reduced_mass,0)))
        self.forcing_term =  self.forcing_term - rhs_dirbc
    
    def init_data(self, mesh): 
        '''
        argument 
        mesh ::: meshe.mesh :::
        '''
        nnodes = np.size(mesh.nodes,0)
        #
        mesh.nodes_data['init_loc'] = mesh.nodes.copy()
        self.u_prev = np.zeros((nnodes*3))
        self.u_prev = self.dirichlet_values
        self.dotu_prev = np.zeros((nnodes*3))
        #rhs = self.forcing_term - np.dot(self.stiffness_matrix,mesh.nodes_data[self.displacement_data])
        #arr_tmp = np.linalg.solve(self.mass_matrix, rhs)
        #mesh.nodes_data[self.ddot_data] = arr_tmp
        self.ddotu_prev = np.zeros((nnodes*3))
        #
        mesh.nodes_data[self.displacement_data] = self.u_prev.reshape((nnodes,3))
        mesh.nodes_data[self.dot_data] = self.dotu_prev.reshape((nnodes,3))
        mesh.nodes_data[self.ddot_data] = self.ddotu_prev.reshape((nnodes,3))
    
    def step(self,mesh):
        '''
        argument 
        mesh ::: meshe.mesh :::
        '''
        u_prev = self.u_prev[self.not_dirichlet_indices]
        dotu_prev = self.dotu_prev[self.not_dirichlet_indices]
        ddotu_prev = self.ddotu_prev[self.not_dirichlet_indices]
        #
        u_pred = u_prev + self.deltat*dotu_prev + 0.5*(self.deltat**2)*(1-2*self.beta)*ddotu_prev
        dotu_pred = dotu_prev + self.deltat*(1-self.gamma)*ddotu_prev
        #
        ddotu_next = np.linalg.solve(self.smat, self.forcing_term - np.dot(self.stiffness_matrix,u_pred))
        #
        u_next = u_pred + (self.deltat**2)*self.beta*ddotu_next
        dotu_next = dotu_pred + self.deltat*self.gamma*ddotu_next
        #
        self.u_prev[self.not_dirichlet_indices] = u_next
        self.dotu_prev[self.not_dirichlet_indices] = dotu_next
        self.ddotu_prev[self.not_dirichlet_indices] = ddotu_next
        nnodes = np.size(mesh.nodes,0)
        mesh.nodes_data[self.displacement_data] = self.u_prev.reshape((nnodes,3))
        mesh.nodes_data[self.dot_data] = self.dotu_prev.reshape((nnodes,3))
        mesh.nodes_data[self.ddot_data]= self.ddotu_prev.reshape((nnodes,3))
        

    
    def solve(self,mesh):
        '''
        solve linear elasttodynamics
        argument 
        mesh ::: meshe.mesh :::
        '''
        #
        # 
        if self.param_dict['STATE_LAW'] == 'HOM_ISO' : 
            young_coeff = self.param_dict['HOM_ISO_YOUNG']
            poisson_coeff = self.param_dict['HOM_ISO_POISSON']
            self.set_homogeneous_isotropic(mesh,young_coeff, poisson_coeff)
        # 
        rho_value = self.param_dict['RHO']
        self.set_constant_rho_data(mesh, rho_value)
        self.build_discrete_operators(mesh,self.boundary_conditions)
        self.init_data(mesh)
        self.smat = self.mass_matrix + self.beta*(self.deltat**2.)*self.stiffness_matrix
        # Temporal Loop 
        n_ite = self.param_dict['NITE']
        dump_ite = self.param_dict['DUMPITE']
        savedir = self.param_dict['DUMPDIR'] 
        # Create savedir 
        if os.path.exists(savedir):
            print(savedir, ' already exists')
        else : 
            os.mkdir(savedir)
        for i in range(n_ite):
            if i % dump_ite == 0 :
                self.calc_stress(mesh)
                #
                mesh.nodes_data['init_loc'] = np.copy(mesh.nodes)
                scaling = self.param_dict['DUMP_DISPLACEMENT_SCALING']
                mesh.nodes += scaling*mesh.nodes_data[self.displacement_data]
                #
                print(np.mean(mesh.nodes_data[self.displacement_data], axis = 0 ))
                save_path = savedir + f"output_{i:04d}.vtk"
                print('dump solution : ', save_path)
                mesh.save_vtk(output_file = save_path)
                mesh.nodes = mesh.nodes_data['init_loc']
            self.step(mesh) 