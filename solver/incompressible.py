from meshe.mesh import *
from fvm.convection import * 
from fvm.diffusion import * 
from fvm.source_term import * 
from fvm.divergence import DivergenceComputer
from fvm.gradient import CellBasedGradient, LSGradient
from tstep.fdts import * 

class IncompressibleSolver():

    def __init__(self,viscosity_data = 'viscosity', source_term_data = 'source',
                 density_data = 'rho', pressure_data = 'pressure', velocity_data = 'velocity',
                 time_scheme = 'backward_euler', diffusion_scheme = 'orthogonal_diffusion',
                 convection_scheme = 'central_differencing', source_scheme = 'implicit_source', 
                 fourier = 0.3, cfl = 0.6):
        '''
        arguments 
        viscosity_data ::: str ::: 
        source_term_data ::: str ::: 
        density_data ::: str ::: 
        pressure_data ::: str ::: 
        velocity_data ::: str ::: 
        time_scheme ::: str ::: 
        diffusion_scheme ::: str ::: 
        convection_scheme ::: str ::: 
        source_scheme ::: str ::: 
        fourier ::: float ::: 
        cfl ::: float ::: 
        '''
        self.velocity_data = velocity_data
        self.pressure_data = pressure_data
        self.density_data = density_data 
        self.viscosity_data = viscosity_data
        self.source_data = source_term_data
        #
        self.constant_operator = False 
        #
        self.fourier = fourier 
        self.cfl = cfl 
        #
        self.time_scheme = time_scheme
        self.diffusion_scheme = diffusion_scheme
        self.convection_scheme = convection_scheme
        self.source_scheme = source_scheme
        #
        self.velocityx_bc ={}
        self.velocityy_bc ={}
        self.velocityz_bc ={}
        self.pressure_bc = {}
        #
        self.lddummy_data = 'p_lapl_dummy'
        self.plop = OrthogonalDiffusion(self.lddummy_data)
        #
        if self.velocity_data != None : 
            if self.convection_scheme == 'central_differencing' :
                self.convop = CentralDiffConvection(velocity_data = self.velocity_data) 
            if self.convection_scheme == 'upwind' :
                self.convop = UpwindConvection(velocity_data = self.velocity_data)
        if self.viscosity_data != None : 
            if self.diffusion_scheme == 'orthogonal_diffusion' : 
                self.diffop = OrthogonalDiffusion(self.viscosity_data)
        if self.source_data != None :
            if self.source_scheme == 'implicit_source':
                self.sourceop = SourceTerm(data_name = self.source_data)
        if self.time_scheme == 'backward_euler' : 
            self.timeop = BackwardEulerScheme()
        else : 
            if self.time_scheme == 'forward_euler':
                self.timeop = ForwardEulerScheme()
        self.divop = DivergenceComputer(self.velocity_data, 'div_'+self.velocity_data)
        #self.gradop = CellBasedGradient(self.pressure_data, 'grad_'+self.pressure_data)
        self.gradop = LSGradient(self.pressure_data, 'grad_'+self.pressure_data, use_boundaries = False)
    
    def set_boundary_conditions(self,boundary_conditions):
        '''
        arguments 
        boundary_conditions ::: dict ::: dictionary of Navier Stokes boundary 
        conditions 
        '''
        vx_bc = {}
        vy_bc = {}
        vz_bc = {}
        p_bc = {}
        for key, val in boundary_conditions.items():
            bc_type = val['type']
            if bc_type == 'inlet':
                bc_val = val['value']
                vx = bc_val[0]
                vy = bc_val[1]
                vz = bc_val[2]
                # Impose the velocity on inlet faces (BC use in velocity pred)
                vx_bc[key] = {'type' : 'dirichlet' , 'value' : vx}
                vy_bc[key] = {'type' : 'dirichlet' , 'value' : vy}
                vz_bc[key] = {'type' : 'dirichlet' , 'value' : vz}
                # Don't know how to set pressure (when solving Poisson Eq.)
                # Maybe Neumann ? 
                p_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])} # to complete 
            if bc_type == 'wall' : 
                # Shall have zero velocity on wall faces (BC use in velocity pred)
                vx_bc[key] = {'type' : 'dirichlet' , 'value' : 0}
                vy_bc[key] = {'type' : 'dirichlet' , 'value' : 0}
                vz_bc[key] = {'type' : 'dirichlet' , 'value' : 0}
                # Shall impose gradP.n = 0 on the boundary face (BC used in Poisson Eq)
                p_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])} # to complete 
            if bc_type == 'outlet':
                p_val = val['value']
                # Don't know how to set velocity (BC used in velocity pred )
                ## Maybe neumann : np.array([0,0,0]) for ensuring that velocity doesn changes 
                ## between adjacent node and Boundary face ? 
                vx_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])}  
                vy_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])}
                vz_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])}
                # Shall impose a pressure value at the outlet (BC used in Poisson Eq.)
                p_bc[key] = {'type' : 'dirichlet' , 'value' : p_val} #
            if bc_type == 'FrontBack' : 
                vx_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])}  
                vy_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])}
                vz_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])}
                # 
                p_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])} 
        self.velocityx_bc = vx_bc
        self.velocityy_bc = vy_bc
        self.velocityz_bc = vz_bc
        self.pressure_bc = p_bc
        print('###########')
        print(self.velocityx_bc)
        print(self.velocityy_bc)
        print(self.velocityz_bc)
        print(self.pressure_bc)
    
    def init_data(self, mesh):
        '''
        arguments 
        mesh ::: meshe.mesh ::: 
        ''' 
        n_elem = np.size(mesh.elements,0)
        n_bndfaces = np.size(mesh.bndfaces,0)
        # init pressure laplacian dummy data 
        arr_tmp = np.ones([n_elem,1])
        mesh.elements_data[self.lddummy_data] = arr_tmp
        arr_tmp = np.ones([n_bndfaces,1])
        mesh.bndfaces_data[self.lddummy_data] = arr_tmp
        # init velocity data 
        if self.velocity_data != None : 
            arr_tmp = np.zeros([n_elem,3])
            mesh.elements_data[self.velocity_data] = arr_tmp
            arr_tmp = np.zeros([n_bndfaces,3])
            mesh.bndfaces_data[self.velocity_data] = arr_tmp
        # init viscosity data 
        if self.viscosity_data != None : 
            arr_tmp = np.zeros((n_elem,1))
            mesh.elements_data[self.viscosity_data] = arr_tmp
            arr_tmp = np.zeros((n_bndfaces,1))
            mesh.bndfaces_data[self.viscosity_data] = arr_tmp
        # init source data 
        if self.source_data != None : 
            arr_tmp = np.zeros((n_elem,1))
            mesh.elements_data[self.source_data] = arr_tmp
            arr_tmp = np.zeros((n_bndfaces,1))
            mesh.bndfaces_data[self.source_data] = arr_tmp 
        # init pressure data 
        if self.pressure_data != None : 
            arr_tmp = np.zeros((n_elem,1))
            mesh.elements_data[self.pressure_data] = arr_tmp
            arr_tmp = np.zeros((n_bndfaces,1))
            mesh.bndfaces_data[self.pressure_data] = arr_tmp 
        # init density data 
        if self.density_data != None : 
            arr_tmp = np.zeros((n_elem,1))
            mesh.elements_data[self.density_data] = arr_tmp
            arr_tmp = np.zeros((n_bndfaces,1))
            mesh.bndfaces_data[self.density_data] = arr_tmp
        pass 
    
    def _set_pl_operators(self,mesh):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        '''
        #
        mat, rhs = self.plop(mesh, self.pressure_bc)
        self.mat_pressure = mat
        self.rhs_pressure = rhs 
    
    def _update_velocity_div(self,mesh, deltat = 1.):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        deltat ::: float ::: 
        '''
        ### calculate divergence of velocity field 
        self.divop(mesh)
        udiv = mesh.elements_data['div_'+self.velocity_data]
        # Multiply by rho divide by deltat
        density_array = mesh.elements_data[self.density_data]
        udiv = (1/deltat)*np.multiply(density_array, udiv)
        mesh.elements_data['div_'+self.velocity_data] = udiv
          
    def _set_operators(self, mesh):
        '''
        arguments 
        mesh ::: meshe.mesh ::: 
        '''
        #
        mat, rhs = self.convop(mesh, self.velocityx_bc)
        self.mat_convx = mat
        self.rhs_convx = rhs 
        #
        mat, rhs = self.convop(mesh, self.velocityy_bc)
        self.mat_convy = mat
        self.rhs_convy = rhs 
        #
        mat, rhs = self.convop(mesh, self.velocityz_bc)
        self.mat_convz = mat
        self.rhs_convz = rhs 
        #
        mat, rhs = self.diffop(mesh, self.velocityx_bc)
        self.mat_viscox = mat 
        self.rhs_viscox = rhs 
        #
        mat, rhs = self.diffop(mesh, self.velocityy_bc)
        self.mat_viscoy = mat 
        self.rhs_viscoy = rhs 
        #
        mat, rhs = self.diffop(mesh, self.velocityz_bc)
        self.mat_viscoz = mat 
        self.rhs_viscoz = rhs 
    
    def advance_velocity(self,mesh):
        '''
        arguments 
        mesh ::: meshe.Mesh ::: 
        '''
        # calc dt 
        meshsize = mesh.elements_volumes**(1/3)
        velocity_array = mesh.elements_data[self.velocity_data]
        viscosity_array = mesh.elements_data[self.viscosity_data]
        density_array = 1. #mesh.elements_data[self.density_data]
        if self.viscosity_data != None : 
            dt_diff = self.timeop._calc_dt_diff(self.fourier,
                                                viscosity_array,
                                                density_array,
                                                meshsize)
        else : 
            dt_diff = np.inf
        if self.velocity_data != None : 
            dt_conv = self.timeop._calc_dt_conv(self.cfl,
                                                velocity_array, 
                                                meshsize)
        else : 
            dt_conv = np.inf 
        dt = np.min([dt_diff,dt_conv])
        # need a fix 
        self.timeop.set_timestep(dt)
        print(dt_diff,dt_conv)
        #self.timeop.set_timestep(1)
        # MISS SOMETHINGS : explicit contribution for the different axis (x, y and z)
        self._set_operators(mesh)
        # Advance Ux 
        implicit_contribution = self.mat_convx + self.mat_viscox 
        explicit_contribution = self.rhs_convx + self.rhs_viscox 
        current_array = np.expand_dims(mesh.elements_data[self.velocity_data][:,0], axis = 1)
        next_ux = self.timeop.step(current_array, 
                                   mesh, 
                                   implicit_contribution, 
                                   explicit_contribution)
        # Advance Uy
        implicit_contribution = self.mat_convy + self.mat_viscoy 
        explicit_contribution = self.rhs_convy + self.rhs_viscoy 
        current_array = np.expand_dims(mesh.elements_data[self.velocity_data][:,1], axis = 1)
        next_uy = self.timeop.step(current_array, 
                                   mesh, 
                                   implicit_contribution, 
                                   explicit_contribution)
        # Advance Uz
        implicit_contribution = self.mat_convz + self.mat_viscoz 
        explicit_contribution = self.rhs_convz + self.rhs_viscoz 
        current_array = np.expand_dims(mesh.elements_data[self.velocity_data][:,2], axis = 1)
        next_uz = self.timeop.step(current_array, 
                                   mesh, 
                                   implicit_contribution, 
                                   explicit_contribution)
        ###
        next_array = mesh.elements_data[self.velocity_data]
        next_array[:,0] = np.squeeze(next_ux)
        next_array[:,1] = np.squeeze(next_uy)
        next_array[:,2] = np.squeeze(next_uz)
        mesh.elements_data[self.velocity_data] = next_array
        pass 
    
    def set_constant_density(self,mesh,density_value):
        '''
        arguments 
        mesh ::: meshe.mesh ::: 
        density_value ::: float ::: fluid density 
        '''
        n_elem = np.size(mesh.elements,0)
        arr_tmp = density_value*np.ones((n_elem,1))
        mesh.elements_data[self.density_data] = arr_tmp
    
    def set_constant_kinematic_viscosity(self, mesh, density_value, dyna_visco):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        density_value ::: float ::: fluid density 
        dyna_visco ::: float ::: fluid dynamics viscosity 
        '''
        n_elem = np.size(mesh.elements,0)
        n_bndfaces = np.size(mesh.bndfaces,0)
        arr_tmp = (dyna_visco/density_value)*np.ones((n_elem,1))
        mesh.elements_data[self.viscosity_data] = arr_tmp
        #
        arr_tmp = (dyna_visco/density_value)*np.ones((n_bndfaces,1))
        mesh.bndfaces_data[self.viscosity_data] = arr_tmp 
    
    def update_boundary_velocity(self, mesh, boundary_conditions):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        boundary_conditions ::: dict :::
        '''
        velocity_arr = np.zeros((np.size(mesh.bndfaces,0),3))
        # Loop over different boundary conditions 
        for bc_key,val in boundary_conditions.items():
            bc_index = mesh._get_bc_index(bc_key)
            type = val['type']
            bc_val = val['value']
            # get index associated with the current bondary condition 
            surfaces_indices =np.squeeze(np.argwhere(mesh.bndfaces_tags == bc_index))
            #if surfaces_indices.shape == () : 
            #    surfaces_indices = [surfaces_indices]         
            if type == 'inlet' : 
                velocity_arr[surfaces_indices,:] = bc_val
            if type == 'wall' : 
                velocity_arr[surfaces_indices,:] = [0, 0, 0]
            if type == 'outlet' : 
                # get bounding elements indices
                bounding_el = np.squeeze(mesh.bndfaces_elem_conn[surfaces_indices])
                bounding_el = bounding_el.astype(int)
                bounding_el_velocities = mesh.elements_data[self.velocity_data][bounding_el]
                velocity_arr[surfaces_indices,:] = bounding_el_velocities
            if type == 'FrontBack' : 
                # get bounding elements indices
                bounding_el = np.squeeze(mesh.bndfaces_elem_conn[surfaces_indices])
                bounding_el = bounding_el.astype(int)
                bounding_el_velocities = mesh.elements_data[self.velocity_data][bounding_el]
                velocity_arr[surfaces_indices,:] = bounding_el_velocities
        mesh.bndfaces_data[self.velocity_data] = velocity_arr
    
    def update_boundary_pressure(self, mesh, boundary_conditions):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        boundary_conditions ::: dict :::
        '''
        pressure_array = np.zeros((np.size(mesh.bndfaces,0),1))
        # Loop over different boundary conditions 
        for bc_key,val in boundary_conditions.items():
            bc_index = mesh._get_bc_index(bc_key)
            type = val['type']
            bc_val = val['value']
            # get index associated with the current bondary condition 
            surfaces_indices =np.squeeze(np.argwhere(mesh.bndfaces_tags == bc_index))       
            if type == 'inlet' or type == 'wall' or type == 'FrontBack': 
                # get bounding elements indices
                print(type)
                bounding_el = np.squeeze(mesh.bndfaces_elem_conn[surfaces_indices])
                bounding_el = bounding_el.astype(int)
                bounding_el_pressure = mesh.elements_data[self.pressure_data][bounding_el]
                pressure_array[surfaces_indices,:] = bounding_el_pressure
            if type == 'outlet' : 
                pressure_array[surfaces_indices,:] = bc_val
        mesh.bndfaces_data[self.pressure_data] = pressure_array 
        
    def step(self,mesh):
        '''
        arguments 
        mesh ::: meshe.Mesh :::
        ''' 
        # 1) Advance velocity 
        # 2) Solve Poisson equation for pressure 
        # 3) Correct velocity 
        pass 

