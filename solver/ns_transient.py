from meshe.mesh import *
from fvm.convection import * 
from fvm.diffusion import * 
from fvm.source_term import * 
from fvm.divergence import DivergenceComputer
from fvm.gradient import CellBasedGradient, LSGradient
from tstep.fdts import * 
from fvm.source_term import *

class TNSSolver():

    def __init__(self,time_scheme = 'backward_euler', diffusion_scheme = 'orthogonal_diffusion',
                 convection_scheme = 'central_differencing', source_scheme = 'implicit_source', 
                 fourier = 0.3, cfl = 0.6):
        '''
        '''
        self.time_scheme = time_scheme
        self.diffusion_scheme = diffusion_scheme
        self.convection_scheme = convection_scheme
        self.source_scheme = source_scheme
        self.fourier = fourier
        self.cfl = cfl 
        #
        self.density_data = 'density'
        self.viscosity_data = 'viscosity'
        self.lddummy_data = 'p_lapl_dummy'
        self.velocity_data = 'velocity'
        self.ustar_data = 'ustar'
        self.pressure_data = 'pressure'
        self.source_data = 'source'
        self.grad_pressure_data = 'grad_'+ self.pressure_data
        self.div_grad_pressure_data = 'div_grad_'+ self.pressure_data
        self.next_pressure_data = 'next_'+self.pressure_data
        self.grad_next_pressure_data = 'grad_'+self.next_pressure_data
        self.div_ustar_data = 'div_'+self.ustar_data
        #
        self.poissonop = OrthogonalDiffusion(self.lddummy_data)
        self.div_ustar_op = DivergenceComputer(self.ustar_data, self.div_ustar_data)
        self.div_gradp_op = DivergenceComputer(self.grad_pressure_data, self.div_grad_pressure_data)
        self.grad_pressure_op = LSGradient(self.pressure_data, self.grad_pressure_data, use_boundaries = False)
        self.grad_nextp_op = LSGradient(self.next_pressure_data, self.grad_next_pressure_data, use_boundaries = False)
        # Convection 
        if self.convection_scheme == 'central_differencing' :
            self.convop = CentralDiffConvection(velocity_data = self.velocity_data) 
        if self.convection_scheme == 'upwind' :
            self.convop = UpwindConvection(velocity_data = self.velocity_data)
        # Diffusion 
        if self.diffusion_scheme == 'orthogonal_diffusion' : 
            self.diffop = OrthogonalDiffusion(self.viscosity_data)
        # Source 
        if self.source_scheme == 'implicit_source':
            self.sourceop = SourceTerm(data_name = self.source_data)
        # Time schema 
        if self.time_scheme == 'backward_euler' : 
            self.timeop = BackwardEulerScheme()
        if self.time_scheme == 'forward_euler':
            self.timeop = ForwardEulerScheme()
    
    def initialize_data(self,mesh):
        '''
        arguments 
        mesh ::: meshe.Mesh :::
        '''
        n_elem = np.size(mesh.elements,0)
        n_bndfaces = np.size(mesh.bndfaces,0)
        # init pressure laplacian dummy data 
        arr_tmp = np.ones([n_elem,1])
        mesh.elements_data[self.lddummy_data] = arr_tmp
        arr_tmp = np.ones([n_bndfaces,1])
        mesh.bndfaces_data[self.lddummy_data] = arr_tmp
        # init velocity data 
        arr_tmp = np.zeros([n_elem,3])
        mesh.elements_data[self.velocity_data] = arr_tmp
        arr_tmp = np.zeros([n_bndfaces,3])
        mesh.bndfaces_data[self.velocity_data] = arr_tmp
        # init ustar data 
        arr_tmp = np.zeros([n_elem,3])
        mesh.elements_data[self.ustar_data] = arr_tmp
        arr_tmp = np.zeros([n_bndfaces,3])
        mesh.bndfaces_data[self.ustar_data] = arr_tmp
        # init viscosity data 
        arr_tmp = np.zeros((n_elem,1))
        mesh.elements_data[self.viscosity_data] = arr_tmp
        arr_tmp = np.zeros((n_bndfaces,1))
        mesh.bndfaces_data[self.viscosity_data] = arr_tmp
        # init source data 
        arr_tmp = np.zeros((n_elem,3))
        mesh.elements_data[self.source_data] = arr_tmp
        arr_tmp = np.zeros((n_bndfaces,3))
        mesh.bndfaces_data[self.source_data] = arr_tmp 
        # init pressure data 
        arr_tmp = np.zeros((n_elem,1))
        mesh.elements_data[self.pressure_data] = arr_tmp
        arr_tmp = np.zeros((n_bndfaces,1))
        mesh.bndfaces_data[self.pressure_data] = arr_tmp 
        # init density data 
        arr_tmp = np.zeros((n_elem,1))
        mesh.elements_data[self.density_data] = arr_tmp
        arr_tmp = np.zeros((n_bndfaces,1))
        mesh.bndfaces_data[self.density_data] = arr_tmp
    
    def set_constant_kinematic_viscosity(self, mesh, density_value, dyna_visco):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        density_value ::: float ::: fluid density 
        dyna_visco ::: float ::: fluid dynamics viscosity 
        '''
        n_elem = np.size(mesh.elements,0)
        n_bndfaces = np.size(mesh.bndfaces,0)
        #
        arr_tmp = density_value*np.ones((n_elem,1))
        mesh.elements_data[self.density_data] = arr_tmp
        arr_tmp = (dyna_visco/density_value)*np.ones((n_elem,1))
        mesh.elements_data[self.viscosity_data] = arr_tmp
        #
        arr_tmp = density_value*np.ones((n_bndfaces,1))
        mesh.bndfaces_data[self.density_data] = arr_tmp
        arr_tmp = (dyna_visco/density_value)*np.ones((n_bndfaces,1))
        mesh.bndfaces_data[self.viscosity_data] = arr_tmp 
    
    def set_boundary_conditions(self,boundary_conditions):
        '''
        arguments 
        boundary_conditions ::: dict ::: dictionary of Navier Stokes boundary 
        conditions 
        '''
        self.boundary_conditions = boundary_conditions
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
                p_bc[key] = {'type' : 'neumann' , 'value' : np.array([0,0,0])} 
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
    
    def set_time_step(self,mesh):
        '''
        arguments 
        mesh ::: meshe.Mesh :::
        '''
        meshsize = mesh.elements_volumes**(1/3)
        velocity_array = mesh.elements_data[self.velocity_data]
        viscosity_array = mesh.elements_data[self.viscosity_data]
        density_array = 1. #mesh.elements_data[self.density_data]
        dt_diff = self.timeop._calc_dt_diff(self.fourier,
                                            viscosity_array,
                                            density_array,
                                            meshsize)
        dt_conv = self.timeop._calc_dt_conv(self.cfl,
                                            velocity_array, 
                                            meshsize)
        print(dt_diff,dt_conv)
        dt = np.min([dt_diff,dt_conv])
        # need a fix 
        self.timeop.set_timestep(dt)
        
    def calc_time_step(self,mesh,velocity_val):
        '''
        arguments :
        mesh ::: meshe.Mesh ::: 
        velocity_vall ::: flaot :::
        '''
        meshsize = mesh.elements_volumes**(1/3)
        velocity_array = np.zeros((np.shape(mesh.elements_data[self.velocity_data])))
        velocity_array[:,0] = velocity_val
        viscosity_array = mesh.elements_data[self.viscosity_data]
        density_array = 1. #mesh.elements_data[self.density_data]
        dt_diff = self.timeop._calc_dt_diff(self.fourier,
                                            viscosity_array,
                                            density_array,
                                            meshsize)
        dt_conv = self.timeop._calc_dt_conv(self.cfl,
                                            velocity_array, 
                                            meshsize)
        print(dt_diff,dt_conv)
        dt = np.min([dt_diff,dt_conv])
        return dt 
        
    
    def set_convective_operators(self,mesh) : 
        '''
        arguments 
        mesh ::: meshe.Mesh :::
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
    
    def set_diffusive_operators(self,mesh):
        '''
        qrguments :
        mesh ::: meshe.Mesh :::
        '''
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
    
    def update_source_data(self,mesh):
        '''
        arguments
        mesh ::: meshe.Mesh ::: 
        '''
        # calc pressure gradient 
        self.grad_pressure_op(mesh)
        mesh.elements_data[self.source_data] = np.divide(-mesh.elements_data[self.grad_pressure_data],
                                                         mesh.elements_data[self.density_data])
    
    def set_source_operators(self,mesh):
        '''
        arguments : 
        mesh ::: meshe.Mesh:::
        '''
        #
        self.update_source_data(mesh)
        #
        source_arr = self.sourceop(mesh)
        self.rhs_sourcex = np.expand_dims(source_arr[:,0], axis = 1)
        self.rhs_sourcey = np.expand_dims(source_arr[:,1], axis = 1)
        self.rhs_sourcez = np.expand_dims(source_arr[:,2], axis = 1)
         
    def projection_step(self,mesh):
        '''
        arguments 
        mesh ::: meshe.Mesh :::
        '''
        #self.set_time_step(mesh)
        #
        self.set_convective_operators(mesh)
        self.set_diffusive_operators(mesh)
        self.set_source_operators(mesh)
        # Calc Ustar by advancing velocity 
        # Advance Ux 
        implicit_contribution = self.mat_convx + self.mat_viscox 
        explicit_contribution = self.rhs_convx + self.rhs_viscox + self.rhs_sourcex
        current_array = np.expand_dims(mesh.elements_data[self.velocity_data][:,0], axis = 1)
        next_ux = self.timeop.step(current_array, 
                                   mesh, 
                                   implicit_contribution, 
                                   explicit_contribution)
        del implicit_contribution, explicit_contribution, current_array
        # Advance Uy
        implicit_contribution = self.mat_convy + self.mat_viscoy 
        explicit_contribution = self.rhs_convy + self.rhs_viscoy + self.rhs_sourcey
        current_array = np.expand_dims(mesh.elements_data[self.velocity_data][:,1], axis = 1)
        next_uy = self.timeop.step(current_array, 
                                   mesh, 
                                   implicit_contribution, 
                                   explicit_contribution)
        del implicit_contribution, explicit_contribution, current_array
        # Advance Uz
        implicit_contribution = self.mat_convz + self.mat_viscoz 
        explicit_contribution = self.rhs_convz + self.rhs_viscoz + self.rhs_sourcez
        current_array = np.expand_dims(mesh.elements_data[self.velocity_data][:,2], axis = 1)
        next_uz = self.timeop.step(current_array, 
                                   mesh, 
                                   implicit_contribution, 
                                   explicit_contribution)
        del implicit_contribution, explicit_contribution, current_array
        ###
        next_array = mesh.elements_data[self.ustar_data]
        next_array[:,0] = np.squeeze(next_ux)
        next_array[:,1] = np.squeeze(next_uy)
        next_array[:,2] = np.squeeze(next_uz)
        mesh.elements_data[self.ustar_data] = next_array
    
    def update_bnd_data(self,mesh):
        '''
        arguments
        mesh ::: meshe.Mesh :::
        '''
        velocity_arr = np.zeros((np.size(mesh.bndfaces,0),3))
        pressure_grad_arr = np.zeros((np.size(mesh.bndfaces,0),3))
        # Loop over different boundary conditions 
        boundary_conditions = self.boundary_conditions
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
                #pressure_grad_arr[surfaces_indices,:] = [0, 0, 0]
            if type == 'wall' : 
                velocity_arr[surfaces_indices,:] = [0, 0, 0]
                #pressure_grad_arr[surfaces_indices,:] = [0, 0, 0]
            if type == 'outlet' : 
                # get bounding elements indices
                bounding_el = np.squeeze(mesh.bndfaces_elem_conn[surfaces_indices])
                bounding_el = bounding_el.astype(int)
                bounding_el_velocities = mesh.elements_data[self.ustar_data][bounding_el]
                velocity_arr[surfaces_indices,:] = bounding_el_velocities
            if type == 'FrontBack' : 
                # get bounding elements indices
                bounding_el = np.squeeze(mesh.bndfaces_elem_conn[surfaces_indices])
                bounding_el = bounding_el.astype(int)
                bounding_el_velocities = mesh.elements_data[self.ustar_data][bounding_el]
                velocity_arr[surfaces_indices,:] = bounding_el_velocities
        mesh.bndfaces_data[self.ustar_data] = velocity_arr
        mesh.bndfaces_data[self.grad_pressure_data] = pressure_grad_arr
    
    def calc_poisson_rhs(self,mesh, deltat = 1.):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        deltat ::: float ::: 
        '''
        ### Update Boundary data for divergence computations 
        self.update_bnd_data(mesh)
        ### calculate divergence of predicted velocity field 
        self.div_ustar_op(mesh)
        ### calculate divergence of current pressure gradient 
        self.div_gradp_op(mesh)
        #
        div_ustar = mesh.elements_data[self.div_ustar_data]
        #print(div_ustar)
        rho = mesh.elements_data[self.density_data]
        div_gradp = mesh.elements_data[self.div_grad_pressure_data]
        #
        rhs = (1/deltat)*np.multiply(rho,div_ustar)+div_gradp
        return rhs 
    
    def set_poisson_operator(self,mesh,deltat=1):
        '''
        arguments 
        mesh ::: meshe.Mesh ::: 
        deltat ::: float ::: time step 
        '''
        poisson_rhs = self.calc_poisson_rhs(mesh,deltat=deltat)
        mat, rhs = self.poissonop(mesh, self.pressure_bc)
        self.mat_pressure = mat
        self.rhs_pressure = rhs + poisson_rhs
    
    def poisson_step(self,mesh):
        '''
        arguments 
        mesh ::: meshe.Mesh:::
        '''
        deltat = self.timeop.dt
        self.set_poisson_operator(mesh,deltat = deltat)
        next_pressure = np.linalg.solve(self.mat_pressure,self.rhs_pressure)
        mesh.elements_data[self.next_pressure_data] = next_pressure
    
    def correction_step(self,mesh):
        '''
        arguments 
        mesh ::: meshe.Mesh :::
        '''
        #calculate pressure gradients 
        self.grad_nextp_op(mesh)
        self.grad_pressure_op(mesh) 
        #
        ustar = mesh.elements_data[self.ustar_data]
        gradp = mesh.elements_data[self.grad_pressure_data]
        gradp_np1 = mesh.elements_data[self.grad_next_pressure_data]
        rho = mesh.elements_data[self.density_data]
        deltat = self.timeop.dt
        deltat_o_rho = deltat*(1/rho)
        #
        u_np1 = ustar + np.multiply(deltat_o_rho,-gradp_np1+gradp)
        #print('USTAR')
        #print(ustar)
        #print(mesh.elements_data[self.next_pressure_data])
        #print(gradp_np1)
        #print(np.multiply(deltat_o_rho,-gradp_np1+gradp))
        #print(u_np1)
        #
        mesh.elements_data[self.velocity_data] = u_np1
        mesh.elements_data[self.pressure_data] = mesh.elements_data[self.next_pressure_data]
        #print(mesh.elements_data[self.source_data])
        #print(mesh.elements_data[self.pressure_data])
        #print(mesh.elements_data[self.velocity_data])
        
        
        
        
        
        
