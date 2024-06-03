from meshe.mesh import *
from fvm.convection import * 
from fvm.diffusion import * 
from fvm.source_term import * 
from fvm.divergence import DivergenceComputer
from fvm.gradient import CellBasedGradient, LSGradient
from tstep.fdts import * 

class SimpleSolver():

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
        # Grid Data
        self.velocity_data = velocity_data
        self.pressure_data = pressure_data
        self.density_data = density_data 
        self.viscosity_data = viscosity_data
        self.source_data = source_term_data
        # Ini boundary conditions schemes 
        self.velocityx_bc ={}
        self.velocityy_bc ={}
        self.velocityz_bc ={}
        self.pressure_bc = {}
        # FVM Schema 
        self.diffusion_scheme = diffusion_scheme
        self.convection_scheme = convection_scheme
        self.source_scheme = source_scheme
        # convection schema 
        if self.convection_scheme == 'central_differencing' :
            self.convop = CentralDiffConvection(velocity_data = self.velocity_data) 
        if self.convection_scheme == 'upwind' :
            self.convop = UpwindConvection(velocity_data = self.velocity_data)
        # diffusion schema 
        if self.diffusion_scheme == 'orthogonal_diffusion' : 
            self.diffop = OrthogonalDiffusion(self.viscosity_data)
        # source term schema
        if self.source_scheme == 'implicit_source':
            self.sourceop = SourceTerm(data_name = self.source_data)
        self.lddummy_data = 'p_lapl_dummy'
        self.plop = OrthogonalDiffusion(self.lddummy_data)
        self.divop = DivergenceComputer(self.velocity_data, 'div_'+self.velocity_data)
        self.gradop = LSGradient(self.pressure_data, 'grad_'+self.pressure_data, use_boundaries = False)
        # Time advancement scheme 
        self.time_scheme = time_scheme 
        self.fourier = fourier 
        self.cfl = cfl 
        if self.time_scheme == 'backward_euler' : 
            self.timeop = BackwardEulerScheme()
        if self.time_scheme == 'forward_euler':
            self.timeop = ForwardEulerScheme()
    
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
            arr_tmp = np.zeros((n_elem,3))
            mesh.elements_data[self.source_data] = arr_tmp
            arr_tmp = np.zeros((n_bndfaces,3))
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
    
    def _set_umatrices(self, mesh):
        '''
        arguments 
        mesh ::: meshe.mesh ::: 
        '''
        #
        mat, rhs = self.convop(mesh, self.velocityx_bc)
        self.mat_convx = mat
        self.rhs_convx = rhs 
        mat, rhs = self.diffop(mesh, self.velocityx_bc)
        self.mat_viscox = mat 
        self.rhs_viscox = rhs 
        self.matx = self.mat_convx + self.mat_viscox
        self.rhsx = self.rhs_convx + self.rhs_viscox
        #
        mat, rhs = self.convop(mesh, self.velocityy_bc)
        self.mat_convy = mat
        self.rhs_convy = rhs 
        mat, rhs = self.diffop(mesh, self.velocityy_bc)
        self.mat_viscoy = mat 
        self.rhs_viscoy = rhs 
        self.maty = self.mat_convy + self.mat_viscoy 
        self.rhsy = self.rhs_convy + self.rhs_viscoy
        #
        mat, rhs = self.convop(mesh, self.velocityz_bc)
        self.mat_convz = mat
        self.rhs_convz = rhs 
        mat, rhs = self.diffop(mesh, self.velocityz_bc)
        self.mat_viscoz = mat 
        self.rhs_viscoz = rhs 
        self.matz = self.mat_convz + self.mat_viscoz
        self.rhsz = self.rhs_convz + self.rhs_viscoz
        #
        self.diag_matx = np.diag(self.matx)
        self.diag_maty = np.diag(self.maty)
        self.diag_matz = np.diag(self.matz)
        self.inv_diag_matx = 1./self.diag_matx
        self.inv_diag_maty = 1./self.diag_maty
        self.inv_diag_matz = 1./self.diag_matz
    
    def _set_source_term(self,mesh):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        '''
        mesh.elements_data[self.source_data] = mesh.elements_data['grad_'+self.pressure_data]
        rhs = self.sourceop(mesh)
        self.rhs_source = rhs
    
    def _momentum_step(self,mesh):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        '''
        self._set_umatrices(mesh)
        self.gradop(mesh)
        self._set_source_term(mesh)
        # update X momentum 
        mat = self.matx 
        rhs = self.rhsx + np.expand_dims(self.rhs_source[:,0], axis = 1)
        solutionx = np.linalg.solve(mat,rhs)
        # update Y momentum 
        mat = self.maty 
        rhs = self.rhsy + np.expand_dims(self.rhs_source[:,1], axis = 1)
        solutiony = np.linalg.solve(mat,rhs)
        # update Z momentum 
        mat = self.matz 
        rhs = self.rhsz + np.expand_dims(self.rhs_source[:,2], axis = 1)
        solutionz = np.linalg.solve(mat,rhs)
        # New velocity 
        mesh.elements_data[self.velocity_data][:,0] = np.squeeze(solutionx)
        mesh.elements_data[self.velocity_data][:,1] = np.squeeze(solutiony)
        mesh.elements_data[self.velocity_data][:,2] = np.squeeze(solutionz)
        print(mesh.elements_data[self.velocity_data])
    
    def _update_hmatrix(self, mesh):
        '''
        arguments 
        mesh ::: meshe.mesh ::: 
        '''
        #
        velocityx = mesh.elements_data[self.velocity_data][:,0]
        velocityy = mesh.elements_data[self.velocity_data][:,1]
        velocityz = mesh.elements_data[self.velocity_data][:,2]
        self.hmatx = np.multiply(self.diag_matx, velocityx) - np.dot(self.matx, velocityx)
        self.hmaty = np.multiply(self.diag_maty, velocityy) - np.dot(self.maty, velocityy)
        self.hmatz = np.multiply(self.diag_matz, velocityz) - np.dot(self.matz, velocityz)

        
    