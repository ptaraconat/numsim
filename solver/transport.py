from meshe.mesh import *
from fvm.convection import * 
from fvm.diffusion import * 
from fvm.source_term import * 
from tstep.fdts import * 

class TransportSolver():
    '''
    '''
    def __init__(self, transported_data, velocity = None, diffusivity = None, 
                 time_scheme = 'backward_euler', diffusion_scheme = 'orthogonal_diffusion',
                 convection_scheme = 'central_differencing',
                 diffusion_coeff = 1., fourier = 0.3, cfl = 0.6):
        '''
        arguments 
        transported_data ::: str ::: name of the transported data 
        velocity ::: str ::: name of the velocity data 
        diffusivity ::: str ::: name of the diffusivity data 
        '''
        self.trans_data = transported_data
        self.velocity_data = velocity 
        self.diffusivity_data = diffusivity
        self.time_scheme = time_scheme
        self.diffusion_scheme = diffusion_scheme
        self.convection_scheme = convection_scheme
        self.fourier = fourier
        self.cfl = cfl 
        # must be replaced with the diffusivity data 
        # some changes are needed in the diffusion schemes
        self.diffusion_coeff = diffusion_coeff
        #
        if self.velocity_data != None : 
            if self.convection_scheme == 'central_differencing' :
                self.convop = CentralDiffConvection(velocity_data = self.velocity_data) 
        if self.diffusivity_data != None : 
            if self.diffusion_scheme == 'orthogonal_diffusion' : 
                self.diffop = OrthogonalDiffusion()
        if self.time_scheme == 'backward_euler' : 
            self.timeop = BackwardEulerScheme()
    
    def _set_operators(self, mesh, boundary_conditions):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        boundary_conditions ::: dictionnary ::: 
        '''
        n_elem = np.size(mesh.elements,0)
        #
        if self.velocity_data != None :
            mat, rhs = self.convop(mesh, boundary_conditions)
        else : 
            mat = np.zeros((n_elem,n_elem))
            rhs = np.zeros((n_elem,1))
        self.mat_conv = mat
        self.rhs_conv = rhs 
        del mat, rhs
        #
        if self.diffusivity_data != None : 
            mat, rhs = self.diffop(mesh, boundary_conditions, self.diffusion_coeff)
        else : 
            mat = np.zeros((n_elem,n_elem))
            rhs = np.zeros((n_elem,1))
        self.mat_diff = mat 
        self.rhs_diff = rhs 
        del mat, rhs 
    
    def step(self,mesh, boundary_conditions ):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        boundary_conditions ::: dict ::: 
        '''
        # calc dt 
        meshsize = mesh.elements_volumes**(1/3)
        velocity_array = mesh.element_data[self.velocity_data]
        # must be improved 
        rho = 1
        if self.diffusivity_data != None : 
            dt_diff = self.timeop._calc_dt_diff(self.fourier,
                                                self.diffusion_coeff,
                                                rho,
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
        self.timeop.set_timestep(dt)
        #
        self._set_operators(mesh, boundary_conditions)
        implicit_contribution = self.mat_conv + self.mat_diff
        explicit_contribution = self.rhs_conv + self.rhs_diff
        current_array = mesh.elements_data[self.trans_data]
        next_array = self.timeop.step(current_array, 
                                         mesh, 
                                         implicit_contribution, 
                                         explicit_contribution)
        mesh.elements_data[self.trans_data] = next_array
        
    def initialize_data(self,mesh):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        '''
        n_elem = np.size(mesh.elements,0)
        n_bndfaces = np.size(mesh.bndfaces,0)
        # init transported data 
        arr_tmp = np.zeros((n_elem,1))
        mesh.elements_data[self.trans_data] = arr_tmp
        arr_tmp = np.zeros((n_bndfaces,1))
        mesh.bndfaces_data[self.trans_data] = arr_tmp
        # init velocity data 
        if self.velocity_data != None : 
            arr_tmp = np.zeros([n_elem,3])
            mesh.elements_data[self.velocity_data] = arr_tmp
            arr_tmp = np.zeros([n_bndfaces,3])
            mesh.bndfaces_data[self.velocity_data] = arr_tmp
        # init diffusivity data 
        if self.diffusivity_data != None : 
            arr_tmp = np.zeros((n_elem,1))
            mesh.elements_data[self.diffusivity_data] = arr_tmp
            arr_tmp = np.zeros((n_bndfaces,1))
            mesh.bndfaces_data[self.diffusivity_data] = arr_tmp
        
    def set_constant_velocity(self,mesh):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        '''
        pass 
    
    def set_constant_diffusivity(self,mesh):
        '''
        arguments 
        mesh ::: meshe.mesh :::
        '''
        pass 
        