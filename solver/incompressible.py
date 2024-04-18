from meshe.mesh import *
from fvm.convection import * 
from fvm.diffusion import * 
from fvm.source_term import * 
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
        if self.velocity_data != None : 
            if self.convection_scheme == 'central_differencing' :
                self.convop = CentralDiffConvection(velocity_data = self.velocity_data) 
        if self.viscosity_data != None : 
            if self.diffusion_scheme == 'orthogonal_diffusion' : 
                self.diffop = OrthogonalDiffusion(self.viscosity_data)
        if self.source_data != None :
            if self.source_scheme == 'implicit_source':
                self.sourceop = SourceTerm(data_name = self.source_data)
        if self.time_scheme == 'backward_euler' : 
            self.timeop = BackwardEulerScheme()
    
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
            print(val)
            bc_type = val['type']
            if bc_type == 'inlet':
                bc_val = val['value']
                vx = bc_val[0]
                vy = bc_val[1]
                vz = bc_val[2]
                #
                vx_bc[key] = {'type' : 'dirichlet' , 'value' : vx}
                vy_bc[key] = {'type' : 'dirichlet' , 'value' : vy}
                vz_bc[key] = {'type' : 'dirichlet' , 'value' : vz}
                p_bc[key] = {'type' : '' , 'value' : None} # to complete 
            if bc_type == 'wall' : 
                #
                vx_bc[key] = {'type' : 'dirichlet' , 'value' : 0}
                vy_bc[key] = {'type' : 'dirichlet' , 'value' : 0}
                vz_bc[key] = {'type' : 'dirichlet' , 'value' : 0}
                p_bc[key] = {'type' : '' , 'value' : None} # to complete 
            if bc_type == 'outlet':
                #
                vx_bc[key] = {'type' : '' , 'value' : None}
                vy_bc[key] = {'type' : '' , 'value' : None}
                vz_bc[key] = {'type' : '' , 'value' : None}
                p_bc[key] = {'type' : '' , 'value' : None} # to complete 
        self.velocityx_bc = vx_bc
        self.velocityy_bc = vy_bc
        self.velocityz_bc = vz_bc
        self.pressure_bc = p_bc
        print('###########')
        print(self.velocityx_bc)
        print(self.velocityy_bc)
        print(self.velocityz_bc)
        print(self.pressure_bc)
