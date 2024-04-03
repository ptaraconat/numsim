import numpy as np 

class TimeSteping : 
    
    def __init__(self):
        '''
        '''
        self.dt = None
        
    def set_timestep(self,time_step) : 
        '''
        argument 
        time_step ::: float ::: time step 
        '''
        self.dt = time_step
        
    def _calc_dt_conv(self,cfl_number,velocity_array, meshsize_array):
        '''
        arguments 
        cfl_number ::: float ::: Courant number
        velocity_arra ::: np.array(n_elem,3) ::: velocities array
        meshsize_array ::: np.array(n_elem,1) ::: array of elements size 
        returns 
        dt ::: float ::: time step 
        '''
        pass
    
    def _calc_dt_diff(self,fourier_number,diffusivity_array, meshsize_array):
        '''
        arguments 
        fourier_number ::: float ::: fourier number
        diffusivity_array ::: np.array(n_elem,1) ::: diffusivities array
        meshsize_array ::: np.array(n_elem,1) ::: array of elements size 
        returns 
        dt ::: float ::: time step 
        '''
        pass
    
class ForwardEulerScheme(TimeSteping):
    
    def __init__(self):
        '''
        '''
        super().__init__()
    
    def step(self,current_array, mesh, implicit_contribution, explicit_contribution):
        '''
        arguments 
        current_array ::: np.array(n_elem,1) ::: 
        mesh ::: numsim.meshe.mesh.Mesh :::
        implicit_contribution ::: np.array(n_elem,n_elem) :::
        explicit_contribution ::: np.array(n_elem,1) :::
        returns 
        advanced_array ::: np.array(n_elem,1) ::: 
        '''
        # f_n+1 = f_n + (dt/V)*(IMP*f_n + EXP)
        one_over_vol = 1. / mesh.elements_volumes
        implicit_contrib = np.dot(implicit_contribution,current_array)
        explicit_contrib = explicit_contribution
        advanced_array = current_array + self.dt*one_over_vol *(implicit_contrib + explicit_contrib)
        return advanced_array

class BackwardEulerScheme(TimeSteping):
    
    def __init__(self):
        '''
        '''
        super().__init__()
    
    def step(self,current_array, mesh, implicit_contribution, explicit_contribution):
        '''
        arguments 
        current_array ::: np.array(n_elem,1) ::: 
        mesh ::: numsim.meshe.mesh.Mesh :::
        implicit_contribution ::: np.array(n_elem,n_elem) :::
        explicit_contribution ::: np.array(n_elem,1) :::
        returns 
        advanced_array ::: np.array(n_elem,1) ::: 
        '''
        # (f_n+1 - f_n)*(V/dt) + (IMP*f_n+1 +EXP)= 0
        # ((V/dt) + IMP)*f_n+1 = (V/dt)*f_n - EXP 
        add_mat = np.diag(np.squeeze(mesh.elements_volumes/self.dt))
        mat = add_mat + implicit_contribution
        add_rhs = np.multiply((mesh.elements_volumes/self.dt),current_array)
        rhs = add_rhs - explicit_contribution
        advanced_array = np.linalg.solve(mat,rhs)
        return advanced_array
