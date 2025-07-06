import numpy as np 
import math

def double_factorial(n):
    """Computes the double factorial (2n - 1)!!"""
    if n <= 0:
        return 1
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result

def gto_normalization_constant(l, m, n, alpha):
    """
    Computes the normalization constant for a primitive Cartesian GTO:
    g(r) = N * (x-Ax)^l (y-Ay)^m (z-Az)^n * exp(-alpha * |r - A|^2)
    """
    L = l + m + n
    num = (2 * alpha) ** (L + 1.5)
    denom = (math.pi ** 1.5) * double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)
    N_squared = num / denom
    N = math.sqrt(N_squared)
    return N

class PrimGauss : 
    '''
    '''

    def __init__(self, atom_center, alpha, l, m, n) : 
        '''
        arguments : 
        atom center ::: array like object (3,) ::: coordinate of the atom center 
        alpha ::: float ::: parameter controlling the exponential decay in the Primitive Gaussian 
        l ::: float ::: Angular momentum parameter 
        m ::: float ::: Angular momentum parameter 
        n ::: float ::: Angular momentum parameter 
        '''
        self.center = atom_center
        self.alpha = alpha
        self.l = l 
        self.m = m 
        self.n = n
        self.norm_constant = gto_normalization_constant(l, m, n, alpha)
    
    def __call__(self, point_in_space) : 
        '''
        arguments : 
        point_in_space ::: tuple of three components ::: Spatial coordinates where we want to evaluate the Basis Function 
        '''
        diffx = point_in_space[0] - self.center[0]
        diffy = point_in_space[1] - self.center[1]
        diffz = point_in_space[2] - self.center[2]

        return self.norm_constant * diffx**self.l * diffy**self.m * diffz**self.n * np.exp(-self.alpha*(diffx**2+diffy**2+diffz**2))

class BasisFunction :
    '''
    '''
    def __init__(self, pg_list, pg_coeff):
        '''
        arguments : 
        pg_list ::: list of PrimGauss objects ::: Primitive Gaussian defining the Basis function 
        pg_coeff ::: list of float ::: weights associated to Primitive Gaussian. Should sum up to 1. 
        '''
        self.pg_list = pg_list
        self.pg_coeff = pg_coeff

    def __call__(self, point_in_space): 
        '''
        arguments : 
        point_in_space ::: tuple of three components ::: Spatial coordinates where we want to evaluate the Basis Function 
        '''
        for i, (coeff, pg) in enumerate(zip(self.pg_coeff, self.pg_list)) : 
            res_tmp = coeff * pg(point_in_space)
            if i == 0 :
                res = res_tmp
            else : 
                res += res_tmp
        return res