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

def get_hermite_coefficients(l1,l2,alpha1,alpha2,coord1, coord2):
    '''
    arguments 
    l1 ::: int ::: Angular momentum number of the first primitive Gaussian
    l2 ::: int ::: Angular momentum number of the second primitive Gaussian
    alpha1 ::: float ::: Exponential decay of the first primitive Gaussian
    alpha2 ::: float ::: Exponential decay of the second primitive Gaussian
    coord1 ::: float ::: Coordinate of the first primitive Gaussian 
    coord2 ::: float ::: Coordinate of the second primitive Gaussian 
    '''
    # Define function for finding former Coeffcients, during recursive 
    # hermite coefficient computation. 
    def get(Eij, t):
        return Eij[t] if 0 <= t < len(Eij) else 0.0
    # Calculate overlap Gaussian parameters 
    mu = alpha1 + alpha2
    P = (alpha1*coord1 + alpha2*coord2)/mu
    # 
    P1 = P - coord1 
    P2 = P - coord2
    # Init the Hermite Coefficient dictionary 
    hc = {}
    hc[(0,0)] = [math.exp(- (alpha1*alpha2)/mu * (coord1 - coord2)**2)]
    # Loop over the first momentum number
    for i in range(l1+1):
        # lLoop over the second momentum number 
        for j in range(l2+1) : 
            #print('E'+str(i)+','+str(j))
            if not(i == 0 and j == 0): 
                coeffs_tmp = []
                # Loop over total number of coefficients 
                # which dis the sum of the two momentum numbers
                for t in range(i+j+1):
                    val = 0
                    if i != 0 : 
                        val += get(hc[(i-1,j)], t-1) / (2 * mu) 
                        val += get(hc[(i-1,j)], t) * P1
                        val += get(hc[(i-1,j)], t+1) * (t + 1)
                    if j != 0 : 
                        val += get(hc[(i,j-1)], t-1)  /(2 * mu)
                        val += get(hc[(i,j-1)], t) * P2
                        val += get(hc[(i,j-1)], t+1) * (t+1)
                    coeffs_tmp.append(val)
                #print(coeffs_tmp)
                hc[(i,j)] = coeffs_tmp
    return hc[(l1,l2)]


def obra_saika_1d_integral(l1,l2,alpha1,alpha2,coord1,coord2):
    '''
    arguments 
    l1 ::: int ::: Angular momentum number of the first primitive Gaussian
    l2 ::: int ::: Angular momentum number of the second primitive Gaussian
    alpha1 ::: float ::: Exponential decay of the first primitive Gaussian
    alpha2 ::: float ::: Exponential decay of the second primitive Gaussian
    coord1 ::: float ::: Coordinate of the first primitive Gaussian 
    coord2 ::: float ::: Coordinate of the second primitive Gaussian 
    returns 
    S ::: float ::: 1D part of the overlap integral. Either Sx, Sy or Sz, depending on 
    the given quantum numbers and coordinate components
    '''
    # init required constant 
    nu = alpha1+alpha2
    P = (alpha1*coord1 + alpha2*coord2)/nu
    P1 = P - coord1
    P2 = P - coord2
    #
    S = {}
    S[(0,0)] = np.sqrt(np.pi/nu) * np.exp(- (coord1- coord2) ** 2 * ((alpha1*alpha2)/nu))
    print(nu,P,P1,P2)
    print((1/(2*nu)))
    print(S[(0,0)])
    ############# S(0,0) ##############
    ########## S(1,0) S(1,0) ##########
    ##### S(2,0)  S(1,1)  S(0,2) ######
    ## S(3,0) S(2,1) S(1,2) S(0,3) ####
    ###################################
    # Loop to build left branch 
    for i in range(1,l1+1) : 
        # Loop starts at 1 since S(0,0) already defined 
        if i > 1 : 
            print(P1,S[(i-1,0)],(1/(2*nu)),(i-1), S[(i-2,0)])
            S[(i,0)] = P1*S[(i-1,0)] + ((1/(2*nu)) * ((i-1) * S[(i-2,0)]))
            print((i,0),S[(i,0)])
        else : 
            S[(i,0)] = P1*S[(i-1,0)] 
    #Loop to build right branch 
    for i in range(1,l2+1) : 
        # Loop starts at 1 since S(0,0) already defined 
        if i > 1 : 
            S[(0,i)] = P2*S[(0,i-1)] +  (1/(2*nu)) * ((i-1) * S[(0,i-2)]) 
        else : 
            S[(0,i)] = P2*S[(0, i-1)] 
    # Midlle branches 
    for i in range(1, l1 +1) :
        for j in range(1, l2 +1):
            S[(i,j)] = P1*S[(i-1,j)] + P2*S[(i,j-1)]
            if i > 1 : 
                S[(i,j)] += (1/(2*nu)) * ((i-1)*S[(i-2,j)])
            if j > 1 : 
                S[(i,j)] += (1/(2*nu)) * ((j-1)*S[(i,j-2)])
    return S[(l1,l2)]


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