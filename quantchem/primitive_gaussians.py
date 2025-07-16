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
    pre_factor = (2 * alpha / math.pi) ** (3 / 4)
    numerator = (4 * alpha) ** (L / 2)
    denom = math.sqrt(double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1))
    N = pre_factor * numerator / denom
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

def obara_saika_1d_kinetic(l1,l2,alpha1,alpha2,coord1,coord2,S):
    '''
    arguments 
    l1 ::: int ::: Angular momentum number of the first primitive Gaussian
    l2 ::: int ::: Angular momentum number of the second primitive Gaussian
    alpha1 ::: float ::: Exponential decay of the first primitive Gaussian
    alpha2 ::: float ::: Exponential decay of the second primitive Gaussian
    coord1 ::: float ::: Coordinate of the first primitive Gaussian 
    coord2 ::: float ::: Coordinate of the second primitive Gaussian 
    S ::: dictionary :::
    returns 
    T :::
    '''
    # init required constant 
    p = alpha1 + alpha2
    mu = (alpha1*alpha2)/(alpha1+alpha2)
    P = (alpha1*coord1 + alpha2*coord2)/(p)
    XP1 = P - coord1 
    XP2 = P - coord2 
    X12 = coord1 - coord2
    #
    T = {}
    #print('XP1 : ', XP1)
    #print('alpha1 : ', alpha1)
    #print('p : ', p)
    #print('2 alpha1 **2 : ' , 2 * alpha1**2)
    #print('XP1 **2 : ', XP1**2)
    #print('1/(2*p) : ', 1. / (2 * p))
    T[(0,0)] = (alpha1 - 2 * alpha1**2 * (XP1**2 + 1. / (2 * p))) * S[(0,0)]
    # build T(i,0)    
    for i in range(1, l1 + 1) :  
        if i == 1 : 
            T[(i,0)] = XP1*T[( i - 1,0)] + (alpha2/p) * (2*alpha1*S[(i,0)])
        else : 
            T[(i,0)] = XP1*T[( i - 1,0)] + (i - 1) / (2 * p) * T[( i - 2, 0)] + (alpha2/p) * (2*alpha1*S[(i,0)] - (i - 1)*S[(i - 2,0)])
    # build T(0,j)
    for j in range(1, l2 + 1):
        if j == 1:
            T[(0, j)] = XP2 * T[(0, j - 1)]
        else:
            T[(0, j)] = XP2 * T[(0, j - 1)] + (j - 1) / (2 * p) * T[(0, j - 2)] + (alpha1/p) * (2*alpha2*S[(0, j)] - (j - 1)*S[(0,j-2)])
    # build T(i,j)
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            term1 = XP1 * T[(i - 1, j)]
            term2 = (i - 1) / (2 * p) * T[(i - 2, j)] if i > 1 else 0.0
            term3 = j / (2 * p) * T[(i - 1, j - 1)]
            term4 = alpha2/p * 2*alpha1*S[(i, j)] 
            term5 = - alpha2/p * (i - 1) * S[(i - 2, j)] if i > 1 else 0.0
            T[(i, j)] = term1 + term2 + term3 + term4 + term5
    #    
    return T[(l1, l2)]

def obra_saika_1d_integral(l1,l2,alpha1,alpha2,coord1,coord2,return_full = False):
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
    p = alpha1 + alpha2
    mu = (alpha1*alpha2)/(alpha1+alpha2)
    P = (alpha1*coord1 + alpha2*coord2)/(p)
    XP1 = P - coord1 
    XP2 = P - coord2 
    X12 = coord1 - coord2
    Kab = np.exp(-mu*X12**2)
    #
    S = {}
    S[(0,0)] = np.sqrt(np.pi/p) * np.exp(-mu*X12**2)  #np.exp(- (coord1- coord2) ** 2 * ((alpha1*alpha2)/nu))
    ############# S(0,0) ##############
    ########## S(1,0) S(1,0) ##########
    ##### S(2,0)  S(1,1)  S(0,2) ######
    ## S(3,0) S(2,1) S(1,2) S(0,3) ####
    ###################################
    # Generate set of (i,0)
    for i in range(1,l1+1) : 
        # Loop starts at 1 since S(0,0) already defined 
        if i > 1 : 
            S[(i,0)] = XP1*S[(i-1,0)] + ((1/(2*p)) * ((i-1) * S[(i-2,0)])) #+ ((1/(2*p)) * ((j-1) * S[(i,j-2)]))
        else : 
            S[(i,0)] = XP1*S[(i-1,0)] 

    for j in range(1, l2 + 1):
        if j == 1:
            S[(0, j)] = XP2 * S[(0, j - 1)]
        else:
            S[(0, j)] = XP2 * S[(0, j - 1)] + (j - 1) / (2 * p) * S[(0, j - 2)]

    # Fill the rest of the table (i, j)
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            term1 = XP1 * S[(i - 1, j)]
            term2 = (i - 1) / (2 * p) * S[(i - 2, j)] if i > 1 else 0.0
            term3 = j / (2 * p) * S[(i - 1, j - 1)]
            S[(i, j)] = term1 + term2 + term3
    if return_full : 
        return S
    else : 
        return S[(l1,l2)]

def primitive_gaussians_overlapp(pg1,pg2):
    '''
    argument 
    pg1 ::: PrimGauss object ::: first primitive gaussian 
    pg2 ::: PrimGauss object ::: second primitive gaussian 
    return 
    overlap ::: float ::: overlap integrals between pg1 and pg2
    '''
    Sx = obra_saika_1d_integral(pg1.l,pg2.l,pg1.alpha,pg2.alpha,pg1.center[0],pg2.center[0])
    Sy = obra_saika_1d_integral(pg1.m,pg2.m,pg1.alpha,pg2.alpha,pg1.center[1],pg2.center[1])
    Sz = obra_saika_1d_integral(pg1.n,pg2.n,pg1.alpha,pg2.alpha,pg1.center[2],pg2.center[2])
    return  Sx * Sy * Sz * pg1.norm_constant * pg2.norm_constant 

def basis_function_overlap(bf1, bf2):
    """
    Compute the overlap between two contracted BasisFunctions.
    Arguments:
        bf1 ::: BasisFunction
        bf2 ::: BasisFunction
    Returns:
        overlap ::: float
    """
    total_overlap = 0.0
    for coeff1, pg1 in zip(bf1.pg_coeff, bf1.pg_list):
        for coeff2, pg2 in zip(bf2.pg_coeff, bf2.pg_list):
            S = primitive_gaussians_overlapp(pg1, pg2)
            total_overlap += coeff1 * coeff2 * S
    return total_overlap

class PrimGauss : 
    '''
    '''

    def __init__(self, atom_center, alpha, l, m, n, normalise = True) : 
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
        if normalise : 
            self.norm_constant = gto_normalization_constant(l, m, n, alpha)
        else : 
            self.norm_constant = 1
    
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