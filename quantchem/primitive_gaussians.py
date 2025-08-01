import numpy as np 
import math
from scipy import special  

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
            T[(0, j)] = XP2 * T[(0, j - 1)] + (alpha1/p) * (2*alpha2*S[(0,j)])
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

def primitive_gaussian_kinetic(pg1,pg2):
    """
    Compute the overlap between two contracted BasisFunctions.
    Arguments:
        bf1 ::: BasisFunction
        bf2 ::: BasisFunction
    Returns:
        kinetic ::: float
    """
    l1 = pg1.l
    m1 = pg1.m
    n1 = pg1.n
    l2 = pg2.l
    m2 = pg2.m
    n2 = pg2.n
    #
    Stmp = obra_saika_1d_integral(l1,l2, pg1.alpha, pg2.alpha, pg1.center[0],pg2.center[0], return_full=True)
    Sij = Stmp[(l1,l2)]
    Tij  = obara_saika_1d_kinetic(l1,l2, pg1.alpha, pg2.alpha, pg1.center[0],pg2.center[0], Stmp)
    #
    Stmp = obra_saika_1d_integral(m1,m2, pg1.alpha, pg2.alpha, pg1.center[1],pg2.center[1], return_full=True)
    Skl = Stmp[(m1,m2)]
    Tkl  = obara_saika_1d_kinetic(m1,m2, pg1.alpha, pg2.alpha, pg1.center[1],pg2.center[1], Stmp)
    #
    Stmp = obra_saika_1d_integral(n1,n2, pg1.alpha, pg2.alpha, pg1.center[2],pg2.center[2], return_full=True)
    Smn = Stmp[(n1,n2)]
    Tmn  = obara_saika_1d_kinetic(n1,n2, pg1.alpha, pg2.alpha, pg1.center[2],pg2.center[2], Stmp)
    # 
    return (Tij*Skl*Smn + Sij*Tkl*Smn + Sij*Skl*Tmn) * pg1.norm_constant * pg2.norm_constant 

def basis_function_kinetic_integral(bf1,bf2) : 
    """
    Compute the overlap between two contracted BasisFunctions.
    Arguments:
        bf1 ::: BasisFunction
        bf2 ::: BasisFunction
    Returns:
        kinetic ::: float
    """
    kinetic = 0.
    for coef1, pg1 in zip(bf1.pg_coeff,bf1.pg_list):
        for coef2, pg2 in zip(bf2.pg_coeff,bf2.pg_list):
            T = primitive_gaussian_kinetic(pg1,pg2)
            kinetic += coef1*coef2*T
    return kinetic

def boys(x,n):
    if x == 0:
        return 1.0/(2*n+1)
    else:
        return special.gammainc(n+0.5,x) * special.gamma(n+0.5) * (1.0/(2*x**(n+0.5)))

def primitive_gaussian_nucat_integral(pg1,pg2,nuclei_coord, debug = False):
    '''
    argument 
    pg1 ::: PrimGauss object ::: first primitive gaussian 
    pg2 ::: PrimGauss object ::: second primitive gaussian 
    return 
    '''
    p = pg1.alpha + pg2.alpha
    P = (pg1.alpha*pg1.center+pg2.alpha*pg2.center)/p
    mu = (pg1.alpha*pg2.alpha)/(pg1.alpha+pg2.alpha)

    R12 = np.linalg.norm(pg1.center-pg2.center)
    RPC = np.linalg.norm(P-nuclei_coord)

    Kab = np.exp(-mu*R12**2)

    # Init : get theta^N_{000000} from Boys function 
    N_max = pg1.l + pg1.m + pg1.n + pg2.l + pg2.m + pg2.n
    x = p*RPC**2
    boys_pre_factor = (2*np.pi)/p*Kab
    theta = {}
    for N in range(N_max + 2) : 
        theta[ ( N , (0,0,0,0,0,0) ) ] = boys_pre_factor * boys(x, N)
    # Step 1) Get theta^N_{i00000} from OS recursion 
    XP1 = P[0] - pg1.center[0]
    XPC = P[0] - nuclei_coord[0]
    N_lim = N_max
    for i in range(1, pg1.l + 1) :
        for N in reversed(range(N_lim)):
            term1 = XP1 * theta[(N , (i-1,0,0,0,0,0))] 
            term2 = ( (i-1)/(2*p) ) * theta[(N , (i-2,0,0,0,0,0))] if i > 1 else 0.
            #term3 = ( (l)/(2*p) )   * theta[(N , (i-1,0,0,l-1,0,0))] 
            term4 = XPC * theta[(N+1 , (i-1,0,0,0,0,0))] 
            term5 = ( (i-1)/(2*p) ) * theta[(N+1 , (i-2,0,0,0,0,0))] if i > 1 else 0.
            #term6 = ( (l)/(2*p) )   * theta[(N+1 , (i-1,0,0,l-1,0,0))] 
            theta[(N , (i,0,0,0,0,0))] = term1 + term2 - term4 - term5
        N_lim -= 1
    # Step 2) Get theta^N_{000l00} from OS recursion 
    XP2 = P[0] - pg2.center[0]
    N_lim = N_max
    for l in range(1, pg2.l + 1) :
        for N in reversed(range(N_lim)):
            term1 = XP2 * theta[(N , (0,0,0,l-1,0,0))] 
            #term2 = ( (i)/(2*p) ) * theta[(N , (i-1,0,0,l-1,0,0))] 
            term3 = ( (l-1)/(2*p) )   * theta[(N , (0,0,0,l-2,0,0))] if l > 1 else 0.
            term4 = XPC * theta[(N+1 , (0,0,0,l-1,0,0))] 
            #term5 = ( (i)/(2*p) ) * theta[(N+1 , (i-1,0,0,l-1,0,0))] 
            term6 = ( (l-1)/(2*p) )   * theta[(N+1 , (0,0,0,l-2,0,0))] if l > 1 else 0.
            theta[(N , (0,0,0,l,0,0))] = term1 + term3 - term4 - term6
        N_lim -= 1
    # Step 3) Get theta^N_{i00l00} from OS recursion
    for i in range(1, pg1.l+1) : 
        for l in range(1, pg2.l + 1) :
            N_lim = N_max - (i + l) + 1
            for N in reversed(range(N_lim)):
                if debug : 
                    print('Computing ::::')
                    print('#######################')
                    print((N , (i,0,0,l,0,0)))
                    print(' Which requires ::::')
                    print('-> ', (N , (i-1,0,0,l,0,0)))
                    print('-> ', (N , (i-2,0,0,0,0,0)))
                    print('-> ', (N , (i-1,0,0,l-1,0,0)))
                    print('-> ', (N+1 , (i-1,0,0,0,0,0)))
                    print('-> ', (N+1 , (i-2,0,0,0,0,0)))
                    print('-> ', (N+1 , (i-1,0,0,l-1,0,0)))
                term1 = XP1 * theta[(N , (i-1,0,0,l,0,0))] 
                term2 = ( (i-1)/(2*p) ) * theta[(N , (i-2,0,0,l,0,0))] if i > 1 else 0.
                term3 = ( (l)/(2*p) )   * theta[(N , (i-1,0,0,l-1,0,0))] 
                term4 = XPC * theta[(N+1 , (i-1,0,0,l,0,0))] 
                term5 = ( (i-1)/(2*p) ) * theta[(N+1 , (i-2,0,0,l,0,0))] if i > 1 else 0.
                term6 = ( (l)/(2*p) )   * theta[(N+1 , (i-1,0,0,l-1,0,0))] 
                theta[(N , (i,0,0,l,0,0))] = term1 + term2 + term3 - term4 - term5 - term6
    # Step 4) Get theta^N_{ij0l00} from OS recursion
    YP1 = P[1] - pg1.center[1]
    YPC = P[1] - nuclei_coord[1]
    for j in range(1, pg1.m+1) : 
        for i in range(0, pg1.l+1) : 
            for l in range(0, pg2.l + 1) :
                N_lim = N_max - (i + j + l) + 1
                for N in reversed(range(N_lim)):
                    term1 = YP1 * theta[(N , (i,j-1,0,l,0,0))] 
                    term2 = ( (j-1)/(2*p) ) * theta[(N , (i,j-2,0,l,0,0))] if j > 1 else 0.
                    #term3 = ( (m)/(2*p) )   * theta[(N , (i,j-1,0,l,m-1,0))] if m > 1 else 0.
                    term4 = YPC * theta[(N+1 , (i,j-1,0,l,0,0))] 
                    term5 = ( (j-1)/(2*p) ) * theta[(N+1 , (i,j-2,0,l,0,0))] if j > 1 else 0.
                    #term6 = ( (m)/(2*p) )   * theta[(N+1 , (i,j-1,0,l-1,m-1,0))] if m > 1 else 0.
                    theta[(N , (i,j,0,l,0,0))] = term1 + term2 - term4 - term5 
    # Step 5) Get theta^N_{i00lm0} from OS recursion 
    YP2 = P[1] - pg2.center[1]
    for m in range(1, pg2.m+1) : 
        for i in range(0, pg1.l+1) : 
            for l in range(0, pg2.l + 1) :
                N_lim = N_max - (i + l + m) + 1
                for N in reversed(range(N_lim)):
                    term1 = YP2 * theta[(N , (i,0,0,l,m-1,0))] 
                    #term2 = ( (j)/(2*p) ) * theta[(N , (i,j-1,0,l,m-1,0))] if j > 1 else 0.
                    term3 = ( (m-1)/(2*p) )   * theta[(N , (i,0,0,l,m-2,0))] if m > 1 else 0.
                    term4 = YPC * theta[(N+1 , (i,0,0,l,m-1,0))] 
                    #term5 = ( (j)/(2*p) ) * theta[(N+1 , (i,j-1,0,l,m-1,0))] if j > 1 else 0.
                    term6 = ( (m-1)/(2*p) )   * theta[(N+1 , (i,0,0,l,m-2,0))] if m > 1 else 0.
                    theta[(N , (i,0,0,l,m,0))] = term1 + term3 - term4 - term6 
    # Step 6) Get theta^N_{ij0lm0} from OS recursion 
    for j in range(1, pg1.m+1): 
        for m in range(1, pg2.m+1): 
            for i in range(0, pg1.l+1): 
                for l in range(0, pg2.l+1): 
                    N_lim = N_max - (i + j + l + m) + 1
                    for N in reversed(range(N_lim)):
                        term1 = YP1 * theta[(N , (i,j-1,0,l,m,0))] 
                        term2 = ((j-1)/(2*p)) * theta[(N , (i,j-2,0,l,m,0))] if j > 1 else 0.
                        term3 = (m/(2*p))      * theta[(N , (i,j-1,0,l,m-1,0))] 
                        term4 = YPC * theta[(N+1 , (i,j-1,0,l,m,0))] 
                        term5 = ((j-1)/(2*p)) * theta[(N+1 , (i,j-2,0,l,m,0))] if j > 1 else 0.
                        term6 = (m/(2*p))      * theta[(N+1 , (i,j-1,0,l,m-1,0))] 
                        theta[(N , (i,j,0,l,m,0))] = term1 + term2 + term3 - term4 - term5 - term6
    # Step 7) Get theta^N_{ijklm0} from OS recursion 
    ZP1 = P[2] - pg1.center[2]
    ZPC = P[2] - nuclei_coord[2]
    for k in range(1, pg1.n + 1) :
        for i in range(0, pg1.l + 1) : 
            for j in range(0, pg1.m + 1):
                for l in range(0, pg2.l + 1):
                    for m in range(0, pg2.m + 1):
                        N_lim = N_max - (i+j+k+l+m) + 1
                        for N in reversed(range(N_lim)):
                            term1 = ZP1 * theta[(N , (i,j,k-1,l,m,0))] 
                            term2 = ((k-1)/(2*p)) * theta[(N , (i,j,k-2,l,m,0))] if k > 1 else 0.
                            #term3 = (n/(2*p))      * theta[(N , (i,j,k-1,l,m,n-1))] 
                            term4 = ZPC * theta[(N+1 , (i,j,k-1,l,m,0))] 
                            term5 = ((k-1)/(2*p)) * theta[(N+1 , (i,j,k-2,l,m,0))] if k > 1 else 0.
                            #term6 = (n/(2*p))      * theta[(N+1 , (i,j,k-1,l,m,n))] 
                            theta[(N,(i,j,k,l,m,0))] = term1 + term2 - term4 - term5 
    # Step 8) Get theta^N_{ij0lmn} from OS recursion 
    ZP2 = P[2] - pg2.center[2]
    for n in range(1, pg2.n + 1) :
        for i in range(0, pg1.l + 1) : 
            for j in range(0, pg1.m + 1):
                for l in range(0, pg2.l + 1):
                    for m in range(0, pg2.m + 1):
                        N_lim = N_max - (i+j+l+m+n) + 1
                        for N in reversed(range(N_lim)):
                            term1 = ZP2 * theta[(N , (i,j,0,l,m,n-1))] 
                            #term2 = ((k)/(2*p)) * theta[(N , (i,j,k-1,l,m,n-1))] 
                            term3 = ((n-1)/(2*p))      * theta[(N , (i,j,0,l,m,n-2))] if n > 1 else 0 
                            term4 = ZPC * theta[(N+1 , (i,j,0,l,m,n-1))] 
                            #term5 = ((k)/(2*p)) * theta[(N+1 , (i,j,k-1,l,m,n-1))] 
                            term6 = ((n-1)/(2*p))      * theta[(N+1 , (i,j,0,l,m,n-2))] if n > 1 else 0
                            theta[(N,(i,j,0,l,m,n))] = term1 + term3 - term4 - term6
    # Step 9) Get theta^N_{ijklmn} from OS recursion 
    for k in range(1, pg1.n + 1):
        for n in range(1, pg2.n + 1):
            for i in range(0, pg1.l + 1) : 
                for j in range(0, pg1.m + 1):
                    for l in range(0, pg2.l + 1):
                        for m in range(0, pg2.m + 1):
                            N_lim = N_max - (i+j+k+l+m+n) + 1
                            for N in reversed(range(N_lim)):
                                term1 = ZP1 * theta[(N , (i,j,k-1,l,m,n))] 
                                term2 = ((k-1)/(2*p)) * theta[(N , (i,j,k-2,l,m,n))] if k > 1 else 0.
                                term3 = (n/(2*p))      * theta[(N , (i,j,k-1,l,m,n-1))] 
                                term4 = ZPC * theta[(N+1 , (i,j,k-1,l,m,n))] 
                                term5 = ((k-1)/(2*p)) * theta[(N+1 , (i,j,k-2,l,m,n))] if k > 1 else 0.
                                term6 = (n/(2*p))      * theta[(N+1 , (i,j,k-1,l,m,n-1))] 
                                theta[(N, (i,j,k,l,m,n))] = term1 + term2 + term3 - term4 - term5 - term6
    return theta[(0, (pg1.l, pg1.m, pg1.n, pg2.l, pg2.m, pg2.n))]

def basis_function_nucat_integral(bf1,bf2,nuclei_coord,nuclei_charge):
    '''
    Compute the nuclear attraction integral from two contracted BasisFunctions.
    Arguments:
        bf1 ::: BasisFunction
        bf2 ::: BasisFunction
        nuclei_coord ::: list of array (3,)
        nuclear_charge ::: list of flaot ::: Zc, nucleus charge
    Returns:
        nuclear_attraction ::: float
    '''
    nuclear_attraction = 0.
    for nucleus_charge, nucleus_coord in zip(nuclei_charge, nuclei_coord) : 
        nuc_contribution = 0.
        for coef1, pg1 in zip(bf1.pg_coeff,bf1.pg_list):
            for coef2, pg2 in zip(bf2.pg_coeff,bf2.pg_list):
                T = pg1.norm_constant * pg2.norm_constant * primitive_gaussian_nucat_integral(pg1,pg2, nucleus_coord)
                nuc_contribution += coef1*coef2*T
        nuclear_attraction += nuc_contribution * (-nucleus_charge)
    return nuclear_attraction

def primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4):
    '''
    argument 
    pg1 ::: PrimGauss object ::: first primitive gaussian 
    pg2 ::: PrimGauss object ::: second primitive gaussian 
    pg3 ::: PrimGauss object ::: third primitive gaussian 
    pg4 ::: PrimGauss object ::: fourth primitive gaussian 
    return  
    Theta ::: float ::: Theta^0_{l1,l2,l3,l4}
    '''
    # init required constant
    # Contract Gaussians into A, B and compute combined Gaussian parameters
    p = pg1.alpha + pg2.alpha
    q = pg3.alpha + pg4.alpha
    P = (pg1.alpha * pg1.center + pg2.alpha * pg2.center) / p
    Q = (pg3.alpha * pg3.center + pg4.alpha * pg4.center) / q
    alpha = p * q / (p + q)  # "reduced" exponent for distance PQ
    PQ = P - Q               
    T = alpha * np.dot(PQ, PQ) 
    mu = (pg1.alpha*pg2.alpha)/(pg1.alpha + pg2.alpha)
    nu = (pg3.alpha*pg4.alpha)/(pg3.alpha + pg4.alpha)
    R12 = np.linalg.norm(pg1.center - pg2.center)
    R34 = np.linalg.norm(pg3.center - pg4.center)
    #
    N_max = pg1.l + pg2.l + pg3.l + pg4.l + pg1.m + pg2.m + pg3.m + pg4.m + pg1.n + pg2.n + pg3.n + pg4.n
    # Step 1) calculate Theta^N_{0000} from boys function 
    Theta = {}
    prefactor = (2*np.pi**(5/2))/(p*q*np.sqrt(p+q))*np.exp(-mu*R12**2)*np.exp(-nu*R34**2)
    for N in range(N_max + 1):
        Theta[(N, (0,0,0,0), (0,0,0,0), (0,0,0,0))] = prefactor * boys(T, N)
    #Step 2) calculate Theta^N_{i000,0000,0000}
    XP1 = P[0] - pg1.center[0]
    XPQ = P[0] - Q[0]
    for i in range(1, pg1.l + 1) :
        N_lim = N_max - i + 1
        for N in reversed(range(N_lim)):
            im1 = (N,(i-1,0,0,0),(0,0,0,0),(0,0,0,0))
            im2 = (N,(i-2,0,0,0),(0,0,0,0),(0,0,0,0))
            Np1_im1 = (N+1,(i-1,0,0,0),(0,0,0,0),(0,0,0,0))
            Np1_im2 = (N+1,(i-2,0,0,0),(0,0,0,0),(0,0,0,0))
            term1 = XP1 * Theta[im1]
            term2 = (alpha/p) * XPQ * Theta[Np1_im1]
            term3 = (i-1)/(2*p) * Theta[im2] if i > 1 else 0.
            term4 = (i-1)/(2*p) * (alpha/p) * Theta[Np1_im2] if i > 1 else 0.
            Theta[(N, (i,0,0,0),(0,0,0,0),(0,0,0,0))] = term1 - term2 + term3 - term4
    #Step 3) calculate Theta^N_{00k0,0000,0000}
    XQ3 = Q[0] - pg3.center[0]
    XQP = Q[0] - P[0]
    for k in range(1, pg3.l + 1) : 
        N_lim = N_max - k + 1
        for N in reversed(range(N_lim)):
            km1 = (N,(0,0,k-1,0),(0,0,0,0),(0,0,0,0))
            km2 = (N,(0,0,k-2,0),(0,0,0,0),(0,0,0,0))
            Np1_km1 = (N+1,(0,0,k-1,0),(0,0,0,0),(0,0,0,0))
            Np1_km2 = (N+1,(0,0,k-2,0),(0,0,0,0),(0,0,0,0))
            term1 = XQ3 * Theta[km1]
            term2 = (alpha/q) * XQP * Theta[Np1_km1]
            term3 = (k-1)/(2*q) * Theta[km2] if k > 1 else 0.
            term4 = (k-1)/(2*q) * (alpha/q) * Theta[Np1_km2] if k > 1 else 0.
            Theta[(N, (0,0,k,0),(0,0,0,0),(0,0,0,0))] = term1 - term2 + term3 - term4
    #Step 4) Calculate Theta^N_{0j00,0000,0000}
    XP2 = P[0] - pg2.center[0]
    for j in range(1, pg2.l + 1) : 
        N_lim = N_max - j + 1 
        for N in reversed(range(N_lim)):
            jm1 = (N,  (0,j-1,0,0), (0,0,0,0), (0,0,0,0))
            jm2 = (N,  (0,j-2,0,0), (0,0,0,0), (0,0,0,0))
            Np1_jm1 = (N+1,  (0,j-1,0,0), (0,0,0,0), (0,0,0,0))
            Np1_jm2 = (N+1,  (0,j-2,0,0), (0,0,0,0), (0,0,0,0))
            term1 = XP2 * Theta[jm1]
            term2 = XPQ * (alpha/p) * Theta[Np1_jm1]
            term3 = (j-1)/(2*p)*Theta[jm2] if j > 1 else 0.
            term4 = (j-1)/(2*p)*(alpha/p)*Theta[Np1_jm2] if j > 1 else 0.
            Theta[(N, (0,j,0,0), (0,0,0,0), (0,0,0,0))] = term1 - term2 + term3 - term4
    #Step 5) Calculate Theta^N_{000l,0000,0000}
    XQ4 = Q[0] - pg4.center[0]
    for l in range(1,pg4.l+1):
        N_lim = N_max - l + 1 
        for N in reversed(range(N_lim)): 
            lm1 = (N, (0,0,0,l-1), (0,0,0,0), (0,0,0,0))
            lm2 = (N, (0,0,0,l-2), (0,0,0,0), (0,0,0,0))
            Np1_lm1 = (N+1, (0,0,0,l-1), (0,0,0,0), (0,0,0,0))
            Np1_lm2 = (N+1, (0,0,0,l-2), (0,0,0,0), (0,0,0,0))
            term1 = XQ4 * Theta[lm1]
            term2 = XQP * (alpha/q) * Theta[Np1_lm1]
            term3 = (l-1)/(2*q)*Theta[lm2] if l > 1 else 0.
            term4 = (l-1)/(2*q)*(alpha/q)*Theta[Np1_lm2] if l > 1 else 0 
            Theta[(N, (0,0,0,l), (0,0,0,0), (0,0,0,0))] = term1 - term2 + term3 - term4
    # Step 6) Calculate Theta^N_{ij00,0000,0000}
    XP1 = P[0] - pg1.center[0]
    XPQ = P[0] - Q[0]
    for j in range(1, pg2.l + 1):
        for i in range(1, pg1.l + 1):
            N_lim = N_max - (i + j) + 1
            for N in reversed(range(N_lim)):
                im1 = (N, (i-1,j,0,0), (0,0,0,0), (0,0,0,0))
                im2 = (N, (i-2,j,0,0), (0,0,0,0), (0,0,0,0))
                Np1_im1 = (N+1, (i-1,j,0,0), (0,0,0,0), (0,0,0,0))
                Np1_im2 = (N+1, (i-2,j,0,0), (0,0,0,0), (0,0,0,0))
                jm1 = (N, (i-1,j-1,0,0), (0,0,0,0), (0,0,0,0))
                Np1_jm1 = (N+1, (i-1,j-1,0,0), (0,0,0,0), (0,0,0,0))
                #Np1_km1 = (N+1, (i,j,k-1,l), (0,0,0,0), (0,0,0,0))
                #Np1_lm1 = (N+1, (i,j,k,l-1), (0,0,0,0), (0,0,0,0))
                #
                term1 = XP1 * Theta[im1] 
                term2 = XPQ * (alpha/p) * Theta[Np1_im1] 
                term3 = (i-1)/(2*p) * Theta[im2] if i > 1 else 0 
                term4 = (i-1)/(2*p) * (alpha/p) * Theta[Np1_im2] if i > 1 else 0 
                term5 = (j)/(2*p) * Theta[jm1] if j > 0 else 0
                term6 = (j)/(2*p) * (alpha/p) * Theta[Np1_jm1] if j > 0 else 0 
                #term7 = (k)/(2*(p+q)) * Theta[Np1_km1] 
                #term8 = (l)/(2*(p+q)) * Theta[Np1_lm1] 
                Theta[(N, (i,j,0,0), (0,0,0,0), (0,0,0,0))] = term1 - term2 + term3 - term4 + term5 - term6
    return Theta[(0,(pg1.l,pg2.l,pg3.l,pg4.l), (pg1.m,pg2.m,pg3.m,pg4.m), (pg1.n,pg2.n,pg3.n,pg4.n))]

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