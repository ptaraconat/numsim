import pytest
import sys as sys 
sys.path.append('.')
from quantchem.primitive_gaussians import *

EPSILON = 1e-6

@pytest.fixture()
def pg_fixture(): 
    pg = PrimGauss(np.array([0,0,0]),0.5, 0, 0, 0)
    return pg

@pytest.fixture()
def bf_fixture():
    pg1 = PrimGauss(np.array([0,0,0]),0.5, 0, 0, 0)
    pg2 = PrimGauss(np.array([0,0,0]),0.5, 0, 0, 0)
    coeff = [0.5,0.5]
    pg_list = [pg1, pg2]
    bf = BasisFunction(pg_list, coeff)
    return bf 

@pytest.fixture()
def bf_fixture1():
    coeff = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
    pg_list = [PrimGauss(np.array([0,0,0]),0.3425250914E+01, 0, 0, 0, normalise = True),
            PrimGauss(np.array([0,0,0]),0.6239137298E+00, 0, 0, 0, normalise = True),
            PrimGauss(np.array([0,0,0]),0.1688554040E+00, 0, 0, 0, normalise = True)]
    bf1 = BasisFunction(pg_list, coeff)
    return bf1

@pytest.fixture()
def bf_fixture2():
    coeff = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
    pg_list = [PrimGauss(np.array([0,0,1.4]),0.3425250914E+01, 0, 0, 0, normalise = True),
            PrimGauss(np.array([0,0,1.4]),0.6239137298E+00, 0, 0, 0, normalise = True),
            PrimGauss(np.array([0,0,1.4]),0.1688554040E+00, 0, 0, 0, normalise = True)]
    bf2 = BasisFunction(pg_list, coeff)
    return bf2

def test_primitive_gaussian(pg_fixture):
    res = pg_fixture((0,0,0))
    print(pg_fixture.norm_constant)
    print(res)
    assertion = np.abs(res - 0.423777) < EPSILON 
    assert assertion

def test_basis_function(bf_fixture):
    res = bf_fixture((0,0,0))
    print(res)
    assertion = np.abs(res - 0.423777) < EPSILON 
    assert assertion

def test_hermite_coef():
    res = get_hermite_coefficients(2, 2, 0.5, 0.4, 0.0, 1.)
    res = np.array(res)
    expected_result = np.array([3.456315081634507, -0.8421106660762044, 4.228860083991372, 
                                -0.18306753610352267, 0.45766884025880655])
    print(res)
    assertion = np.all(np.abs(res-expected_result) < EPSILON)
    assert assertion 

def test_1d_os_integral():
    # Compare with analytical value for two 1D s-Gaussians
    alpha1 = 0.5
    alpha2 = 0.5
    x1 = 0.0
    x2 = 0.0
    S00 = obra_saika_1d_integral(0, 0, alpha1, alpha2, x1, x2)
    analytical = np.sqrt(np.pi / (alpha1 + alpha2))
    print(S00, analytical)
    assertion = S00 == analytical
    assert assertion

def test_1d_os_integral2():
    # Parameters
    l1 = 2       # p orbital on center A
    l2 = 2       # d orbital on center B
    alpha1 = 0.6  # Gaussian exponent for center A
    alpha2 = 0.7  # Gaussian exponent for center B
    coord1 = 0.0  # center A
    coord2 = 1.0  # center B
    # Compute the 1D integral (say, along x-axis)
    Sx = obra_saika_1d_integral(l1, l2, alpha1, alpha2, coord1, coord2)
    print(Sx)
    expected = 0.3563525572480466 # derived by hand 
    assertion = np.abs(Sx-expected) < EPSILON
    assert assertion

def test_1d_os_integral3():
    def integrand(x):
        pg1 = PrimGauss(np.array([0,0,0]),0.4, 2, 0, 0, normalise = False)
        pg2 = PrimGauss(np.array([1.1,0,0]),0.7, 3, 0, 1, normalise = False)
        result = (x-pg1.center[0])**pg1.l * (x-pg2.center[0])**pg2.l * np.exp(-pg1.alpha*(x-pg1.center[0])**2) * np.exp(-pg2.alpha*(x-pg2.center[0])**2)
        return result
    x = np.linspace(-10,10,1000)
    y = integrand(x)
    expected = np.trapz(y,x)
    # Compute the 1D integral (say, along x-axis)
    Sx = obra_saika_1d_integral(2, 3, 0.4, 0.7, 0, 1.1)
    print(Sx)
    print(expected)
    assertion = np.abs(Sx-expected) < EPSILON
    assert assertion

def test_prim_gauss_overlapp():
    pg1 = PrimGauss(np.array([0,0,0]),0.5, 3, 2, 7, normalise = True)
    pg2 = PrimGauss(np.array([0,0,0]),0.5, 3, 2, 7, normalise = True)
    overlap = primitive_gaussians_overlapp(pg1,pg2)
    print(overlap)
    expected = 1
    assertion = np.abs(overlap-expected) < EPSILON
    assert assertion 

def test_bf_overlap(bf_fixture1,bf_fixture2):
    S11 = overlap = basis_function_overlap(bf_fixture1,bf_fixture1)
    S12 = overlap = basis_function_overlap(bf_fixture1,bf_fixture2)
    S21 = overlap = basis_function_overlap(bf_fixture2,bf_fixture1)
    S22 = overlap = basis_function_overlap(bf_fixture2,bf_fixture2)
    print(S11,S12,S22,S21)
    assertion = np.abs(S11 - 1.) < EPSILON
    assertion = assertion and np.abs(S12 - 0.65931821) < EPSILON
    assertion = assertion and np.abs(S21 - 0.65931821) < EPSILON
    assertion = assertion and np.abs(S22 - 1.) < EPSILON
    print(assertion) 
    assert assertion 

def test_prim_gauss_kin_integral():
    l1 = 0
    l2 = 0
    alpha1 = 0.5
    alpha2 = 0.6
    coord1 = 0 
    coord2 = 1.4
    S = obra_saika_1d_integral(l1,l2,alpha1,alpha2,coord1,coord2,return_full = True)
    res = obara_saika_1d_kinetic(l1,l2,alpha1,alpha2,coord1,coord2,S)

    assertion = np.abs(res - -0.018658) < EPSILON
    assert assertion





def d2_prim_gauss_1d(x, alpha, A, l):
    r = x - A
    base = np.exp(-alpha * r**2)
    
    term1 = l * (l - 1) * r**(l - 2) if l >= 2 else 0.0
    term2 = -2 * alpha * (2 * l + 1) * r**l if l >= 0 else 0.0
    term3 = 4 * alpha**2 * r**(l + 2)
    
    return base * (term1 + term2 + term3)

# Primitive gaussienne simple (l=0)
def prim_gauss_1d(x, alpha, A, l):
    return (x - A)**l * np.exp(-alpha * (x - A)**2)

# Paramètres
l1 = 0
l2 = 1
alpha1 = 0.5
alpha2 = 0.6
coord1 = 0.0
coord2 = 1.4

x = np.linspace(-5,5,10000)
y = -0.5*prim_gauss_1d(x,alpha1,coord1,l1) * d2_prim_gauss_1d(x,alpha2, coord2, l2)
expected = np.trapz(y,x)

S = obra_saika_1d_integral(l1, l2, alpha1, alpha2, coord1, coord2, return_full=True)
T_os = obara_saika_1d_kinetic(l1, l2, alpha1, alpha2, coord1, coord2, S)
print(f"Intégrale numérique = {expected:.6f}")
print(f"Obara-Saika       = {T_os:.6f}")
print(f"Différence        = {abs(expected - T_os):.6e}")
print(f"Ratio       = {expected/T_os:.6e}")


#import matplotlib.pyplot as plt 
#plt.plot(x,y)
#plt.show()