import pytest
import sys as sys 
from scipy.integrate import nquad
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
    l1 = 0.4
    l2 = 0.3
    l1 = 10
    l2 = 3
    center1 = 0
    center2 = 1.1
    alpha1 = 0.4
    alpha2 = 0.6
    def integrand(x,l1,l2,alpha1,alpha2,center1,center2):
        pg1 = PrimGauss(np.array([center1,0,0]),alpha1, l1, 0, 0, normalise = False)
        pg2 = PrimGauss(np.array([center2,0,0]),alpha2, l2, 0, 1, normalise = False)
        result = (x-pg1.center[0])**pg1.l * (x-pg2.center[0])**pg2.l * np.exp(-pg1.alpha*(x-pg1.center[0])**2) * np.exp(-pg2.alpha*(x-pg2.center[0])**2)
        return result
    x = np.linspace(-10,10,1000)
    y = integrand(x,l1,l2,alpha1,alpha2,center1,center2)
    expected = np.trapz(y,x)
    # Compute the 1D integral (say, along x-axis)
    Sx = obra_saika_1d_integral(l1, l2, alpha1, alpha2, center1, center2)
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
# Second order derivative of simple 1D primitive gaussian
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

def test_prim_gauss_kin_integral2():
    # Paramètres
    l1 = 2
    l2 = 5
    alpha1 = 0.5
    alpha2 = 0.6
    coord1 = 0.0
    coord2 = 1.4

    x = np.linspace(-5,5,10000)
    y = prim_gauss_1d(x,alpha1,coord1,l1) * d2_prim_gauss_1d(x,alpha2, coord2, l2)
    expected = -0.5*np.trapz(y,x)

    S = obra_saika_1d_integral(l1, l2, alpha1, alpha2, coord1, coord2, return_full=True)
    T_os = obara_saika_1d_kinetic(l1, l2, alpha1, alpha2, coord1, coord2, S)
    print(f"Intégrale numérique = {expected:.6f}")
    print(f"Obara-Saika       = {T_os:.6f}")
    print(f"Différence        = {abs(expected - T_os):.6e}")
    assertion = abs(expected - T_os)/expected < EPSILON
    print(abs(expected - T_os)/expected)
    assert assertion

def test_basis_function_kinetic_integral(bf_fixture1,bf_fixture2):
    T11 = basis_function_kinetic_integral(bf_fixture1,bf_fixture1)
    T22 = basis_function_kinetic_integral(bf_fixture2,bf_fixture2)
    T12 = basis_function_kinetic_integral(bf_fixture1,bf_fixture2)
    T21 = basis_function_kinetic_integral(bf_fixture2,bf_fixture1)
    print(T11, T22, T12, T21)
    assertion = np.abs(T11 - 0.76003188) < EPSILON
    assertion = assertion and np.abs(T12 - 0.23645466 ) < EPSILON
    assertion = assertion and np.abs(T21 - 0.23645466 ) < EPSILON
    assertion = assertion and np.abs(T22 - 0.76003188) < EPSILON
    assert assertion

def gaussian_3d(r, center, alpha, l, m, n):
    x, y, z = r[0]-center[0], r[1]-center[1], r[2]-center[2]
    return (x**l)*(y**m)*(z**n)*np.exp(-alpha * np.dot(r - center, r - center))

def integrand(r, pg1, pg2, R_nuc):
    chi1 = gaussian_3d(r, pg1.center, pg1.alpha, pg1.l, pg1.m, pg1.n)
    chi2 = gaussian_3d(r, pg2.center, pg2.alpha, pg2.l, pg2.m, pg2.n)
    r_RC = np.linalg.norm(r - R_nuc)
    if r_RC < 1e-8:
        return 0.0  # évite division par 0
    return chi1 * chi2 / r_RC

def numeric_integral(pg1, pg2, R_nuc):
    bounds = [[-8, 8], [-8, 8], [-8, 8]]
    result, err = nquad(lambda x, y, z: integrand(np.array([x, y, z]), pg1, pg2, R_nuc),
                        bounds)
    return result

def test_nuclear_integral_X():
    l1 = 4
    m1 = 0
    n1 = 0 
    l2 = 0 
    m2 = 0 
    n2 = 0 
    alpha1 = 0.5
    alpha2 = 0.6
    center1 = np.array([0,0,0])
    center2 = np.array([1.4,0,0]) 
    pg1 = PrimGauss(center1,alpha1,l1,m1,n1)
    pg2 = PrimGauss(center2,alpha2,l2,m2,n2)
    nuc_coord = np.array([0.7,0,0])
    nucat = primitive_gaussian_nucat_integral(pg1,pg2,nuc_coord)
    print(nucat)
    #numint = numeric_integral(pg1, pg2, nuc_coord)
    numint = 5.554494040059354
    print(numint)
    assertion = np.abs(nucat - numint) < EPSILON
    assert assertion

def test_nuclear_integral_X2():
    l1 = 0
    m1 = 0
    n1 = 0 
    l2 = 4 
    m2 = 0 
    n2 = 0 
    alpha1 = 0.5
    alpha2 = 0.6
    center1 = np.array([0,0,0])
    center2 = np.array([1.4,0,0]) 
    pg1 = PrimGauss(center1,alpha1,l1,m1,n1)
    pg2 = PrimGauss(center2,alpha2,l2,m2,n2)
    nuc_coord = np.array([0.7,0,0])
    nucat = primitive_gaussian_nucat_integral(pg1,pg2,nuc_coord)
    print(nucat)
    #numint = numeric_integral(pg1, pg2, nuc_coord)
    numint = 4.296154316896881
    print(numint)
    assertion = np.abs(nucat - numint) < EPSILON
    assert assertion

def test_nuclear_integral_X3():
    l1 = 2
    m1 = 0
    n1 = 0 
    l2 = 4 
    m2 = 0 
    n2 = 0 
    alpha1 = 0.5
    alpha2 = 0.6
    center1 = np.array([0,0,0])
    center2 = np.array([1.4,0,0]) 
    pg1 = PrimGauss(center1,alpha1,l1,m1,n1)
    pg2 = PrimGauss(center2,alpha2,l2,m2,n2)
    nuc_coord = np.array([0.7,0,0])
    nucat = primitive_gaussian_nucat_integral(pg1,pg2,nuc_coord, debug = True)
    print(nucat)
    #numint = numeric_integral(pg1, pg2, nuc_coord)
    numint = 1.544416952942481
    print(numint)
    assertion = np.abs(nucat-numint) < EPSILON
    assert assertion 

def test_nuclear_integral_Y():
    l1 = 2
    m1 = 1
    n1 = 0 
    l2 = 4 
    m2 = 0 
    n2 = 0 
    alpha1 = 0.5
    alpha2 = 0.6
    center1 = np.array([0,0,0])
    center2 = np.array([1.4,0,0]) 
    pg1 = PrimGauss(center1,alpha1,l1,m1,n1)
    pg2 = PrimGauss(center2,alpha2,l2,m2,n2)
    nuc_coord = np.array([0.7,0.1,0])
    nucat = primitive_gaussian_nucat_integral(pg1,pg2,nuc_coord, debug = True)
    print(nucat)
    #numint = numeric_integral(pg1, pg2, nuc_coord)
    numint = 0.022040716823910643
    print(numint)
    assertion = np.abs(nucat-numint) < EPSILON
    assert assertion 

def test_nuclear_integral_Y2():
    l1 = 2
    m1 = 0
    n1 = 0 
    l2 = 4 
    m2 = 4
    n2 = 0 
    alpha1 = 0.5
    alpha2 = 0.6
    center1 = np.array([0,0,0])
    center2 = np.array([1.4,0,0]) 
    pg1 = PrimGauss(center1,alpha1,l1,m1,n1)
    pg2 = PrimGauss(center2,alpha2,l2,m2,n2)
    nuc_coord = np.array([0.7,0.1,0])
    nucat = primitive_gaussian_nucat_integral(pg1,pg2,nuc_coord, debug = True)
    print(nucat)
    #numint = numeric_integral(pg1, pg2, nuc_coord)
    numint = 0.7395853019635226
    print(numint)
    assertion = np.abs(nucat-numint)<EPSILON
    assert assertion

def test_nuclear_integral_Y3():
    l1 = 2
    m1 = 3
    n1 = 0 
    l2 = 4 
    m2 = 4
    n2 = 0 
    alpha1 = 0.5
    alpha2 = 0.6
    center1 = np.array([0,0,0])
    center2 = np.array([1.4,0,0]) 
    pg1 = PrimGauss(center1,alpha1,l1,m1,n1)
    pg2 = PrimGauss(center2,alpha2,l2,m2,n2)
    nuc_coord = np.array([0.7,0.1,0])
    nucat = primitive_gaussian_nucat_integral(pg1,pg2,nuc_coord, debug = False)
    print(nucat)
    #numint = numeric_integral(pg1, pg2, nuc_coord)
    numint = 0.07316591815671031
    print(numint) 
    assertion = np.abs(nucat-numint) < EPSILON
    assert assertion

def test_nuclear_integral_Z():
    l1 = 0
    m1 = 0
    n1 = 0
    l2 = 0 
    m2 = 0
    n2 = 1 
    alpha1 = 0.5
    alpha2 = 0.6
    center1 = np.array([0,0,0])
    center2 = np.array([1.4,0,0]) 
    pg1 = PrimGauss(center1,alpha1,l1,m1,n1)
    pg2 = PrimGauss(center2,alpha2,l2,m2,n2)
    nuc_coord = np.array([0.7,0.,0.5])
    nucat = primitive_gaussian_nucat_integral(pg1,pg2,nuc_coord, debug = False)
    print(nucat)
    #numint = numeric_integral(pg1, pg2, nuc_coord)
    numint = 0.47297524878811215
    print(numint)
    assertion = np.abs(nucat-numint) < EPSILON
    assert assertion

def test_nuclear_integralZ2():
    l1 = 0
    m1 = 0
    n1 = 1
    l2 = 0 
    m2 = 0
    n2 = 1 
    alpha1 = 0.5
    alpha2 = 0.6
    center1 = np.array([0,0,0])
    center2 = np.array([1.4,0,0]) 
    pg1 = PrimGauss(center1,alpha1,l1,m1,n1)
    pg2 = PrimGauss(center2,alpha2,l2,m2,n2)
    nuc_coord = np.array([0.7,0.,0.5])
    nucat = primitive_gaussian_nucat_integral(pg1,pg2,nuc_coord, debug = False)
    print(nucat)
    #numint = numeric_integral(pg1, pg2, nuc_coord)
    numint = 1.098049159515406
    print(numint)
    assertion = np.abs(nucat-numint) < EPSILON
    assert assertion

def test_nuclear_integralZ3():
    l1 = 2
    m1 = 3
    n1 = 1
    l2 = 4
    m2 = 7
    n2 = 3
    alpha1 = 0.5
    alpha2 = 0.6
    center1 = np.array([0,0,0])
    center2 = np.array([1.4,0,0]) 
    pg1 = PrimGauss(center1,alpha1,l1,m1,n1)
    pg2 = PrimGauss(center2,alpha2,l2,m2,n2)
    nuc_coord = np.array([0.7,0.,0.5])
    nucat = primitive_gaussian_nucat_integral(pg1,pg2,nuc_coord, debug = False)
    print(nucat)
    #numint = numeric_integral(pg1, pg2, nuc_coord)
    numint = 9.830324064878733
    print(numint)
    assertion = np.abs(nucat-numint) < EPSILON
    assert assertion

def test_basis_function_nucat_integral(bf_fixture1,bf_fixture2):
    nuclei_coord = [np.array([0,0,0]), np.array([0,0,1.4])]
    nuclei_charge = [1, 1]
    V11 = basis_function_nucat_integral(bf_fixture1,bf_fixture1,nuclei_coord,nuclei_charge)
    V22 = basis_function_nucat_integral(bf_fixture2,bf_fixture2,nuclei_coord,nuclei_charge)
    V12 = basis_function_nucat_integral(bf_fixture1,bf_fixture2,nuclei_coord,nuclei_charge)
    V21 = basis_function_nucat_integral(bf_fixture2,bf_fixture1,nuclei_coord,nuclei_charge)
    print(V11, V22, V12, V21)
    assertion = np.abs(V11 + 1.8804408905227796) < EPSILON
    assertion = assertion and np.abs(V22 + 1.8804408905227796) < EPSILON
    assertion = assertion and np.abs(V12 + 1.1948346220535697) < EPSILON
    assertion = assertion and np.abs(V21 + 1.1948346220535697) < EPSILON
    assert assertion

# Intégrande 6D
def elecrep_integrand(r1x, r1y, r1z, r2x, r2y, r2z, pg1, pg2, pg3, pg4):
    r1 = np.array([r1x, r1y, r1z])
    r2 = np.array([r2x, r2y, r2z])
    g1 = gaussian_3d(r1, pg1.center, pg1.alpha, pg1.l, pg1.m, pg1.n)
    g2 = gaussian_3d(r1, pg2.center, pg2.alpha, pg2.l, pg2.m, pg2.n)
    g3 = gaussian_3d(r2, pg3.center, pg3.alpha, pg3.l, pg3.m, pg3.n)
    g4 = gaussian_3d(r2, pg4.center, pg4.alpha, pg4.l, pg4.m, pg4.n)
    r12 = np.linalg.norm(r1 - r2)
    if r12 < 1e-10:
        return 0.0
    return g1 * g2 * g3 * g4 / r12

def test_elecrep_integral():
    l1 = 0
    l2 = 0
    l3 = 0
    l4 = 0
    m1 = 0 
    m2 = 0
    m3 = 0
    m4 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    alpha1 = 0.5
    alpha2 = 0.4
    alpha3 = 0.4
    alpha4 = 0.2
    coord1 = np.array([0   ,0, 0])
    coord2 = np.array([1.4 ,0, 0])
    coord3 = np.array([0.1 ,0, 0])
    coord4 = np.array([0   ,0, 0.8])
    pg1 = PrimGauss(coord1,alpha1,l1,m1,n1,normalise=True)
    pg2 = PrimGauss(coord2,alpha2,l2,m2,n2,normalise=True)
    pg3 = PrimGauss(coord3,alpha3,l3,m3,n3,normalise=True)
    pg4 = PrimGauss(coord4,alpha4,l4,m4,n4,normalise=True)
    elecrep = primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4)
    #
    # Intégration numérique (bornes raisonnables)
    #bounds = [[-5, 5]] * 6
    #args = (pg1, pg2, pg3, pg4)
    #numerical_result, error = nquad(elecrep_integrand, bounds, args=args)
    #
    #cpg1 = covert_pg(pg1)
    #cpg2 = covert_pg(pg2)
    #cpg3 = covert_pg(pg3)
    #cpg4 = covert_pg(pg4)
    #Eri_integral = ElectronRepulsion()
    expected = 30.00776693243499
    assertion = np.abs(elecrep-expected) < EPSILON
    assert assertion 

def test_elecrep_integral2():
    l1 = 14
    l2 = 0
    l3 = 0
    l4 = 0
    m1 = 0 
    m2 = 0
    m3 = 0
    m4 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    alpha1 = 0.5
    alpha2 = 0.4
    alpha3 = 0.4
    alpha4 = 0.2
    coord1 = np.array([0   ,0, 0])
    coord2 = np.array([1.4 ,0, 0])
    coord3 = np.array([0.1 ,0, 0])
    coord4 = np.array([0   ,0, 0.8])
    pg1 = PrimGauss(coord1,alpha1,l1,m1,n1,normalise=True)
    pg2 = PrimGauss(coord2,alpha2,l2,m2,n2,normalise=True)
    pg3 = PrimGauss(coord3,alpha3,l3,m3,n3,normalise=True)
    pg4 = PrimGauss(coord4,alpha4,l4,m4,n4,normalise=True)
    elecrep = primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4)
    expected = 325237.0249171882
    assertion = np.abs(elecrep-expected)<EPSILON
    assert assertion

def test_elecrep_integral3():
    l1 = 3
    l2 = 5
    l3 = 0
    l4 = 0
    m1 = 0 
    m2 = 0
    m3 = 0
    m4 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    alpha1 = 0.5
    alpha2 = 0.4
    alpha3 = 0.4
    alpha4 = 0.2
    coord1 = np.array([0   ,0.2, 0.1])
    coord2 = np.array([0.9 ,0.1, 0])
    coord3 = np.array([0.4 ,0.1, 0])
    coord4 = np.array([0   ,0, 0.8])
    pg1 = PrimGauss(coord1,alpha1,l1,m1,n1,normalise=True)
    pg2 = PrimGauss(coord2,alpha2,l2,m2,n2,normalise=True)
    pg3 = PrimGauss(coord3,alpha3,l3,m3,n3,normalise=True)
    pg4 = PrimGauss(coord4,alpha4,l4,m4,n4,normalise=True)
    elecrep = primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4)
    expected = 247.71014025859444
    assertion = np.abs(elecrep-expected) < EPSILON
    assert assertion

def test_elecrep_inegral4() : 
    l1 = 4
    l2 = 2
    l3 = 3
    l4 = 0
    m1 = 0 
    m2 = 0
    m3 = 0
    m4 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    alpha1 = 0.5
    alpha2 = 0.4
    alpha3 = 0.4
    alpha4 = 0.2
    coord1 = np.array([0   ,0.2, 0.1])
    coord2 = np.array([0.9 ,0.1, 0])
    coord3 = np.array([0.4 ,0.1, 0])
    coord4 = np.array([0   ,0, 0.8])
    pg1 = PrimGauss(coord1,alpha1,l1,m1,n1,normalise=True)
    pg2 = PrimGauss(coord2,alpha2,l2,m2,n2,normalise=True)
    pg3 = PrimGauss(coord3,alpha3,l3,m3,n3,normalise=True)
    pg4 = PrimGauss(coord4,alpha4,l4,m4,n4,normalise=True)
    elecrep = primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4)
    expected = -9.210608449875055
    assertion = np.abs(elecrep-expected) < EPSILON
    assert assertion

def test_elecrep_integral5() : 
    l1 = 3
    l2 = 2
    l3 = 1
    l4 = 5
    m1 = 0 
    m2 = 0
    m3 = 0
    m4 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    alpha1 = 0.5
    alpha2 = 0.4
    alpha3 = 0.4
    alpha4 = 0.2
    coord1 = np.array([0   ,0.2, 0.1])
    coord2 = np.array([0.9 ,0.1, 0])
    coord3 = np.array([0.4 ,0.1, 0])
    coord4 = np.array([0   ,0, 0.8])
    pg1 = PrimGauss(coord1,alpha1,l1,m1,n1,normalise=True)
    pg2 = PrimGauss(coord2,alpha2,l2,m2,n2,normalise=True)
    pg3 = PrimGauss(coord3,alpha3,l3,m3,n3,normalise=True)
    pg4 = PrimGauss(coord4,alpha4,l4,m4,n4,normalise=True)
    elecrep = primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4)
    expected = 126.1453627613943
    assertion = np.abs(elecrep-expected) < EPSILON
    assert assertion

def test_elecrep_integral6() : 
    l1 = 0
    l2 = 2
    l3 = 0
    l4 = 1
    m1 = 0
    m2 = 4
    m3 = 0
    m4 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    alpha1 = 0.5
    alpha2 = 0.4
    alpha3 = 0.4
    alpha4 = 0.2
    coord1 = np.array([0   ,0.2, 0.1])
    coord2 = np.array([0.9 ,0.1, 0])
    coord3 = np.array([0.4 ,0.1, 0])
    coord4 = np.array([0   ,0, 0.8])
    pg1 = PrimGauss(coord1,alpha1,l1,m1,n1,normalise=True)
    pg2 = PrimGauss(coord2,alpha2,l2,m2,n2,normalise=True)
    pg3 = PrimGauss(coord3,alpha3,l3,m3,n3,normalise=True)
    pg4 = PrimGauss(coord4,alpha4,l4,m4,n4,normalise=True)
    elecrep = primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4)
    expected = 3.608400391086203
    assertion = np.abs(elecrep - expected) < EPSILON
    assert assertion 

def test_elecrep_integral7():
    l1 = 1
    l2 = 1
    l3 = 1
    l4 = 1
    m1 = 1
    m2 = 1
    m3 = 1
    m4 = 4
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    alpha1 = 0.5
    alpha2 = 0.4
    alpha3 = 0.4
    alpha4 = 0.2
    coord1 = np.array([0   ,0.2, 0.1])
    coord2 = np.array([0.9 ,0.1, 0])
    coord3 = np.array([0.4 ,0.1, 0])
    coord4 = np.array([0   ,0, 0.8])
    pg1 = PrimGauss(coord1,alpha1,l1,m1,n1,normalise=True)
    pg2 = PrimGauss(coord2,alpha2,l2,m2,n2,normalise=True)
    pg3 = PrimGauss(coord3,alpha3,l3,m3,n3,normalise=True)
    pg4 = PrimGauss(coord4,alpha4,l4,m4,n4,normalise=True)
    elecrep = primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4)
    expected = 2.2093550398620616
    assertion = np.abs(elecrep-expected) < EPSILON 
    assert assertion 

def test_elecrep_integral8():
    l1 = 0
    l2 = 0
    l3 = 0
    l4 = 1
    m1 = 0
    m2 = 0
    m3 = 2
    m4 = 0
    n1 = 0
    n2 = 3
    n3 = 0
    n4 = 0
    alpha1 = 0.5
    alpha2 = 0.4
    alpha3 = 0.4
    alpha4 = 0.2
    coord1 = np.array([0   ,0.2, 0.1])
    coord2 = np.array([0.9 ,0.1, 0])
    coord3 = np.array([0.4 ,0.1, 0])
    coord4 = np.array([0   ,0, 0.8])
    pg1 = PrimGauss(coord1,alpha1,l1,m1,n1,normalise=True)
    pg2 = PrimGauss(coord2,alpha2,l2,m2,n2,normalise=True)
    pg3 = PrimGauss(coord3,alpha3,l3,m3,n3,normalise=True)
    pg4 = PrimGauss(coord4,alpha4,l4,m4,n4,normalise=True)
    elecrep = primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4)
    expected = 0.8579692541876706
    assertion = np.abs(elecrep-expected)/np.mean([elecrep,expected]) < EPSILON
    assert assertion

def test_elecrep_integral9():
    l1 = 0
    l2 = 0
    l3 = 2
    l4 = 0
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    n1 = 0
    n2 = 1
    n3 = 0
    n4 = 5
    alpha1 = 0.5
    alpha2 = 0.4
    alpha3 = 0.4
    alpha4 = 0.2
    coord1 = np.array([0   ,0.2, 0.1])
    coord2 = np.array([0.9 ,0.1, 0])
    coord3 = np.array([0.4 ,0.1, 0])
    coord4 = np.array([0   ,0, 0.8])
    pg1 = PrimGauss(coord1,alpha1,l1,m1,n1,normalise=True)
    pg2 = PrimGauss(coord2,alpha2,l2,m2,n2,normalise=True)
    pg3 = PrimGauss(coord3,alpha3,l3,m3,n3,normalise=True)
    pg4 = PrimGauss(coord4,alpha4,l4,m4,n4,normalise=True)
    elecrep = primitive_gaussian_elecrep_integral(pg1,pg2,pg3,pg4)
    expected = 16.772162465618546
    assertion = np.abs(elecrep-expected)/np.mean([elecrep,expected])*100 < EPSILON
    assert assertion 
