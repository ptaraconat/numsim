import pytest
import sys as sys 
sys.path.append('.')
from quantchem.primitive_gaussians import *

@pytest.fixture()
def pg_fixture(): 
    pg = PrimGauss(np.array([0,0,0]),0.5, 0, 0, 0)
    return pg

pg1 = PrimGauss(np.array([0,0,0]),0.3425250914E+01, 0, 0, 0)
pg2 = PrimGauss(np.array([0,0,0]),0.6239137298E+00, 0, 0, 0)
pg3 = PrimGauss(np.array([0,0,0]),0.1688554040E+00, 0, 0, 0)
coeff = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
pg_list = [pg1, pg2, pg3]
bf = BasisFunction(pg_list, coeff)

zs = np.linspace(0,10,100)
points = (0,0,zs)

res1 = pg1(points)
res2 = pg2(points)
res3 = pg3(points)
bf_res = bf(points)

#import matplotlib.pyplot as plt 
#plt.plot(zs,res1,'b-')
#plt.plot(zs,res2,'r-')
#plt.plot(zs,res3,'g-')
#plt.plot(zs, bf_res, 'k--o')
#plt.grid()
#plt.show()
#plt.close()
