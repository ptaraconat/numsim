import numpy as np 

def tet4_basis1(xi,eta,psi) : 
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the basis function
    '''
    return eta

def tet4_basis2(xi,eta,psi) : 
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the basis function
    '''
    return psi

def tet4_basis3(xi,eta,psi) : 
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the basis function
    '''
    return 1 - xi - eta - psi

def tet4_basis4(xi,eta,psi) : 
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the basis function
    '''
    return xi 

class Tet4 : 
    '''
    '''
    def __init__(self) : 
        '''
        arguments : 
        
        '''
        # Gauss quadrature setting
        # Source : Code Aster documentation 
        a = (5 - np.sqrt(5))/(20)
        b = (5+3*np.sqrt(5))/(20)
        self.refel_gauss_coords = np.array([[a,a,a],
                                            [a,a,b],
                                            [a,b,a],
                                            [b,a,a]])
        self.refel_gauss_weights = (1/24)*np.array([1.,1.,1.,1.])
        # basis function 
        # The same formalism as Code Aster has been used 
        self.bfunc1 = tet4_basis1
        self.bfunc2 = tet4_basis2
        self.bfunc3 = tet4_basis3
        self.bfunc4 = tet4_basis4
        self.basis_functions = [tet4_basis1,
                                tet4_basis2,
                                tet4_basis3,
                                tet4_basis4]
        # Reference nodes coordinates 
        # Formalism of Code aster used here 
        self.refnode1 = np.array([0, 1, 0])
        self.refnode2 = np.array([0, 0, 1])
        self.refnode3 = np.array([0, 0, 0])
        self.refnode4 = np.array([1, 0, 0])
        self.refnodes = np.array([[0, 1, 0],
                                  [0, 0, 1],
                                  [0, 0, 0],
                                  [1, 0, 0]])
        # 
        self.nnodes = 4 
        self.element_nodes = None
    
    def get_bf_array(self,coordinates) : 
        '''
        arguments 
        coordinates ::: float np.array (3) ::: Local coordinates 
        returns 
        bf_arr ::: np.array(float) (nnodes) ::: basis function values  
        '''
        # Reference frame coordinates
        xi = coordinates[0]
        eta = coordinates[1]
        psi = coordinates[2]
        #
        bf_arr = np.asarray([self.basis_functions[i](xi,eta,psi) for i in range(self.nnodes)])
        print(bf_arr)
        print(np.shape(bf_arr))
        return bf_arr
    
    def mapping(self, coordinates) : 
        '''
        arguments 
        coordinates ::: float np.array (3) ::: Local coordinates 
        returns 
        global_coords ::: float np.array (3) ::: global coordinates  
        '''
        pass
        
        
        
        