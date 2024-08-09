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

def tet4_basis1_dxi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0 

def tet4_basis1_deta(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 1

def tet4_basis1_dpsi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0 

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

def tet4_basis2_dxi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0 

def tet4_basis2_deta(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0 

def tet4_basis2_dpsi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 1

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

def tet4_basis3_dxi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return -1

def tet4_basis3_deta(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return -1

def tet4_basis3_dpsi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return -1

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

def tet4_basis4_dxi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 1

def tet4_basis4_deta(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0  

def tet4_basis4_dpsi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0 

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
        self.basis_functions = [tet4_basis1,
                                tet4_basis2,
                                tet4_basis3,
                                tet4_basis4]
        self.basis_functions_derivatives = [[tet4_basis1_dxi,tet4_basis1_deta,tet4_basis1_dpsi],
                                            [tet4_basis2_dxi,tet4_basis2_deta,tet4_basis2_dpsi],
                                            [tet4_basis3_dxi,tet4_basis3_deta,tet4_basis3_dpsi],
                                            [tet4_basis4_dxi,tet4_basis4_deta,tet4_basis4_dpsi]]
        # Reference nodes coordinates 
        # Formalism of Code aster used here 
        self.refnodes = np.array([[0, 1, 0],
                                  [0, 0, 1],
                                  [0, 0, 0],
                                  [1, 0, 0]])
        # 
        self.nnodes = 4 
        self.element_nodes = self.refnodes
    
    def set_element(self,element_nodes):
        '''
        arguments 
        element_nodes ::: np.array(float) (nnodes,3) ::: global coordinates of the element
        '''
        self.element_nodes = element_nodes
    
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
        #bf_arr = np.expand_dims(bf_arr, axis = 1)
        return bf_arr
    
    def get_dbf_array(self,coordinates) : 
        '''
        arguments 
        coordinates ::: float np.array (3) ::: Local coordinates 
        returns 
        dbf_arr ::: np.array(float) (nnodes,3) ::: derivatives of the basis functions
        '''
        # Reference frame coordinates
        xi = coordinates[0]
        eta = coordinates[1]
        psi = coordinates[2]
        #
        ndim = 3
        dbf_arr = np.asarray([[self.basis_functions_derivatives[i][j](xi,eta,psi) for i in range(self.nnodes)] for j in range(ndim)])
        dbf_arr = np.transpose(dbf_arr)
        return dbf_arr
    
    def calc_global_dbf_array(self,coordinates, inv_jacobian):
        '''
        arguments 
        coordinates ::: float np.array (3) ::: Local coordinates 
        inv_jacobian ::: float np.array (3,3) ::: jacobian matrix evaluated at coordinates
        returns 
        global_dbf_arr ::: np.array(float) (nnodes,3) ::: derivatives of the basis functions. 
        '''
        # Calculate basis functions derivatives, with respect to local coordinates/reference frame (xi, eta, psi). 
        local_dbf_array = self.get_dbf_array(coordinates)
        # Tensor product with the inverse jacobian, in order to get the basis functions derivatives,
        # with respect to global coordinates (x, y, z)
        print(np.shape(local_dbf_array))
        print(np.shape(inv_jacobian))
        print(local_dbf_array)
        print(inv_jacobian)
        global_dbf_arr = np.dot(local_dbf_array, inv_jacobian)
        return global_dbf_arr
    
    def mapping(self, coordinates) : 
        '''
        arguments 
        coordinates ::: float np.array (3) ::: Local coordinates 
        returns 
        global_coords ::: float np.array (3) ::: global coordinates  
        '''
        bf_arr = self.get_bf_array(coordinates)
        global_coords = np.dot(bf_arr, self.element_nodes)
        return global_coords
    
    def calc_jacobian(self, coordinates) : 
        '''
        arguments 
        coordinates ::: float np.array (3) ::: Local coordinates 
        returns 
        jacobian ::: float np.array (3,3) ::: Jacobian Matrix
        det ::: float ::: determinant of the Jacobian Matrix
        inv_jacobian ::: float np.array (3,3) ::: Inverse of the Jacobian matrix
        '''
        dbf_arr = self.get_dbf_array(coordinates)
        #jacobian = np.dot(np.transpose(dbf_arr),self.element_nodes)
        jacobian = np.dot(np.transpose(self.element_nodes),dbf_arr)
        det = np.linalg.det(jacobian)
        inv_jacobian = np.linalg.inv(jacobian)
        return jacobian, det, inv_jacobian
        
        
        
        