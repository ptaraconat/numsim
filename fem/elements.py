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
    
    def calc_stifness_integrand(self, coordinates, state_matrix):
        '''
        arguments : 
        coordinates ::: float np.array (3) ::: Local coordinates 
        state_matrix ::: float np.array () ::: 
        returns ::: 
        integrand ::: float np.array (4,4) ::: integrand for the stifness matrix computation 
        '''
        _, det_jacobian, inv_jacobian = self.calc_jacobian(coordinates)
        global_dbf = self.calc_global_dbf_array(coordinates, inv_jacobian)
        integrand = det_jacobian*np.dot(np.dot(global_dbf,state_matrix), np.transpose(global_dbf))
        return integrand
    
    def set_state_matrices(self, state_arr):
        '''
        arguments : 
        state_arr ::: float np.array (nnodes,3,3) or (3,3) ::: state matrix 
        '''
        if np.shape(state_arr) == (3,3) : 
            state_mat = np.zeros((self.nnodes,3,3))
            for i in range(self.nnodes):
                state_mat[i,:,:] = state_arr
            self.state_matrices = state_mat
        else : 
            if np.shape(state_arr) == (self.nnodes,3,3):
                self.state_matrices = state_arr
            else : 
                print('The state matrix do not have the good shape : ', np.shape(state_arr))
    
    def calc_stifness_matrix(self):
        '''
        arguments : 
        returns : 
        el_stiffness_mat ::: float np.array (nnodes,nnodes) ::: Element stiffness matrix
        '''
        el_stiffness_mat = np.zeros((self.nnodes,self.nnodes))
        for i in range(self.nnodes):
            gauss_pt_coordinates = self.refel_gauss_coords[i,:]
            gauss_pt_weight = self.refel_gauss_weights[i]
            state_matrix = self.state_matrices[i,:]
            add = self.calc_stifness_integrand(gauss_pt_coordinates, state_matrix)
            el_stiffness_mat += gauss_pt_weight*add
            print(i)
            print(add)
        return el_stiffness_mat
        
        
        
        