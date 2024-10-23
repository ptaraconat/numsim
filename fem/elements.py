import numpy as np 

############## TRI3 Basis functions #############
def tri3_basis1(xi,eta,psi):
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the basis function
    '''
    return 1 - xi - eta

def tri3_basis1_dxi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return -1

def tri3_basis1_deta(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return -1

def tri3_basis1_dpsi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    # comes from the coursera course : psi = 1-xi-eta
    return 0

###########################
def tri3_basis2(xi,eta,psi):
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the basis function
    '''
    return xi

def tri3_basis2_dxi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 1

def tri3_basis2_deta(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0

def tri3_basis2_dpsi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0

#############################
def tri3_basis3(xi,eta,psi):
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the basis function
    '''
    return eta

def tri3_basis3_dxi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0

def tri3_basis3_deta(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 1

def tri3_basis3_dpsi(xi,eta,psi) :
    '''
    arguments :
    xi ::: float ::: First Coordinate in the parent element frame
    eta ::: float ::: Second Coordinate in the parent element frame
    psi ::: float ::: Third Coordinate in the parent element frame
    returns : 
    ::: float ::: value of the derivative of the basis function
    ''' 
    return 0

############## TET4 Basis functions #############
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

#############################
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

#############################
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


class FemConstructor():
    '''
    '''
    def __init__(self):
        '''
        '''
        #
        self.ngauss_pt = None 
        self.refel_gauss_coords = None 
        self.refel_gauss_weights = None 
        #
        self.basis_functions = None 
        self.basis_functions_derivatives = None 
        #
        self.refnodes = None 
        self.nnodes = None 
        self.element_nodes = None 
    
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
        ndim = self.eldim
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
    
    def calc_stifness_matrix(self):
        '''
        arguments : 
        returns : 
        el_stiffness_mat ::: float np.array (nnodes*vardim,nnodes*vardim) ::: Element stiffness matrix
        '''
        el_stiffness_mat = np.zeros((self.nnodes*self.vardim,self.nnodes*self.vardim))
        for i in range(self.ngauss_pt):
            gauss_pt_coordinates = self.refel_gauss_coords[i,:]
            gauss_pt_weight = self.refel_gauss_weights[i]
            state_matrix = self.state_matrices[i,:]
            add = self.calc_stifness_integrand(gauss_pt_coordinates, state_matrix)
            el_stiffness_mat += gauss_pt_weight*add
        return el_stiffness_mat
    
    def calc_mass_matrix(self):
        '''
        arguments : 
        returns : 
        el_mass_mat ::: float np.array (nnodes*vardim,nnodes*vardim) ::: Element mass matrix
        '''
        el_mass_mat = np.zeros((self.nnodes*self.vardim,self.nnodes*self.vardim))
        for i in range(self.ngauss_pt):
            gauss_pt_coordinates = self.refel_gauss_coords[i,:]
            gauss_pt_weight = self.refel_gauss_weights[i]
            rho_value = self.rho_values[i]
            add = self.calc_massmat_integrand(gauss_pt_coordinates, rho_value)
            el_mass_mat += gauss_pt_weight*add
        return el_mass_mat
    
    def calc_global_stiffness_matrix(self, mesh, state_data):
        '''
        arguments 
        mesh ::: meshe.mesh
        state_data ::: str ::: name of the state data 
        returns 
        global_stiffness :: 
        '''
        nnodes = np.size(mesh.nodes,0)
        vdim = self.vardim
        nelem = np.size(mesh.elements,0)
        global_stiffness = np.zeros((nnodes*vdim,nnodes*vdim))
        for i in range(nelem) : 
            element = mesh.elements[i,:]
            element_nodes = mesh.nodes[element]
            state_matrix = mesh.nodes_data[state_data][element]
            self.set_state_matrices(state_matrix)
            self.set_element(element_nodes)
            local_stiffness = self.calc_stifness_matrix()
            connectivity = self.get_connectivity(element)
            global_stiffness[np.ix_(connectivity, connectivity)] += local_stiffness
        return global_stiffness
    
    def interpolate_rho_values(self, rho_values): 
        '''
        arguments : 
        rho_value ::: float np.array(nnodes) ::: rho values at nodes 
        returns 
        rho_array ::: float np.array(nnodes) ::: rho values at gauss points
        '''
        rho_array = np.zeros((self.ngauss_pt))
        for i in range(self.ngauss_pt): 
            gauss_pt_coordinates = self.refel_gauss_coords[i,:]
            basis_functions = self.get_bf_array(gauss_pt_coordinates)
            #basis_functions = np.expand_dims(basis_functions,axis = (1,2))
            gauss_pt_rho= np.sum(basis_functions*rho_values,axis = 0)
            rho_array[i] = gauss_pt_rho
        return rho_array

    def set_rho_values(self, rho_value):
        '''
        arguments : 
        rho_value ::: float np.array(nnodes) or float ::: rho values
        '''
        if np.shape(rho_value) == () : 
            rho_array = np.zeros((self.nnodes))
            for i in range(self.nnodes):
                rho_array[i] = rho_value
            self.rho_values = rho_array
        else : 
            if np.shape(rho_value) == (self.nnodes,):
                rho_array = self.interpolate_rho_values(rho_value)
                self.rho_values = rho_array
            else : 
                print('The state matrix do not have the good shape : ', np.shape(rho_value))

    def calc_global_mass_matrix(self, mesh, rho_data):
        '''
        arguments 
        mesh ::: meshe.mesh
        rho_data ::: str ::: name of rho data 
        returns 
        global_mass :: 
        '''
        nnodes = np.size(mesh.nodes,0)
        vdim = self.vardim
        nelem = np.size(mesh.elements,0)
        global_mass = np.zeros((nnodes*vdim,nnodes*vdim))
        for i in range(nelem) : 
            element = mesh.elements[i,:]
            element_nodes = mesh.nodes[element]
            rho_values = mesh.nodes_data[rho_data][element]
            #state_matrix = mesh.nodes_data[state_data][element]
            self.set_rho_values(rho_values)
            self.set_element(element_nodes)
            local_mass = self.calc_mass_matrix()
            connectivity = self.get_connectivity(element)
            #print(local_mass)
            global_mass[np.ix_(connectivity, connectivity)] += local_mass
        return global_mass

class Tri3(FemConstructor):
    '''
    '''
    def __init__(self,variable_dimension = 1):
        '''
        '''
        # Gauss points 
        self.ngauss_pt = 3
        self.refel_gauss_coords = np.array([[1/6,1/6,0],
                                            [2/3,1/6,0],
                                            [1/6,2/3,0]])
        self.refel_gauss_weights = (1/6)*np.array([1.,1.,1.])
        # basis function 
        # The same formalism as Code Aster has been used 
        self.basis_functions = [tri3_basis1,
                                tri3_basis2,
                                tri3_basis3]
        #self.basis_functions_derivatives = [[tri3_basis1_dxi,tri3_basis1_deta,tri3_basis1_dpsi],
        #                                    [tri3_basis2_dxi,tri3_basis2_deta,tri3_basis2_dpsi],
        #                                    [tri3_basis3_dxi,tri3_basis3_deta,tri3_basis3_dpsi]]
        self.basis_functions_derivatives = [[tri3_basis1_dxi,tri3_basis1_deta],
                                            [tri3_basis2_dxi,tri3_basis2_deta],
                                            [tri3_basis3_dxi,tri3_basis3_deta]]
        # Reference nodes coordinates 
        # Formalism of Code aster used here 
        self.refnodes = np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 1, 0]])
        # 
        self.nnodes = 3
        self.eldim = 2
        self.vardim = variable_dimension
        # init element_nodes 
        self.element_nodes = self.refnodes
    
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
        jacobian = np.dot(np.transpose(self.element_nodes),dbf_arr)
        #
        a1 = jacobian[:,0]
        a2 = jacobian[:,1]
        det = np.linalg.norm(np.cross(a1,a2))
        #
        t1 = a1/np.linalg.norm(a1)
        t2 = a2/np.linalg.norm(a2)
        jacobian2 = np.array([[a1.dot(t1),a2.dot(t1)],
                              [a1.dot(t2),a2.dot(t2)]])
        inv_jacobian = np.linalg.inv(jacobian2)
        #
        return jacobian, det, inv_jacobian
    
    def calc_bndflux_integrand(self, coordinates, flux):
        '''
        arguments : 
        coordinates ::: float np.array (3) ::: Local coordinates
        flux ::: float dimension = self.vardim ::: Boundary flux (Neumann BC)  
        returns ::: 
        integrand ::: float np.array (4,4) ::: integrand for the stifness matrix computation 
        '''
        _, det_jacobian, _ = self.calc_jacobian(coordinates)
        bf_array = self.get_bf_array(coordinates)
        bf_array = np.expand_dims(bf_array, axis = 1)
        integrand = det_jacobian*bf_array*flux
        integrand = np.reshape(integrand,(self.nnodes*self.vardim,1))
        #integrand = np.reshape(np.transpose(integrand),(self.nnodes*self.vardim))
        return integrand 
    
    def interpolate_fluxes(self, fluxes): 
        '''
        arguments : 
        fluxes ::: float np.array (nnodes,3)  ::: state matrix 
        returns : 
        fluxes_arr ::: float np.array (ngauss_points,3) ::: fluxes interpolated at gauss point
        '''
        fluxes_arr = np.zeros((self.ngauss_pt,self.vardim))
        for i in range(self.ngauss_pt): 
            gauss_pt_coordinates = self.refel_gauss_coords[i,:]
            basis_functions = self.get_bf_array(gauss_pt_coordinates)
            basis_functions = np.expand_dims(basis_functions,axis = (1))
            gauss_pt_flux = np.sum(basis_functions*fluxes,axis = 0)
            fluxes_arr[i,:] = gauss_pt_flux
        return fluxes_arr
    
    def set_fluxes(self, fluxes):
        '''
        arguments : 
        fluxes ::: float np.array (nnodes,3)  or (3)::: fluxes
        '''
        if self.vardim == 3 : 
            if np.shape(fluxes) == (3,) : 
                fluxes_arr = np.zeros((self.nnodes,self.vardim))
                for i in range(self.nnodes):
                    fluxes_arr[i,:] = fluxes
                self.gauss_point_fluxes = fluxes_arr
            else : 
                if np.shape(fluxes) == (self.nnodes,3):
                    fluxes_arr = self.interpolate_fluxes(fluxes)
                    self.gauss_point_fluxes = fluxes_arr
                else : 
                    print('The state matrix do not have the good shape : ', np.shape(fluxes))
        if self.vardim == 1 : 
            if np.shape(fluxes) == () : 
                fluxes_arr = np.zeros((self.nnodes,self.vardim))
                for i in range(self.nnodes):
                    fluxes_arr[i,:] = fluxes
                self.gauss_point_fluxes = fluxes_arr
            else : 
                if np.shape(fluxes) == (self.nnodes,1):
                    fluxes_arr = self.interpolate_fluxes(fluxes)
                    self.gauss_point_fluxes = fluxes_arr
                else : 
                    print('The state matrix do not have the good shape : ', np.shape(fluxes))

class Tet4Scalar(FemConstructor) : 
    '''
    '''
    def __init__(self) : 
        '''
        arguments : 
        
        '''
        # Gauss quadrature setting
        # Source : Code Aster documentation 
        self.ngauss_pt = 4
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
        self.vardim = 1
        self.eldim = 3
        self.element_nodes = self.refnodes
    
    def get_connectivity(self, element):
        '''
        argument 
        element ::: np.array(int) (nelnodes) ::: element connectivity
        return 
        mat_conn ::: np.array(int) (nelnodes*vardim,) ::: local connectivity 
        to gloabal connectivity for global matrix-vector weak form
        '''
        return element
    
    def interpolate_state_mat(self, state_arr): 
        '''
        arguments : 
        state_arr ::: float np.array (nnodes,3,3) or (3,3) ::: state matrix 
        '''
        state_matrix = np.zeros((self.ngauss_pt,3,3))
        for i in range(self.ngauss_pt): 
            gauss_pt_coordinates = self.refel_gauss_coords[i,:]
            basis_functions = self.get_bf_array(gauss_pt_coordinates)
            basis_functions = np.expand_dims(basis_functions,axis = (1,2))
            gauss_pt_state_matrix = np.sum(basis_functions*state_arr,axis = 0)
            state_matrix[i,:,:] = gauss_pt_state_matrix
        return state_matrix
    
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
                state_mat = self.interpolate_state_mat(state_arr)
                self.state_matrices = state_mat
            else : 
                print('The state matrix do not have the good shape : ', np.shape(state_arr))

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

    def calc_massmat_integrand(self, coordinates, rho_value):
        '''
        arguments : 
        coordinates ::: float np.array (3) ::: Local coordinates 
        rho_value ::: float  ::: 
        returns ::: 
        integrand ::: float np.array (4,4) ::: integrand for the stifness matrix computation 
        '''
        _, det_jacobian, _ = self.calc_jacobian(coordinates)
        bf_array = self.get_bf_array(coordinates)
        bf_array = np.expand_dims(bf_array, axis = 1)
        integrand = det_jacobian*bf_array*rho_value*np.transpose(bf_array)
        return integrand 
        
class Tet4Vector(FemConstructor) : 
    '''
    '''
    def __init__(self) : 
        '''
        arguments : 
        
        '''
        # Gauss quadrature setting
        # Source : Code Aster documentation 
        self.ngauss_pt = 4
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
        self.refcentroid = np.mean(self.refnodes, axis = 1)
        # 
        self.nnodes = 4 
        self.vardim = 3
        self.eldim = 3
        self.element_nodes = self.refnodes 

    def calc_global_dbf_array_symgrad(self,coordinates, inv_jacobian):
        '''
        arguments 
        coordinates ::: float np.array (3) ::: Local coordinates 
        inv_jacobian ::: float np.array (3,3) ::: jacobian matrix evaluated at coordinates
        returns 
        global_dbf_arr ::: np.array(float) (nnodes*ndim,6) ::: derivatives of the basis functions, 
        arrange for assessing a symmetric gradient tensor  
        '''
        # Calculate basis functions derivatives, with respect to local coordinates/reference frame (xi, eta, psi). 
        local_dbf_array = self.get_dbf_array(coordinates)
        # Tensor product with the inverse jacobian, in order to get the basis functions derivatives,
        # with respect to global coordinates (x, y, z)
        scalar_dbf_arr = np.dot(local_dbf_array, inv_jacobian)
        #
        dn1dx = scalar_dbf_arr[0,0]
        dn2dx = scalar_dbf_arr[1,0]
        dn3dx = scalar_dbf_arr[2,0]
        dn4dx = scalar_dbf_arr[3,0]
        dn1dy = scalar_dbf_arr[0,1]
        dn2dy = scalar_dbf_arr[1,1]
        dn3dy = scalar_dbf_arr[2,1]
        dn4dy = scalar_dbf_arr[3,1]
        dn1dz = scalar_dbf_arr[0,2]
        dn2dz = scalar_dbf_arr[1,2]
        dn3dz = scalar_dbf_arr[2,2]
        dn4dz = scalar_dbf_arr[3,2]
        #
        row1 = np.array([dn1dx, 0, 0, dn2dx, 0, 0, dn3dx, 0, 0, dn4dx, 0, 0])
        row2 = np.array([0, dn1dy, 0, 0, dn2dy, 0, 0, dn3dy, 0, 0, dn4dy, 0])
        row3 = np.array([0, 0, dn1dz, 0, 0, dn2dz, 0, 0, dn3dz, 0, 0, dn4dz])
        row4 = 0.5*np.array([dn1dy, dn1dx, 0, dn2dy, dn2dx, 0, dn3dy, dn3dx, 0, dn4dy, dn4dx, 0])
        row5 = 0.5*np.array([dn1dz, 0, dn1dx, dn2dz, 0, dn2dx, dn3dz, 0, dn3dx, dn4dz, 0, dn4dx])
        row6 = 0.5*np.array([0, dn1dz, dn1dy, 0, dn2dz, dn2dy, 0, dn3dz, dn3dy, 0, dn4dz, dn4dy])
        symgrad_dbf_arr = np.zeros((self.nnodes*3,6))
        symgrad_dbf_arr[:,0] = row1
        symgrad_dbf_arr[:,1] = row2
        symgrad_dbf_arr[:,2] = row3
        symgrad_dbf_arr[:,3] = row4
        symgrad_dbf_arr[:,4] = row5
        symgrad_dbf_arr[:,5] = row6
        return symgrad_dbf_arr
    
    def calc_stifness_integrand(self, coordinates, state_matrix):
        '''
        arguments : 
        coordinates ::: float np.array (3) ::: Local coordinates 
        state_matrix ::: float np.array (6,6) ::: 
        returns ::: 
        integrand ::: float np.array (nnodes*ndim,nnodes*ndim) ::: integrand for the stifness matrix computation 
        '''
        _, det_jacobian, inv_jacobian = self.calc_jacobian(coordinates)
        global_dbf = self.calc_global_dbf_array_symgrad(coordinates, inv_jacobian)
        integrand = det_jacobian*np.dot(np.dot(global_dbf,state_matrix), np.transpose(global_dbf))
        return integrand
    
    def get_bf_array_for_vec_variable(self, coordinates) : 
        '''
        arguments 
        coordinates ::: float np.array (3) ::: Local coordinates 
        returns 
        bf_arr ::: np.array(float) (3*nnodes,3) ::: basis function values arranged for vector variable 
        '''
        bf_array = self.get_bf_array(coordinates)
        n1 = bf_array[0]
        n2 = bf_array[1]
        n3 = bf_array[2]
        n4 = bf_array[3]
        row1 = [n1, 0, 0, n2, 0, 0, n3, 0, 0, n4, 0, 0]
        row2 = [0, n1, 0, 0, n2, 0, 0, n3, 0, 0, n4, 0]
        row3 = [0, 0, n1, 0, 0, n2, 0, 0, n3, 0, 0, n4]
        vv_bf = np.zeros((self.nnodes*3,3))
        vv_bf[:,0] = row1
        vv_bf[:,1] = row2
        vv_bf[:,2] = row3
        return vv_bf

    def calc_massmat_integrand(self, coordinates, rho_value):
        '''
        arguments : 
        coordinates ::: float np.array (3) ::: Local coordinates 
        rho_value ::: float  ::: 
        returns ::: 
        integrand ::: float np.array (nnodes*ndim,nnodes*ndim) ::: integrand for the mass matrix computation 
        '''
        _, det_jacobian, _ = self.calc_jacobian(coordinates)
        bf_array = self.get_bf_array_for_vec_variable(coordinates)
        integrand = det_jacobian*rho_value*np.dot(bf_array,np.transpose(bf_array))
        return integrand 
    
    def set_state_matrices(self, state_arr):
        '''
        arguments : 
        state_arr ::: float np.array (nnodes,6,6) or (6,6) ::: state matrix 
        '''
        if np.shape(state_arr) == (6,6) : 
            state_mat = np.zeros((self.nnodes,6,6))
            for i in range(self.nnodes):
                state_mat[i,:,:] = state_arr
            self.state_matrices = state_mat
        else : 
            if np.shape(state_arr) == (self.nnodes,6,6):
                state_mat = self.interpolate_state_mat(state_arr)
                self.state_matrices = state_mat
            else : 
                print('The state matrix do not have the good shape : ', np.shape(state_arr))

    def interpolate_state_mat(self, state_arr): 
        '''
        arguments : 
        state_arr ::: float np.array (nnodes,6,6) or (6,6) ::: state matrix 
        '''
        state_matrix = np.zeros((self.ngauss_pt,6,6))
        for i in range(self.ngauss_pt): 
            gauss_pt_coordinates = self.refel_gauss_coords[i,:]
            basis_functions = self.get_bf_array(gauss_pt_coordinates)
            basis_functions = np.expand_dims(basis_functions,axis = (1,2))
            gauss_pt_state_matrix = np.sum(basis_functions*state_arr,axis = 0)
            state_matrix[i,:,:] = gauss_pt_state_matrix
            # To be continued
        return state_matrix

    def get_connectivity(self, element):
        '''
        argument 
        element ::: np.array(int) (nelnodes) ::: element connectivity
        return 
        mat_conn ::: np.array(int) (nelnodes*vardim,) ::: local connectivity 
        to gloabal connectivity for global matrix-vector weak form
        '''
        mat_conn = []
        for el in element : 
            mat_conn.append(3*el)
            mat_conn.append(3*el+1)
            mat_conn.append(3*el+2)
        mat_conn = np.asarray(mat_conn)
        return mat_conn
    
    def calc_stress_tensor(self,disp_arr,state_arr):
        '''
        arguments 
        disp_arr ::: np.array (nel_nodes,3) ::: displacement vectors for each element nodes 
        state_arr ::: np.array (nel_nodes,6,6) ::: state matrices at each element nodes
        return 
        stress_tensor_arr ::: np.array (6) ::: flatten stress tensor 
        '''
        #
        elcentroid = self.refcentroid
        basis_functions = self.get_bf_array(elcentroid)
        basis_functions = np.expand_dims(basis_functions,axis = (1,2))
        centroid_state_matrix = np.sum(basis_functions*state_arr,axis = 0)
        #
        _, _, inv_jacobian = self.calc_jacobian(elcentroid)
        symgrad_dbf_arr = self.calc_global_dbf_array_symgrad(elcentroid, inv_jacobian)
        flatten_disp_arr = disp_arr.flatten()
        strain_tensor_arr = np.dot(np.transpose(symgrad_dbf_arr),flatten_disp_arr)
        stress_tensor_arr = np.dot(centroid_state_matrix,strain_tensor_arr)
        return stress_tensor_arr