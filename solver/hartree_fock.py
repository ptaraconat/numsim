from quantchem.primitive_gaussians import * 
from scipy import linalg

def nuclear_nuclear_repulsion_energy(atom_coords, zlist):
    
    assert (len(atom_coords) == len(zlist))
    natoms = len(zlist)
    E_NN = 0
    for i in range(natoms):
        Zi = zlist[i]
        for j in range(natoms):
            if j > i:
                Zj = zlist[j]
                Rijx = atom_coords[i][0] - atom_coords[j][0]
                Rijy = atom_coords[i][1] - atom_coords[j][1]
                Rijz = atom_coords[i][2] - atom_coords[j][2]
                Rijx_squared = Rijx*Rijx
                Rijy_squared = Rijy*Rijy
                Rijz_squared = Rijz*Rijz             
                Rij = math.sqrt(Rijx_squared + Rijy_squared + Rijz_squared)      
                E_NN += (Zi*Zj)/Rij
                
    return E_NN

class HartreeFockSolver:
    '''
    '''
    def __init__(self):
        '''
        '''
        pass 

class RestrictedHartreeFock(HartreeFockSolver): 
    '''
    '''
    def __init__(self, basis_functions = None, atom_coordinates = None, nuclei_charges = None, n_electrons = None):
        '''
        '''
        self.basis_functions = basis_functions
        self.atom_coordinates = atom_coordinates
        self.n_electrons = n_electrons
        self.nuclei_charges = nuclei_charges
        self.density = np.zeros((len(self.basis_functions), len(self.basis_functions)))
        self.mos = np.zeros((len(self.basis_functions), n_electrons))
    
    def initialize_overlapp_matrix(self) : 
        '''
        '''
        n_bf = len(self.basis_functions)
        S = np.zeros((n_bf,n_bf))
        for i, bfi in enumerate(self.basis_functions): 
            for j, bfj in enumerate(self.basis_functions):
                S[i,j] = basis_function_overlap(bfi, bfj)
        self.overlapp = S
    def initialize_core_matrices(self): 
        '''
        '''
        n_bf = len(self.basis_functions)
        # init kinetic matrix 
        T = np.zeros((n_bf,n_bf))
        # init nuclei attraction mat 
        V = np.zeros((n_bf,n_bf))
        for i, bfi in enumerate(self.basis_functions): 
            for j, bfj in enumerate(self.basis_functions):
                T[i,j] = basis_function_kinetic_integral(bfi, bfj)
                V[i,j] = basis_function_nucat_integral(bfi, bfj, self.atom_coordinates, self.nuclei_charges)
        self.kinetic = T 
        self.nucat = V
        self.core_hamiltonian = T + V 
    def initialize_electron_repulsion_matrix(self):
        '''
        '''
        n_bf = len(self.basis_functions)
        ERI = np.zeros((n_bf, n_bf,n_bf,n_bf))
        for i,bfi in enumerate(self.basis_functions):
            for j, bfj in enumerate(self.basis_functions):
                for k, bfk in enumerate(self.basis_functions):
                    for l, bfl in enumerate(self.basis_functions): 
                        ERI[i,j,k,l] = basis_function_elecrep_integral(bfi,bfj,bfk,bfl)
        self.elecrep = ERI
    def update_gmat(self): 
        '''
        '''
        n_bf = len(self.basis_functions)
        Gmat = np.zeros((n_bf,n_bf))
        for i in range(n_bf):
            for j in range(n_bf):
                for k in range(n_bf): 
                    for l in range(n_bf):
                        term1 = self.elecrep[i,j,k,l]
                        term2 = self.elecrep[i,j,k,j]
                        Gmat[i,j] += self.density[k,l] * (term1 - 0.5 * term2) 
        self.gmat = Gmat 
    def update_density(self):
        '''
        '''
        n_bf = len(self.basis_functions)
        density = np.zeros((n_bf,n_bf))
        number_occupied_orbital = int(self.n_electrons*0.5)
        for i in range(n_bf) : 
            for j in range(n_bf): 
                for k in range(number_occupied_orbital) : 
                    density[i,j] += self.mos[i,k]*self.mos[j,k]
        # Slice the occupied molecular orbitals
        #C_occ = self.mos[:, :number_occupied_orbitals]  # shape (n_bf, n_occ)
        # Build density matrix: P = 2 * C_occ @ C_occ.T
        #self.density = 2 * np.dot(C_occ, C_occ.T)
        self.density = 2*density   
    def calculate_energy(self):
        '''
        '''
        energy = 0
        n_bf = len(self.basis_functions)
        for i in range(n_bf):
            for j in range(n_bf) : 
                energy += self.density[i,j] * (self.core_hamiltonian[i,j] + 0.5 * self.gmat[i,j])
        return energy  
    def solve(self,max_iter = 20, tol = 1e-5):
        '''
        '''
        # precompute integrals 
        self.initialize_overlapp_matrix()
        self.initialize_core_matrices()
        self.initialize_electron_repulsion_matrix()
        # guess first density and get associated energy 
        # By default density is initialized to 0 
        self.update_gmat()
        energy = self.calculate_energy()
        #Hartree Fock loop
        for step in range(max_iter) : 
            former_energy = energy
            # Calculate Fock matrix 
            self.update_gmat()
            F = self.core_hamiltonian + self.gmat
            # S^{-1/2} S S^{-1/2}
            S_inverse = linalg.inv(self.overlapp)
            S_inverse_sqrt = linalg.sqrtm(S_inverse)
            # S^{-1/2} F S^{-1/2}
            F_unitS = np.dot(S_inverse_sqrt, np.dot(F, S_inverse_sqrt))
            eigenvalues, eigenvectors = linalg.eigh(F_unitS)
            self.mos = np.dot(S_inverse_sqrt, eigenvectors)
            # calculate new density matrix 
            self.update_density()
            # calculate new energy 
            energy = self.calculate_energy()
            # check for convergence 
            if np.abs(former_energy - energy) < tol :
                print('HF loop converged at step ', step)
                print('electronic energy : ', energy)
                return energy
