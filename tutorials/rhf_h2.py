import sys as sys 
sys.path.append('../')
from solver.hartree_fock import * 

# number of electrons in molecule 
n_elec = 2 
# setup nuclear charges 
nuclei_charge = [1, 1]
# 
distances = [round(i*0.1,3) for i in range(4,61)] 
molecule_coordinates = [ [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, distance])] for distance in distances]
energies = []
for atom_coordinates in molecule_coordinates : 
    # Define Basis functions 
    coeff = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
    pg_list = [PrimGauss(atom_coordinates[0],0.3425250914E+01, 0, 0, 0, normalise = True),
            PrimGauss(atom_coordinates[0],0.6239137298E+00, 0, 0, 0, normalise = True),
            PrimGauss(atom_coordinates[0],0.1688554040E+00, 0, 0, 0, normalise = True)]
    bf1 = BasisFunction(pg_list, coeff)
    #
    coeff = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
    pg_list = [PrimGauss(atom_coordinates[1],0.3425250914E+01, 0, 0, 0, normalise = True),
            PrimGauss(atom_coordinates[1],0.6239137298E+00, 0, 0, 0, normalise = True),
            PrimGauss(atom_coordinates[1],0.1688554040E+00, 0, 0, 0, normalise = True)]
    bf2 = BasisFunction(pg_list, coeff)
    # Init RHF solver 
    rhf = RestrictedHartreeFock(basis_functions = [bf1, bf2],
                                atom_coordinates = atom_coordinates,
                                nuclei_charges = nuclei_charge,
                                n_electrons = n_elec)
    electronic_energy = rhf.solve(tol = 1e-10)
    nucat_energy = nuclear_nuclear_repulsion_energy(atom_coordinates, nuclei_charge)
    print('molecule energy : ', electronic_energy + nucat_energy)
    energies.append(electronic_energy + nucat_energy)
print('Optimal_distance : ', distances[np.argmin(energies)])

# Know bond length
atom_coordinates = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.4])]
# Define Basis functions 
coeff = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
pg_list = [PrimGauss(atom_coordinates[0],0.3425250914E+01, 0, 0, 0, normalise = True),
           PrimGauss(atom_coordinates[0],0.6239137298E+00, 0, 0, 0, normalise = True),
           PrimGauss(atom_coordinates[0],0.1688554040E+00, 0, 0, 0, normalise = True)]
bf1 = BasisFunction(pg_list, coeff)
#
coeff = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
pg_list = [PrimGauss(atom_coordinates[1],0.3425250914E+01, 0, 0, 0, normalise = True),
           PrimGauss(atom_coordinates[1],0.6239137298E+00, 0, 0, 0, normalise = True),
           PrimGauss(atom_coordinates[1],0.1688554040E+00, 0, 0, 0, normalise = True)]
bf2 = BasisFunction(pg_list, coeff)
# Init RHF solver 
rhf = RestrictedHartreeFock(basis_functions = [bf1, bf2],
                            atom_coordinates = atom_coordinates,
                            nuclei_charges = nuclei_charge,
                            n_electrons = n_elec)
electronic_energy = rhf.solve(tol = 1e-10)
nucat_energy = nuclear_nuclear_repulsion_energy(atom_coordinates, nuclei_charge)
print('molecule energy : ', electronic_energy + nucat_energy)

import matplotlib.pyplot as plt 
plt.plot(np.array(distances)*0.529, energies)
plt.grid()
plt.show()