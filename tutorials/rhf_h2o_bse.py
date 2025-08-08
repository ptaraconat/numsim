import sys as sys 
sys.path.append('../')
from solver.hartree_fock import * 

n_electrons = 10
atom_types = [8, 1, 1]  # O, H, H
atom_positions = [np.array([0., 0., 0.]), 
                  np.array([1.43141111, 0.        , 1.1083169 ]), 
                  np.array([-1.43141111,  0.        ,  1.1083169 ])]
# Initialisation et résolution RHF
rhf = RestrictedHartreeFock(
    basis_functions='sto-3g',
    atom_coordinates=atom_positions,
    nuclei_charges=atom_types,
    n_electrons=n_electrons
)
electronic_energy = rhf.solve(tol=1e-8)
nuclear_repulsion = nuclear_nuclear_repulsion_energy(atom_positions, atom_types)
print("Énergie électronique (Hartree) :", electronic_energy)
print("Énergie répulsion nucléaire (Hartree) :", nuclear_repulsion)
print("Énergie totale de la molécule (Hartree) :", electronic_energy + nuclear_repulsion)
