import sys
sys.path.append('../')
from solver.hartree_fock import *
import numpy as np

# Atome de Soufre
n_electrons = 16  # Soufre Z=16
atom_types = [16]
atom_positions = [np.array([0.0, 0.0, 0.0])]

# Initialisation et résolution RHF avec ta classe modifiée qui accepte 'sto-3g' en basis_functions
rhf = RestrictedHartreeFock(
    basis_functions='sto-3g',
    atom_coordinates=atom_positions,
    nuclei_charges=atom_types,
    n_electrons=n_electrons
)

electronic_energy = rhf.solve(tol=1e-8)
nuclear_repulsion = nuclear_nuclear_repulsion_energy(atom_positions, atom_types)

print("Mon code - Énergie électronique (Hartree) :", electronic_energy)
print("Mon code - Énergie répulsion nucléaire (Hartree) :", nuclear_repulsion)
print("Mon code - Énergie totale (Hartree) :", electronic_energy + nuclear_repulsion)
