import sys as sys 
sys.path.append('../')
from solver.hartree_fock import * 
import math

# Géométrie H2O (en Bohr)
OH_distance = 0.958 * 1.8897  # ~1.81 Bohr
angle_deg = 104.5
angle_rad = np.radians(angle_deg / 2)

atom_coordinates = [
    np.array([0.0, 0.0, 0.0]),  # Oxygène
    np.array([OH_distance * np.sin(angle_rad), 0.0, OH_distance * np.cos(angle_rad)]),  # H gauche
    np.array([-OH_distance * np.sin(angle_rad), 0.0, OH_distance * np.cos(angle_rad)])  # H droite
]

nuclei_charges = [8, 1, 1]
n_electrons = 10

# STO-3G coefficients et exposants (exemples simplifiés)
coeff_s = [0.1543289673, 0.5353281423, 0.4446345422]
exps_O_s = [130.70932, 23.808861, 6.4436083]
exps_H_s = [3.42525091, 0.62391373, 0.16885540]

coeff_p = [0.1543289673, 0.5353281423, 0.4446345422]
exps_O_p = [5.0331513, 1.1695961, 0.3803890]

# Créer les BasisFunctions
# Oxygène s
pg_list_O_s = [PrimGauss(atom_coordinates[0], exp, 0, 0, 0, normalise=True) for exp in exps_O_s]
bf_O_s = BasisFunction(pg_list_O_s, coeff_s)

# Oxygène p orbitals
pg_list_O_px = [PrimGauss(atom_coordinates[0], exp, 1, 0, 0, normalise=True) for exp in exps_O_p]
bf_O_px = BasisFunction(pg_list_O_px, coeff_p)

pg_list_O_py = [PrimGauss(atom_coordinates[0], exp, 0, 1, 0, normalise=True) for exp in exps_O_p]
bf_O_py = BasisFunction(pg_list_O_py, coeff_p)

pg_list_O_pz = [PrimGauss(atom_coordinates[0], exp, 0, 0, 1, normalise=True) for exp in exps_O_p]
bf_O_pz = BasisFunction(pg_list_O_pz, coeff_p)

# Hydrogène gauche s
pg_list_H1 = [PrimGauss(atom_coordinates[1], exp, 0, 0, 0, normalise=True) for exp in exps_H_s]
bf_H1 = BasisFunction(pg_list_H1, coeff_s)

# Hydrogène droite s
pg_list_H2 = [PrimGauss(atom_coordinates[2], exp, 0, 0, 0, normalise=True) for exp in exps_H_s]
bf_H2 = BasisFunction(pg_list_H2, coeff_s)

# Assemblage base
basis_functions = [bf_O_s, bf_O_px, bf_O_py, bf_O_pz, bf_H1, bf_H2]

# Initialisation et résolution RHF
rhf = RestrictedHartreeFock(
    basis_functions=basis_functions,
    atom_coordinates=atom_coordinates,
    nuclei_charges=nuclei_charges,
    n_electrons=n_electrons
)

electronic_energy = rhf.solve(tol=1e-8)
nuclear_repulsion = nuclear_nuclear_repulsion_energy(atom_coordinates, nuclei_charges)

print("Énergie électronique (Hartree) :", electronic_energy)
print("Énergie répulsion nucléaire (Hartree) :", nuclear_repulsion)
print("Énergie totale de la molécule (Hartree) :", electronic_energy + nuclear_repulsion)
