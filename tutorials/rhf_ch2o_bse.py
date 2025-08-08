import sys as sys 
sys.path.append('../')
from solver.hartree_fock import * 
# --- 1. Define the geometry (in Bohr) ---
atom_positions = [
    np.array([0.000, 0.000, 0.000]),        # C
    np.array([0.000, 0.000, 2.278]),        # O
    np.array([1.832, 0.000, -0.597]),       # H1
    np.array([-1.832, 0.000, -0.597])       # H2
]
atom_types = [6, 8, 1, 1]   # C, O, H, H
n_electrons = None
# Initialisation et résolution RHF
rhf = RestrictedHartreeFock(
    atom_positions,
    atom_types,
    basis_functions='sto-3g',
    n_electrons=n_electrons
)
electronic_energy = rhf.solve(max_iter = 100, tol=1e-8)
nuclear_repulsion = nuclear_nuclear_repulsion_energy(atom_positions, atom_types)

print("Énergie électronique (Hartree) :", electronic_energy)
print("Énergie répulsion nucléaire (Hartree) :", nuclear_repulsion)
print("Énergie totale de la molécule (Hartree) :", electronic_energy + nuclear_repulsion)
