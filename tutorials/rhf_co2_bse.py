import sys as sys 
sys.path.append('../')
from solver.hartree_fock import * 

# --- Géométrie CO2 (en Bohr) ---
atom_positions = [
    np.array([0.0, 0.0, 0.0]),          # Carbone
    np.array([0.0, 0.0,  2.192]), # Oxygène droit
    np.array([0.0, 0.0, -2.192])  # Oxygène gauche
]
atom_types = [6, 8, 8]
# Initialisation et résolution RHF
rhf = RestrictedHartreeFock(
    basis_functions='sto-3g',
    atom_coordinates=atom_positions,
    nuclei_charges=atom_types,
)
electronic_energy = rhf.solve(tol=1e-8)
nuclear_repulsion = nuclear_nuclear_repulsion_energy(atom_positions, atom_types)

print("Énergie électronique (Hartree) :", electronic_energy)
print("Énergie répulsion nucléaire (Hartree) :", nuclear_repulsion)
print("Énergie totale de la molécule (Hartree) :", electronic_energy + nuclear_repulsion)
