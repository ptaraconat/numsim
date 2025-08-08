import numpy as np
import plotly.graph_objects as go

def generate_grid(atom_coords, spacing=0.3, padding=3.0):
    atom_coords = np.array(atom_coords)
    mins = atom_coords.min(axis=0) - padding
    maxs = atom_coords.max(axis=0) + padding

    x = np.arange(mins[0], maxs[0], spacing)
    y = np.arange(mins[1], maxs[1], spacing)
    z = np.arange(mins[2], maxs[2], spacing)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    shape = X.shape
    spacing_tuple = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    return grid_points, shape, spacing_tuple, X, Y, Z

def evaluate_mo_on_grid(grid_points, basis_functions, mo_coeffs, orbital_index):
    values = np.zeros(len(grid_points))
    for mu, bf in enumerate(basis_functions):
        phi_vals = np.array([bf(p) for p in grid_points])
        values += mo_coeffs[mu, orbital_index] * phi_vals
    return values

def plot_orbital_volume(values, X, Y, Z, atom_coords=None, atom_charges=None, iso_range=0.05):
    V = values.reshape(X.shape)

    fig = go.Figure()

    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=V.flatten(),
        isomin=-iso_range,
        isomax=iso_range,
        opacity=0.15,
        surface_count=30,
        colorscale='RdBu',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    if atom_coords is not None and atom_charges is not None:
        for coord, Z in zip(atom_coords, atom_charges):
            fig.add_trace(go.Scatter3d(
                x=[coord[0]], y=[coord[1]], z=[coord[2]],
                mode='markers',
                marker=dict(
                    size=8 + 3 * Z,
                    color=Z,
                    colorscale='Viridis',
                    opacity=1.0,
                    line=dict(width=1, color='black')
                ),
                name=f'Noyau Z={Z}'
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (Bohr)',
            yaxis_title='Y (Bohr)',
            zaxis_title='Z (Bohr)',
            aspectmode='data'
        ),
        title='Orbital Molecular 3D avec positions atomiques',
        margin=dict(t=30, l=0, r=0, b=0)
    )
    fig.show()

