import pygmsh
import meshio

with pygmsh.geo.Geometry() as geom:
    polygon = geom.add_polygon(
        [
            [0.0, 0.0],
            [1.0, -0.2],
            [1.1, 1.2],
            [0.1, 0.7],
        ],
        mesh_size=0.1,
    )
    # Tag the edges of the polygon to define boundaries
    boundary_tags = {'left': 0, 'top': 1, 'right': 2, 'bottom': 3}
    for tag, edge_index in boundary_tags.items():
        geom.add_physical([polygon.lines[edge_index]], label=tag)
    mesh = geom.generate_mesh()

# mesh.points, mesh.cells, ...
# mesh.write("out.vtk")

#print(mesh.points)
#print(mesh.cells)

for cell_block in mesh.cells:
    # cell_block is a meshio.CellBlock object
    cell_type = cell_block.type
    cell_data = cell_block.data

    print(f"{cell_type} element: {cell_data}")

# Loop over the mesh cells and print the tag of the boundary
for cell_block in mesh.cells:
    # cell_block is a meshio.CellBlock object
    cell_type = cell_block.type
    cell_data = cell_block.data
    if cell_type == 'line':
        for line_index, phys_group in enumerate(mesh.cell_data[cell_type]['gmsh:physical']):
            if phys_group in boundary_tags.values():
                boundary_tag = [tag for tag, index in boundary_tags.items() if index == phys_group][0]
                print(f"Line {line_index} belongs to the boundary: {boundary_tag}")