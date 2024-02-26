import gmsh 

gmsh.initialize()

gmsh.model.add('test_model')

radius = 0.5
height = 1 
mesh_size = 0.25

factory = gmsh.model.geo
factory.addPoint(0, 0, 0,  tag = 1)
factory.addPoint(0, radius, 0,  tag = 2)
factory.addPoint(0, -radius, 0, tag = 3)
circle_curve1 = factory.addCircleArc(2,1,3, tag = 1)
circle_curve2 = factory.addCircleArc(3,1,2, tag = 2)
circle = factory.addCurveLoop([1,2], tag = 3)
print(circle)
# Create a surface from the circle
surface = factory.addPlaneSurface([circle],tag = 1)
print(surface)
# Extrude the surface
extruded_volume = factory.extrude([(2,surface)],0,0,height)
print(extruded_volume)
volume = gmsh.model.addPhysicalGroup(3, [extruded_volume[1][1]], name="fluid")
surface1 = gmsh.model.addPhysicalGroup(2, [extruded_volume[0][1]], tag = 101, name="surface1")
surface2 = gmsh.model.addPhysicalGroup(2, [extruded_volume[2][1]], tag = 102, name="surface2")
surface3 = gmsh.model.addPhysicalGroup(2, [extruded_volume[3][1]], tag = 103, name="surface3")

# Generate the mesh
factory.synchronize()
gmsh.model.mesh.setSize(extruded_volume, mesh_size)


gmsh.model.mesh.generate(3)
