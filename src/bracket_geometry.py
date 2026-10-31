from pathlib import Path

import numpy as np
import gmsh

"""Run everything with pixi: pixi run python src/bracket_geometry.py"""

gmsh.initialize()

# create geometry model
gmsh.model.add("3D L-Bracket")

# define geom. dimensions in meters
L, H, W, t = 0.12, 0.10, 0.06, 0.01

# equipment-mounted flange
e_flange = gmsh.model.occ.addBox(0,0,0, L, W, t)
# frame-mounted flange
f_flange = gmsh.model.occ.addBox(0,0,0, t, W, H)

# join them into one solid using fuse
bracket, _ = gmsh.model.occ.fuse(
    [(3, e_flange)], [(3, f_flange)]
)

# create a single mounting hole
# hole = gmsh.model.occ.addCylinder(
#     0.08, W/2, -t, 0,0,3*t, 0.006
# )

# create two holes on the frame-mounted flange
for y in [W*0.25, W*0.75]:
    hole = gmsh.model.occ.addCylinder(-t, y, 0.7*H, 3*t, 0, 0, 0.006)

    # subtract the holes from the geometry
    bracket, _ = gmsh.model.occ.cut(bracket, [(3, hole)])

# create two holes in the equipment-mounted flange
for y in [W*0.25, W*0.75]:
    holex = gmsh.model.occ.addCylinder(0.7 * L,y, -t, 0, 0, 3 * t, 0.006)

    # subtract the hole from the geometry
    bracket, _ = gmsh.model.occ.cut(bracket, [(3, holex)])

# subtract the cylinder from the bracket (bracket - cylinder)
# bracket, _ = gmsh.model.occ.cut(
#     bracket, [(3,hole)]
# )

# push geometry into the model through sync
gmsh.model.occ.synchronize()

# get the finished bracket solid
volumes = gmsh.model.getEntities(3)
assert len(volumes) == 1, "Expected one solid bracket"

# tag the 3D material domain
gmsh.model.addPhysicalGroup(3, [volumes[0][1]],1)
gmsh.model.setPhysicalName(3, 1, "Bracket")

# find exterior boundary surfaces
surfaces = sorted({tag for dim, tag in gmsh.model.getBoundary(volumes, oriented=False) if dim == 2})

# create lists of tags for each boundary condition type for linear elasticity
fixed = []
loaded = []
free = []

# geometric tolerance (meters)
eps = 1e-5

# now, for each surface, we want to sort every surface into one of three boxes
for s in surfaces:
    # calcuate the mins and maxes of each bounding box
    xmin, ymin, zmin, xmax, ymax, zmax = (gmsh.model.getBoundingBox(2, s))

    # back 2d surface attached to the frame
    if abs(xmin) < eps and abs(xmax) < eps:
        fixed.append(s)

    # equipment-mounted top face
    elif abs(zmin - t) < eps and abs(zmax - t) < eps:
        loaded.append(s)

    else:
        free.append(s)

assert fixed, "No fixed surfaces identified"
assert loaded, "No loaded surfaces identified"

# verify every exterior surface belongs to exactly one boundary group:
# the union of the groups must equal the full surface set (nothing missed),
# and the group sizes must sum to the surface count (nothing counted twice)
grouped = set(fixed) | set(loaded) | set(free)
assert grouped == set(surfaces), f"Unassigned surfaces: {set(surfaces) - grouped}"
assert len(fixed) + len(loaded) + len(free) == len(surfaces), "A surface is in more than one group"

# create named surface groups
for tag, name, entities in [(11, "Fixed", fixed), (12, "Loaded", loaded), (13, "Free", free)]:
    if entities:
        gmsh.model.addPhysicalGroup(2, entities, tag)
        gmsh.model.setPhysicalName(2, tag, name)
        print(name, entities)


# generate unstructured 3d meshes from coarse to fine (max element size in meters),
# each saved next to this script as bracket_tagged_<name>.msh
mesh_sizes = {"coarse": 0.008, "medium": 0.005, "fine": 0.003, "extra_fine": 0.002}
out_dir = Path(__file__).parent

for name, mesh_size in mesh_sizes.items():
    # clear the previous mesh so each one is generated from the geometry alone
    gmsh.model.mesh.clear()
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(out_dir / f"bracket_tagged_{name}.msh"))

    # ELEMENT-QUALITY CHECKS - this is important because poorly scored tetrahedra (close to zero hurt FEM solution accuracy)
    types, tags, _ = gmsh.model.mesh.getElements(3)

    tet_tags = np.concatenate([t for typ, t in zip(types, tags) if typ == 4])

    quality = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")

    print(f"{name} (h={mesh_size}): {len(tet_tags)} tets")
    print("  Min quality:", np.min(quality))
    print("  Mean quality,", np.mean(quality))

gmsh.finalize()

# a mesh can be viewed with running the following in the terminal:
# pixi run gmsh src/bracket_tagged_fine.msh
