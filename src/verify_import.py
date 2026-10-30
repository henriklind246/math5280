"""Verify the tagged mesh can be imported correctly in FEniCSx"""
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio

"""Run with pixi: pixi run python src/verify_import.py"""

# locate the mesh next to this script, so it works from any working directory
mesh_file = Path(__file__).parent / "bracket_tagged.msh"

# import mesh and physical tags
data = gmshio.read_from_msh(
    mesh_file,
    MPI.COMM_WORLD,
    rank=0,
    gdim=3
)

mesh = data.mesh
cell_tags = data.cell_tags
facet_tags = data.facet_tags

# confirm successful import
assert cell_tags is not None
assert facet_tags is not None
assert mesh.geometry.dim == 3

print("Physical Groups:", data.physical_groups)

print("Cell tags:", np.unique(cell_tags.values))

print("Boundary tags:", np.unique(facet_tags.values))

# verify expected physical groups
assert set(np.unique(cell_tags.values)) == {1}
assert set(np.unique(facet_tags.values)) == {11, 12, 13}

print("Mesh and physical tags imported successfully")


