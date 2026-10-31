"""Verify the tagged meshes can be imported correctly in FEniCSx"""
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio

"""Run with pixi: pixi run python src/verify_import.py"""

# check every mesh written by bracket_geometry.py, coarse to fine
# (names must match the mesh_sizes keys there; files live next to this script)
mesh_names = ["coarse", "medium", "fine", "extra_fine", "ultra_fine"]

for name in mesh_names:
    mesh_file = Path(__file__).parent / f"bracket_tagged_{name}.msh"

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

    print(f"--- {name} ---")
    print("Physical Groups:", data.physical_groups)

    print("Cells:", mesh.topology.index_map(3).size_global)

    print("Cell tags:", np.unique(cell_tags.values))

    print("Boundary tags:", np.unique(facet_tags.values))

    # verify expected physical groups
    assert set(np.unique(cell_tags.values)) == {1}
    assert set(np.unique(facet_tags.values)) == {11, 12, 13, 14}

print("All meshes and physical tags imported successfully")
