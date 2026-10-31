"""Mesh convergence study for the bracket elasticity solve.
Generate the meshes first with bracket_geometry.py, then
run with pixi: pixi run python src/mesh_convergence.py"""
from pathlib import Path

from mpi4py import MPI

from solve_elasticity import solve

# meshes from coarse to fine (names must match the mesh_sizes keys in bracket_geometry.py)
mesh_names = ["coarse", "medium", "fine", "extra_fine", "ultra_fine"]

# quantities compared between refinements; the global max von Mises is left out on purpose,
# since sharp edges (e.g. where the clamped face meets a free face) are singular in theory and
# can grow without bound under refinement, while the rounded fillet has a finite peak stress
quantities = {"max_disp": "max displacement", "mean_loaded_uz": "mean loaded uz",
              "max_vm_fillet": "max von Mises on fillet"}

# practical convergence target: < 1% change between successive refinements
tolerance = 0.01

src_dir = Path(__file__).parent
results = {}

for name in mesh_names:
    results[name] = solve(src_dir / f"bracket_tagged_{name}.msh")

    if MPI.COMM_WORLD.rank == 0:
        r = results[name]
        print(f"{name}: {r['num_dofs']} dofs, {r['iterations']} CG iterations, "
              f"max displacement {r['max_disp']*1000:.6f} mm, mean loaded uz {r['mean_loaded_uz']*1000:.6f} mm, "
              f"fillet max von Mises {r['max_vm_fillet']/1e6:.4f} MPa (global {r['max_vm']/1e6:.4f} MPa)")

# MESH CONVERGENCE CHECK - if successive refinements produce increasingly similar values,
# that's evidence of convergence
if MPI.COMM_WORLD.rank == 0:
    for key, label in quantities.items():
        print(f"\nMesh convergence ({label}):")
        for coarser, finer in zip(mesh_names, mesh_names[1:]):
            change = abs(results[finer][key] - results[coarser][key]) / abs(results[finer][key])
            status = "converged" if change < tolerance else "not converged"
            print(f"  {coarser} -> {finer}: {change:.2%} change ({status})")
