from pathlib import Path

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, la, mesh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem

"""Run with: pixi run python src/solve_elasticity.py
Or import solve() from another script, e.g. mesh_convergence.py"""

FIXED = 11
LOADED = 12
FILLET = 14

#####################
# material properties (aluminum)
#####################
# Youngs modulus describing a materials stiffness (stress / strain)
E = 69e9
# poissons ratio describing how a material squeezes or bulges sideways -(lateral strain/axial strain)
nu = 0.33

mu = E / (2 * (1 + nu))
lam = nu * E / ((1 + nu) * (1 - 2 * nu))


def epsilon(w):
    return ufl.sym(ufl.grad(w))


def sigma(w):
    return lam * ufl.tr(epsilon(w)) * ufl.Identity(3) + 2.0 * mu * epsilon(w)


# building nullspace just matters for speed, not accuracy, e.g., a solve without it goes through ~220 iterations, while solving with it goes through about 56
def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity."""
    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=PETSc.ScalarType) for i in range(6)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm) for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)


def solve(mesh_file, degree=2):
    """Solve linear elasticity on a tagged gmsh mesh and return a dict of results
    (scalar quantities in SI units, plus the mesh, displacement and von Mises functions)."""

    ####################
    # 1. import mesh from gmsh
    ###################
    data = gmshio.read_from_msh(
        mesh_file,
        MPI.COMM_WORLD,
        rank=0,
        gdim=3
    )

    msh = data.mesh
    facet_tags = data.facet_tags

    assert facet_tags is not None

    ###############
    # 2. vector-valued function space
    ##############

    # second-order elements by default
    V = fem.functionspace(msh, ("Lagrange", degree, (3,)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    ####################
    # 3. Fixed displacement boundary condition
    ####################

    fdim = msh.topology.dim - 1

    fixed_facets = facet_tags.find(FIXED)
    loaded_facets = facet_tags.find(LOADED)

    assert len(fixed_facets) > 0
    assert len(loaded_facets) > 0

    fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)

    bc = fem.dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), fixed_dofs, V)

    ##########################
    # 4. applied traction (Pa)
    ##########################
    traction = fem.Constant(msh, np.array([0,0,-1e5], dtype=PETSc.ScalarType))

    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    ########################
    # 5. linear elasticity weak form
    #######################
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(traction, v) * ds(LOADED)

    #######################
    # 6. Solve the linear system
    #######################

    # solve with CG and GAMG
    # ksp stands for krylov subspace solver
    # cg is the conjugate gradient
    # gamg is algebraic multigrid
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="bracket_", petsc_options={"ksp_type": "cg","pc_type":"gamg", "ksp_rtol":1e-9,"ksp_max_it":1000, "ksp_error_if_not_converged":True})

    ns = build_nullspace(V)
    problem.A.setNearNullSpace(ns)

    uh = problem.solve()
    uh.name = "displacement"
    uh.x.scatter_forward()

    assert problem.solver.getConvergedReason() > 0

    ######################
    # 7. stress post-processing
    ######################
    stress = sigma(uh)

    stress_dev = (stress - (1.0 / 3.0) * ufl.tr(stress) * ufl.Identity(3))

    # Von Mises equivalent stress (Pa) that converts stresses into a scalar
    von_mises = ufl.sqrt(1.5 * ufl.inner(stress_dev, stress_dev))

    # P2 displacement -> cellwise linear stress
    Q = fem.functionspace(msh, ("DG", 1))

    vm_expr = fem.Expression(von_mises, Q.element.interpolation_points)

    vm_h = fem.Function(Q)
    vm_h.name = "von_mises_stress"
    vm_h.interpolate(vm_expr)

    # max computed von mises stress
    local_max = np.max(vm_h.x.array)
    max_vm = msh.comm.allreduce(local_max, op=MPI.MAX)

    # max von Mises in the cells touching the fillet; the rounded fillet has a finite peak stress
    # that converges with refinement, while sharp edges elsewhere (e.g. where the clamped face meets
    # a free face) are singular in theory, so the global max is not guaranteed to converge
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    fillet_cells = mesh.compute_incident_entities(msh.topology, facet_tags.find(FILLET), fdim, msh.topology.dim)
    fillet_vm = vm_h.x.array[Q.dofmap.list[fillet_cells].ravel()]
    local_max_fillet = np.max(fillet_vm) if fillet_vm.size else 0.0
    max_vm_fillet = msh.comm.allreduce(local_max_fillet, op=MPI.MAX)

    # mean stress over bracket volume
    total_stress = msh.comm.allreduce(fem.assemble_scalar(fem.form(von_mises * ufl.dx)), op=MPI.SUM)
    volume = msh.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=msh))), op=MPI.SUM)

    # mean vertical displacement on loaded surface
    loaded_area = msh.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(LOADED))), op=MPI.SUM)
    loaded_uz = msh.comm.allreduce(fem.assemble_scalar(fem.form(uh[2] * ds(LOADED))), op=MPI.SUM) / loaded_area

    ######################
    # 8. Calculate maximum displacement
    ######################
    displacements = uh.x.array.reshape((-1, 3))
    magnitudes = np.linalg.norm(displacements, axis=1)

    local_max = np.max(magnitudes)
    max_disp = msh.comm.allreduce(local_max, op=MPI.MAX)

    return {
        "max_disp": max_disp,
        "mean_loaded_uz": loaded_uz,
        "max_vm": max_vm,
        "max_vm_fillet": max_vm_fillet,
        "mean_vm": total_stress / volume,
        "num_dofs": V.dofmap.index_map.size_global * V.dofmap.index_map_bs,
        "iterations": problem.solver.getIterationNumber(),
        "mesh": msh,
        "displacement": uh,
        "von_mises": vm_h,
    }


if __name__ == "__main__":
    src_dir = Path(__file__).parent
    results = solve(src_dir / "bracket_tagged_fine.msh")

    if MPI.COMM_WORLD.rank == 0:
        print(f"Maximum von Mises: {results['max_vm']/1e6:.4f} MPa")
        print(f"Maximum von Mises on fillet: {results['max_vm_fillet']/1e6:.4f} MPa")
        print(f"Mean von Mises: {results['mean_vm']/1e6:.4f} MPa")
        print(f"Mean loaded uz: {results['mean_loaded_uz']*1000:.6f} mm")
        print(f"Maximum displacement:{results['max_disp']:.6e} m")
        print(f"Maximum displacement:{results['max_disp']*1000:.6f} mm")

    #####################
    # 9. Export displacement and von Mises stress for visualization
    #####################
    # VTX (ADIOS2 .bp) keeps the P2 displacement and DG1 stress exactly; XDMF only
    # handles fields on the mesh's own P1 nodes. Open the .bp folders in ParaView.
    msh = results["mesh"]
    with io.VTXWriter(msh.comm, "bracket_displacement.bp", [results["displacement"]]) as file:
        file.write(0.0)
    with io.VTXWriter(msh.comm, "bracket_stress.bp", [results["von_mises"]]) as file:
        file.write(0.0)

    if msh.comm.rank == 0:
        print("Elasticity solve completed.")
