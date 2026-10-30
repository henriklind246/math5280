import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
from ufl import facet

"""Run with: pixi run python src/solve_elasticity.py"""

####################
# 1. import mesh from gmsh
###################
data = gmshio.read_from_msh(
    "bracket_tagged.msh",
    MPI.COMM_WORLD,
    rank=0,
    gdim=3
)

msh = data.mesh
facet_tags = data.facet_tags

assert facet_tags is not None

FIXED = 11
LOADED = 12

#####################
# 2. define bracket material properties (aluminum)
#####################
# Youngs modulus describing a materials stiffness (stress / strain)
E = 69e9
# poissons ratio describing how a material squeezes or bulges sideways -(lateral strain/axial strain)
nu = 0.33

mu = E / (2 * (1 + nu))
lam = nu * E / ((1 + nu) * (1 - 2 * nu))

###############
# 3. vector-valued function space
##############
V = fem.functionspace(msh, ("Lagrange", 1, (3,)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

####################
# 4. Fixed displacement boundary condition
####################

fdim = msh.topology.dim - 1

fixed_facets = facet_tags.find(FIXED)
loaded_facets = facet_tags.find(LOADED)

assert len(fixed_facets) > 0
assert len(loaded_facets) > 0

fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)

bc = fem.dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), fixed_dofs, V)

##########################
# 5. applied traction (Pa)
##########################
traction = fem.Constant(msh, np.array([0,0,-1e5], dtype=PETSc.ScalarType))

ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

########################
# 6. linear elasticity weak form
#######################
def epsilon(w):
    return ufl.sym(ufl.grad(w))

def sigma(w):
    return lam * ufl.tr(epsilon(w)) * ufl.Identity(3) + 2.0 * mu * epsilon(w)

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(traction, v) * ds(LOADED)

#######################
# 7. Solve the linear system
#######################

problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="bracket_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

uh = problem.solve()
uh.name = "displacement"
uh.x.scatter_forward()

assert problem.solver.getConvergedReason() > 0

######################
# 8. Calculate maximum displacement
######################

displacements = uh.x.array.reshape((-1, 3))
magnitudes = np.linalg.norm(displacements, axis=1)

local_max = np.max(magnitudes)
max_disp = msh.comm.allreduce(local_max, op=MPI.MAX)

if msh.comm.rank == 0:
    print(f"Maximum displacement:{max_disp:.6e} m")
    print(f"Maximum displacement:{max_disp*1000:.6f} mm")

#####################
# 9. Export displacement for visualization
#####################
with io.XDMFFile(msh.comm, "bracket_displacement.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

if msh.comm.rank == 0:
    print("Elasticity solve completed.")

