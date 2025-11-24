# use Lagrange multiplier
from firedrake import *
import csv
import os
import sys

# ---------------------------
# User configuration (top)
# ---------------------------

# Choose initial condition: "hopf" or "E3" or "iso"
IC_CHOICE = "hopf" 

# Domain and mesh parameters: choose sizes and resolutions depending on IC
if IC_CHOICE.lower() == "hopf":
    # Hopf fibre
    Lx, Ly, Lz = 4, 4, 10
elif IC_CHOICE.lower() == "e3":
    # E3 field
    Lx, Ly, Lz = 4, 4, 48
elif IC_CHOICE.lower() == "iso":
    # IsoHelix
    Lx, Ly, Lz = 8, 8, 20

elif IC_CHOICE.lower() == "hesse":
    Lx, Ly, Lz = 40, 40, 40

else:
    raise ValueError("Unknown IC_CHOICE: %r" % IC_CHOICE)

# Domain / mesh parameters
periodic = False
closed = True # closed domain means B\cdot n = 0 on each faces
# Polynomial degree
POLY_DEGREE = 1

# Time-stepping
DT = 1.0
T_FINAL = 10000
TAU = 100

# Output / logging
OUTPUT_DIR = "output"
DATA_FILENAME = "output/data.csv"
PVD_FILENAME = os.path.join(OUTPUT_DIR, "parker.pvd")

# Boundary identifiers used where DirichletBC expects names.
# For periodic True, only "on_boundary" is meaningful here.
# For non-periodic, we'll include "top" and "bottom" explicitly.
if periodic:
    DIRICHLET_IDS = ("on_boundary",)
else:
    DIRICHLET_IDS = ("on_boundary", "top", "bottom")

# Solver parameter snippets can be placed here if you want to tweak later
# e.g. PROJ_SOLVER_PARAMS = {...}

# ---------------------------
# End of user configuration
# ---------------------------


# ---------------------------
# Utility: create mesh & coordinate shift
# ---------------------------
def create_mesh(Lx, Ly, Lz, periodic):
    """
    Build a quadrilateral base mesh with (Nx,Ny) cells on (Lx,Ly),
    then extrude vertically with Nz layers to height Lz.

    Returns the extruded mesh, centered around (0,0,0).
    """
    # RectangleMesh signature: RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
    Nx = 4
    Ny = 4
    base = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)

    # Extrude: ExtrudedMesh(base_mesh, layers, height, periodic=...)
    mesh = ExtrudedMesh(base, Lz, 1, periodic=periodic)

    # center coordinates so the domain is [-Lx/2, Lx/2] x [-Ly/2, Ly/2] x [-Lz/2, Lz/2]
    # hesse, the domain is [-20, 20] x [-20, 20] x [0, 40]

    mesh.coordinates.dat.data[:, 0] -= Lx/2
    mesh.coordinates.dat.data[:, 1] -= Ly/2
    if IC_CHOICE.lower() != "hesse":
        mesh.coordinates.dat.data[:, 2] -= Lz/2
    
    return mesh


mesh = create_mesh(Lx, Ly, Lz, periodic)
x, y, z0 = X = SpatialCoordinate(mesh)

# ---------------------------
# Function spaces
# ---------------------------
k = POLY_DEGREE
Vg = VectorFunctionSpace(mesh, "Q", k)
Vg_ = FunctionSpace(mesh, "Q", k)
Vc = FunctionSpace(mesh, "NCE", k)
Vd = FunctionSpace(mesh, "NCF", k)
Vn = FunctionSpace(mesh, "DQ", k-1)
VR = FunctionSpace(mesh, "R", 0)

# Mixed unknowns: [B, u, A, E, lmbda_e, lmbda_m]
Z = MixedFunctionSpace([Vd, Vd, Vc, Vc, Vc, VR, VR])
z = Function(Z)
z_prev = Function(Z)
z_test = TestFunction(Z)
(B, u, A, E, j, lmbda_e, lmbda_m) = split(z)
(Bt, ut, At, Et, jt, lmbda_et, lmbda_mt) = split(z_test)
(Bp, up, Ap, Ep, jp, lmbda_ep, lmbda_mp) = split(z_prev)

# Convenient references to subfunctions
(B_, u_, A_, E_, j_, lmbda_e_, lmbda_m_) = z.subfunctions
B_.rename("MagneticField")
E_.rename("ElectricField")
u_.rename("Velocity")
A_.rename("MagneticPotential")
j_.rename("Current")

# ---------------------------
# Initial condition constructors
# ---------------------------
def make_B_init_hopf(X):
    # Hopf initial field (your original expression)
    x, y, z0 = X
    w1 = 3
    w2 = 2
    s = 1
    deno = 1 + dot(X, X)
    coeff = 4*sqrt(s)/((pi * deno * deno * deno)*sqrt(w1**2+w2**2))
    B_init = as_vector([
        coeff*2*(w2*y-w1*x*z0),
        -coeff*2*(w2*x+w1*y*z0),
        coeff*w1*(-1+x**2+y**2-z0**2)
    ])
    return B_init

def make_B_init_E3(X):
    # E3 field (your original multipole/twisted tubes)
    x, y, z0 = X
    x_c = [1, -1, 1, -1, 1, -1]
    y_c = 0
    z_c = [-20, -12, -4, 4, 12, 20]
    a = sqrt(2.0)
    k_twist = 1.0
    l = 2.0
    R = FunctionSpace(mesh, "R", 0)
    zero = Function(R).assign(0)
    B_0 = 1 # change it to zeor, if you want remove the background magnetic field
    B_x = 0
    B_y = 0
    B_z = B_0
    for i in range(6):
        coeff = exp((-(x-x_c[i])**2/a**2) - ((y - y_c)**2/a**2) - ((z0 - z_c[i])**2/l**2))
        B_x = B_x + coeff * (2 * k_twist * B_0/a * (-(y-y_c)))
        B_y = B_y + coeff * (2 * k_twist * B_0/a * ((x-x_c[i])))
    B_init = as_vector([B_x, B_y, B_z])
    return B_init

def make_B_init_iso(X):
    #IsoHelix with phi = pi from https://doi.org/10.1137/140967404
    x, y, z0 = X
    B0 = 1
    ar = sqrt(2)
    az = 2
    phi = pi
    (X1, Y1, Z1) = SpatialCoordinate(mesh)
    coeff = 2*B0*z0/az**2 * exp(-(x**2 + y**2)/ar**2 - z0**2/az**2)

    B_init = as_vector([coeff * phi * y,
                       -coeff * phi * x,
                       B0])
    return B_init

def make_B_init_hesse(X):
    x, y, z0 = X
    t = 2
    Bx = -2
    By = -z0 - t*(1 - z0**2)/(1 + z0**2/25)**2/(1 + x**2/25)
    Bz =y
    B_init = as_vector([Bx, By, Bz])
    
    return B_init


def make_B_init(choice):
    if choice.lower() == "hopf":
        return make_B_init_hopf(X)

    elif choice.lower() == "e3" or choice.lower() == "E3".lower():
        return make_B_init_E3(X)

    elif choice.lower() == "iso" or choice.lower() == "iso".lower():
        return make_B_init_iso(X)
    
    elif choice.lower() == "hesse" or choice.lower() == "hesse".lower():
        return make_B_init_hesse(X)

    else:
        raise ValueError("Unknown IC choice: %r" % choice)

# Build the selected B_init (UFL expression)
B_init_expr = make_B_init(IC_CHOICE)
# boundary value for projection

if closed:
    B_init_bc_t = 0
    B_init_bc_b = 0
else:
    B_init_bc_t = as_vector([0, 0, 1])
    B_init_bc_b = as_vector([0, 0, 1])
# ---------------------------
# Helper: build linear / nonlinear solvers (kept generic)
# ---------------------------
def build_linear_solver(a, L, u_sol, bcs, aP=None, solver_parameters = None, options_prefix=None):
    problem = LinearVariationalProblem(a, L, u_sol, bcs=bcs, aP=aP)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=solver_parameters,
                                     options_prefix=options_prefix)
    return solver

def build_nonlinear_solver(F, z_sol, bcs, Jp=None, solver_parameters = None, options_prefix=None):
    problem = NonlinearVariationalProblem(F, z_sol, bcs, Jp=Jp)
    solver = NonlinearVariationalSolver(problem,
                solver_parameters=solver_parameters,
                options_prefix=options_prefix)
    return solver

# ---------------------------
# Project initial B to be divergence-free and satisfy BCs
# ---------------------------
def project_initial_conditions(B_init_expr):
    Zp = MixedFunctionSpace([Vd, Vn])
    zp = Function(Zp)
    (Bp_proj, p) = split(zp)
    test_B, test_p = TestFunctions(Zp)
    
    # not bc for p
    bcs_proj = [DirichletBC(Zp.sub(0), 0, sub) for sub in DIRICHLET_IDS]
#if not closed:
#        bcs_proj += [
#                     DirichletBC(Zp.sub(0), B_init_bc_t, "top"),
##                     DirichletBC(Zp.sub(0), B_init_bc_b, "bottom"),
#        ]

    L = (
        0.5*inner(Bp_proj, Bp_proj)*dx
        - inner(B_init_expr, Bp_proj)*dx
        - inner(p, div(Bp_proj))*dx
    )
    Fp = derivative(L, zp, TestFunction(Zp))

    gamma = Constant(1E5)
    Up = 0.5*(inner(Bp_proj, Bp_proj) + inner(div(Bp_proj) * gamma, div(Bp_proj)) + inner(p * (1/gamma), p))*dx
    Jp = derivative(derivative(Up, zp), zp)

    spp = {
        "mat_type": "nest",
        "snes_type": "ksponly",
        "ksp_type": "minres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_pc_type": "cholesky",
        "fieldsplit_pc_factor_mat_solver_type": "mumps",
        "ksp_atol": 1.0e-5,
        "ksp_rtol": 1.0e-5,
        "ksp_minres_nutol": 1E-8,
        "ksp_convergence_test": "skip",
    }

    solve(Fp == 0, zp, bcs_proj, Jp=Jp, solver_parameters=spp,
          options_prefix="B_init_div_free_projection")
    return zp.subfunctions[0]  # return projected B

# ---------------------------
# Helicity / magnetic potential solver utilities
# ---------------------------
def build_helicity_solver():
    # Solve curl-curl u = curl^{-1} B  (weak: curl(u), curl(v) = <B, curl(v)> )
    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    u_sol = Function(Vc)

    a = inner(curl(u), curl(v)) * dx
    B_proj = project_initial_conditions(B_init_expr)
    L = inner(B_proj, curl(v)) * dx

    # small regularization for kernel
    beta = Constant(0.1)
    Jp_curl = a + inner(beta * u, v) * dx

    bcs_curl = [DirichletBC(Vc, 0, sub) for sub in DIRICHLET_IDS]

    sparams = {
        "snes_type": "ksponly",
        "ksp_type": "minres",
        "ksp_max_it": 1000,
        "pc_type": "cholesky",
        "ksp_norm_type": "preconditioned",
        "ksp_minres_nutol": 1E-8,
    }

    solver = build_linear_solver(a, L, u_sol, bcs_curl, Jp_curl, sparams, options_prefix="helicity")
    return solver

helicity_solver = build_helicity_solver()

def riesz_map(functional):
    """
    Map a residual (vector in dual) to the function space via the preconditioner.
    This follows your original pattern using helicity_solver's PC.
    """
    function = Function(functional.function_space().dual())
    with functional.dat.vec as x, function.dat.vec as y:
        helicity_solver.snes.ksp.pc.apply(x, y)
    return function

def compute_potential(B):
    """
    Compute A such that curl(A) approximates B. Uses the helicity solver and a lifting
    trick if the solver residual is large (your original strategy).
    """
    # Solve the helicity problem (internal to helicity_solver)
    helicity_solver.solve()
    problem = helicity_solver._problem
    # check residual; if large, perform lifting
    if helicity_solver.snes.ksp.getResidualNorm() > 0.01:
        r = assemble(problem.F, bcs=problem.bcs)
        rstar = r.riesz_representation(riesz_map=riesz_map, bcs=problem.bcs)
        c = assemble(action(r, problem.u)) / assemble(action(r, rstar))
        ulft = Function(Vc, name="u_lifted")
        ulft.assign(problem.u - c * rstar)
        A = ulft
    else:
        A = problem.u

    diff = norm(curl(A) - B, "L2")
    if mesh.comm.rank == 0:
        print(f"[compute_potential] ||curl(A)-B||_L2 = {diff:.8e}", flush=True)
    A_ = Function(Vc, name="MagneticPotential")
    A_.project(A)
    return A_

def potential_solver_direct(B):
    """
    Alternative direct nonlinear solve for curl-curl A = curl^{-1} B (kept as in original)
    """
    Afunc = Function(Vc)
    v = TestFunction(Vc)
    F_curl  = inner(curl(Afunc), curl(v)) * dx - inner(B, curl(v)) * dx

    sp_helicity = {  
           "ksp_type":"gmres",
           "pc_type": "ilu",
    }
    bcs_curl = [DirichletBC(Vc, 0, sub) for sub in DIRICHLET_IDS]
    solver = build_nonlinear_solver(F_curl, Afunc, bcs_curl, solver_parameters = sp_helicity, options_prefix="solver_curlcurl")
    solver.solve()
    return Afunc

# ---------------------------
# Initialize z_prev: project B and compute A at t=0
# ---------------------------
proj_B0 = project_initial_conditions(B_init_expr)
z_prev.sub(0).project(proj_B0)
z_prev.sub(2).project(compute_potential(proj_B0))
z.assign(z_prev)  # initialize current solution

# ---------------------------
# Time-discrete weak form (as in your original code)
# ---------------------------
dt = Constant(DT)
t = Constant(0.0)
tau = Constant(TAU)


def form_energy(B):
    return dot(B, B)

def form_dissipation(B, j):
    return 2 * tau * inner(cross(B, j), cross(B, j)) 


def form_helicity(A, B):
    if periodic:
        harmonic = Function(Vd)
        harmonic.project(B - curl(A))
        return dot(A, B + harmonic)
    else:
        return dot(A, B)

B_avg = B
#B_avg = (B + Bp)/2
#A_avg = (A + Ap)/2

F = (
        inner((B - Bp)/dt, Bt) * dx
    + inner(curl(E), Bt) * dx
    + 2 * lmbda_e * inner(B_avg, Bt) * dx # LM for energy
    
    + inner((A-Ap)/dt, At) * dx
    + inner(E, At) * dx
    + 2 * lmbda_m * inner(B_avg, At) * dx # LM for helicity

    + inner(E, Et) * dx
    + inner(cross(u, B_avg), Et) * dx

    + inner(u, ut) * dx
    - tau * inner(cross(j, B_avg), ut) * dx
    
    + inner(j, jt) * dx
    - inner(B_avg, curl(jt)) * dx

    # energy law
    + 1/dt * inner(form_energy(B) - form_energy(Bp), lmbda_et) * dx
    + inner(form_dissipation(B_avg, j), lmbda_et) * dx
    # helicity 
    + 1/dt * inner(form_helicity(A, B) - form_helicity(Ap, Bp), lmbda_mt) * dx
)

# Build BCs for the full mixed system Z
bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)-2) for subdomain in DIRICHLET_IDS]
#if not closed:
#bcs += [DirichletBC(Z.sub(0), B_init_bc_t, "top"),
#           DirichletBC(Z.sub(0), B_init_bc_b, "bottom")]
# Nonlinear solver params you used originally
lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "ksp_type":"preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type":"mumps"
}

fs = {
    "mat_type": "matfree",
    "snes_monitor": None, 
    "ksp_type": "fgmres",
    "ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_0_fields": "0, 1, 2, 3, 4",
    "pc_fieldsplit_1_fields": "5, 6",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "python",
#"ksp_monitor": None,
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "lu",
        "assembled_pc_factor_mat_solver_type": "mumps",
    },
    "fieldsplit_1": {
        "ksp_type": "gmres",
#"ksp_monitor": None,
        "pc_type": "none",
        "ksp_max_it": 2, 
        "ksp_convergence_test": "skip",

    },

}
sp = None
pb = NonlinearVariationalProblem(F, z, bcs=bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

# ---------------------------
# Diagnostics: helicity, divB, energy
# ---------------------------
def compute_helicity(A_func, B_func):
    if periodic:
        harmonic = Function(Vd)
        harmonic.project(B_func - curl(A_func))
        diff = norm(harmonic,"L2")
        # generalized helicity
        return assemble(inner(A_func, B_func + harmonic) * dx)
    else:
        A = potential_solver_direct(B_func)
        return assemble(inner(A, B_func) * dx)

def compute_divB(B_func):
    return norm(div(B_func), "L2")

def compute_energy(B_func, A_func):
    if periodic:
        harmonic = Function(Vd)
        harmonic.project(B_func - curl(A_func))
        return assemble(inner(B_func - harmonic, B_func - harmonic) * dx)
    else:
        return assemble(inner(B_func, B_func) * dx)

# ---------------------------
# Prepare output directory and CSV/log file
# ---------------------------
pvd = VTKFile(PVD_FILENAME)

pvd.write(*z.subfunctions, time=float(t))


fieldnames = ["t", "helicity", "energy", "divB", "j_max", "lambda_max", "xi_max", "beta", "helicity_GV", "helicity_r"]
if mesh.comm.rank == 0:
    with open(DATA_FILENAME, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


# Save t=0 diagnostics
helicity = compute_helicity(z.sub(2), z.sub(0)) # A, B
divB = compute_divB(z.sub(0)) # B
energy = compute_energy(z.sub(0), z.sub(2)) # B , A

# ------------------ Time-stepping loop (keep t, dt as Constant; use floats for control) ------------------

# Python-side variables for loop control and comparison
t_val = 0.0                       # current physical time (float)
initial_dt_val = float(DT)        # initial step size
increase_dt_after = 20            # after how many steps to increase dt
big_dt_val = 100.0                # larger step size after the threshold

# ensure Firedrake Constants start from consistent values
t.assign(0.0)
dt.assign(initial_dt_val)

timestep = 0
# write the initial fields
if mesh.comm.rank == 0:
    row = {
        "t": float(t_val),
        "helicity": float(helicity),
        "energy": float(energy),
        "divB": float(divB),
    }
    with open(DATA_FILENAME, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

while t_val < T_FINAL - 1e-12:
    # determine desired step size (float)
    desired_dt = initial_dt_val if (timestep <= increase_dt_after) else big_dt_val

    # do not exceed the remaining time to reach T_FINAL
    remaining = T_FINAL - t_val
    next_dt = min(desired_dt, remaining)

    # synchronize Firedrake Constant dt with the actual float value
    dt.assign(next_dt)

    # advance the UFL Constant t to the new physical time
    t.assign(t_val + next_dt)

    if mesh.comm.rank == 0:
        print(f"[Time] Step {timestep:4d}: advancing t {t_val:.6f} -> {t_val + next_dt:.6f} (dt={next_dt})", flush=True)

    # solve the nonlinear system
    solver.solve()

    # update previous solution
    z_prev.assign(z)

    # update Python-side time tracker
    t_val += next_dt

    # diagnostics
    helicity = compute_helicity(z.sub(2), z.sub(0))
    divB = compute_divB(z.sub(0))
    energy = compute_energy(z.sub(0), z.sub(2))

    if mesh.comm.rank == 0:
        row = {
        "t": float(t_val),
        "helicity": float(helicity),
        "energy": float(energy),
        "divB": float(divB),
        }
        with open(DATA_FILENAME, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
        print(f"[Diagnostics] t={t_val:.6f} helicity={helicity:.6e}, energy={energy:.6e}, divB={divB:.6e}", flush=True)

    # write VTK/PVD output (stored in 'output/' folder)
    pvd.write(*z.subfunctions, time=float(t_val))
    timestep += 1

# ------------------ End of time-stepping loop ------------------

