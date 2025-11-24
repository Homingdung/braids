# use Lagrange multiplier
from firedrake import *
import csv
import os
import sys

# parameters 
output = True
ic = "hopf" # hopf or E3 
# closed: periodic = False, closed = True
# periodic: periodic = True, closed = True
# line-tied: periodic = False, closed = False
bc = "closed"

if bc == "line-tied":
    periodic = False
    closed = False

elif bc == "closed":
    periodic = False
    closed = True

elif bc == "periodic":
    periodic = True # no top and bottom label
    closed = True

time_discr = "adaptive" # uniform or adaptive

if ic == "hopf":
    Lx, Ly, Lz = 8, 8, 20
    Nx, Ny, Nz = 4, 4, 10
elif ic == "E3":
    Lx, Ly, Lz = 8, 8, 48
    Nx, Ny, Nz = 4, 4, 24


if periodic:
    dirichlet_ids = ("on_boundary",)
else:
    dirichlet_ids = ("on_boundary", "top", "bottom")


order = 1  # polynomial degree
tau = Constant(1)
t = Constant(0)
dt = Constant(1)
T = 10000

base = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(base, Lz, 1, periodic=periodic)
mesh.coordinates.dat.data[:, 0] -= Lx/2
mesh.coordinates.dat.data[:, 1] -= Ly/2
mesh.coordinates.dat.data[:, 2] -= Lz/2

Vg = VectorFunctionSpace(mesh, "Q", order)
Vg_ = FunctionSpace(mesh, "Q", order)
Vc = FunctionSpace(mesh, "NCE", order)
Vd = FunctionSpace(mesh, "NCF", order)
Vn = FunctionSpace(mesh, "DQ", order-1)
VR = FunctionSpace(mesh, "R", 0)

# Mixed unknowns: [B, u, A, E, lmbda_e, lmbda_m]
Z = MixedFunctionSpace([Vd, Vd, Vc, Vc, Vc, VR, VR])
z = Function(Z)
z_prev = Function(Z)
z_test = TestFunction(Z)
(B, u, A, E, j, lmbda_e, lmbda_m) = split(z)
(Bt, ut, At, Et, jt, lmbda_et, lmbda_mt) = split(z_test)
(Bp, up, Ap, Ep, jp, lmbda_ep, lmbda_mp) = split(z_prev)

(X0, Y0, Z0) = x = SpatialCoordinate(mesh)

# Hopf fibre
if ic == "hopf":
    w1 = 3
    w2 = 2
    s = 1
    deno = 1 + dot(x, x)
    coeff = 4*sqrt(s)/((pi * deno * deno * deno)*sqrt(w1**2+w2**2))
    B_init = as_vector([coeff*2*(w2*Y0-w1*X0*Z0), -coeff*2*(w2*X0+w1*Y0*Z0), coeff*w1*(-1+X0**2+Y0**2-Z0**2)])

elif ic == "E3":
    x_c = [1, -1, 1, -1, 1, -1]
    y_c = 0
    z_c = [-20, -12, -4, 4, 12, 20]
    a = sqrt(2)
    # strength of twist
    k = 5
    l = 2
    B_0 = 1

    B_z = B_0
    B_x = 0
    B_y = 0
    # background magnetic field
    B_b = as_vector([0, 0, B_0])
    for i in range(6):
        coeff = exp((-(X0-x_c[i])**2/(a**2)) - ((Y0 - y_c)**2/(a**2)) - ((Z0 - z_c[i])**2/(l**2))) 
        B_x += coeff * ((2 * k * B_0/a) * (-(Y0-y_c)))
        B_y += coeff * ((2 * k * B_0/a) * ((X0-x_c[i])))

    B_init = as_vector([B_x, B_y, B_z]) - B_b

# Convenient references to subfunctions
(B_, u_, A_, E_, j_, lmbda_e_, lmbda_m_) = z.subfunctions
B_.rename("MagneticField")
E_.rename("ElectricField")
u_.rename("Velocity")
A_.rename("MagneticPotential")
j_.rename("Current")


def project_initial_conditions(B_init):
    Zp = MixedFunctionSpace([Vd, Vn])
    zp = Function(Zp)
    (Bp_proj, p) = split(zp)
    test_B, test_p = TestFunctions(Zp)
    
    # not bc for p
    bcs_proj = [DirichletBC(Zp.sub(0), 0, sub) for sub in dirichlet_ids]
#if not closed:
#        bcs_proj += [
#                     DirichletBC(Zp.sub(0), B_init_bc_t, "top"),
##                     DirichletBC(Zp.sub(0), B_init_bc_b, "bottom"),
#        ]

    L = (
        0.5*inner(Bp_proj, Bp_proj)*dx
        - inner(B_init, Bp_proj)*dx
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

B_recover = Function(Vd, name="RecoverdMagneticField")
if output:
    pvd = VTKFile("output/parker.pvd")
    pvd.write(*z.subfunctions, time=float(t))
    if ic == "E3" and bc == "closed":
        pvd1 = VTKFile("output/recover.pvd")
        B_recover.project(z.sub(0) + B_b)
        pvd1.write(B_recover, time=float(t))

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
    bcs_curl = [DirichletBC(Vc, 0, sub) for sub in dirichlet_ids]
    solver = build_nonlinear_solver(F_curl, Afunc, bcs_curl, solver_parameters = sp_helicity, options_prefix="solver_curlcurl")
    solver.solve()
    return Afunc


proj_B0 = project_initial_conditions(B_init)
z_prev.sub(0).project(proj_B0)
z_prev.sub(2).project(potential_solver_direct(proj_B0))
z.assign(z_prev)  # initialize current solution

def build_helicity_solver(B):
    # Solve curl-curl u = curl^{-1} B  (weak: curl(u), curl(v) = <B, curl(v)> )
    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    u_sol = Function(Vc)

    a = inner(curl(u), curl(v)) * dx
    #B_proj = project_initial_conditions(B_init)
    L = inner(B, curl(v)) * dx

    # small regularization for kernel
    beta = Constant(0.1)
    Jp_curl = a + inner(beta * u, v) * dx

    bcs_curl = [DirichletBC(Vc, 0, sub) for sub in dirichlet_ids]

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

helicity_solver = build_helicity_solver(B)

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
bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)-2) for subdomain in dirichlet_ids]
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

sp = None
pb = NonlinearVariationalProblem(F, z, bcs=bcs)
time_stepper = NonlinearVariationalSolver(pb, solver_parameters = sp)

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
        return assemble(inner(A_func, B_func) * dx)

def compute_divB(B_func):
    return norm(div(B_func), "L2")

def compute_energy(B_func, A_func):
    if periodic:
        harmonic = Function(Vd)
        harmonic.project(B_func - curl(A_func))
        return assemble(inner(B_func - harmonic, B_func - harmonic) * dx)
    else:
        return assemble(inner(B_func, B_func) * dx)

# define files
data_filename = "output/data.csv"
fieldnames = ["t", "helicity", "energy", "divB"]
if mesh.comm.rank == 0:
    with open(data_filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Save t=0 diagnostics
helicity = compute_helicity(z.sub(2), z.sub(0)) # A, B
divB = compute_divB(z.sub(0)) # B
energy = compute_energy(z.sub(0), z.sub(2)) # B , A

# ------------------ Time-stepping loop (keep t, dt as Constant; use floats for control) ------------------
timestep = 0
# write the initial fields
if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "helicity": float(helicity),
        "energy": float(energy),
        "divB": float(divB),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
        print(f"{row}")

while (float(t) < float(T) + 1.0e-10):
    if float(t) + float(dt) > float(T):
        dt.assign(T - t)
    if float(dt)<=1e-14:
        break
    t.assign(t + dt)
    if mesh.comm.rank == 0:
        print(RED % f"Solving for t = {float(t):.4f}, dofs = {Z.dim()}, initial condition = {ic}, time discretisation = {time_discr}, dt={float(dt)}, T={T}, bc={bc}", flush=True)
    
    time_stepper.solve()
    
    helicity = compute_helicity(z.sub(2), z.sub(0)) # A, B
    divB = compute_divB(z.sub(0)) # B
    energy = compute_energy(z.sub(0), z.sub(2)) # B , A


    if time_discr == "adaptive":
        #E_new = compute_energy(z.sub(0), diff)
        #dE = abs(E_new-E_old) / E_old
        if timestep > 100:
            dt.assign(100)
            tau.assign(0.1)
    
    if mesh.comm.rank == 0:
        row = {
            "t": float(t),
            "helicity": float(helicity),
            "energy": float(energy),
            "divB": float(divB),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            print(f"{row}")

    if output:
        #if timestep % 10 == 0:
        pvd.write(*z.subfunctions,time=float(t))
        if ic == "E3" and bc == "closed":
            B_recover.project(z.sub(0) + B_b)
            pvd1.write(B_recover, time=float(t))
    timestep += 1
    z_prev.assign(z)




