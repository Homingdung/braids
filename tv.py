# avfet method for relaxation 
from firedrake import *
import csv
import os
import sys
from ufl import atan2   
import numpy as np
# complete elliptic integral of the first and the second kind
from scipy.special import ellipk, ellipe
# parameters 
output = True
ic = "tv" # hopf or E3 
bc = "closed"

if bc == "line-tied":
    periodic = False

elif bc == "closed":
    periodic = False

elif bc == "periodic":
    periodic = True # no top and bottom label

time_discr = "adaptive" # uniform or adaptive

if ic == "hopf":
    Lx, Ly, Lz = 8, 8, 20
    Nx, Ny, Nz = 8, 8, 10
elif ic == "E3":
    Lx, Ly, Lz = 8, 8, 48
    Nx, Ny, Nz = 4, 4, 24

elif ic == "tv":
    Lx, Ly, Lz = 8, 8, 8
    Nx, Ny, Nz = 8, 8, 8

if periodic:
    dirichlet_ids = ("on_boundary",)
else:
    dirichlet_ids = ("on_boundary", "top", "bottom")


order = 1  # polynomial degree
tau = Constant(10)
t = Constant(0)
dt = Constant(1)
T = 10000

base = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(base, Lz, 1, periodic=periodic)
mesh.coordinates.dat.data[:, 0] -= Lx/2
mesh.coordinates.dat.data[:, 1] -= Ly/2
#mesh.coordinates.dat.data[:, 2] -= Lz/2

Vg = VectorFunctionSpace(mesh, "Q", order)
Vg_ = FunctionSpace(mesh, "Q", order)
Vc = FunctionSpace(mesh, "NCE", order)
Vd = FunctionSpace(mesh, "NCF", order)
Vn = FunctionSpace(mesh, "DQ", order-1)
Real = FunctionSpace(mesh, "R", 0)
zero = Function(Real).assign(0)

# Mixed unknowns: [B, j, H, u, E]
Z = MixedFunctionSpace([Vd, Vc, Vc, Vd, Vc])
z = Function(Z)
(B , j,  H, u, E) = split(z)
(Bt, jt, Ht, ut, Et) = split(TestFunction(Z))

z_prev = Function(Z)
(Bp, jp, Hp, up, Ep) = split(z_prev)
B_avg = (B + Bp)/2
E_avg = E
#E_avg = (E + Ep)/2
H_avg = H
#H_avg = (H + Hp)/2  # or H?
j_avg = j
#j_avg = (j + jp)/2  # or j?
u_avg = u
#u_avg = (u + up)/2  # or u?


X0, Y0, Z0 = SpatialCoordinate(mesh)
# constants
mu0 = 1.0
I0  = 1.0
I   = 1.0
R   = 2.0
a   = 0.2
d   = 0.0
q   = 1.0
L   = 0.3
eps = 1e-12
pi  = np.pi


def A_I_compute(X0, Y0, Z0):
    # r_perp
    r_perp = np.sqrt(Y0**2 + (Z0 + d)**2)
    r_perp_safe = np.sqrt(r_perp**2 + eps)

    # rho
    rho_sq = X0**2 + (r_perp - R)**2
    rho = np.sqrt(rho_sq)
    rho_safe = np.sqrt(rho_sq + eps)

    # chi
    chi = np.where(a - rho > 0.0, 1.0, 0.0)

    # hat_theta
    hat_theta = np.stack((
        np.zeros_like(X0),
        -(Z0 + d) / r_perp_safe,
        Y0 / r_perp_safe
    ), axis=-1)

    # elliptic parameters
    k_a = 2 * np.sqrt(r_perp_safe * R / (4 * r_perp_safe * R + a**2))
    k   = 2 * np.sqrt(r_perp_safe * R / ((r_perp_safe + R)**2 + X0**2))

    # elliptic integrals
    K_ellip = ellipk(k**2)   # SciPy uses m = k^2
    E_ellip = ellipe(k**2)

    # A_cal and A_diff
    A_cal = (1 / k) * ((2 - k**2) * K_ellip - 2 * E_ellip)
    A_diff = ((2 - k**2) * E_ellip - 2 * (1 - k**2) * K_ellip) / (k**2 * (1 - k**2))

    # inside / outside
    A_I_ex = mu0 * I / (2 * pi) * np.sqrt(R / r_perp_safe) * A_cal
    A_I_in = mu0 * I / (2 * pi) * np.sqrt(R / r_perp_safe) * (A_cal + A_diff * (k - k_a))

    # final A
    A_exp = chi * A_I_in + (1 - chi) * A_I_ex

    return A_exp[..., None] * hat_theta

m = Vg_.mesh()
W = VectorFunctionSpace(m, Vg_.ufl_element())
X = assemble(project(m.coordinates, W))

A_I = Function(Vg)
A_I.dat.data[:] = A_I_compute(*X.dat.data_ro.T)
A_init = Function(Vc).project(A_I)
B_I = curl(A_init)

r_perp      = sqrt(Y0**2 + (Z0 + d)**2)
r_perp_safe = sqrt(r_perp**2 + eps)

rho      = sqrt(X0**2 + (r_perp - R)**2)
rho_safe = sqrt(rho**2 + eps)

# Heaviside chi(a - rho)
chi_ = conditional(a - rho > 0.0, 1.0, 0.0)

# ---------------------------------------------------------------
#  Inside the square root (Titov eq. 16)
# ---------------------------------------------------------------
inside_sqrt = (1.0 / R**2) + \
              2.0 * chi_ * ((a - rho) / a**2) * (I**2 / I0**2) * (1.0 - (rho**2)/(a**2))

inside_sqrt_safe = conditional(inside_sqrt > 0.0, inside_sqrt, 0.0)
term1 = sqrt(inside_sqrt_safe)

term2 = 1.0 / r_perp_safe

# B_theta
prefactor = mu0 * I0 / (2.0 * pi)
B_theta = prefactor * (term1 + term2 - (1.0 / R))

hat_theta = as_vector((
    0.0,
    -(Z0 + d) / r_perp_safe,
    Y0 / r_perp_safe
))

# B_theta
B_theta = B_theta * hat_theta

# B_q
r_p = as_vector([X0 - L, Y0, Z0 + d])
r_m = as_vector([X0 + L, Y0, Z0 + d])
r_p_3 = sqrt(dot(r_p, r_p)) ** 3
r_m_3 = sqrt(dot(r_m, r_m)) ** 3

B_q = q * (r_p/r_p_3 - r_m/r_m_3)
B_init = B_I + B_theta + B_q

B_init = Function(Vd).project(B_init, form_compiler_parameters={"quadrature_degree": 12})

F = (
      inner((B-Bp)/dt, Bt) * dx
    + inner(curl(E_avg), Bt) * dx
    - inner(B_avg, curl(jt)) * dx
    + inner(j_avg, jt) * dx
    - inner(cross(Et, H_avg), u) * dx
    + inner(E_avg, Et) * dx
    + inner(u_avg, ut) * dx
    - tau * inner(cross(j_avg, H_avg)/(dot(Bp, Bp)+1e-5), ut) * dx
    + inner(H_avg, Ht) * dx
    - inner(B_avg, Ht) * dx
    )

# Boundary conditions
# bcs for top and bottom
B_init_bc = as_vector([0, 0, 1])

bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids]
if bcs == "line-tied":
    bcs += DirichletBC(Z.sub(0), B_init_bc, "top")
    bcs += DirichletBC(Z.sub(0), B_init_bc, "bottom")

lu = {
	"mat_type": "aij",
	"snes_type": "newtonls",
        "snes_monitor": None,
        "ksp_monitor": None,
        "ksp_type":"preonly",
	"pc_type": "lu",
        "pc_factor_mat_solver_type":"mumps"
}
sp = lu
       
(B_, j_, H_, u_, E_) = z.subfunctions
B_.rename("MagneticField")
E_.rename("ElectricField")
H_.rename("HCurlMagneticField")
j_.rename("Current")
u_.rename("Velocity")

def project_initial_conditions(B_init):
    # Need to project the initial conditions
    # such that div(B) = 0 and B·n = 0
    Zp = MixedFunctionSpace([Vd, Vn])
    zp = Function(Zp)
    (B, p) = split(zp)
    if bcs == "line-tied":
        bcp = [
                DirichletBC(Zp.sub(0), 0, "on_boundary"), 
                DirichletBC(Zp.sub(0), B_init_bc, "top"),
                DirichletBC(Zp.sub(0), B_init_bc, "bottom")
        ]
    else:
        bcp = [DirichletBC(Zp.sub(0), 0, subdomain) for subdomain in dirichlet_ids]
        # Write Lagrangian
    L = (
          0.5*inner(B, B)*dx
        - inner(B_init, B)*dx
        - inner(p, div(B))*dx
        )

    Fp = derivative(L, zp, TestFunction(Zp))
    spp = {
        "mat_type": "nest",
        "snes_type": "ksponly",
        "snes_monitor": None,
        "ksp_monitor": None,
        "ksp_max_it": 1000,
        "ksp_norm_type": "preconditioned",
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
    gamma = Constant(1E5)
    Up = 0.5*(inner(B, B) + inner(div(B) * gamma, div(B)) + inner(p * (1/gamma), p))*dx
    Jp = derivative(derivative(Up, zp), zp)
    solve(Fp == 0, zp, bcp, Jp=Jp, solver_parameters=spp,
            options_prefix="B_init_div_free_projection")
    return zp.subfunctions[0]

B_.assign(project_initial_conditions(B_init))
z_prev.assign(z)

B_recover = Function(Vd, name="RecoveredMagneticField")
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

def build_nonlinear_solver(F, z, bcs, Jp=None, solver_parameters = None, options_prefix=None):
    problem = NonlinearVariationalProblem(F, z, bcs, Jp=Jp)
    solver = NonlinearVariationalSolver(problem,
                solver_parameters=solver_parameters,
                options_prefix=options_prefix)
    return solver


def helicity_solver():
    # Spaces for magnetic potential computation
    # If using periodic boundary conditions, we need to modify
    # this to account for the harmonic form [0, 0, 1]^T
    # using Yang's solver

    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    u_sol = Function(Vc)

    # weak form of curl-curl problem 
    a = inner(curl(u), curl(v)) * dx
    L = inner(B_, curl(v)) * dx
    beta = Constant(0.1)
    Jp_curl = a + inner(beta * u, v) * dx
    bcs_curl = [DirichletBC(Vc, 0, subdomain) for subdomain in dirichlet_ids]
    rtol = 1E-8
    preconditioner = True
    if preconditioner:
        pc_type = "lu"
    else:
        pc_type = "none"
    sparams = {
        "snes_type": "ksponly",
        # "ksp_type": "lsqr",
        "ksp_type": "minres",
        "ksp_max_it": 1000,
        "ksp_convergence_test": "skip",
        #"ksp_monitor": None,
        "pc_type": pc_type,
        "ksp_norm_type": "preconditioned",
        "ksp_minres_nutol": 1E-8,
        }

    solver = build_linear_solver(a, L, u_sol, bcs_curl, Jp_curl, sparams, options_prefix="helicity")
    return solver


helicity_solver = helicity_solver()

def riesz_map(functional):
    function = Function(functional.function_space().dual())
    with functional.dat.vec as x, function.dat.vec as y:
        helicity_solver.snes.ksp.pc.apply(x, y)
    return function


def compute_helicity_energy(B):
    helicity_solver.solve()
    problem = helicity_solver._problem
    if helicity_solver.snes.ksp.getResidualNorm() > 0.01:
        # lifting strategy
        r = assemble(problem.F, bcs=problem.bcs)
        rstar = r.riesz_representation(riesz_map=riesz_map, bcs=problem.bcs)
        rstar.rename("RHS")
        # lft = uh - inner(r, uh)/inner(r, rstar) * rstar
        c = assemble(action(r, problem.u)) / assemble(action(r, rstar))
        ulft = Function(Vc, name="u_lifted")
        ulft.assign(problem.u - c * rstar)
        A = ulft
    else:
        A = problem.u
    diff = norm(curl(A) - B, "L2")
    if mesh.comm.rank == 0:
        print(f"magnetic potential: ||curl(A) - B||_L2 = {diff:.8e}", flush=True)
    A_ = Function(Vc, name="MagneticPotential")
    A_.project(A)
    curlA = Function(Vd, name="CurlA")
    curlA.project(curl(A))
    diff_ = Function(Vd, name="CurlAMinusB")
    diff_.project(B-curlA)
    #VTKFile("output/magnetic_potential.pvd").write(curlA, diff_, A_)
    if bc=="closed":
        return assemble(inner(A, B)*dx), diff, diff_, assemble(inner(B, B) * dx)
    else: 
        return assemble(inner(A, B + diff_)*dx), diff, diff_, assemble(inner(B - diff_, B-diff_) * dx)
       
def compute_Bn(B):
    n = FacetNormal(mesh)
    return assemble(inner(dot(B, n), dot(B, n))*ds_v)

def compute_divB(B):
    return norm(div(B), "L2")

# solver
time_stepper = build_nonlinear_solver(F, z, bcs, solver_parameters=sp, options_prefix="time_stepper")

# define files
data_filename = "output/data.csv"
fieldnames = ["t", "helicity", "energy", "divB"]
if mesh.comm.rank == 0:
    with open(data_filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# store the initial value
helicity, diff, diff_, energy = compute_helicity_energy(z.sub(0))
divB = compute_divB(z.sub(0))


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


timestep = 0
#E_old = compute_energy(z_prev.sub(0), diff_)

while (float(t) < float(T) + 1.0e-10):
    if float(t) + float(dt) > float(T):
        dt.assign(T - float(t))
    if float(dt) <=1e-14:
        break
    t.assign(t + dt)
    if mesh.comm.rank == 0:
        print(RED % f"Solving for t = {float(t):.4f}, dofs = {Z.dim()}, initial condition = {ic}, time discretisation = {time_discr}, dt={float(dt)}, T={T}, bc={bc}", flush=True)
    
    time_stepper.solve()
    
    # monitor
    helicity, diff, diff_, energy= compute_helicity_energy(z.sub(0))
    divB = compute_divB(z.sub(0))

    if time_discr == "adaptive":
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
            B_b = as_vector([0, 0, 1])
            B_recover.project(z.sub(0) + B_b)
            pvd1.write(B_recover, time=float(t))
    timestep += 1
    z_prev.assign(z)



