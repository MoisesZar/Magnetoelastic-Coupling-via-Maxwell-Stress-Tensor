import numpy as np
import os
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl

# -------------------------
# Constantes del material
# -------------------------
n0 = 0.8388e-9
dNP = int(14e-9/n0)*n0
Num_x, Num_y, Num_z = 25, 25, 25
dx, dy, dz = dNP*Num_x, dNP*Num_y, dNP*Num_z
E = 9.5e9
nu = 0.4
mu = E/(2*(1+nu))
lamb = E*nu/((1+nu)*(1-2*nu))

# -------------------------
# Definir dominio
# -------------------------
domain = mesh.create_box(MPI.COMM_WORLD,
                         [[0,0,0],[dx,dy,dz]],
                         [Num_x, Num_y, Num_z],
                         cell_type=mesh.CellType.hexahedron)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
Q = fem.functionspace(domain, ("DG", 0, (domain.geometry.dim,)))

# -------------------------
# Condiciones de borde
# -------------------------
def clamped_boundary(x):
    return np.isclose(x[2], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0,0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
T = fem.Constant(domain, default_scalar_type((0,0,0)))
ds = ufl.Measure("ds", domain=domain)

# -------------------------
# Funciones helper
# -------------------------
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lamb*ufl.nabla_div(u)*ufl.Identity(len(u)) + 2*mu*epsilon(u)

# -------------------------
# Lista de directorios de kappa
# -------------------------
kappa_dirs = sorted([d for d in os.listdir("fuerzas_kappa") if d.startswith("kappa_")])
print(f"📂 Directorios encontrados: {kappa_dirs}")

# -------------------------
# Archivo de salida único con columna para φ
# -------------------------
out_file = open("epsilons_todas_kappa.txt", "w")
header = ("kappa\tB\tExx_global\tEyy_global\tEzz_global\t" "Exx_x0\tEyy_x0\tEzz_x0\tExx_y0\tEyy_y0\tEzz_y0\t" "Exx_zmax\tEyy_zmax\tEzz_zmax\n")
out_file.write(header)

# -------------------------
# Loop sobre kappa
# -------------------------
for kappa_dir in kappa_dirs:
    kappa = float(kappa_dir.split('_')[1])
    print(f"\n🔹 Procesando κ = {kappa}")
    
    # Lista de archivos Force para esta kappa
    force_files = sorted([f for f in os.listdir(f"fuerzas_kappa/{kappa_dir}") 
                      if f.startswith("Force") and f.endswith(".dat")]) 
    
    # Buscar el archivo correspondiente a B=0
    force0_file = None
    for f in force_files:
        B_str = f.split('_Bx_')[-1].replace('.dat', '').replace('_', '.')
        B_str = B_str.replace('neg', '-')
        B = float(B_str)
        if abs(B) < 1e-12:
            force0_file = f
            break
        
    
    if force0_file is None:
        print(f"  ⚠️ No se encontró archivo para B=0 en κ={kappa}, usando el primer archivo como referencia")
        force0_file = force_files[0]
    
    print(f"  📍 Referencia (B=0): {force0_file}")
    Force0 = np.loadtxt(f"fuerzas_kappa/{kappa_dir}/{force0_file}", skiprows=1)
    
    for ffile in force_files:
        # Saltar el archivo de referencia (B=0) si no queremos compararlo consigo mismo
        if ffile == force0_file:
            print(f"  ⏭️  Saltando archivo de referencia: {ffile}")
            continue
        
        # Extraer B del nombre
        B_str = ffile.split('_Bx_')[-1].replace('.dat', '').replace('_', '.')
        B_str = B_str.replace('neg', '-')
        B = float(B_str)
        print(f"  📊 B = {B:.2f}")
        
        # Cargar Force (campo aplicado)
    
        Force = np.loadtxt(f"fuerzas_kappa/{kappa_dir}/{ffile}", skiprows=1)


        
        # --- Resolver con Force (campo aplicado) ---
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = fem.Function(Q)
        f.x.array[:] = Force.flatten()
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
        problem = LinearProblem(a, L, bcs=[bc],
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                petsc_options_prefix="elasticity_field")
        uh = problem.solve()
        
        # --- Resolver con Force0 (referencia B=0) ---
        f.x.array[:] = Force0.flatten()
        problem = LinearProblem(a, L, bcs=[bc],
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                petsc_options_prefix="elasticity_ref")
        uh0 = problem.solve()
        
        # Diferencia (deformación relativa al caso sin campo)
        uhdiff = fem.Function(V)
        uhdiff.x.array[:] = uh.x.array - uh0.x.array
        
        # Tensor de deformación
        eps = epsilon(uhdiff)
        V_eps = fem.functionspace(domain, ("DG", 0))
        
        # Calcular componentes
        eps_xx = fem.Function(V_eps)
        eps_yy = fem.Function(V_eps)
        eps_zz = fem.Function(V_eps)
        eps_xx.interpolate(fem.Expression(eps[0,0], V_eps.element.interpolation_points))
        eps_yy.interpolate(fem.Expression(eps[1,1], V_eps.element.interpolation_points))
        eps_zz.interpolate(fem.Expression(eps[2,2], V_eps.element.interpolation_points))
        
        # Promedios globales
        mean_exx = np.mean(eps_xx.x.array)
        mean_eyy = np.mean(eps_yy.x.array)
        mean_ezz = np.mean(eps_zz.x.array)
        
        # Coordenadas de centroides
        cell_midpoints = np.array([
            domain.geometry.x[domain.topology.connectivity(domain.topology.dim, 0).links(c)].mean(axis=0)
            for c in range(domain.topology.index_map(domain.topology.dim).size_local)
        ])
        eps_xx_vals = eps_xx.x.array
        eps_yy_vals = eps_yy.x.array
        eps_zz_vals = eps_zz.x.array
        
        xmin, ymin, zmin = np.min(cell_midpoints, axis=0)
        xmax, ymax, zmax = np.max(cell_midpoints, axis=0)
        tol = 0.05
        
        # Máscaras para planos
        mask_x0 = np.abs(cell_midpoints[:,0] - xmin) < tol*(xmax-xmin)
        mask_y0 = np.abs(cell_midpoints[:,1] - ymin) < tol*(ymax-ymin)
        mask_zmax = np.abs(cell_midpoints[:,2] - zmax) < tol*(zmax-zmin)
        
        # Promedios en planos
        def safe_mean(vals, mask):
            return np.mean(vals[mask]) if np.any(mask) else np.nan
        
        exx_x0 = safe_mean(eps_xx_vals, mask_x0)
        eyy_x0 = safe_mean(eps_yy_vals, mask_x0)
        ezz_x0 = safe_mean(eps_zz_vals, mask_x0)
        
        exx_y0 = safe_mean(eps_xx_vals, mask_y0)
        eyy_y0 = safe_mean(eps_yy_vals, mask_y0)
        ezz_y0 = safe_mean(eps_zz_vals, mask_y0)
        
        exx_zmax = safe_mean(eps_xx_vals, mask_zmax)
        eyy_zmax = safe_mean(eps_yy_vals, mask_zmax)
        ezz_zmax = safe_mean(eps_zz_vals, mask_zmax)
        
        # Escribir resultados
        out_file.write(f"{kappa:.6f}\t{B:.6e}\t{mean_exx:.6e}\t{mean_eyy:.6e}\t{mean_ezz:.6e}\t"
                       f"{exx_x0:.6e}\t{eyy_x0:.6e}\t{ezz_x0:.6e}\t"
                       f"{exx_y0:.6e}\t{eyy_y0:.6e}\t{ezz_y0:.6e}\t"
                       f"{exx_zmax:.6e}\t{eyy_zmax:.6e}\t{ezz_zmax:.6e}\n")

out_file.close()
print("✅ Archivo epsilons_todas_kappa.txt generado con referencia a B=0 para cada κ")
