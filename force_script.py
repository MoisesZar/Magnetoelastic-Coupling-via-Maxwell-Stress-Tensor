import oommfc as mc
import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import tqdm
import os

# ----------------------------
# Constantes
# ----------------------------
n0 = 0.8388e-9
alpha = 0.5

dNP = int(14e-9/n0)*n0
Num_x, Num_y, Num_z = 25, 25, 25
dx, dy, dz = dNP*Num_x, dNP*Num_y, dNP*Num_z

Ms = 4*mm.consts.muB*6.022e23*5170*1000/234.15
def Ms_value(point):
    x, y, z = point
    if (dNP<x<dx-dNP) and (dNP<y<dy-dNP) and (dNP<z<dx-dNP):
        return Ms
    else:
        return 0

# ----------------------------
# Valores de kappa
# ----------------------------
kappa_values = [0.1, 0.5, 1, 2,5]

# ----------------------------
# Crear directorio
# ----------------------------
os.makedirs("fuerzas_kappa", exist_ok=True)

# ----------------------------
# Crear sistema y malla (solo una vez)
# ----------------------------
system = mm.System(name='Barrido_kappa')

cell = (dNP, dNP, dNP)
region = df.Region(p1=(0,0,0), p2=(dx, dy, dz))
mesh = df.Mesh(region=region, cell=cell)

# Guardar mesh
mesh2 = np.array([val for val in mesh])
np.savetxt("fuerzas_kappa/mesh2.dat", mesh2, fmt="%.6e", header="x y z")

# ----------------------------
# Bucle principal sobre kappa
# ----------------------------
for kappa in tqdm.tqdm(
        kappa_values,
        desc="🔹 Variando kappa",
        mininterval=5,
        dynamic_ncols=True):
    
    B_values = np.arange(-2, 2.1, 0.1)
    B_values=mm.consts.mu0*Ms*kappa*B_values
    # calcular K correspondiente
    #Ms=np.sqrt(2/kappa)*Ms0
    K=kappa*mm.consts.mu0*Ms**2/2

    kappa_dir = f"fuerzas_kappa/kappa_{kappa}"
    os.makedirs(kappa_dir, exist_ok=True)

    # ----------------------------
    # Bucle sobre B
    # ----------------------------
    for B in tqdm.tqdm(B_values, desc=f"  📊 B para κ={kappa}", leave=False):

        system.m = df.Field(mesh, nvdim=3,
                            value=(0,0,1),
                            norm=Ms_value,
                            valid="norm")

        # Energía
        system.energy = (
            mm.UniaxialAnisotropy(K=K, u=(0,0,1)) +
            mm.Demag() +
            mm.Zeeman(H=(B/mm.consts.mu0,0,0))
        )

        system.dynamics = mm.Damping(alpha=alpha) + mm.Precession(gamma0=mm.consts.gamma0)

        # Minimizar
        md = mc.MinDriver()
        md.drive(system,verbose=0)

        # ----------------------------
        # Calcular fuerzas
        # ----------------------------
        M = system.m

        H_ext = df.Field(mesh, nvdim=3, value=(B/mm.consts.mu0,0,0))
        H_int1 = mc.compute(system.energy.demag.effective_field, system,verbose=0)
        H_int = H_ext + H_int1

        B_field = mm.consts.mu0 * (H_int + M)
        BHdot = B_field.dot(H_int)

        Txx = B_field.x * H_int.x - 0.5 * BHdot
        Txy = B_field.x * H_int.y
        Txz = B_field.x * H_int.z
        Tyy = B_field.y * H_int.y - 0.5 * BHdot
        Tyz = B_field.y * H_int.z
        Tzz = B_field.z * H_int.z - 0.5 * BHdot

        fx = -(Txx.diff('x') + Txy.diff('y') + Txz.diff('z'))
        fy = -(Txy.diff('x') + Tyy.diff('y') + Tyz.diff('z'))
        fz = -(Txz.diff('x') + Tyz.diff('y') + Tzz.diff('z'))

        def force_value(pos):
            return (fx(pos), fy(pos), fz(pos))

        Force_field = df.Field(mesh, nvdim=3, value=force_value)
        Force_array = np.array([val for val in Force_field])

        # Guardar archivo
        B_str = f"{B:.2f}".replace('.', '_').replace('-', 'neg')
        filename = f"{kappa_dir}/Force_kappa{kappa}_Bx_{B_str}.dat"

        np.savetxt(filename, Force_array, fmt="%.6e", header="Fx Fy Fz")

print("✅ Todas las simulaciones completadas.")
