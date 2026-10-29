# Magnetoelastic Simulation: Maxwell Stress to Navier Strain

A Python pipeline to simulate the **magneto-elastic response** of assemblies
of magnetic nanoparticles under an external magnetic field. The workflow
couples:

1. **Micromagnetics** (Ubermag / OOMMF) to compute the equilibrium
   magnetization `M(r)` under an applied field `B`.
2. **Maxwell stress tensor** `T̂(r)` to obtain the magnetic force density
   `f(r) = -∇·T̂`.
3. **Linear elasticity** (FEniCSx) to solve the Navier equation for the
   displacement field `u(r)` and the strain tensor `ε̂`.

The simulation is used to quantify whether **magnetic interactions between
nanoparticles can produce observable magnetostriction** at the level of
interparticle strain.

The setup models a simple-cubic assembly of magnetic nanoparticles with a
common uniaxial easy axis along `z`, clamped at the bottom (`z = 0`), and
subjected to an external field along `x` (perpendicular to the easy axis).


<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; text-align: center;">
  <div>
    <img width="400" alt="anim_zx_lento" src="https://github.com/user-attachments/assets/acb55846-8029-431d-a95e-0c23fb8e161f" />
    <p style="font-size: 14px; margin-top: 8px;">
      <b>Fig. 1:</b> Magnetization as a function of the applied field along x (xz plane).
    </p>
  </div>
  <div>
    <img width="400" alt="anim_fuerza_yz_ciclo_lento" src="https://github.com/user-attachments/assets/2ee7a2a5-751f-43c6-892f-46a2a4e36f1b" />
    <p style="font-size: 14px; margin-top: 8px;">
      <b>Fig. 2:</b> x-component of the magnetic force density (yz plane).
    </p>
  </div>
</div>


---

## Physics

### Magnetic subsystem

Each simulation cell is a nanoparticle with:

- saturation magnetization `Ms`
- uniaxial anisotropy constant `K` along `z`
- an applied field `H_ext = (B/μ0, 0, 0)`
- dipolar (demag) interactions

The equilibrium magnetization is found by minimizing the total energy
(`UniaxialAnisotropy + Demag + Zeeman`).

### Maxwell stress tensor and force density

From the micromagnetic solution, the internal field is

    H_int(r) = H_ext + H_demag(r)

and the magnetic field is

    B_field(r) = μ0 (H_int(r) + M(r)).

The Maxwell stress tensor is

    T_{αβ} = B_α H_β − (1/2)(H · B) δ_{αβ}

and the force density is its divergence:

    f_α(r) = − Σ_β ∂_β T_{αβ}

### Elastic subsystem (Navier equation)

The force density is inserted as a body force in the isotropic linear
elastic equation:

    f(r) = −(λ + μ) ∇(∇·u) − μ ∇² u

with Lamé parameters

    λ = ν Y / [(1 + ν)(1 − 2ν)]
    μ = Y / [2(1 + ν)]

The system is solved with **FEniCSx** using Lagrange P1 vector elements.
Boundary conditions:

- **bottom surface** (`z = 0`): clamped, `u = 0`
- **all other surfaces**: free

### Strain and magnetostriction

The strain tensor is

    ε_{αβ} = (1/2)(∂_α u_β + ∂_β u_α)

with diagonal components

    ε_{αα} = ΔL_α / L_α      (strain along each axis)

and volumetric strain

    ΔV/V = ε_xx + ε_yy + ε_zz

The deformation is computed **relative to the zero-field case** (`B = 0`):

    ε_rel(r) = ε(B)(r) − ε(0)(r)

so that the elastic reference configuration is subtracted.

### Magnetic hardness parameter κ

The relative weight of anisotropy vs magnetostatic energy is parametrized by

    κ = 2K / (μ0 Ms²)

- `κ ≪ 1` → soft magnetic system (anisotropy dominates weakly)
- `κ ≫ 1` → hard magnetic system (anisotropy dominates strongly)

The script sweeps `κ ∈ {0.1, 0.5, 1, 2, 5}` and the normalized field
`B/(μ0 Ms κ) ∈ [−2, 2]`.
<div style="text-align: center;">
  <img width="800" alt="Delta_V2" src="https://github.com/user-attachments/assets/e740edd7-22e2-4ccd-979f-37ca994f1b59" />
  <p style="font-size: 14px; margin-top: 8px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;">
    <b>Figure 3:</b> Volumetric strain resulting from magnetic interactions under an external magnetic field. (a) A fixed demagnetizing factor (\(1/3\)) is considered while varying \(\kappa\). (b) A fixed \(\kappa = 5\) is considered while varying the overall shape of the system.
  </p>
</div>
---

MoisesZar
moisesg.zarzoza@uclm.es

## Requirements

- Python 3.9+
- [Ubermag](https://ubermag.github.io/) (`oommfc`, `discretisedfield`, `micromagneticmodel`)
- **OOMMF** installed and available in `PATH`
- [FEniCSx](https://fenicsproject.org/) ≥ 0.7 (`dolfinx`, `ufl`, `basix`)
- `mpi4py` (for FEniCSx parallel runs)
- `numpy`, `matplotlib`, `tqdm`, `Pillow` (for the GIF)
