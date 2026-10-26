#!/usr/bin/env python
# coding: utf-8
"""
Post-processing of the magnetic force density fields.

Produces:
  1. PNG frames of one force component (Fx, Fy or Fz) in one orthogonal
     plane (zx, xy or yz), one image per applied field B, using a SymLog
     color scale.
  2. An animated GIF cycling through B forward and backward for a fixed κ.

Inputs:  fuerzas_kappa/kappa_<κ>/Force_kappa<κ>_Bx_<B>.dat
Outputs: fuerzas_kappa/kappa_<κ>_scalar_<plane>/*.png
         anim_fuerza_<plane>_ciclo_lento.gif

Run after force_script_kappa.py.
"""

import os
import glob
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')               # headless backend (no display needed)
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from PIL import Image

import discretisedfield as df


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
KAPPA = 5                           # κ to visualize
PLANE = "yz"                        # "zx", "xy" or "yz"
LIN_PCT = 50.0                      # percentile for SymLog linear threshold
VMAX_PCT = 99.9                     # percentile for upper color limit
LINSCALE = 0.5                      # SymLog linear-region scale
FRAME_MS = 300                      # duration of each GIF frame (ms)


# ----------------------------------------------------------------------
# Mesh (must match the simulation)
# ----------------------------------------------------------------------
n0 = 0.8388e-9
dNP = int(14e-9 / n0) * n0
Nx, Ny, Nz = 25, 25, 25
dx, dy, dz = dNP * Nx, dNP * Ny, dNP * Nz

mesh = df.Mesh(
    region=df.Region(p1=(0, 0, 0), p2=(dx, dy, dz)),
    cell=(dNP, dNP, dNP),
)

xs_nm = np.arange(Nx) * dNP * 1e9
ys_nm = np.arange(Ny) * dNP * 1e9
zs_nm = np.arange(Nz) * dNP * 1e9


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def load_force(kappa, B):
    """Load a force density array from disk as a df.Field."""
    B_str = f"{B:.2f}".replace('.', '_').replace('-', 'neg')
    path = f"fuerzas_kappa/kappa_{kappa}/Force_kappa{kappa}_Bx_{B_str}.dat"
    arr = np.loadtxt(path, comments='#')
    return df.Field(mesh, nvdim=3, value=arr.reshape(Nx, Ny, Nz, 3))


def parse_B(fname):
    """Parse B from a filename like `..._Bx_neg6_17.dat`."""
    m = re.search(r"_Bx_(neg)?(\d+)_(\d+)\.dat$", os.path.basename(fname))
    if m is None:
        return None
    sign = -1 if m.group(1) else 1
    return sign * float(f"{m.group(2)}.{m.group(3)}")


def get_Bs(kappa):
    """Return the sorted list of B values available for a given κ."""
    folder = f"fuerzas_kappa/kappa_{kappa}"
    files = [
        f for f in glob.glob(os.path.join(folder, "Force_*.dat"))
        if parse_B(f) is not None
    ]
    files.sort(key=parse_B)
    return [parse_B(f) for f in files]


# ----------------------------------------------------------------------
# Configuration of the three planes
# ----------------------------------------------------------------------
PLANES = {
    'zx': dict(color=1, xlab='x (nm)', ylab='z (nm)'),
    'xy': dict(color=2, xlab='x (nm)', ylab='y (nm)'),
    'yz': dict(color=0, xlab='y (nm)', ylab='z (nm)'),
}
COMP_NAMES = ['Fx', 'Fy', 'Fz']


# ----------------------------------------------------------------------
# PNG generation
# ----------------------------------------------------------------------
def make_png(B_val, plane, kappa, vmax, lin, out_path):
    """Create a PNG of one force component in one plane at field B_val."""
    cfg = PLANES[plane]
    arr = load_force(kappa, B_val).array

    if plane == 'zx':
        j = Ny // 2
        C = arr[:, j, :, cfg['color']].T
        xs_p, ys_p = xs_nm, zs_nm
    elif plane == 'xy':
        k = Nz // 2
        C = arr[:, :, k, cfg['color']].T
        xs_p, ys_p = xs_nm, ys_nm
    else:  # 'yz'
        i = Nx // 2
        C = arr[i, :, :, cfg['color']].T
        xs_p, ys_p = ys_nm, zs_nm

    comp_name = COMP_NAMES[cfg['color']]

    norm = SymLogNorm(
        linthresh=lin,
        linscale=LINSCALE,
        vmin=-vmax,
        vmax=+vmax,
        base=10,
    )

    fig, ax = plt.subplots(figsize=(7, 6.2), dpi=130)
    im = ax.imshow(
        C, origin='lower', cmap='coolwarm', norm=norm,
        extent=[xs_p[0], xs_p[-1], ys_p[0], ys_p[-1]],
        aspect='equal', interpolation='bilinear',
    )

    ax.set_xlabel(cfg['xlab'], fontsize=14)
    ax.set_ylabel(cfg['ylab'], fontsize=14)
    ax.set_title(
        f'{comp_name} [{plane}]   κ={kappa}   B = {B_val:+.2f} T\n'
        f'SymLog: linthresh={lin:.1e}, linscale={LINSCALE}, vmax={vmax:.1e}',
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{comp_name}  (N/m³)', fontsize=11)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)

    ticks = np.array([-vmax, -lin, 0, lin, vmax])
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([
        f'{-vmax:.1e}', f'{-lin:.1e}', '0', f'{lin:.1e}', f'{vmax:.1e}',
    ], fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_pngs(kappa):
    """Generate one PNG per (plane, B) for a given κ."""
    Bs = get_Bs(kappa)
    print(f"Frames: {len(Bs)}  B ∈ [{min(Bs):+.2f}, {max(Bs):+.2f}] T")

    # Global statistics across all frames (for a shared color scale)
    all_F = {c: [] for c in range(3)}
    for B in Bs:
        arr = load_force(kappa, B).array
        for c in range(3):
            all_F[c].append(arr[..., c].ravel())

    vmax = {c: np.percentile(np.abs(np.concatenate(all_F[c])), VMAX_PCT)
            for c in range(3)}
    lin = {c: np.percentile(np.abs(np.concatenate(all_F[c])), LIN_PCT)
           for c in range(3)}

    print(f"\nvmax   (p{VMAX_PCT}): |Fx|={vmax[0]:.2e}  |Fy|={vmax[1]:.2e}  |Fz|={vmax[2]:.2e}")
    print(f"linthr (p{LIN_PCT}): Fx={lin[0]:.2e}  Fy={lin[1]:.2e}  Fz={lin[2]:.2e}  linscale={LINSCALE}")

    for plane in PLANES:
        c = PLANES[plane]['color']
        outdir = f"fuerzas_kappa/kappa_{kappa}_scalar_{plane}"
        os.makedirs(outdir, exist_ok=True)
        print(f"\n=== Plane {plane} → {outdir}/ ===")
        for i, B in enumerate(Bs):
            sign = 'neg' if B < 0 else 'pos'
            fname = f"{plane}_kappa{kappa}_B_{sign}{abs(B):.2f}.png".replace('.', '_')
            try:
                make_png(B, plane, kappa, vmax[c], lin[c],
                         os.path.join(outdir, fname))
                print(f"  [{i+1:2d}/{len(Bs)}] {fname}")
            except Exception as e:
                print(f"  ! B={B:+.2f}: {e}")


# ----------------------------------------------------------------------
# GIF generation
# ----------------------------------------------------------------------
def generate_gif(kappa, plane, frame_ms=FRAME_MS):
    """Create a forward-and-back GIF from the PNG frames."""
    folder = f"fuerzas_kappa/kappa_{kappa}_scalar_{plane}"
    files = glob.glob(os.path.join(folder, '*.png'))

    def order_key(fname):
        base = os.path.basename(fname)
        m = re.search(r'_(pos|neg)(\d+)_(\d+)', base)
        if m:
            sign = m.group(1)
            value = float(f"{m.group(2)}.{m.group(3)}")
            return value if sign == 'pos' else -value
        return 0

    # Forward: from +B to −B
    forward = sorted(files, key=order_key, reverse=True)
    # Backward: exact reverse (from −B to +B)
    backward = forward[::-1]
    cycle = forward + backward

    print(f"\nGIF frames: {len(cycle)} (forward {len(forward)} + backward {len(backward)})")
    print(f"Frame duration: {frame_ms} ms")
    print(f"Total duration: {len(cycle) * frame_ms / 1000:.2f} s")

    frames = [Image.open(f).resize((600, 540)) for f in cycle]
    out_gif = f"anim_fuerza_{plane}_ciclo_lento.gif"
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
    )
    print(f"Saved {out_gif}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    generate_pngs(KAPPA)
    generate_gif(KAPPA, PLANE)
