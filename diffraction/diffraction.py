import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.colors import LinearSegmentedColormap


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_sponge_mask(Nx: int, Ny: int, width: int = 22, strength: float = 3.5) -> np.ndarray:
    mask = np.ones((Nx, Ny), dtype=float)

    ix = np.minimum(np.arange(Nx), np.arange(Nx)[::-1])
    iy = np.minimum(np.arange(Ny), np.arange(Ny)[::-1])
    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    d = np.minimum(IX, IY)

    w = max(1, int(width))
    s = np.clip((w - d) / w, 0.0, 1.0)
    mask *= np.exp(-strength * s * s)
    return mask


def build_single_slit_wall(Nx: int, Ny: int, wall_i: int, slit_center_j: int, slit_half_height: int, thickness: int = 2):
    pec = np.zeros((Nx, Ny), dtype=bool)

    j0 = max(0, slit_center_j - slit_half_height)
    j1 = min(Ny - 1, slit_center_j + slit_half_height)

    for ti in range(thickness):
        ii = wall_i + ti
        if 0 <= ii < Nx:
            pec[ii, :] = True
            pec[ii, j0:j1 + 1] = False

    return pec


def soft_line_source(t: float, f0: float, t_ramp: float) -> float:
    envelope = 1.0 - np.exp(-(t / max(t_ramp, 1e-30)) ** 2)
    return envelope * np.sin(2.0 * np.pi * f0 * t)


def main():

    eps0 = 8.854187817e-12
    mu0 = 4e-7 * np.pi
    c0 = 1.0 / np.sqrt(eps0 * mu0)

    Nx, Ny = 360, 240
    Lx, Ly = 1.2, 0.8
    dx, dy = Lx / Nx, Ly / Ny

    dt_max = 1.0 / (c0 * np.sqrt((1.0 / dx**2) + (1.0 / dy**2)))
    dt = 0.98 * dt_max

    nsteps = 1400
    save_every = 2

    Ez = np.zeros((Nx, Ny), dtype=float)
    Hx = np.zeros((Nx, Ny - 1), dtype=float)
    Hy = np.zeros((Nx - 1, Ny), dtype=float)

    sponge = make_sponge_mask(Nx, Ny, width=26, strength=4.0)
    sponge_Hx = sponge[:, :-1]
    sponge_Hy = sponge[:-1, :]

    wall_i = int(0.55 * Nx)         
    slit_center_j = Ny // 2          
    slit_half_height = int(0.06 * Ny)  
    wall_thickness = 3
    pec = build_single_slit_wall(Nx, Ny, wall_i, slit_center_j, slit_half_height, thickness=wall_thickness)

    src_i = int(0.10 * Nx)
    src_j0 = int(0.15 * Ny)
    src_j1 = int(0.85 * Ny)

    wavelength = 0.08 
    f0 = c0 / wavelength
    t_ramp = 8.0 / f0 

    ensure_dir("outputs")
    out_gif = "outputs/diffraction.gif"

    dark_blue = (0.03, 0.10, 0.35)
    cmap = LinearSegmentedColormap.from_list("white_to_darkblue", [(1.0, 1.0, 1.0), dark_blue])

    gain = 20.0 

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(
        np.abs(np.tanh(gain * Ez)).T,
        origin="lower",
        extent=(0, Lx, 0, Ly),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("2D FDTD TEz: single-slit diffraction", color="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.tight_layout(pad=0)

    frames = []
    t = 0.0

    print(f"Nx={Nx}, Ny={Ny}, dx={dx:.3e}, dy={dy:.3e}")
    print(f"dt_max={dt_max:.3e}, dt={dt:.3e}, steps={nsteps}")
    print(f"wall_i={wall_i}, slit_half_height={slit_half_height}, src_i={src_i}, f0={f0:.3e}")

    for n in range(nsteps):

        Hx -= (dt / (mu0 * dy)) * (Ez[:, 1:] - Ez[:, :-1])
        Hy += (dt / (mu0 * dx)) * (Ez[1:, :] - Ez[:-1, :])

        dHy_dx = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx
        dHx_dy = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
        Ez[1:-1, 1:-1] += (dt / eps0) * (dHy_dx - dHx_dy)

        src_val = soft_line_source(t, f0=f0, t_ramp=t_ramp)
        Ez[src_i, src_j0:src_j1] += src_val

        Ez[pec] = 0.0

        Ez *= sponge
        Hx *= sponge_Hx
        Hy *= sponge_Hy

        t += dt

        if n % save_every == 0:
            im.set_data(np.abs(np.tanh(gain * Ez)).T)
            ax.set_title(f"2D FDTD TEz: single-slit diffraction   t = {t:.3e}", color="black")

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            frames.append(img.copy())

    plt.close(fig)

    print(f"Saving GIF: {out_gif} ({len(frames)} frames)")
    imageio.mimsave(out_gif, frames, fps=25)
    print("Done.")


if __name__ == "__main__":
    main()
