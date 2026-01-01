import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.colors import LinearSegmentedColormap


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_sponge_mask(Nx, Ny, width=18, strength=3.0):
    mask = np.ones((Nx, Ny))
    ix = np.minimum(np.arange(Nx), np.arange(Nx)[::-1])
    iy = np.minimum(np.arange(Ny), np.arange(Ny)[::-1])
    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    d = np.minimum(IX, IY)
    s = np.clip((width - d) / width, 0.0, 1.0)
    mask *= np.exp(-strength * s * s)
    return mask


def gaussian_source(X, Y, x0, y0, sigma):
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2.0 * sigma**2))


def main():

    eps0 = 8.854187817e-12
    mu0 = 4e-7 * np.pi
    c0 = 1.0 / np.sqrt(eps0 * mu0)

    Nx = Ny = 240
    Lx = Ly = 1.0
    dx, dy = Lx / Nx, Ly / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    dt_max = 1.0 / (c0 * np.sqrt((1.0 / dx**2) + (1.0 / dy**2)))
    dt = 0.98 * dt_max

    nsteps = 900
    save_every = 2

    Ez = np.zeros((Nx, Ny))
    Hx = np.zeros((Nx, Ny - 1))
    Hy = np.zeros((Nx - 1, Ny))

    Ez[:] = gaussian_source(X, Y, 0.35 * Lx, 0.50 * Ly, sigma=0.025)

    sponge = make_sponge_mask(Nx, Ny)
    sponge_Hx = sponge[:, :-1]
    sponge_Hy = sponge[:-1, :]

    ensure_dir("outputs")
    out_gif = "outputs/fdtd_2d.gif"

    gain = 18.0

    dark_blue = (0.05, 0.15, 0.45)
    cmap = LinearSegmentedColormap.from_list(
        "white_to_darkblue",
        [(1.0, 1.0, 1.0), dark_blue]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
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
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("2D FDTD Maxwell (TEz)", color="black")

    for spine in ax.spines.values():
        spine.set_color("black")

    fig.tight_layout(pad=0)

    frames = []
    t = 0.0

    print(f"dt = {dt:.3e}, steps = {nsteps}")

    for n in range(nsteps):

        Hx -= (dt / (mu0 * dy)) * (Ez[:, 1:] - Ez[:, :-1])
        Hy += (dt / (mu0 * dx)) * (Ez[1:, :] - Ez[:-1, :])

        dHy_dx = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx
        dHx_dy = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
        Ez[1:-1, 1:-1] += (dt / eps0) * (dHy_dx - dHx_dy)

        Ez *= sponge
        Hx *= sponge_Hx
        Hy *= sponge_Hy

        t += dt

        if n % save_every == 0:
            im.set_data(np.abs(np.tanh(gain * Ez)).T)
            ax.set_title(f"2D FDTD Maxwell (TEz)   t = {t:.3e}", color="black")

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            frames.append(img.copy())

    plt.close(fig)

    print(f"Saving GIF: {out_gif}")
    imageio.mimsave(out_gif, frames, fps=25)
    print("Done.")


if __name__ == "__main__":
    main()
