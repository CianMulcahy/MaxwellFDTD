import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def gaussian_pulse(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def main():

    eps0 = 8.854187817e-12
    mu0 = 4e-7 * np.pi
    c0 = 1.0 / np.sqrt(eps0 * mu0)

    Nx = 800
    L = 1.0
    dx = L / Nx
    x = np.arange(Nx) * dx

    S = 0.99
    dt = S * dx / c0

    nsteps = 1800
    save_every = 3

    Ez = np.zeros(Nx, dtype=float)

    Hy = np.zeros(Nx - 1, dtype=float)

    x0 = 0.25 * L
    sigma = 0.03 * L
    Ez[:] = gaussian_pulse(x, x0, sigma)

    Ez_left_old = Ez[0]
    Ez_right_old = Ez[-1]

    mur_coeff = (c0 * dt - dx) / (c0 * dt + dx)

    out_gif = "outputs/fdtd_1d.gif"
    ensure_dir(os.path.dirname(out_gif))


    fig, ax = plt.subplots(figsize=(9, 4))
    line_E, = ax.plot(x, Ez, label="Ez(x,t)")

    xH = x[:-1] + 0.5 * dx
    line_H, = ax.plot(xH, Hy, label="Hy(x,t)")
    ax.set_xlim(0, L)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("Field amplitude")
    ax.set_title("1D FDTD Maxwell (Yee) with Mur ABC")
    ax.legend(loc="upper right")
    fig.tight_layout()

    frames = []
    t = 0.0

    for n in range(nsteps):
        Hy -= (dt / (mu0 * dx)) * (Ez[1:] - Ez[:-1])

        Ez[1:-1] -= (dt / (eps0 * dx)) * (Hy[1:] - Hy[:-1])

        Ez0_new = Ez[1] + mur_coeff * (Ez[1] - Ez_left_old)
        Ez_left_old = Ez[0]
        Ez[0] = Ez0_new

        EzN_new = Ez[-2] + mur_coeff * (Ez[-2] - Ez_right_old)
        Ez_right_old = Ez[-1]
        Ez[-1] = EzN_new

        t += dt

        if n % save_every == 0:
            line_E.set_ydata(Ez)
            line_H.set_ydata(Hy)

            ax.set_title(f"1D FDTD Maxwell (Yee) with Mur ABC   t = {t:.3e} s   S = {S:.2f}")

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
