import numpy as np
import matplotlib.pyplot as plt

# Helpers for image and mesh plots
def show_image(Z, xlabel, ylabel):
    plt.figure()
    plt.imshow(
        Z,
        cmap="gray",
        interpolation="nearest",
        extent=(0.5, Z.shape[1] + 0.5, Z.shape[0] + 0.5, 0.5),
        aspect="auto"
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def show_mesh(Z, xlabel, ylabel, zlabel):
    cols, rows = np.meshgrid(
        np.arange(1, Z.shape[1] + 1),
        np.arange(1, Z.shape[0] + 1)
    )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        cols, rows, Z,
        cmap="viridis",
        edgecolor=(0, 0, 0, 0.3),
        linewidth=0.25,
        rstride=1,
        cstride=1,
        antialiased=True
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=30, azim=-60)

    fig.tight_layout()


# FFT — grating
w = 64
X1, Y1 = np.meshgrid(np.arange(1, w + 1), np.arange(1, w + 1))
X = X1 - w / 2
Y = Y1 - w / 2

Z = np.sin(2 * np.pi * 0.1 * X)  # Frequency: 0.1 cycles/pixel

show_mesh(Z, "x", "y", "z")
show_image(Z, "x", "y")

FZ = np.fft.fftshift(np.fft.fft2(Z))

show_mesh(np.abs(FZ), "wx", "wy", "abs(F)")
show_image(np.abs(FZ), "wx", "wy")

# Different frequency
Z = np.sin(2 * np.pi * 0.2 * X)  # Frequency: 0.2 cycles/pixel

show_image(Z, "x", "y")

FZ = np.fft.fftshift(np.fft.fft2(Z))
show_image(np.abs(FZ), "wx", "wy")

# Different orientation
Z = np.sin(2 * np.pi * 0.1 * (X + Y))

show_image(Z, "x", "y")

FZ = np.fft.fftshift(np.fft.fft2(Z))
show_image(np.abs(FZ), "wx", "wy")

# Gaussian — standard deviation: 5 pixels
Z = np.exp((-X**2 - Y**2) / (2 * 5**2))

show_mesh(Z, "x", "y", "z")
show_image(Z, "x", "y")

FZ = np.fft.fftshift(np.fft.fft2(Z))

show_mesh(np.abs(FZ), "wx", "wy", "abs(F)")
show_image(np.abs(FZ), "wx", "wy")

# Box
Z = np.zeros((w, w))
vi = np.arange(-5, 6)

# Subtract 1 to convert MATLAB indices to Python indices.
indices = w // 2 + vi - 1
Z[np.ix_(indices, indices)] = 1

show_mesh(Z, "x", "y", "z")
show_image(Z, "x", "y")

FZ = np.fft.fftshift(np.fft.fft2(Z))

show_mesh(np.abs(FZ), "wx", "wy", "abs(F)")
show_image(np.abs(FZ), "wx", "wy")

plt.show()