import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def show_image(image, title, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray", interpolation="nearest", aspect="auto")
    ax.set_box_aspect(1)  # Equivalent to MATLAB's axis square
    ax.set_title(title)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    fig.tight_layout()


# Phase information is crucial for reconstructing
# the spatial structure of an image.

# Load image
img = loadmat("tree.mat")["IN"].astype(float)

# Fourier transform, magnitude, and phase
IMG = np.fft.fft2(img)
MOD = np.abs(IMG)
PHI = np.angle(IMG)

# Original image
show_image(img, "Original image")

# Log-magnitude spectrum: avoid taking log(0)
log_magnitude = np.log(
    np.maximum(np.fft.fftshift(MOD), np.finfo(float).tiny)
)
show_image(log_magnitude, "Log-magnitude spectrum", "wx", "wy")

# Phase spectrum
show_image(np.fft.fftshift(PHI), "Phase spectrum", "wx", "wy")

# Reconstruction using magnitude only (zero phase)
magnitude_only = np.real(
    np.fft.fftshift(np.fft.ifft2(MOD))
)
show_image(magnitude_only, "Magnitude-only reconstruction", "x", "y")

# Reconstruction using phase only (unit magnitude)
phase_only = np.real(
    np.fft.ifft2(np.exp(1j * PHI))
)
show_image(phase_only, "Phase-only reconstruction", "x", "y")

plt.show()