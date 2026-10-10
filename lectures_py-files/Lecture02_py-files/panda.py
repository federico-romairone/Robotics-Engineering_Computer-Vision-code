import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import correlate

# Read image and convert to grayscale
tmp = np.asarray(Image.open("panda.png").convert("RGB"), dtype=float)
im = np.floor(
    0.298936 * tmp[:, :, 0]
    + 0.587043 * tmp[:, :, 1]
    + 0.114021 * tmp[:, :, 2]
    + 0.5
).astype(np.uint8)

fsize = 33

# Coordinates for the Gaussian kernel
coords = np.arange(fsize) - (fsize - 1) / 2
X, Y = np.meshgrid(coords, coords)

# MATLAB 1:3:9 gives sigma = 1, 4, 7.
for sigma in range(1, 10, 3):
    # Equivalent to fspecial('gaussian', fsize, sigma)
    h = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    h[h < np.finfo(float).eps * h.max()] = 0
    h /= h.sum()

    # Filtering with zero padding, as in MATLAB imfilter
    out = correlate(im.astype(float), h, mode="constant", cval=0)
    out = np.floor(np.clip(out, 0, 255) + 0.5).astype(np.uint8)

    # Display both the filtered image and its kernel
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(out, cmap="gray")
    axes[0].set_title(f"Filtered image: sigma = {sigma}")
    axes[0].axis("off")

    axes[1].imshow(h, cmap="gray")
    axes[1].set_title(f"Gaussian kernel: sigma = {sigma}")
    axes[1].axis("off")

    fig.tight_layout()

plt.show()