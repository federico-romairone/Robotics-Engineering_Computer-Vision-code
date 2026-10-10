import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import convolve2d
from scipy.ndimage import median_filter

rng = np.random.default_rng()


def show_image(image, title):
    plt.figure()
    plt.imshow(image, cmap="gray", interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.axis("off")


def show_profiles(original, averaged, median):
    # MATLAB row 40 corresponds to Python index 39.
    x = np.arange(1, original.shape[1] + 1)

    plt.figure()
    plt.plot(x, original[39, :], "r", label="Original")
    plt.plot(x, averaged[39, :], "b", label="Average filter")
    plt.plot(x, median[39, :], "g", label="Median filter")
    plt.xlabel("Column")
    plt.ylabel("Intensity")
    plt.legend()


def average_filter(image):
    # Zero padding matches MATLAB conv2(..., 'same').
    return convolve2d(image, K, mode="same", boundary="fill", fillvalue=0)


def median_filter_3x3(image):
    # Zero padding matches MATLAB medfilt2's default.
    return median_filter(image, size=(3, 3), mode="constant", cval=0)


# 3 × 3 averaging kernel
K = np.ones((3, 3)) / 9

# ==================
# Synthetic images
# ==================
A = np.zeros((101, 101))
A[:, 49:] = 1  # MATLAB A(:, 50:end)

show_image(A, "Synthetic image")

# Gaussian noise: standard deviation = 0.1
B = A + 0.1 * rng.standard_normal(A.shape)
show_image(B, "Synthetic image with Gaussian noise")

x = np.arange(1, A.shape[1] + 1)
plt.figure()
plt.plot(x, A[39, :], "r", label="Original")
plt.plot(x, B[39, :], "b", label="Noisy")
plt.xlabel("Column")
plt.ylabel("Intensity")
plt.legend()

# Calculate both filters before comparing their profiles.
outc = average_filter(B)
outm = median_filter_3x3(B)

show_image(outc, "Smoothing by averaging")
show_image(outm, "Non-linear filter: median")
show_profiles(A, outc, outm)

# ==================
# Real-world images
# ==================
IN = loadmat("tree.mat")["IN"].astype(float)
show_image(IN, "Original image")

# Gaussian noise: standard deviation = 20
img = IN + 20 * rng.standard_normal(IN.shape)
show_image(img, "Gaussian noise: standard deviation = 20")

# Smoothing by averaging
outc = average_filter(img)
show_image(outc, "Gaussian noise: average filter")

# Non-linear filter
outm = median_filter_3x3(img)
show_image(outm, "Gaussian noise: median filter")

# Salt-and-pepper noise
density = 0.2

# Select approximately 20% of pixels for corruption.
selected = rng.random(IN.shape) < density
noise_values = rng.random(IN.shape)

mask1 = selected & (noise_values < 0.5)
mask2 = selected & (noise_values >= 0.5)

out = IN.copy()
out[mask1] = 0
out[mask2] = 149  # Preserve the salt intensity in the MATLAB code.

show_image(out, "Salt-and-pepper noise: density = 0.2")

# Smoothing by averaging
outc = average_filter(out)
show_image(outc, "Salt-and-pepper noise: average filter")

# Non-linear filter
outm = median_filter_3x3(out)
show_image(outm, "Salt-and-pepper noise: median filter")

plt.show()