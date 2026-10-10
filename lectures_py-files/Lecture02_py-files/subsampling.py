import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import convolve2d

# 1D low-pass filter, represented as a row vector
lpf = np.array([[1, 4, 6, 4, 1]], dtype=float) / 16

# Load image
IN = loadmat("tree.mat")["IN"]


def show_image(image, title):
    plt.figure()
    plt.imshow(image, cmap="gray", interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.axis("off")


show_image(IN, "Original image")

# Downsample directly by a factor of 4
show_image(IN[::4, ::4], "Downsampling by 4 without filtering")

# Apply the horizontal and vertical filters.
# MATLAB conv2 uses full convolution by default.
tmp = convolve2d(IN.astype(float), lpf, mode="full")
tmp = convolve2d(tmp, lpf.T, mode="full")

# Downsample by a factor of 2
tmp1 = tmp[::2, ::2]

# Filter again horizontally and vertically
tmp2 = convolve2d(tmp1, lpf, mode="full")
tmp2 = convolve2d(tmp2, lpf.T, mode="full")

# Downsample again by a factor of 2
show_image(tmp2[::2, ::2], "Two stages of filtering and downsampling")

plt.show()