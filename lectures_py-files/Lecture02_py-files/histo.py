import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def to_uint8(image):
    # Round and clip, matching MATLAB's uint8 conversion.
    return np.floor(np.clip(image, 0, 255) + 0.5).astype(np.uint8)


def show_image_and_histogram(image, title):
    plt.figure()
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

    plt.figure()
    plt.hist(
        to_uint8(image).ravel(),
        bins=256,
        range=(-0.5, 255.5),
        color="black"
    )
    plt.xlim(0, 255)
    plt.title(f"{title} — histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels")


def equalize_histogram(image):
    image = to_uint8(image)

    # Count pixels at each intensity level.
    histogram = np.bincount(image.ravel(), minlength=256)

    # Normalized cumulative distribution function.
    cdf = histogram.cumsum().astype(float)
    cdf /= cdf[-1]

    # Map each input intensity to its equalized value.
    lookup_table = to_uint8(255 * cdf)
    return lookup_table[image]


# Histogram
data = loadmat("tree.mat")
IN = data["IN"].astype(np.float64)

show_image_and_histogram(IN, "Original image")

# Contrast stretching (also useful for image normalization)
M = IN.max()
m = IN.min()

if M > m:
    out = ((IN - m) / (M - m)) * 255
else:
    out = np.zeros_like(IN)

show_image_and_histogram(out, "Contrast stretching")

# Histogram equalization
OUT = equalize_histogram(IN)

show_image_and_histogram(OUT, "Histogram equalization")

plt.show()