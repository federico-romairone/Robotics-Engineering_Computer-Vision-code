import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read image and convert to grayscale
rgb = np.asarray(Image.open("panda.png").convert("RGB"), dtype=float)
img = np.floor(
    0.298936 * rgb[:, :, 0]
    + 0.587043 * rgb[:, :, 1]
    + 0.114021 * rgb[:, :, 2]
    + 0.5
).astype(np.uint8)

rr, cc = img.shape


def show_image_and_histogram(image, title):
    plt.figure()
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.axis("off")

    plt.figure()
    plt.hist(
        image.ravel(),
        bins=256,
        range=(-0.5, 255.5),
        color="black"
    )
    plt.xlim(0, 255)
    plt.title(f"{title} — histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels")


show_image_and_histogram(img, "Original image")

# Flatten using MATLAB's column-major order
v = img.ravel(order="F")

# Randomly shuffle the pixel values
rng = np.random.default_rng()
v = rng.permutation(v)

# Reshape using the same column-major order
img2 = v.reshape((rr, cc), order="F")

show_image_and_histogram(img2, "Shuffled image")

plt.show()