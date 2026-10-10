import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

rng = np.random.default_rng()


def show_image(image, title):
    plt.figure()
    plt.imshow(image, cmap="gray", interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.axis("off")


# Load image
data = loadmat("tree.mat")
IN = data["IN"]

show_image(IN, "Original image")

# Gaussian noise: standard deviation = 10
out = IN.astype(float) + 10 * rng.standard_normal(IN.shape)
show_image(out, "Gaussian noise: standard deviation = 10")

# Gaussian noise: standard deviation = 30
out = IN.astype(float) + 30 * rng.standard_normal(IN.shape)
show_image(out, "Gaussian noise: standard deviation = 30")

# Noise distribution
a = rng.standard_normal(100)

plt.figure()
plt.hist(a, bins=21)
plt.title("Gaussian noise distribution (100 samples)")
plt.xlabel("Value")
plt.ylabel("Frequency")

a = rng.standard_normal(10000)

plt.figure()
plt.hist(a, bins=21)
plt.title("Gaussian noise distribution (10000 samples)")
plt.xlabel("Value")
plt.ylabel("Frequency")

# Salt-and-pepper noise
IN = IN.astype(float)
rr, cc = IN.shape
maxv = IN.max()
density = 0.3

# Select approximately 30% of pixels for corruption.
selected = rng.random((rr, cc)) < density
noise_values = rng.random((rr, cc))

mask1 = selected & (noise_values < 0.5)   # Pepper
mask2 = selected & (noise_values >= 0.5)  # Salt

out = IN.copy()
out[mask1] = 0
out[mask2] = maxv

show_image(out, "Salt-and-pepper noise: density = 0.3")

plt.show()