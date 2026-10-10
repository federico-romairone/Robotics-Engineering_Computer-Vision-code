import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# DIGITAL IMAGE REPRESENTATION
A = np.array([
    [0.490, 0.310, 0.200],
    [0.177, 0.813, 0.011],
    [0.000, 0.010, 0.990]
])
print(A)

plt.figure()
plt.imshow(A, cmap="gray", interpolation="nearest", aspect="auto")

# Load the variable IN from tree.mat
data = loadmat("tree.mat")
IN = data["IN"]

print(IN[::40, ::40])

plt.figure()
plt.imshow(IN[::40, ::40], cmap="gray",
           interpolation="nearest", aspect="auto")

# 3D surface plot
Z = IN[::5, ::5].astype(float)
X, Y = np.meshgrid(
    np.arange(1, Z.shape[1] + 1),
    np.arange(1, Z.shape[0] + 1)
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", rstride=1, cstride=1)

plt.figure()
plt.imshow(IN.astype(float), cmap="gray",
           interpolation="nearest", aspect="auto")

# SPATIAL RESOLUTION
step = 1
for a in range(4):
    plt.figure()
    plt.imshow(IN[::step, ::step], cmap="gray",
               interpolation="nearest", aspect="auto")
    step *= 2

# TRUECOLOR IMAGES
img = plt.imread("left_#290.bmp")

plt.figure()
plt.imshow(img, interpolation="nearest", aspect="auto")

# COLORMAPS
for cmap_name in ["gray", "pink", "jet"]:
    plt.figure()
    plt.imshow(IN.astype(float), cmap=cmap_name,
               interpolation="nearest", aspect="auto")

# Display the RGB values of a 256-color jet colormap
jet = plt.get_cmap("jet")(np.linspace(0, 1, 256))[:, :3]
print(jet)

plt.show()