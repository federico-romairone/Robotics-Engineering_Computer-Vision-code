import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata

# Load image
data = loadmat("tree.mat")
IN = data["IN"]
rr, cc = IN.shape


def show_image(image, title=None):
    plt.figure()
    plt.imshow(image, cmap="gray", interpolation="nearest", aspect="auto")
    if title:
        plt.title(title)


show_image(IN)
zf = 2.8  # Zooming factor

# Forward warping
# MATLAB rounds positive half-integers upward.
ri = np.floor(zf * np.arange(1, rr + 1) + 0.5).astype(int)
ci = np.floor(zf * np.arange(1, cc + 1) + 0.5).astype(int)

tmp = np.zeros((ri[-1], ci[-1]), dtype=IN.dtype)

# Convert MATLAB's 1-based indices to Python's 0-based indices.
tmp[np.ix_(ri - 1, ci - 1)] = IN

r0 = int(np.ceil(ri[-1] / 2))
c0 = r0
vv = np.arange(-(rr // 2), rr // 2 + 1)

rows = r0 + vv - 1
cols = c0 + vv - 1
out1 = tmp[np.ix_(rows, cols)]

show_image(out1, "forward warping")

# Backward warping
X, Y = np.meshgrid(
    np.arange(1, cc + 1),
    np.arange(1, rr + 1)
)

XC = X - cc // 2
YC = Y - rr // 2

Xn = XC / zf + cc // 2
Yn = YC / zf + rr // 2

out2 = griddata(
    points=np.column_stack((X.ravel(), Y.ravel())),
    values=IN.astype(float).ravel(),
    xi=(Xn, Yn),
    method="linear"
)

show_image(out2, "backward warping")

plt.show()