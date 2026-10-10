import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate


def demosaicing(image):
    """Demosaic a GRBG Bayer image into a uint8 RGB image."""
    im = np.asarray(image, dtype=np.float64)

    if im.ndim not in (2, 3):
        raise ValueError("Input must be a 2D image or a 3-channel RGB image.")
    if im.ndim == 3 and im.shape[2] != 3:
        raise ValueError("A color input must have exactly 3 channels.")

    M, N = im.shape[:2]

    # Bayer pattern:
    # G R G R
    # B G B G
    red_mask = np.zeros((M, N))
    green_mask = np.zeros((M, N))
    blue_mask = np.zeros((M, N))

    red_mask[0::2, 1::2] = 1
    green_mask[0::2, 0::2] = 1
    green_mask[1::2, 1::2] = 1
    blue_mask[1::2, 0::2] = 1

    if im.ndim == 2:
        R = im * red_mask
        G = im * green_mask
        B = im * blue_mask
    else:
        R = im[:, :, 0] * red_mask
        G = im[:, :, 1] * green_mask
        B = im[:, :, 2] * blue_mask

    cross_kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=float) / 4

    diagonal_kernel = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ], dtype=float) / 4

    def imfilter(channel, kernel):
        # Match MATLAB imfilter's default zero padding.
        return correlate(channel, kernel, mode="constant", cval=0.0)

    # Interpolate missing green values
    G = G + imfilter(G, cross_kernel)

    # Interpolate blue at red locations, then green locations
    B1 = imfilter(B, diagonal_kernel)
    B2 = imfilter(B + B1, cross_kernel)
    B = B + B1 + B2

    # Interpolate red at blue locations, then green locations
    R1 = imfilter(R, diagonal_kernel)
    R2 = imfilter(R + R1, cross_kernel)
    R = R + R1 + R2

    output = np.stack((R, G, B), axis=-1)

    # Match MATLAB uint8: saturate and round nonnegative values.
    output = np.floor(np.clip(output, 0, 255) + 0.5).astype(np.uint8)

    return output


img = plt.imread("left_#290.bmp")
output = demosaicing(img)

plt.figure()
plt.imshow(img)
plt.axis("off")

plt.figure()
plt.imshow(output)
plt.axis("off")

plt.show()