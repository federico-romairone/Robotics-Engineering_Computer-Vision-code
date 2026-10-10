import matplotlib.pyplot as plt
import numpy as np

def bilinear_interpolation(img: np.ndarray, x: float, y: float) -> float:
    """
    Compute the light intensity of an output image by bilinear interpolation.

    :param img: original image
    :param x: x-coordinate obtained by some anti-transformation from the processed to the original image (so could be float)
    :param y: y-coordinate obtained by some anti-transformation from the processed to the original image (so could be float)

    :return out_I: Intensity of the pixel in the processed image.
    """
    # compute dimentions of the original image
    r = len(img)
    c = len(img[0])
    # compute left-bottom pixel coordinates of the original image
    i = int(np.floor(y))
    j = int(np.floor(x))
    # compute distances from left-bottom pixel
    a = y - i
    b = x - j
    # compute weight for neighbour pixels (near -> more weight)
    left_top_weight = (1-a) * b
    right_top_weight = a * b
    left_bottom_weight = (1-a) * (1-b)
    right_bottom_weight = a * (1-b)
    # weight pixel intensity with check for out of boundaries
    out_I = (
        (0 if (i < 0 or i >= r or j+1 < 0 or j+1 >= c) else left_top_weight * img[i][j+1]) + 
        (0 if (i+1 < 0 or i+1 >= r or j+1 < 0 or j+1 >= c) else right_top_weight * img[i+1][j+1]) + 
        (0 if (i < 0 or i >= r or j < 0 or j >= c) else left_bottom_weight * img[i][j]) +
        (0 if (i+1 < 0 or i+1 >= r or j < 0 or j >= c) else right_bottom_weight * img[i+1][j])
    )
    return out_I;

def backward_warp(src: np.ndarray, inv_funct, out_dim: tuple[int, int]) -> np.ndarray:
    """
    Compute the transformation of a source image by means of backward warping and returns the output.

    :param src: input image
    :param inv_funct: adopted inverse function
    :param out_dim: dimentions of output image grid

    :return out: the computed image
    """
    # Empty output image
    out = np.zeros(out_dim)
    # For each pixel in output matrix
    for y_out in range(out_dim[0]):
        for x_out in range(out_dim[1]):
            # Map the theoretrical pixel in source image
            (t_x_src, t_y_src) = inv_funct(x_out, y_out)
            out[y_out][x_out] = bilinear_interpolation(src, t_x_src, t_y_src)
    return out;

def inv_translation(x: int, y: int, h: float, v: float) -> tuple[float, float]:
    x_src = x - h;
    y_src = y - v;
    return (x_src, y_src);

def inv_rotation(x: int, y: int, angle: float, rotation_center: tuple[int, int]) -> tuple[float, float]:
    # R^-1 = [cos(angle) sin(angle); -sin(angle) cos(angle)]
    # Coordinate's change: the origin is in (x_c, y_c)
    x_tilde = x - rotation_center[0]
    y_tilde = y - rotation_center[1]
    # Rotation and rollback of coordinates
    x_src = x_tilde * np.cos(angle) + y_tilde * np.sin(angle) + rotation_center[0]
    y_src = - x_tilde * np.sin(angle) + y_tilde * np.cos(angle) + rotation_center[1]
    return (x_src, y_src);

def inv_scaling(x: int, y: int, a: float, b: float) -> tuple[float, float]:
    x_src = x / a
    y_src = y / b
    return (x_src, y_src);

def main():
    # load original image
    img = plt.imread("../data/Lab01/boccadasse.jpg")
    gray = img.mean(axis=2)
    dim = gray.shape

    # Translation
    h = 50
    v = 50
    translated = backward_warp(gray, lambda x, y: inv_translation(x, y, h, v), dim)

    # Rotation
    angle = np.pi / 4
    rotation_center = (dim[1]/2, dim[0]/2)
    rotated = backward_warp(gray, lambda x, y: inv_rotation(x, y, angle, rotation_center), dim)

    # Scale
    a = 2.5
    b = 2.5
    scaled = backward_warp(gray, lambda x, y: inv_scaling(x, y, a, b), dim)

    # Plot
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
    plt.title("Original")

    plt.subplot(3, 2, 3)
    plt.imshow(translated, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Translated by vector [{h}, {v}]")

    plt.subplot(3, 2, 4)
    plt.imshow(rotated, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Rotated by angle {(angle/(2*np.pi))*360:.2f} degrees")

    plt.subplot(3, 2, 5)
    plt.imshow(scaled, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Scaled by {a}x{b}")

    plt.show()

    return 0;

if __name__ == "__main__":
    main();