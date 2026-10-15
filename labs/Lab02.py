import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d 
from scipy.ndimage import median_filter

rng = np.random.default_rng()

def print_data(matrix: np.ndarray) -> None:
    print(f"\tshape = {matrix.shape}\n\tdtype = {matrix.dtype}\n\tmin = {matrix.min()}\n\tmax = {matrix.max()}")

def show_with_histogram(img: np.ndarray, title: str):
    plt.figure()

    plt.subplot(2,1,1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

    plt.subplot(2,1,2)
    plt.hist(
            np.clip(img, 0, 255).astype(np.uint8).ravel(),
            bins=256,
            range=(-0.5,255.5)
        )
    plt.xlim(0, 255)
    plt.title(f"{title} — histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels");7

def show_comparison(img, noisey, noise_type, box_filtered, gaussian_filtered, median_filtered, n) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(f"{noise_type} noise, {n}x{n} filters")

    for ax, (name, im) in zip(axes[0], [("Original", img), ("Noisy", noisey)]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.set_title(name)
    axes[0, 2].hist(np.clip(noisey, 0, 255).astype(np.uint8).ravel(), bins=256, range=(-0.5, 255.5))
    axes[0, 2].set_title("Noisy histogram")

    for col, (name, im) in enumerate([("Box", box_filtered), ("Gaussian", gaussian_filtered), ("Median", median_filtered)]):
        axes[1, col].imshow(im, cmap="gray", vmin=0, vmax=255); axes[1, col].set_title(name)
        axes[2, col].hist(np.clip(im, 0, 255).astype(np.uint8).ravel(), bins=256, range=(-0.5, 255.5))
        axes[2, col].set_xlim(0, 255)

    for ax in [*axes[0, :2], *axes[1]]:
        ax.axis("off")
    fig.tight_layout();


def show_kernel(ker: np.ndarray, title: str) -> None:
    plt.figure()
    plt.imshow(ker, cmap="viridis")
    plt.colorbar()
    plt.title(title);

def add_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    return img + sigma * rng.standard_normal(img.shape)

def add_salt_and_pepper(img: np.ndarray, density: float) -> np.ndarray:
    R, C = img.shape
    # select entries with probability
    selected_entries = rng.random((R, C)) < density
    # compute random numbers in [0.0, 1.0]
    noise_0_1 = rng.random((R, C))
    # define pepper-salt masks
    salt_mask = selected_entries & (noise_0_1 >= 0.5)
    pepper_mask = selected_entries & (noise_0_1 < 0.5)
    # compute noisey image
    noisey = img.copy()
    noisey[salt_mask] = img.max()
    noisey[pepper_mask] = 0

    return noisey;

def get_box_kernel(n: int) -> np.ndarray:
    # n must be odd in order to have a kernel center and greater than 3
    if n < 3: n = 3;
    if n % 2 == 0: n -=1     
    return (1/(n**2)) * np.ones((n,n));

def get_gaussian_kernel(n: int, sigma: float | None) -> np.ndarray:
    # n must be odd in order to have a kernel center and greater than 3
    if n < 3: n = 3;
    if n % 2 == 0: n -=1
    # if not given, use the rule for sigma: half-width should be 3*sigma
    half = n//2
    sigma = sigma or (1/3 * half)
    
    a = np.arange(-half, half+1)
    X, Y = np.meshgrid(a, a)
    ker = sigma * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # normalise wrt the sum in order to preserve the general intensity
    return ker / ker.sum();

def linear_filter(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
    return convolve2d(img, ker, mode="same", boundary="fill", fillvalue=0);

def median_filt(img: np.ndarray, n: int) -> np.ndarray:
    return median_filter(img, size=(n, n), mode="constant", cval=0);

def get_sharpening_ker_7by7() -> np.ndarray:
    box_kernel = get_box_kernel(7)
    impulse = np.zeros((7,7))
    impulse[3][3] = 1
    return impulse + (impulse - box_kernel)

def main() -> int:
    path = "../data/Lab02/i235.png"

    # simulation config
    sigma = 20
    density = 0.2

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"Not valid path for \"{path}\"")
        return 1

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"\nImage successfully loaded and converted to greyscale:")
    print_data(img)
    show_with_histogram(img, "Original image")

    # convert the image to float for filter operations
    img_f = img.astype(np.float64)
    print(f"\nImage's entries successfully converted to float:")
    print_data(img_f)

    img_gaussian = add_gaussian(img_f, sigma)
    print("\nImage with Gaussian noise:")
    print_data(img_gaussian)
    show_with_histogram(img_gaussian, "Image with Gaussian noise")

    img_sp = add_salt_and_pepper(img_f, density)
    print("\nImage with Salt and Pepper noise")
    print_data(img_sp)
    show_with_histogram(img_sp, "Image with Salt and Pepper noise")

    for n in [3, 7]:
        box_ker = get_box_kernel(n);
        gaussian_ker = get_gaussian_kernel(n, None)
        
        show_kernel(box_ker, f"Box kernel {n}x{n}")
        show_kernel(gaussian_ker, f"Gaussian kernel {n}x{n}")
        
        for noisey, noise_type in [(img_gaussian, "gaussian"), (img_sp, "salt and pepper")]:
            box_filtered = linear_filter(noisey, box_ker)
            print(f"\nImage ({noise_type}) with Box filter {n}x{n}:")
            print_data(box_filtered)
            
            gaussian_filtered = linear_filter(noisey, gaussian_ker)
            print(f"\nImage ({noise_type}) with Gaussian filter {n}x{n}:")
            print_data(gaussian_filtered)

            median_filtered = median_filt(noisey, n);
            print(f"\nImage ({noise_type}) with Median filter {n}x{n}:")
            print_data(median_filtered)

            show_comparison(img_f, noisey, noise_type, box_filtered, gaussian_filtered, median_filtered, n)

    sharp_ker = get_sharpening_ker_7by7()
    show_kernel(sharp_ker, f"Sharpening kernel 7x7")
    
    sharpened = linear_filter(img_f, sharp_ker)
    print("\nImage with sharpening filter 7x7:")
    print_data(sharpened)
    show_with_histogram(sharpened, f"Sharpening filter 7x7")

    plt.show();

    return 0;

if __name__ == "__main__":
    main()