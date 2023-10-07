import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# task 1
image = matplotlib.image.imread("leaf.jpg")
height = image.shape[0]
width = image.shape[1]
intensities = (0.299 * image[:, :, 0]) + \
    (0.587 * image[:, :, 1]) + \
    (0.114 * image[:, :, 2])
grayscale_image = np.stack(
    (intensities, intensities, intensities), axis=2).astype(int)

plt.imshow(image)
plt.axis("off")
plt.show()

plt.imshow(grayscale_image)
plt.axis("off")
plt.show()

# task 2

pillow_image = Image.fromarray(grayscale_image.astype(np.uint8))
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 16
blurred_image = pillow_image.filter(
    ImageFilter.Kernel((3, 3), gaussian_kernel.flatten()))

plt.imshow(blurred_image)
plt.axis("off")
plt.show()

# task 3

gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

gy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])
gx_image = blurred_image.filter(ImageFilter.Kernel((3, 3), gx.flatten(), 1))
gy_image = blurred_image.filter(ImageFilter.Kernel((3, 3), gy.flatten(), 1))
gx_numpy = np.array(gx_image)
gy_numpy = np.array(gy_image)
g_numpy = np.sqrt(gx_numpy[:, :, 0].astype(np.uint64) **
                  2 + gy_numpy[:, :, 0].astype(np.uint64)**2)
theta = np.arctan2(gx_numpy[:, :, 0], gy_numpy[:, :, 0])
plt.imshow(gx_numpy)
plt.axis("off")
plt.show()

plt.imshow(gy_numpy)
plt.axis("off")
plt.show()

final_gradient = np.maximum(gx_numpy, gy_numpy)


# task 4

x_vals = np.array([np.arange(width) for _ in range(height)])
y_vals = np.array([[i]*width for i in range(height)])
plt.imshow(final_gradient)
plt.axis("off")
plt.show()


def nmsuppression(iy, ix):
    # magnitude of pixel's gradient
    magnitude = g_numpy[iy][ix]
    # in y direction
    if theta[iy][ix] > np.pi/4:
        # local maximum
        if 0 < iy and iy+1 < height and (magnitude < g_numpy[iy+1][ix] or magnitude < g_numpy[iy-1][ix]):
            # suppress by setting to 0
            final_gradient[iy][ix] = 0
    # in x direction
    else:
        if 0 < ix and ix+1 < width and (magnitude < g_numpy[iy][ix+1] or magnitude < g_numpy[iy][ix-1]):
            # suppress by setting to 0
            final_gradient[iy][ix] = 0


vnmsuppression = np.vectorize(nmsuppression)

vnmsuppression(y_vals, x_vals)

plt.imshow(final_gradient)
plt.axis("off")
plt.show()
