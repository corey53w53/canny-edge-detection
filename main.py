import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# task 1
image = matplotlib.image.imread("flower.jpg")
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
g_numpy = np.sqrt(gx_numpy.astype(np.uint64) **
                  2 + gy_numpy.astype(np.uint64)**2)
theta = np.arctan2(gx_numpy, gy_numpy)
plt.imshow(gx_numpy)
plt.axis("off")
plt.show()

plt.imshow(gy_numpy)
plt.axis("off")
plt.show()
