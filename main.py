import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

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

pillow_image = Image.fromarray(grayscale_image.astype(np.uint8))
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 16
blurred_image = pillow_image.filter(
    ImageFilter.Kernel((3, 3), gaussian_kernel.flatten()))

plt.imshow(blurred_image)
plt.axis("off")
plt.show()
