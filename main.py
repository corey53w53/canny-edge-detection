import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
