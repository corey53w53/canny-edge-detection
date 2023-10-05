import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
img = Image.open("flower.jpg")
print(img)
print(type(img))
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 16.0
blurred_image = img.filter(
    ImageFilter.Kernel((3, 3), gaussian_kernel.flatten()))

plt.imshow(img)
plt.axis("off")
plt.show()

plt.imshow(blurred_image)
plt.axis("off")
plt.show()
