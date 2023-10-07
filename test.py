from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
img = Image.open("flower.jpg")

gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

gy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])
gx_image = img.filter(ImageFilter.Kernel((3, 3), gx.flatten(), 1))
gy_image = img.filter(ImageFilter.Kernel((3, 3), gy.flatten(), 1))
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
