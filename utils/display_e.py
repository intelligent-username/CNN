"""
Display some of the EMNIST pictures to get an idea of what it looks like
Not necessary for the project, just for visualization. Produced image is used in README.
"""

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

images = idx2numpy.convert_from_file("data/EMNIST/raw/emnist-balanced-train-images-idx3-ubyte")

# choose how many to show
N = 200
side = int(N**0.5)
# If N is not a perfect square, some images will be left out


# normalize to 0–1
imgs = images[:N] / 255.0

# assemble grid
grid = np.zeros((side*28, side*28))
for i in range(side):
    for j in range(side):
        k = i*side + j
        grid[i*28:(i+1)*28, j*28:(j+1)*28] = imgs[k]

plt.figure(figsize=(8,8))
plt.imshow(grid, cmap="gray")
plt.axis("off")
plt.show()

