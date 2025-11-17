"""
Display some sample images from the Synthwave90k dataset. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_root = os.path.join(project_root, "data")

candidate_paths = [
	os.path.join(data_root, "Synthwave90k", "processed"),
	os.path.join(data_root, "Synthwave90k", "raw"),
	os.path.join(data_root, "Synthwave90k")
]

dataset_dir = None
for p in candidate_paths:
	if os.path.isdir(p):
		dataset_dir = p
		break

if dataset_dir is None:
	raise SystemExit("Synthwave90k dataset directory not found.")

transform = transforms.Compose([
	transforms.Resize((128, 128)),  # scale to a small, uniform size for display
	transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# choose how many to show
N = 64
N = min(N, len(dataset))
side = int(np.sqrt(N))

# build grid
grid = np.zeros((side * 128, side * 128, 3), dtype=float)
for i in range(side):
	for j in range(side):
		k = i * side + j
		if k >= N:
			break
		img = dataset[k][0].numpy()  # tensor -> numpy (C, H, W)
		# convert to HWC and scale 0–1
		img = np.transpose(img, (1, 2, 0))
		# if image is single-channel, expand to 3 channels
		if img.shape[2] == 1:
			img = np.tile(img, (1, 1, 3))
		grid[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128, :] = img

plt.figure(figsize=(8, 8))
plt.imshow(grid)
plt.axis("off")
plt.show()

