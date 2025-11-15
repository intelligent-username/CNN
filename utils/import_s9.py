"""
Import the Synthwave90k dataset (if available locally) using torchvision's
"""

import os
from torchvision import datasets, transforms
import pandas as pd


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_root = os.path.join(project_root, "data")

# Common Synthwave90k locations inside the repo
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
	raise SystemExit(
		"Synthwave90k dataset directory not found. \n"
		"Place the dataset in one of: {}".format(candidate_paths)
	)

# Basic transforms — convert to tensor so we can inspect shape
transform = transforms.Compose([
	transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# === Read out some basic info ===
num_samples = len(dataset)
sample_shape = dataset[0][0].shape if num_samples > 0 else (0, 0, 0)
classes = dataset.classes

info = {
	"num_samples": [num_samples],
	"image_shape": [tuple(sample_shape)],
	"num_classes": [len(classes)],
	"classes": [classes],
	"path": [dataset_dir]
}

df = pd.DataFrame(info)
print(df)

