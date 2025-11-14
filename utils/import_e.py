"""
Import (downoad) and unzip the standard EMNIST dataset from torchvision's datasets
Then print basic info.
"""

import os
import torch
from torchvision import datasets, transforms
import pandas as pd


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
folder = os.path.join(project_root, "data")

# --- Load EMNIST (Letters split) ---
transform = transforms.ToTensor()
emnist = datasets.EMNIST(
    root=folder,
    split="letters",
    train=True,
    download=True,
    transform=transform
)

# ATP, data is downloaded and unzipped


# === Read out some basic info ===
num_samples = len(emnist)
sample_shape = emnist[0][0].shape
classes = emnist.classes            # labels 1–26 mapped to letters

info = {
    "num_samples": [num_samples],
    "image_shape": [tuple(sample_shape)],
    "num_classes": [len(classes)],
    "classes": [classes]
}

df = pd.DataFrame(info)
print(df)
