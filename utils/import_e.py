"""
Import (download) and unzip the full EMNIST dataset from torchvision's datasets
Then print basic info.
"""

import os
import torch
from torchvision import datasets, transforms
import pandas as pd

print("Importing EMNIST dataset...")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
folder = os.path.join(project_root, "data")

# --- Load EMNIST (all classes: digits + letters) ---
transform = transforms.ToTensor()
emnist = datasets.EMNIST(
    root=folder,
    split="byclass",      # all 62 classes
    train=True,
    download=True,
    transform=transform
)

# === Read out some basic info ===
num_samples = len(emnist)
sample_shape = emnist[0][0].shape
classes = emnist.classes  # labels 0–61 mapped to digits and letters

info = {
    "num_samples": [num_samples],
    "image_shape": [tuple(sample_shape)],
    "num_classes": [len(classes)],
    "classes": [classes]
}

df = pd.DataFrame(info)
print(df)

print("----------------------")
print("Done")
