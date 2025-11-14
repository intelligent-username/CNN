import torch
from torchvision import datasets, transforms
import pandas as pd

# --- Load EMNIST (Letters split) ---
transform = transforms.ToTensor()
emnist = datasets.EMNIST(
    root="data",
    split="letters",
    train=True,
    download=True,
    transform=transform
)

# --- Extract basic info ---
num_samples = len(emnist)
sample_shape = emnist[0][0].shape   # (1, 28, 28)
classes = emnist.classes            # labels 1–26 mapped to letters

info = {
    "num_samples": [num_samples],
    "image_shape": [tuple(sample_shape)],
    "num_classes": [len(classes)],
    "classes": [classes]
}

df = pd.DataFrame(info)
print(df)
