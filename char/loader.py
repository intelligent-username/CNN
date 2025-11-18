"""
Load the data in a way that PyTorch can use for training
"""

import torch as th
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define the normalization transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load EMNIST
emnist = datasets.EMNIST(root="data/EMNIST", split="byclass", train=True, download=False, transform=transform)

# Split dataset

dataset_size = len(emnist)

train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size

train_data, val_data = random_split(emnist, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers = 4)
val_loader = DataLoader(val_data, batch_size=64, num_workers = 4)

# Note: num_workers is how many threads are used for data loading (AMD celebrating rn)
