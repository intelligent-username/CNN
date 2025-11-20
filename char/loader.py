"""
Load the data in a way that PyTorch can use for training
"""

import os
import torch as th
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def build_loaders(batch_size=2048, num_workers=4, val_fraction=0.1):
    transform = transforms.ToTensor()

    print("Processing data...")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    emnist_root = os.path.join(project_root, "data")

    emnist = datasets.EMNIST(
        root=emnist_root,
        split="byclass",
        train=True,
        download=True,
        transform=transform
    )

    print("Done processing.")


    print("Here are some basic details.")
    dataset_size = len(emnist)

    train_size = int((1.0 - val_fraction) * dataset_size)
    if train_size >= dataset_size:
        train_size = dataset_size - 1
    val_size = dataset_size - train_size

    train_data, val_data = random_split(emnist, [train_size, val_size])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_loader, val_loader, val_data
