"""
Load the data in a way that PyTorch can use.
Augments the training data to improve robustness.
"""

# Fix this too

import os
import torch as th
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def build_loaders(batch_size=512, num_workers=4, val_fraction=0.1, use_test=False):
    print("Processing data...")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    emnist_root = os.path.join(project_root, "data")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    emnist_full = datasets.EMNIST(
        root=emnist_root,
        split="byclass",
        train=True,
        download=True,
        transform=train_transform
    )

    dataset_size = len(emnist_full)
    train_size = int((1.0 - val_fraction) * dataset_size)
    val_size = dataset_size - train_size

    # Split
    train_data, val_data = random_split(emnist_full, [train_size, val_size])

    # Transforms
    train_data.dataset.transform = train_transform
    val_data.dataset.transform = val_transform

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    if use_test:
        test_emnist = datasets.EMNIST(
            root=emnist_root,
            split="byclass",
            train=False,
            download=True,
            transform=val_transform
        )
        test_loader = DataLoader(
            test_emnist,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return train_loader, val_loader, test_loader, val_data

    print("Done processing.")
    return train_loader, val_loader, val_data


if __name__ == "__main__":
    print("[SynthText Loader] Running loader self-test :)")
    train_loader, val_loader, val_data = build_loaders(
        batch_size=64,
        num_workers=4,
        val_fraction=0.1
    )
    print("Test finished.")
