"""
Load the data in a way that PyTorch can use.
Separates training augmentations from validation purity using a custom wrapper.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def build_loaders(batch_size=512, num_workers=4, val_fraction=0.1, use_test=False):
    print("Processing data...")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    emnist_root = os.path.join(project_root, "data")

    # 1. Define Transforms
    # Harder data for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Pure data for validation/testing
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. Determine Split Indices
    # We load one copy just to calculate the random split indices
    base_dataset = datasets.EMNIST(
        root=emnist_root, split="byclass", train=True, download=True, transform=None
    )
    
    dataset_size = len(base_dataset)
    train_size = int((1.0 - val_fraction) * dataset_size)
    val_size = dataset_size - train_size

    # Get the subset objects to extract indices
    # We use a fixed generator for reproducibility
    train_subset_temp, val_subset_temp = random_split(
        base_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_indices = train_subset_temp.indices
    val_indices = val_subset_temp.indices

    # 3. Instantiate Target Datasets
    # Create two distinct dataset objects with their respective transforms baked in
    train_full = datasets.EMNIST(
        root=emnist_root, split="byclass", train=True, download=True, transform=train_transform
    )
    val_full = datasets.EMNIST(
        root=emnist_root, split="byclass", train=True, download=True, transform=val_transform
    )

    # 4. Create Final Subsets
    # Map the random indices to the correctly transformed datasets
    train_data = Subset(train_full, train_indices)
    val_data = Subset(val_full, val_indices)

    # 5. Create Loaders
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    if use_test:
        test_emnist = datasets.EMNIST(
            root=emnist_root,
            split="byclass",
            train=False,
            download=True,
            transform=val_transform
        )
        test_loader = DataLoader(test_emnist, shuffle=False, **loader_kwargs)
        return train_loader, val_loader, test_loader, val_data

    print("Done processing.")
    return train_loader, val_loader, val_data
