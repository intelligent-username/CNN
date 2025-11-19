"""
Bare bones template for Synthwave90k for loading the dataset into PyTorch DataLoaders
"""

import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

def build_loaders(batch_size=2048, num_workers=4, val_fraction=0.1):
    """
    Build train and validation DataLoaders for Synthwave90k dataset.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        val_fraction: Fraction of data to use for validation
        
    Returns:
        train_loader, val_loader, val_data
    """
    pass
