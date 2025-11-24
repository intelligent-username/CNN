"""
Load Synthwave90k into PyTorch DataLoaders
"""

import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import Dataset


class HuggingFaceSyn90k(Dataset):
    """
    Wraps the Hugging Face Synth90k dataset so PyTorch can use it.
    """
    def __init__(self, hf_dataset_split, transform=None):
        self.hf_ds = hf_dataset_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item["image"]      # PIL Image
        label = item["label"]    # raw text string
        
        if self.transform:
            img = self.transform(img)

        return img, label


def build_loaders(batch_size=2048, num_workers=4, val_fraction=0.1):
    """
    Build train and validation DataLoaders for Synthwave90k dataset.
    """
    print("[Synth90k Loader] Starting processing...")

    # Path resolution
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    syn90k_root = os.path.join(project_root, "data", "Synth90k")
    os.makedirs(syn90k_root, exist_ok=True)

    print(f"[Synth90k Loader] Using dataset cache directory: {syn90k_root}")

    # Download / load from HF
    print("[Synth90k Loader] Loading HuggingFace dataset...")
    ds = load_dataset(
        "priyank-m/MJSynth_text_recognition",
        cache_dir=syn90k_root
    )
    print("[Synth90k Loader] Done loading dataset metadata.")

    # Transform
    print("[Synth90k Loader] Setting up transforms...")
    transform = transforms.ToTensor()

    print("[Synth90k Loader] Wrapping dataset...")
    dataset = HuggingFaceSyn90k(ds["train"], transform=transform)

    print(f"[Synth90k Loader] Total samples: {len(dataset)}")
    print("[Synth90k Loader] Splitting into train/validation...")

    # Split
    dataset_size = len(dataset)
    train_size = int((1.0 - val_fraction) * dataset_size)
    if train_size >= dataset_size:
        train_size = dataset_size - 1
    val_size = dataset_size - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    print(f"[Synth90k Loader] Train: {train_size}, Validation: {val_size}")
    print("[Synth90k Loader] Building DataLoaders...")

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

    print("[Synth90k Loader] DataLoaders ready.")
    return train_loader, val_loader, val_data


if __name__ == "__main__":
    print("[Synth90k Loader] Running loader self-test...")
    train_loader, val_loader, val_data = build_loaders(
        batch_size=64,
        num_workers=4,
        val_fraction=0.1
    )
    print("[Synth90k Loader] Self-test complete.")
