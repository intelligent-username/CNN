"""
Load SynthText (HuggingFace: wendlerc/CaptionedSynthText) into PyTorch DataLoaders
"""

import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import Dataset


class HuggingFaceSynthText(Dataset):
    """
    Wraps the Hugging Face SynthText dataset so PyTorch can use it.
    """
    def __init__(self, hf_dataset_split, transform=None):
        self.hf_ds = hf_dataset_split
        self.transform = transform
        print(f"[HuggingFaceSynthText] Dataset wrapper created for {len(self.hf_ds)} samples.")

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item["image"]      # PIL Image
        label = item["text"]     # text label / caption

        if self.transform:
            img = self.transform(img)

        if idx % 100000 == 0 and idx > 0:
            print(f"[HuggingFaceSynthText] Accessed {idx} samples...")

        return img, label


def build_loaders(batch_size=2048, num_workers=4, val_fraction=0.1):
    """
    Build train and validation DataLoaders for SynthText dataset.
    """
    print("[SynthText Loader] Starting processing...")

    # Path resolution
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    synthtext_root = os.path.join(project_root, "data", "SynthText")
    os.makedirs(synthtext_root, exist_ok=True)

    print(f"[SynthText Loader] Using dataset cache directory: {synthtext_root}")

    # Download / load from HF
    print("[SynthText Loader] Loading HuggingFace dataset...")
    ds = load_dataset(
        "wendlerc/CaptionedSynthText",
        cache_dir=synthtext_root
    )
    print("[SynthText Loader] Done loading dataset metadata.")
    print(f"[SynthText Loader] Available splits: {list(ds.keys())}")
    print(f"[SynthText Loader] Number of samples in 'train': {len(ds['train'])}")

    # Transform
    print("[SynthText Loader] Setting up transforms...")
    transform = transforms.ToTensor()

    # Wrap dataset
    print("[SynthText Loader] Wrapping dataset...")
    dataset = HuggingFaceSynthText(ds["train"], transform=transform)

    print(f"[SynthText Loader] Total samples: {len(dataset)}")
    print("[SynthText Loader] Splitting into train/validation...")

    # Split
    dataset_size = len(dataset)
    train_size = int((1.0 - val_fraction) * dataset_size)
    if train_size >= dataset_size:
        train_size = dataset_size - 1
    val_size = dataset_size - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    print(f"[SynthText Loader] Train: {train_size}, Validation: {val_size}")
    print("[SynthText Loader] Building DataLoaders...")

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

    print("[SynthText Loader] DataLoaders ready.")
    return train_loader, val_loader, val_data


if __name__ == "__main__":
    print("[SynthText Loader] Running loader self-test :)")
    train_loader, val_loader, val_data = build_loaders(
        batch_size=64,
        num_workers=4,
        val_fraction=0.1
    )
    print("Test finished.")
