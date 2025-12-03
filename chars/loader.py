"""
Load SynthText (HuggingFace) into PyTorch DataLoaders.
"""

import os
import pandas as pd
import torch as th
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset


class HuggingFaceSynthText(Dataset):
    def __init__(self, hf_dataset_split, transform=None):
        self.hf_ds = hf_dataset_split
        self.transform = transform
        print(f"[HuggingFaceSynthText] Wrapped {len(self.hf_ds)} samples.")

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]

        # Native SynthText HF format:
        #   item["image"]  → PIL image
        #   item["text"]   → list of word strings (ignored for now)
        img = item["image"]
        text = item["text"]

        if self.transform:
            img = self.transform(img)

        return img, text


def build_loaders(batch_size=512, num_workers=4, val_fraction=0.1):
    print("Processing SynthText...")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    synth_root = os.path.join(project_root, "data", "SynthText", "raw")

    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Fixed: use real SynthText, not CaptionedSynthText
    ds = load_dataset(
        "aimagelab/synthtext",
        cache_dir=synth_root
    )["train"]

    full_dataset = HuggingFaceSynthText(ds, transform=None)

    dataset_size = len(full_dataset)
    train_size = int(dataset_size * (1 - val_fraction))
    val_size = dataset_size - train_size

    train_data, val_data = random_split(full_dataset, [train_size, val_size])

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

    print("Done processing SynthText.")
    return train_loader, val_loader, val_data


if __name__ == "__main__":
    print("[SynthText Loader] Running loader self-test")

    train_loader, val_loader, val_data = build_loaders(
        batch_size=64,
        num_workers=4,
        val_fraction=0.1
    )

    batch = next(iter(train_loader))
    imgs, labels = batch

    df = pd.DataFrame({
        "train_batches": [len(train_loader)],
        "val_batches": [len(val_loader)],
        "train_size": [len(train_loader.dataset)],
        "val_size": [len(val_loader.dataset)],
        "sample_batch_shape": [tuple(imgs.shape)],
        "example_text": [labels[0]],
    })

    print(df)
