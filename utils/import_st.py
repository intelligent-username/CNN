"""
Download the SynthText dataset from HuggingFace and write it into the data/ folder.
"""

import os
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

print("Starting up...")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_root = os.path.join(project_root, "data")
synthtext_root = os.path.join(data_root, "SynthText", "raw")
os.makedirs(synthtext_root, exist_ok=True)

# Check if dataset already exists
if any(os.scandir(synthtext_root)):
    print(f"[SynthText DownDownloader] Dataset folder {synthtext_root} already exists. Skipping download.")
    download_needed = False
else:
    print(f"[SynthText Downloader] Dataset folder {synthtext_root} is empty. Downloading...")
    download_needed = True

if download_needed:
    # Load SynthText dataset from Hugging Face, store shards locally
    ds = load_dataset(
        "wendlerc/CaptionedSynthText",
        cache_dir=synthtext_root
    )
    print("[SynthText Downloader] Dataset downloaded successfully.")
else:
    # Still load the dataset object from cache
    ds = load_dataset(
        "wendlerc/CaptionedSynthText",
        cache_dir=synthtext_root
    )
    print("[SynthText Downloader] Dataset loaded from cache.")

print(f"[SynthText Downloader] Available splits: {list(ds.keys())}")
print(f"[SynthText Downloader] Number of samples in 'train' split: {len(ds['train'])}")

class HuggingFaceSynthText(Dataset):
    def __init__(self, hf_dataset_split, transform=None):
        self.hf_ds = hf_dataset_split
        self.transform = transform
        print(f"[HuggingFaceSynthText] Dataset wrapper created for {len(self.hf_ds)} samples.")

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item["image"]
        label = item["text"]
        if self.transform:
            img = self.transform(img)
        if idx % 100000 == 0 and idx > 0:
            print(f"[HuggingFaceSynthText] Accessed {idx} samples...")
        return img, label

# Minimal transform: convert to tensor
transform = transforms.ToTensor()

# Wrap the HF dataset without assuming any training split
dataset = HuggingFaceSynthText(ds["train"], transform=transform)

# Print dataset stats
num_samples = len(dataset)
sample_shape = dataset[0][0].shape if num_samples > 0 else (0, 0, 0)

info = {
    "num_samples": [num_samples],
    "image_shape": [tuple(sample_shape)],
}

df = pd.DataFrame(info)
print(df)
print("[SynthText Downloader] Done.")
print("----------------------")

