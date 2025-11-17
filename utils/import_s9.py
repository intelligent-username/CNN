import os
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_root = os.path.join(project_root, "data")
syn90k_root = os.path.join(data_root, "Synth90k")
os.makedirs(syn90k_root, exist_ok=True)

# Load from Hugging Face dataset
ds = load_dataset("priyank-m/MJSynth_text_recognition")  # ~12 GB :contentReference[oaicite:0]{index=0}

class HuggingFaceSyn90k(Dataset):
    def __init__(self, hf_dataset_split, transform=None):
        self.hf_ds = hf_dataset_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item["image"]
        # Hugging Face returns a PIL image by default for image feature
        if self.transform:
            img = self.transform(img)
        return img, item["label"]

transform = transforms.ToTensor()
dataset = HuggingFaceSyn90k(ds["train"], transform=transform)

num_samples = len(dataset)
sample_shape = dataset[0][0].shape if num_samples > 0 else (0, 0, 0)

info = {
    "num_samples": [num_samples],
    "image_shape": [tuple(sample_shape)],
    "splits": ["train"],
}

df = pd.DataFrame(info)
print(df)
