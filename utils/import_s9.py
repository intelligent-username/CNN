import os
import tarfile
import urllib.request
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_root = os.path.join(project_root, "data")
syn90k_root = os.path.join(data_root, "Synth90k")
os.makedirs(syn90k_root, exist_ok=True)

tar_url = "http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz"
tar_path = os.path.join(syn90k_root, "mjsynth.tar.gz")

if not os.path.exists(tar_path.replace(".tar.gz", "")):
    urllib.request.urlretrieve(tar_url, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=syn90k_root)
    os.remove(tar_path)

label_url = "https://download.openmmlab.com/mmocr/data/mixture/Syn90k/label.txt"
label_path = os.path.join(syn90k_root, "label.txt")

if not os.path.exists(label_path):
    urllib.request.urlretrieve(label_url, label_path)

class Synth90kDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                rel_path = parts[0]
                label = " ".join(parts[1:])
                img_path = os.path.join(self.root_dir, rel_path)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.ToTensor()
images_root = os.path.join(syn90k_root, "mnt", "ramdisk", "max", "90kDICT32px")
dataset = Synth90kDataset(root_dir=images_root, label_file=label_path, transform=transform)

num_samples = len(dataset)
sample_shape = dataset[0][0].shape if num_samples > 0 else (0, 0, 0)
labels_sample = [lbl for _, lbl in dataset.samples[:10]]

info = {
    "num_samples": [num_samples],
    "image_shape": [tuple(sample_shape)],
    "num_classes": ["N/A — text labels vary"],
    "sample_labels": [labels_sample],
    "path": [images_root]
}

df = pd.DataFrame(info)
print(df)
