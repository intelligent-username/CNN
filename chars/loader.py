"""SynthText (CaptionedSynthText) dataloading utilities.

This module is used by chars/train.py via build_loaders().

The HuggingFace dataset returns samples shaped like:
- jpg: PIL image
- json: dict containing ocr_annotation { bounding_boxes: [quad...], text: [str...] }

This loader trains on *every word* in each image by returning all word crops.
The DataLoader collate function flattens per-image word lists into one big list
of (crop, text) for the batch.
"""

from __future__ import annotations

import os
from typing import Callable, List, Tuple

import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _synthtext_cache_dir() -> str:
    return os.path.join(_project_root(), "data", "SynthText", "raw")


def _quad_to_bbox(quad: List[List[float]], w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x1 = max(int(min(xs)), 0)
    y1 = max(int(min(ys)), 0)
    x2 = min(int(max(xs)), w)
    y2 = min(int(max(ys)), h)
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


class CaptionedSynthTextAllWordsDataset(Dataset):
    """Returns (list[word_crop_tensor], list[word_text]) per image."""

    def __init__(
        self,
        split: str = "train",
        transform: Callable | None = None,
        cache_dir: str | None = None,
        target_height: int = 32,
        max_width: int = 512,
    ):
        self.cache_dir = cache_dir or _synthtext_cache_dir()
        self.ds = load_dataset("wendlerc/CaptionedSynthText", cache_dir=self.cache_dir, split=split)
        self.transform = transform or transforms.ToTensor()
        self.target_height = int(target_height)
        self.max_width = int(max_width)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[int(idx)]
        img: Image.Image = item["jpg"]
        meta = item.get("json", {})
        ocr = meta.get("ocr_annotation", {})

        texts: List[str] = list(ocr.get("text", []))
        bbs = list(ocr.get("bounding_boxes", []))

        crops: List[torch.Tensor] = []
        crop_texts: List[str] = []

        if texts and bbs:
            for text, quad in zip(texts, bbs):
                if not isinstance(text, str):
                    continue
                text = text.strip("\n")
                if text == "":
                    continue

                x1, y1, x2, y2 = _quad_to_bbox(quad, img.size[0], img.size[1])
                # Filter tiny boxes (helps avoid degenerate conv/pool shapes)
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue

                crop_img = img.crop((x1, y1, x2, y2))

                # Normalize crop sizes: fixed height, capped width.
                if self.target_height > 0 and crop_img.size[1] > 0:
                    new_w = int(round(crop_img.size[0] * (self.target_height / crop_img.size[1])))
                    new_w = max(new_w, 1)
                    if self.max_width > 0:
                        new_w = min(new_w, self.max_width)
                    crop_img = crop_img.resize((new_w, self.target_height), resample=Image.BILINEAR)

                crop_tensor = self.transform(crop_img)
                crops.append(crop_tensor)
                crop_texts.append(text)

        return crops, crop_texts


def _flatten_words_collate(batch):
    """Flatten per-image word lists into per-batch word lists."""
    crops: List[torch.Tensor] = []
    texts: List[str] = []
    for crop_list, text_list in batch:
        crops.extend(crop_list)
        texts.extend(text_list)
    return crops, texts


def build_loaders(
    batch_size: int = 16,
    num_workers: int = 0,
    use_test: bool = True,
    val_frac: float = 0.02,
    test_frac: float = 0.02,
    seed: int = 1337,
):
    """Build train/val/test loaders.

    Returns: (train_loader, val_loader, test_loader, val_subset)
    """

    full = CaptionedSynthTextAllWordsDataset(split="train")
    n = len(full)
    val_n = int(n * val_frac)
    test_n = int(n * test_frac) if use_test else 0
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("Split fractions too large; no samples left for training")

    gen = torch.Generator().manual_seed(seed)
    splits = random_split(full, [train_n, val_n, test_n] if use_test else [train_n, val_n], generator=gen)
    train_ds = splits[0]
    val_ds = splits[1]
    test_ds = splits[2] if use_test else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_flatten_words_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_flatten_words_collate)
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_flatten_words_collate)
        if test_ds is not None
        else None
    )

    val_subset = Subset(val_ds, list(range(min(512, len(val_ds)))))
    return train_loader, val_loader, test_loader, val_subset
