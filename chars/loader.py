import os
import json
import torch
from typing import Callable, List, Tuple
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

from text import MAX_LABEL_LEN, tokenize_text

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
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else (0, 0, w, h)

def collate_fn(batch):
    """Pads variable width images and stacks pre-tokenized targets."""
    batch = [b for b in batch if b is not None and b[0] is not None and b[1] is not None]
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
        
    images, targets = zip(*batch)
    # Compute mean padding value from batch statistics
    batch_mean = torch.stack([img.mean() for img in images]).mean().item()
    # Reversing dimensions for pad_sequence: (C, W, H) -> (W, C, H)
    images = [img.permute(2, 0, 1) for img in images]
    padded_imgs = pad_sequence(images, batch_first=True, padding_value=batch_mean)
    # Back to (B, C, H, W)
    padded_imgs = padded_imgs.permute(0, 2, 3, 1)
    target_tensor = torch.stack(targets)
    return padded_imgs, target_tensor

class CaptionedSynthTextWordDataset(Dataset):
    def __init__(self, split: str = "train", transform: Callable | None = None, 
                 cache_dir: str | None = None, target_height: int = 32, 
                 max_width: int = 512, max_cap: int | None = None,
                 max_steps: int = MAX_LABEL_LEN):
        self.cache_dir = cache_dir or _synthtext_cache_dir()
        print(f"[LOADER] Loading dataset split: {split}...")
        self.ds = load_dataset("wendlerc/CaptionedSynthText", cache_dir=self.cache_dir, split=split)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45160, 0.47395, 0.46847], std=[0.28209, 0.266675, 0.27435])
        ])
        self.target_height, self.max_width = int(target_height), int(max_width)
        self.max_steps = int(max_steps)
        
        # Suffix index file with cap if provided to avoid overwriting full index
        suffix = f"_{max_cap}" if max_cap else ""
        index_path = os.path.join(self.cache_dir, f"{split}_index{suffix}.json")
        
        if os.path.exists(index_path):
            print(f" Loading cached index from {index_path}")
            with open(index_path, "r") as f: self.index = json.load(f)
        else:
            print(f" No index found. Building index (Cap: {max_cap})...")
            print("This will take a while...")
            self.index = []
            for i, item in enumerate(self.ds):
                # Stop early if cap is reached
                if max_cap is not None and i >= max_cap:
                    print(f" Hit max_cap {max_cap}. Stopping index.")
                    break
                    
                if i % 5000 == 0: print(f"[PROGRESS] Indexed {i} images...")
                ocr = item.get("json", {}).get("ocr_annotation", {})
                for j, text in enumerate(ocr.get("text", [])):
                    if isinstance(text, str) and text.strip(): self.index.append((i, j))
            
            with open(index_path, "w") as f: json.dump(self.index, f)
        
        print(f"[LOADER] Dataset initialized with {len(self.index)} words.")

    def __len__(self) -> int: return len(self.index)

    def __getitem__(self, idx: int):
        img_idx, word_idx = self.index[idx]
        item = self.ds[img_idx]
        img, ocr = item["jpg"], item.get("json", {}).get("ocr_annotation", {})
        
        if word_idx >= len(ocr["text"]): return None
        
        text = ocr["text"][word_idx].strip()
        quad = ocr["bounding_boxes"][word_idx]
        
        x1, y1, x2, y2 = _quad_to_bbox(quad, img.size[0], img.size[1])
        crop_img = img.crop((x1, y1, x2, y2))
        
        if self.target_height > 0 and crop_img.size[1] > 0:
            ratio = self.target_height / crop_img.size[1]
            new_w = min(max(int(round(crop_img.size[0] * ratio)), 1), self.max_width)
            crop_img = crop_img.resize((new_w, self.target_height), resample=Image.BILINEAR)
        
        tokens = tokenize_text(text, max_len=self.max_steps)
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        return self.transform(crop_img), token_tensor

def build_loaders(batch_size: int = 16, num_workers: int = 6, use_test: bool = True, 
                  val_frac: float = 0.02, test_frac: float = 0.02, seed: int = 777,
                  max_steps: int = MAX_LABEL_LEN,
                  max_cap: int | None = None):
    
    # Pass max_cap and max_steps to dataset init
    full = CaptionedSynthTextWordDataset(split="train", max_cap=max_cap, max_steps=max_steps)
    n = len(full)
    
    val_n = int(n * val_frac)
    test_n = int(n * test_frac) if use_test else 0
    train_n = n - val_n - test_n
    
    # Safety for small caps
    if train_n <= 0:
        train_n = n // 2
        val_n = n - train_n
        test_n = 0
    
    print(f"Splitting: Train={train_n}, Val={val_n}, Test={test_n}")
    
    splits = random_split(full, [train_n, val_n, test_n] if test_n > 0 else [train_n, val_n], 
                          generator=torch.Generator().manual_seed(seed))

    loader_args = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True, 
                   "persistent_workers": True, "prefetch_factor": 4, "collate_fn": collate_fn}
    
    train_loader = DataLoader(splits[0], shuffle=True, **loader_args)
    val_loader = DataLoader(splits[1], shuffle=False, **loader_args)
    test_loader = DataLoader(splits[2], shuffle=False, **loader_args) if (use_test and test_n > 0) else None
    
    return train_loader, val_loader, test_loader, Subset(splits[1], list(range(min(512, len(splits[1])))))
