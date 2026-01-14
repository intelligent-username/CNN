import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from loader import CaptionedSynthTextWordDataset

def compute_stats(max_samples=100000, num_workers=8):
    """
    Roughly find mean and standard deviation of the images
    Prints result but does not save to a file
    Used in data loader for normalization
    Results:
        mean=[0.45160260801311486, 0.47395370089428446, 0.4684704135438617]
        std=[0.28209423291119723, 0.26667546357847927, 0.274351691481741]
    Note: if you run this script as the loader is right now, you'll get means of ~0 and stds of ~1
    """
    ds = CaptionedSynthTextWordDataset(split="train")
    
    n = min(len(ds), max_samples)
    print(f"Computing stats over {n} samples...")
    
    from torch.utils.data import Subset
    subset = Subset(ds, range(n))
    loader = DataLoader(subset, batch_size=256, num_workers=num_workers, 
                        collate_fn=lambda x: [item for item in x if item[0] is not None])
    
    mean = np.zeros(3)
    std = np.zeros(3)
    pixel_count = 0
    processed = 0
    
    for batch_idx, batch in enumerate(loader):
        for img, _ in batch:
            img_np = img.numpy()  # Shape: (C, H, W)
            n_pixels = img_np.shape[1] * img_np.shape[2]
            mean += img_np.sum(axis=(1, 2))
            std += (img_np ** 2).sum(axis=(1, 2))
            pixel_count += n_pixels
        processed += len(batch)
        if batch_idx % 10 == 4:
            print(f"Batch {batch_idx + 1}: Processed {processed}/{n} samples ({100*processed/n:.1f}%)")
    
    print("Computing...")
    mean /= pixel_count
    std = np.sqrt(std / pixel_count - mean ** 2)
    
    print(f"\nStatistics:")
    print(f"mean={mean.tolist()}")
    print(f"std={std.tolist()}")

if __name__ == "__main__":
    compute_stats()
