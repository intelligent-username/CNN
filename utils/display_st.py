"""
Display sample images from SynthText.
Samples 1 image every 1,200 images, up to 100 images.
Tried to keep decently high DPI :)
"""

# Note: don't switch to matplotlib since it's purely on the CPU and will be slow.

import os
import math
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.utils import make_grid, save_image
from datasets import Dataset as HFDataset 
from PIL import Image
import io
import sys

SAMPLE_RATE = 1200
MAX_IMAGES = 100    
IMAGE_SIZE = 512    

# Device for acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), "..")) 
    
data_root = os.path.join(project_root, "data")
synthtext_raw = os.path.join(data_root, "SynthText", "raw")
save_path = os.path.join(project_root, "imgs")
os.makedirs(save_path, exist_ok=True)

arrow_files = []
if not os.path.isdir(synthtext_raw):
    print(f"Error: Directory not found: {synthtext_raw}")
    sys.exit(1)
    
for root, dirs, files in os.walk(synthtext_raw):
    for f in files:
        if f.endswith(".arrow") or "data-" in f: 
            arrow_files.append(os.path.join(root, f))

arrow_files.sort()

if not arrow_files:
    print("Error: No data files found. Check your directory structure.")
    sys.exit(1)

print(f"Found {len(arrow_files)} data files.")

images = []
collected_count = 0
global_scan_index = 0

# Determine the correct column key for images
image_key = None
for f in arrow_files:
    try:
        temp_ds = HFDataset.from_file(f)
        cols = temp_ds.column_names
        for key in ['img', 'image', 'pixel_values', 'jpg', 'picture']:
            if key in cols:
                image_key = key
                break
        if image_key: break
    except:
        continue

if not image_key:
    image_key = "image" 

print(f"Using image column: {image_key}")
print(f"Starting scan. Target: {MAX_IMAGES} images (1 every {SAMPLE_RATE})")

for arrow in arrow_files:
    filename = os.path.basename(arrow)
    try:
        ds = HFDataset.from_file(arrow)
        count_in_file = len(ds)
        # print(f"Processing file: {filename} ({count_in_file} rows)")
        
    except Exception as e:
        print(f"Skipping 'final' file for ease of use: {filename}. Error: {e}")
        continue 
    
    for i in range(count_in_file):
        if global_scan_index % SAMPLE_RATE == 0:
            try:
                img_data = ds[i][image_key] 
                
                if not isinstance(img_data, Image.Image):
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        img_data = Image.open(io.BytesIO(img_data['bytes']))
                    elif isinstance(img_data, np.ndarray):
                        img_data = Image.fromarray(img_data)
                
                if hasattr(img_data, 'mode') and img_data.mode != 'RGB':
                    img_data = img_data.convert('RGB')
                
                # Accelerate this onto GPU otherwise it'll take forever
                # 1. Convert to Tensor
                img_tensor = F.to_tensor(img_data)
                
                # 2. Move to GPU
                img_tensor = img_tensor.to(device)
                
                # 3. Resize on GPU
                img_tensor = F.resize(img_tensor, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
                
                images.append(img_tensor)
                collected_count += 1
                
                # Reduced logging: Only print every 20 images
                if collected_count % 20 == 0 or collected_count == MAX_IMAGES:
                    print(f"Collected image {collected_count}/{MAX_IMAGES} (File: {filename}, Index: {i})")
                
                if collected_count >= MAX_IMAGES:
                    break
            except Exception as e:
                pass
        
        global_scan_index += 1
        
    if collected_count >= MAX_IMAGES:
        break

print(f"Scan complete. Collected {collected_count} images.")

if collected_count == 0:
    print("Error: No images collected.")
    sys.exit(1)

print("Building grid on GPU...")

batch_tensor = torch.stack(images)
side = int(math.ceil(math.sqrt(collected_count)))


# Don't want whitespace
# padding=0, etc.
grid_tensor = make_grid(batch_tensor, nrow=side, padding=0)

out_file = os.path.join(save_path, "SynthText.png")
print("Saving image...")

save_image(grid_tensor, out_file)

print(f"Grid saved to {out_file} :)")
