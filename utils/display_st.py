"""
Display sample images from SynthText.
Samples 1 image every 5,000 images, up to 100 images.
Handles corrupt files and dynamically sizes the output grid.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import Dataset as HFDataset 
from PIL import Image
import io
import sys

SAMPLE_RATE = 1000
MAX_IMAGES = 100    
IMAGE_SIZE = 128    

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

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), 
])

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
        print(f"Processing file: {filename} ({count_in_file} rows)")
        
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
                
                img_tensor = transform(img_data)
                images.append(img_tensor)
                collected_count += 1
                
                print(f"Collected image {collected_count}/{MAX_IMAGES} (File: {filename}, Index: {i}, Global: {global_scan_index})")
                
                if collected_count >= MAX_IMAGES:
                    break
            except:
                pass
        
        global_scan_index += 1
        
    if collected_count >= MAX_IMAGES:
        break

print(f"Scan complete. Collected {collected_count} images.")

if collected_count == 0:
    print("Error: No images collected.")
    sys.exit(1)

side = int(np.ceil(np.sqrt(collected_count)))
grid_h, grid_w = side * IMAGE_SIZE, side * IMAGE_SIZE
grid = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

# Find images and place in grid
for idx, img_tensor in enumerate(images):
    row = idx // side
    col = idx % side
    
    img = img_tensor.numpy().transpose(1, 2, 0)
    
    y_start, y_end = row * IMAGE_SIZE, (row + 1) * IMAGE_SIZE
    x_start, x_end = col * IMAGE_SIZE, (col + 1) * IMAGE_SIZE
    
    if y_end <= grid_h and x_end <= grid_w:
        grid[y_start:y_end, x_start:x_end, :] = img

# --- Display and Save ---
plt.figure(figsize=(40, 40))
plt.imshow(grid)
plt.axis("off")
out_file = os.path.join(save_path, "SynthText.png")

print("Saving image to imgs/")

plt.savefig(out_file, dpi=300)
plt.show()
print(f"Grid saved to {out_file}")
