"""
Train the SynthText OCR model.
Includes:
- CRAFT for text detection, cropping
- Reading order reconstruction
- Variable-width batching
- Checkpointing
"""

# Make sure to update detect() and crop_and_resize() function calls to match CRAFT

import os
import time
import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loader import build_loaders
from model import SynthText_CRNN
from text import tokenize_text, collate_fn

os.makedirs("../models", exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # NOTE: batch_size is "images per batch"; each image has ~10 words on average,
    # so effective word-crops per batch can be ~10x this.
    train_loader, val_loader, test_loader, val_subset = build_loaders(
        batch_size=2, num_workers=0
    )

    model = SynthText_CRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler(enabled=(device.type=='cuda'))

    save_location = "../models/ocr_attn_checkpoint.pth"
    start_epoch = 0
    if os.path.isfile(save_location):
        print("Loading checkpoint...")
        ckpt = torch.load(save_location, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        start_epoch = ckpt['epoch'] + 1

    num_epochs = 40

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            start_time = time.time()
            running_loss = 0
            seen_words = 0

            for crops, texts in train_loader:
                # loader returns *all* word crops/texts for the batch
                if not crops:
                    continue
                tokenized_targets = [tokenize_text(t) for t in texts]
                batch_inputs, batch_targets = collate_fn(crops, tokenized_targets, device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(batch_inputs)
                    T, B, C = outputs.shape
                    targets = batch_targets.permute(1, 0)
                    loss = criterion(outputs.reshape(T*B, C), targets.reshape(T*B))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * B
                seen_words += B

            avg_train_loss = running_loss / max(seen_words, 1)
            epoch_duration = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Duration: {epoch_duration:.2f}s")

            # Validation
            model.eval()
            val_loss = 0
            val_total = 0
            with torch.no_grad():
                for crops, texts in val_loader:
                    if not crops:
                        continue
                    tokenized_targets = [tokenize_text(t) for t in texts]
                    batch_inputs, batch_targets = collate_fn(crops, tokenized_targets, device)
                    outputs = model(batch_inputs)
                    T, B, C = outputs.shape
                    targets = batch_targets.permute(1, 0)
                    loss = criterion(outputs.reshape(T*B, C), targets.reshape(T*B))
                    val_loss += loss.item() * B
                    val_total += B
            avg_val_loss = val_loss / max(val_total, 1)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
            }, save_location)

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
        }, save_location)

if __name__ == "__main__":
    main()
