"""
Train the SynthText OCR model.
Includes:
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
from text import tokenize_text

os.makedirs("../models", exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # NOTE: batch_size is "images per batch"; each image has ~10 words on average,
    # so effective word-crops per batch can be ~10x this. 
    # num_workers is for data loading parallelism, might break fi you're on Windows (in which case set to 0)
    train_loader, val_loader, test_loader, val_subset = build_loaders(
        batch_size=6, num_workers=4
    )
    # Could use a larger batch size but I need to use my laptop while the model trains

    model = SynthText_CRNN(max_steps=21).to(device)  # Decoder limited to 21 characters (99.9% of English words are <= this length)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler(enabled=(device.type=='cuda'))

    save_location = "../models/OCR_ATTN.pth"
    global_step = 0
    if os.path.isfile(save_location):
        print("Loading checkpoint...")
        ckpt = torch.load(save_location, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        global_step = ckpt.get('global_step', 0)

    # Early stopping and divergence parameters
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5  # Number of validation checks to wait for improvement
    min_delta = 0.001    # Threshold for significant improvement
    val_interval = 5000
    divergence_threshold = 2.0  # Val loss cannot be > 2x Train loss

    def get_infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    train_gen = get_infinite_loader(train_loader)

    try:
        print("Starting infinite training loop...")
        while True:
            model.train()
            crops, texts = next(train_gen)
            
            if crops is None or len(texts) == 0:
                continue
            
            batch_inputs = crops.to(device)
            tokenized_list = [tokenize_text(t, max_len=model.max_steps) for t in texts]
            batch_targets = torch.stack(tokenized_list).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(batch_inputs)
                T, B, C = outputs.shape
                outputs = outputs.permute(1, 0, 2)
                loss = criterion(outputs.reshape(B*T, C), batch_targets.reshape(B*T))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
            train_loss_val = loss.item()

            if global_step % 100 == 0:
                print(f"Step {global_step} | Train Loss: {train_loss_val:.4f}")

            if global_step % val_interval == 0:
                model.eval()
                val_loss_acc = 0
                val_count = 0
                print(f"\n[CHECKPOINT] Step {global_step}: Running Validation...")
                
                with torch.no_grad():
                    # Check first 100 batches of validation for speed
                    for i, (v_crops, v_texts) in enumerate(val_loader):
                        if i > 100: break
                        if v_crops is None or len(v_texts) == 0: continue
                        
                        v_inputs = v_crops.to(device)
                        v_tokens = torch.stack([tokenize_text(t, max_len=model.max_steps) for t in v_texts]).to(device)
                        
                        v_out = model(v_inputs)
                        v_T, v_B, v_C = v_out.shape
                        v_out = v_out.permute(1, 0, 2)
                        v_loss = criterion(v_out.reshape(v_B*v_T, v_C), v_tokens.reshape(v_B*v_T))
                        val_loss_acc += v_loss.item() * v_B
                        val_count += v_B
                
                avg_val_loss = val_loss_acc / max(val_count, 1)
                divergence = avg_val_loss / max(train_loss_val, 1e-6)
                
                print(f"Step {global_step} | Val Loss: {avg_val_loss:.4f} | Ratio: {divergence:.2f}x")

                if divergence > divergence_threshold:
                    print(f"[STOPPING]: Model diverged (Val Loss too high). Stopping.")
                    break

                if avg_val_loss < (best_val_loss - min_delta):
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    print(f"New best model found. Saving...")
                    torch.save({
                        'global_step': global_step,
                        'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict(),
                    }, save_location)
                else:
                    patience_counter += 1
                    print(f"No significant improvement. Patience: {patience_counter}/{patience_limit}")

                if patience_counter >= patience_limit:
                    print(f"[INFO] Early stopping triggered at step {global_step}.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint.")
        print("DO not ctrl + C again, otherwise progress might be lost.")
        torch.save({
            'global_step': global_step,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
        }, save_location)

if __name__ == "__main__":
    main()
