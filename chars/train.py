"""
Train the SynthText OCR model.
Includes:
- Reading order reconstruction
- Variable-width batching
- Checkpointing (now with Metadata Persistence)
"""

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

    train_loader, val_loader, test_loader, val_subset = build_loaders(
        batch_size=24, num_workers=6
    )

    model = SynthText_CRNN(max_steps=21).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler(enabled=(device.type=='cuda'))

    save_location = "../models/OCR_ATTN.pth"
    
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if os.path.isfile(save_location):
        print("Loading checkpoint...")
        ckpt = torch.load(save_location, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        
        # Restore metadata to prevent "Amnesia"
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        
        print(f"Resumed from Step {global_step} | Best Loss: {best_val_loss:.4f} | Patience: {patience_counter}")

    patience_limit = 1
    min_delta = 0.001
    val_interval = 700
    divergence_threshold = 2.0 

    def get_infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    train_gen = get_infinite_loader(train_loader)

    try:
        print("Starting training loop...")
        while True: # "Epochs" are basically obsolete since SynthText is so big
            model.train()
            crops, texts = next(train_gen)
            
            if crops is None or len(texts) == 0:
                continue
            
            batch_inputs = crops.to(device)
            # Tokenize manually since loader now returns raw strings
            tokenized_list = [torch.tensor(tokenize_text(t, max_len=model.max_steps), dtype=torch.long) for t in texts]
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
                    for i, (v_crops, v_texts) in enumerate(val_loader):
                        if i > 100: break # Validate on fixed subset for speed
                        if v_crops is None or len(v_texts) == 0: continue
                        
                        v_inputs = v_crops.to(device)
                        v_tokens = torch.stack([torch.tensor(tokenize_text(t, max_len=model.max_steps), dtype=torch.long) for t in v_texts]).to(device)
                        
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
                        'best_val_loss': best_val_loss,   # Persist this!
                        'patience_counter': patience_counter # Persist this!
                    }, save_location)
                else:
                    patience_counter += 1
                    print(f"No significant improvement. Patience: {patience_counter}/{patience_limit}")

                if patience_counter >= patience_limit:
                    print(f"[INFO] Early stopping triggered at step {global_step}.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint.")
        print("DON'T ctrl + C again, otherwise progress might be lost.")
        torch.save({
            'global_step': global_step,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter
        }, save_location)

if __name__ == "__main__":
    main()
