"""
Train the SynthText OCR model.
Includes:
- Reading order reconstruction
- Variable-width batching
- Checkpointing (now with Metadata Persistence)
"""

import sys
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
    optimizer = torch.optim.Adam(model.parameters(), lr=3.14159e-4)
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
        patience_counter = 0

        print("Checkpoint loaded.")
        print("|—————————————————————————")        
        print(f"| Resuming from Step {global_step}\n| Best Loss: {best_val_loss:.4f}\n| Patience Counter: {patience_counter}")
        print("|—————————————————————————")

    patience_limit = 10
    min_delta = 0.003
    val_interval = 25
    divergence_threshold = 2.0 

    num_epochs = 2

    try:
        num_epochs = 2
        max_samples_per_epoch = 612000
        batch_size = train_loader.batch_size
        max_batches = max_samples_per_epoch // batch_size

        log_interval = 250                      # log train loss only every 200 steps
        val_interval = log_interval * 4         # every 4, validate
        patience_limit = 10
        min_delta = 0.001
        divergence_threshold = 1.5

        print("Starting training loop...\n\n")

        start_time = time.perf_counter()
        last_log_time = start_time
        ema_train_loss = None
        ema_alpha = 0.1
        ema_val_loss = None
        ema_alpha_val = 0.2

        stop_training = False

        for epoch in range(num_epochs):
            ema_train_loss = None
            if stop_training:
                break

            print(f"=== Epoch {epoch+1}/{num_epochs} ===")

            model.train()
            loader_iter = iter(train_loader)

            for step in range(max_batches):
                try:
                    crops, texts = next(loader_iter)
                except StopIteration:
                    break

                if crops is None or len(texts) == 0:
                    continue

                batch_inputs = crops.to(device)
                batch_targets = torch.stack([
                    torch.tensor(tokenize_text(t, max_len=model.max_steps), dtype=torch.long)
                    for t in texts
                ]).to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(batch_inputs)      # (T, B, C)
                    T, B, C = outputs.shape
                    outputs = outputs.permute(1, 0, 2) # (B, T, C)
                    loss = criterion(
                        outputs.reshape(B * T, C),
                        batch_targets.reshape(B * T)
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                train_loss_val = loss.item()

                # Update EMA
                if ema_train_loss is None:
                    ema_train_loss = train_loss_val
                else:
                    ema_train_loss = ema_alpha * train_loss_val + (1.0 - ema_alpha) * ema_train_loss

                # Log only every log_interval
                if global_step % log_interval == 0:
                    now = time.perf_counter()
                    step_time = now - last_log_time
                    elapsed = now - start_time
                    last_log_time = now
                    print(
                        f"\n------------------ Step {global_step} ------------------"
                        f"\n| Loss: {train_loss_val:.4f} \t \t |"
                        f"\tEMA: {ema_train_loss:.4f} \t|\n"
                        f"------------- ΔT: {step_time:.2f}s, T: {elapsed:.0f} seconds -----------"
                    )

                # Validation
                if global_step % val_interval == 0:
                    if global_step % val_interval == 0:
                        model.eval()
                        val_loss_acc = 0.0
                        val_count = 0

                        with torch.no_grad():
                            for i, (v_crops, v_texts) in enumerate(val_loader):
                                if i > 100:
                                    break
                                if v_crops is None or len(v_texts) == 0:
                                    continue

                                v_inputs = v_crops.to(device)
                                v_targets = torch.stack([
                                    torch.tensor(tokenize_text(t, max_len=model.max_steps), dtype=torch.long)
                                    for t in v_texts
                                ]).to(device)

                                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                                    v_out = model(v_inputs)
                                    v_T, v_B, v_C = v_out.shape
                                    v_out = v_out.permute(1, 0, 2)
                                    v_loss = criterion(
                                        v_out.reshape(v_B * v_T, v_C),
                                        v_targets.reshape(v_B * v_T)
                                    )

                                val_loss_acc += v_loss.item() * v_B
                                val_count += v_B

                        avg_val_loss = val_loss_acc / max(val_count, 1)

                        # EMA update
                        if ema_val_loss is None:
                            ema_val_loss = avg_val_loss
                        else:
                            ema_val_loss = ema_alpha_val * avg_val_loss + (1.0 - ema_alpha_val) * ema_val_loss

                        divergence = avg_val_loss / max(train_loss_val, 1e-6)

                        print(
                            f"\n🟩=================🟩"
                            f"\n [Validation @ Step {global_step}] "
                            f"\n Loss: {avg_val_loss:.4f} "
                            f"\n VLoss EMA: {ema_val_loss:.4f} "
                            f"\n best: {best_val_loss:.4f} "
                            f"\n ratio: {divergence:.2f}x"
                            f"\n🟩===============🟩"
                        )

                        # EMA-based patience check
                        if ema_val_loss < best_val_loss - min_delta:
                            best_val_loss = ema_val_loss
                            patience_counter = 0
                            torch.save({
                                "global_step": global_step,
                                "model_state": model.state_dict(),
                                "optim_state": optimizer.state_dict(),
                                "best_val_loss": best_val_loss,
                                "patience_counter": patience_counter,
                            }, save_location)
                        else:
                            patience_counter += 1
                            if patience_counter >= patience_limit:
                                stop_training = True
                                model.train()
                                break

                        model.train()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint.")
        torch.save({
            'global_step': global_step,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter
        }, save_location)
        print("Saved")
        return
    
    except Exception as e:
        print("Some other error occured. Might be a bug. Saving checkpoint.")
        print("DO NOT ctrl + C")

        torch.save({
            'global_step': global_step,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter
        }, save_location)

        print("Model checkpointed")

        print("---\nCrash log:", e)
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
