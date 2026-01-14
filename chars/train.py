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
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loader import build_loaders
from model import SynthText_CRNN

from text import PAD_ID, VOCAB_SIZE, MAX_LABEL_LEN

os.makedirs("../models", exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    max_steps = MAX_LABEL_LEN

    train_loader, val_loader, test_loader, val_subset = build_loaders(
        batch_size=64, num_workers=6, max_steps=max_steps
    )

    learning_rate = 0.025
    # For now I'm going with this learning rate
    # It's relatively high but seems to be working well
    # Anything below 0.01 was useless and completely oscillatory
    # 0.05 semeed like the "upper bound" (also oscillatory) some room to experiment still
    # But the current model is still very undefit, requires more training. Will update once finished

    model = SynthText_CRNN(max_steps=max_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
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
        
        # To continue from last point in the data
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        # best_val_loss = float('inf')

        # patience_counter = ckpt.get('patience_counter', 0)
        patience_counter = 0

        # Teacher force
        tf_ratio = ckpt.get('tf_ratio', 0.5)

        print("Checkpoint loaded.")
        print("│—————————————————————————")        
        print(
            f"│ Resuming from Step {global_step}\n"
            f"│ Best Validation Loss: {best_val_loss:.4f}\n"
            f"│ Patience Counter: {patience_counter}"
            f"│ Teacher Forcing: {tf_ratio}"
            )
        print("│—————————————————————————")

    patience_limit = 10
    min_delta = 0.003
    val_interval = 25
    divergence_threshold = 2.0 

    try:
        num_epochs = 2
        max_samples_per_epoch = 6126000
        batch_size = train_loader.batch_size
        max_batches = max_samples_per_epoch // batch_size

        log_interval = 250                      # log the training loss only every 250 steps
        val_interval = log_interval * 4         # every 4, validate on the validation set
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
            if stop_training:
                break

            model.train()

            print(f"=== Epoch {epoch+1}/{num_epochs} ===")
            
            loader_iter = iter(train_loader)
            ema_train_loss = None

            for step in range(max_batches):
                try:
                    crops, batch_targets = next(loader_iter)
                except StopIteration:
                    break

                if crops.numel() == 0 or batch_targets.numel() == 0:
                    continue

                batch_inputs = crops.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(batch_inputs, targets=batch_targets, teacher_forcing_ratio=tf_ratio)

                    if outputs.dim() != 3:
                        raise RuntimeError(f"Expected model outputs to be 3D, got shape: {tuple(outputs.shape)}")

                    # TBH idk what's going wrong I'm just going to do it by cases
                    if outputs.shape[0] == batch_targets.shape[0]:
                        # (B, T, C)
                        B, T, C = outputs.shape
                        outputs_flat = outputs.reshape(B * T, C)
                    elif outputs.shape[1] == batch_targets.shape[0]:
                        # (T, B, C)
                        T, B, C = outputs.shape
                        outputs_flat = outputs.permute(1, 0, 2).reshape(B * T, C)
                    else:
                        raise RuntimeError(
                            "Model output shape does not match batch size. "
                            f"outputs={tuple(outputs.shape)}, batch_targets={tuple(batch_targets.shape)}"
                        )

                    batch_targets_flat = batch_targets[:, :T].reshape(B * T)
                    loss = criterion(outputs_flat, batch_targets_flat)


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
                        f"┌─────────────────────\t Step {global_step} \t───────────────────────┐\n"
                        f"│ Loss: {train_loss_val:.4f} \t\t|\t\t\tEMA: {ema_train_loss:.4f} \t\n"
                        f"└──────────\t ΔT: {step_time:.2f}s, T: {elapsed:.0f} seconds \t───────────────┘\n"
                    )

                # Validation
                if global_step % val_interval == 0:
                    model.eval()
                    val_loss_acc = 0.0
                    val_count = 0

                    with torch.no_grad():
                        for i, (v_crops, v_targets) in enumerate(val_loader):
                            if i > 100:
                                break
                            if v_crops.numel() == 0 or v_targets.numel() == 0:
                                continue

                            v_inputs = v_crops.to(device, non_blocking=True)
                            v_targets = v_targets.to(device, non_blocking=True)

                            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                                v_out = model(v_inputs, targets=None)
                                if v_out.dim() != 3:
                                    raise RuntimeError(f"Expected validation outputs to be 3D, got shape: {tuple(v_out.shape)}")

                                if v_out.shape[0] == v_targets.shape[0]:
                                    # (B, T, C)
                                    B, T, C = v_out.shape
                                    v_out_flat = v_out.reshape(B * T, C)
                                elif v_out.shape[1] == v_targets.shape[0]:
                                    # (T, B, C)
                                    T, B, C = v_out.shape
                                    v_out_flat = v_out.permute(1, 0, 2).reshape(B * T, C)
                                else:
                                    raise RuntimeError(
                                        "Validation output shape does not match batch size. "
                                        f"v_out={tuple(v_out.shape)}, v_targets={tuple(v_targets.shape)}"
                                    )

                                v_loss = criterion(v_out_flat, v_targets[:, :T].reshape(B * T))


                            val_loss_acc += v_loss.item() * B
                            val_count += B

                    # Mostly unnecessary, just in case something goes wrong later:
                    # if val_count == 0:
                    #     print("Skipping validation: val_count zero (batches were empty), DEBUG?")
                    #     model.train()
                    #     continue

                    avg_val_loss = val_loss_acc / val_count

                    # Exponential Moving Average Math
                    if ema_val_loss is None:
                        ema_val_loss = avg_val_loss
                    else:
                        ema_val_loss = ema_alpha_val * avg_val_loss + (1.0 - ema_alpha_val) * ema_val_loss

                    divergence = avg_val_loss / max(train_loss_val, 1e-6)

                    print(
                        f"\n🟩════════════════════════🟩"
                        f"\n [Validation @ Step {global_step}] "
                        f"\n Loss: {avg_val_loss:.4f} "
                        f"\n VLoss EMA: {ema_val_loss:.4f} "
                        f"\n best: {best_val_loss:.4f} "
                        f"\n ratio: {divergence:.2f}x"
                        f"\n🟩════════════════════════🟩"
                    )

                    # Teacher forcing update
                    if divergence > 1.1:
                            tf_ratio = max(0, tf_ratio - 0.005)
                            # Decreasing it gradually since I don't want the model to go too craz
                    # Patience Check
                    if ema_val_loss < best_val_loss - min_delta:
                        best_val_loss = ema_val_loss
                        patience_counter = 0
                        tmp_path = save_location + ".tmp"
                        torch.save({
                            'global_step': global_step,
                            'model_state': model.state_dict(),
                            'optim_state': optimizer.state_dict(),
                            'best_val_loss': best_val_loss,
                            'patience_counter': patience_counter,
                            'tf_ratio': tf_ratio
                        }, tmp_path)
                        shutil.move(tmp_path, save_location)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience_limit:
                            stop_training = True
                            model.train()
                            break
                        
                    model.train()

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint.")
        tmp_path = save_location + ".tmp"
        torch.save({
            'global_step': global_step,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'tf_ratio': tf_ratio
        }, tmp_path)
        shutil.move(tmp_path, save_location)
        print("│Model Saved")
        print("└─Pytorch, etc. might now take a while to free up memory, disk, and CPU usage. But the model is safe.")
        return
    
    except Exception as e:
        print("Some other error occured. Might be a bug. Saving checkpoint.")
        print("DO NOT ctrl + C")

        try:
            model.cpu()  # Move off GPU before save to avoid CUDA errors
            tmp_path = save_location + ".tmp"
            torch.save({
                'global_step': global_step,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'tf_ratio': tf_ratio
            }, tmp_path)
            shutil.move(tmp_path, save_location)
            print("Model checkpointed")
        except Exception as save_err:
            print(f"Failed to save checkpoint: {save_err}")

        print("----\nCrash log:", e)
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
