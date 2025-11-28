"""
Actually train the CNN.
Saves to ../models/emnist_cnn_full.pth
Evaluated with char/eval.py
Can continue training by simply re-running this script.
"""

import os
import time

import traceback
import torch
import torch.nn as nn
import traceback
from model import EMNIST_VGG
from loader import build_loaders

os.makedirs("../models", exist_ok=True)

def main():

    # Note: num_workers > 1 usually breaks the interruption handling on Windows
    # Mac/Linux/WSL don't have tihs problem since they're not forced to use spawn
    # So I'd recommend them if you're planning on interrupting the training
    train_loader, val_loader, val_data = build_loaders(batch_size=512, num_workers=4, val_fraction=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

    save_location = "../models/EMNIST_CNN.pth"
    if os.path.isfile(save_location) and os.path.getsize(save_location) > 0:
        print(f"Existing model found at {save_location}, loading...")
        model = torch.load(save_location, map_location=device, weights_only=False)
        model = model.to(device)
    else:
        model = EMNIST_VGG(num_classes=62).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler = torch.amp.GradScaler('cuda')

    # NOTE: Empirically, I have found that, after 10 epochs, training loss goes down but validation loss starts going up. 
    # In most cases, that would be a good place to stop.
    # But, with THIS data, even when validation loss is increasing, since it's so augmented and the model is being tuned, letting it "overfit" actually makes it perform better on the actual test set.
    # Counterintuitive, and I can't quite explain why this is happening, but, after much experimentation, it just simply is.

    # To add the validation checks and early stopping,
    # just un-comment the 'patience' counter, threshold, and checker
    # on lines 58, 59, 60, and 124-132

    num_epochs = 40

    # If validation gets worse for too long, stop training

    # best_val_loss = float('inf')
    # patience_counter = 0
    # patience_threshold = 2

    print("Starting training...")

    try:
        # Gradient Descent :)
        for epoch in range(num_epochs):
            start_time = time.time()

            model.train()
            running_loss = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)

            avg_train_loss = running_loss / len(train_loader.dataset)
            epoch_duration = time.time() - start_time

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Duration: {epoch_duration:.2f} seconds"
            )
            
            # NOTE: removing this validation testing will speed up the training by ~8%
            # Validation doesn't need to be done every epoch, either
            # (for example, do it every 3 just to keep an eye on overfitting)
            # But either way, validation doesn't help with the training itself,
            # it's just there for early stopping and as a reference for the user (YOU)

            # Enter Evaluation mode real quick
            model.eval()

            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
            avg_val_loss = val_loss / val_total
            val_accuracy = val_correct / val_total
            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            model.train()
            # Go back to training mode

            # Early termination check

            # if avg_val_loss < best_val_loss:
            #     best_val_loss = avg_val_loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1

            # if patience_counter >= patience_threshold:
            #     print(f"No improvement for {patience_threshold} epochs. Stopping training.")
            #     break


    except KeyboardInterrupt:
        torch.save(model, save_location)

        print("\nTraining interrupted by user")
        print(traceback.format_exc())
        print("Saving current progress...")
        print("DO NOT ctrl + C")
    except Exception:
        torch.save(model, save_location)
        print("Training crashed for some reason")
        print(traceback.format_exc())
        print(Exception)
        print("Saving current progress...")
        print("DO NOT ctrl + C")

    # Save model (architecture + weights)
    torch.save(model, save_location)

    # Saves weights only (basically useless since the architecture is so tiny anyway)
    # torch.save(model.state_dict(), save_location)

    # NOTE: If you save weights only, make sure to go back up to line 35 and set weights_only=True

    print("Training ended. Model saved.")

if __name__ == "__main__":
    # Just for safety, we *could* add:
    
    # import torch.multiprocessing as mp
    # mp.set_start_method('fork', force=True)

    # But it'll proabbly break on Windows 
    # And it's not necessary on UNIX-like systems
    # So comment it out
    # But might need later

    main()
