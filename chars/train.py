"""
Actually train the CNN.
Saves to ../models/emnist_cnn_full.pth
Evaluated with char/eval.py
Can continue training by simply re-running this script.
"""

# This loop is basically done

import os
import time

import traceback
import torch
import torch.nn as nn
from model import EMNIST_VGG
from loader import build_loaders

os.makedirs("../models", exist_ok=True)

def main():
    """"
    Main training loop for SynthText model.
    """
    
    
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

    num_epochs = 40

    print("Starting training...")

    try:
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

    except KeyboardInterrupt:
        torch.save(model, save_location)

        print("\nTraining interrupted by user")
        print(traceback.format_exc())
        print("Saving current progress...")
    except Exception:
        torch.save(model, save_location)
        print("Training crashed for some reason")
        print(traceback.format_exc())

    torch.save(model, save_location)

    print("Training ended. Model saved.")

if __name__ == "__main__":
    main()
