"""
Actually train the CNN.
Saves to ../models/emnist_cnn_full.pth
Evaluated with char/eval.py
Can continue training by simply re-running this script.
"""

import os
import time

import torch
import torch.nn as nn
import traceback
from model import EMNIST_VGG

os.makedirs("../models", exist_ok=True)

def main():
    from loader import build_loaders

    train_loader, val_loader, val_data = build_loaders(batch_size=512, num_workers=4, val_fraction=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

    save_location = "../models/emnist_cnn_full.pth"
    if os.path.isfile(save_location) and os.path.getsize(save_location) > 0:
        print(f"Existing model found at {save_location}, loading...")
        model = torch.load(save_location, map_location=device)
        model = model.to(device)
    else:
        model = EMNIST_VGG(num_classes=62).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler = torch.amp.GradScaler('cuda')

    # Feel free to end earlier if it plateaus too hard
    # This is just here as a nice default lenght/safeguard
    num_epochs = 60

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

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        print(traceback.format_exc())
        print(KeyboardInterrupt)
        print("Saving current progress...")
        print("DO NOT ctrl + C")
    except Exception:
        print("Training crashed for some reason")
        print(traceback.format_exc())
        print(Exception)
        print("Saving current progress...")
        print("DO NOT ctrl + C")

    torch.save(model, save_location)
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
