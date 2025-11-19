"""
Actually train the CNN (Updated for PyTorch 2.x+)
"""

import torch
import torch.nn as nn
from model import EMNIST_VGG

def main():
    from loader import build_loaders

    # Ensure loader uses batch_size=1024 or 2048
    train_loader, val_loader, val_data = build_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

    model = EMNIST_VGG(num_classes=62).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler = torch.amp.GradScaler('cuda')

    num_epochs = 50

    print("Starting training...")

    try:
        # Gradient Descent :)
        for epoch in range(num_epochs):
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

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}"
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        print("Saving current progress...")
        print("DO NOT ctrl + C")
    except Exception:
        print("Training crashed for reason")
        print("Saving current progress...")
        print("DO NOT ctrl + C")

    torch.save(model, "emnist_cnn_full.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
