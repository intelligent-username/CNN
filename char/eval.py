"""
Evaluating the model's accuracy using the official EMNIST test set.
"""

import torch
from loader import build_loaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = build_loaders(batch_size=512, num_workers=0, use_test=True)

    model = torch.load(
        "../models/emnist_cnn_full.pth",
        map_location=device,
        weights_only=False
    )
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = criterion(out, labels)

            total_loss += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
            

    print("Average Loss", total_loss / total)
    print("Accuracy", correct / total)

if __name__ == "__main__":
    main()

