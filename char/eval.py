"""
Evaluate the model's accuracy using the official EMNIST test set.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from loader import build_loaders
from model import EMNIST_VGG

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load only the test loader
    _, _, test_loader, _ = build_loaders(batch_size=1024, num_workers=6, use_test=True)

    # Initialize model architecture first
    model = EMNIST_VGG(num_classes=62).to(device)
    
    # Load the saved state dictionary (weights)
    model.load_state_dict(torch.load(
        "../models/EMNIST_CNN.pth",
        map_location=device,
        weights_only=False
    ))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    print("Starting evaluation on Test Set...")
    
    with torch.no_grad():
        # Iterate over test_loader, not val_loader
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = criterion(out, labels)

            total_loss += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4%}")

if __name__ == "__main__":
    main()
