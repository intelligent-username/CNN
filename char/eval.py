"""
Evaluate the CNN for accuracy
"""

import torch
from loader import val_loader
from model import EMNIST_VGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EMNIST_VGG(num_classes=62)
model.load_state_dict(torch.load("emnist_cnn.pth"))
model = model.to(device)
model.eval()  # Turn off dropout

criterion = torch.nn.CrossEntropyLoss()

total_loss = 0
correct = 0
total_samples = 0

with torch.no_grad():  # Disable gradient computation
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)  # sum over batch

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

avg_loss = total_loss / total_samples
accuracy = correct / total_samples

print(f"Validation Loss: {avg_loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Optional: compute per-class accuracy or confusion matrix
# Useful for spotting which letters or digits are most often misclassified
