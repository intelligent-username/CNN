"""
Actually train the CNN
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loader import train_loader, val_loader, val_data
from model import EMNIST_VGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EMNIST_VGG(num_classes=62).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam is best?

num_epochs = 200

# Gradient Descent :)
for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0) 
    
    avg_train_loss = running_loss / len(train_loader.dataset)  # weighted average
    
    # --- Validation ---
    model.eval()  # evaluation mode
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_labels).item() * val_images.size(0)
            
            predicted = val_outputs.argmax(dim=1)
            correct += (predicted == val_labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}")

# Save full model
torch.save(model, "emnist_cnn_full.pth")
print("Training complete. Models saved.")
