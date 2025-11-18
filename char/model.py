"""
The VGG-style model that will be trained
"""

# Note that an input layer isn't needed since the first convolutional layer will work with the images directly

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Generic convolutional block: 2 conv layers + ReLU + MaxPool"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

class EMNIST_VGG(nn.Module):
    """
    The actual CNN that will be trained.
    Brought to you by composition.
    """
    
    def __init__(self, num_classes=62):
        super(EMNIST_VGG, self).__init__()
        
        # The two blocks
        self.block1 = ConvBlock(in_channels=1, out_channels=32)
        self.block2 = ConvBlock(in_channels=32, out_channels=64)

        # Flatten layer (no parameters needed, only reshaping)
        self.flatten = nn.Flatten()

        # Two Dense layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)   # 28x28 -> 14x14 -> 7x7
        self.dropout = nn.Dropout(p=0.5)        # Because regularization is important
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
