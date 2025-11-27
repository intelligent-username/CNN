"""
This is the VGG-style model that will be trained
"""

# Note: input layers aren't needed since the first convolutional layer will work with the images directly

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Generic convolutional block: 2 conv layers + ReLU + MaxPool"""
    def __init__(self, in_channels, out_channels, padding=1, pool_kernel=2, pool_stride=2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
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
        self.conv1 = ConvBlock(in_channels=1, out_channels=32, pool_kernel=2, pool_stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, pool_stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128, pool_stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=256, pool_stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Flatten layer (no parameters needed, only reshaping)
        self.flatten = nn.Flatten()

        # (Since the Dense layers just take flat inputs)

        # Two fully-connected layers

        # For the first layer, notice that, due to the stride and pool sizes, we need to adjust the input size to 256 * 5 * 5
        self.fc1 = nn.Linear(256 * 5 * 5, 256)
        self.dropout = nn.Dropout(p=0.5)

        # Classifier
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
