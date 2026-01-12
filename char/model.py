"""
VGG-style CNN for EMNIST character classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolutional block: 2 conv layers, LeakyReLU, MaxPool"""
    def __init__(self, in_channels, out_channels, padding=1, pool_kernel=2, pool_stride=2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
    
    def forward(self, x):
        # CHANGE 1: LeakyReLU prevents "dead neurons," critical for 62-class differentiation.
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.pool(x)
        return x

class EMNIST_VGG(nn.Module):
    """
    The actual CNN that will be trained.
    Brought to you by composition.
    """

    def __init__(self, num_classes=62):
        super(EMNIST_VGG, self).__init__()
        
        # The four blocks
        self.conv1 = ConvBlock(in_channels=1, out_channels=32, pool_kernel=2, pool_stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, pool_stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = ConvBlock(in_channels=64, out_channels=128, pool_stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = ConvBlock(in_channels=128, out_channels=256, pool_stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        # CHANGE 2: Spatial Dropout. 
        # Drops entire feature maps to force redundancy, unlike standard dropout.
        self.spatial_drop = nn.Dropout2d(p=0.1)

        # Flatten layer (no parameters needed, only reshaping)
        self.flatten = nn.Flatten()

        # Two fully-connected layers

        # CHANGE 3: Expanded Width (256 -> 512). 
        # Your Keras model used 512; 256 is a bottleneck for 62 classes.
        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.bn_fc = nn.BatchNorm1d(512) # Added BN to the dense layer for stability
        self.dropout = nn.Dropout(p=0.5)

        # Classifier
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.spatial_drop(x) # Apply mild spatial regularization
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.spatial_drop(x) 
        
        x = self.conv4(x)
        x = self.bn4(x)
        
        x = self.flatten(x)
        
        # Dense Pass
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
