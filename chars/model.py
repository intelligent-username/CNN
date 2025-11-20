"""
Model Architecture (for Word Recognition)
"""

import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Generic convolutional block for feature extraction"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        pass

    def forward(self, x):
        pass

class Synthwave90k_CNN(nn.Module):
    """
    CNN architecture for Synthwave90k text recognition.
    """
    
    def __init__(self, num_classes=None):
        super(Synthwave90k_CNN, self).__init__()
        pass

    def forward(self, x):
        pass
