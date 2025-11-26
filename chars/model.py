"""
RCNN End-to-End OCR Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from __future__ import annotations
from typing import List

# NOTE: nn.Module inheritance enables PyTorch's autograd functionality automatically.

class LSTM(nn.Module):
    """
    LSTM layer for sequence modeling.
    We'll use two of these for the final model, after the CNN layers.
    """
    
    hidden_size: int
    num_layers: int
    bidirectional: bool
    lstm: nn.LSTM

    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        """
        Initialize the Long Short-Term Memory layer.
        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state `h`.
            num_layers (int): Number of recurrent layers.
            bidirectional (bool): If True, becomes a bidirectional LSTM.
        """

        super().__init__()

        # Store for later use
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )


    def forward(self, x):
        """
        Forward pass through the LSTM layer.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        Returns:
            output (Tensor): Output tensor of shape (batch_size, seq_length, hidden_size * num_directions).
        """

        # PyTorch's LSTM handles this for me
        # Note that `cells` is just the cell states, not really needed, but returned by default.
        output, cells = self.lstm(x)
        return output

    # NOTE: PyTorch's nn doesn't let you redefine 'backward'


class ConvLayer(nn.Module):
    """
    A single convolutional layer followed by ReLU and MaxPool.
    """
    conv: nn.Module         # Type of convolution
    activation: nn.Module   # Activation function
    pool: nn.Module         # Pooling layer
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_kernel: int = 2,
        pool_stride: int = 2,
        conv_type=nn.Conv2d,
        activation_type=nn.ReLU,
        pool_type=nn.MaxPool2d
    ):
        super().__init__()
        self.conv = conv_type(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation_type()
        self.pool = pool_type(pool_kernel, pool_stride)

    def forward(self, x):
        
        # Simply apply the filter, pass to the activation, and pool for final result
        x = self.pool(self.activation(self.conv(x)))
        return x
    
class RecBlock(nn.Module):
    """
    The Block of Recurrent Layers
    """

    layers: List[nn.Module]

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvBlock(nn.Module):
    """
    The Block of Convolutional Layers
    """

    layers: List[nn.Module]

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SynthText_CRNN(nn.Module):
    """
    CNN architecture for SynthText text recognition.
    """

    conv_block: ConvBlock
    rec_block: RecBlock
    dense_layer: nn.Linear
    
    def __init__(self):
        """
        Instantiate the SPECIFIC model architecture.
        Note that CL stands for Convolutional Layer and RL stands for Recurrent Layer.
        """


        super(SynthText_CRNN, self).__init__()

        # Take advantage of defaults defined earlier btw
        
        # First convolution layers
        Layerz = [
            ConvLayer(in_channels=1, out_channels=64),
            ConvLayer(in_channels=64, out_channels=128),
            ConvLayer(in_channels=128, out_channels=256, pool_kernel=2, pool_stride=(2,1)),  # height pooling only
            ConvLayer(in_channels=256, out_channels=512, pool_kernel=2, pool_stride=(2,1)),  # height pooling only
            ConvLayer(in_channels=512, out_channels=512),
            ConvLayer(in_channels=512, out_channels=512, pool_kernel=2, pool_stride=(2,1)),  # height pooling only
            ConvLayer(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0, pool_kernel=1, pool_stride=1),
        ]
        self.conv_block = ConvBlock(layers=Layerz)

        # Then recurrent layers
        self.layerz = [
            LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True),
            LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True),  # corrected input_size for stacking
        ]
    
        self.rec_block = RecBlock(layers=self.layerz)

        # 99 output classes since:
        #    - 26 for lowercase
        #    - 26 for uppercase
        #    - 10 for digits
        #    - 38 for special characters
        #          [!, @, #, $, %, ^, &, etc. etc.]
        #          and then some room for random, unknown characters       
        self.dense_layer = nn.Linear(512, 99)


    def forward(self, x):
        # Apply conv block
        x = self.conv_block(x)  # shape: (B, C, H, W)

        # Collapse height dimension
        x = x.squeeze(2)  # assuming final H=1; shape: (B, C, W)
        x = x.permute(0, 2, 1)  # shape: (B, W, C) for LSTM input

        # Apply recurrent block
        x = self.rec_block(x)  # shape: (B, W, hidden*2)

        # Dense layer applied at each time step
        x = self.dense_layer(x)  # shape: (B, W, num_classes)

        # Permute to (W, B, num_classes) for CTC compatibility
        x = x.permute(1, 0, 2)
        return x
