"""
RCNN + Encoder/Decoder OCR Model
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: nn.Module inheritance enables PyTorch's autograd functionality automatically
# So don't touch it

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
        super().__init__()
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
        output, cells = self.lstm(x)
        return output


class AdditiveAttention(nn.Module):
    """
    Bahdanau additive attention for encoder-decoder alignment.
    """
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, attn_dim)
        self.dec_proj = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        # encoder_outputs: (B, T, E)
        # hidden: (B, D)
        enc = self.enc_proj(encoder_outputs)
        dec = self.dec_proj(hidden).unsqueeze(1)
        scores = self.v(torch.tanh(enc + dec))
        attn_weights = torch.softmax(scores, dim=1)
        context = (attn_weights * encoder_outputs).sum(dim=1)
        return context, attn_weights


class ConvLayer(nn.Module):
    """An individual convolution layer"""
    conv: nn.Module
    activation: nn.Module
    pool: nn.Module

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
        x = self.pool(self.activation(self.conv(x)))
        return x

class RecBlock(nn.Module):
    """The block of Recurrent Layers"""
    layers: List[nn.Module]

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvBlock(nn.Module):
    """The block of Convolutional Layers"""
    layers: List[nn.Module]

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SynthText_CRNN(nn.Module):
    """The final OCR model with all of the blocks together"""
    conv_block: ConvBlock
    rec_block: RecBlock
    dense_layer: nn.Linear

    def __init__(self):
        super(SynthText_CRNN, self).__init__()

        Layerz = [
            ConvLayer(in_channels=1, out_channels=64),
            ConvLayer(in_channels=64, out_channels=128),
            ConvLayer(in_channels=128, out_channels=256, pool_kernel=2, pool_stride=(2,1)),
            ConvLayer(in_channels=256, out_channels=512, pool_kernel=2, pool_stride=(2,1)),
            ConvLayer(in_channels=512, out_channels=512),
            ConvLayer(in_channels=512, out_channels=512, pool_kernel=2, pool_stride=(2,1)),
            ConvLayer(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0, pool_kernel=1, pool_stride=1),
        ]
        self.conv_block = ConvBlock(layers=Layerz)

        self.layerz = [
            LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True),
            LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True),
        ]
        self.rec_block = RecBlock(layers=self.layerz)

        # Attention layer integrates with decoder hidden states
        self.attention = AdditiveAttention(enc_dim=512, dec_dim=512, attn_dim=256)

        # Decoder LSTM for character-by-character prediction
        self.decoder = nn.LSTMCell(input_size=512, hidden_size=512)

        self.dense_layer = nn.Linear(512, 99)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        encoder_seq = self.rec_block(x)

        B, T, E = encoder_seq.size()
        outputs = []

        h = torch.zeros(B, 512, device=x.device)
        c = torch.zeros(B, 512, device=x.device)

        steps = 25  # fixed decode length placeholder
        for _ in range(steps):
            context, _ = self.attention(encoder_seq, h)
            h, c = self.decoder(context, (h, c))
            out = self.dense_layer(h)
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(1, 0, 2)
        return outputs
