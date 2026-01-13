"""
RCNN + Encoder/Decoder OCR Model
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from text import MAX_LABEL_LEN, VOCAB_SIZE

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
    ln: nn.LayerNorm

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
        
        # LayerNorm to stabilize gradients
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.ln = nn.LayerNorm(output_size)

    def forward(self, x):
        output, cells = self.lstm(x)
        output = self.ln(output)
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
    """An individual convolution layer with BatchNorm"""

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
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation_type()
        self.pool = pool_type(pool_kernel, pool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class RecBlock(nn.Module):
    """The block of Recurrent Layers"""
    # NOTE: was 2 now is 1

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
    """Final OCR with all blocks together"""
    conv_block: ConvBlock
    rec_block: RecBlock
    dense_layer: nn.Linear

    def __init__(self, num_classes: int = VOCAB_SIZE, max_steps: int = MAX_LABEL_LEN):
        super().__init__()

        self.num_classes = num_classes
        self.max_steps = max_steps

        Layerz = [
            ConvLayer(in_channels=3, out_channels=32),
            ConvLayer(in_channels=32, out_channels=64),
            ConvLayer(in_channels=64, out_channels=128, pool_kernel=(2,1), pool_stride=(2,1)),
            ConvLayer(in_channels=128, out_channels=256, pool_kernel=(2,1), pool_stride=(2,1)),
            ConvLayer(in_channels=256, out_channels=256),
            ConvLayer(in_channels=256, out_channels=256, pool_kernel=(1,2), pool_stride=(1,2)),
            ConvLayer(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, pool_kernel=1, pool_stride=1),
        ]
        self.conv_block = ConvBlock(layers=Layerz)

        self.layerz = [
            LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True),
        ]
        self.rec_block = RecBlock(layers=self.layerz)
        
        # LayerNorm after encoder
        self.encoder_ln = nn.LayerNorm(512)

        self.attention = AdditiveAttention(enc_dim=512, dec_dim=512, attn_dim=256)
        self.decoder = nn.LSTMCell(input_size=512 + num_classes, hidden_size=512)  # include previous token embedding
        self.decoder_ln = nn.LayerNorm(512)  # LayerNorm after decoder cell
        self.embed = nn.Embedding(num_classes, num_classes)  # one-hot embedding
        self.dense_layer = nn.Linear(512, num_classes)

    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        """
        x: images (B, C, H, W)
        targets: LongTensor (B, max_steps) with ground-truth indices
        """
        x = self.conv_block(x)
        if x.dim() == 4 and x.size(2) != 1:
            x = F.adaptive_avg_pool2d(x, (1, x.size(3)))
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        encoder_seq = self.rec_block(x)  # (B, T, E)
        encoder_seq = self.encoder_ln(encoder_seq)  # LayerNorm stabilization

        B, T, E = encoder_seq.size()
        outputs = []

        # initialize decoder hidden state from mean of encoder
        h = torch.tanh(encoder_seq.mean(dim=1) @ nn.Parameter(torch.randn(E, 512, device=x.device)))
        c = torch.zeros_like(h)

        # start token (assume 0)
        input_token = torch.zeros(B, dtype=torch.long, device=x.device)

        for t in range(self.max_steps):
            token_embed = self.embed(input_token)  # (B, num_classes)
            decoder_input = torch.cat([encoder_seq.mean(dim=1), token_embed], dim=1)
            h, c = self.decoder(decoder_input, (h, c))
            h = self.decoder_ln(h)  # LayerNorm to stabilize decoder gradients
            out = self.dense_layer(h)
            outputs.append(out.unsqueeze(1))

            # RNN's teacher forcing
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t]
            else:
                input_token = out.argmax(dim=1)

        outputs = torch.cat(outputs, dim=1)  # (B, max_steps, num_classes)
        return outputs.permute(1, 0, 2)      # (max_steps, B, num_classes)

