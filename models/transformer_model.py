import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, C)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class Transformer1DModel(nn.Module):
    def __init__(self, num_channels=3, num_classes=5, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        # Project 3-channel signal to embedding dimension
        self.input_proj = nn.Linear(num_channels, d_model)

        # Positional encoding
        self.pos = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)

        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)

        # Take last time-step
        x = x[:, -1, :]
        return self.fc(x)
