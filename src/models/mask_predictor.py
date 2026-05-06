from __future__ import annotations
import torch.nn as nn


class MaskPredictor(nn.Module):
    """Paper Table 1(b): 5 conv layers, 3x3 stride 1, channels 4,8,16,32,1."""
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 4, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.body(x)
