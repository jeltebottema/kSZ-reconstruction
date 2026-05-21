"""
unet3d.py
=========
Small 3D U-Net for box-to-box noisy-Tb -> v_z regression.

Topology (base_channels=16):
    encoder: [1 -> 16 -> 32 -> 64 -> 128]   at full/half/quarter/eighth res
    bottleneck:                       128 -> 256
    decoder mirrors, with skip concatenations.

At HII_DIM=128 and base=16 this is ~6 M parameters. Memory per sample
at float32 is ~900 MB of activations (single batch); fine on an 8 GB
GPU, on CPU it works but is slow.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(c_in,  c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(inplace=True),
            nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.pool = nn.AvgPool3d(2)
        self.conv = DoubleConv(c_in, c_out)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(c_in, c_in // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(c_in // 2 + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        # pad/crop if shapes disagree by 1 (rare with power-of-two inputs)
        if x.shape[-3:] != skip.shape[-3:]:
            dz, dy, dx = (s - t for s, t in zip(skip.shape[-3:], x.shape[-3:]))
            x = torch.nn.functional.pad(x, (0, dx, 0, dy, 0, dz))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16,
                 depth: int = 4):
        super().__init__()
        assert depth >= 2
        self.depth = depth
        chans: Sequence[int] = [base * (2 ** i) for i in range(depth + 1)]  # e.g. [16,32,64,128,256]

        self.stem = DoubleConv(in_channels, chans[0])
        self.downs = nn.ModuleList([Down(chans[i], chans[i + 1]) for i in range(depth)])
        self.ups   = nn.ModuleList([
            Up(c_in=chans[i + 1], c_skip=chans[i], c_out=chans[i]) for i in range(depth - 1, -1, -1)
        ])
        self.head = nn.Conv3d(chans[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = [self.stem(x)]
        h = skips[0]
        for d in self.downs:
            h = d(h)
            skips.append(h)
        # h is the bottleneck; pop it and feed the rest as skips
        h = skips.pop()
        for u in self.ups:
            h = u(h, skips.pop())
        return self.head(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
