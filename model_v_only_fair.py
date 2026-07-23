#!/usr/bin/env python3
"""Capacity-fair V-only baseline: same `VisualEncoder` as AV plus a
`ResBlock(64, 128)` head matching AV's downstream depth, ~476K params.
Tests whether the V-only ceiling is head-capacity-limited."""

from __future__ import annotations

import torch
import torch.nn as nn

from model_av import VisualEncoder
from train import ResBlock


class VOnlyFairWordResNet(nn.Module):
    """V-only counterpart with the same downstream head depth as AV."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.visual = VisualEncoder()
        self.block2 = ResBlock(64, 128, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.visual(video)             # (B, 64, 40, 50)
        x = self.block2(x)                 # (B, 128, 20, 25)
        x = self.gap(x).flatten(1)         # (B, 128)
        x = self.dropout(x)
        return self.fc(x)
