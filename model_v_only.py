#!/usr/bin/env python3
"""Visual-only baseline — reuses `model_av.VisualEncoder` so the lipreading
representation is literally the same module the AV model uses."""

from __future__ import annotations

import torch
import torch.nn as nn

from model_av import VisualEncoder


class VOnlyWordResNet(nn.Module):
    """Visual-only counterpart of `WordResNet` / `AVWordResNet`."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.visual = VisualEncoder()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, 1, T, 88, 88)
        x = self.visual(video)              # (B, 64, 40, 50)
        x = self.gap(x).flatten(1)          # (B, 64)
        x = self.dropout(x)
        return self.fc(x)                   # (B, num_classes)
