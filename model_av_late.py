#!/usr/bin/env python3
"""Late-fusion AV variant (D3.2): independent A and V branches concatenated
at the penult, no mid-block gate, no α. ~526K params (vs AV-fused's 522K)."""

from __future__ import annotations

import torch
import torch.nn as nn

from model_av import VisualEncoder
from train import ResBlock


class AVLateFusionWordResNet(nn.Module):
    """Independent A and V branches with penult-level concatenation."""

    def __init__(self, num_classes: int):
        super().__init__()
        # Audio stream — mirrors `train.WordResNet`'s body up to GAP.
        self.audio_block1 = ResBlock(1, 64, stride=2)
        self.audio_block2 = ResBlock(64, 128, stride=2)
        self.audio_gap = nn.AdaptiveAvgPool2d(1)

        # Visual stream — lean variant (64-d GAP, matches `VOnlyWordResNet`).
        self.visual = VisualEncoder()
        self.visual_gap = nn.AdaptiveAvgPool2d(1)

        # Late-fusion head
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 + 64, num_classes)

    def forward(self, audio: torch.Tensor,
                video: torch.Tensor | None = None) -> torch.Tensor:
        """`video=None` substitutes a zero visual penultimate (probe)."""
        # Audio path: (B, 1, 80, 99) → (B, 128)
        a = self.audio_block1(audio)
        a = self.audio_block2(a)
        a = self.audio_gap(a).flatten(1)

        # Visual path: (B, 1, T, 88, 88) → (B, 64)
        if video is None:
            v = torch.zeros(a.size(0), 64, device=a.device, dtype=a.dtype)
        else:
            v = self.visual(video)             # (B, 64, 40, 50)
            v = self.visual_gap(v).flatten(1)  # (B, 64)

        # Concat penults → fc
        x = torch.cat([a, v], dim=1)           # (B, 192)
        x = self.dropout(x)
        return self.fc(x)                      # (B, num_classes)
