#!/usr/bin/env python3
"""Additive-gate AV variant (D3.10): same as `model_av.AVWordResNet` but
the gate is additive (a + α·σ(...)) rather than multiplicative (a · (1 + α·σ(...))).
Probes whether sharp α tuning and gate-inhibition signatures are specific
to multiplicative gain control. Param count matches AV-fused exactly."""

from __future__ import annotations

import torch
import torch.nn as nn

from model_av import VisualEncoder
from train import ResBlock


class AdditiveCrossModalGate(nn.Module):
    """Additive cross-modal gate: a_out = a_mid + α · σ(W_a·a_mid + W_v·v_mid).
    Same module signature as `model_av.CrossModalGate` for drop-in replacement."""

    def __init__(self, channels: int = 64, alpha_init: float = 0.2):
        super().__init__()
        self.Wa = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.Wv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, a_mid: torch.Tensor, v_mid: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.Wa(a_mid) + self.Wv(v_mid))
        return a_mid + self.alpha * g


class AVAdditiveWordResNet(nn.Module):
    """Mid-fusion AV model with an additive (not multiplicative) gate."""

    def __init__(self, num_classes: int, alpha_init: float = 0.2):
        super().__init__()
        # Audio path
        self.audio_block1 = ResBlock(1, 64, stride=2)
        self.audio_block2 = ResBlock(64, 128, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

        # Visual path
        self.visual = VisualEncoder()

        # Mid-level fusion (additive)
        self.gate = AdditiveCrossModalGate(channels=64, alpha_init=alpha_init)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # audio: (B, 1, 80, 99); video: (B, 1, 50, 88, 88) or None.
        a_mid = self.audio_block1(audio)
        if video is None:
            v_mid = torch.zeros_like(a_mid)
        else:
            v_mid = self.visual(video)
        a_fused = self.gate(a_mid, v_mid)
        x = self.audio_block2(a_fused)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
