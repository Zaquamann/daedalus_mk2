#!/usr/bin/env python3
"""Early-fusion AV variant (D3.1): video is interpolated to mel-shaped
channels and concatenated at the input, then a single 2D-conv stack handles
both modalities. No VisualEncoder, no gate, no α. ~530K params."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from train import ResBlock


CV = 4         # number of video-projection channels concat'd with the mel.
TARGET_H = 80  # match mel n_mels
TARGET_W = 99  # match mel n_frames


def _video_to_mel_channels(video: torch.Tensor) -> torch.Tensor:
    """(B, 1, T, H, W) video → (B, CV, TARGET_H, TARGET_W) — CV evenly-spaced
    frames, bilinear-resized to mel-frame dims for input-level concat."""
    B, _, T, H, W = video.shape
    idx = torch.linspace(0, T - 1, steps=CV, device=video.device).round().long()
    sub = video[:, 0, idx, :, :]                 # (B, CV, H, W)
    sub = F.interpolate(sub, size=(TARGET_H, TARGET_W),
                        mode="bilinear", align_corners=False)
    return sub                                    # (B, CV, 80, 99)


class AVEarlyFusionWordResNet(nn.Module):
    """Early-fusion AV model: concat at input, single 2D-conv stack."""

    def __init__(self, num_classes: int):
        super().__init__()
        # block2 widened 128→192 to absorb the visual capacity that AV-fused's
        # VisualEncoder provided.
        self.block1 = ResBlock(1 + CV, 64, stride=2)
        self.block2 = ResBlock(64, 192, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, audio: torch.Tensor,
                video: torch.Tensor | None = None) -> torch.Tensor:
        """`video=None` substitutes a zero video projection (probe)."""
        if video is None:
            v_proj = torch.zeros(audio.size(0), CV, TARGET_H, TARGET_W,
                                  device=audio.device, dtype=audio.dtype)
        else:
            v_proj = _video_to_mel_channels(video)
            v_proj = v_proj.to(audio.dtype)
        x = torch.cat([audio, v_proj], dim=1)    # (B, 1+CV, 80, 99)
        x = self.block1(x)                        # (B, 64, 40, 50)
        x = self.block2(x)                        # (B, 192, 20, 25)
        x = self.gap(x).flatten(1)                # (B, 192)
        x = self.dropout(x)
        return self.fc(x)                         # (B, num_classes)
