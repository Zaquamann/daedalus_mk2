#!/usr/bin/env python3
"""AV word recognizer WITH temporal recurrence (Q14: with-vs-without recurrence).

Identical to `AVWordResNet` (model_av.py) in every respect EXCEPT a temporal GRU
runs over the T=50 visual-frame axis of `v_mid` BEFORE the multiplicative
`CrossModalGate`, so the visual gate signal becomes temporally *contextual*
rather than static per-frame. This is form (a) of the Q14 spec ("temporal GRU
over v_mid before fusion"), keeping **mid_mult parity** (the fusion gate is the
unchanged multiplicative gate from model_av.py).

Design (faithful to the spec):
  v_mid = VisualEncoder(video)            # (B, 64, 40, 50)  — dim 3 (=50) is time
  seq   = v_mid.mean(dim=2)               # (B, 64, 50)      — 40-pooled per-frame vec
  seq   = seq.permute(0, 2, 1)            # (B, 50, 64)      — (batch, seq=T, feat)
  h, _  = GRU(seq)                        # (B, 50, gru_hidden)
  v'    = Linear(gru_hidden, 64)(h)       # (B, 50, 64)
  v_mid'= broadcast v' across the 40 axis # (B, 64, 40, 50)  — fed into the gate

Random init, no `load_audio_pretrained` (train both branches from scratch, exactly
like train_av.py). Param-matched to AVWordResNet within ~10% via `gru_hidden`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from train import ResBlock
from model_av import CrossModalGate, VisualEncoder


class AVRecurrentWordResNet(nn.Module):
    """AVWordResNet + a temporal GRU over the v_mid frame axis (Q14 mechanism)."""

    def __init__(self, num_classes: int, alpha_init: float = 0.2,
                 gru_hidden: int = 64, gru_layers: int = 1):
        super().__init__()
        # Audio path — identical to AVWordResNet.
        self.audio_block1 = ResBlock(1, 64, stride=2)
        self.audio_block2 = ResBlock(64, 128, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

        # Visual path — identical encoder.
        self.visual = VisualEncoder()

        # Temporal recurrence over the 50-frame axis of v_mid (the Q14 mechanism).
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)
        self.vgru = nn.GRU(input_size=64, hidden_size=self.gru_hidden,
                           num_layers=self.gru_layers, batch_first=True)
        self.vproj = nn.Linear(self.gru_hidden, 64)

        # Mid-level multiplicative fusion — unchanged (mid_mult parity).
        self.gate = CrossModalGate(channels=64, alpha_init=alpha_init)

    def _recurrent_vmid(self, v_mid: torch.Tensor) -> torch.Tensor:
        """(B,64,40,50) -> temporally-contextualized (B,64,40,50)."""
        B, C, Hh, T = v_mid.shape           # C=64, Hh=40, T=50
        seq = v_mid.mean(dim=2)             # (B, 64, 50)  pool the 40 (spatial) axis
        seq = seq.permute(0, 2, 1)          # (B, 50, 64)  -> (batch, seq, feat)
        out, _ = self.vgru(seq)             # (B, 50, gru_hidden)
        out = self.vproj(out)               # (B, 50, 64)
        out = out.permute(0, 2, 1)          # (B, 64, 50)
        v_ctx = out.unsqueeze(2).expand(B, C, Hh, T)  # broadcast across the 40 axis
        return v_ctx.contiguous()

    def forward(self, audio: torch.Tensor,
                video: torch.Tensor | None = None) -> torch.Tensor:
        # audio: (B,1,80,99); video: (B,1,100|50,88,88) or None.
        a_mid = self.audio_block1(audio)          # (B, 64, 40, 50)
        if video is None:
            v_mid_ctx = torch.zeros_like(a_mid)   # matches AVWordResNet None-path
        else:
            v_mid = self.visual(video)            # (B, 64, 40, 50)
            v_mid_ctx = self._recurrent_vmid(v_mid)
        a_fused = self.gate(a_mid, v_mid_ctx)     # (B, 64, 40, 50)
        x = self.audio_block2(a_fused)            # (B, 128, 20, 25)
        x = self.gap(x).flatten(1)                # (B, 128)
        x = self.dropout(x)
        return self.fc(x)                         # (B, num_classes)
