#!/usr/bin/env python3
"""Parallel / late-fusion AV recognizer with a learned reliability gate.

Motivation (debugger task #5): the committed fusions (multiplicative
`AVWordResNet`, additive `AVAdditiveWordResNet`) let video enter ONLY as a
bounded modulation of the audio carrier `a_mid`; the AV readout always flows
through `audio_block2 -> fc`. There is NO independent video->classifier path,
so when audio is corrupted the AV decision is dragged below video-only
(d'_AV < d'_V; gain_over_best < 1 at high sigma) — a violation of the
biological "integration is never worse than the better single cue" bound.

Fix topology: give video a fully INDEPENDENT route to the logits and combine
the two readouts at the LOGIT level with a LEARNED reliability gate that can
down-weight an unreliable modality:

    logit_a = audio_fc(audio_penult)          # pure audio readout
    logit_v = visual_fc(visual_penult)        # pure, independent video readout
    [w_a, w_v] = softmax(rel_gate([a_pen, v_pen]))   # per-sample reliability
    logits   = w_a * logit_a + w_v * logit_v

Because the weights are a per-sample convex combination conditioned on both
penultimates, the network can learn to route to video when audio is noisy
(w_a -> 0, logits -> logit_v, so d'_AV -> d'_V floor) and to audio when audio
is clean (w_a -> 1). At equal reliability w_a ~ w_v ~ 0.5 and the average of
two independent equal-d' cues yields ~sqrt(2)*d' — the optimal-cue-combination
target. The gate only LEARNS this if trained with audio-noise augmentation
across the sigma range (see train_av_latefusion.py); the architecture alone
does not guarantee the floor, the trained gate does.

The audio branch mirrors `train.WordResNet`'s body (audio_block1/2 -> GAP ->
fc), so it stays weight-compatible with the A-only checkpoint, and the visual
branch mirrors `VOnlyFairWordResNet`'s proven-capable readout (VisualEncoder ->
ResBlock(64->128) -> GAP -> Linear(128->C)), giving the video head the same
downstream depth as the audio path / the d'_V specialist (D312 fix).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_av import VisualEncoder, CrossModalGate
from train import ResBlock


class AVLateFusionReliabilityWordResNet(nn.Module):
    """Independent audio & video logit readouts fused by a learned reliability
    gate. `forward` returns the fused logits tensor by default (so it is a drop
    -in for `AVWordResNet(audio, video)` in the d' harness); pass
    `return_parts=True` during training to also get the per-modality logits and
    the gate weights for auxiliary supervision.
    """

    def __init__(self, num_classes: int, alpha_init: float = 0.2,
                 use_mid_gate: bool = False):
        super().__init__()
        # --- Audio branch (mirrors WordResNet body up to GAP) ---
        self.audio_block1 = ResBlock(1, 64, stride=2)
        self.audio_block2 = ResBlock(64, 128, stride=2)
        self.audio_gap = nn.AdaptiveAvgPool2d(1)
        self.audio_fc = nn.Linear(128, num_classes)

        # --- Visual branch (independent readout). D312 fix: mirror the audio
        #     path's downstream depth — VisualEncoder(64-d) -> ResBlock(64->128)
        #     -> GAP(128-d) -> Linear(128->C), i.e. VOnlyFairWordResNet's
        #     proven-capable topology. The prior 64-d readout had NO ResBlock and
        #     could not fit the lipread task, capping the video head at d'~1.94
        #     (PROVEN root cause of the E1c bar-(i) fail). ---
        self.visual = VisualEncoder()
        self.visual_block2 = ResBlock(64, 128, stride=2)
        self.visual_gap = nn.AdaptiveAvgPool2d(1)
        self.visual_fc = nn.Linear(128, num_classes)

        # --- Optional mid-block cross-modal gate (default OFF: keep the audio
        #     path purely audio so video's route to the decision is genuinely
        #     independent). Kept available because the lead spec'd "gate
        #     optional". ---
        self.use_mid_gate = bool(use_mid_gate)
        if self.use_mid_gate:
            self.gate = CrossModalGate(channels=64, alpha_init=alpha_init)

        # --- Learned reliability gate over the two readouts ---
        # Reads BOTH penultimates (detached, so it estimates reliability without
        # distorting the readouts) and emits a per-sample convex weight.
        self.rel_gate = nn.Sequential(
            nn.Linear(128 + 128, 64),   # audio_pen(128) + video_pen(128, post-ResBlock)
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor | None = None,
        audio_dead: torch.Tensor | None = None,
        video_dead: torch.Tensor | None = None,
        return_parts: bool = False,
    ):
        """audio: (B, 1, 80, 99); video: (B, 1, T, 88, 88) or None.

        `video=None` substitutes a zero visual penult (audio-via-AV probe).

        `audio_dead` / `video_dead` are optional (B,) boolean masks for per-sample
        MODALITY DROPOUT. A dead audio stream zeroes the mel input (so a_mid =
        audio_block1(0) — identical to the d' harness's audio_kind="zero"); a dead
        video stream zeroes v_mid (identical to video_kind="zero" / video=None).
        Training passes these so the reliability gate sees FULLY-dead streams
        (out-of-distribution for the sigma-noise alone), and the d' ablation eval
        reproduces the same dead-stream representations via a zero mel / video=None
        — so the floor-when-a-stream-dies behaviour is trained and measured the
        same way.
        """
        if audio_dead is not None:
            keep_a = (~audio_dead).view(-1, 1, 1, 1).to(audio.dtype)
            audio = audio * keep_a
        a_mid = self.audio_block1(audio)                 # (B, 64, 40, 50)
        if video is None:
            v_mid = torch.zeros_like(a_mid)
        else:
            v_mid = self.visual(video)                   # (B, 64, 40, 50)
            if video_dead is not None:
                keep_v = (~video_dead).view(-1, 1, 1, 1).to(v_mid.dtype)
                v_mid = v_mid * keep_v

        a_in = self.gate(a_mid, v_mid) if self.use_mid_gate else a_mid
        a = self.audio_block2(a_in)                      # (B, 128, 20, 25)
        a_pen = self.audio_gap(a).flatten(1)             # (B, 128)
        # Video readout now mirrors the audio path's depth (ResBlock -> GAP).
        # v_mid (64-ch) is still what the optional mid-gate reads above.
        v = self.visual_block2(v_mid)                    # (B, 128, 20, 25)
        v_pen = self.visual_gap(v).flatten(1)            # (B, 128)

        logit_a = self.audio_fc(self.dropout(a_pen))     # (B, C)
        logit_v = self.visual_fc(self.dropout(v_pen))    # (B, C)

        # Reliability weights from (detached) penults: estimate, don't distort.
        rel_in = torch.cat([a_pen.detach(), v_pen.detach()], dim=1)  # (B, 256)
        w = torch.softmax(self.rel_gate(rel_in), dim=1)             # (B, 2)
        w_a, w_v = w[:, 0:1], w[:, 1:2]

        logits = w_a * logit_a + w_v * logit_v           # (B, C)
        if return_parts:
            return logits, logit_a, logit_v, w
        return logits

    def load_audio_pretrained(self, checkpoint_state_dict: dict) -> None:
        """Map A-only checkpoint keys (block1/block2/fc) -> audio_block1/
        audio_block2/audio_fc. Visual + gates stay freshly initialized."""
        renames = {
            "block1": "audio_block1",
            "block2": "audio_block2",
            "fc": "audio_fc",
        }
        new_sd = {}
        for k, v in checkpoint_state_dict.items():
            for old, new in renames.items():
                if k.startswith(old + "."):
                    new_sd[new + k[len(old):]] = v
                    break
            else:
                new_sd[k] = v
        self.load_state_dict(new_sd, strict=False)
