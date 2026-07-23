#!/usr/bin/env python3
"""Extract AV / A-only / V-fair penult features for 32 val samples, save .npy."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_av import AVWordResNet
from model_v_only_fair import VOnlyFairWordResNet
from paired_dataset import PairedAVDataset
from train import WordResNet


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# 32 is enough to see shapes + variances; 05_linear_probe.py needs more
# samples (180 classes) so this bumps to 1000. Drop back to 32 for a quick
# look — the mechanics are identical.
N_SAMPLES = 1000
BATCH = 32


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for path in ("models/av_fused.pt", "models/audio_only_filtered.pt",
                 "models/video_only_fair.pt", "processed/splits.pt"):
        if not os.path.exists(os.path.join(ROOT, path)):
            sys.exit(f"missing {path} — see top-level README quickstart")

    ds = PairedAVDataset(t_stride=2)
    splits = torch.load(os.path.join(ROOT, "processed", "splits.pt"),
                         weights_only=False)
    val_idx = np.asarray(splits["val_idx"])[:N_SAMPLES]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    av_ckpt = torch.load(os.path.join(ROOT, "models/av_fused.pt"),
                          map_location=device, weights_only=False)
    av = AVWordResNet(len(av_ckpt["label_to_idx"])).to(device)
    av.load_state_dict(av_ckpt["model_state_dict"])
    av.eval()

    a_ckpt = torch.load(os.path.join(ROOT, "models/audio_only_filtered.pt"),
                         map_location=device, weights_only=False)
    a_only = WordResNet(len(a_ckpt["label_to_idx"])).to(device)
    a_only.load_state_dict(a_ckpt["model_state_dict"])
    a_only.eval()

    v_ckpt = torch.load(os.path.join(ROOT, "models/video_only_fair.pt"),
                         map_location=device, weights_only=False)
    v_fair = VOnlyFairWordResNet(len(v_ckpt["label_to_idx"])).to(device)
    v_fair.load_state_dict(v_ckpt["model_state_dict"])
    v_fair.eval()

    print(f"Extracting penults for {N_SAMPLES} samples (batch={BATCH})...")
    av_penult_chunks, av_vzero_chunks = [], []
    a_pen_chunks, v_pen_chunks = [], []
    labels_chunks = []
    with torch.no_grad():
        for start in range(0, len(val_idx), BATCH):
            batch_idx = val_idx[start:start + BATCH]
            mel_b, vid_b, y_b = [], [], []
            for i in batch_idx:
                mel, video, y = ds[int(i)]
                mel_b.append(mel); vid_b.append(video); y_b.append(int(y))
            mel_t = torch.stack(mel_b).unsqueeze(1).to(device)
            vid_t = torch.stack(vid_b).to(device)

            # AV penult: forward up to GAP, skip dropout + fc.
            a_mid = av.audio_block1(mel_t)
            v_mid = av.visual(vid_t)
            a_fused = av.gate(a_mid, v_mid)
            av_penult_chunks.append(
                av.gap(av.audio_block2(a_fused)).flatten(1).cpu().numpy())

            # AV penult with v_mid=0 — the calibration-cliff condition for 05.
            v_zero = torch.zeros_like(a_mid)
            a_fused_z = av.gate(a_mid, v_zero)
            av_vzero_chunks.append(
                av.gap(av.audio_block2(a_fused_z)).flatten(1).cpu().numpy())

            # A-only penult: WordResNet has block1/block2/gap directly.
            a_pen_chunks.append(
                a_only.gap(a_only.block2(a_only.block1(mel_t))).flatten(1).cpu().numpy())

            # V-fair penult: visual → block2 → gap.
            v_pen_chunks.append(
                v_fair.gap(v_fair.block2(v_fair.visual(vid_t))).flatten(1).cpu().numpy())

            labels_chunks.append(np.asarray(y_b, dtype=np.int64))
    av_penult = np.concatenate(av_penult_chunks, axis=0)
    av_vzero = np.concatenate(av_vzero_chunks, axis=0)
    a_pen = np.concatenate(a_pen_chunks, axis=0)
    v_pen = np.concatenate(v_pen_chunks, axis=0)
    labels = np.concatenate(labels_chunks, axis=0)

    print(f"\nshapes:")
    print(f"  AV penult        = {av_penult.shape}   (128-d, post-GAP, pre-fc)")
    print(f"  AV penult v=0    = {av_vzero.shape}   (gate sees zero v_mid)")
    print(f"  A-only penult    = {a_pen.shape}   (128-d, no fusion)")
    print(f"  V-fair penult    = {v_pen.shape}   (128-d, capacity-matched)")
    print(f"  labels           = {labels.shape}")

    def _summary(name, arr):
        # Per-feature variance is a quick "is anything alive in this dim".
        var = arr.var(axis=0)
        print(f"  {name:<14}  mean_var={var.mean():.4f}  "
              f"dead_dims(<1e-6)={int((var < 1e-6).sum())}/{arr.shape[1]}")
    print("\nper-feature variance summary:")
    _summary("AV",        av_penult)
    _summary("AV (v=0)",  av_vzero)
    _summary("A-only",    a_pen)
    _summary("V-fair",    v_pen)

    np.savez(os.path.join(OUT_DIR, "04_features.npz"),
              AV=av_penult, AV_vzero=av_vzero,
              A_only=a_pen, V_fair=v_pen,
              labels=labels, val_idx=val_idx)
    print(f"\nSaved features to: {os.path.join(OUT_DIR, '04_features.npz')}")


if __name__ == "__main__":
    main()
