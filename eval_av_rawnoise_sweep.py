#!/usr/bin/env python3
"""Inference σ_a sweep + modality-dropout for the trained AV checkpoints.

Usage:
    python eval_av_rawnoise_sweep.py models/av_fused.pt
    python eval_av_rawnoise_sweep.py models/av_fused_rawnoise.pt
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (
    BATCH_SIZE, T_STRIDE,
    _NoisyAudioView, _ValAVView, _accuracy, _forward_AV,
)
from dataset_raw_noisy import RawNoisyAVDataset
from model_av import AVWordResNet


SIGMA_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def main(ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, weights_only=False)
    print(f"Checkpoint: {ckpt_path}")
    print(f"  best val_acc: {ckpt.get('best_val_acc', float('nan')):.4%}")
    print(f"  noise_kind:   {ckpt.get('noise_kind', '-')}")
    print(f"  α at best:    {ckpt.get('alpha_at_best', float('nan')):.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    val_idx = ckpt["val_idx"]
    n_classes = len(ckpt["label_to_idx"])
    model = AVWordResNet(n_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # σ_a sweep
    sweep_csv = os.path.splitext(ckpt_path)[0] + "_av_noise_sweep.csv"
    print(f"\n{'σ_a/rms':>8} | {'val_acc':>8}")
    print("-" * 22)
    sweep = []
    for sigma in SIGMA_LEVELS:
        view = _NoisyAudioView(base, val_idx, sigma_mult=sigma, seed=0)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        out = _forward_AV(model, loader, device,
                          video_kind="real", audio_kind="real")
        acc = _accuracy(out["preds"], out["labels"])
        sweep.append((sigma, acc))
        print(f"{sigma:8.4f} | {acc:8.4%}")
    with open(sweep_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["sigma_per_rms", "AV_acc"])
        for s, a in sweep:
            w.writerow([f"{s:.4f}", f"{a:.6f}"])
    print(f"Saved {sweep_csv}")

    # Modality-dropout sanity
    print("\nModality-dropout sanity (clean-input inference):")
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    out_full = _forward_AV(model, loader, device,
                           video_kind="real", audio_kind="real")
    out_zv = _forward_AV(model, loader, device,
                        video_kind="zero", audio_kind="real")
    out_za = _forward_AV(model, loader, device,
                        video_kind="real", audio_kind="zero")
    out_both = _forward_AV(model, loader, device,
                          video_kind="zero", audio_kind="zero")
    rows = [
        ("AV_full", _accuracy(out_full["preds"], out_full["labels"])),
        ("audio_only_video_zero", _accuracy(out_zv["preds"], out_zv["labels"])),
        ("video_only_audio_zero", _accuracy(out_za["preds"], out_za["labels"])),
        ("both_zero", _accuracy(out_both["preds"], out_both["labels"])),
    ]
    for name, acc in rows:
        print(f"  {name:>22s}: {acc:8.4%}")
    drop_csv = os.path.splitext(ckpt_path)[0] + "_modality_dropout.csv"
    with open(drop_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["condition", "AV_acc"])
        for name, acc in rows:
            w.writerow([name, f"{acc:.6f}"])
    print(f"Saved {drop_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python eval_av_rawnoise_sweep.py <ckpt>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
