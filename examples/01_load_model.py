#!/usr/bin/env python3
"""Load each trained checkpoint and print param count, val acc, val-set sha."""

import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_av import AVWordResNet
from model_av_additive import AVAdditiveWordResNet
from model_av_early import AVEarlyFusionWordResNet
from model_av_late import AVLateFusionWordResNet
from model_v_only_fair import VOnlyFairWordResNet
from train import WordResNet


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("A-only (audio_only_filtered)",  "models/audio_only_filtered.pt", WordResNet),
    ("V-only fair (video_only_fair)", "models/video_only_fair.pt",     VOnlyFairWordResNet),
    ("AV-fused (mid-mult)",           "models/av_fused.pt",            AVWordResNet),
    ("AV-additive (D3.10)",           "models/av_fused_additive.pt",   AVAdditiveWordResNet),
    ("AV-late (D3.2)",                "models/av_fused_late.pt",       AVLateFusionWordResNet),
    ("AV-early (D3.1)",               "models/av_fused_early.pt",      AVEarlyFusionWordResNet),
]


def _val_sha_from_ckpt(ckpt: dict) -> str:
    # Most checkpoints record val_idx_sha256 directly; the older A-only one
    # doesn't, so recompute from the saved val_idx if needed.
    if "val_idx_sha256" in ckpt:
        return ckpt["val_idx_sha256"]
    if "val_idx" in ckpt:
        arr = np.asarray(ckpt["val_idx"], dtype=np.int64)
        return hashlib.sha256(arr.tobytes()).hexdigest()
    return "(not recorded)"


def main():
    print(f"{'Model':<32}  {'class':<24}  {'params':>10}  {'val_acc':>8}  {'val sha':<12}")
    print("-" * 96)
    for name, rel_path, cls in MODELS:
        path = os.path.join(ROOT, rel_path)
        if not os.path.exists(path):
            print(f"{name:<32}  {'(missing checkpoint)':<24}  "
                  f"see top-level README quickstart")
            continue

        # weights_only=False: checkpoints store label_to_idx + config alongside
        # state_dict, so we need the unrestricted loader.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(len(ckpt["label_to_idx"]))
        model.load_state_dict(ckpt["model_state_dict"])
        # Trainable params only — running BN buffers don't count.
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        acc = ckpt.get("best_val_acc", float("nan"))
        sha = _val_sha_from_ckpt(ckpt)[:12]
        print(f"{name:<32}  {cls.__name__:<24}  {n_params:>10,}  "
              f"{acc:>7.2%}  {sha}")

    # AV-fused architecture peek — useful when you forget what blocks live
    # inside (audio_block1, visual, gate, audio_block2, gap, dropout, fc).
    ck_path = os.path.join(ROOT, "models/av_fused.pt")
    if os.path.exists(ck_path):
        ckpt = torch.load(ck_path, map_location="cpu", weights_only=False)
        av = AVWordResNet(len(ckpt["label_to_idx"]))
        av.load_state_dict(ckpt["model_state_dict"])
        print("\nAV-fused module tree:")
        for cname, mod in av.named_children():
            n = sum(p.numel() for p in mod.parameters())
            print(f"  {cname:<14} {type(mod).__name__:<22} {n:>8,} params")


if __name__ == "__main__":
    main()
