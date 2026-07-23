#!/usr/bin/env python3
"""TEMP DEBUG INSTRUMENT (debugger, task #12) — single-variable decomposition of
WHY the late-fusion video HEAD is weak (clean acc 33.7% in-fusion @ ep58, vs the
standalone fair-V specialist 86.6% @ ep185 / ~72% @ ep60).

Three variables differ between the in-fusion late head (33.7%) and the specialist:
  (A) ARCHITECTURE: the specialist video readout is
        VisualEncoder -> ResBlock(64->128) -> GAP -> Linear(128->C)   [VOnlyFairWordResNet]
      the late-fusion video head OMITS the ResBlock:
        VisualEncoder -> GAP -> Linear(64->C)                         [late head]
  (B) REGIME: in-fusion training adds audio-noise aug, modality dropout (12% of
      samples have video zeroed -> head not supervised there), the reliability
      gate, aux_w=0.5, and a SHORT T_max=60 cosine schedule (LR->~0 by ep60).
  (C) EPOCHS: 60 vs 200.

This script isolates (A) and (B) with ONE controlled clean-video run. It trains
BOTH architectures in the SAME epoch loop on BYTE-IDENTICAL batches (same shuffle,
same lip-flip aug per sample), clean video, 60 ep, T_max=60 cosine, seed 0,
AdamW lr1e-3 wd1e-2, dropout 0.3 before the classifier — i.e. EVERYTHING the
late-fusion trainer uses for the video path EXCEPT audio noise / modality dropout
/ gate / aux. The ONLY difference between the two models is the ResBlock head.

Reads off:
  * (A) architecture effect  = M_fair_acc@60 - M_late_acc@60   (single variable: ResBlock)
  * (B) in-fusion regime eff = M_late_acc@60(standalone clean) - 33.7%(in-fusion head)
  * (C) epochs headroom      = 86.6%(specialist @185) - M_fair_acc@60
Self-validation: M_fair must track the committed fair-V curve (~70% by ep60).

Writes ONLY to analysis/deepdive/ (no production model/curve is touched).
Run:  CUDA_VISIBLE_DEVICES=1 python analysis/deepdive/diag_video_head_arch.py
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from dataset_raw_noisy import RawNoisyAVDataset
from model_av import VisualEncoder
from model_v_only_fair import VOnlyFairWordResNet

EPOCHS = int(os.environ.get("EPOCHS", "60"))
SEED = int(os.environ.get("SEED", "0"))
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-2
T_STRIDE = 2
NUM_WORKERS = int(os.environ.get("WORKERS", "12"))
OUT_CSV = os.path.join(HERE, "D312_video_head_arch.csv")
SPLITS = os.path.join(ROOT, "processed", "splits.pt")


class LateHeadVideoOnly(nn.Module):
    """EXACT late-fusion video readout as a standalone classifier:
    VisualEncoder -> AdaptiveAvgPool2d(1) -> Dropout(0.3) -> Linear(64->C).
    Mirrors AVLateFusionReliabilityWordResNet's video path (self.visual,
    self.visual_gap, self.dropout, self.visual_fc) with NO ResBlock head."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.visual = VisualEncoder()
        self.visual_gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.visual_fc = nn.Linear(64, num_classes)

    def forward(self, video):
        v_mid = self.visual(video)                 # (B,64,40,50)
        v_pen = self.visual_gap(v_mid).flatten(1)  # (B,64)
        return self.visual_fc(self.dropout(v_pen))


class _VView(torch.utils.data.Dataset):
    """Clean video + label; lip flip (prob .5) on train — identical to both the
    fair-V trainer and the late-fusion trainer's video augmentation."""

    def __init__(self, base, indices, augment):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = int(self.indices[k])
        _mel, video, label = self.base[idx]
        if self.augment and torch.rand(1).item() < 0.5:
            video = torch.flip(video, dims=[-1])
        return video, label


@torch.no_grad()
def _val_acc(model, loader, device, autocast_kw):
    model.eval()
    correct = total = 0
    for video, y in loader:
        video = video.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(**autocast_kw):
            logits = model(video)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")
    autocast_kw = {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}

    base = RawNoisyAVDataset(t_stride=T_STRIDE, noise=False, return_video=True)
    num_classes = len(base.label_to_idx)
    s = torch.load(SPLITS, weights_only=False)
    train_idx, val_idx = s["train_idx"], s["val_idx"]
    print(f"classes={num_classes}  train={len(train_idx)}  val={len(val_idx)}  "
          f"seed={SEED}  epochs={EPOCHS}  T_max={EPOCHS}  workers={NUM_WORKERS}",
          flush=True)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(_VView(base, train_idx, True), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                              generator=g, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(_VView(base, val_idx, False), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    m_fair = VOnlyFairWordResNet(num_classes).to(device)   # WITH ResBlock head
    m_late = LateHeadVideoOnly(num_classes).to(device)     # NO ResBlock head
    nf = sum(p.numel() for p in m_fair.parameters())
    nl = sum(p.numel() for p in m_late.parameters())
    print(f"M_fair params={nf:,} (with ResBlock64->128)   "
          f"M_late params={nl:,} (no ResBlock)", flush=True)

    # torch.compile: TRAIN-SPEED ONLY (numerically identical eager math); the
    # committed fair-V run used it on this codebase. COMPILE=0 falls back to eager.
    use_compile = os.environ.get("COMPILE", "1") == "1"
    fair_run = torch.compile(m_fair) if use_compile else m_fair
    late_run = torch.compile(m_late) if use_compile else m_late
    print(f"compile={use_compile}", flush=True)

    crit = nn.CrossEntropyLoss()
    opt_f = optim.AdamW(m_fair.parameters(), lr=LR, weight_decay=WD)
    opt_l = optim.AdamW(m_late.parameters(), lr=LR, weight_decay=WD)
    sch_f = optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=EPOCHS, eta_min=1e-6)
    sch_l = optim.lr_scheduler.CosineAnnealingLR(opt_l, T_max=EPOCHS, eta_min=1e-6)

    rows = []
    best_f = best_l = 0.0
    print(f"\n{'Ep':>3} | {'fairTrL':>7} {'fairVA':>6} | {'lateTrL':>7} {'lateVA':>6} "
          f"| {'time':>5}", flush=True)
    print("-" * 56, flush=True)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        fair_run.train(); late_run.train()
        lf = ll = tot = 0.0
        for video, y in train_loader:
            video = video.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # same batch -> both models; separate optimizers (no interference)
            opt_f.zero_grad()
            with torch.autocast(**autocast_kw):
                loss_f = crit(fair_run(video), y)
            loss_f.backward(); opt_f.step()
            opt_l.zero_grad()
            with torch.autocast(**autocast_kw):
                loss_l = crit(late_run(video), y)
            loss_l.backward(); opt_l.step()
            bs = y.size(0)
            lf += loss_f.item() * bs; ll += loss_l.item() * bs; tot += bs
        lf /= tot; ll /= tot
        va_f = _val_acc(fair_run, val_loader, device, autocast_kw)
        va_l = _val_acc(late_run, val_loader, device, autocast_kw)
        best_f = max(best_f, va_f); best_l = max(best_l, va_l)
        sch_f.step(); sch_l.step()
        dt = time.time() - t0
        rows.append((epoch, lf, va_f, ll, va_l, dt))
        print(f"{epoch:3d} | {lf:7.4f} {va_f:6.1%} | {ll:7.4f} {va_l:6.1%} "
              f"| {dt:4.0f}s", flush=True)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "fair_train_loss", "fair_val_acc",
                        "late_train_loss", "late_val_acc", "epoch_time_s"])
            for r in rows:
                w.writerow([r[0]] + [f"{x:.6f}" for x in r[1:]])

    print(f"\n[done] M_fair best={best_f:.4f} (final {rows[-1][2]:.4f})  "
          f"M_late best={best_l:.4f} (final {rows[-1][4]:.4f})", flush=True)
    print(f"[decompose] (A) architecture (ResBlock) = fair-late @ep{EPOCHS} = "
          f"{rows[-1][2]-rows[-1][4]:+.3f}", flush=True)
    print(f"[decompose] (B) in-fusion regime = M_late_standalone - 0.337 = "
          f"{rows[-1][4]-0.337:+.3f}", flush=True)
    print(f"[decompose] (C) epochs headroom (fair) = 0.866 - M_fair@ep{EPOCHS} = "
          f"{0.866-rows[-1][2]:+.3f}", flush=True)
    print(f"[saved] {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
