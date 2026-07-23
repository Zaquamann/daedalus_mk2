#!/usr/bin/env python3
"""Train the MATCHED FEEDFORWARD baseline for Q14 (recurrent vs feedforward control).

This is the single-variable control for the Q14 recurrence comparison. It is a BYTE-FOR-
BYTE mirror of train_av_recurrent.py's training recipe (PairedAVDataset, shared
processed/splits.pt, random init / no audio-pretrain, bf16 autocast, AdamW + cosine
T_max=200, 200 epochs, best-val checkpoint, per-epoch CSV flush) EXCEPT it trains the
plain AVWordResNet — i.e. the recurrent model with the temporal GRU REMOVED. The ONLY
architectural difference from av_fused_recurrent.pt is the GRU-over-v_mid; seed (0),
schedule, epochs, batch, LR, weight-decay, augmentation, dataset, and val split
(sha 03c5a87a) are identical.

WHY this exists: the canonical av_fused.pt was trained at seed=42 WITH torch.compile, so
recurrent(seed0,eager) - av_fused.pt confounds recurrence with seed + compile-path. This
control is seed0 + --no-compile, so recurrent - this = the PURE recurrence effect
(single-variable), and this - av_fused.pt = run-to-run variance (same architecture, a
different draw) which calibrates how large a clean-val delta must be to be meaningful.

RNG-PARITY TRICK (to match data order, not just the recipe): the recurrent script
constructs a throwaway AVWordResNet inside its param-match report (consuming D_ff RNG
draws); this script symmetrically constructs a throwaway AVRecurrentWordResNet (D_rec
draws) in its param-match report. Both runs therefore consume D_rec + D_ff draws from the
seed-0 MT19937 stream before the first DataLoader iteration, so the per-epoch shuffle and
the (single, identically-shaped) dropout draws SHOULD coincide epoch-for-epoch. The
audio path / visual encoder / fc init bit-identically (constructed before the GRU in
both). IRREDUCIBLE residual: the gate's Wa/Wv/alpha init differs (the GRU's param-init
draws shift where the gate's draws fall in the stream) — a ~8k-param, second-order
difference inherent to comparing two different-sized nets. (Parity is by-construction,
not verified against the already-running recurrent process — the this-vs-av_fused.pt
calibration bounds any residual.)

  --no-compile is REQUIRED on the pod (no Python.h -> torch.compile unavailable) and is
  also the right choice here (matches the recurrent run's eager training path).

Outputs: models/av_fused_ff_baseline_seed0<tag>.pt + analysis/deepdive/D3_ff_baseline<tag>_log.csv
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train import spec_augment, stratified_split
from paired_dataset import PairedAVDataset
from model_av import AVWordResNet
from model_av_recurrent import AVRecurrentWordResNet


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(SCRIPT_DIR, "processed", "splits.pt")

# Recipe constants — IDENTICAL to train_av_recurrent.py.
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
TEST_SIZE = 0.33
ALPHA_INIT = 0.2
T_STRIDE = 2
USE_BF16 = True
USE_COMPILE = True
NUM_WORKERS = 4
VAL_SHA_PIN = "03c5a87acdcf07ad"
GRU_HIDDEN_MATCH = 64   # the gru_hidden the recurrent run used (for RNG-parity throwaway)


class _AVAugmentedView(torch.utils.data.Dataset):
    """Verbatim from train_av_recurrent.py (same augmentation -> same aug draws)."""

    def __init__(self, base, indices, augment):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = int(self.indices[k])
        mel, video, label = self.base[idx]
        mel = mel.unsqueeze(0)                       # (1, 80, 99)
        if self.augment:
            mel = spec_augment(mel)
            if torch.rand(1).item() < 0.5:
                video = torch.flip(video, dims=[-1])
        return mel, video, label


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=NUM_EPOCHS,
                    help="stop after this many epochs (LR schedule still T_max=200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-tag", type=str, default="",
                    help="suffix for ckpt/log paths, e.g. _precheck")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    use_compile = USE_COMPILE and not args.no_compile
    model_path = os.path.join(SCRIPT_DIR, "models",
                              f"av_fused_ff_baseline_seed{args.seed}{args.out_tag}.pt")
    curve_csv = os.path.join(SCRIPT_DIR, "analysis", "deepdive",
                             f"D3_ff_baseline{args.out_tag}_log.csv")
    os.makedirs(os.path.dirname(curve_csv), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base = PairedAVDataset(t_stride=T_STRIDE)
    labels = base.labels
    label_to_idx = base.label_to_idx
    idx_to_label = base.idx_to_label
    config = base.config
    num_classes = len(label_to_idx)
    print(f"Loaded {len(base)} paired samples, {num_classes} classes", flush=True)

    if os.path.exists(SPLITS_PATH):
        s = torch.load(SPLITS_PATH, weights_only=False)
        train_idx, val_idx = s["train_idx"], s["val_idx"]
        print(f"Loaded shared splits from {SPLITS_PATH}", flush=True)
    else:
        train_idx, val_idx = stratified_split(labels, TEST_SIZE, args.seed)

    train_hash = _hash_idx(train_idx)
    val_hash = _hash_idx(val_idx)
    print(f"train_idx sha256: {train_hash}", flush=True)
    print(f"val_idx   sha256: {val_hash}", flush=True)
    assert val_hash.startswith(VAL_SHA_PIN), (
        f"VAL PIN MISMATCH: got {val_hash[:16]}, expected {VAL_SHA_PIN}")
    assert len(val_idx) == 5244, f"val N={len(val_idx)} != 5244"
    print(f"VAL PIN OK ({VAL_SHA_PIN}, N={len(val_idx)})", flush=True)

    train_ds = _AVAugmentedView(base, train_idx, augment=True)
    val_ds = _AVAugmentedView(base, val_idx, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # The matched control = recurrent model with the GRU removed.
    model = AVWordResNet(num_classes, alpha_init=ALPHA_INIT).to(device)

    # Param-match report vs the RECURRENT model. Constructing AVRecurrentWordResNet here
    # is also the RNG-PARITY trick (mirror of the recurrent script's throwaway AVWordResNet):
    # it consumes D_rec draws so both runs reach the same MT19937 state before training.
    n_total = sum(p.numel() for p in model.parameters())
    rec_total = sum(p.numel() for p in
                    AVRecurrentWordResNet(num_classes, gru_hidden=GRU_HIDDEN_MATCH).parameters())
    ratio = n_total / rec_total
    print(f"PARAMS feedforward(matched)={n_total:,} | recurrent={rec_total:,} "
          f"| ratio={ratio:.4f} | seed={args.seed} | arch=AVWordResNet (NO GRU)", flush=True)
    print(f"bf16={USE_BF16} compile={use_compile} max_epochs={args.max_epochs} "
          f"T_max(sched)={NUM_EPOCHS}", flush=True)

    autocast_kw = {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": USE_BF16}
    compiled = torch.compile(model, mode="default") if use_compile else model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc",
                            "epoch_time_s", "peak_gpu_gib", "alpha")}
    best_val_acc, best_val_loss, best_epoch = 0.0, float("inf"), 0

    def _flush_csv():
        with open(curve_csv, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_s,"
                    "peak_gpu_gib,alpha\n")
            for i in range(len(hist["train_loss"])):
                f.write(f"{i+1},{hist['train_loss'][i]:.6f},{hist['train_acc'][i]:.6f},"
                        f"{hist['val_loss'][i]:.6f},{hist['val_acc'][i]:.6f},"
                        f"{hist['epoch_time_s'][i]:.3f},{hist['peak_gpu_gib'][i]:.3f},"
                        f"{hist['alpha'][i]:.6f}\n")

    print(f"\n{'Ep':>3} | {'trL':>7} {'trA':>6} | {'vaL':>7} {'vaA':>6} "
          f"| {'a':>5} | {'t':>5} {'G':>5}", flush=True)
    print("-" * 60, flush=True)

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model.train()
        tr_correct = tr_total = 0
        tr_loss = 0.0
        for mel, video, y in train_loader:
            mel = mel.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(**autocast_kw):
                logits = compiled(mel, video)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * y.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.size(0)
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        model.eval()
        va_correct = va_total = 0
        va_loss = 0.0
        with torch.no_grad():
            for mel, video, y in val_loader:
                mel = mel.to(device, non_blocking=True)
                video = video.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(**autocast_kw):
                    logits = compiled(mel, video)
                    loss = criterion(logits, y)
                va_loss += loss.item() * y.size(0)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_total += y.size(0)
        va_loss /= va_total
        va_acc = va_correct / va_total

        epoch_t = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                if device.type == "cuda" else 0.0)
        alpha = float(model.gate.alpha.detach())

        for k, v in (("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("val_loss", va_loss), ("val_acc", va_acc),
                     ("epoch_time_s", epoch_t), ("peak_gpu_gib", peak),
                     ("alpha", alpha)):
            hist[k].append(v)

        print(f"{epoch:3d} | {tr_loss:7.4f} {tr_acc:6.1%} | {va_loss:7.4f} {va_acc:6.1%} "
              f"| {alpha:5.2f} | {epoch_t:4.0f}s {peak:4.1f}G", flush=True)
        _flush_csv()

        if va_acc > best_val_acc:
            best_val_acc, best_val_loss, best_epoch = va_acc, va_loss, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx, "idx_to_label": idx_to_label,
                "config": config, "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss, "epoch": best_epoch,
                "train_idx": train_idx, "val_idx": val_idx,
                "train_idx_sha256": train_hash, "val_idx_sha256": val_hash,
                "alpha_init": ALPHA_INIT, "alpha_at_best": alpha,
                "arch": "av_fused_feedforward_matched_seed0", "seed": args.seed,
            }, model_path)
        scheduler.step()

    print(f"\nDone. best val_acc={best_val_acc:.4f} @ep{best_epoch} | "
          f"saved {model_path}", flush=True)


if __name__ == "__main__":
    main()
