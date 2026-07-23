#!/usr/bin/env python3
"""Train the late-fusion AV variant (D3.2). Same training recipe as
`train_av.py`; the only varied factor is the architecture (concat at the
penult instead of mid-block gate)."""

from __future__ import annotations

import hashlib
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train import spec_augment, stratified_split
from paired_dataset import PairedAVDataset
from model_av_late import AVLateFusionWordResNet


# Config (mirrors train_av.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(SCRIPT_DIR, "processed", "splits.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "av_fused_late.pt")
CURVE_PNG = os.path.join(SCRIPT_DIR, "analysis", "deepdive", "D3_retrain_late_curves.png")
CURVE_CSV = os.path.join(SCRIPT_DIR, "analysis", "deepdive", "D3_retrain_late_log.csv")

BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
TEST_SIZE = 0.33
RANDOM_SEED = 42
T_STRIDE = 2
USE_BF16 = True
USE_COMPILE = True
NUM_WORKERS = 4


# AV dataset wrapper (same as train_av.py)
class _AVAugmentedView(torch.utils.data.Dataset):

    def __init__(self, base: PairedAVDataset, indices: np.ndarray,
                 augment: bool):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = int(self.indices[k])
        mel, video, label = self.base[idx]
        mel = mel.unsqueeze(0)
        if self.augment:
            mel = spec_augment(mel)
            if torch.rand(1).item() < 0.5:
                video = torch.flip(video, dims=[-1])
        return mel, video, label


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _save_curves(history: dict, num_classes: int, best_val_acc: float) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="train")
    ax.plot(epochs, history["val_acc"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy (best val: {best_val_acc:.1%}, {num_classes} classes)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(CURVE_PNG), exist_ok=True)
    fig.savefig(CURVE_PNG, dpi=130)
    plt.close(fig)


def _write_curves_csv(history: dict) -> None:
    os.makedirs(os.path.dirname(CURVE_CSV), exist_ok=True)
    with open(CURVE_CSV, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,"
                 "epoch_time_s,peak_gpu_gib\n")
        n = len(history["train_loss"])
        for i in range(n):
            f.write(
                f"{i+1},{history['train_loss'][i]:.6f},"
                f"{history['train_acc'][i]:.6f},"
                f"{history['val_loss'][i]:.6f},"
                f"{history['val_acc'][i]:.6f},"
                f"{history['epoch_time_s'][i]:.3f},"
                f"{history['peak_gpu_gib'][i]:.3f}\n"
            )


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    base = PairedAVDataset(t_stride=T_STRIDE)
    labels = base.labels
    label_to_idx = base.label_to_idx
    idx_to_label = base.idx_to_label
    config = base.config
    num_classes = len(label_to_idx)

    print(f"Loaded {len(base)} paired samples, {num_classes} classes")

    # Shared split
    if os.path.exists(SPLITS_PATH):
        s = torch.load(SPLITS_PATH, weights_only=False)
        train_idx, val_idx = s["train_idx"], s["val_idx"]
        print(f"Loaded shared splits from {SPLITS_PATH}")
    else:
        train_idx, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)
        os.makedirs(os.path.dirname(SPLITS_PATH), exist_ok=True)
        torch.save({
            "train_idx": train_idx, "val_idx": val_idx,
            "random_seed": RANDOM_SEED, "test_size": TEST_SIZE,
            "dataset_path": "processed/dataset_av.pt",
            "n_samples": len(train_idx) + len(val_idx),
        }, SPLITS_PATH)
        print(f"Wrote shared splits to {SPLITS_PATH}")

    train_hash = _hash_idx(train_idx)
    val_hash = _hash_idx(val_idx)
    print(f"train_idx sha256: {train_hash}")
    print(f"val_idx   sha256: {val_hash}")
    # Defensive: val sha must match the canonical Tier-0 hash.
    assert val_hash.startswith("03c5a87a"), \
        f"val_idx hash drift: {val_hash[:16]} != 03c5a87a…"

    train_ds = _AVAugmentedView(base, train_idx, augment=True)
    val_ds = _AVAugmentedView(base, val_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = AVLateFusionWordResNet(num_classes).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_audio = (sum(p.numel() for p in model.audio_block1.parameters())
                + sum(p.numel() for p in model.audio_block2.parameters()))
    n_visual = sum(p.numel() for p in model.visual.parameters())
    n_fc = sum(p.numel() for p in model.fc.parameters())
    print(f"Model parameters: total={n_total:,}, "
          f"audio_branch={n_audio:,}, visual_branch={n_visual:,}, "
          f"fc={n_fc:,}")
    print(f"video t_stride: {T_STRIDE} (effective fps = {100 // T_STRIDE})")
    print(f"bf16 autocast: {USE_BF16}")
    print(f"torch.compile: {USE_COMPILE}")
    autocast_kw = {"device_type": "cuda", "dtype": torch.bfloat16,
                    "enabled": USE_BF16}

    compiled = torch.compile(model, mode="default") if USE_COMPILE else model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6,
    )

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [],
               "epoch_time_s": [], "peak_gpu_gib": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    epoch_times: list[float] = []
    peak_gpu_gib = 0.0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} "
          f"| {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>5} | {'GPU':>5}")
    print("-" * 76)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
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
            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for mel, video, y in val_loader:
                mel = mel.to(device, non_blocking=True)
                video = video.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(**autocast_kw):
                    logits = compiled(mel, video)
                    loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_t = time.time() - t0
        epoch_times.append(epoch_t)
        epoch_peak = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                      if device.type == "cuda" else 0.0)
        peak_gpu_gib = max(peak_gpu_gib, epoch_peak)
        history["epoch_time_s"].append(epoch_t)
        history["peak_gpu_gib"].append(epoch_peak)

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} "
              f"| {val_loss:8.4f} | {val_acc:6.1%} "
              f"| {epoch_t:4.1f}s | {epoch_peak:4.1f}G")

        # Best-by-val_acc snapshot
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "config": config,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "epoch": best_epoch,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_idx_sha256": train_hash,
                "val_idx_sha256": val_hash,
                "architecture": "late_fusion_d3_2",
            }, MODEL_PATH)
        scheduler.step()

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.1%} (epoch {best_epoch})")
    print(f"Best val loss:     {best_val_loss:.4f}")
    print(f"Mean time/epoch: {np.mean(epoch_times):.1f}s "
          f"(median {np.median(epoch_times):.1f}s)")
    print(f"Peak GPU: {peak_gpu_gib:.2f} GiB")
    print(f"Saved model to: {MODEL_PATH}")

    _save_curves(history, num_classes, best_val_acc)
    _write_curves_csv(history)
    print(f"Saved curves to: {CURVE_PNG}")
    print(f"Saved curves CSV to: {CURVE_CSV}")


if __name__ == "__main__":
    main()
