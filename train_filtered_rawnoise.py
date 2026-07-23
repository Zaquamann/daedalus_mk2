#!/usr/bin/env python3
"""A-only training with Gaussian noise on the raw audio waveform (mel
recomputed on the fly per epoch). Validation uses σ_a=0 (clean mel)."""

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

from dataset_raw_noisy import RawNoisyAVDataset
from train import WordResNet, stratified_split


# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(SCRIPT_DIR, "processed", "splits.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "audio_only_rawnoise_filtered.pt")
CURVE_PNG = os.path.join(SCRIPT_DIR, "analysis", "audio_only_rawnoise_filtered_curves.png")
CURVE_CSV = os.path.join(SCRIPT_DIR, "analysis", "audio_only_rawnoise_filtered_curves.csv")

BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
TEST_SIZE = 0.33
RANDOM_SEED = 42
NOISE_RANGE = (0.001, 0.05)        # σ_a / audio_rms (debugger-recommended)
NUM_WORKERS = 4                    # smoke-test sweet spot
# Early stopping intentionally disabled — see train_filtered.py for rationale.


class _AudioView(torch.utils.data.Dataset):
    """Subset wrapper over RawNoisyAVDataset (return_video=False).

    Yields `(mel[1, 80, 99], label)` — channel dim already added so this is a
    drop-in for `WordResNet`'s forward.
    """

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        mel, label = self.base[int(self.indices[k])]
        return mel.unsqueeze(0), label


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
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(CURVE_PNG), exist_ok=True)
    fig.savefig(CURVE_PNG, dpi=130)
    plt.close(fig)


def _write_curves_csv(history: dict) -> None:
    os.makedirs(os.path.dirname(CURVE_CSV), exist_ok=True)
    with open(CURVE_CSV, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_s\n")
        for i, (tl, ta, vl, va, et) in enumerate(zip(
            history["train_loss"], history["train_acc"],
            history["val_loss"], history["val_acc"],
            history["epoch_time_s"],
        ), start=1):
            f.write(f"{i},{tl:.6f},{ta:.6f},{vl:.6f},{va:.6f},{et:.3f}\n")


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Datasets — share the same underlying paired manifest, differ in noise flag.
    train_base = RawNoisyAVDataset(noise=True, noise_range=NOISE_RANGE,
                                    return_video=False)
    val_base = RawNoisyAVDataset(noise=False, return_video=False)

    labels = train_base.labels
    label_to_idx = train_base.label_to_idx
    idx_to_label = train_base.idx_to_label
    config = train_base.config
    num_classes = len(label_to_idx)

    print(f"Loaded {len(train_base)} paired samples, {num_classes} classes")
    print(f"Filter: dropped speakers={config['dropped_speakers']}, "
          f"dropped (speaker, group)={config['dropped_speaker_groups']}")
    print(f"Noise range (σ_a / audio_rms) at TRAIN: "
          f"[{NOISE_RANGE[0]:.4f}, {NOISE_RANGE[1]:.4f}] uniform per sample")
    print(f"Noise at VAL: σ_a = 0  (clean mel)")

    # Splits
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

    train_ds = _AudioView(train_base, train_idx)
    val_ds = _AudioView(val_base, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = WordResNet(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_time_s": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    epoch_times: list[float] = []
    peak_gpu_gib = 0.0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} "
          f"| {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>5}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += y_batch.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate (clean)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * y_batch.size(0)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += y_batch.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        epoch_t = time.time() - t0
        epoch_times.append(epoch_t)
        history["epoch_time_s"].append(epoch_t)
        if device.type == "cuda":
            peak_gpu_gib = max(peak_gpu_gib,
                               torch.cuda.max_memory_allocated() / (1024 ** 3))

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} "
              f"| {val_loss:8.4f} | {val_acc:6.1%} | {epoch_t:4.1f}s")

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
                "noise_range": NOISE_RANGE,
                "noise_kind": "raw_audio_gaussian",
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
