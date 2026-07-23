#!/usr/bin/env python3
"""Audio-only baseline on the same filtered partition as the AV run. Reuses
`WordResNet` + `WordDataset` from `train.py`; mels come from
`processed/dataset_av.pt` so train/val splits match `train_av.py`."""

from __future__ import annotations

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

from train import WordDataset, WordResNet, stratified_split


# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "processed", "dataset_av.pt")
SPLITS_PATH = os.path.join(SCRIPT_DIR, "processed", "splits.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "audio_only_filtered.pt")
CURVE_PNG = os.path.join(SCRIPT_DIR, "analysis", "audio_only_filtered_curves.png")
CURVE_CSV = os.path.join(SCRIPT_DIR, "analysis", "audio_only_filtered_curves.csv")

BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
TEST_SIZE = 0.33
RANDOM_SEED = 42
# Early stopping intentionally disabled — we always run all 200 epochs and
# take the best val_acc checkpoint, so cosine annealing has time to settle.


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
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for i, (tl, ta, vl, va) in enumerate(zip(
            history["train_loss"], history["train_acc"],
            history["val_loss"], history["val_acc"],
        ), start=1):
            f.write(f"{i},{tl:.6f},{ta:.6f},{vl:.6f},{va:.6f}\n")


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"missing {DATA_PATH}; build it with `python paired_dataset.py`"
        )

    data = torch.load(DATA_PATH, weights_only=False)
    spectrograms = data["spectrograms"]
    labels = data["labels"]
    label_to_idx = data["label_to_idx"]
    idx_to_label = data["idx_to_label"]
    config = data["config"]
    num_classes = len(label_to_idx)

    print(f"Loaded {len(labels)} samples, {num_classes} classes")
    print(f"Spectrogram shape: {tuple(spectrograms.shape)}")
    print(f"Filter: dropped speakers={config['dropped_speakers']}, "
          f"dropped (speaker, group)={config['dropped_speaker_groups']}")

    # Shared train/val partition with the AV experiment.
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

    train_ds = WordDataset(spectrograms[train_idx], labels[train_idx], augment=True)
    val_ds = WordDataset(spectrograms[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

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

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    epoch_times: list[float] = []

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} "
          f"| {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>5}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        epoch_times.append(time.time() - t0)

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} "
              f"| {val_loss:8.4f} | {val_acc:6.1%} | {epoch_times[-1]:4.1f}s")

        # Best by val_acc — early stopping intentionally disabled.
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
            }, MODEL_PATH)
        scheduler.step()

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.1%} (epoch {best_epoch})")
    print(f"Best val loss:     {best_val_loss:.4f}")
    print(f"Mean time/epoch: {np.mean(epoch_times):.1f}s "
          f"(median {np.median(epoch_times):.1f}s)")
    print(f"Saved model to: {MODEL_PATH}")

    _save_curves(history, num_classes, best_val_acc)
    _write_curves_csv(history)
    print(f"Saved curves to: {CURVE_PNG}")
    print(f"Saved curves CSV to: {CURVE_CSV}")


if __name__ == "__main__":
    main()
