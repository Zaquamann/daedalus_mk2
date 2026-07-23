#!/usr/bin/env python3
"""Audio-only training with Gaussian noise (std=0.3) on the mel during
training only. Same recipe as `train.py` otherwise."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from train import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    RANDOM_SEED,
    TEST_SIZE,
    WEIGHT_DECAY,
    WordDataset,
    WordResNet,
    stratified_split,
)

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "processed", "dataset.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "processed", "model_noisy.pt")

NOISE_STD = 0.3


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    data = torch.load(DATA_PATH, weights_only=False)
    spectrograms = data["spectrograms"]
    labels = data["labels"]
    label_to_idx = data["label_to_idx"]
    idx_to_label = data["idx_to_label"]
    config = data["config"]
    num_classes = len(label_to_idx)

    print(f"Loaded {len(labels)} samples, {num_classes} classes")
    print(f"Spectrogram shape: {spectrograms.shape}")
    print(f"Training with Gaussian noise std={NOISE_STD}")

    # Stratified train/val split
    train_idx, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)

    train_dataset = WordDataset(spectrograms[train_idx], labels[train_idx], augment=True)
    val_dataset = WordDataset(spectrograms[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = WordResNet(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6,
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Gaussian noise augmentation (training only)
            X_batch = X_batch + torch.randn_like(X_batch) * NOISE_STD
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

        # Validate (clean)
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

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} | {val_loss:8.4f} | {val_acc:6.1%}")

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "config": config,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
                "noise_std": NOISE_STD,
            }, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no val loss improvement for {PATIENCE} epochs)")
                break

        scheduler.step()

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val accuracy (clean): {best_val_acc:.1%}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
