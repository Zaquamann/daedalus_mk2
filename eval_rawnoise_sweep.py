#!/usr/bin/env python3
"""Inference-time noise sweep on a trained A-only model.

Evaluates a checkpoint on the val partition with σ_a fixed across 8 levels
(`σ_a / audio_rms` ∈ {0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5}) and records
val accuracy. The sweep is deterministic: noise seed is fixed across all
levels so the underlying noise realisation per sample is identical sample-to-
sample, only the magnitude varies.

Usage:
    python eval_rawnoise_sweep.py models/audio_only_filtered.pt
    python eval_rawnoise_sweep.py models/audio_only_rawnoise_filtered.pt

Writes a CSV next to the checkpoint:  <basename>_noise_sweep.csv
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset_raw_noisy import RawNoisyAVDataset
from train import WordResNet


SIGMA_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5)
BATCH_SIZE = 64
NUM_WORKERS = 4
NOISE_SEED = 0


class _FixedSigmaView(Dataset):
    """Wraps RawNoisyAVDataset to inject a FIXED sigma_a / audio_rms.

    Same noise realisation per sample across calls (deterministic per index)
    so the sweep isolates magnitude effects.
    """

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray,
                 sigma_mult: float, seed: int = NOISE_SEED):
        # Force the underlying dataset to NO noise; we'll add it manually.
        assert base.noise is False, "wrap a clean (noise=False) base"
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sigma_mult = float(sigma_mult)
        self.seed = int(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = int(self.indices[k])
        # Re-implement the audio-load + pad path so we can inject deterministic
        # fixed-sigma noise. We bypass the base's __getitem__ to avoid re-doing
        # the mel twice.
        from paired_dataset import _read_wav, _pad_audio, _wav_to_log_mel
        audio = _read_wav(self.base.audio_paths[idx])
        if self.sigma_mult > 0:
            audio_rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
            sigma = self.sigma_mult * audio_rms
            rng = np.random.default_rng(self.seed + idx)
            noise = rng.standard_normal(len(audio)).astype(np.float32) * sigma
            audio = audio + noise
        pad_left = int(self.base.pad_offsets[idx])
        audio_p = _pad_audio(audio, pad_left)
        mel = _wav_to_log_mel(audio_p).astype(np.float32)
        mel_t = torch.from_numpy(mel).unsqueeze(0)             # (1, 80, 99)
        return mel_t, int(self.base.labels[idx])


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(X)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main(ckpt_path: str) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, weights_only=False)
    n_classes = len(ckpt["label_to_idx"])
    val_idx = ckpt["val_idx"]
    sha = ckpt.get("val_idx_sha256")
    print(f"Checkpoint: {ckpt_path}")
    print(f"  best val_acc (training): {ckpt.get('best_val_acc', float('nan')):.4f}")
    print(f"  val_idx sha256: {sha}")
    print(f"  noise_kind: {ckpt.get('noise_kind', '—')}")

    base = RawNoisyAVDataset(noise=False, return_video=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WordResNet(n_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    out_csv = os.path.splitext(ckpt_path)[0] + "_noise_sweep.csv"
    print(f"\n{'σ_a/rms':>8} | {'val_acc':>8}")
    print("-" * 22)

    rows = []
    for sigma in SIGMA_LEVELS:
        ds = _FixedSigmaView(base, val_idx, sigma_mult=sigma)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
        acc = _eval(model, loader, device)
        print(f"{sigma:8.4f} | {acc:8.4%}")
        rows.append((sigma, acc))

    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["sigma_per_rms", "val_acc"])
        for sigma, acc in rows:
            w.writerow([f"{sigma:.4f}", f"{acc:.6f}"])
    print(f"\nSaved sweep to {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python eval_rawnoise_sweep.py <ckpt_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
