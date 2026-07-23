#!/usr/bin/env python3
"""Direction-1 deep-dive driver: 3×3 accuracy matrix (D1.1), cross-trained
sanity (D1.6), heterogeneous-noise comparison (D1.7). Also defines reusable
noise views (`_NoisyVideoView`, `_FrameDropView`, `_NoisyAVView`) imported by
other Phase-B scripts. Outputs in `analysis/deepdive/`."""

from __future__ import annotations

import csv
import os
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analyze_av_msi import (
    BATCH_SIZE, T_STRIDE, _NoisyAudioView, _ValAVView,
    _accuracy, _forward_A, _forward_AV, _forward_V, _load_models,
)
from dataset_raw_noisy import RawNoisyAVDataset
from paired_dataset import _pad_audio, _read_wav, _wav_to_log_mel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "deepdive")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_WORKERS = 4


# Reusable noise views

class _NoisyVideoView(Dataset):
    """Val partition with deterministic per-pixel Gaussian on the video.

    Noise std = `sigma_mult * per_clip_pixel_std`. RNG seeded per-sample so
    repeated forwards on the same view are bit-stable. Audio is untouched.
    """

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray,
                 sigma_mult: float, seed: int = 0):
        assert base.noise is False, "pass a clean base; noise is injected here"
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sigma_mult = float(sigma_mult)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = int(self.indices[k])
        # Clean audio (cached mel via the underlying base).
        mel, v, y = self.base[idx]                          # v in [0, 1], (1, T, 88, 88)
        if self.sigma_mult > 0.0:
            v_np = v.numpy()
            std = float(v_np.std())
            sigma = self.sigma_mult * std
            rng = np.random.default_rng(self.seed + idx)
            noise = rng.standard_normal(v_np.shape).astype(np.float32) * sigma
            v = torch.from_numpy((v_np + noise).astype(np.float32))
        return mel, v, y


class _FrameDropView(Dataset):
    """Val partition with `n_drop` frames zeroed (random per clip, deterministic)."""

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray,
                 n_drop: int, seed: int = 0):
        assert base.noise is False
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.n_drop = int(n_drop)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = int(self.indices[k])
        mel, v, y = self.base[idx]                          # v: (1, T, 88, 88)
        T = v.shape[1]
        if self.n_drop > 0:
            rng = np.random.default_rng(self.seed + idx)
            drop = rng.choice(T, size=min(self.n_drop, T), replace=False)
            v_np = v.numpy().copy()
            v_np[:, drop, :, :] = 0.0
            v = torch.from_numpy(v_np)
        return mel, v, y


class _NoisyAVView(Dataset):
    """Combined audio + video noise (σ_a × σ_v iso-perf grid)."""

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray,
                 sigma_a_mult: float, sigma_v_mult: float, seed: int = 0):
        assert base.noise is False
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sigma_a = float(sigma_a_mult)
        self.sigma_v = float(sigma_v_mult)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = int(self.indices[k])

        # Audio: load raw, optionally inject Gaussian, then mel.
        audio = _read_wav(self.base.audio_paths[idx])
        if self.sigma_a > 0:
            rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
            sigma = self.sigma_a * rms
            rng_a = np.random.default_rng(self.seed + idx)
            noise = rng_a.standard_normal(len(audio)).astype(np.float32) * sigma
            audio = audio + noise
        pad_left = int(self.base.pad_offsets[idx])
        mel = torch.from_numpy(_wav_to_log_mel(_pad_audio(audio, pad_left))
                               .astype(np.float32))

        # Video: load from memmap, optionally inject per-pixel Gaussian.
        v_np = np.array(self.base._videos[idx])
        if self.base.t_stride > 1:
            v_np = v_np[:: self.base.t_stride]
        v_np = v_np.astype(np.float32)[np.newaxis, ...] / 255.0  # (1, T, 88, 88)
        if self.sigma_v > 0:
            std = float(v_np.std())
            sigma = self.sigma_v * std
            rng_v = np.random.default_rng(self.seed + idx + 10_000_000)
            v_np = v_np + (rng_v.standard_normal(v_np.shape).astype(np.float32)
                            * sigma)
        v = torch.from_numpy(v_np)
        return mel, v, int(self.base.labels[idx])


# Loader helpers

def _loader(view) -> DataLoader:
    return DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


# D1.1 — Full 3×3 accuracy matrix

def D1_1_matrix_3x3(models, val_idx: np.ndarray, base: RawNoisyAVDataset,
                    device: torch.device) -> dict:
    """Emit `D1_3x3_clean.csv` plus return the dict for downstream use."""
    A_model, _ = models["A"]
    V_model, _ = models["V"]
    AV_model, _ = models["AV"]

    view = _ValAVView(base, val_idx)
    loader = _loader(view)

    # A-only model — by architecture takes only audio.
    a_preds, _, a_labels = _forward_A(A_model, loader, device)
    a_clean = _accuracy(a_preds, a_labels)

    # V-only model — by architecture takes only video.
    v_preds, _, v_labels = _forward_V(V_model, loader, device)
    v_clean = _accuracy(v_preds, v_labels)

    # AV-trained model — three input conditions.
    out_full = _forward_AV(AV_model, loader, device,
                            video_kind="real", audio_kind="real")
    out_a_only = _forward_AV(AV_model, loader, device,
                              video_kind="zero", audio_kind="real")
    out_v_only = _forward_AV(AV_model, loader, device,
                              video_kind="real", audio_kind="zero")
    av_full = _accuracy(out_full["preds"], out_full["labels"])
    av_a_only = _accuracy(out_a_only["preds"], out_a_only["labels"])
    av_v_only = _accuracy(out_v_only["preds"], out_v_only["labels"])

    # Layout: rows = trained network, cols = input condition.
    NA_A = "n/a (no V input)"
    NA_V = "n/a (no A input)"
    rows = [
        ("A_trained",  f"{a_clean:.6f}",  NA_A,             NA_A),
        ("V_trained",  NA_V,              f"{v_clean:.6f}", NA_V),
        ("AV_trained", f"{av_a_only:.6f}", f"{av_v_only:.6f}", f"{av_full:.6f}"),
    ]
    out_csv = os.path.join(OUT_DIR, "D1_3x3_clean.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["trained_model", "input_A_only",
                    "input_V_only", "input_AV"])
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out_csv}")

    # Pretty-print
    print("\n  D1.1 — 3×3 accuracy matrix (clean):")
    print(f"  {'':<14}{'A-only':>22}{'V-only':>22}{'AV':>22}")
    for name, c1, c2, c3 in rows:
        f1 = c1 if c1.startswith("n/a") else f"{float(c1)*100:>21.2f}%"
        f2 = c2 if c2.startswith("n/a") else f"{float(c2)*100:>21.2f}%"
        f3 = c3 if c3.startswith("n/a") else f"{float(c3)*100:>21.2f}%"
        print(f"  {name:<14}{f1}{f2}{f3}")

    return dict(
        a_clean=a_clean, v_clean=v_clean,
        av_full=av_full, av_audio_only=av_a_only, av_video_only=av_v_only,
        a_labels=a_labels, a_preds=a_preds,
        v_labels=v_labels, v_preds=v_preds,
        av_preds=out_full["preds"], av_labels=out_full["labels"],
    )


# D1.6 — Cross-trained sanity: no leakage

def D1_6_cross_sanity(d11: dict) -> None:
    """Verify A-only & V-only reproduce their checkpoint accuracies on the
    val partition (i.e. the AV pairing didn't shuffle labels). Pure audit;
    cross-feeding an A-only model with video / V-only with audio is
    architecturally impossible (those models don't accept the other tensor).
    """
    out_csv = os.path.join(OUT_DIR, "D1_cross_trained_sanity.csv")
    a_match = (d11["a_preds"] == d11["a_labels"]).mean()
    v_match = (d11["v_preds"] == d11["v_labels"]).mean()
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "feed",
                    "obtained_acc", "checkpoint_recorded",
                    "delta_pp", "note"])
        w.writerow([
            "A_only", "val_audio",
            f"{a_match:.6f}", "0.927000",
            f"{(a_match - 0.927)*100:.4f}",
            "architectural: A-only model has no V input",
        ])
        w.writerow([
            "V_only_fair", "val_video",
            f"{v_match:.6f}", "0.865600",
            f"{(v_match - 0.8656)*100:.4f}",
            "architectural: V-only model has no A input",
        ])
    print(f"  wrote {out_csv}")
    print(f"  A_only on val audio: {a_match*100:.2f}% (ckpt 92.70%)")
    print(f"  V_fair on val video: {v_match*100:.2f}% (ckpt 86.56%)")


# D1.7 — Heterogeneous noise on AV

# Symmetric pair (σ_a, σ_v) — picked so each side is at "comparable
# degradation" in its own units. Used together with D1.4's grid (which
# already covers the full Cartesian product). σ_a is σ/audio_rms; σ_v is
# σ/per-clip-pixel-std.
SYMMETRIC_PAIRS = (
    (0.000, 0.00),
    (0.005, 0.05),
    (0.010, 0.10),
    (0.050, 0.20),
    (0.100, 0.40),
    (0.200, 0.80),
)


def D1_7_heterogeneous(models, val_idx: np.ndarray, base: RawNoisyAVDataset,
                       device: torch.device) -> None:
    """AV under σ_a-alone, σ_v-alone, both symmetric.

    The σ_a-alone curve is re-evaluated here in-script (small extra cost,
    keeps the script self-contained); σ_v-alone and symmetric are evaluated
    here too. After D1.4 finishes its full grid, post-processing produces a
    super-set; this script suffices on its own.
    """
    AV_model, _ = models["AV"]

    a_levels = (0.0, 0.005, 0.01, 0.05, 0.1, 0.2)
    v_levels = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)

    print("\n  D1.7 — heterogeneous noise on AV:")
    out_csv = os.path.join(OUT_DIR, "D1_heterogeneous_noise.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["regime", "sigma_a_per_rms",
                    "sigma_v_per_pixstd", "AV_acc"])

        # σ_a alone (σ_v = 0)
        print(f"    {'σ_a-only':>10} | {'σ_a':>6} | {'val_acc':>8}")
        for sa in a_levels:
            view = _NoisyAVView(base, val_idx, sigma_a_mult=sa,
                                sigma_v_mult=0.0, seed=0)
            out = _forward_AV(AV_model, _loader(view), device,
                              video_kind="real", audio_kind="real")
            acc = _accuracy(out["preds"], out["labels"])
            print(f"    {'σ_a-only':>10} | {sa:6.4f} | {acc:8.4%}")
            w.writerow(["sigma_a_only", f"{sa:.4f}", "0.0000",
                         f"{acc:.6f}"])

        # σ_v alone (σ_a = 0)
        print(f"    {'σ_v-only':>10} | {'σ_v':>6} | {'val_acc':>8}")
        for sv in v_levels:
            view = _NoisyAVView(base, val_idx, sigma_a_mult=0.0,
                                sigma_v_mult=sv, seed=0)
            out = _forward_AV(AV_model, _loader(view), device,
                              video_kind="real", audio_kind="real")
            acc = _accuracy(out["preds"], out["labels"])
            print(f"    {'σ_v-only':>10} | {sv:6.4f} | {acc:8.4%}")
            w.writerow(["sigma_v_only", "0.0000", f"{sv:.4f}",
                         f"{acc:.6f}"])

        # Symmetric pairs
        print(f"    {'symmetric':>10} | {'σ_a':>6} | {'σ_v':>6} | {'val_acc':>8}")
        for sa, sv in SYMMETRIC_PAIRS:
            view = _NoisyAVView(base, val_idx, sigma_a_mult=sa,
                                sigma_v_mult=sv, seed=0)
            out = _forward_AV(AV_model, _loader(view), device,
                              video_kind="real", audio_kind="real")
            acc = _accuracy(out["preds"], out["labels"])
            print(f"    {'symmetric':>10} | {sa:6.4f} | {sv:6.4f} | {acc:8.4%}")
            w.writerow(["both_symmetric", f"{sa:.4f}", f"{sv:.4f}",
                         f"{acc:.6f}"])
    print(f"  wrote {out_csv}")


# Main

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading models...")
    models = _load_models(device)
    print(f"  A:  {models.get('A')[1].get('best_val_acc', 0)*100:.2f}%")
    print(f"  V:  {models.get('V')[1].get('best_val_acc', 0)*100:.2f}%  "
          f"({models.get('_V_path')})")
    print(f"  AV: {models.get('AV')[1].get('best_val_acc', 0)*100:.2f}%")

    # Shared val partition
    splits = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                        weights_only=False)
    val_idx = splits["val_idx"]
    if hasattr(val_idx, "numpy"):
        val_idx = val_idx.numpy()
    print(f"  N val: {len(val_idx)}")

    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)

    print("\nD1.1 — 3×3 accuracy matrix (clean)...")
    d11 = D1_1_matrix_3x3(models, val_idx, base, device)

    print("\nD1.6 — Cross-trained sanity...")
    D1_6_cross_sanity(d11)

    print("\nD1.7 — Heterogeneous noise on AV...")
    D1_7_heterogeneous(models, val_idx, base, device)

    print("\nDone. Artifacts in analysis/deepdive/:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("D1_") and f.endswith(".csv"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
