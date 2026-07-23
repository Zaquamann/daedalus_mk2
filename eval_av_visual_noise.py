#!/usr/bin/env python3
"""Visual-noise robustness sweeps (D1.2 σ_v, D1.3 frame-drop, D1.4 σ_a × σ_v
iso-perf grid) for V-only and AV models. Writes CSVs under
`analysis/deepdive/`. Run: `python eval_av_visual_noise.py`."""

from __future__ import annotations

import csv
import os
import sys
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (
    BATCH_SIZE, T_STRIDE, _accuracy, _forward_AV, _forward_V, _load_models,
)
from analyze_av_deepdive import (
    OUT_DIR, NUM_WORKERS, _NoisyVideoView, _FrameDropView, _NoisyAVView,
)
from dataset_raw_noisy import RawNoisyAVDataset
from model_av import AVWordResNet


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AV_RAW_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused_rawnoise.pt")

SIGMA_V_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
FRAME_DROP_LEVELS = (0, 10, 20, 30, 40, 50)
SIGMA_A_GRID = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)         # 7 levels
SIGMA_V_GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)              # 7 levels


def _loader(view) -> DataLoader:
    return DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def _maybe_load_av_rawnoise(device: torch.device):
    if not os.path.exists(AV_RAW_CKPT):
        return None
    ckpt = torch.load(AV_RAW_CKPT, weights_only=False)
    m = AVWordResNet(len(ckpt["label_to_idx"])).to(device).eval()
    m.load_state_dict(ckpt["model_state_dict"])
    return m


# D1.2 — σ_v sweep

def D1_2_sigma_v(models, val_idx: np.ndarray, base: RawNoisyAVDataset,
                  device: torch.device, av_raw=None) -> None:
    V_model, _ = models["V"]
    AV_model, _ = models["AV"]

    out_csv = os.path.join(OUT_DIR, "D1_sigma_v_sweep.csv")
    print(f"\n  D1.2 — σ_v sweep (per-pixel Gaussian, σ_v / per-clip-pixel-std):")
    header = ["sigma_v_per_pixstd", "V_only_acc", "AV_acc"]
    if av_raw is not None:
        header.append("AV_rawnoise_acc")

    rows = []
    print("    " + "".join(f"{c:>22}" for c in header))
    for sv in SIGMA_V_LEVELS:
        view = _NoisyVideoView(base, val_idx, sigma_mult=sv, seed=0)
        loader = _loader(view)
        v_pred, _, v_lab = _forward_V(V_model, loader, device)
        v_acc = _accuracy(v_pred, v_lab)
        out = _forward_AV(AV_model, loader, device,
                          video_kind="real", audio_kind="real")
        av_acc = _accuracy(out["preds"], out["labels"])
        row = [f"{sv:.4f}", f"{v_acc:.6f}", f"{av_acc:.6f}"]
        if av_raw is not None:
            out_r = _forward_AV(av_raw, loader, device,
                                 video_kind="real", audio_kind="real")
            row.append(f"{_accuracy(out_r['preds'], out_r['labels']):.6f}")
        rows.append(row)
        cells = [f"{sv:>22.4f}", f"{v_acc*100:>21.2f}%", f"{av_acc*100:>21.2f}%"]
        if av_raw is not None:
            cells.append(f"{float(row[-1])*100:>21.2f}%")
        print("    " + "".join(cells))

    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out_csv}")


# D1.3 — Frame-drop sweep

def D1_3_frame_drop(models, val_idx: np.ndarray, base: RawNoisyAVDataset,
                     device: torch.device, av_raw=None) -> None:
    V_model, _ = models["V"]
    AV_model, _ = models["AV"]

    out_csv = os.path.join(OUT_DIR, "D1_frame_drop.csv")
    print(f"\n  D1.3 — frame-drop sweep (zero N of 50 frames at random):")
    header = ["n_frames_dropped", "V_only_acc", "AV_acc"]
    if av_raw is not None:
        header.append("AV_rawnoise_acc")

    rows = []
    print("    " + "".join(f"{c:>22}" for c in header))
    for n in FRAME_DROP_LEVELS:
        view = _FrameDropView(base, val_idx, n_drop=n, seed=0)
        loader = _loader(view)
        v_pred, _, v_lab = _forward_V(V_model, loader, device)
        v_acc = _accuracy(v_pred, v_lab)
        out = _forward_AV(AV_model, loader, device,
                          video_kind="real", audio_kind="real")
        av_acc = _accuracy(out["preds"], out["labels"])
        row = [str(n), f"{v_acc:.6f}", f"{av_acc:.6f}"]
        if av_raw is not None:
            out_r = _forward_AV(av_raw, loader, device,
                                 video_kind="real", audio_kind="real")
            row.append(f"{_accuracy(out_r['preds'], out_r['labels']):.6f}")
        rows.append(row)
        cells = [f"{n:>22d}", f"{v_acc*100:>21.2f}%", f"{av_acc*100:>21.2f}%"]
        if av_raw is not None:
            cells.append(f"{float(row[-1])*100:>21.2f}%")
        print("    " + "".join(cells))

    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out_csv}")


# D1.4 — σ_a × σ_v iso-perf grid

def D1_4_iso_perf_grid(models, val_idx: np.ndarray,
                        base: RawNoisyAVDataset, device: torch.device,
                        av_raw=None) -> None:
    """For AV (and AV-rawnoise if available) — full Cartesian σ_a × σ_v grid."""
    AV_model, _ = models["AV"]

    targets = [("AV_clean", AV_model)]
    if av_raw is not None:
        targets.append(("AV_rawnoise", av_raw))

    out_csv = os.path.join(OUT_DIR, "D1_iso_perf_grid.csv")
    print(f"\n  D1.4 — σ_a × σ_v iso-perf grid "
          f"({len(SIGMA_A_GRID)}×{len(SIGMA_V_GRID)} = "
          f"{len(SIGMA_A_GRID)*len(SIGMA_V_GRID)} cells per model):")

    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "sigma_a_per_rms",
                    "sigma_v_per_pixstd", "AV_acc"])
        for name, m in targets:
            print(f"    {name}:")
            print(f"      σ_a \\ σ_v | " +
                  " | ".join(f"{sv:>6.3f}" for sv in SIGMA_V_GRID))
            for sa in SIGMA_A_GRID:
                row_cells = []
                for sv in SIGMA_V_GRID:
                    view = _NoisyAVView(base, val_idx, sigma_a_mult=sa,
                                        sigma_v_mult=sv, seed=0)
                    loader = _loader(view)
                    out = _forward_AV(m, loader, device,
                                      video_kind="real", audio_kind="real")
                    acc = _accuracy(out["preds"], out["labels"])
                    w.writerow([name, f"{sa:.4f}", f"{sv:.4f}",
                                f"{acc:.6f}"])
                    row_cells.append(f"{acc*100:>6.2f}")
                print(f"      {sa:9.4f} | " + " | ".join(row_cells))
    print(f"  wrote {out_csv}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading models...")
    models = _load_models(device)
    print(f"  V:  {models['V'][1].get('best_val_acc', 0)*100:.2f}%  "
          f"({models['_V_path']})")
    print(f"  AV: {models['AV'][1].get('best_val_acc', 0)*100:.2f}%")
    av_raw = _maybe_load_av_rawnoise(device)
    if av_raw is not None:
        print(f"  AV-rawnoise loaded ({AV_RAW_CKPT})")

    splits = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                        weights_only=False)
    val_idx = splits["val_idx"]
    if hasattr(val_idx, "numpy"):
        val_idx = val_idx.numpy()

    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)

    D1_2_sigma_v(models, val_idx, base, device, av_raw)
    D1_3_frame_drop(models, val_idx, base, device, av_raw)
    D1_4_iso_perf_grid(models, val_idx, base, device, av_raw)

    print("\nDone. Artifacts in analysis/deepdive/:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("D1_") and f.endswith(".csv"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
