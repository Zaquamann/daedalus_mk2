#!/usr/bin/env python3
"""Phase D — saliency + lesion analysis on the AV model (D5.4 MEI, D5.5
temporal saliency, D5.6 spatial saliency, D5.7 GradCAM, D5.8 block2 lesion).
Run: `python phase_d_saliency.py`."""

from __future__ import annotations

import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from analyze_av_msi import (
    BATCH_SIZE, T_STRIDE, _ValAVView, _accuracy, _load_models,
)
from dataset_raw_noisy import RawNoisyAVDataset
from model_av import AVWordResNet


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "deepdive")
os.makedirs(OUT_DIR, exist_ok=True)


# Activation cache: a_mid, v_mid one-pass (D5.4 / D5.8 reuse). Spatial and
# temporal saliency iterate the loader because the perturbation is at the input.

@torch.no_grad()
def _cache_a_v_mid(model: AVWordResNet, loader, device):
    a_mids, v_mids, ys = [], [], []
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        a_mids.append(model.audio_block1(mel).cpu())
        v_mids.append(model.visual(vid).cpu())
        ys.append(y)
    return torch.cat(a_mids, 0), torch.cat(v_mids, 0), torch.cat(ys, 0)


@torch.no_grad()
def _forward_AV_from_cache(model, a_cache, v_cache, labels_cache,
                            device, batch_size: int = 256,
                            block2_mask: torch.Tensor | None = None) -> dict:
    preds, labels = [], []
    n = a_cache.shape[0]
    for i in range(0, n, batch_size):
        a_mid = a_cache[i:i+batch_size].to(device, non_blocking=True)
        v_mid = v_cache[i:i+batch_size].to(device, non_blocking=True)
        a_fused = model.gate(a_mid, v_mid)
        x = model.audio_block2(a_fused)
        if block2_mask is not None:
            x = x * block2_mask.view(1, -1, 1, 1)
        pen = model.gap(x).flatten(1)
        logits = model.fc(model.dropout(pen))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(labels_cache[i:i+batch_size].numpy())
    return dict(preds=np.concatenate(preds),
                labels=np.concatenate(labels))


# D5.4 — per-channel MEI distribution on AV.block2

@torch.no_grad()
def _collect_block2_stats(model: AVWordResNet, loader, device):
    """Mean activation per channel for {AV-full, A-only-via-v_zero,
    V-only-via-audio_zero, zero-zero} — feeds MEI per channel."""

    sums = {"AV": [], "A_via_vzero": [], "V_via_azero": [], "ZZ": []}
    labels = []
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)

        # AV full
        a_mid = model.audio_block1(mel)
        v_mid = model.visual(vid)
        a_fused = model.gate(a_mid, v_mid)
        b2 = model.audio_block2(a_fused).mean(dim=(2, 3))  # (B, 128)
        sums["AV"].append(b2.cpu().numpy())

        # AV with v=0
        v_zero = torch.zeros_like(a_mid)
        a_fused = model.gate(a_mid, v_zero)
        b2 = model.audio_block2(a_fused).mean(dim=(2, 3))
        sums["A_via_vzero"].append(b2.cpu().numpy())

        # AV with audio=0
        mel_zero = torch.zeros_like(mel)
        a_mid_z = model.audio_block1(mel_zero)
        a_fused = model.gate(a_mid_z, v_mid)
        b2 = model.audio_block2(a_fused).mean(dim=(2, 3))
        sums["V_via_azero"].append(b2.cpu().numpy())

        # Both zero
        a_fused = model.gate(a_mid_z, v_zero)
        b2 = model.audio_block2(a_fused).mean(dim=(2, 3))
        sums["ZZ"].append(b2.cpu().numpy())

        labels.append(y.numpy())
    sums = {k: np.concatenate(v) for k, v in sums.items()}
    return sums, np.concatenate(labels)


def D5_4_top_mei_channels(model, loader, device):
    print("\n  D5.4 — Per-channel MEI distribution (AV.block2):")
    sums, _ = _collect_block2_stats(model, loader, device)
    mean_AV = sums["AV"].mean(axis=0)
    mean_A = sums["A_via_vzero"].mean(axis=0)
    mean_V = sums["V_via_azero"].mean(axis=0)
    mean_Z = sums["ZZ"].mean(axis=0)
    # MEI per channel: (AV − max(A, V)) / max(|A|, |V|, ε)
    den = np.maximum(np.abs(mean_A), np.abs(mean_V)) + 1e-6
    mei = (mean_AV - np.maximum(mean_A, mean_V)) / den
    super_add = (mean_AV - (mean_A + mean_V - mean_Z))
    n_ch = len(mean_AV)
    out_csv = os.path.join(OUT_DIR, "D5_top_mei_channels.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["channel", "mean_AV", "mean_A_via_vzero",
                    "mean_V_via_azero", "mean_ZZ", "MEI", "super_additivity"])
        for c in range(n_ch):
            w.writerow([c, f"{mean_AV[c]:.6f}", f"{mean_A[c]:.6f}",
                        f"{mean_V[c]:.6f}", f"{mean_Z[c]:.6f}",
                        f"{mei[c]:.6f}", f"{super_add[c]:.6f}"])
    top = np.argsort(-mei)[:8]
    print(f"  top-8 MEI channels: " +
          ", ".join(f"ch{c}=MEI{mei[c]:+.3f}" for c in top))
    print(f"  median MEI = {np.median(mei):+.3f}")
    print(f"  super-additive (mean_AV > mean_A + mean_V − mean_ZZ): "
          f"{int((super_add > 0).sum())}/{n_ch}")
    print(f"  wrote {out_csv}")


# D5.5 — Temporal saliency: zero-out a contiguous frame window in v_mid.
# Mask the 4D feature map at the spatial-temporal axis after the visual
# encoder, but for clarity & cheapness we mask the input video frames
# directly (model.visual handles the rest).

@torch.no_grad()
def _forward_with_temporal_mask(model, loader, device, a_cache,
                                  t_start: int, t_end: int) -> float:
    preds, labels = [], []
    cursor = 0
    for mel, vid, y in loader:
        bs = mel.shape[0]
        vid = vid.to(device, non_blocking=True).clone()
        vid[:, :, t_start:t_end, :, :] = 0.0
        a_mid = a_cache[cursor:cursor+bs].to(device, non_blocking=True)
        v_mid = model.visual(vid)
        a_fused = model.gate(a_mid, v_mid)
        x = model.audio_block2(a_fused)
        pen = model.gap(x).flatten(1)
        logits = model.fc(model.dropout(pen))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
        cursor += bs
    return _accuracy(np.concatenate(preds), np.concatenate(labels))


def D5_5_temporal_saliency(model, loader, device, a_cache,
                             baseline_acc: float):
    print("\n  D5.5 — Temporal saliency (zero contiguous frame window):")
    window = 10                                              # 10 frames = 200 ms
    n_frames = 50
    out_csv = os.path.join(OUT_DIR, "D5_temporal_saliency.csv")
    rows = []
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["t_start_frame", "t_end_frame",
                    "AV_acc", "delta_pp_from_baseline"])
        for t in range(0, n_frames - window + 1, 5):
            t0 = time.time()
            acc = _forward_with_temporal_mask(
                model, loader, device, a_cache, t, t + window)
            delta = (acc - baseline_acc) * 100.0
            rows.append((t, t + window, acc, delta))
            w.writerow([t, t + window, f"{acc:.6f}", f"{delta:+.4f}"])
            print(f"    frames [{t:>2d}, {t+window:>2d}): "
                  f"acc={acc:.4%} Δ={delta:+.3f} pp "
                  f"({time.time()-t0:.1f}s)")
    print(f"  wrote {out_csv}")

    # PNG
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ts = np.asarray([r[0] for r in rows])
    accs = np.asarray([r[2] for r in rows])
    ms_per_frame = 1000.0 / 50.0  # T=50 frames per 1s clip
    ax.bar(ts * ms_per_frame, baseline_acc - accs,
            width=window * ms_per_frame * 0.9, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("window start (ms)")
    ax.set_ylabel("accuracy drop (baseline − masked)")
    ax.set_title("D5.5 — Temporal saliency of visual stream "
                  "(200 ms zero-window)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "D5_temporal_saliency.png")
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  wrote {out_png}")


# D5.6 — Spatial saliency: zero 8×8 patches on the (88, 88) input.

@torch.no_grad()
def _forward_with_spatial_mask(model, loader, device, a_cache,
                                  h0: int, w0: int, hsize: int, wsize: int):
    """Spatial saliency: zero an (hsize×wsize) input patch, re-run only
    the visual encoder + downstream. audio_block1's output is cached.
    """
    preds, labels = [], []
    cursor = 0
    for mel, vid, y in loader:
        bs = mel.shape[0]
        vid = vid.to(device, non_blocking=True).clone()
        vid[:, :, :, h0:h0+hsize, w0:w0+wsize] = 0.0
        a_mid = a_cache[cursor:cursor+bs].to(device, non_blocking=True)
        v_mid = model.visual(vid)
        a_fused = model.gate(a_mid, v_mid)
        x = model.audio_block2(a_fused)
        pen = model.gap(x).flatten(1)
        logits = model.fc(model.dropout(pen))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
        cursor += bs
    return _accuracy(np.concatenate(preds), np.concatenate(labels))


def D5_6_spatial_saliency(model, loader, device, a_cache,
                            baseline_acc: float):
    """Coarser 5×5 grid of 16×16 patches (= 88/16 ≈ 5)."""
    print("\n  D5.6 — Spatial saliency (16×16 patch mask, 5×5 grid):")
    H = W = 88
    patch = 16
    grid_h = (H + patch - 1) // patch
    grid_w = (W + patch - 1) // patch
    out_csv = os.path.join(OUT_DIR, "D5_spatial_saliency.csv")
    drops = np.zeros((grid_h, grid_w))
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["row_idx", "col_idx", "h0", "w0", "patch",
                    "AV_acc", "delta_pp_from_baseline"])
        for ih in range(grid_h):
            for iw in range(grid_w):
                t0 = time.time()
                acc = _forward_with_spatial_mask(
                    model, loader, device, a_cache,
                    min(ih * patch, H - patch), min(iw * patch, W - patch),
                    patch, patch)
                delta = (acc - baseline_acc) * 100.0
                drops[ih, iw] = -delta
                w.writerow([ih, iw, ih * patch, iw * patch, patch,
                            f"{acc:.6f}", f"{delta:+.4f}"])
                print(f"    cell ({ih},{iw}): acc={acc:.4%} "
                      f"Δ={delta:+.2f} pp ({time.time()-t0:.1f}s)")
    print(f"  wrote {out_csv}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(drops, cmap="hot", aspect="equal", origin="upper",
                    extent=(0, W, H, 0))
    ax.set_title("D5.6 — Spatial saliency (acc drop on 16×16 mask)")
    fig.colorbar(im, ax=ax, label="pp drop")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "D5_spatial_saliency.png")
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  wrote {out_png}")


# D5.7 — Visual GradCAM. Per plan §9.5: skip on zero-gradient.

def D5_7_gradcam_visual(model, loader, device, n_max_per_class: int = 5):
    print("\n  D5.7 — Visual GradCAM (gradient × activation on v_mid):")
    # Probe gradient flow once.
    model.eval()
    test_batch = next(iter(loader))
    mel, vid, y = test_batch
    mel = mel.unsqueeze(1).to(device).requires_grad_(False)
    vid = vid.to(device).requires_grad_(True)
    a_mid = model.audio_block1(mel)
    v_mid = model.visual(vid)
    a_fused = model.gate(a_mid, v_mid)
    x = model.audio_block2(a_fused)
    pen = model.gap(x).flatten(1)
    logits = model.fc(model.dropout(pen))
    # Use logit of the true class.
    target = logits.gather(1, y.to(device).view(-1, 1)).sum()
    target.backward()
    grad_norm = float(vid.grad.abs().mean()) if vid.grad is not None else 0.0
    print(f"    gradient norm on video input = {grad_norm:.3e}")
    if grad_norm < 1e-8:
        print("  [WARN] vanishing gradient on visual input — skipping per plan §9.5.")
        with open(os.path.join(OUT_DIR, "D5_gradcam_visual_SKIPPED.txt"),
                   "w") as f:
            f.write("Skipped: vanishing gradient through gate.sigmoid "
                    f"({grad_norm:.3e} mean abs). See plan §9.5.\n")
        return

    # If gradient flows, aggregate per-class.
    n_classes_seen = set()
    accum = {}     # class → (count, summed grad-cam map (H, W))
    H = W = 88
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device)
        vid = vid.to(device).requires_grad_(True)
        a_mid = model.audio_block1(mel)
        v_mid = model.visual(vid)
        a_fused = model.gate(a_mid, v_mid)
        x = model.audio_block2(a_fused)
        pen = model.gap(x).flatten(1)
        logits = model.fc(model.dropout(pen))
        # Sum logits for the true class over the batch (a single backward).
        target = logits.gather(1, y.to(device).view(-1, 1)).sum()
        model.zero_grad()
        target.backward()
        grad = vid.grad.detach()                         # (B, 1, T, H, W)
        sal = (grad * vid.detach()).abs().mean(dim=(1, 2))  # (B, H, W)
        sal_np = sal.cpu().numpy()
        for cls, smap in zip(y.numpy(), sal_np):
            cls = int(cls)
            if cls not in accum:
                accum[cls] = [0, np.zeros((H, W), dtype=np.float64)]
            if accum[cls][0] < n_max_per_class:
                accum[cls][0] += 1
                accum[cls][1] += smap
            n_classes_seen.add(cls)
    print(f"    aggregated GradCAM over {len(accum)} classes")
    if not accum:
        return

    # Visualize the average over all classes
    global_avg = sum(v[1] / max(1, v[0]) for v in accum.values()) / len(accum)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(global_avg, cmap="hot", aspect="equal")
    ax.set_title("D5.7 — Visual GradCAM (grad×act, mean over classes)")
    fig.colorbar(im, ax=ax, label="|∂logit/∂pixel · pixel|")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "D5_gradcam_visual.png")
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  wrote {out_png}")


# D5.8 — block-2 channel lesion (128 channels of audio_block2.conv2 output)
# Uses cache for 100× speedup.

def D5_8_block2_lesion(model, a_cache, v_cache, labels_cache, device,
                        baseline_acc: float):
    print("\n  D5.8 — Block-2 channel lesion (128 channels, cached):")
    n_ch = 128
    out_csv = os.path.join(OUT_DIR, "D5_block2_lesion.csv")
    impacts = np.zeros(n_ch)
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["channel", "AV_acc_baseline", "AV_acc_lesioned",
                    "delta_pp"])
        for c in range(n_ch):
            mask = torch.ones(n_ch, device=device)
            mask[c] = 0.0
            t0 = time.time()
            out = _forward_AV_from_cache(model, a_cache, v_cache,
                                           labels_cache, device,
                                           block2_mask=mask)
            acc = _accuracy(out["preds"], out["labels"])
            delta = (baseline_acc - acc) * 100.0
            impacts[c] = delta
            w.writerow([c, f"{baseline_acc:.6f}", f"{acc:.6f}",
                        f"{delta:+.4f}"])
            if c % 32 == 0:
                print(f"    [{c:>3d}/{n_ch}] Δ={delta:+.2f} pp "
                      f"({time.time()-t0:.1f}s)")
    print(f"  wrote {out_csv}")
    top = np.argsort(-impacts)[:8]
    print(f"  top-8 block2 channels by impact: " +
          ", ".join(f"ch{c}={impacts[c]:+.2f}pp" for c in top))


# Main

def main() -> None:
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)
    val_idx = splits["val_idx"]
    if hasattr(val_idx, "numpy"):
        val_idx = val_idx.numpy()

    models = _load_models(device)
    AV = models["AV"][0]
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

    # Cache a_mid, v_mid one-shot (reused by D5.8 + temporal/spatial speedup).
    print("Building activation cache...")
    t0 = time.time()
    a_cache, v_cache, labels_cache = _cache_a_v_mid(AV, loader, device)
    print(f"  cached a_mid={tuple(a_cache.shape)}, "
          f"v_mid={tuple(v_cache.shape)} in {time.time()-t0:.1f}s")

    # Baseline (use cache to verify the pipeline).
    out = _forward_AV_from_cache(AV, a_cache, v_cache, labels_cache, device)
    baseline = _accuracy(out["preds"], out["labels"])
    print(f"  AV_clean baseline = {baseline:.4%}")
    assert 0.9560 <= baseline <= 0.9590, f"baseline OOD: {baseline:.4%}"
    print("  [OK] sanity OK")

    D5_4_top_mei_channels(AV, loader, device)
    D5_5_temporal_saliency(AV, loader, device, a_cache, baseline)
    D5_6_spatial_saliency(AV, loader, device, a_cache, baseline)
    D5_7_gradcam_visual(AV, loader, device)
    D5_8_block2_lesion(AV, a_cache, v_cache, labels_cache, device, baseline)

    print("\nPhase D done.")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("D5_") and (f.endswith(".csv") or f.endswith(".png")
                                      or f.endswith(".txt")):
            print(f"  {f}")


if __name__ == "__main__":
    main()
