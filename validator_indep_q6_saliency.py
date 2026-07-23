#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of Q6 VIDEO-OCCLUSION saliency
(D5_temporal_saliency / D5_spatial_saliency). phase_d_saliency NOT imported —
reimplemented from the AV submodules. Mask the VIDEO INPUT, recompute v_mid =
visual(masked); a_mid from real audio is unchanged.

Temporal (D5.5): 10-frame (200 ms) contiguous windows, step 5, over the 50 strided
frames; AV acc + delta vs baseline. Anchor: frames[20,30) -74.29pp (the max-drop window
= the articulation peak ~400-600 ms).
Spatial (D5.6): 16x16 input patch on a 6x6 grid (h0/w0 = min(i*16, 72)); AV acc + delta.
Anchor: center-lip patch -6.16pp (the max-drop patch).

Load-bearing claim = the ARGMAX window/patch + magnitude: temporal max-drop at frames
[20,30) ~-74pp; spatial max-drop at a center patch ~-6pp. baseline AV 0.956712.

Run on dev-codex:
    python validator_indep_q6_saliency.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q6_saliency.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
WINDOW = 10
N_FRAMES = 50
PATCH = 16
ANCHOR_TEMPORAL_WIN = (20, 30)
ANCHOR_TEMPORAL_PP = -74.29
ANCHOR_SPATIAL_PP = -6.16
BASELINE = 0.956712


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--t-stride", type=int, default=2)
    ap.add_argument("--expect-sha", default=EXPECT_SHA)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    sys.path.insert(0, args.root)
    from model_av import AVWordResNet

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print("[FATAL] val sha mismatch"); sys.exit(2)

    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels_np = dav["spectrograms"]
    mels_np = mels_np.numpy() if hasattr(mels_np, "numpy") else np.asarray(mels_np)
    labels_all = np.asarray(dav["labels"]).astype(np.int64)
    n_all = len(labels_all)
    T_FRAMES, H, W = dav["video_shape"]
    cache_path = dav.get("video_cache_path")
    if not cache_path or not os.path.exists(cache_path):
        cache_path = os.path.join(args.root, "data", "visual", "cache",
                                  dav.get("video_cache_name", "videos_88_100.uint8"))
    videos = np.memmap(cache_path, dtype=np.uint8, mode="r", shape=(n_all, T_FRAMES, H, W))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stride = max(1, int(args.t_stride))

    ck = torch.load(os.path.join(args.root, "models", "av_fused.pt"),
                    weights_only=False, map_location="cpu")
    AV = AVWordResNet(len(ck["label_to_idx"]))
    AV.load_state_dict(ck["model_state_dict"])
    AV = AV.to(device).eval()

    class Vw(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k])
            mel = torch.from_numpy(mels_np[g]).unsqueeze(0)
            v = np.array(videos[g])
            if stride > 1: v = v[::stride]
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, vid, int(labels_all[g])

    dl = DataLoader(Vw(), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    @torch.no_grad()
    def _masked_acc(mask_fn):
        preds, labs = [], []
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True).clone()
            mask_fn(vid)
            a_mid = AV.audio_block1(mel)
            v_mid = AV.visual(vid)
            pen = AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)
            preds.append(AV.fc(pen).argmax(1).cpu().numpy())
            labs.append(y.numpy())
        p = np.concatenate(preds); l = np.concatenate(labs)
        return float((p == l).mean())

    # baseline (no mask) — self-check
    base = _masked_acc(lambda vid: None)
    print(f"[baseline] AV clean = {base:.6f} (anchor {BASELINE})", flush=True)
    sc_ok = abs(base - BASELINE) < 5e-4

    rows = []

    # ---- temporal ----
    print("\n[temporal] 10-frame (200ms) windows, step 5:", flush=True)
    temporal = []
    for t in range(0, N_FRAMES - WINDOW + 1, 5):
        def mf(vid, t=t):
            vid[:, :, t:t + WINDOW, :, :] = 0.0
        acc = _masked_acc(mf)
        delta = (acc - base) * 100.0
        temporal.append((t, t + WINDOW, acc, delta))
        rows.append(["temporal", f"[{t},{t+WINDOW})", f"{acc:.6f}", f"{delta:+.4f}"])
        print(f"    frames[{t:>2d},{t+WINDOW:>2d}) acc={acc:.4f} Δ={delta:+.3f}pp", flush=True)
    t_min = min(temporal, key=lambda r: r[3])  # most negative delta
    win_match = (t_min[0], t_min[1]) == ANCHOR_TEMPORAL_WIN
    t_pp_ok = abs(t_min[3] - ANCHOR_TEMPORAL_PP) <= 1.5
    print(f"  max-drop window = [{t_min[0]},{t_min[1]}) Δ={t_min[3]:+.3f}pp "
          f"(anchor [20,30) {ANCHOR_TEMPORAL_PP}) win_match={win_match} pp_ok={t_pp_ok}",
          flush=True)

    # ---- spatial 6x6 grid, 16x16 patch ----
    print("\n[spatial] 16x16 patch, 6x6 grid:", flush=True)
    grid_h = (H + PATCH - 1) // PATCH
    grid_w = (W + PATCH - 1) // PATCH
    spatial = []
    for ih in range(grid_h):
        for iw in range(grid_w):
            h0 = min(ih * PATCH, H - PATCH); w0 = min(iw * PATCH, W - PATCH)
            def mf(vid, h0=h0, w0=w0):
                vid[:, :, :, h0:h0 + PATCH, w0:w0 + PATCH] = 0.0
            acc = _masked_acc(mf)
            delta = (acc - base) * 100.0
            spatial.append((ih, iw, h0, w0, acc, delta))
            rows.append(["spatial", f"({ih},{iw})", f"{acc:.6f}", f"{delta:+.4f}"])
    s_min = min(spatial, key=lambda r: r[5])
    # center cells of a 6x6 grid: rows/cols 2-3
    is_center = s_min[0] in (2, 3) and s_min[1] in (2, 3)
    s_pp_ok = abs(s_min[5] - ANCHOR_SPATIAL_PP) <= 2.0
    print(f"  max-drop patch = cell({s_min[0]},{s_min[1]}) h0={s_min[2]} w0={s_min[3]} "
          f"Δ={s_min[5]:+.3f}pp (anchor center-lip {ANCHOR_SPATIAL_PP}) "
          f"center={is_center} pp_ok={s_pp_ok}", flush=True)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q6_saliency.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kind", "window_or_cell", "AV_acc", "delta_pp"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)

    all_ok = sc_ok and win_match and t_pp_ok and is_center and s_pp_ok
    print("\n[VERDICT]", flush=True)
    print(f"  baseline self-check ........ {'OK' if sc_ok else 'FAIL'}", flush=True)
    print(f"  temporal max-drop @[20,30) . {'OK' if win_match else 'FLAG'} "
          f"(Δ{t_min[3]:+.2f}pp, pp_ok={t_pp_ok})", flush=True)
    print(f"  spatial max-drop center .... {'OK' if is_center else 'FLAG'} "
          f"(Δ{s_min[5]:+.2f}pp, pp_ok={s_pp_ok})", flush=True)
    if all_ok:
        print("[GO] Q6 occlusion saliency reproduced: temporal peak at the articulation "
              "window [20,30), spatial peak at the center-lip patch.", flush=True)
    else:
        print(f"[NO-GO/FLAG] sc_ok={sc_ok} win_match={win_match} t_pp_ok={t_pp_ok} "
              f"is_center={is_center} s_pp_ok={s_pp_ok} -> report to lead.", flush=True)


if __name__ == "__main__":
    main()
