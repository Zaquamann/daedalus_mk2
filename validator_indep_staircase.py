#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q6/Q8 AV word-decodability
staircase (D5_layer_decodability_word.csv, AV_full rows):

    a_mid_gap 0.2796 -> v_mid_gap 0.4678 -> gate_out_gap 0.7599 -> block2_gap 0.9432

Method (independent of phase_a_deepdive.py / phase_f_flow.py — neither imported):
  * Rebuild the four AV GAP activation sites directly from the trained submodules
    over the pinned val set, in val_idx order (matching phase_a's shuffle=False
    loader so the StratifiedKFold fold assignment is identical):
        a_mid     = audio_block1(mel)              -> a_mid_gap     = mean over (H,W)
        v_mid     = visual(vid)                    -> v_mid_gap
        a_fused   = a_mid*(1+alpha*sigmoid(Wa a_mid + Wv v_mid))  -> gate_out_gap
        b2        = audio_block2(a_fused)          -> block2_gap
  * Self-check: fc(gap(b2)) argmax accuracy must reproduce AV fp32 clean-val
    0.956712 (guards the activation path, same role as the lesion baseline).
  * Re-run phase_f's exact 5-fold word probe: StratifiedKFold(5, shuffle=True,
    random_state=0); per-fold StandardScaler; LogisticRegression(max_iter=1500,
    C=1.0); mean top-1 accuracy over folds.

fp32, no autocast.

Run on dev-codex:
    python validator_indep_staircase.py --root /scratch/daedelus \
        --ckpt /scratch/daedelus/models/av_fused.pt \
        --out  /scratch/daedelus/analysis/validator_indep_staircase_av_fused.csv
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
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except ModuleNotFoundError:
    HAVE_SKLEARN = False

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
BASELINE_FP32_REF = 0.956712
# AV_full word-decodability staircase the report cites (rounded to 0.280/0.468/0.760/0.943)
REPORT = {"a_mid_gap": 0.279558, "v_mid_gap": 0.467771,
          "gate_out_gap": 0.759915, "block2_gap": 0.943173}


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _probe_5fold(X, y, max_iter: int = 1500, C: float = 1.0, seed: int = 0) -> float:
    """Mirror phase_f_flow._probe_5fold: z-score per fold, LR, mean top-1 acc."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        clf.fit(X_tr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X_te)))
    return float(np.mean(accs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--t-stride", type=int, default=2)
    ap.add_argument("--video-cache", default=None)
    ap.add_argument("--expect-sha", default=EXPECT_SHA)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    sys.path.insert(0, args.root)
    from model_av import AVWordResNet  # noqa: E402

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}")
    if args.expect_sha and val_sha != args.expect_sha:
        print(f"[FATAL] val sha {val_sha} != expected {args.expect_sha}; STOP.")
        sys.exit(2)

    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels = dav["spectrograms"]
    mels_np = mels.numpy() if hasattr(mels, "numpy") else np.asarray(mels)
    labels_all = np.asarray(dav["labels"]).astype(np.int64)
    n_all = len(labels_all)
    T_FRAMES, H, W = dav["video_shape"]
    cache_path = args.video_cache or dav.get("video_cache_path")
    cache_name = dav.get("video_cache_name", "videos_88_100.uint8")
    if not cache_path or not os.path.exists(cache_path):
        alt = os.path.join(args.root, "data", "visual", "cache", cache_name)
        cache_path = alt if os.path.exists(alt) else cache_path
    if not cache_path or not os.path.exists(cache_path):
        print(f"[FATAL] video memmap not found ({cache_path}); STOP.")
        sys.exit(3)
    videos = np.memmap(cache_path, dtype=np.uint8, mode="r",
                       shape=(n_all, T_FRAMES, H, W))

    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    if ckpt.get("val_idx_sha256") and ckpt["val_idx_sha256"] != val_sha:
        print("[FATAL] ckpt val sha != splits sha; STOP."); sys.exit(2)
    n_classes = len(ckpt["label_to_idx"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AVWordResNet(n_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    alpha = model.gate.alpha.detach()
    print(f"[ckpt] {args.ckpt}  alpha={float(alpha):.4f}  device={device}")
    stride = max(1, int(args.t_stride))

    class V(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k])
            mel = torch.from_numpy(mels_np[g]).unsqueeze(0)
            v = np.array(videos[g])
            if stride > 1: v = v[::stride]
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, vid, int(labels_all[g])

    dl = DataLoader(V(), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    sites = ["a_mid_gap", "v_mid_gap", "gate_out_gap", "block2_gap"]
    feats = {s: [] for s in sites}
    ys, correct, total = [], 0, 0

    print("[build] forwarding AV submodules + GAP over val (val_idx order) ...")
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            a_mid = model.audio_block1(mel)
            v_mid = model.visual(vid)
            g = torch.sigmoid(model.gate.Wa(a_mid) + model.gate.Wv(v_mid))
            a_fused = a_mid * (1.0 + alpha * g)
            b2 = model.audio_block2(a_fused)
            pen = model.gap(b2).flatten(1)
            logits = model.fc(model.dropout(pen))
            y = y.to(device)
            correct += (logits.argmax(1) == y).sum().item()
            total += int(y.numel())
            feats["a_mid_gap"].append(a_mid.mean(dim=(2, 3)).cpu().numpy())
            feats["v_mid_gap"].append(v_mid.mean(dim=(2, 3)).cpu().numpy())
            feats["gate_out_gap"].append(a_fused.mean(dim=(2, 3)).cpu().numpy())
            feats["block2_gap"].append(b2.mean(dim=(2, 3)).cpu().numpy())
            ys.append(y.cpu().numpy())

    base = correct / total
    print(f"[baseline] AV clean-val acc = {base:.6f}  (fp32 ref {BASELINE_FP32_REF})  "
          f"delta={base-BASELINE_FP32_REF:+.6f}")
    if abs(base - BASELINE_FP32_REF) > 0.002:
        print("[WARN] baseline deviates — activation path suspect; staircase NOT trustworthy.")

    y = np.concatenate(ys)
    X = {s: np.concatenate(feats[s]).astype(np.float64) for s in sites}

    out = args.out or os.path.join(args.root, "analysis",
                                   "validator_indep_staircase_av_fused.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Always persist the independently-rebuilt GAP activations + labels so the
    # exact phase_f probe can be run the moment sklearn is authorized.
    acts_path = out.replace(".csv", "_acts.npz")
    np.savez(acts_path, labels=y, val_idx=val_idx, baseline_acc=np.float64(base),
             **{s: X[s] for s in sites})
    print(f"[acts] saved rebuilt GAP activations -> {acts_path}  "
          f"(shapes: " + ", ".join(f"{s}{X[s].shape}" for s in sites) + ")")

    if not HAVE_SKLEARN:
        print("\n[skip] scikit-learn unavailable on this interpreter — the "
              "model-side activation rebuild + baseline self-check are done and "
              "verified; the 5-fold LR staircase is PENDING an sklearn decision. "
              "Activations saved above for an exact probe run.")
        return

    print("\n[staircase] AV_full word-decodability (5-fold, StandardScaler, "
          "LR max_iter=1500 C=1.0, random_state=0):")
    rows = []
    for s in sites:
        acc = _probe_5fold(X[s], y)
        rep = REPORT[s]
        rows.append((s, acc, rep, acc - rep))
        print(f"    {s:<14s} mine={acc:.6f}  report={rep:.6f}  "
              f"delta={acc-rep:+.6f}")
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["site", "acc_5fold_mine", "acc_5fold_report", "delta"])
        for s, a, r, d in rows:
            w.writerow([s, f"{a:.6f}", f"{r:.6f}", f"{d:+.6f}"])
    print(f"[out] wrote {out}")


if __name__ == "__main__":
    main()
