#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q13 CROSS-VARIANT viseme-decodability
ordering (AV_INTEGRATION_TIER1_CROSS_VARIANT.md §5, "5-fold LR on penult"):

    mid-fusion (mult & add) > late > early

TIER1 §5 AV_full viseme decodability: mid_mult 0.9183, mid_add 0.9239, late 0.8966,
early 0.8339. Load-bearing claim = the ORDERING (mid sharpens viseme structure most).

Independence: imports only the trained variant ARCHITECTURE classes (AVWordResNet,
AVAdditiveWordResNet, AVLateFusionWordResNet, AVEarlyFusionWordResNet) + the allowed
viseme_class taxonomer + sklearn. phase_t1_cross_variant / phase_f NOT imported. Penult
captured generically via a forward hook on each model's classifier Linear (out_features
== num_classes); probe = phase_f _probe_5fold (per-fold StandardScaler, LR max_iter=1500,
C=1, SKF(5,shuffle,rs=0)), viseme target, drop "other" (4930/5244) — the exact protocol
that reproduced mid_mult 0.918 in validator_indep_probes.

Self-check clean acc per variant: mid_mult 0.956712, mid_add 0.957666, late 0.953661,
early 0.939359.

Run on dev-codex:
    python validator_indep_q13_variant_viseme.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q13_variant_viseme.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
CLEAN_ACC = {"mid_mult": 0.956712, "mid_add": 0.957666, "late": 0.953661, "early": 0.939359}
TIER1_VISEME = {"mid_mult": 0.9183, "mid_add": 0.9239, "late": 0.8966, "early": 0.8339}
TOL = 0.005  # 0.5% relative LR-probe guardrail (mean); early has larger fold std


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _probe_5fold_scaled(X, y, max_iter=1500, C=1.0, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        clf.fit(Xtr, y[tr])
        accs.append(float((clf.predict(Xte) == y[te]).mean()))
    return float(np.mean(accs))


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
    from model_av_additive import AVAdditiveWordResNet
    from model_av_late import AVLateFusionWordResNet
    from model_av_early import AVEarlyFusionWordResNet
    from analyze_av_phonetics import viseme_class  # ALLOWED taxonomer

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
    mdir = os.path.join(args.root, "models")
    stride = max(1, int(args.t_stride))

    class Vw(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k])
            mel = torch.from_numpy(mels_np[g]).unsqueeze(0)   # [1,80,99]
            v = np.array(videos[g])
            if stride > 1: v = v[::stride]
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, vid, int(labels_all[g])

    dl = DataLoader(Vw(), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    VARIANTS = [
        ("mid_mult", "av_fused.pt",          AVWordResNet),
        ("mid_add",  "av_fused_additive.pt", AVAdditiveWordResNet),
        ("late",     "av_fused_late.pt",     AVLateFusionWordResNet),
        ("early",    "av_fused_early.pt",    AVEarlyFusionWordResNet),
    ]

    # labels + viseme mask (shared)
    ck0 = torch.load(os.path.join(mdir, "av_fused.pt"), weights_only=False, map_location="cpu")
    idx_to_label = ck0["idx_to_label"]
    ys = np.asarray([int(labels_all[int(g)]) for g in val_idx], dtype=np.int64)
    visemes = np.asarray([viseme_class(idx_to_label[int(l)]) for l in ys])
    keep = visemes != "other"
    vk = visemes[keep]
    print(f"[masks] viseme keep {int(keep.sum())}/{len(ys)}", flush=True)

    rows, flags = [], []
    results = {}
    for flavour, ckname, cls in VARIANTS:
        ck = torch.load(os.path.join(mdir, ckname), weights_only=False, map_location="cpu")
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        m = m.to(device).eval()
        n_classes = len(ck["label_to_idx"])

        # find classifier Linear (out_features == n_classes); hook its input = penult
        clf_layer = None
        for name, mod in m.named_modules():
            if isinstance(mod, nn.Linear) and mod.out_features == n_classes:
                clf_layer = mod
        if clf_layer is None:
            print(f"[FATAL] {flavour}: no classifier Linear found"); sys.exit(2)
        cap = {}
        def hook(mod, inp, out, cap=cap):
            cap["pen"] = inp[0].detach()
        h = clf_layer.register_forward_hook(hook)

        pens, preds = [], []
        with torch.no_grad():
            for mel, vid, y in dl:
                mel = mel.to(device, non_blocking=True)
                vid = vid.to(device, non_blocking=True)
                logits = m(mel, vid)
                pens.append(cap["pen"].cpu().numpy())
                preds.append(logits.argmax(1).cpu().numpy())
        h.remove()
        pen = np.concatenate(pens).astype(np.float64)
        pr = np.concatenate(preds)
        acc = float((pr == ys).mean())
        ref_acc = CLEAN_ACC[flavour]
        acc_ok = abs(acc - ref_acc) < 1e-3
        print(f"\n[{flavour}] penult dim={pen.shape[1]} clean acc={acc:.6f} "
              f"(ref {ref_acc}) {'OK' if acc_ok else '**FLAG'}", flush=True)

        vis = _probe_5fold_scaled(pen[keep], vk)
        ref_v = TIER1_VISEME[flavour]
        rel = (vis - ref_v) / ref_v * 100.0
        vis_ok = abs(vis - ref_v) / ref_v <= TOL
        print(f"    viseme decodability = {vis:.6f} (TIER1 {ref_v}, rel {rel:+.3f}%) "
              f"{'OK' if vis_ok else '**FLAG'}", flush=True)
        results[flavour] = vis
        if not acc_ok: flags.append((flavour, "clean_acc", acc, ref_acc))
        if not vis_ok: flags.append((flavour, "viseme", vis, ref_v))
        rows.append([flavour, f"{acc:.6f}", f"{ref_acc}", f"{vis:.6f}", f"{ref_v}", f"{rel:+.3f}%"])
        del m
        if device.type == "cuda": torch.cuda.empty_cache()

    # ordering check
    mid = min(results["mid_mult"], results["mid_add"])
    order_ok = (mid > results["late"]) and (results["late"] > results["early"])
    print(f"\n[ordering] mid_mult {results['mid_mult']:.4f} / mid_add {results['mid_add']:.4f} "
          f"> late {results['late']:.4f} > early {results['early']:.4f}  = {order_ok}", flush=True)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q13_variant_viseme.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "clean_acc", "clean_acc_ref", "viseme_decodability",
                    "tier1_ref", "rel"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)

    all_ok = order_ok and not flags
    print("\n[VERDICT]", flush=True)
    print(f"  per-variant clean acc + viseme within {TOL*100:g}% .. {'OK' if not flags else 'FLAG'}", flush=True)
    print(f"  ordering mid > late > early .............. {order_ok}", flush=True)
    if all_ok:
        print("[GO] Q13 cross-variant viseme ordering reproduced: mid-fusion (mult & add) "
              "> late > early.", flush=True)
    else:
        print(f"[NO-GO/FLAG] flags={flags} order_ok={order_ok} -> report to lead.", flush=True)


if __name__ == "__main__":
    main()
