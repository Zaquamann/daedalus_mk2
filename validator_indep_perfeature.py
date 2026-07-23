#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q11 per-feature A-vs-AV accuracy
deltas (sklearn-free; model top-1 accuracy per stimulus bin):

  onset:  /r/ +4.97 | /b/ +4.35 | /w/ +4.32   (biggest named-phoneme gains)
  length: Medium (5-7) +3.62  (gains most)

Method (phase-faithful, phase scripts NOT imported): forward A (WordResNet) and
AV (AVWordResNet) over the pinned val set, take top-1 preds, group each sample by
the TRUE label's onset/length/vowel category, per-group acc = mean(pred==true),
delta = AV - A. Reuses only the label->category taxonomers (get_onset /
get_length_group / get_vowel_group) and the trained submodules.

Self-check: overall A=0.926964 / AV=0.956712 (fp32 anchors). fp32, no autocast.

Run on dev-codex:
    python validator_indep_perfeature.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_perfeature.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
REF = {"A": 0.926964, "AV": 0.956712}
REPORT = {  # (grouping, bin) -> delta the report cites
    ("onset", "/r/"): 0.0497, ("onset", "/b/"): 0.0435, ("onset", "/w/"): 0.0432,
    ("length", "Medium (5-7)"): 0.0362,
}


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _per_group_accuracy(preds, labels, idx_to_label, key_fn):
    correct, total = defaultdict(int), defaultdict(int)
    for p, t in zip(preds, labels):
        g = key_fn(idx_to_label[int(t)])
        total[g] += 1
        if p == t:
            correct[g] += 1
    return {g: (correct[g] / total[g], total[g]) for g in total}


def main() -> None:
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
    from train import WordResNet
    from model_av import AVWordResNet
    from analyze_phoneme_accuracy import get_onset, get_length_group, get_vowel_group

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print("[FATAL] val sha != expected; STOP."); sys.exit(2)

    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels = dav["spectrograms"]
    mels_np = mels.numpy() if hasattr(mels, "numpy") else np.asarray(mels)
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

    def _load(cls, name):
        ck = torch.load(os.path.join(mdir, name), weights_only=False, map_location="cpu")
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        return m.to(device).eval(), ck

    A, a_ck = _load(WordResNet, "audio_only_filtered.pt")
    AV, av_ck = _load(AVWordResNet, "av_fused.pt")
    idx_to_label = av_ck["idx_to_label"]
    stride = max(1, int(args.t_stride))

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

    a_preds, av_preds, ys = [], [], []
    print("[fwd] A + AV top-1 over val ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            xa = A.gap(A.block2(A.block1(mel))).flatten(1)
            a_preds.append(A.fc(xa).argmax(1).cpu().numpy())
            a_mid = AV.audio_block1(mel); v_mid = AV.visual(vid)
            xv = AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)
            av_preds.append(AV.fc(xv).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    a_preds = np.concatenate(a_preds); av_preds = np.concatenate(av_preds)
    labels = np.concatenate(ys)

    accA = float((a_preds == labels).mean())
    accAV = float((av_preds == labels).mean())
    print(f"[self-check] A={accA:.6f} (ref {REF['A']}) | AV={accAV:.6f} (ref {REF['AV']})", flush=True)

    groupings = [("onset", get_onset), ("length", get_length_group), ("vowel", get_vowel_group)]
    out = args.out or os.path.join(args.root, "analysis", "validator_indep_perfeature.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allrows = []
    for gname, fn in groupings:
        ga = _per_group_accuracy(a_preds, labels, idx_to_label, fn)
        gav = _per_group_accuracy(av_preds, labels, idx_to_label, fn)
        rows = []
        for g in ga:
            aacc, n = ga[g]; avacc, _ = gav[g]
            rows.append((g, n, aacc, avacc, avacc - aacc))
        rows.sort(key=lambda r: -r[4])
        print(f"\n[{gname}] top deltas (AV - A):", flush=True)
        for g, n, aacc, avacc, d in rows[:6]:
            tag = ""
            if (gname, g) in REPORT:
                tag = f"  [report delta {REPORT[(gname, g)]:+.4f}]"
            print(f"  {str(g):>14s} n={n:>4d}: A={aacc:.4f} AV={avacc:.4f} delta={d:+.4f}{tag}", flush=True)
        for g, n, aacc, avacc, d in rows:
            allrows.append((gname, g, n, aacc, avacc, d))
    n_neg = sum(1 for r in allrows if r[5] < 0)
    print(f"\n[all-positive check] {len(allrows)} bins, {n_neg} with negative delta", flush=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["grouping", "bin", "n", "A_acc", "AV_acc", "delta"])
        for gname, g, n, aacc, avacc, d in allrows:
            w.writerow([gname, g, n, f"{aacc:.6f}", f"{avacc:.6f}", f"{d:+.6f}"])
    print(f"[out] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
