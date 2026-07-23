#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q11 CONFUSION-RESCUE pair counts
(phonetic_clustering_av/confusion_rescue_pairs.csv): audio-only word confusions that
COLLAPSE under AV. Report (Q11) cites TEN/TURN 6→1 and INBOX/OUTBOX 5→0.

Definition (reimplemented from analyze_av_phonetics.confusion_rescue_pairs, NOT imported):
for each model, over MISCLASSIFIED items (pred != true), form the unordered pair
{pred_word, true_word} and increment that pair's count; rescue = A_count − AV_count.
Counts are deterministic integers → must match the anchor EXACTLY (no tolerance).

Independence: own A (audio_only_filtered.pt) + AV (av_fused.pt) eager-fp32 forwards on the
pinned val set; _eval_audio/_eval_av/confusion_rescue_pairs NOT imported. Self-check
acc_A 0.926964, acc_AV 0.956712 (the canonical preds that the anchor was built from).

Run on dev-codex:
    python validator_indep_q11_confusion_pairs.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q11_confusion_pairs.csv
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
ACC_A = 0.926964
ACC_AV = 0.956712
# Anchor (confusion_rescue_pairs.csv top-30): (word_i, word_j, A_count, AV_count)
ANCHOR = [
    ("TEN", "TURN", 6, 1), ("INBOX", "OUTBOX", 5, 0), ("SEVENTEEN", "SEVENTY", 6, 2),
    ("DOWN", "GO", 4, 0), ("TEN", "TIME", 4, 0), ("FORTY", "THIRTY", 4, 0),
    ("START", "STOP", 4, 0), ("MONDAY", "NINETY", 4, 0), ("SEND", "SOUND", 4, 0),
    ("NINETEEN", "NINETY", 6, 3), ("DOCUMENT", "DOCUMENTS", 3, 0), ("DOWN", "ON", 3, 0),
    ("NINETEEN", "TWENTY", 3, 0), ("TAB", "TURN", 3, 0), ("DAD", "TAB", 3, 0),
    ("MUM", "RUN", 3, 0), ("ONE", "RUN", 4, 2), ("PAST", "TO", 3, 1),
    ("MAY", "MUM", 2, 0), ("MUM", "ONE", 2, 0), ("NOVEMBER", "REMINDER", 2, 0),
    ("ALL", "ON", 2, 0), ("SET", "SHUT", 2, 0), ("FIVE", "FOUR", 2, 0),
    ("MAY", "PLAY", 2, 0), ("FULL", "TO", 2, 0), ("FOURTEEN", "TWENTY", 2, 0),
    ("SIXTEEN", "THIRTEEN", 2, 0), ("FIFTY", "THIRTY", 2, 0), ("SCROLL", "SLEEP", 2, 0),
]
CITED = [("TEN", "TURN", 6, 1), ("INBOX", "OUTBOX", 5, 0)]  # the two load-bearing pairs


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _key(a, b):
    return tuple(sorted([str(a), str(b)]))


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
    from train import WordResNet
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
    mdir = os.path.join(args.root, "models")
    stride = max(1, int(args.t_stride))

    def _load(cls, name):
        ck = torch.load(os.path.join(mdir, name), weights_only=False, map_location="cpu")
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        return m.to(device).eval(), ck

    A, ckA = _load(WordResNet, "audio_only_filtered.pt")
    AV, ckAV = _load(AVWordResNet, "av_fused.pt")
    idx_to_label = ckAV["idx_to_label"]
    print("[ckpt] A (audio_only_filtered) + AV (av_fused) loaded", flush=True)

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

    preds_a, preds_av, labs = [], [], []
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            preds_a.append(A(mel).argmax(1).cpu().numpy())
            preds_av.append(AV(mel, vid).argmax(1).cpu().numpy())
            labs.append(y.numpy())
    preds_a = np.concatenate(preds_a)
    preds_av = np.concatenate(preds_av)
    labs = np.concatenate(labs)

    acc_a = float((preds_a == labs).mean())
    acc_av = float((preds_av == labs).mean())
    sc_ok = abs(acc_a - ACC_A) < 5e-4 and abs(acc_av - ACC_AV) < 5e-4
    print(f"[self-check] acc_A={acc_a:.6f} (ref {ACC_A}) acc_AV={acc_av:.6f} (ref {ACC_AV}) "
          f"{'OK' if sc_ok else '**FAIL'}", flush=True)

    # reimplemented confusion_rescue_pairs: unordered {pred,true} over misclassified items
    pair = defaultdict(lambda: [0, 0])
    for p, t in zip(preds_a, labs):
        if p != t:
            pair[_key(idx_to_label[int(p)], idx_to_label[int(t)])][0] += 1
    for p, t in zip(preds_av, labs):
        if p != t:
            pair[_key(idx_to_label[int(p)], idx_to_label[int(t)])][1] += 1

    # ---- compare every anchor pair against my full reproduced dict ----
    rows, flags = [], []
    print("\n[anchor pairs — reproduced vs anchor]", flush=True)
    for wi, wj, ra, rav in ANCHOR:
        a, av = pair.get(_key(wi, wj), [0, 0])
        ok = (a == ra) and (av == rav)
        if not ok:
            flags.append((wi, wj, (a, av), (ra, rav)))
        tag = "OK" if ok else "**FLAG"
        cited = "  <CITED>" if (wi, wj, ra, rav) in CITED else ""
        print(f"  {wi}/{wj}: A={a} AV={av} (anchor {ra}/{rav}) rescue={a-av} {tag}{cited}",
              flush=True)
        rows.append([wi, wj, a, av, a - av, ra, rav, "OK" if ok else "FLAG"])

    # explicit cited-pair gate
    cited_ok = True
    for wi, wj, ra, rav in CITED:
        a, av = pair.get(_key(wi, wj), [0, 0])
        if not (a == ra and av == rav):
            cited_ok = False

    # my own top-15 by rescue (descending), for the record
    mine = sorted([(k, v[0], v[1]) for k, v in pair.items()],
                  key=lambda r: (-(r[1] - r[2]), -r[1]))[:15]
    print("\n[my top-15 rescued pairs]", flush=True)
    for (wi, wj), a, av in mine:
        print(f"  {wi}/{wj}: A={a} AV={av} rescue={a-av}", flush=True)

    out = args.out or os.path.join(args.root, "analysis",
                                   "validator_indep_q11_confusion_pairs.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["word_i", "word_j", "A_count", "AV_count", "rescue",
                    "anchor_A", "anchor_AV", "FLAG"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)

    all_ok = sc_ok and cited_ok and not flags
    print("\n[VERDICT]", flush=True)
    print(f"  self-check acc A/AV ............ {'OK' if sc_ok else 'FAIL'}", flush=True)
    print(f"  cited TEN/TURN 6→1, INBOX/OUTBOX 5→0 .. {'OK' if cited_ok else 'FLAG'}", flush=True)
    print(f"  all 30 anchor pairs exact ...... {'OK' if not flags else f'FLAG {flags}'}", flush=True)
    if all_ok:
        print("[GO] Q11 confusion-rescue pair counts reproduced exactly.", flush=True)
    else:
        print(f"[NO-GO/FLAG] sc_ok={sc_ok} cited_ok={cited_ok} flags={flags} -> report to lead.",
              flush=True)


if __name__ == "__main__":
    main()
