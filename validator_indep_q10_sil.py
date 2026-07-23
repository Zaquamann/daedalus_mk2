#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q10 viseme-silhouette CAVEAT
(artifact analysis/phonetic_clustering_av/viseme_silhouette.csv): viseme-cluster
tightness on the penult is NEGATIVE for EVERY model (A_clean -0.017778, A_noisy
-0.016164, AV_clean -0.007644, AV_noisy -0.008755), AV less-negative than A — clusters
tighten under AV but never cleanly separate in absolute terms.

INDEPENDENCE: I FORWARD the four frozen checkpoints MYSELF on the pinned clean val
(sha 03c5a87a, N=5244), eager fp32, .eval() — analyze_av_phonetics's eval/forward NOT
imported. I reuse ONLY the allowed label->category taxonomer (analyze_av_phonetics.
viseme_class) and the stats primitive sklearn.metrics.silhouette_score. Penult =
gap(block2).flatten(1) (128-d). Silhouette spec reproduced verbatim: keep visemes !=
"other", metric="euclidean".

Models (clean/noisy = different TRAINED checkpoints, all eval'd on the SAME clean val):
  A_clean  audio_only_filtered.pt        (anchor val_acc 0.926964)
  A_noisy  audio_only_noisy_filtered.pt  (summary val_acc 0.929443)
  AV_clean av_fused.pt                   (anchor val_acc 0.956712)
  AV_noisy av_fused_noisy.pt             (summary val_acc 0.951754)

GATE: deterministic (no LR probe). The load-bearing caveat is SIGN + ORDERING — confirm
all four silhouettes NEGATIVE and AV_clean > A_clean (less negative); values reproduced
within a small absolute tol of the artifact. Self-check val_acc on the two anchored
models exact. Any miss -> flag to lead.

Run on dev-codex:
    python validator_indep_q10_sil.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q10_silhouette.csv
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
from sklearn.metrics import silhouette_score

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
VAL_ACC = {"A_clean": 0.926964, "A_noisy": 0.929443,
           "AV_clean": 0.956712, "AV_noisy": 0.951754}
ARTIFACT = {"A_clean": -0.017778, "A_noisy": -0.016164,
            "AV_clean": -0.007644, "AV_noisy": -0.008755}
CKPTS = [("A_clean", "audio_only_filtered.pt", "audio"),
         ("A_noisy", "audio_only_noisy_filtered.pt", "audio"),
         ("AV_clean", "av_fused.pt", "av"),
         ("AV_noisy", "av_fused_noisy.pt", "av")]
SIL_ABS_TOL = 3e-3   # small-number deterministic silhouette; sign+ordering are the claim


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
    from train import WordResNet
    from model_av import AVWordResNet
    from analyze_av_phonetics import viseme_class   # ALLOWED taxonomer

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print("[FATAL] val sha != expected; STOP."); sys.exit(2)

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

    models = {}
    idx_to_label = None
    for tag, fname, kind in CKPTS:
        cls = WordResNet if kind == "audio" else AVWordResNet
        m, ck = _load(cls, fname)
        models[tag] = (m, kind)
        if idx_to_label is None:
            idx_to_label = ck["idx_to_label"]

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

    feats = {t: [] for t, _, _ in CKPTS}
    preds = {t: [] for t, _, _ in CKPTS}
    ys = []
    print("[fwd] penult for A_clean/A_noisy/AV_clean/AV_noisy on clean val ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            for tag, (m, kind) in models.items():
                if kind == "audio":
                    x = m.block2(m.block1(mel))
                    pen = m.gap(x).flatten(1)
                else:
                    a_mid = m.audio_block1(mel); v_mid = m.visual(vid)
                    x = m.audio_block2(m.gate(a_mid, v_mid))
                    pen = m.gap(x).flatten(1)
                feats[tag].append(pen.cpu().numpy())
                preds[tag].append(m.fc(pen).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys).astype(np.int64)
    feats = {t: np.concatenate(v).astype(np.float64) for t, v in feats.items()}
    preds = {t: np.concatenate(v) for t, v in preds.items()}

    # self-check accuracies
    print("[self-check val_acc]", flush=True)
    sc_ok = True
    for tag, _, _ in CKPTS:
        acc = float((preds[tag] == y).mean())
        ref = VAL_ACC[tag]
        tol = 5e-4 if tag in ("A_clean", "AV_clean") else 5e-3
        ok = abs(acc - ref) < tol
        sc_ok = sc_ok and ok
        print(f"    {tag:>9s} acc={acc:.6f} (ref {ref}) {'OK' if ok else '** FLAG'}", flush=True)

    # viseme labels + keep mask (verbatim: ignore "other" only)
    visemes = np.asarray([viseme_class(idx_to_label[int(l)]) for l in y])
    keep = visemes != "other"
    vk = visemes[keep]
    print(f"[masks] viseme keep N={int(keep.sum())}/{len(y)} ; "
          f"classes={sorted(set(vk.tolist()))}", flush=True)

    rows, flags = [], []
    print("\n[silhouette] euclidean, viseme grouping (keep != other):", flush=True)
    sil = {}
    for tag, _, _ in CKPTS:
        sval = float(silhouette_score(feats[tag][keep], vk, metric="euclidean"))
        sil[tag] = sval
        ref = ARTIFACT[tag]
        d = sval - ref
        neg = sval < 0
        ok = abs(d) <= SIL_ABS_TOL
        tag_flag = "" if (ok and neg) else "  ** FLAG"
        if not (ok and neg):
            flags.append((tag, sval, ref))
        print(f"    {tag:>9s} sil={sval:+.6f} (artifact {ref:+.6f}) Δ={d:+.2e} "
              f"neg={neg}{tag_flag}", flush=True)
        rows.append([tag, "viseme", f"{sval:.6f}", f"{ref:.6f}", f"{d:+.6f}",
                     "negative" if neg else "POSITIVE"])

    av_less_neg = sil["AV_clean"] > sil["A_clean"]
    all_neg = all(sil[t] < 0 for t, _, _ in CKPTS)
    print(f"\n[caveat checks] all four NEGATIVE = {all_neg}; "
          f"AV_clean ({sil['AV_clean']:+.6f}) > A_clean ({sil['A_clean']:+.6f}) "
          f"(AV less negative) = {av_less_neg}", flush=True)
    print(f"[caveat checks] AV_noisy ({sil['AV_noisy']:+.6f}) > A_noisy "
          f"({sil['A_noisy']:+.6f}) = {sil['AV_noisy'] > sil['A_noisy']}", flush=True)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q10_silhouette.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "grouping", "silhouette_score", "artifact", "delta", "sign"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)

    print("\n[VERDICT]", flush=True)
    print(f"  self-check val_acc ........... {'OK' if sc_ok else 'FAIL'}", flush=True)
    print(f"  all silhouettes negative ..... {all_neg}", flush=True)
    print(f"  AV less-negative than A ...... {av_less_neg}", flush=True)
    print(f"  values within {SIL_ABS_TOL:g} abs ...... {not flags}", flush=True)
    if sc_ok and all_neg and av_less_neg and not flags:
        print("[GO] Q10 viseme-silhouette caveat reproduced: negative for every model, "
              "AV tighter (less negative) than A, values match artifact.", flush=True)
    else:
        print(f"[NO-GO/FLAG] flags={flags} sc_ok={sc_ok} all_neg={all_neg} "
              f"av_less_neg={av_less_neg} -> report to lead (no self-reconcile).", flush=True)


if __name__ == "__main__":
    main()
