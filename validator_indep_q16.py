#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of Q16 (single-modality REINTERPRETATION):
does joint AV training change how the audio modality is *linearly represented*?

Compares linear-DECODABILITY (word/onset/viseme, D5 5-fold probe) of:
  - A_only net first block (block1_gap), vs
  - AV net audio branch mid-rep (a_mid_gap), on IDENTICAL audio,
and the geometry dissociation linear-CKA(A_block1, AV_a_mid).

INDEPENDENCE: I do NOT load processed/deepdive_act_cache.pt (the generator's path).
I FORWARD audio_only_filtered.pt and av_fused.pt myself on the pinned val
(sha 03c5a87a, N=5244), fp32 eager, no autocast. a_mid is captured (1) in a real-video
pass (also yields av_pred for the AV self-check + the CKA partner) and (2) in a SEPARATE
ZEROED-VIDEO-INPUT pass — if audio_block1 truly ignores video, the two a_mid arrays are
bit-identical (max|Δ|=0), a strictly stronger leak test than the report's same-pass diff.

Probes/metric reimplemented verbatim-equivalent (NOT imported):
  _probe_5fold  : D5 = StratifiedKFold(5,shuffle,rs=0), per-fold StandardScaler,
                  LogisticRegression(max_iter=1500,C=1.0).
  _linear_cka   : centered, ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F).
ALLOWED reuse (label->category taxonomers only): analyze_phoneme_accuracy.get_onset,
analyze_av_phonetics.viseme_class. Masks match the generator: onset keep =
(onset!='vowel')&(onset!='other'); viseme keep = viseme!='other'.

GATE (no per-Q anchor spec from lead yet -> key to the report CSV at the project
guardrail): each load-bearing decodability acc reproduced within 0.5% RELATIVE of the
Q16 CSV value; CKA within 1e-3 ABS (deterministic). a_mid leak max|Δ| must be 0.
Self-check A 0.926964 / AV 0.956712. Any miss -> FLAG to lead (no self-reconcile).

Run on dev-codex:
    python validator_indep_q16.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q16.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
REF = {"A": 0.926964, "AV": 0.956712}

# --- Q16 CSV (the report numbers I must independently reproduce) ---
# (section/readout/model) -> reported value
REPORT = {
    ("word", "A_only_block1", "acc"): 0.424676,
    ("word", "A_only_block1", "bal"): 0.357341,
    ("word", "AV_a_mid_vzero", "acc"): 0.279367,
    ("word", "AV_a_mid_vzero", "bal"): 0.197571,
    ("word", "gap_pp"): 14.5309,
    ("onset", "A_only_block1", "acc"): 0.552801,
    ("onset", "A_only_block1", "bal"): 0.452036,
    ("onset", "AV_a_mid_vzero", "acc"): 0.455304,
    ("onset", "AV_a_mid_vzero", "bal"): 0.329097,
    ("onset", "gap_pp"): 9.7497,
    ("viseme", "A_only_block1", "acc"): 0.631440,
    ("viseme", "A_only_block1", "bal"): 0.454231,
    ("viseme", "AV_a_mid_vzero", "acc"): 0.570183,
    ("viseme", "AV_a_mid_vzero", "bal"): 0.307447,
    ("viseme", "gap_pp"): 6.1258,
    ("geometry", "linear_CKA"): 0.976481,
}
# the generator's own published-anchor self-check targets (1.8.0-era D5/D4), informational
PUB = {("word", "AV"): 0.279558, ("word", "A"): 0.426774,
       ("viseme", "AV"): 0.569777, ("viseme", "A"): 0.631643, "cka": 0.976481}

REL_GUARD = 5e-3   # project >=0.5% LR-probe reproduction guardrail (relative)
CKA_ABS = 1e-3     # deterministic CKA -> tight absolute


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _probe_5fold(X, y, max_iter=1500, C=1.0, seed=0):
    """D5 / phase_f_flow._probe_5fold: 5-fold SKF, per-fold StandardScaler, LR."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, bal = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        bal.append(balanced_accuracy_score(y[te], pred))
    return float(np.mean(accs)), float(np.mean(bal))


def _linear_cka(X, Y):
    """phase_f_flow._linear_cka: centered linear CKA."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = (X.T @ Y).reshape(-1)
    num = float(np.dot(num, num))
    den_x = float(np.linalg.norm(X.T @ X, "fro"))
    den_y = float(np.linalg.norm(Y.T @ Y, "fro"))
    return num / (den_x * den_y + 1e-12)


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
    from analyze_phoneme_accuracy import get_onset
    from analyze_av_phonetics import viseme_class

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

    def _load(cls, name):
        ck = torch.load(os.path.join(mdir, name), weights_only=False, map_location="cpu")
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        return m.to(device).eval(), ck

    A, _ = _load(WordResNet, "audio_only_filtered.pt")
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

    a_block1_l, amid_full_l, amid_vz_l = [], [], []
    a_pred, av_pred, ys = [], [], []
    print("[fwd] A(block1) + AV a_mid (real-video pass + zeroed-video-input pass) ...",
          flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            # A_only block1 + full (for acc self-check)
            x1 = A.block1(mel)
            a_block1_l.append(A.gap(x1).flatten(1).cpu().numpy())
            a_pen = A.gap(A.block2(x1)).flatten(1)
            a_pred.append(A.fc(a_pen).argmax(1).cpu().numpy())
            # AV real-video full pipeline -> av_pred + a_mid(full-video)
            a_mid = AV.audio_block1(mel)
            v_mid = AV.visual(vid)
            pen_f = AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)
            av_pred.append(AV.fc(AV.dropout(pen_f)).argmax(1).cpu().numpy())
            amid_full_l.append(AV.gap(a_mid).flatten(1).cpu().numpy())
            # SEPARATE zeroed-video-INPUT pass: recompute a_mid; run visual on zeros.
            # If video leaks into the audio branch this would differ from a_mid(full).
            vid0 = torch.zeros_like(vid)
            a_mid_z = AV.audio_block1(mel)
            _ = AV.visual(vid0)
            amid_vz_l.append(AV.gap(a_mid_z).flatten(1).cpu().numpy())
            ys.append(y.numpy())

    y = np.concatenate(ys).astype(np.int64)
    a_block1 = np.concatenate(a_block1_l).astype(np.float64)
    amid_full = np.concatenate(amid_full_l).astype(np.float64)
    amid_vz = np.concatenate(amid_vz_l).astype(np.float64)
    accA = float((np.concatenate(a_pred) == y).mean())
    accAV = float((np.concatenate(av_pred) == y).mean())
    print(f"[self-check] A={accA:.6f} (ref {REF['A']}) AV={accAV:.6f} (ref {REF['AV']})",
          flush=True)
    sc_ok = abs(accA - REF["A"]) < 5e-4 and abs(accAV - REF["AV"]) < 5e-4

    # ---- 1) airtight identical-audio: a_mid video-independent (leak test) ----
    diff = np.abs(amid_vz - amid_full)
    max_d, mean_d = float(diff.max()), float(diff.mean())
    print(f"[identical-audio] a_mid(zero-video-input) vs a_mid(real-video): "
          f"max|Δ|={max_d:.3e} mean|Δ|={mean_d:.3e}", flush=True)
    leak_ok = max_d < 1e-5

    # readout label vectors + masks (match generator exactly)
    onsets = np.asarray([get_onset(idx_to_label[int(l)]) for l in y])
    visemes = np.asarray([viseme_class(idx_to_label[int(l)]) for l in y])
    keep_o = (onsets != "vowel") & (onsets != "other")
    keep_v = visemes != "other"
    readouts = [("word", y, slice(None)),
                ("onset", onsets, keep_o),
                ("viseme", visemes, keep_v)]
    print(f"[masks] onset keep N={int(keep_o.sum())}/{len(y)} | "
          f"viseme keep N={int(keep_v.sum())}/{len(y)}", flush=True)

    rows, flags = [], []

    def _chk(key, val, ref, tol_rel=REL_GUARD, abs_tol=None):
        if ref is None:
            return ""
        if abs_tol is not None:
            ok = abs(val - ref) <= abs_tol
            d = f"{val - ref:+.3e}(abs)"
        else:
            ok = abs(val - ref) <= tol_rel * abs(ref)
            d = f"{(val - ref) / ref * 100:+.3f}%"
        if not ok:
            flags.append((key, val, ref, d))
        return d + ("" if ok else "  ** FLAG")

    # ---- 2) decodability (D5 probe) on identical audio ----
    print("\n[decodability] D5 5-fold linear probe on identical audio", flush=True)
    dec = {}
    for tname, yv, keep in readouts:
        Xa = a_block1 if keep is slice(None) else a_block1[keep]
        Xv = amid_vz if keep is slice(None) else amid_vz[keep]
        yk = yv if keep is slice(None) else yv[keep]
        a_acc, a_bal = _probe_5fold(Xa, yk)
        v_acc, v_bal = _probe_5fold(Xv, yk)
        gap = (a_acc - v_acc) * 100
        dec[tname] = (a_acc, a_bal, v_acc, v_bal, gap)
        da = _chk((tname, "A_only_block1", "acc"), a_acc, REPORT.get((tname, "A_only_block1", "acc")))
        dva = _chk((tname, "AV_a_mid_vzero", "acc"), v_acc, REPORT.get((tname, "AV_a_mid_vzero", "acc")))
        dgap = _chk((tname, "gap_pp"), gap, REPORT.get((tname, "gap_pp")), abs_tol=0.05)
        print(f"  {tname:>7s} | A_block1 acc={a_acc:.6f} {da:>14s} | "
              f"AV_a_mid acc={v_acc:.6f} {dva:>14s} | gap={gap:+.4f}pp {dgap}", flush=True)
        rows += [
            ["decodability", tname, "A_only_block1", "acc_5fold", f"{a_acc:.6f}",
             _fmt(REPORT.get((tname, "A_only_block1", "acc"))), da],
            ["decodability", tname, "A_only_block1", "bal_acc_5fold", f"{a_bal:.6f}",
             _fmt(REPORT.get((tname, "A_only_block1", "bal"))),
             _chk((tname, "A_only_block1", "bal"), a_bal, REPORT.get((tname, "A_only_block1", "bal")))],
            ["decodability", tname, "AV_a_mid_vzero", "acc_5fold", f"{v_acc:.6f}",
             _fmt(REPORT.get((tname, "AV_a_mid_vzero", "acc"))), dva],
            ["decodability", tname, "AV_a_mid_vzero", "bal_acc_5fold", f"{v_bal:.6f}",
             _fmt(REPORT.get((tname, "AV_a_mid_vzero", "bal"))),
             _chk((tname, "AV_a_mid_vzero", "bal"), v_bal, REPORT.get((tname, "AV_a_mid_vzero", "bal")))],
            ["decodability", tname, "block1_minus_a_mid", "acc_gap_pp", f"{gap:.4f}",
             _fmt(REPORT.get((tname, "gap_pp"))), dgap],
        ]

    # ---- 3) geometry/function dissociation: linear CKA ----
    cka = _linear_cka(a_block1, amid_full)
    dcka = _chk(("geometry", "linear_CKA"), cka, REPORT.get(("geometry", "linear_CKA")), abs_tol=CKA_ABS)
    print(f"\n[geometry] linear CKA(A_block1, AV_a_mid) = {cka:.6f} "
          f"(report {REPORT[('geometry', 'linear_CKA')]:.6f}) {dcka}", flush=True)
    rows.append(["geometry", "a_mid_vs_block1", "A_x_AV", "linear_CKA", f"{cka:.6f}",
                 _fmt(REPORT.get(("geometry", "linear_CKA"))), dcka])

    # ---- informational: distance to the generator's 1.8.0 published anchors ----
    print("\n[informational] vs generator's published (1.8.0) anchors:", flush=True)
    for nm, anc in [(("word", "AV"), dec["word"][2]), (("word", "A"), dec["word"][0]),
                    (("viseme", "AV"), dec["viseme"][2]), (("viseme", "A"), dec["viseme"][0]),
                    ("cka", cka)]:
        a = PUB[nm]
        print(f"    {str(nm):>16s} mine={anc:.6f} pub={a:.6f} Δ={(anc - a) / a * 100:+.3f}%", flush=True)

    rows += [
        ["identical_audio", "a_mid", "AV_vzero_vs_full", "max_abs_diff", f"{max_d:.3e}",
         "0.000e+00", "OK" if leak_ok else "** FLAG"],
        ["identical_audio", "a_mid", "AV_vzero_vs_full", "mean_abs_diff", f"{mean_d:.3e}", "", ""],
        ["selfcheck", "model_acc", "A/AV", "accuracy", f"{accA:.6f}/{accAV:.6f}",
         f"{REF['A']}/{REF['AV']}", "OK" if sc_ok else "** FLAG"],
    ]

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q16.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "readout", "model", "metric", "value", "report", "delta_vs_report"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)

    # ---- verdict ----
    print("\n[VERDICT]", flush=True)
    print(f"  model self-check ............ {'OK' if sc_ok else 'FAIL'}", flush=True)
    print(f"  a_mid video-independence .... {'OK (max|Δ|=0)' if leak_ok else 'FAIL'}", flush=True)
    if flags and (sc_ok and leak_ok):
        print(f"  decodability/CKA ............ {len(flags)} value(s) > guardrail", flush=True)
    if (not flags) and sc_ok and leak_ok:
        print("[GO] all Q16 load-bearing numbers reproduced within guardrail; "
              "a_mid video-independent; CKA bit-close.", flush=True)
    else:
        print(f"[NO-GO/FLAG] {len(flags)} flag(s); sc_ok={sc_ok} leak_ok={leak_ok} -> "
              f"report to lead (no self-reconcile). flags={flags}", flush=True)


def _fmt(v):
    return "" if v is None else (f"{v:.6f}" if v < 1.5 else f"{v:.4f}")


if __name__ == "__main__":
    main()
