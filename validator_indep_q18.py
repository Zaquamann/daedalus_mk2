#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of Q18 (per-layer audio-only / v_mid=0
decodability) + ADJUDICATION of the AV_clean_v_zero penult 0.512209 vs
published_anchor 0.526320 (-1.41pp), the only self-check >0.5%.

My own forward (deepdive_act_cache NOT loaded). The generator's own docstring says
0.526320 came from D4_linprobe_class.csv = the _linprobe_5fold variant (NO scaler,
max_iter=2000), whereas Q18's NEW 0.512209 uses _probe_5fold (z-score scaler,
max_iter=1500). DECISIVE TEST: rebuild the v_zero penult ONCE and run BOTH probe
variants on it.
  - if D5(scaler/1500) ~0.512 AND D4(no-scaler/2000) ~0.526  => the -1.41pp is purely
    a probe-variant difference on identical activations -> anchor not apples-to-apples
    -> BENIGN (no real discrepancy).
  - if either lands elsewhere -> real discrepancy -> flag to lead/debugger.

Structural (deterministic): a_mid identical full-vs-vzero (video-independent);
cliff = full - v_zero at penult ~0.4315; audio-only penult 0.512 << A-only 0.899 <<
full-AV 0.944.

Self-check: A 0.926964 / AV 0.956712. fp32, no autocast. A-block1_gap is the
separate -0.21pp sklearn-drift item (reported, NOT gated — it is the debugger's task).

Run on dev-codex:
    python validator_indep_q18.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q18.csv
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
ANCHOR_VZERO_PENULT_D4 = 0.526320
# Q18 D5 (_probe_5fold) reported acc per (cond,layer)
REPORT = {
    ("A_only", "block1_gap"): 0.424676, ("A_only", "block2_gap"): 0.898741,
    ("A_only", "penult"): 0.898741,
    ("AV_clean_full", "a_mid_gap"): 0.279367, ("AV_clean_full", "gate_out_gap"): 0.760678,
    ("AV_clean_full", "block2_gap"): 0.943744, ("AV_clean_full", "penult"): 0.943744,
    ("AV_clean_v_zero", "a_mid_gap"): 0.279367,
    ("AV_clean_v_zero", "gate_out_gap"): 0.346491,
    ("AV_clean_v_zero", "block2_gap"): 0.512209, ("AV_clean_v_zero", "penult"): 0.512209,
}
CLIFF_PENULT = 0.431535


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


def _linprobe_5fold(X, y, max_iter=2000, C=1.0, seed=0):
    """D4/D2 / _linprobe_5fold: 5-fold SKF, NO scaler, LR (n_jobs ignored 1.8+)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, bal = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        bal.append(balanced_accuracy_score(y[te], pred))
    return float(np.mean(accs)), float(np.mean(bal))


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
        return m.to(device).eval()

    A = _load(WordResNet, "audio_only_filtered.pt")
    AV = _load(AVWordResNet, "av_fused.pt")
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

    acc_layers = {"A_block1": [], "A_block2": [],
                  "AVf_amid": [], "AVf_gate": [], "AVf_blk2": [],
                  "AVz_gate": [], "AVz_blk2": []}
    a_pred, av_pred, ys = [], [], []
    print("[fwd] A(block1/block2) + AV full/v_zero (a_mid/gate/block2) ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            # A_only
            x1 = A.block1(mel); x2 = A.block2(x1)
            acc_layers["A_block1"].append(A.gap(x1).flatten(1).cpu().numpy())
            a_pen = A.gap(x2).flatten(1)
            acc_layers["A_block2"].append(a_pen.cpu().numpy())
            a_pred.append(A.fc(a_pen).argmax(1).cpu().numpy())
            # AV full
            a_mid = AV.audio_block1(mel); v_mid = AV.visual(vid)
            gate_f = AV.gate(a_mid, v_mid); blk2_f = AV.audio_block2(gate_f)
            pen_f = AV.gap(blk2_f).flatten(1)
            acc_layers["AVf_amid"].append(AV.gap(a_mid).flatten(1).cpu().numpy())
            acc_layers["AVf_gate"].append(AV.gap(gate_f).flatten(1).cpu().numpy())
            acc_layers["AVf_blk2"].append(pen_f.cpu().numpy())
            av_pred.append(AV.fc(AV.dropout(pen_f)).argmax(1).cpu().numpy())
            # AV v_zero (v_mid := 0)
            gate_z = AV.gate(a_mid, torch.zeros_like(a_mid)); blk2_z = AV.audio_block2(gate_z)
            acc_layers["AVz_gate"].append(AV.gap(gate_z).flatten(1).cpu().numpy())
            acc_layers["AVz_blk2"].append(AV.gap(blk2_z).flatten(1).cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys)
    acc_layers = {k: np.concatenate(v).astype(np.float64) for k, v in acc_layers.items()}
    accA = float((np.concatenate(a_pred) == y).mean())
    accAV = float((np.concatenate(av_pred) == y).mean())
    print(f"[self-check] A={accA:.6f} (ref {REF['A']}) AV={accAV:.6f} (ref {REF['AV']})", flush=True)
    # deterministic guardrail: AV a_mid is video-independent (same in full and v_zero)
    print(f"[guardrail] AV a_mid is computed pre-gate -> identical full/v_zero by construction", flush=True)

    rows, flags = [], []

    def _emit(cond, layer, X, variant="D5"):
        if variant == "D5":
            a, b = _probe_5fold(X, y)
        else:
            a, b = _linprobe_5fold(X, y)
        rep = REPORT.get((cond, layer))
        d = "" if rep is None else f"{(a-rep)*100:+.3f}pp"
        tag = ""
        if rep is not None and abs(a - rep) > 0.004:   # ~0.4pp LR cross-version bar
            tag = "  ** FLAG >0.4pp"; flags.append((cond, layer, a, rep))
        print(f"  [{variant}] {cond:>16s} {layer:>12s} acc={a:.6f} bal={b:.6f} "
              f"rep={rep} {d}{tag}", flush=True)
        rows.append([variant, cond, layer, f"{a:.6f}", f"{b:.6f}",
                     "" if rep is None else f"{rep:.6f}", d])
        return a, b

    print("\n[D5 _probe_5fold: scaler + max_iter=1500] (Q18's NEW measurement)", flush=True)
    a_b1, _ = _emit("A_only", "block1_gap", acc_layers["A_block1"])
    a_pen, _ = _emit("A_only", "penult", acc_layers["A_block2"])
    avf_amid, _ = _emit("AV_clean_full", "a_mid_gap", acc_layers["AVf_amid"])
    avf_gate, _ = _emit("AV_clean_full", "gate_out_gap", acc_layers["AVf_gate"])
    avf_pen, _ = _emit("AV_clean_full", "penult", acc_layers["AVf_blk2"])
    _emit("AV_clean_v_zero", "gate_out_gap", acc_layers["AVz_gate"])
    avz_pen_d5, _ = _emit("AV_clean_v_zero", "penult", acc_layers["AVz_blk2"])

    print("\n[D4 _linprobe_5fold: NO scaler + max_iter=2000] (the ANCHOR's variant)", flush=True)
    avz_pen_d4, _ = _emit("AV_clean_v_zero", "penult", acc_layers["AVz_blk2"], variant="D4")

    # structural
    cliff = avf_pen - avz_pen_d5
    print(f"\n[structural] cliff full-minus-vzero penult (D5) = {cliff:.6f} (rep {CLIFF_PENULT})", flush=True)
    print(f"[structural] audio-only penult {avz_pen_d5:.4f} << A-only penult {a_pen:.4f} "
          f"<< full-AV penult {avf_pen:.4f} : "
          f"{avz_pen_d5 < a_pen < avf_pen}", flush=True)
    mono = avf_amid < avf_gate < avf_pen
    print(f"[structural] full-AV staircase monotone a_mid<gate<penult = {mono}", flush=True)

    # ADJUDICATION
    d5_ok = abs(avz_pen_d5 - 0.512209) <= 0.004
    d4_ok = abs(avz_pen_d4 - ANCHOR_VZERO_PENULT_D4) <= 0.004
    verdict = ("BENIGN: anchor 0.526320 is the D4 no-scaler variant; -1.41pp is the "
               "scaler-vs-no-scaler probe difference on identical activations"
               if (d5_ok and d4_ok) else
               "DISCREPANCY: probe-variant hypothesis NOT confirmed -> flag to debugger")
    print(f"\n[ADJUDICATION] D5(scaler)={avz_pen_d5:.6f} (~0.512209? {d5_ok}) | "
          f"D4(no-scaler)={avz_pen_d4:.6f} (~0.526320? {d4_ok})", flush=True)
    print(f"[ADJUDICATION] {verdict}", flush=True)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q18.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "condition", "layer", "acc_5fold", "bal_acc_5fold", "report_acc", "delta_pp"])
        for r in rows:
            w.writerow(r)
        w.writerow(["_struct", "cliff_penult", "", f"{cliff:.6f}", "", f"{CLIFF_PENULT}", ""])
        w.writerow(["_adjudication", "vzero_penult_D5", "", f"{avz_pen_d5:.6f}", "", "0.512209", ""])
        w.writerow(["_adjudication", "vzero_penult_D4", "", f"{avz_pen_d4:.6f}", "", "0.526320", ""])
        w.writerow(["_adjudication", "verdict", "", verdict, "", "", ""])
    print(f"\n[out] wrote {out}", flush=True)
    if flags:
        print(f"[FLAGS (note: A_only/block1_gap is the separate debugger task #23)] {flags}", flush=True)
    else:
        print("[PASS] D5 numbers reproduced; adjudication resolved.", flush=True)


if __name__ == "__main__":
    main()
