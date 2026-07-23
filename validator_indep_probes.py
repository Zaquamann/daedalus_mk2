#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the sklearn-gated penult probes for
Q10 and Q11 (my own probe code; phase_a_deepdive / phase_e_geometry / phase_f_flow
NOT imported — only the trained submodules + the label->category taxonomers reused).

Q11 — D5 layer-decodability at the PENULT (phase_f `_probe_5fold`: StandardScaler,
LR max_iter=1500 C=1.0, StratifiedKFold(5, shuffle, rs=0)); target masks per phase_f:
    word   : all samples
    onset  : drop onset in {vowel, other}
    viseme : drop viseme == other
  report penult acc:
    word    AV 0.943173 > A 0.902745 > V 0.821509
    viseme  AV 0.918256 > V 0.872008 > A 0.831237
    onset   AV 0.908224 > A 0.817640 > V 0.783552

Q10 — D2 three-condition viseme probe (phase_e `D2_5_three_cond_viseme`: NO scaler,
LR max_iter=2000 C=1.0 n_jobs=-1, SKF(5, shuffle, rs=0), viseme target, drop "other"),
on the penult of five conditions:
    AV_full 0.920081 | AV_v_zero 0.677688 | AV_audio_zero 0.812576 |
    A_only 0.832454 | V_fair 0.873631

Self-check: penult->fc accuracies must reproduce A 0.926964 / V 0.864989 /
AV 0.956712 ; AV_v_zero(=video-zeroed) 0.008391 ; AV_audio_zero(=audio-zeroed) ~0.4443.

Guardrail: report each delta; >0.5% relative off the report -> FLAG (do not reconcile).
fp32, no autocast.

Run on dev-codex:
    python validator_indep_probes.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_probes.csv
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
REF = {"A": 0.926964, "V": 0.864989, "AV": 0.956712,
       "AV_v_zero": 0.008391, "AV_audio_zero": 0.444317}
TOL = 0.005  # 0.5% relative guardrail
D5_REPORT = {
    ("A_only", "word"): 0.902745, ("V_fair", "word"): 0.821509, ("AV_full", "word"): 0.943173,
    ("A_only", "onset"): 0.817640, ("V_fair", "onset"): 0.783552, ("AV_full", "onset"): 0.908224,
    ("A_only", "viseme"): 0.831237, ("V_fair", "viseme"): 0.872008, ("AV_full", "viseme"): 0.918256,
}
D2_REPORT = {
    "AV_full": 0.920081, "AV_v_zero": 0.677688, "AV_audio_zero": 0.812576,
    "A_only": 0.832454, "V_fair": 0.873631,
}


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _probe_5fold_scaled(X, y, max_iter=1500, C=1.0, seed=0):
    """phase_f _probe_5fold: z-score per fold, LR, mean top-1 acc."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        clf.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(accs))


def _linprobe_5fold(X, y, max_iter=2000, C=1.0, seed=0):
    """phase_e/phase_a no-scaler probe: LR(n_jobs=-1), mean top-1 + balanced acc."""
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
    from model_v_only_fair import VOnlyFairWordResNet
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

    A, _ = _load(WordResNet, "audio_only_filtered.pt")
    V, _ = _load(VOnlyFairWordResNet, "video_only_fair.pt")
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

    pen = {k: [] for k in ["A_only", "V_fair", "AV_full", "AV_v_zero", "AV_audio_zero"]}
    ys = []
    cc = {k: 0 for k in pen}; tot = 0
    print("[fwd] penult for A / V / AV_full / AV_v_zero / AV_audio_zero ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            yb = y.to(device)
            # A
            pa = A.gap(A.block2(A.block1(mel))).flatten(1)
            cc["A_only"] += (A.fc(A.dropout(pa)).argmax(1) == yb).sum().item()
            pen["A_only"].append(pa.cpu().numpy())
            # V
            pv = V.gap(V.block2(V.visual(vid))).flatten(1)
            cc["V_fair"] += (V.fc(V.dropout(pv)).argmax(1) == yb).sum().item()
            pen["V_fair"].append(pv.cpu().numpy())
            # AV full
            a_mid = AV.audio_block1(mel); v_mid = AV.visual(vid)
            pf = AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)
            cc["AV_full"] += (AV.fc(AV.dropout(pf)).argmax(1) == yb).sum().item()
            pen["AV_full"].append(pf.cpu().numpy())
            # AV v_zero (video zeroed -> v_mid := 0)
            pvz = AV.gap(AV.audio_block2(AV.gate(a_mid, torch.zeros_like(a_mid)))).flatten(1)
            cc["AV_v_zero"] += (AV.fc(AV.dropout(pvz)).argmax(1) == yb).sum().item()
            pen["AV_v_zero"].append(pvz.cpu().numpy())
            # AV audio_zero (mel := 0)
            a_mid0 = AV.audio_block1(torch.zeros_like(mel))
            paz = AV.gap(AV.audio_block2(AV.gate(a_mid0, v_mid))).flatten(1)
            cc["AV_audio_zero"] += (AV.fc(AV.dropout(paz)).argmax(1) == yb).sum().item()
            pen["AV_audio_zero"].append(paz.cpu().numpy())
            ys.append(y.numpy()); tot += int(yb.numel())

    labels = np.concatenate(ys)
    for k in pen:
        pen[k] = np.concatenate(pen[k]).astype(np.float64)
    print("[self-check penult->fc accuracy]", flush=True)
    for k in ["A_only", "V_fair", "AV_full", "AV_v_zero", "AV_audio_zero"]:
        acc = cc[k] / tot
        ref = REF.get({"A_only": "A", "V_fair": "V", "AV_full": "AV"}.get(k, k))
        print(f"  {k:<14s} {acc:.6f}  (ref {ref})  delta={acc-ref:+.6f}", flush=True)

    # label categories
    onsets = np.asarray([get_onset(idx_to_label[int(l)]) for l in labels])
    visemes = np.asarray([viseme_class(idx_to_label[int(l)]) for l in labels])
    keep_v = visemes != "other"
    keep_o = (onsets != "vowel") & (onsets != "other")
    print(f"[masks] viseme keep {keep_v.sum()}/{len(labels)} | "
          f"onset keep {keep_o.sum()}/{len(labels)}", flush=True)

    rows = []
    flags = []

    # ---- Q11 D5 penult decodability (scaled probe) ----
    print("\n[Q11 D5 penult decodability] (scaler, max_iter=1500)", flush=True)
    D5_targets = {
        "word": (labels, np.ones(len(labels), bool)),
        "onset": (onsets, keep_o),
        "viseme": (visemes, keep_v),
    }
    for model_key in ["A_only", "V_fair", "AV_full"]:
        for tname, (y, keep) in D5_targets.items():
            acc = _probe_5fold_scaled(pen[model_key][keep], y[keep])
            rep = D5_REPORT[(model_key, tname)]
            rel = (acc - rep) / rep
            tag = "  ** FLAG >0.5%" if abs(rel) > TOL else ""
            if abs(rel) > TOL: flags.append(("D5", model_key, tname, acc, rep, rel))
            print(f"  {model_key:<8s} {tname:<7s} mine={acc:.6f} report={rep:.6f} "
                  f"rel={rel:+.4%}{tag}", flush=True)
            rows.append(("D5_" + tname, model_key, f"{acc:.6f}", f"{rep:.6f}", f"{rel:+.6f}"))

    # ---- Q10 D2 three-condition viseme probe (no scaler, max_iter=2000) ----
    print("\n[Q10 D2 three-cond viseme probe] (no scaler, max_iter=2000, drop 'other')", flush=True)
    yv = visemes[keep_v]
    for cond in ["AV_full", "AV_v_zero", "AV_audio_zero", "A_only", "V_fair"]:
        acc, bal = _linprobe_5fold(pen[cond][keep_v], yv)
        rep = D2_REPORT[cond]
        rel = (acc - rep) / rep
        tag = "  ** FLAG >0.5%" if abs(rel) > TOL else ""
        if abs(rel) > TOL: flags.append(("D2", cond, "viseme", acc, rep, rel))
        print(f"  {cond:<14s} mine={acc:.6f} (bal {bal:.6f}) report={rep:.6f} "
              f"rel={rel:+.4%}{tag}", flush=True)
        rows.append(("D2_viseme", cond, f"{acc:.6f}", f"{rep:.6f}", f"{rel:+.6f}"))

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_probes.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["probe", "condition", "acc_mine", "acc_report", "rel_delta"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)
    if flags:
        print(f"[FLAGS] {len(flags)} probe(s) >0.5% off report:", flush=True)
        for fam, c, t, a, r, rel in flags:
            print(f"   {fam} {c}/{t}: mine={a:.6f} report={r:.6f} rel={rel:+.4%}", flush=True)
    else:
        print("[PASS] all probes within 0.5% of report.", flush=True)


if __name__ == "__main__":
    main()
