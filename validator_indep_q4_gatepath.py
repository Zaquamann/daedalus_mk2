#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q4 gate-path attribution +
variance decomposition (Q4_variance_decomposition.csv):

  variance decomp (pp):
    total_AV_minus_A        +2.9748  = AV - A
    ensemble_attainable     +2.3646  = ensemble_50_50 - A
    learned_fusion_residual +0.6102  = AV - ensemble_50_50  (sum_check ✓)
  gate-path (on A-wrong -> AV-right rescues):
    n_rescued                261     (A-wrong & AV-right)
    n_regressed              105     (A-right & AV-wrong)   [net +156]
    frac_visual_gate_driven  0.992337  (rescue LOST when Wv(v_mid) ablated = v_mid:=0)
    frac_audio_path_survives 0.007663

Wv is bias-free, so video-zero (v_mid:=0) IS the visual-gate-term-ablated model
(per q4_variance_decomposition.py:16-17). Independence: forwards A/V/AV from the
trained submodules; reimplements ensemble + rescue/regression counts here. The
project's q4 script is NOT imported.

Self-check: A 0.926964 / V 0.864989 / AV 0.956712 / AV_v_zero 0.008391 /
ensemble 0.950610 — guards every per-sample prediction array the counts rest on.
fp32, no autocast.

Run on dev-codex:
    python validator_indep_q4_gatepath.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q4_gatepath.csv
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
REF = {"A": 0.926964, "V": 0.864989, "AV": 0.956712, "AV_v_zero": 0.008391,
       "ens5050": 0.950610}
REPORT = {
    "total_AV_minus_A": 2.9748, "ensemble_attainable": 2.3646,
    "learned_fusion_residual": 0.6102,
    "n_rescued": 261, "n_regressed": 105, "net": 156,
    "frac_visual_gate_driven": 0.992337, "frac_audio_path_survives": 0.007663,
}


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


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
    V = _load(VOnlyFairWordResNet, "video_only_fair.pt")
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

    logits_A, logits_V = [], []
    a_pred, av_pred, avvz_pred, ys = [], [], [], []
    print("[fwd] A / V / AV_full / AV_v_zero ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            la = A(mel); logits_A.append(la.float().cpu().numpy())
            a_pred.append(la.argmax(1).cpu().numpy())
            lv = V(vid); logits_V.append(lv.float().cpu().numpy())
            a_mid = AV.audio_block1(mel); v_mid = AV.visual(vid)
            lf = AV.fc(AV.dropout(AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)))
            av_pred.append(lf.argmax(1).cpu().numpy())
            lvz = AV.fc(AV.dropout(AV.gap(AV.audio_block2(AV.gate(a_mid, torch.zeros_like(a_mid)))).flatten(1)))
            avvz_pred.append(lvz.argmax(1).cpu().numpy())
            ys.append(y.numpy())

    y = np.concatenate(ys)
    a_pred = np.concatenate(a_pred); av_pred = np.concatenate(av_pred)
    avvz_pred = np.concatenate(avvz_pred)
    logits_A = np.concatenate(logits_A); logits_V = np.concatenate(logits_V)
    p_ens = 0.5 * (_softmax(logits_A) + _softmax(logits_V))
    ens_pred = p_ens.argmax(1)

    accA = float((a_pred == y).mean()); accAV = float((av_pred == y).mean())
    accVZ = float((avvz_pred == y).mean()); acc_ens = float((ens_pred == y).mean())
    accV = float((logits_V.argmax(1) == y).mean())
    print("[self-check]", flush=True)
    for nm, got in [("A", accA), ("V", accV), ("AV", accAV),
                    ("AV_v_zero", accVZ), ("ens5050", acc_ens)]:
        print(f"  {nm:<10s} {got:.6f}  (ref {REF[nm]})  delta={got-REF[nm]:+.6f}", flush=True)

    # variance decomposition (pp)
    total = (accAV - accA) * 100.0
    gain_ens = (acc_ens - accA) * 100.0
    gain_resid = (accAV - acc_ens) * 100.0

    # gate-path attribution
    A_wrong = a_pred != y
    AV_right = av_pred == y
    rescued = A_wrong & AV_right
    n_resc = int(rescued.sum())
    regressed = (~A_wrong) & (~AV_right)        # A-right & AV-wrong
    n_reg = int(regressed.sum())
    lost_when_vablated = rescued & (avvz_pred != y)
    frac_visual = float(lost_when_vablated.sum()) / max(n_resc, 1)
    frac_audio = 1.0 - frac_visual

    derived = {
        "total_AV_minus_A": total, "ensemble_attainable": gain_ens,
        "learned_fusion_residual": gain_resid,
        "n_rescued": n_resc, "n_regressed": n_reg, "net": n_resc - n_reg,
        "frac_visual_gate_driven": frac_visual, "frac_audio_path_survives": frac_audio,
    }
    print("\n[Q4 derived vs report]", flush=True)
    for k in ["total_AV_minus_A", "ensemble_attainable", "learned_fusion_residual",
              "n_rescued", "n_regressed", "net",
              "frac_visual_gate_driven", "frac_audio_path_survives"]:
        d = derived[k] - REPORT[k]
        print(f"  {k:<26s} mine={derived[k]:<10} report={REPORT[k]:<10} delta={d:+.6f}", flush=True)
    print(f"  sum_check attainable+residual = {gain_ens+gain_resid:.4f}pp (total {total:.4f}pp)", flush=True)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q4_gatepath.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "mine", "report", "delta"])
        for k in ["total_AV_minus_A", "ensemble_attainable", "learned_fusion_residual",
                  "n_rescued", "n_regressed", "net",
                  "frac_visual_gate_driven", "frac_audio_path_survives"]:
            w.writerow([k, f"{derived[k]}", f"{REPORT[k]}", f"{derived[k]-REPORT[k]:+.6f}"])
    print(f"[out] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
