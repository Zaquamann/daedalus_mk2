#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of two sklearn-free load-bearing numbers:

  Q2/Q6/Q8 — late 50/50 softmax ensemble of the A + V specialists:
      ensemble_50_50 = 0.950610 ; AV - ensemble = +0.6102 pp
  Q3 — AV model on a single hard-zeroed stream (D1_3x3_clean):
      input_A_only (video zeroed -> v_mid := 0)      = 0.008391
      input_V_only (audio zeroed -> mel  := 0)       = 0.444699

Faithful to the project's exact semantics (read, NOT imported):
  * ensemble (phase_a_deepdive.D4_1): p = 0.5*(softmax(A_logits)+softmax(V_logits)),
    numpy max-subtracted softmax, argmax.
  * zeroing (analyze_av_msi._forward_AV): audio_kind="zero" => mel = zeros_like(mel);
    video_kind="zero" => v_mid = zeros_like(a_mid) (HARD zero, not visual(0)).

Independence: forwards A (WordResNet), V (VOnlyFairWordResNet), AV (AVWordResNet)
from the trained submodules; reimplements ensemble + zeroing here. phase_a_deepdive
and analyze_av_msi are NOT imported.

Self-check: A=0.926964 / V=0.864989 / AV=0.956712 (fp32 anchors) — guards the path.
fp32, no autocast.

Run on dev-codex:
    python validator_indep_ensemble_zero.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_ensemble_zero.csv
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
REF = {"A": 0.926964, "V": 0.864989, "AV": 0.956712}
REPORT = {
    "ensemble_50_50": 0.950610,
    "AV_minus_5050_pp": 0.6102,
    "av_input_A_only": 0.008391,   # video zeroed (v_mid := 0)
    "av_input_V_only": 0.444699,   # audio zeroed (mel := 0)
}


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _softmax(x):  # verbatim from phase_a_deepdive.D4_1_late_ensemble
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


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
        if ck.get("val_idx_sha256") and ck["val_idx_sha256"] != val_sha:
            print(f"[FATAL] {name} val sha mismatch; STOP."); sys.exit(2)
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
    pa, pv, pav_full, pav_aonly, pav_vonly, ys = [], [], [], [], [], []
    print("[fwd] A / V / AV (full + a_only[v_mid=0] + v_only[mel=0]) over val ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            # A specialist (full forward)
            la = A(mel)
            logits_A.append(la.float().cpu().numpy())
            pa.append(la.argmax(1).cpu().numpy())
            # V specialist (full forward)
            lv = V(vid)
            logits_V.append(lv.float().cpu().numpy())
            pv.append(lv.argmax(1).cpu().numpy())
            # AV full
            a_mid = AV.audio_block1(mel)
            v_mid = AV.visual(vid)
            x = AV.audio_block2(AV.gate(a_mid, v_mid))
            pav_full.append(AV.fc(AV.dropout(AV.gap(x).flatten(1))).argmax(1).cpu().numpy())
            # AV input_A_only: video zeroed -> v_mid := 0 (hard)
            v0 = torch.zeros_like(a_mid)
            xa = AV.audio_block2(AV.gate(a_mid, v0))
            pav_aonly.append(AV.fc(AV.dropout(AV.gap(xa).flatten(1))).argmax(1).cpu().numpy())
            # AV input_V_only: audio zeroed -> mel := 0
            a_mid0 = AV.audio_block1(torch.zeros_like(mel))
            xv = AV.audio_block2(AV.gate(a_mid0, v_mid))
            pav_vonly.append(AV.fc(AV.dropout(AV.gap(xv).flatten(1))).argmax(1).cpu().numpy())
            ys.append(y.numpy())

    labels = np.concatenate(ys)
    logits_A = np.concatenate(logits_A); logits_V = np.concatenate(logits_V)
    accA = float((np.concatenate(pa) == labels).mean())
    accV = float((np.concatenate(pv) == labels).mean())
    accAV = float((np.concatenate(pav_full) == labels).mean())
    print(f"[self-check] A={accA:.6f} (ref {REF['A']}) | V={accV:.6f} (ref {REF['V']}) "
          f"| AV={accAV:.6f} (ref {REF['AV']})", flush=True)
    bad = [nm for nm, got, ref in [("A", accA, REF['A']), ("V", accV, REF['V']),
                                   ("AV", accAV, REF['AV'])] if abs(got - ref) > 0.002]
    if bad:
        print(f"[WARN] self-check off for {bad} — derived numbers suspect.")

    # ---- Q3 zeroed-stream accuracies ----
    av_a_only = float((np.concatenate(pav_aonly) == labels).mean())
    av_v_only = float((np.concatenate(pav_vonly) == labels).mean())

    # ---- Q2 50/50 softmax ensemble ----
    p_a = _softmax(logits_A); p_v = _softmax(logits_V)
    p_ens = 0.5 * (p_a + p_v)
    acc_ens = float((p_ens.argmax(1) == labels).mean())
    av_minus_ens_pp = (accAV - acc_ens) * 100.0

    derived = {
        "ensemble_50_50": acc_ens,
        "AV_minus_5050_pp": av_minus_ens_pp,
        "av_input_A_only": av_a_only,
        "av_input_V_only": av_v_only,
    }
    print("\n[derived vs report]", flush=True)
    for k in ["av_input_A_only", "av_input_V_only", "ensemble_50_50", "AV_minus_5050_pp"]:
        print(f"  {k:<18s}: mine={derived[k]:.6f}  report={REPORT[k]:.6f}  "
              f"delta={derived[k]-REPORT[k]:+.6f}", flush=True)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_ensemble_zero.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "mine", "report", "delta"])
        w.writerow(["A_acc", f"{accA:.6f}", f"{REF['A']:.6f}", f"{accA-REF['A']:+.6f}"])
        w.writerow(["V_acc", f"{accV:.6f}", f"{REF['V']:.6f}", f"{accV-REF['V']:+.6f}"])
        w.writerow(["AV_acc", f"{accAV:.6f}", f"{REF['AV']:.6f}", f"{accAV-REF['AV']:+.6f}"])
        for k in ["av_input_A_only", "av_input_V_only", "ensemble_50_50", "AV_minus_5050_pp"]:
            w.writerow([k, f"{derived[k]:.6f}", f"{REPORT[k]:.6f}", f"{derived[k]-REPORT[k]:+.6f}"])
    print(f"[out] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
