#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q10 RSA + CKA numbers
(sklearn-free: scipy.stats.spearmanr + scipy.spatial.distance.pdist + numpy):

  RSA (Spearman rho of cosine class-mean RDMs on the 128-d penultimate):
      AV-vs-A 0.786232 | AV-vs-V 0.742174 | A-vs-V 0.467092   (16110 pairs)
  CKA (linear):
      A.block1_gap <-> AV.a_mid_gap = 0.976481  ("pre-gate ~identical")
      A.penult     <-> AV.penult    = 0.751425  ("after gate diverges")

Independence: forwards A (WordResNet), V (VOnlyFairWordResNet), AV (AVWordResNet)
from the trained submodules over the pinned val set (val_idx order); reimplements
class-mean / RDM / linear-CKA here; phase_a_deepdive.py is NOT imported. The 5-fold
probe number (0.920) is a SEPARATE sklearn quantity, not covered here.

Self-check: penult->fc accuracies must reproduce A 0.926964 / V 0.864989 /
AV 0.956712 (fp32 anchors) or the extracted features are wrong.

fp32, no autocast.

Run on dev-codex:
    python validator_indep_rsa_cka.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_rsa_cka.csv
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
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
REF = {"A": 0.926964, "V": 0.864989, "AV": 0.956712}
REPORT_RSA = {"AV_vs_A": 0.786232, "AV_vs_V": 0.742174, "A_vs_V": 0.467092}
REPORT_CKA = {"A.block1_gap~AV.a_mid_gap": 0.976481, "A.penult~AV.penult": 0.751425}


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _class_mean(feats, labels, n_classes):
    means = np.zeros((n_classes, feats.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for f, l in zip(feats, labels):
        means[int(l)] += f
        counts[int(l)] += 1
    nz = counts > 0
    means[nz] /= counts[nz, None]
    return means


def _class_rdm(feats, labels, n_classes):
    return pdist(_class_mean(feats, labels, n_classes), metric="cosine")


def _linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = (X.T @ Y).reshape(-1)
    num = float(np.dot(num, num))
    den_x = float(np.linalg.norm(X.T @ X, "fro"))
    den_y = float(np.linalg.norm(Y.T @ Y, "fro"))
    return num / (den_x * den_y + 1e-12)


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

    A_b1, A_pen, V_pen, AV_amid, AV_pen, ys = [], [], [], [], [], []
    ca = cv = cav = tot = 0
    print("[fwd] A/V/AV over val ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            yb = y.to(device)
            # A
            b1 = A.block1(mel)
            b2 = A.block2(b1)
            penA = A.gap(b2).flatten(1)
            ca += (A.fc(A.dropout(penA)).argmax(1) == yb).sum().item()
            A_b1.append(b1.mean((2, 3)).cpu().numpy())
            A_pen.append(penA.cpu().numpy())
            # V
            vf = V.visual(vid)
            vb2 = V.block2(vf)
            penV = V.gap(vb2).flatten(1)
            cv += (V.fc(V.dropout(penV)).argmax(1) == yb).sum().item()
            V_pen.append(penV.cpu().numpy())
            # AV
            a_mid = AV.audio_block1(mel)
            v_mid = AV.visual(vid)
            af = AV.gate(a_mid, v_mid)
            ab2 = AV.audio_block2(af)
            penAV = AV.gap(ab2).flatten(1)
            cav += (AV.fc(AV.dropout(penAV)).argmax(1) == yb).sum().item()
            AV_amid.append(a_mid.mean((2, 3)).cpu().numpy())
            AV_pen.append(penAV.cpu().numpy())
            ys.append(y.numpy())
            tot += int(yb.numel())

    accA, accV, accAV = ca / tot, cv / tot, cav / tot
    print(f"[self-check] A={accA:.6f} (ref {REF['A']}) | V={accV:.6f} (ref {REF['V']}) "
          f"| AV={accAV:.6f} (ref {REF['AV']})", flush=True)
    for nm, got, ref in [("A", accA, REF['A']), ("V", accV, REF['V']), ("AV", accAV, REF['AV'])]:
        if abs(got - ref) > 0.002:
            print(f"[WARN] {nm} self-check off by {got-ref:+.6f} — features suspect.")

    labels = np.concatenate(ys)
    n_classes = int(labels.max()) + 1
    A_b1 = np.concatenate(A_b1).astype(np.float64)
    A_pen = np.concatenate(A_pen).astype(np.float64)
    V_pen = np.concatenate(V_pen).astype(np.float64)
    AV_amid = np.concatenate(AV_amid).astype(np.float64)
    AV_pen = np.concatenate(AV_pen).astype(np.float64)

    # RSA
    rdm_a = _class_rdm(A_pen, labels, n_classes)
    rdm_v = _class_rdm(V_pen, labels, n_classes)
    rdm_av = _class_rdm(AV_pen, labels, n_classes)
    rsa = {
        "AV_vs_A": float(spearmanr(rdm_av, rdm_a)[0]),
        "AV_vs_V": float(spearmanr(rdm_av, rdm_v)[0]),
        "A_vs_V":  float(spearmanr(rdm_a, rdm_v)[0]),
    }
    print(f"\n[RSA] (n_pairs={len(rdm_a)})", flush=True)
    for k in rsa:
        print(f"  {k:>8s}: mine={rsa[k]:.6f} report={REPORT_RSA[k]:.6f} "
              f"delta={rsa[k]-REPORT_RSA[k]:+.6f}", flush=True)

    # CKA
    cka = {
        "A.block1_gap~AV.a_mid_gap": _linear_cka(A_b1, AV_amid),
        "A.penult~AV.penult": _linear_cka(A_pen, AV_pen),
    }
    print("\n[CKA]", flush=True)
    for k in cka:
        print(f"  {k}: mine={cka[k]:.6f} report={REPORT_CKA[k]:.6f} "
              f"delta={cka[k]-REPORT_CKA[k]:+.6f}", flush=True)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_rsa_cka.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["metric", "key", "mine", "report", "delta"])
        for k in rsa:
            w.writerow(["RSA_spearman", k, f"{rsa[k]:.6f}", f"{REPORT_RSA[k]:.6f}",
                        f"{rsa[k]-REPORT_RSA[k]:+.6f}"])
        for k in cka:
            w.writerow(["linear_CKA", k, f"{cka[k]:.6f}", f"{REPORT_CKA[k]:.6f}",
                        f"{cka[k]-REPORT_CKA[k]:+.6f}"])
    print(f"[out] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
