#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q9 inverse-effectiveness sweeps.

Reproduces, on dev-codex, WITHOUT importing the project eval scripts
(analyze_av_msi / eval_av_visual_noise — their forwards/sweeps are reimplemented
here; only the deterministic noise formula + signal-processing primitives +
trained submodules are reused):

  * E1 audio-noise sweep  (AV vs A-only):   peak AV-A +49.64pp at sigma_a 0.02
        (A 0.355454 / AV 0.851831), clean +2.97pp, sigma_a 0.50 +8.09pp
  * D1.2 video-noise sweep (AV vs V-fair):  peak AV-V +57.23pp at sigma_v 0.40
        (AV 0.774218 / V 0.201945)

Noise (verbatim from _NoisyAudioView / _NoisyVideoView, deterministic seed+idx):
  audio: rms=sqrt(mean(a^2)+1e-12); a += default_rng(seed+idx).standard_normal*sigma_a*rms; then log-mel
  video: std=v.std(); v += default_rng(seed+idx).standard_normal*sigma_v*std

Self-check (sigma=0): A=0.926964, V=0.864989, AV=0.956712 (all fp32, independently
confirmed earlier). If clean rows miss these, the path is wrong -> noised rows void.

fp32, no autocast — matches the regime the E1/D1 CSVs were computed in.

Run on dev-codex:
    python validator_indep_noise_ie.py --root /scratch/daedelus \
        --out-a /scratch/daedelus/analysis/validator_indep_E1_sigma_a.csv \
        --out-v /scratch/daedelus/analysis/validator_indep_D1_sigma_v.csv
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
SIGMA_A = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
SIGMA_V = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
REF = {"A": 0.926964, "V": 0.864989, "AV": 0.956712}  # fp32 clean anchors
# report's load-bearing peaks (from local CSVs):
REPORT_E1 = {0.0: (0.926964, 0.956712), 0.02: (0.355454, 0.851831), 0.5: (0.039664, 0.120519)}
REPORT_D1 = {0.0: (0.864989, 0.956712), 0.4: (0.201945, 0.774218)}


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--t-stride", type=int, default=2)
    ap.add_argument("--expect-sha", default=EXPECT_SHA)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-a", default=None)
    ap.add_argument("--out-v", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    sys.path.insert(0, args.root)
    from train import WordResNet
    from model_v_only_fair import VOnlyFairWordResNet
    from model_av import AVWordResNet
    from dataset_raw_noisy import RawNoisyAVDataset
    from paired_dataset import _read_wav, _wav_to_log_mel, _pad_audio

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print(f"[FATAL] val sha != expected; STOP."); sys.exit(2)

    base = RawNoisyAVDataset(noise=False, t_stride=args.t_stride, return_video=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = os.path.join(args.root, "models")

    def _load(cls, name):
        ck = torch.load(os.path.join(mdir, name), weights_only=False)
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        if ck.get("val_idx_sha256") and ck["val_idx_sha256"] != val_sha:
            print(f"[FATAL] {name} val sha mismatch; STOP."); sys.exit(2)
        return m.to(device).eval()

    A = _load(WordResNet, "audio_only_filtered.pt")
    V = _load(VOnlyFairWordResNet, "video_only_fair.pt")
    AV = _load(AVWordResNet, "av_fused.pt")
    print("[ckpt] A/V/AV loaded; sha-pinned to val", flush=True)

    stride = base.t_stride

    class AudioNoiseView(Dataset):
        def __init__(self, sigma_mult, seed=0):
            self.s = float(sigma_mult); self.seed = int(seed)
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            idx = int(val_idx[k])
            audio = _read_wav(base.audio_paths[idx])
            if self.s > 0:
                rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
                rng = np.random.default_rng(self.seed + idx)
                audio = audio + rng.standard_normal(len(audio)).astype(np.float32) * (self.s * rms)
            mel = torch.from_numpy(_wav_to_log_mel(
                _pad_audio(audio, int(base.pad_offsets[idx]))).astype(np.float32))
            v = np.array(base._videos[idx])
            if stride > 1: v = v[::stride]
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, vid, int(base.labels[idx])

    class VideoNoiseView(Dataset):
        def __init__(self, sigma_mult, seed=0):
            self.s = float(sigma_mult); self.seed = int(seed)
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            idx = int(val_idx[k])
            mel, v, y = base[idx]            # clean mel + clean video (1,T,88,88) in [0,1]
            if self.s > 0:
                v_np = v.numpy()
                rng = np.random.default_rng(self.seed + idx)
                noise = rng.standard_normal(v_np.shape).astype(np.float32) * (self.s * float(v_np.std()))
                v = torch.from_numpy((v_np + noise).astype(np.float32))
            return mel, v, int(y)

    def _loader(view):
        return DataLoader(view, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    @torch.no_grad()
    def acc_A(loader):
        c = t = 0
        for mel, _v, y in loader:
            x = mel.unsqueeze(1).to(device, non_blocking=True)
            p = A(x).argmax(1).cpu()
            c += (p == y).sum().item(); t += y.numel()
        return c / t

    @torch.no_grad()
    def acc_V(loader):
        c = t = 0
        for _mel, v, y in loader:
            p = V(v.to(device, non_blocking=True)).argmax(1).cpu()
            c += (p == y).sum().item(); t += y.numel()
        return c / t

    @torch.no_grad()
    def acc_AV(loader):
        c = t = 0
        for mel, vid, y in loader:
            mel = mel.unsqueeze(1).to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            a_mid = AV.audio_block1(mel)
            v_mid = AV.visual(vid)
            a_fused = AV.gate(a_mid, v_mid)
            x = AV.audio_block2(a_fused)
            pen = AV.gap(x).flatten(1)
            p = AV.fc(AV.dropout(pen)).argmax(1).cpu()
            c += (p == y).sum().item(); t += y.numel()
        return c / t

    # ---- E1: sigma_a sweep (AV vs A) ----
    out_a = args.out_a or os.path.join(args.root, "analysis", "validator_indep_E1_sigma_a.csv")
    os.makedirs(os.path.dirname(out_a), exist_ok=True)
    print("\n[E1] sigma_a sweep (AV vs A-only):", flush=True)
    rows_a = []
    for sa in SIGMA_A:
        ld = _loader(AudioNoiseView(sa))
        a = acc_A(ld); av = acc_AV(ld)
        rows_a.append((sa, a, av, av - a))
        tag = ""
        if sa in REPORT_E1:
            ra, rav = REPORT_E1[sa]
            tag = f"  [report A={ra:.6f} AV={rav:.6f} dA={av-a:+.6f}]"
        print(f"  sigma_a={sa:6.4f}: A={a:.6f} AV={av:.6f} d={av-a:+.6f}{tag}", flush=True)
    with open(out_a, "w") as f:
        w = csv.writer(f)
        w.writerow(["sigma_per_rms", "A_acc", "AV_acc", "AV_minus_A"])
        for sa, a, av, d in rows_a:
            w.writerow([f"{sa:.4f}", f"{a:.6f}", f"{av:.6f}", f"{d:.6f}"])
    print(f"[out] wrote {out_a}", flush=True)

    # ---- D1.2: sigma_v sweep (AV vs V) ----
    out_v = args.out_v or os.path.join(args.root, "analysis", "validator_indep_D1_sigma_v.csv")
    print("\n[D1.2] sigma_v sweep (AV vs V-fair):", flush=True)
    rows_v = []
    for sv in SIGMA_V:
        ld = _loader(VideoNoiseView(sv))
        vv = acc_V(ld); av = acc_AV(ld)
        rows_v.append((sv, vv, av, av - vv))
        tag = ""
        if sv in REPORT_D1:
            rv, rav = REPORT_D1[sv]
            tag = f"  [report V={rv:.6f} AV={rav:.6f} dV={av-vv:+.6f}]"
        print(f"  sigma_v={sv:6.4f}: V={vv:.6f} AV={av:.6f} d={av-vv:+.6f}{tag}", flush=True)
    with open(out_v, "w") as f:
        w = csv.writer(f)
        w.writerow(["sigma_v_per_pixstd", "V_only_acc", "AV_acc", "AV_minus_V"])
        for sv, vv, av, d in rows_v:
            w.writerow([f"{sv:.4f}", f"{vv:.6f}", f"{av:.6f}", f"{d:.6f}"])
    print(f"[out] wrote {out_v}", flush=True)

    # ---- self-checks ----
    a0 = [r for r in rows_a if r[0] == 0.0][0]
    v0 = [r for r in rows_v if r[0] == 0.0][0]
    print("\n[self-check sigma=0 vs fp32 anchors]")
    print(f"  A : {a0[1]:.6f} (ref {REF['A']}) d={a0[1]-REF['A']:+.6f}")
    print(f"  V : {v0[1]:.6f} (ref {REF['V']}) d={v0[1]-REF['V']:+.6f}")
    print(f"  AV: {a0[2]:.6f} / {v0[2]:.6f} (ref {REF['AV']})")


if __name__ == "__main__":
    main()
