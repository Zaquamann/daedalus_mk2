#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q13 noise-robustness RANKING
across the four integration-stage variants (the load-bearing Q13 claim):

    mid_mult (av_fused) > mid_add (additive) > late > early   at sigma_a = 0.05

Reproduces the D1_cross_variant_noise.csv AV_acc column at sigma_a in {0.0, 0.05},
fp32. The sigma_a=0 row is a per-variant self-check against the cross-variant
clean accuracies (mult 0.956712 / late 0.953661 / add 0.957666 / early 0.939359).

Independence: imports only the trained variant ARCHITECTURE classes (AVWordResNet,
AVLateFusionWordResNet, AVAdditiveWordResNet, AVEarlyFusionWordResNet) — each does
its own internal fusion in forward(audio, video) — plus an independently
reimplemented audio-noise view + eval loop. phase_t1_cross_variant.py is NOT imported.

Audio noise (verbatim deterministic formula, seed+idx):
  rms=sqrt(mean(a^2)+1e-12); a += default_rng(seed+idx).standard_normal*sigma_a*rms; log-mel.

Run on dev-codex:
    python validator_indep_stage_ranking.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_stage_ranking.csv
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
SIGMAS = (0.0, 0.05)
# report values from D1_cross_variant_noise.csv  (flavour -> {sigma: acc})
REPORT = {
    "mid_mult": {0.0: 0.956712, 0.05: 0.609268},
    "mid_add":  {0.0: 0.957666, 0.05: 0.559878},
    "late":     {0.0: 0.953661, 0.05: 0.280320},
    "early":    {0.0: 0.939359, 0.05: 0.226163},
}


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


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
    from model_av import AVWordResNet
    from model_av_late import AVLateFusionWordResNet
    from model_av_additive import AVAdditiveWordResNet
    from model_av_early import AVEarlyFusionWordResNet
    from dataset_raw_noisy import RawNoisyAVDataset
    from paired_dataset import _read_wav, _wav_to_log_mel, _pad_audio

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print("[FATAL] val sha != expected; STOP."); sys.exit(2)

    base = RawNoisyAVDataset(noise=False, t_stride=args.t_stride, return_video=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stride = base.t_stride
    mdir = os.path.join(args.root, "models")

    VARIANTS = [
        ("mid_mult", "av_fused.pt",          AVWordResNet),
        ("mid_add",  "av_fused_additive.pt", AVAdditiveWordResNet),
        ("late",     "av_fused_late.pt",     AVLateFusionWordResNet),
        ("early",    "av_fused_early.pt",    AVEarlyFusionWordResNet),
    ]

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

    def _loader(view):
        return DataLoader(view, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    @torch.no_grad()
    def acc(model, loader):
        c = t = 0
        for mel, vid, y in loader:
            mel = mel.unsqueeze(1).to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            p = model(mel, vid).argmax(1).cpu()
            c += (p == y).sum().item(); t += y.numel()
        return c / t

    # pre-build the loaders once per sigma (shared across variants)
    loaders = {sig: _loader(AudioNoiseView(sig)) for sig in SIGMAS}

    rows = []
    print("\n[stage ranking] AV_acc by variant x sigma_a (fp32):", flush=True)
    results = {}
    for flavour, ckname, cls in VARIANTS:
        ck = torch.load(os.path.join(mdir, ckname), weights_only=False, map_location="cpu")
        if ck.get("val_idx_sha256") and ck["val_idx_sha256"] != val_sha:
            print(f"[FATAL] {ckname} val sha mismatch; STOP."); sys.exit(2)
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        m = m.to(device).eval()
        results[flavour] = {}
        for sig in SIGMAS:
            a = acc(m, loaders[sig])
            results[flavour][sig] = a
            rep = REPORT[flavour][sig]
            rows.append((flavour, sig, a, rep, a - rep))
            print(f"  {flavour:>8s} sigma_a={sig:.2f}: mine={a:.6f} report={rep:.6f} "
                  f"delta={a-rep:+.6f}", flush=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ranking at 0.05
    order = sorted(results, key=lambda f: -results[f][0.05])
    print("\n[ranking @ sigma_a=0.05] " +
          " > ".join(f"{f}({results[f][0.05]:.4f})" for f in order), flush=True)
    print("[report ranking]          mid_mult(0.6093) > mid_add(0.5599) > "
          "late(0.2803) > early(0.2262)", flush=True)

    out = args.out or os.path.join(args.root, "analysis",
                                   "validator_indep_stage_ranking.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["flavour", "sigma_a", "acc_mine", "acc_report", "delta"])
        for fl, sig, a, r, d in rows:
            w.writerow([fl, f"{sig:.4f}", f"{a:.6f}", f"{r:.6f}", f"{d:+.6f}"])
    print(f"[out] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
