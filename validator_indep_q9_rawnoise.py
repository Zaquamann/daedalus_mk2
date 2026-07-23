#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the Q9 RAWNOISE-pair sweep
(E1_inverse_effectiveness_4model.csv A_rawnoise / AV_rawnoise columns): training with
raw-waveform noise FLATTENS the inverse-effectiveness curve — the rawnoise-trained pair
stays high (~0.89-0.96) through sigma_a 0.10 while the clean pair collapses.

SECONDARY: the two main Q9 peaks (+49.64pp @σ_a 0.02, +57.23pp @σ_v 0.40) are already
verified (validator_indep_E1_sigma_a / D1_sigma_v). This closes the rawnoise aside.

Independence: same reimplemented additive-noise view + eval loop as validator_indep_
noise_ie (analyze_av_msi NOT imported); only swaps the checkpoints to the rawnoise-trained
A (audio_only_rawnoise_filtered.pt) and AV (av_fused_rawnoise.pt).

Self-check σ=0 (E1_4model): A_rawnoise 0.932494, AV_rawnoise 0.958429.

Run on dev-codex:
    python validator_indep_q9_rawnoise.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q9_rawnoise.csv
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
# E1_4model rawnoise anchors: sigma -> (A_rawnoise, AV_rawnoise); 0.02 A is nan in artifact
ANCHOR = {
    0.0: (0.932494, 0.958429), 0.001: (0.931541, 0.958047), 0.005: (0.933066, 0.959001),
    0.01: (0.932494, 0.959573), 0.02: (float("nan"), 0.957285), 0.05: (0.923722, 0.952136),
    0.1: (0.894355, 0.938215), 0.2: (0.734935, 0.847445), 0.5: (0.212624, 0.421243),
}


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
    from dataset_raw_noisy import RawNoisyAVDataset
    from paired_dataset import _read_wav, _wav_to_log_mel, _pad_audio

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print("[FATAL] val sha mismatch"); sys.exit(2)

    base = RawNoisyAVDataset(noise=False, t_stride=args.t_stride, return_video=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = os.path.join(args.root, "models")
    stride = base.t_stride

    def _load(cls, name):
        ck = torch.load(os.path.join(mdir, name), weights_only=False)
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        return m.to(device).eval()

    A = _load(WordResNet, "audio_only_rawnoise_filtered.pt")
    AV = _load(AVWordResNet, "av_fused_rawnoise.pt")
    print("[ckpt] A_rawnoise + AV_rawnoise loaded", flush=True)

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
    def acc_A(loader):
        c = t = 0
        for mel, _v, y in loader:
            x = mel.unsqueeze(1).to(device, non_blocking=True)
            p = A(x).argmax(1).cpu()
            c += (p == y).sum().item(); t += y.numel()
        return c / t

    @torch.no_grad()
    def acc_AV(loader):
        c = t = 0
        for mel, vid, y in loader:
            mel = mel.unsqueeze(1).to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            pen = AV.gap(AV.audio_block2(AV.gate(AV.audio_block1(mel), AV.visual(vid)))).flatten(1)
            p = AV.fc(AV.dropout(pen)).argmax(1).cpu()
            c += (p == y).sum().item(); t += y.numel()
        return c / t

    print("\n[Q9 rawnoise] sigma_a sweep (A_rawnoise + AV_rawnoise):", flush=True)
    rows, flags = [], []
    res = {}
    for sa in SIGMA_A:
        ld = _loader(AudioNoiseView(sa))
        a = acc_A(ld); av = acc_AV(ld)
        res[sa] = (a, av)
        ra, rav = ANCHOR[sa]
        da = (a - ra) if not np.isnan(ra) else float("nan")
        dav = av - rav
        # gate AV (and A where anchor finite) within 0.5% rel; high-uncertainty sigma may drift
        av_ok = abs(dav) / rav <= 0.005
        a_ok = np.isnan(ra) or abs(da) / ra <= 0.005
        if not av_ok: flags.append((sa, "AV", av, rav))
        if not a_ok: flags.append((sa, "A", a, ra))
        print(f"  sigma_a={sa:6.4f}: A={a:.6f} (anc {ra}) AV={av:.6f} (anc {rav}) "
              f"dAV={dav:+.6f} {'OK' if (av_ok and a_ok) else '**FLAG'}", flush=True)
        rows.append([f"{sa:.4f}", f"{a:.6f}", f"{ra}", f"{av:.6f}", f"{rav}", f"{dav:+.6f}"])

    # load-bearing: rawnoise pair stays high through sigma_a 0.10
    a010, av010 = res[0.1]
    a005, av005 = res[0.05]
    flat_ok = (av010 >= 0.92 and a010 >= 0.88 and av005 >= 0.94 and a005 >= 0.91)
    sc_ok = abs(res[0.0][0] - 0.932494) < 5e-4 and abs(res[0.0][1] - 0.958429) < 5e-4
    print(f"\n[flatten check] σ=0.05 A {a005:.4f}/AV {av005:.4f} ; σ=0.10 A {a010:.4f}/AV "
          f"{av010:.4f} — pair stays high (clean pair collapses to A 0.134/AV 0.609 @0.05) "
          f"= {flat_ok}", flush=True)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q9_rawnoise.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sigma_per_rms", "A_rawnoise", "anchor_A", "AV_rawnoise", "anchor_AV", "dAV"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)

    print("\n[VERDICT]", flush=True)
    print(f"  self-check σ=0 (A 0.9325 / AV 0.9584) .. {'OK' if sc_ok else 'FAIL'}", flush=True)
    print(f"  rawnoise pair stays high thru σ_a 0.10 . {'OK' if flat_ok else 'FLAG'}", flush=True)
    print(f"  anchor rows within 0.5% ................ {'OK' if not flags else f'FLAG {flags}'}", flush=True)
    if sc_ok and flat_ok and not flags:
        print("[GO] Q9 rawnoise IE-flattening reproduced.", flush=True)
    else:
        print(f"[NO-GO/FLAG] sc_ok={sc_ok} flat_ok={flat_ok} flags={flags} -> report to lead.", flush=True)


if __name__ == "__main__":
    main()
