#!/usr/bin/env python3
"""DEBUGGER diag (read-only, Task #34): isolate the ONE axis that flips the
STEEP curve A (models/av_fused_av_noise_sweep.csv) vs the GENTLE curve B
(analysis/validator_indep_q9_rawnoise.csv AV_rawnoise col).

Single-variable design: ONE shared _NoisyAudioView (the exact production view
curve A uses) feeds BOTH checkpoints the SAME mel/vid batch; the only thing that
differs between the two output columns is the model weights. So if clean->curveA
and rawnoise->curveB, the checkpoint is proven causal and noise-protocol / seed /
val-pin / dtype are ruled out (held literally constant, same tensors).

Also reconciles refuting fact #1 (sigma=0 fresh-decode vs cached-mel).
No production code edited; reuses production helpers verbatim.
"""
import os, sys, hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analyze_av_msi import BATCH_SIZE, T_STRIDE, _NoisyAudioView, _accuracy
from dataset_raw_noisy import RawNoisyAVDataset
from model_av import AVWordResNet

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"  # A6000 default (Q3 bit-exact reproducer)
device = torch.device(DEV)
SIGMA = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

# published anchors (from the two CSVs on disk)
CURVE_A = {0.0:0.956712,0.001:0.956712,0.005:0.948894,0.01:0.927536,0.02:0.851831,
           0.05:0.609268,0.1:0.376049,0.2:0.217201,0.5:0.120519}      # av_fused.pt
CURVE_B = {0.0:0.958429,0.001:0.958047,0.005:0.959001,0.01:0.959573,0.02:0.957285,
           0.05:0.952136,0.1:0.938406,0.2:0.847445,0.5:0.421434}      # av_fused_rawnoise.pt


def load_av(name):
    ck = torch.load(os.path.join("models", name), weights_only=False)
    m = AVWordResNet(len(ck["label_to_idx"]))
    m.load_state_dict(ck["model_state_dict"])
    return m.to(device).eval(), ck


@torch.no_grad()
def fwd_av(model, mel, vid):
    a_mid = model.audio_block1(mel)
    v_mid = model.visual(vid)
    pen = model.gap(model.audio_block2(model.gate(a_mid, v_mid))).flatten(1)
    return model.fc(model.dropout(pen)).argmax(1)


def main():
    print(f"[device] {DEV}", flush=True)
    av_clean, ckc = load_av("av_fused.pt")
    av_raw, ckr = load_av("av_fused_rawnoise.pt")
    val_idx = np.asarray(ckc["val_idx"], dtype=np.int64)
    sha = hashlib.sha256(val_idx.tobytes()).hexdigest()
    print(f"[val] N={len(val_idx)} sha={sha[:16]} clean_nk={ckc.get('noise_kind','-')} "
          f"raw_nk={ckr.get('noise_kind','-')}", flush=True)

    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)

    print(f"\n{'sigma':>7} | {'clean_AV':>9} {'(curveA)':>9} {'dA':>9} | "
          f"{'raw_AV':>9} {'(curveB)':>9} {'dB':>9} | dtype", flush=True)
    print("-" * 86, flush=True)
    rows = []
    for sig in SIGMA:
        view = _NoisyAudioView(base, val_idx, sigma_mult=sig, seed=0)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        cc = ct = rc = rt = 0
        dt = None
        for mel, vid, y in loader:
            mel = mel.unsqueeze(1).to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            dt = str(mel.dtype)
            pc = fwd_av(av_clean, mel, vid).cpu()
            pr = fwd_av(av_raw, mel, vid).cpu()
            cc += (pc == y).sum().item(); ct += y.numel()
            rc += (pr == y).sum().item(); rt += y.numel()
        acc_c, acc_r = cc / ct, rc / rt
        dA = acc_c - CURVE_A[sig]; dB = acc_r - CURVE_B[sig]
        print(f"{sig:7.4f} | {acc_c:9.6f} {CURVE_A[sig]:9.6f} {dA:+9.6f} | "
              f"{acc_r:9.6f} {CURVE_B[sig]:9.6f} {dB:+9.6f} | {dt}", flush=True)
        rows.append((sig, acc_c, acc_r, dA, dB))

    # ---- refuting fact #1: sigma=0 fresh-decode vs cached-mel, BOTH ckpts ----
    print("\n[sigma=0 fresh-vs-cached reconciliation]", flush=True)
    d = torch.load(os.path.join("processed", "dataset_av.pt"), weights_only=False)
    cached = d["spectrograms"]  # (N,80,99) float32 cached log-mel

    class CachedMelView(Dataset):
        def __init__(self, base, idx):
            self.base = base; self.idx = np.asarray(idx, dtype=np.int64)
        def __len__(self): return len(self.idx)
        def __getitem__(self, k):
            i = int(self.idx[k])
            mel = torch.from_numpy(np.asarray(cached[i], dtype=np.float32))
            v = np.array(self.base._videos[i])
            if self.base.t_stride > 1: v = v[:: self.base.t_stride]
            v = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, v, int(self.base.labels[i])

    ld = DataLoader(CachedMelView(base, val_idx), batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=4, pin_memory=True)
    cc = ct = rc = rt = 0
    # also measure max |fresh_mel - cached_mel| over val to test the docstring claim
    fresh_view = _NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=0)
    md = 0.0
    for k in range(0, len(val_idx), 500):  # sample every 500th to bound cost
        mfresh, _, _ = fresh_view[k]
        i = int(val_idx[k])
        md = max(md, float(np.abs(mfresh.numpy() - np.asarray(cached[i], np.float32)).max()))
    for mel, vid, y in ld:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        pc = fwd_av(av_clean, mel, vid).cpu()
        pr = fwd_av(av_raw, mel, vid).cpu()
        cc += (pc == y).sum().item(); ct += y.numel()
        rc += (pr == y).sum().item(); rt += y.numel()
    print(f"  CACHED-mel : clean_AV={cc/ct:.6f}  raw_AV={rc/rt:.6f}", flush=True)
    print(f"  FRESH-mel  : clean_AV={rows[0][1]:.6f}  raw_AV={rows[0][2]:.6f}  (sigma=0 row above)", flush=True)
    print(f"  max|fresh_mel - cached_mel| over sampled val = {md:.3e}", flush=True)


if __name__ == "__main__":
    main()
