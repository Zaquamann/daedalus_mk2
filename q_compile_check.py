#!/usr/bin/env python3
"""Decide whether torch.compile / bf16 is load-bearing for the AV clean anchor.

torch.compile is unavailable on this pod (Triton's runtime JIT needs Python.h,
which is absent and uninstallable without root). The eval harness analyze_av_msi.py
runs EAGER (no autocast, no compile). This script reproduces the clean-AV val
accuracy on the pinned val partition THREE ways, all no-compile:
  (1) eager fp32   (what the harness actually does)
  (2) eager bf16-autocast
and compares to the published anchor 0.956712. Read-only on av_fused.pt.
"""
import hashlib
import os
import sys

import numpy as np
import torch

from dataset_raw_noisy import RawNoisyAVDataset
from model_av import AVWordResNet

ANCHOR = 0.956712
PIN = "03c5a87acdcf07ad"
T_STRIDE = 2
BS = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] device={device} torch={torch.__version__}", flush=True)

base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
ck = torch.load("models/av_fused.pt", weights_only=False)
val_idx = np.asarray(ck["val_idx"]).astype(np.int64)
sha = hashlib.sha256(val_idx.tobytes()).hexdigest()
print(f"[pin] val N={len(val_idx)} sha16={sha[:16]}", flush=True)
assert len(val_idx) == 5244 and sha.startswith(PIN), "VAL PIN MISMATCH"

n_classes = len(ck["label_to_idx"])
model = AVWordResNet(n_classes).to(device)
model.load_state_dict(ck["model_state_dict"])
model.eval()


def _batch(idxs):
    mels, vids, labs = [], [], []
    for i in idxs:
        m, v, l = base[int(i)]
        mels.append(torch.as_tensor(m))
        vids.append(torch.as_tensor(v))
        labs.append(int(l))
    mb = torch.stack(mels)                 # (B,80,99)
    if mb.dim() == 3:
        mb = mb.unsqueeze(1)               # (B,1,80,99)  matches _forward_AV
    vb = torch.stack(vids)                 # (B,1,T,88,88)
    return mb.to(device), vb.to(device), torch.tensor(labs)


@torch.no_grad()
def _acc(use_bf16):
    correct = total = 0
    for s in range(0, len(val_idx), BS):
        mb, vb, lb = _batch(val_idx[s:s + BS])
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                a_mid = model.audio_block1(mb)
                v_mid = model.visual(vb)
                a_fused = model.gate(a_mid, v_mid)
                x = model.audio_block2(a_fused)
                pen = model.gap(x).flatten(1)
                logits = model.fc(model.dropout(pen))
        else:
            a_mid = model.audio_block1(mb)
            v_mid = model.visual(vb)
            a_fused = model.gate(a_mid, v_mid)
            x = model.audio_block2(a_fused)
            pen = model.gap(x).flatten(1)
            logits = model.fc(model.dropout(pen))
        correct += (logits.argmax(1).cpu() == lb).sum().item()
        total += lb.numel()
    return correct / total


acc_fp32 = _acc(False)
acc_bf16 = _acc(True)
print(f"[result] eager fp32 clean-AV acc = {acc_fp32:.6f}  (anchor {ANCHOR:.6f}, "
      f"d={acc_fp32 - ANCHOR:+.6f})", flush=True)
print(f"[result] eager bf16 clean-AV acc = {acc_bf16:.6f}  (anchor {ANCHOR:.6f}, "
      f"d={acc_bf16 - ANCHOR:+.6f})", flush=True)
print("DONE", flush=True)
