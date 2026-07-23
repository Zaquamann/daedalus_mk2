#!/usr/bin/env python3
"""TEMP DEBUG INSTRUMENT (debugger, task #16, Q2) — analytical clean-acc ceiling
for the late-fusion reliability-gate topology with FROZEN v2 heads. NO training.

R2 residual: clean fused val acc = 0.854, target >= 0.95. Question: is 0.95
reachable by THIS topology (fused = w_a*logit_a + w_v*logit_v, w_a+w_v=1, per
sample) with the current ep60 heads? Compute the upper bounds:

  audio_acc / video_acc  = the two endpoints (w_a=1 / w_a=0)
  union                  = P(audio right OR video right) = best HARD-routing gate
  convex_oracle          = P(exists w in [0,1]: argmax(w*la+(1-w)*lv)==y)
                           = the TRUE ceiling of this convex-gate topology, frozen heads
  equal_sum              = acc of (la+lv) [diagnostic: unconstrained MLE-ish sum,
                           NOT reachable by the convex gate]
  best_global_w          = best single fixed w_a over the whole val set

Interpretation:
  convex_oracle < 0.95  -> 0.95 UNREACHABLE by any gate with THESE heads; the
                           deficit is in the HEADS (head quality), so R2's path to
                           0.95 is via better heads = TRAINING BUDGET (if heads are
                           still climbing) — NOT a gate fix.
  convex_oracle >= 0.95 -> a perfect gate WOULD reach 0.95 with these heads; the
                           actual 0.854 then reflects GATE miscalibration.

Eval is fp32, model.eval() (dropout=identity) — matches the d' harness forward.
Reads the ARCHIVED v2 ckpt so it is robust to a concurrent retrain overwrite.
Run:  CUDA_VISIBLE_DEVICES=1 python analysis/deepdive/diag_q2_gate_ceiling.py
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from analyze_av_msi import RawNoisyAVDataset, _NoisyAudioView, T_STRIDE, BATCH_SIZE
from model_av_latefusion import AVLateFusionReliabilityWordResNet

CKPT = os.environ.get("LATE_CKPT",
                      os.path.join(ROOT, "models", "av_fused_latefusion_v2_60ep.pt"))


@torch.no_grad()
def main():
    device = torch.device("cuda")
    ck = torch.load(CKPT, weights_only=False)
    model = AVLateFusionReliabilityWordResNet(
        len(ck["label_to_idx"]), use_mid_gate=ck.get("use_mid_gate", False))
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device).eval()
    print(f"ckpt={os.path.basename(CKPT)}  best_val_acc(meta)={ck.get('best_val_acc'):.4f}"
          f"  video_head(meta)={ck.get('val_acc_video_at_best'):.4f}"
          f"  audio_head(meta)={ck.get('val_acc_audio_at_best'):.4f}"
          f"  w_a_mean(meta)={ck.get('w_a_mean_at_best'):.4f}", flush=True)

    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    val_idx = torch.load(os.path.join(ROOT, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    loader = DataLoader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=0),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=16,
                        pin_memory=True)

    LA, LV, W, Y = [], [], [], []
    for mel, vid, y in loader:
        m1 = mel.unsqueeze(1).to(device, non_blocking=True)
        vd = vid.to(device, non_blocking=True)
        fused, la, lv, w = model(m1, vd, return_parts=True)
        LA.append(la.float().cpu()); LV.append(lv.float().cpu())
        W.append(w.float().cpu()); Y.append(y)
    la = torch.cat(LA).numpy(); lv = torch.cat(LV).numpy()
    w = torch.cat(W).numpy(); y = torch.cat(Y).numpy()
    n = len(y)
    wa = w[:, 0:1]   # gate's actual per-sample audio weight
    print(f"clean val n={n}", flush=True)

    a_pred = la.argmax(1); v_pred = lv.argmax(1)
    a_ok = (a_pred == y); v_ok = (v_pred == y)
    # actual gate fused
    fused_actual = (wa * la + (1 - wa) * lv).argmax(1)
    actual_ok = (fused_actual == y)
    # equal sum (diagnostic, NOT convex)
    sum_ok = ((la + lv).argmax(1) == y)

    audio_acc = a_ok.mean(); video_acc = v_ok.mean()
    union = (a_ok | v_ok).mean()
    actual_acc = actual_ok.mean()
    sum_acc = sum_ok.mean()

    # convex oracle: per-sample, does ANY w in [0,1] give the correct argmax?
    grid = np.linspace(0.0, 1.0, 201)
    any_correct = np.zeros(n, dtype=bool)
    best_global = -1.0; best_global_w = None
    for wv in grid:
        pred = (wv * la + (1.0 - wv) * lv).argmax(1)
        ok = (pred == y)
        any_correct |= ok
        ga = ok.mean()
        if ga > best_global:
            best_global = ga; best_global_w = wv
    convex_oracle = any_correct.mean()

    print("\n=== Q2 CLEAN-ACC CEILINGS (frozen v2 heads, convex reliability gate) ===",
          flush=True)
    print(f"  audio head acc            = {audio_acc:.4f}", flush=True)
    print(f"  video head acc            = {video_acc:.4f}", flush=True)
    print(f"  actual gate fused acc     = {actual_acc:.4f}   (meta {ck.get('best_val_acc'):.4f})",
          flush=True)
    print(f"  union (best hard-routing) = {union:.4f}", flush=True)
    print(f"  CONVEX-ORACLE CEILING     = {convex_oracle:.4f}   <-- topology ceiling, these heads",
          flush=True)
    print(f"  best GLOBAL fixed w_a     = {best_global:.4f}  at w_a={best_global_w:.3f}",
          flush=True)
    print(f"  equal-sum (la+lv) acc     = {sum_acc:.4f}   (diagnostic, non-convex)",
          flush=True)
    print(f"  gate actual w_a: mean={wa.mean():.4f} std={wa.std():.4f} "
          f"p10={np.percentile(wa,10):.3f} p50={np.percentile(wa,50):.3f} "
          f"p90={np.percentile(wa,90):.3f}", flush=True)
    gap = 0.95 - convex_oracle
    verdict = ("STRUCTURAL-vs-HEADS: 0.95 UNREACHABLE with these heads "
               "(deficit is head quality -> budget if heads still climbing)"
               if convex_oracle < 0.95 else
               "GATE-LIMITED: a perfect gate reaches 0.95 with these heads "
               "(actual<oracle = gate miscalibration)")
    print(f"\n  0.95 - convex_oracle = {gap:+.4f}  =>  {verdict}", flush=True)

    out = os.path.join(HERE, "D312_q2_gate_ceiling.txt")
    with open(out, "w") as f:
        f.write(f"ckpt={os.path.basename(CKPT)} n={n}\n")
        f.write(f"audio_acc={audio_acc:.4f}\nvideo_acc={video_acc:.4f}\n")
        f.write(f"actual_fused={actual_acc:.4f}\nunion={union:.4f}\n")
        f.write(f"convex_oracle={convex_oracle:.4f}\n")
        f.write(f"best_global_w_acc={best_global:.4f} at w_a={best_global_w:.3f}\n")
        f.write(f"equal_sum={sum_acc:.4f}\n")
        f.write(f"wa_mean={wa.mean():.4f} wa_std={wa.std():.4f}\n")
        f.write(f"0.95-convex_oracle={gap:+.4f}\nverdict={verdict}\n")
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
