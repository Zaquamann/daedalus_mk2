#!/usr/bin/env python3
"""TEMP DEBUG (task #19) — diagnose the two E1d residuals (FLAG#1 AV<best-single at
low sigma; FLAG#2 sigma-flat / 'superoptimal' d'_AV) as REAL gate dilution vs the
D311 cross-observer artefact.

READ-ONLY / CHEAP. No training, no production edits. Reuses the canonical harness
helpers (pair selection, d' formula) verbatim from dprime_latefusion.py; only adds
(a) a readout of the actual reliability-gate weight w_a, and (b) a FORCED-w_a sweep
that overrides ONLY the convex gate weight (w_a*logit_a + (1-w_a)*logit_v) and
recomputes pairwise d' — the single-variable causal test for 'is the gate weight
the cause of the E1d dilution?'.

Predictions:
  - If the gate is a near-CONSTANT video-leaning weight (cause candidate):
    actual w_a ~ 0.37 with small std on E1d audio-strong trials at ALL sigma;
    and forcing w_a=1 (pure audio) lifts E1d d' from ~3.4 to ~4.5 (gL->~1, floor
    recovered) while forcing w_a=1 on E1c DROPS d' below the video floor (breaks
    the locked bar) -> proves the E1c/E1d tension is a single constant weight.
  - Video-dead probe AVl(m1, None): w_a should -> ~1 (gate routes to audio when
    video is zeroed) -> confirms dA_fus_late is the pure audio channel.

Run: CUDA_VISIBLE_DEVICES=1 python analysis/deepdive/diag_gate_e1d_readout.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import dprime_latefusion as dlf  # noqa: E402
from analyze_av_msi import _NoisyAudioView, BATCH_SIZE  # noqa: E402

dlf.NW = 6
FORCED_WA = [0.0, 0.25, 0.375, 0.5, 0.75, 1.0]
SIGMAS = {"e1d": [0.0, 0.04, 0.22], "e1c": [0.0, 0.05]}


def _stats(x):
    return (f"mean={x.mean():.3f} std={x.std():.3f} min={x.min():.3f} "
            f"p10={np.percentile(x, 10):.3f} p50={np.percentile(x, 50):.3f} "
            f"p90={np.percentile(x, 90):.3f} max={x.max():.3f}")


def _dp_pairs(prob, lab, pair_ids):
    lp = dlf._logp(prob)
    return np.array([abs(dlf._dprime_pair(lp, lab, i, j)) for i, j in pair_ids])


@torch.no_grad()
def _collect(AVl, A, loader, device):
    """One pass: standalone-audio prob, late logit_a/logit_v, actual gate w,
    and the video-dead gate w_a. All on the SAME noisy batch."""
    la, lv, ww, ap, wdead, labs = [], [], [], [], [], []
    for mel, vid, y in loader:
        m1 = mel.unsqueeze(1).to(device, non_blocking=True)
        vd = vid.to(device, non_blocking=True)
        _, lo_a, lo_v, w = AVl(m1, vd, return_parts=True)
        _, _, _, wd = AVl(m1, None, return_parts=True)   # video dead
        la.append(lo_a.cpu().numpy())
        lv.append(lo_v.cpu().numpy())
        ww.append(w[:, 0].cpu().numpy())
        wdead.append(wd[:, 0].cpu().numpy())
        ap.append(A(m1).softmax(1).cpu().numpy())
        labs.append(y.numpy())
    cat = np.concatenate
    return cat(la), cat(lv), cat(ww), cat(wdead), cat(ap), cat(labs)


def _softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run(design, AVl, A, V, base, val_idx, device):
    pair_ids, dV_pair = dlf._select_pairs(design, A, V, base, val_idx, device)
    pair_classes = sorted({c for p in pair_ids for c in p})
    dV_mean = float(np.nanmean(dV_pair))
    print(f"\n========== {design.upper()}  n_pairs={len(pair_ids)} "
          f"n_pair_classes={len(pair_classes)}  dV_mean={dV_mean:.3f} ==========",
          flush=True)
    for sg in SIGMAS[design]:
        loader = DataLoader(_NoisyAudioView(base, val_idx, sigma_mult=sg, seed=0),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=dlf.NW, pin_memory=True)
        la, lv, wa, wdead, ap, lab = _collect(AVl, A, loader, device)
        dA_std = _dp_pairs(ap, lab, pair_ids)
        best = float(np.nanmean(np.maximum(dA_std, dV_pair)))
        mask = np.isin(lab, pair_classes)
        print(f"\n--- sigma={sg:.3f}  dA_std={np.nanmean(dA_std):.2f}  "
              f"dV={dV_mean:.2f}  best_single={best:.2f} ---", flush=True)
        print(f"  ACTUAL gate w_a (all val n={len(wa)}):        {_stats(wa)}",
              flush=True)
        print(f"  ACTUAL gate w_a (E1d/E1c-pair trials n={int(mask.sum())}): "
              f"{_stats(wa[mask])}", flush=True)
        print(f"  VIDEO-DEAD probe w_a (all val):              {_stats(wdead)}",
              flush=True)
        # forced-w_a sweep: override ONLY the convex weight
        print("  forced w_a -> d'_AV (pairs)  | gL=d'/best_single", flush=True)
        for w_force in FORCED_WA:
            fused = w_force * la + (1.0 - w_force) * lv
            d = float(np.nanmean(_dp_pairs(_softmax_np(fused), lab, pair_ids)))
            print(f"      w_a={w_force:5.3f}:  d'_AV={d:5.2f}   gL={d / best:5.3f}",
                  flush=True)
        # actual gate (per-sample w) for reference
        fused_act = wa[:, None] * la + (1.0 - wa)[:, None] * lv
        d_act = float(np.nanmean(_dp_pairs(_softmax_np(fused_act), lab, pair_ids)))
        print(f"      ACTUAL  :  d'_AV={d_act:5.2f}   gL={d_act / best:5.3f}",
              flush=True)


def main():
    device = torch.device("cuda")
    base = dlf.RawNoisyAVDataset(noise=False, t_stride=dlf.T_STRIDE,
                                 return_video=True)
    val_idx = torch.load(os.path.join(dlf.SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    models = dlf._load_models(device)
    A, V = models["A"][0], models["V"][0]
    ck = torch.load(dlf.LATE_CKPT, weights_only=False)
    AVl = dlf.AVLateFusionReliabilityWordResNet(
        len(ck["label_to_idx"]), use_mid_gate=ck.get("use_mid_gate", False))
    AVl.load_state_dict(ck["model_state_dict"])
    AVl = AVl.to(device).eval()
    print(f"ckpt={os.path.basename(dlf.LATE_CKPT)} acc={ck.get('best_val_acc')} "
          f"w_a_meta={ck.get('w_a_mean_at_best')}", flush=True)
    run("e1d", AVl, A, V, base, val_idx, device)
    run("e1c", AVl, A, V, base, val_idx, device)


if __name__ == "__main__":
    main()
