#!/usr/bin/env python3
"""TEMP DEBUG (task #25, D315) — WHY does candidate-(c) FAIL the E1d-clean-audio
property on the FINAL ep185 model? The ep185 forced-w_a sweep bifurcated the cause:
  - gate WEIGHT is the dominant lever: actual w_a=0.418 -> d'=3.71; forcing
    w_a=0.75 -> 4.30 (clean single-variable causal +0.59).
  - the fused audio head CAPS at 4.33 (w_a=1.0) < standalone A 4.61 (residual 0.28);
    the sweep peak IS at w_a=1.0, so the convex logit combination cannot exceed its
    own audio head -> even a perfect gate gives gL=0.938 < 1.0 at sigma0.
This probe proves the MECHANISM behind each (all read-only, ep185, clean val):
  T1 (case-b): is AVl's audio head degraded vs standalone A on E1d pairs in
      ACCURACY (scale-invariant) or only in CALIBRATION (acc equal, d' gap from
      softmax temperature)?
  T2 (case-a, signal PRESENT?): on E1d-pair trials, do the video-head CONFIDENCE
      features [entropy, max-prob, margin] separate video-WRONG from video-right,
      or is the video head OVERCONFIDENT-when-wrong (the signal the gate needs is
      absent)?
  T3 (case-a, signal USED?): does the ACTUAL gate w_a track video-wrongness on E1d
      (higher w_a when video is wrong), or does it ignore it (mapping never learned
      for the rare E1d regime)?

READ-ONLY / CHEAP. No training, no production edits.
Run: CUDA_VISIBLE_DEVICES=0 LATE_CKPT=models/av_fused_latefusion.pt \
     python analysis/deepdive/diag_e1d_gate_failure_D315.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

import dprime_latefusion as dlf  # noqa: E402
from analyze_av_msi import _NoisyAudioView, BATCH_SIZE  # noqa: E402

dlf.NW = 6
SEED = 0


def _smx(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _conf_np(logit):
    """per-head [entropy, max-prob, top1-top2 margin] from logits."""
    p = _smx(logit)
    ent = -(p * np.log(p + 1e-12)).sum(1, keepdims=True)
    srt = np.sort(p, 1)
    return np.concatenate([ent, srt[:, -1:], srt[:, -1:] - srt[:, -2:-1]], 1)


def auc(X, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X, y, cv=4, scoring="roc_auc").mean()


def _dp_pairs(prob, lab, pair_ids):
    lp = dlf._logp(prob)
    return np.array([abs(dlf._dprime_pair(lp, lab, i, j)) for i, j in pair_ids])


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
    print(f"ckpt={os.path.basename(dlf.LATE_CKPT)} acc={ck.get('best_val_acc')}",
          flush=True)

    pair_ids, _dV = dlf._select_pairs("e1d", A, V, base, val_idx, device)
    pair_classes = sorted({c for p in pair_ids for c in p})

    loader = DataLoader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=SEED),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=dlf.NW, pin_memory=True)
    LA, LV, WA, PA_std, ys = [], [], [], [], []
    with torch.no_grad():
        for mel, vid, y in loader:
            m1 = mel.unsqueeze(1).to(device, non_blocking=True)
            vd = vid.to(device, non_blocking=True)
            _, la, lv, w = AVl(m1, vd, return_parts=True)
            LA.append(la.cpu().numpy())
            LV.append(lv.cpu().numpy())
            WA.append(w[:, 0].cpu().numpy())
            PA_std.append(A(m1).softmax(1).cpu().numpy())
            ys.append(y.numpy())
    LA, LV = np.concatenate(LA), np.concatenate(LV)
    WA = np.concatenate(WA)
    PA = np.concatenate(PA_std)      # standalone A probs
    ys = np.concatenate(ys)
    e1d = np.isin(ys, pair_classes)

    PLA = _smx(LA)                   # late-fusion audio-head probs
    vcorr = (LV.argmax(1) == ys)
    a_late_corr = (LA.argmax(1) == ys)
    a_std_corr = (PA.argmax(1) == ys)

    print(f"\nE1d-pair trials n={int(e1d.sum())}  "
          f"(video-head acc on E1d={vcorr[e1d].mean():.3f})", flush=True)

    # ---- T1: case-b — is the late audio head degraded vs standalone A? ----
    dA_late = float(np.nanmean(_dp_pairs(PLA, ys, pair_ids)))
    dA_std = float(np.nanmean(_dp_pairs(PA, ys, pair_ids)))
    print("\n[T1 case-b: AVl audio head vs standalone A on E1d pairs]", flush=True)
    print(f"   d'   late-audio-head={dA_late:.2f}   standalone-A={dA_std:.2f}   "
          f"gap={dA_std - dA_late:.2f}", flush=True)
    print(f"   acc  late-audio-head={a_late_corr[e1d].mean():.3f}   "
          f"standalone-A={a_std_corr[e1d].mean():.3f}   "
          f"(E1d-class trials; scale-invariant)", flush=True)
    # temperature check: rescale late-audio logits to match A's logit scale
    sA = float(np.std(PA))  # not used for scale; do a direct temp-fit on d'
    for T in (0.5, 0.7, 1.0, 1.5, 2.0):
        dT = float(np.nanmean(_dp_pairs(_smx(LA / T), ys, pair_ids)))
        print(f"     temp T={T}: d'(late-audio)={dT:.2f}", flush=True)
    print("   -> acc gap => REPRESENTATION degraded; acc equal & a temp recovers d' "
          "=> CALIBRATION only", flush=True)

    # ---- T2: case-a — is the unreliability signal PRESENT in conf_v on E1d? ----
    CV = _conf_np(LV)
    yv_wrong = (~vcorr).astype(int)
    print("\n[T2 case-a: video unreliability signal PRESENT in conf_v on E1d?]",
          flush=True)
    a_e1d = auc(CV[e1d], yv_wrong[e1d])
    print(f"   AUC(conf_v -> video-WRONG) on E1d trials = {a_e1d:.3f}", flush=True)
    for k, name in enumerate(["entropy", "max-prob", "margin"]):
        wm = CV[e1d & ~vcorr, k].mean()
        rm = CV[e1d & vcorr, k].mean()
        print(f"     {name:>8}: WRONG={wm:.3f}  right={rm:.3f}", flush=True)
    print("   -> high max-prob / low entropy when WRONG => overconfident-when-wrong "
          "(signal weak/absent)", flush=True)

    # ---- T3: case-a — does the ACTUAL gate w_a USE it on E1d? ----
    print("\n[T3 case-a: does ACTUAL gate w_a track video-wrongness on E1d?]",
          flush=True)
    ww = WA[e1d & ~vcorr].mean()
    wr = WA[e1d & vcorr].mean()
    print(f"   actual w_a | video-WRONG (E1d) = {ww:.3f}", flush=True)
    print(f"   actual w_a | video-right (E1d) = {wr:.3f}", flush=True)
    print(f"   delta = {ww - wr:+.3f}   (optimal: push w_a->~0.75-1.0 when video "
          f"wrong; ~0 delta => gate ignores it)", flush=True)
    # contrast: video-head overconfidence E1d (audio-strong) vs E1c (video-strong)
    pair_ids_c, _ = dlf._select_pairs("e1c", A, V, base, val_idx, device)
    pc = sorted({c for p in pair_ids_c for c in p})
    e1c = np.isin(ys, pc)
    print(f"\n[contrast] video-head max-prob when WRONG: "
          f"E1d={CV[e1d & ~vcorr, 1].mean():.3f}  E1c={CV[e1c & ~vcorr, 1].mean():.3f}"
          f"  (overconfident-when-wrong if high)", flush=True)
    print("D315_RC=0", flush=True)


if __name__ == "__main__":
    main()
