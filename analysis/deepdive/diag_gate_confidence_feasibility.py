#!/usr/bin/env python3
"""TEMP DEBUG (task #20/#22) — WHERE is the per-trial reliability signal the gate
needs to route on E1d? The gate currently reads the two PENULTS (detached). The
feasibility probe showed those barely encode 'which head is right' (disagreement
routing AUC 0.585; video-head-correct 0.661) — because a penult encodes WORD
IDENTITY, not confidence. The natural reliability signal is the LOGIT CONFIDENCE
(a confusable video head has low top1-top2 margin / high entropy). This probe
compares, on clean val, how well each FEATURE SET decodes the routing target:

  PEN   = [a_pen, v_pen]                          (256-d; what rel_gate reads now)
  CONF  = per-head {entropy, max-prob, top1-top2 margin} of softmax(logit)  (6-d)
  BOTH  = PEN + CONF

  decode-1: video-head-CORRECT            (all clean val)
  decode-2: 'audio is the right head'     (DISAGREEMENT trials = exactly one right)

If CONF >> PEN, the fix direction is: give the rel_gate the logit-confidence
features (candidate c), not just penults — the signal it needs exists, just not
in the vectors it currently reads.

READ-ONLY / CHEAP. No model training, no production edits.
Run: CUDA_VISIBLE_DEVICES=1 python analysis/deepdive/diag_gate_confidence_feasibility.py
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


def auc(X, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X, y, cv=4, scoring="roc_auc").mean()


def conf_feats(logit):
    """per-head reliability features: entropy, max-prob, top1-top2 margin."""
    z = logit - logit.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(1, keepdims=True)
    mx = p.max(1, keepdims=True)
    srt = np.sort(p, axis=1)
    margin = (srt[:, -1] - srt[:, -2]).reshape(-1, 1)
    return np.concatenate([ent, mx, margin], axis=1)


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
    print(f"ckpt={os.path.basename(dlf.LATE_CKPT)}", flush=True)

    pair_ids, _dV = dlf._select_pairs("e1d", A, V, base, val_idx, device)
    pair_classes = sorted({c for p in pair_ids for c in p})

    store = {}
    AVl.audio_gap.register_forward_hook(
        lambda m, i, o: store.__setitem__("a", o.flatten(1).detach().cpu().numpy()))
    AVl.visual_gap.register_forward_hook(
        lambda m, i, o: store.__setitem__("v", o.flatten(1).detach().cpu().numpy()))

    loader = DataLoader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=SEED),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=dlf.NW, pin_memory=True)

    apen, vpen, lga, lgv, ys = [], [], [], [], []
    with torch.no_grad():
        for mel, vid, y in loader:
            m1 = mel.unsqueeze(1).to(device, non_blocking=True)
            vd = vid.to(device, non_blocking=True)
            _logits, la, lv, _w = AVl(m1, vd, return_parts=True)
            apen.append(store["a"])
            vpen.append(store["v"])
            lga.append(la.cpu().numpy())
            lgv.append(lv.cpu().numpy())
            ys.append(y.numpy())
    apen, vpen = np.concatenate(apen), np.concatenate(vpen)
    lga, lgv = np.concatenate(lga), np.concatenate(lgv)
    ys = np.concatenate(ys)
    vcorr = (lgv.argmax(1) == ys).astype(int)
    acorr = (lga.argmax(1) == ys).astype(int)
    e1d = np.isin(ys, pair_classes)

    PEN = np.concatenate([apen, vpen], axis=1)
    CONF = np.concatenate([conf_feats(lga), conf_feats(lgv)], axis=1)
    BOTH = np.concatenate([PEN, CONF], axis=1)

    print(f"\nclean val n={len(ys)}  vhead_acc={acorr.mean():.3f}(audio) "
          f"{vcorr.mean():.3f}(video)   E1d-class video_acc={vcorr[e1d].mean():.3f}",
          flush=True)

    print("\n[decode-1] video-head-CORRECT (all clean val):", flush=True)
    for name, X in [("PEN", PEN), ("CONF", CONF), ("BOTH", BOTH)]:
        print(f"    {name:>5}: CV ROC-AUC = {auc(X, vcorr):.3f}", flush=True)

    dis = (vcorr != acorr)
    print(f"\n[decode-2] 'audio is the right head' on DISAGREEMENT (n={int(dis.sum())}):",
          flush=True)
    for name, X in [("PEN", PEN), ("CONF", CONF), ("BOTH", BOTH)]:
        print(f"    {name:>5}: CV ROC-AUC = {auc(X[dis], acorr[dis]):.3f}", flush=True)

    # the E1d-specific routing question: among E1d-pair trials, decode video-wrong
    print(f"\n[decode-3] video-head-WRONG among E1d-pair trials "
          f"(n={int(e1d.sum())}, video_acc={vcorr[e1d].mean():.3f}):", flush=True)
    yv = (vcorr[e1d] == 0).astype(int)
    for name, X in [("PEN", PEN[e1d]), ("CONF", CONF[e1d]), ("BOTH", BOTH[e1d])]:
        print(f"    {name:>5}: CV ROC-AUC = {auc(X, yv):.3f}", flush=True)
    print("CONF_RC=0", flush=True)


if __name__ == "__main__":
    main()
