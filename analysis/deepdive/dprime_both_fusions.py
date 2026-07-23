"""TEMP DEBUG INSTRUMENT (debugger, task #5) — faster, tighter equivalent of
E1c (dprime_precision_test) + E1d (dprime_precision_balanced).

Single sweep per design; per sigma evaluates A, AV-multiplicative
(av_fused.pt / AVWordResNet) and AV-additive (av_fused_additive.pt /
AVAdditiveWordResNet) on the SAME noisy-mel batch. Pair selection, sigma grid,
seed=0 and the d' formula are copied verbatim from the committed scripts; only
the fusion model varies, and both fusions see byte-identical inputs.

num_workers is raised for speed; results are independent of worker count because
the audio noise is np.random.default_rng(seed+idx) (per-idx deterministic) and
shuffle=False preserves order. Self-validates: the 'mult' column must reproduce
the committed E1c/E1d CSV.
"""
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (RawNoisyAVDataset, _load_models, _NoisyAudioView,
                            _forward_A, _forward_V, T_STRIDE, BATCH_SIZE,
                            SCRIPT_DIR)
from model_av_additive import AVAdditiveWordResNet

EPS = 1e-7
MIN_TRIALS = 12
N_PAIRS = 15
NW = 32   # workers: speed only — identical results (per-idx-seeded noise)

SIGMA_E1C = [0.0, 0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
             0.010, 0.012, 0.015, 0.020, 0.030, 0.050, 0.080]
SIGMA_E1D = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040,
             0.050, 0.060, 0.080, 0.100, 0.130, 0.170, 0.220]
DV_MIN = 0.5                       # E1c
DV_LO, DV_HI, DA_MARGIN, DA_HI = 0.8, 3.2, 0.6, 6.0   # E1d


def _logp(probs):
    return np.log(np.clip(probs, EPS, 1.0))


def _dprime_pair(logp, labels, i, j):   # verbatim
    mi, mj = (labels == i), (labels == j)
    if mi.sum() < MIN_TRIALS or mj.sum() < MIN_TRIALS:
        return np.nan
    llr = logp[:, j] - logp[:, i]
    di, dj = llr[mi], llr[mj]
    sd = np.sqrt((di.var(ddof=1) + dj.var(ddof=1)) / 2.0)
    if not np.isfinite(sd) or sd < 1e-9:
        return np.nan
    return float((dj.mean() - di.mean()) / sd)


def _loader(view):
    return DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NW, pin_memory=True)


@torch.no_grad()
def _probs_AV(model, loader, device):
    """model(audio, video) is identical to analyze_av_msi._forward_AV
    (real,real) — same submodule sequence, dropout is identity in eval."""
    out = []
    for mel, vid, _y in loader:
        m1 = mel.unsqueeze(1).to(device, non_blocking=True)
        vd = vid.to(device, non_blocking=True)
        out.append(model(m1, vd).softmax(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _combined(A, AVm, AVa, loader, device):
    ap, mp, pp, labs = [], [], [], []
    for mel, vid, y in loader:
        m1 = mel.unsqueeze(1).to(device, non_blocking=True)
        vd = vid.to(device, non_blocking=True)
        ap.append(A(m1).softmax(1).cpu().numpy())
        mp.append(AVm(m1, vd).softmax(1).cpu().numpy())
        pp.append(AVa(m1, vd).softmax(1).cpu().numpy())
        labs.append(y.numpy())
    return (np.concatenate(ap), np.concatenate(mp),
            np.concatenate(pp), np.concatenate(labs))


def _select_pairs(design, A, V, base, val_idx, device):
    cl = _loader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=0))
    a_pred, a_prob, labels = _forward_A(A, cl, device)
    v_pred, v_prob, _ = _forward_V(V, cl, device)
    a_logp0, v_logp = _logp(a_prob), _logp(v_prob)

    conf = {}
    src = a_pred if design == "e1c" else v_pred     # E1c: audio-confused; E1d: video-confused
    for t, p in zip(labels, src):
        if t != p:
            key = (min(t, p), max(t, p))
            conf[key] = conf.get(key, 0) + 1
    cand = sorted(conf.items(), key=lambda kv: -kv[1])

    pairs = []
    for (i, j), c in cand:
        if (labels == i).sum() < MIN_TRIALS or (labels == j).sum() < MIN_TRIALS:
            continue
        dV = _dprime_pair(v_logp, labels, i, j)
        dA = _dprime_pair(a_logp0, labels, i, j)
        if design == "e1c":
            if not np.isfinite(dV) or abs(dV) < DV_MIN or not np.isfinite(dA):
                continue
            pairs.append((i, j, abs(dV), abs(dA)))
        else:
            if not (np.isfinite(dV) and np.isfinite(dA)):
                continue
            dVa, dAa = abs(dV), abs(dA)
            if DV_LO <= dVa <= DV_HI and (dAa - dVa) >= DA_MARGIN and dAa <= DA_HI:
                pairs.append((i, j, dVa, dAa))
        if len(pairs) >= N_PAIRS:
            break
    pair_ids = [(i, j) for i, j, *_ in pairs]
    dV_pair = np.array([abs(_dprime_pair(v_logp, labels, i, j))
                        for i, j in pair_ids])
    return pair_ids, dV_pair


def run_design(design, models, AVa, base, val_idx, device):
    A, V, AVm = models["A"][0], models["V"][0], models["AV"][0]
    sigma = SIGMA_E1C if design == "e1c" else SIGMA_E1D
    pair_ids, dV_pair = _select_pairs(design, A, V, base, val_idx, device)
    dV_mean = float(np.nanmean(dV_pair))
    print(f"\n### {design.upper()}  n_pairs={len(pair_ids)}  dV_mean={dV_mean:.3f}",
          flush=True)
    print("sigma   dA     dV     dAV_m  dAV_a  gain_m gain_a", flush=True)

    rows = []
    for sg in sigma:
        loader = _loader(_NoisyAudioView(base, val_idx, sigma_mult=sg, seed=0))
        a_prob, m_prob, a2_prob, lab = _combined(A, AVm, AVa, loader, device)
        a_lp, m_lp, a2_lp = _logp(a_prob), _logp(m_prob), _logp(a2_prob)
        dA = np.array([abs(_dprime_pair(a_lp, lab, i, j)) for i, j in pair_ids])
        dM = np.array([abs(_dprime_pair(m_lp, lab, i, j)) for i, j in pair_ids])
        dAdd = np.array([abs(_dprime_pair(a2_lp, lab, i, j)) for i, j in pair_ids])
        best = np.maximum(dA, dV_pair)
        gM = float(np.nanmean(dM / best))
        gAdd = float(np.nanmean(dAdd / best))
        mdA, mdM, mdAdd = (float(np.nanmean(dA)), float(np.nanmean(dM)),
                           float(np.nanmean(dAdd)))
        rows.append((sg, mdA, dV_mean, mdM, mdAdd, gM, gAdd))
        print(f"{sg:6.4f} {mdA:5.2f}  {dV_mean:5.2f}  {mdM:5.2f}  {mdAdd:5.2f}  "
              f"{gM:5.3f}  {gAdd:5.3f}", flush=True)

    out = os.path.join(SCRIPT_DIR, "analysis", "deepdive",
                       f"D310_{design}_both_fusions.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_a", "dprime_A", "dprime_V",
                    "dprime_AV_mult", "dprime_AV_add",
                    "gain_over_best_mult", "gain_over_best_add"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}"] + [f"{x:.4f}" for x in r[1:]])
    print(f"[saved] {out}", flush=True)
    return rows


def main():
    device = torch.device("cuda")
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    val_idx = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    models = _load_models(device)

    ck = torch.load(os.path.join(SCRIPT_DIR, "models", "av_fused_additive.pt"),
                    weights_only=False)
    AVa = AVAdditiveWordResNet(len(ck["label_to_idx"]))
    AVa.load_state_dict(ck["model_state_dict"])     # strict
    AVa = AVa.to(device).eval()
    print(f"AV_mult  alpha={float(models['AV'][0].gate.alpha):.4f} "
          f"acc={models['AV'][1].get('best_val_acc')}", flush=True)
    print(f"AV_add   alpha={float(AVa.gate.alpha):.4f} "
          f"acc={ck.get('best_val_acc')}", flush=True)

    run_design("e1c", models, AVa, base, val_idx, device)
    run_design("e1d", models, AVa, base, val_idx, device)


if __name__ == "__main__":
    main()
