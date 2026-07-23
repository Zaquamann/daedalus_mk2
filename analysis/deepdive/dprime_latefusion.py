"""Eval the parallel/late-fusion reliability model (task #7 fix) through the
SAME d' harness the debugger used for the committed fusions (dprime_both_fusions.py).

Per design (E1c / E1d) and per sigma it evaluates, on the SAME noisy-mel batch:
  - A           (audio-only WordResNet, committed)
  - AV-mult     (av_fused.pt / AVWordResNet, committed)  -> self-validation column
  - AV-late     (av_fused_latefusion.pt / AVLateFusionReliabilityWordResNet, the fix)

Pair selection, sigma grids, seed=0 and the d' formula are copied verbatim from
dprime_both_fusions.py so the AV-mult column reproduces the committed E1c/E1d
result and the AV-late column is directly comparable. The fix is judged by
gain_over_best_late = mean(d'_late / max(d'_A, d'_V)) >= ~1.0 across the grid
(floor held at d'_V when audio dies) without regressing the E1d crossover (~sqrt2).

Run (pod, repo root): python analysis/deepdive/dprime_latefusion.py
Optional: LATE_CKPT=models/av_fused_latefusion_ep40.pt  (eval a mid-checkpoint).
"""
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from analyze_av_msi import (RawNoisyAVDataset, _load_models, _NoisyAudioView,
                            _forward_A, _forward_V, T_STRIDE, BATCH_SIZE,
                            SCRIPT_DIR)
from model_av_latefusion import AVLateFusionReliabilityWordResNet

EPS = 1e-7
MIN_TRIALS = 12
N_PAIRS = 15
NW = 32

SIGMA_E1C = [0.0, 0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
             0.010, 0.012, 0.015, 0.020, 0.030, 0.050, 0.080]
SIGMA_E1D = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040,
             0.050, 0.060, 0.080, 0.100, 0.130, 0.170, 0.220]
DV_MIN = 0.5
DV_LO, DV_HI, DA_MARGIN, DA_HI = 0.8, 3.2, 0.6, 6.0

LATE_CKPT = os.environ.get("LATE_CKPT",
                           os.path.join(SCRIPT_DIR, "models",
                                        "av_fused_latefusion.pt"))


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
def _combined(A, AVm, AVl, loader, device):
    """Per batch, on byte-identical inputs: standalone A; mult fused; late fused;
    and ABLATION-SURVIVAL readouts for both fusions —
      *_af = audio channel surviving (video zeroed via video=None -> v_mid=0),
      *_vf = video channel surviving (audio zeroed via a zero mel input).
    Both zeroings match analyze_av_msi._forward_AV's video_kind/audio_kind="zero".
    """
    ap, mp, lp = [], [], []
    laf, lvf, maf, mvf = [], [], [], []
    labs = []
    for mel, vid, y in loader:
        m1 = mel.unsqueeze(1).to(device, non_blocking=True)
        vd = vid.to(device, non_blocking=True)
        z = torch.zeros_like(m1)
        ap.append(A(m1).softmax(1).cpu().numpy())
        mp.append(AVm(m1, vd).softmax(1).cpu().numpy())
        lp.append(AVl(m1, vd).softmax(1).cpu().numpy())
        laf.append(AVl(m1, None).softmax(1).cpu().numpy())   # late, video dead
        lvf.append(AVl(z, vd).softmax(1).cpu().numpy())       # late, audio dead
        maf.append(AVm(m1, None).softmax(1).cpu().numpy())   # mult, video dead
        mvf.append(AVm(z, vd).softmax(1).cpu().numpy())       # mult, audio dead
        labs.append(y.numpy())
    cat = np.concatenate
    return (cat(ap), cat(mp), cat(lp), cat(laf), cat(lvf),
            cat(maf), cat(mvf), cat(labs))


def _select_pairs(design, A, V, base, val_idx, device):   # verbatim
    cl = _loader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=0))
    a_pred, a_prob, labels = _forward_A(A, cl, device)
    v_pred, v_prob, _ = _forward_V(V, cl, device)
    a_logp0, v_logp = _logp(a_prob), _logp(v_prob)

    conf = {}
    src = a_pred if design == "e1c" else v_pred
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


def run_design(design, models, AVl, base, val_idx, device):
    A, V, AVm = models["A"][0], models["V"][0], models["AV"][0]
    sigma = SIGMA_E1C if design == "e1c" else SIGMA_E1D
    pair_ids, dV_pair = _select_pairs(design, A, V, base, val_idx, device)
    dV_mean = float(np.nanmean(dV_pair))
    print(f"\n### {design.upper()}  n_pairs={len(pair_ids)}  dV_mean={dV_mean:.3f}",
          flush=True)
    print("BAR(i) gain_over_best>=~1 ; BAR(ii) ablation survival: "
          "dA_fus~dA_std & dV_fus~dV_std (late survives, mult collapses)",
          flush=True)
    print("sigma   dA_std dV_std dAVm  dAVl  gM    gL   | dAfL  dVfL  dAfM  dVfM",
          flush=True)

    def _dp_pairs(prob, lab):
        lp = _logp(prob)
        return np.array([abs(_dprime_pair(lp, lab, i, j)) for i, j in pair_ids])

    rows = []
    for sg in sigma:
        loader = _loader(_NoisyAudioView(base, val_idx, sigma_mult=sg, seed=0))
        a_prob, m_prob, l_prob, laf, lvf, maf, mvf, lab = _combined(
            A, AVm, AVl, loader, device)
        dA = _dp_pairs(a_prob, lab)
        dM = _dp_pairs(m_prob, lab)
        dL = _dp_pairs(l_prob, lab)
        dAfL, dVfL = _dp_pairs(laf, lab), _dp_pairs(lvf, lab)   # late ablations
        dAfM, dVfM = _dp_pairs(maf, lab), _dp_pairs(mvf, lab)   # mult ablations
        best = np.maximum(dA, dV_pair)
        gM = float(np.nanmean(dM / best))
        gL = float(np.nanmean(dL / best))
        nm = lambda x: float(np.nanmean(x))
        mdA, mdM, mdL = nm(dA), nm(dM), nm(dL)
        mAfL, mVfL, mAfM, mVfM = nm(dAfL), nm(dVfL), nm(dAfM), nm(dVfM)
        rows.append((sg, mdA, dV_mean, mdM, mdL, gM, gL,
                     mAfL, mVfL, mAfM, mVfM))
        print(f"{sg:6.4f} {mdA:5.2f}  {dV_mean:5.2f}  {mdM:5.2f}  {mdL:5.2f}  "
              f"{gM:5.3f} {gL:5.3f} | {mAfL:5.2f} {mVfL:5.2f}  {mAfM:5.2f} "
              f"{mVfM:5.2f}", flush=True)

    # Survival summary (BAR ii): channel d' under ablation vs standalone, averaged
    # over the grid. Late should be ~1.0 (survives); mult << 1 (collapses).
    arr = np.array(rows, dtype=float)
    dA_std_col, dAfL_col, dAfM_col = arr[:, 1], arr[:, 7], arr[:, 9]
    dVfL_col, dVfM_col = arr[:, 8], arr[:, 10]
    eps = 1e-9
    print(f"[survival] late: dA_fus/dA_std={np.nanmean(dAfL_col/(dA_std_col+eps)):.3f}"
          f"  dV_fus/dV_std={np.nanmean(dVfL_col/(dV_mean+eps)):.3f}"
          f"  || mult: dA_fus/dA_std={np.nanmean(dAfM_col/(dA_std_col+eps)):.3f}"
          f"  dV_fus/dV_std={np.nanmean(dVfM_col/(dV_mean+eps)):.3f}", flush=True)

    out = os.path.join(SCRIPT_DIR, "analysis", "deepdive",
                       f"D310_{design}_latefusion.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_a", "dprime_A_std", "dprime_V_std",
                    "dprime_AV_mult", "dprime_AV_late",
                    "gain_over_best_mult", "gain_over_best_late",
                    "dA_fus_late", "dV_fus_late",
                    "dA_fus_mult", "dV_fus_mult"])
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

    ck = torch.load(LATE_CKPT, weights_only=False)
    AVl = AVLateFusionReliabilityWordResNet(
        len(ck["label_to_idx"]), use_mid_gate=ck.get("use_mid_gate", False))
    AVl.load_state_dict(ck["model_state_dict"])       # strict
    AVl = AVl.to(device).eval()
    print(f"AV_mult  alpha={float(models['AV'][0].gate.alpha):.4f} "
          f"acc={models['AV'][1].get('best_val_acc')}", flush=True)
    print(f"AV_late  ckpt={os.path.basename(LATE_CKPT)} "
          f"acc={ck.get('best_val_acc')} "
          f"vid_head={ck.get('val_acc_video_at_best')} "
          f"w_a={ck.get('w_a_mean_at_best')} noise={ck.get('noise_range')}",
          flush=True)

    run_design("e1c", models, AVl, base, val_idx, device)
    run_design("e1d", models, AVl, base, val_idx, device)


if __name__ == "__main__":
    main()
