"""TEMP DEBUG INSTRUMENT (debugger) — behaviour-3 benchmark-artefact check.

E1d "super-optimality" (observed d'_AV > sqrt(d'_A^2 + d'_V^2), AV/pred up to
~1.14) is suspected to be a MEASUREMENT artefact: the sqrt-sum benchmark pairs
d'_A from the STANDALONE audio net with d'_AV from the FUSED net. If the fused
net's own audio pathway is a better audio detector than the standalone net,
sqrt(d'_A_standalone^2 + d'_V^2) UNDER-estimates the fair benchmark and the
"super-optimality" is spurious.

Test: across the E1d sigma grid, recompute d'_A from the FUSED model's
audio-only-via-AV readout (video zeroed: _forward_AV(video_kind="zero",
audio_kind="real")) and compare to the standalone d'_A used in the benchmark.
Verdict CONFIRMED (artefact) if d'_A_fused > d'_A_standalone where AV/pred>1, and
using d'_A_fused brings AV/pred down toward/below 1.0.

Same E1d pair selection / sigma grid / seed=0 / d' formula as
dprime_precision_balanced.py. Run anywhere with the models+data (hardware-robust:
d' is an aggregate; cross-GPU fp drift is negligible and the pod run already
reproduced the committed CSVs).
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (RawNoisyAVDataset, _load_models, _NoisyAudioView,
                            _forward_A, _forward_V, _forward_AV,
                            T_STRIDE, BATCH_SIZE, SCRIPT_DIR)

EPS = 1e-7
MIN_TRIALS = 12
N_PAIRS = 15
DV_LO, DV_HI, DA_MARGIN, DA_HI = 0.8, 3.2, 0.6, 6.0
SIGMA = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040,
         0.050, 0.060, 0.080, 0.100, 0.130, 0.170, 0.220]
NW = 8


def _logp(p):
    return np.log(np.clip(p, EPS, 1.0))


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    val_idx = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    models = _load_models(device)
    A, V, AV = models["A"][0], models["V"][0], models["AV"][0]

    # ---- E1d pair selection (verbatim: video-confused, audio stronger) ----
    cl = _loader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=0))
    _, a_prob, labels = _forward_A(A, cl, device)
    v_pred, v_prob, _ = _forward_V(V, cl, device)
    a_logp0, v_logp = _logp(a_prob), _logp(v_prob)
    conf = {}
    for t, p in zip(labels, v_pred):
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
        if not (np.isfinite(dV) and np.isfinite(dA)):
            continue
        dV, dA = abs(dV), abs(dA)
        if DV_LO <= dV <= DV_HI and (dA - dV) >= DA_MARGIN and dA <= DA_HI:
            pairs.append((i, j, dV, dA))
        if len(pairs) >= N_PAIRS:
            break
    pair_ids = [(i, j) for i, j, *_ in pairs]
    dV_pair = np.array([abs(_dprime_pair(v_logp, labels, i, j)) for i, j in pair_ids])
    dV_mean = float(np.nanmean(dV_pair))
    print(f"n_pairs={len(pair_ids)}  dV_standalone_mean={dV_mean:.3f}", flush=True)
    print("sigma   dA_std dA_fus dV_std dV_fus  dAV   "
          "AVp_std AVp_within  gB_std gB_within", flush=True)

    rows = []
    for sg in SIGMA:
        ld = _loader(_NoisyAudioView(base, val_idx, sigma_mult=sg, seed=0))
        _, a_std_prob, lab = _forward_A(A, ld, device)                    # standalone audio
        fz = _forward_AV(AV, ld, device, video_kind="zero", audio_kind="real")  # fused audio-only (video zeroed)
        vz = _forward_AV(AV, ld, device, video_kind="real", audio_kind="zero")  # fused video-only (audio zeroed)
        av = _forward_AV(AV, ld, device, video_kind="real", audio_kind="real")  # full AV
        a_std_lp = _logp(a_std_prob)
        a_fz_lp = _logp(fz["probs"])
        v_fz_lp = _logp(vz["probs"])
        av_lp = _logp(av["probs"])

        dA_std = np.array([abs(_dprime_pair(a_std_lp, lab, i, j)) for i, j in pair_ids])
        dA_fz = np.array([abs(_dprime_pair(a_fz_lp, lab, i, j)) for i, j in pair_ids])
        dV_fz = np.array([abs(_dprime_pair(v_fz_lp, lab, i, j)) for i, j in pair_ids])
        dAV = np.array([abs(_dprime_pair(av_lp, lab, i, j)) for i, j in pair_ids])
        pred_std = np.sqrt(dA_std ** 2 + dV_pair ** 2)        # cross-observer benchmark (committed)
        pred_within = np.sqrt(dA_fz ** 2 + dV_fz ** 2)        # within-fused-model benchmark

        m = lambda x: float(np.nanmean(x))
        avp_std = m(dAV / pred_std)
        avp_within = m(dAV / pred_within)
        gb_std = m(dAV / np.maximum(dA_std, dV_pair))
        gb_within = m(dAV / np.maximum(dA_fz, dV_fz))
        rows.append((sg, m(dA_std), m(dA_fz), dV_mean, m(dV_fz), m(dAV),
                     avp_std, avp_within, gb_std, gb_within))
        print(f"{sg:6.4f} {m(dA_std):5.2f}  {m(dA_fz):5.2f}  {dV_mean:5.2f} "
              f"{m(dV_fz):5.2f}  {m(dAV):5.2f}  {avp_std:6.3f}  {avp_within:9.3f}  "
              f"{gb_std:6.3f}  {gb_within:9.3f}", flush=True)

    out = os.path.join(SCRIPT_DIR, "analysis", "deepdive",
                       "D310_behaviour3_artefact.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_a", "dprimeA_standalone", "dprimeA_fused_videozero",
                    "dprimeV_standalone", "dprimeV_fused_audiozero", "dprimeAV",
                    "AV_over_pred_crossobs", "AV_over_pred_withinmodel",
                    "gain_over_best_crossobs", "gain_over_best_withinmodel"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}"] + [f"{x:.4f}" for x in r[1:]])
    print(f"[saved] {out}", flush=True)

    # summary verdict aid
    dA_std = np.array([r[1] for r in rows]); dA_fz = np.array([r[2] for r in rows])
    dV_std = np.array([r[3] for r in rows]); dV_fz = np.array([r[4] for r in rows])
    print(f"\nsigmas dA_fused > dA_standalone: {int((dA_fz > dA_std).sum())}/{len(rows)}"
          f"  (artefact hypo predicts ALL)", flush=True)
    print(f"sigmas dV_fused > dV_standalone: {int((dV_fz > dV_std).sum())}/{len(rows)}", flush=True)
    print(f"fused video-zero dA range: {dA_fz.min():.2f}-{dA_fz.max():.2f}  "
          f"fused audio-zero dV range: {dV_fz.min():.2f}-{dV_fz.max():.2f}  "
          f"(standalone dA {dA_std.min():.2f}-{dA_std.max():.2f}, dV {dV_std[0]:.2f})", flush=True)


if __name__ == "__main__":
    main()
