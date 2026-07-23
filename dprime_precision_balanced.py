"""Precision-currency MLE test, v2 — designed to BRACKET equal sensitivity.

v1 (dprime_precision_test.py) selected audio-confusable pairs, on which video is
the stronger cue (d'_V > d'_A); adding audio noise only pushed the two senses
further apart, so it sampled only the falling side of the gain curve and could
not test the "max benefit at equal sensitivity" prediction.

v2 selects pairs where AUDIO is the stronger but degradable cue (clean
d'_A > d'_V, both informative). Sweeping audio noise then slides d'_A DOWN
THROUGH d'_V, bracketing the equal-sensitivity point. The MLE prediction:

    relative gain  d'_AV / max(d'_A, d'_V) = sqrt(1 + (d'_min/d'_max)^2)

peaks (-> sqrt(2)=1.414 for an optimal integrator; k*sqrt(2) for one that reaches
a constant fraction k of the optimal quadratic sum) exactly where d'_A == d'_V.
A peak at the crossover confirms the colleague's claim; its height vs sqrt(2)
measures how close the integrator is to Bayesian-optimal.

Video held clean (d'_V is a fixed per-pair constant). Same E1 machinery.
"""
import os, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (
    RawNoisyAVDataset, _load_models, _NoisyAudioView,
    _forward_A, _forward_V, _forward_AV,
    T_STRIDE, BATCH_SIZE, SCRIPT_DIR,
)

EPS = 1e-7
MIN_TRIALS = 12
N_PAIRS = 15
# keep pairs where BOTH cues are informative AND audio starts clearly stronger,
# so an audio-noise sweep brackets the d'_A == d'_V crossover:
DV_LO, DV_HI = 0.8, 3.2      # video informative but not saturating
DA_MARGIN = 0.6              # clean d'_A must exceed d'_V by at least this
DA_HI = 6.0

SIGMA = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040,
         0.050, 0.060, 0.080, 0.100, 0.130, 0.170, 0.220]


def _logp(probs):
    return np.log(np.clip(probs, EPS, 1.0))


def _dprime_pair(logp, labels, i, j):
    mi, mj = (labels == i), (labels == j)
    if mi.sum() < MIN_TRIALS or mj.sum() < MIN_TRIALS:
        return np.nan
    llr = logp[:, j] - logp[:, i]
    di, dj = llr[mi], llr[mj]
    sd = np.sqrt((di.var(ddof=1) + dj.var(ddof=1)) / 2.0)
    if not np.isfinite(sd) or sd < 1e-9:
        return np.nan
    return float((dj.mean() - di.mean()) / sd)


def main():
    device = torch.device("cuda")
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    val_idx = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    models = _load_models(device)
    ck = models["A"][1]
    idx_to_label = ck.get("idx_to_label") or {v: k for k, v in ck["label_to_idx"].items()}

    clean = _NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=0)
    cl = DataLoader(clean, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)
    _, a_prob, labels = _forward_A(models["A"][0], cl, device)
    v_pred, v_prob, _ = _forward_V(models["V"][0], cl, device)
    a_logp0, v_logp = _logp(a_prob), _logp(v_prob)

    # candidate pairs: those the VIDEO model confuses (video weaker there) so that
    # audio can be the stronger cue; ranked by video-confusion mass.
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

    print(f"val N={len(labels)}   selected {len(pairs)} pairs "
          f"(audio stronger, brackets equal sensitivity):")
    for i, j, dV, dA in pairs:
        print(f"  {idx_to_label[i]:>14s} / {idx_to_label[j]:<14s}  "
              f"clean d'_A={dA:5.2f}  d'_V={dV:5.2f}")
    pair_ids = [(i, j) for i, j, *_ in pairs]
    dV_pair = np.array([abs(_dprime_pair(v_logp, labels, i, j)) for i, j in pair_ids])
    dV_mean = float(np.nanmean(dV_pair))

    rows = []
    for sg in SIGMA:
        view = _NoisyAudioView(base, val_idx, sigma_mult=sg, seed=0)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        _, a_prob, lab = _forward_A(models["A"][0], loader, device)
        av = _forward_AV(models["AV"][0], loader, device,
                         video_kind="real", audio_kind="real")
        a_lp, av_lp = _logp(a_prob), _logp(av["probs"])
        dA = np.array([abs(_dprime_pair(a_lp, lab, i, j)) for i, j in pair_ids])
        dAV = np.array([abs(_dprime_pair(av_lp, lab, i, j)) for i, j in pair_ids])
        pred = np.sqrt(dA ** 2 + dV_pair ** 2)
        gain = dAV / np.maximum(dA, dV_pair)
        gain_opt = pred / np.maximum(dA, dV_pair)
        m = lambda x: float(np.nanmean(x))
        rows.append((sg, m(dA), dV_mean, m(dAV), m(pred),
                     m(dAV) / m(pred) if m(pred) else np.nan,
                     m(gain), m(gain_opt), m(np.abs(dA - dV_pair))))
        print(f"sig={sg:6.4f}  d'_A={m(dA):5.2f}  d'_V={dV_mean:5.2f}  "
              f"d'_AV={m(dAV):5.2f}  pred={m(pred):5.2f}  AV/pred="
              f"{m(dAV)/m(pred) if m(pred) else float('nan'):4.2f}  "
              f"gain={m(gain):4.2f} (opt {m(gain_opt):4.2f})  "
              f"|A-V|={m(np.abs(dA-dV_pair)):4.2f}")

    out = os.path.join(SCRIPT_DIR, "analysis", "msi",
                       "E1d_dprime_precision_balanced.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_a", "dprime_A", "dprime_V", "dprime_AV",
                    "dprime_pred_opt", "AV_over_pred", "gain_over_best",
                    "gain_opt", "abs_dA_minus_dV"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}"] + [f"{x:.4f}" for x in r[1:]])

    cross = min(rows, key=lambda r: r[8])
    gpeak = max(rows, key=lambda r: r[6])
    print(f"\nEqual-sensitivity crossover (min |d'_A-d'_V|): sig={cross[0]:.4f}  "
          f"d'_A={cross[1]:.2f} d'_V={cross[2]:.2f}  gain={cross[6]:.3f}  "
          f"(optimal sqrt2={cross[7]:.3f})")
    print(f"Relative-gain PEAK: sig={gpeak[0]:.4f}  d'_A={gpeak[1]:.2f} "
          f"d'_V={gpeak[2]:.2f}  gain={gpeak[6]:.3f}  |A-V|={gpeak[8]:.2f}")
    obs = np.array([r[3] for r in rows]); prd = np.array([r[4] for r in rows])
    print(f"Quadratic-summation fit: mean observed d'_AV / predicted = "
          f"{float(np.mean(obs/prd)):.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
