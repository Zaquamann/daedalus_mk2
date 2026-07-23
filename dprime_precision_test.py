"""Precision-currency version of the MLE multisensory test.

Accuracy hides the MLE law behind a ceiling. The law is stated in sensitivity
(d'), the signal-detection measure of how many standard deviations separate two
stimuli on the model's internal decision variable. For a confusable word pair
(i, j) the decision variable is the per-trial log-likelihood ratio

    LLR = log P(word j) - log P(word i)              (= logit_j - logit_i)

and sensitivity is the separation of that variable between the two true classes
in pooled-SD units:

    d' = ( mean LLR | true=j  -  mean LLR | true=i ) / sqrt((var_i + var_j)/2)

Optimal (Bayesian) integration of two independent cues predicts

    d'_AV = sqrt(d'_A^2 + d'_V^2)                     (quadratic summation)

so the RELATIVE sensitivity gain over the better single cue,

    d'_AV / max(d'_A, d'_V) = sqrt(1 + (d'_min/d'_max)^2),

is largest (-> sqrt(2) = 1.414) exactly when the two senses are EQUALLY
sensitive (d'_A == d'_V), and shrinks toward 1 when one sense dominates. That is
the colleague's claim, in the currency the proof is actually written in.

Method: video held clean (d'_V is a fixed per-pair constant; audio noise cannot
touch the video stream). Audio is degraded through a noise grid that slides d'_A
down through d'_V. Pairs are chosen data-drivenly: the word pairs the audio model
actually confuses (so audio sensitivity is finite, not at ceiling) AND on which
the video model still has real signal (so integration has something to combine).

Reuses the exact E1 eval machinery (same models, same noisy-audio view, same
pinned val split, seed=0).
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
MIN_TRIALS = 12          # per word, for a stable per-pair d'
N_PAIRS = 15             # how many confusable pairs to average over
DV_MIN = 0.5             # keep pairs where video has real signal (clean d'_V >= this)

# Audio-noise grid that brackets the d'_A == d'_V crossover.
SIGMA = [0.0, 0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
         0.010, 0.012, 0.015, 0.020, 0.030, 0.050, 0.080]


def _logp(probs):
    return np.log(np.clip(probs, EPS, 1.0))


def _dprime_pair(logp, labels, i, j):
    """d' for separating words i and j on the LLR decision variable."""
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

    # ---- clean pass: pick confusable pairs + fixed per-pair d'_V -------------
    clean = _NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=0)
    cl = DataLoader(clean, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)
    a_pred, a_prob, labels = _forward_A(models["A"][0], cl, device)
    _, v_prob, _ = _forward_V(models["V"][0], cl, device)
    a_logp0, v_logp = _logp(a_prob), _logp(v_prob)

    # audio confusion mass per unordered pair
    conf = {}
    for t, p in zip(labels, a_pred):
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
        if not np.isfinite(dV) or abs(dV) < DV_MIN or not np.isfinite(dA):
            continue
        pairs.append((i, j, abs(dV), abs(dA)))
        if len(pairs) >= N_PAIRS:
            break

    print(f"val N={len(labels)}   selected {len(pairs)} confusable pairs "
          f"(audio-confusable, video d'>= {DV_MIN}):")
    for i, j, dV, dA in pairs:
        print(f"  {idx_to_label[i]:>14s} / {idx_to_label[j]:<14s}  "
              f"clean d'_A={dA:5.2f}  d'_V={dV:5.2f}  conf={conf[(i,j)]}")
    pair_ids = [(i, j) for i, j, *_ in pairs]

    # fixed video sensitivity per pair (unsigned magnitude)
    dV_pair = np.array([abs(_dprime_pair(v_logp, labels, i, j))
                        for i, j in pair_ids])
    dV_mean = float(np.nanmean(dV_pair))

    # ---- sweep audio noise --------------------------------------------------
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
        pred = np.sqrt(dA ** 2 + dV_pair ** 2)            # optimal prediction
        gain = dAV / np.maximum(dA, dV_pair)              # over best single cue
        gain_opt = pred / np.maximum(dA, dV_pair)

        m = lambda x: float(np.nanmean(x))
        rows.append((sg, m(dA), dV_mean, m(dAV), m(pred),
                     m(dAV) / m(pred) if m(pred) else np.nan,
                     m(gain), m(gain_opt), m(np.abs(dA - dV_pair))))
        print(f"sig={sg:6.4f}  d'_A={m(dA):5.2f}  d'_V={dV_mean:5.2f}  "
              f"d'_AV={m(dAV):5.2f}  pred=sqrt(A^2+V^2)={m(pred):5.2f}  "
              f"AV/pred={m(dAV)/m(pred) if m(pred) else float('nan'):4.2f}  "
              f"gain={m(gain):4.2f} (opt {m(gain_opt):4.2f})  "
              f"|A-V|={m(np.abs(dA-dV_pair)):4.2f}")

    out = os.path.join(SCRIPT_DIR, "analysis", "msi", "E1c_dprime_precision_sweep.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_a", "dprime_A", "dprime_V", "dprime_AV",
                    "dprime_pred_opt", "AV_over_pred", "gain_over_best",
                    "gain_opt", "abs_dA_minus_dV"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}"] + [f"{x:.4f}" for x in r[1:]])

    cross = min(rows, key=lambda r: r[8])     # min |d'_A - d'_V|
    gpeak = max(rows, key=lambda r: r[6])     # max relative gain
    print(f"\nEqual-sensitivity crossover (min |d'_A-d'_V|): "
          f"sig={cross[0]:.4f}  d'_A={cross[1]:.2f} d'_V={cross[2]:.2f}  "
          f"gain={cross[6]:.3f}  (optimal would be {cross[7]:.3f})")
    print(f"Relative-gain PEAK: sig={gpeak[0]:.4f}  d'_A={gpeak[1]:.2f} "
          f"d'_V={gpeak[2]:.2f}  gain={gpeak[6]:.3f}")
    obs = np.array([r[3] for r in rows]); prd = np.array([r[4] for r in rows])
    print(f"Quadratic-summation fit across sweep: mean observed d'_AV / "
          f"predicted = {float(np.mean(obs/prd)):.3f}  "
          f"(1.0 = optimal, <1 = under-integration)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
