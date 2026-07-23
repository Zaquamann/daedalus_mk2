"""Test the colleague's MLE claim: multisensory benefit is maximal when the two
unisensory cues are EQUALLY reliable.

Method: video is held clean (video-only reliability fixed at 86.50%). Audio is
degraded through a FINE noise grid that brackets the point where audio-only
accuracy crosses video-only accuracy (= equal reliability). At each level we
measure A-only, AV, and the benefit of AV over the BEST single modality, and
check whether that benefit peaks at the crossover.

Reuses the exact eval machinery validated for E1 (same models, same noisy-audio
view, same pinned val split, seed=0) so the shared sigma points reproduce E1.
"""
import os, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (
    RawNoisyAVDataset, _load_models, _NoisyAudioView,
    _forward_A, _forward_AV, _accuracy, T_STRIDE, BATCH_SIZE, SCRIPT_DIR,
)

V_CLEAN = 0.864989  # video-only (fair) on clean video — fixed; audio noise can't touch video

# Fine grid bracketing the A==V crossover (E1: A=0.883 at 0.005, A=0.692 at 0.010)
SIGMA = [0.0, 0.002, 0.004, 0.005, 0.0055, 0.006, 0.0065, 0.007,
         0.0075, 0.008, 0.009, 0.010, 0.012, 0.015, 0.020, 0.030]

def main():
    device = torch.device("cuda")
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    val_idx = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    models = _load_models(device)
    print(f"val N={len(val_idx)}   V_only(clean)={V_CLEAN:.4f}\n")

    rows = []
    for sg in SIGMA:
        view = _NoisyAudioView(base, val_idx, sigma_mult=sg, seed=0)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        a_preds, _, labels = _forward_A(models["A"][0], loader, device)
        acc_a = _accuracy(a_preds, labels)
        av = _forward_AV(models["AV"][0], loader, device,
                         video_kind="real", audio_kind="real")
        acc_av = _accuracy(av["preds"], labels)
        best = max(acc_a, V_CLEAN)
        rows.append((sg, acc_a, V_CLEAN, acc_av,
                     acc_av - acc_a, acc_av - V_CLEAN, acc_av - best,
                     abs(acc_a - V_CLEAN)))
        print(f"sig={sg:6.4f}  A={acc_a:.4f}  V={V_CLEAN:.4f}  AV={acc_av:.4f}  "
              f"AV-A={acc_av-acc_a:+.4f}  AV-V={acc_av-V_CLEAN:+.4f}  "
              f"AV-best={acc_av-best:+.4f}  |A-V|={abs(acc_a-V_CLEAN):.4f}")

    out = os.path.join(SCRIPT_DIR, "analysis", "msi", "E1b_reliability_match_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_a", "A_acc", "V_acc", "AV_acc",
                    "AV_minus_A", "AV_minus_V", "AV_minus_best", "abs_A_minus_V"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}"] + [f"{x:.6f}" for x in r[1:]])

    # locate empirical equal-reliability point and the benefit peak
    eq = min(rows, key=lambda r: r[7])
    peak = max(rows, key=lambda r: r[6])
    print(f"\nEqual-reliability point (min |A-V|): sig={eq[0]:.4f}  "
          f"A={eq[1]:.4f}  AV-best={eq[6]:+.4f}")
    print(f"Benefit-over-best PEAK:              sig={peak[0]:.4f}  "
          f"A={peak[1]:.4f}  AV-best={peak[6]:+.4f}")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
