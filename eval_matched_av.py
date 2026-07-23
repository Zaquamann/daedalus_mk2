"""Matched-reliability AV point: degrade BOTH audio and video so A-only and
V-only land at the same accuracy (~75%), then measure AV under joint noise.

sigma_a, sigma_v from D1_iso_perf_lookup.csv (target_acc=0.75). All three streams
measured on the SAME jointly-degraded inputs (one loader), pinned val set, seed 0.
Also reports A-only at sigma_a=0.06 to show what that (much larger) noise does.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (BATCH_SIZE, T_STRIDE, _load_models,
                            _forward_A, _forward_V, _forward_AV, _NoisyAudioView)
from analyze_av_deepdive import _NoisyAVView
from dataset_raw_noisy import RawNoisyAVDataset

HERE = os.path.dirname(os.path.abspath(__file__))
SIG_A, SIG_V = 0.008487, 0.212586        # iso-perf-75 (both -> ~75%)


def acc(p, y):
    return float((p == y).mean())


def loader(view):
    return DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)


def main():
    device = torch.device("cuda")
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    val_idx = torch.load(os.path.join(HERE, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    models = _load_models(device)

    # matched point: both streams noised, all models read the same loader
    ld = loader(_NoisyAVView(base, val_idx, sigma_a_mult=SIG_A,
                             sigma_v_mult=SIG_V, seed=0))
    aP, aprob, lab = _forward_A(models["A"][0], ld, device)   # A reads noisy audio
    vP, vprob, _ = _forward_V(models["V"][0], ld, device)     # V reads noisy video
    av = _forward_AV(models["AV"][0], ld, device, video_kind="real", audio_kind="real")
    A, V, AV = acc(aP, lab), acc(vP, lab), acc(av["preds"], lab)

    # pooled = combine the two unisensory OUTPUTS without the learned gate (late fusion).
    # AV(gate) - pooled = the genuine multisensory integration beyond statistical pooling.
    pooled_avg = acc((aprob + vprob).argmax(1), lab)                                  # 50/50 softmax avg
    pooled_bayes = acc((np.log(aprob + 1e-12) + np.log(vprob + 1e-12)).argmax(1), lab)  # indep-Bayes
    pooled = max(pooled_avg, pooled_bayes)                                            # conservative

    print("=== MATCHED-RELIABILITY POINT (both streams noised) ===")
    print(f"sigma_a={SIG_A}  sigma_v={SIG_V}  (N={len(lab)})")
    print(f"  A-only         : {A*100:6.3f}%")
    print(f"  V-only         : {V*100:6.3f}%")
    print(f"  pooled (avg)   : {pooled_avg*100:6.3f}%")
    print(f"  pooled (bayes) : {pooled_bayes*100:6.3f}%")
    print(f"  AV (gate)      : {AV*100:6.3f}%")
    print(f"  |A - V|        : {abs(A-V)*100:6.3f} pp   (matched check)")
    print(f"  genuine MSI = AV - pooled(best) : +{(AV-pooled)*100:6.3f} pp")
    print(f"  AV - best single                : +{(AV-max(A,V))*100:6.3f} pp")

    with open(os.path.join(HERE, "analysis", "msi", "E1e_matched_75.csv"), "w") as f:
        f.write("condition,sigma_a,sigma_v,A_acc,V_acc,AV_acc,"
                "pooled_avg,pooled_bayes,AV_minus_pooled\n")
        f.write(f"matched_75,{SIG_A},{SIG_V},{A:.6f},{V:.6f},{AV:.6f},"
                f"{pooled_avg:.6f},{pooled_bayes:.6f},{AV-pooled:.6f}\n")


if __name__ == "__main__":
    main()
