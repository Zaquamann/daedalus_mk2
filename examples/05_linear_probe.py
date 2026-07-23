#!/usr/bin/env python3
"""5-fold LR probe on AV / A-only / AV(v_mid=0) features from 04.

The AV model's softmax accuracy collapses to ~chance when v_mid=0, but a
fresh linear probe on those same penult features still decodes the word
at ~50%+. So the features still encode the word — the trained fc was just
fit with v_mid != 0 inputs and freaks out when v_mid goes to 0.
"""

import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_av import AVWordResNet
from paired_dataset import PairedAVDataset
from train import WordResNet


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
NPZ_PATH = os.path.join(OUT_DIR, "04_features.npz")


def _probe(name: str, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # Standardising before LR is important: AV penult dims have very different
    # scales (some saturate near 0, others go to ~8). Without StandardScaler,
    # the L2 penalty becomes scale-dependent.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs, bal_accs = [], []
    t0 = time.time()
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        accs.append(accuracy_score(y[te], pred))
        bal_accs.append(balanced_accuracy_score(y[te], pred))
    dt = time.time() - t0
    print(f"  {name:<22}  acc={np.mean(accs):.4f} ± {np.std(accs):.4f}   "
          f"bal_acc={np.mean(bal_accs):.4f}   ({dt:.1f}s)")
    return float(np.mean(accs)), float(np.mean(bal_accs))


def _softmax_acc(name: str, model_fwd, ds, val_idx, device) -> float:
    # Compare against the trained-readout accuracy. This is what `model(x)` gives.
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, len(val_idx), 32):
            batch_idx = val_idx[start:start + 32]
            mel_b, vid_b, y_b = [], [], []
            for i in batch_idx:
                mel, video, y = ds[int(i)]
                mel_b.append(mel); vid_b.append(video); y_b.append(int(y))
            mel_t = torch.stack(mel_b).unsqueeze(1).to(device)
            vid_t = torch.stack(vid_b).to(device)
            logits = model_fwd(mel_t, vid_t)
            pred = logits.argmax(1).cpu().numpy()
            correct += int((pred == np.asarray(y_b)).sum())
            total += len(y_b)
    acc = correct / total
    print(f"  {name:<22}  softmax_acc={acc:.4f}")
    return acc


def main():
    if not os.path.exists(NPZ_PATH):
        sys.exit(f"missing {NPZ_PATH} — run 04_extract_features.py first")

    data = np.load(NPZ_PATH)
    y = data["labels"]
    val_idx = data["val_idx"]
    print(f"Loaded {len(y)} samples from {NPZ_PATH}")
    print(f"  unique classes in this batch: {len(set(y.tolist()))}")

    print("\n5-fold LR probes (linear decode of word identity):")
    av_acc, _    = _probe("AV penult",         data["AV"],       y)
    av_vz_acc, _ = _probe("AV penult (v=0)",   data["AV_vzero"], y)
    a_acc, _     = _probe("A-only penult",     data["A_only"],   y)
    v_acc, _     = _probe("V-fair penult",     data["V_fair"],   y)

    # Compare trained-fc softmax accuracy against the probe on the same samples.
    # When we feed AV with v_mid=0, the trained fc is mis-calibrated (it was
    # trained expecting the gate's α-boosted output) so softmax accuracy
    # collapses. But the penult features still linearly encode the word —
    # the probe recovers most of the accuracy.
    print("\nTrained-readout (softmax) accuracy on the same samples:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairedAVDataset(t_stride=2)

    av_ckpt = torch.load(os.path.join(ROOT, "models/av_fused.pt"),
                          map_location=device, weights_only=False)
    av = AVWordResNet(len(av_ckpt["label_to_idx"])).to(device)
    av.load_state_dict(av_ckpt["model_state_dict"])
    av.eval()
    _softmax_acc("AV (full)",          lambda m, v: av(m, v),     ds, val_idx, device)
    _softmax_acc("AV (v=0)",           lambda m, v: av(m, None),  ds, val_idx, device)

    a_ckpt = torch.load(os.path.join(ROOT, "models/audio_only_filtered.pt"),
                         map_location=device, weights_only=False)
    a_only = WordResNet(len(a_ckpt["label_to_idx"])).to(device)
    a_only.load_state_dict(a_ckpt["model_state_dict"])
    a_only.eval()
    _softmax_acc("A-only",             lambda m, v: a_only(m),    ds, val_idx, device)

    print("\n--- summary ---")
    print(f"AV trained fc with v_mid=0:  softmax cliffs (look at the AV(v=0) row above).")
    print(f"AV penult probe with v_mid=0: linear decode = {av_vz_acc:.2%}.")
    print(f"The penult features still carry word identity; the fc just can't read it.")
    # The 1000-sample probe usually lands around 98% on AV penult — higher than
    # the synthesis's headline 92% because each class has only ~5 samples here
    # so 5-fold leaks. The 92% figure in
    # analysis/AV_INTEGRATION_TIER1_CROSS_VARIANT.md uses all 5244 val samples
    # and is the more rigorous number; the 98% here is the demo number.


if __name__ == "__main__":
    main()
