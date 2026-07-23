#!/usr/bin/env python3
"""Run 5 val samples through AV and A-only; flag AV-rescue cases."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_av import AVWordResNet
from paired_dataset import PairedAVDataset
from train import WordResNet


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# How many samples to show.
N_DISPLAY = 5
# How many val samples to scan when hunting for rescue cases.
N_SCAN = 200


def _top3(logits: torch.Tensor, idx_to_label: dict):
    probs = logits.softmax(dim=-1)
    vals, idxs = probs.topk(3)
    return [(idx_to_label[int(i)], float(v)) for v, i in zip(vals, idxs)]


def main():
    for path in ("models/av_fused.pt", "models/audio_only_filtered.pt",
                 "processed/dataset_av.pt", "processed/splits.pt"):
        if not os.path.exists(os.path.join(ROOT, path)):
            sys.exit(f"missing {path} — see top-level README quickstart")

    ds = PairedAVDataset(t_stride=2)
    idx_to_label = ds.idx_to_label
    splits = torch.load(os.path.join(ROOT, "processed", "splits.pt"),
                         weights_only=False)
    val_idx = np.asarray(splits["val_idx"])

    a_ckpt = torch.load(os.path.join(ROOT, "models/audio_only_filtered.pt"),
                         map_location="cpu", weights_only=False)
    a_model = WordResNet(len(a_ckpt["label_to_idx"]))
    a_model.load_state_dict(a_ckpt["model_state_dict"])
    a_model.eval()

    av_ckpt = torch.load(os.path.join(ROOT, "models/av_fused.pt"),
                          map_location="cpu", weights_only=False)
    av_model = AVWordResNet(len(av_ckpt["label_to_idx"]))
    av_model.load_state_dict(av_ckpt["model_state_dict"])
    av_model.eval()

    # Scan val samples; tag each as RESCUE (AV right, A wrong), AGREE
    # (both right), or MISS (both wrong). Display a didactic mix:
    # 1 AGREE + 3 RESCUE + 1 MISS so the pattern is visible at a glance.
    print(f"Scanning first {N_SCAN} val samples to find AV-rescue cases...\n")
    results = []
    with torch.no_grad():
        for i in val_idx[:N_SCAN]:
            idx = int(i)
            mel, video, label = ds[idx]
            truth = idx_to_label[int(label)]
            mel_in = mel.unsqueeze(0).unsqueeze(0)
            vid_in = video.unsqueeze(0)
            a_pred = _top3(a_model(mel_in)[0], idx_to_label)
            av_pred = _top3(av_model(mel_in, vid_in)[0], idx_to_label)
            a_right = a_pred[0][0] == truth
            av_right = av_pred[0][0] == truth
            tag = "RESCUE" if (av_right and not a_right) else \
                  "AGREE " if (av_right and a_right) else \
                  "MISS  "
            results.append((idx, truth, tag, a_pred, av_pred))

    rescues = [r for r in results if r[2] == "RESCUE"]
    agrees  = [r for r in results if r[2] == "AGREE "]
    misses  = [r for r in results if r[2] == "MISS  "]
    # Order: 1 AGREE (baseline), 3 RESCUE (the AV-wins case), 1 MISS (so the
    # reader sees a case where AV is also wrong). Backfill any missing slots
    # from whichever bucket has spares.
    display = (agrees[:1] + rescues[:3] + misses[:1])
    if len(display) < N_DISPLAY:
        spare = [r for r in (rescues + agrees + misses) if r not in display]
        display = (display + spare)[:N_DISPLAY]

    for idx, truth, tag, a_pred, av_pred in display:
        print(f"[idx={idx:<5}]  truth={truth!r:<18}  [{tag}]")
        print(f"  A-only  top-3:  "
              + ", ".join(f"{w}={p:.3f}" for w, p in a_pred))
        print(f"  AV-fused top-3: "
              + ", ".join(f"{w}={p:.3f}" for w, p in av_pred))
        print()
    print(f"Tallies in the first {N_SCAN} val samples: "
          f"{len(agrees)} AGREE, {len(rescues)} RESCUE, {len(misses)} MISS.")


if __name__ == "__main__":
    main()
