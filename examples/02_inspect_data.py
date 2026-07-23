#!/usr/bin/env python3
"""Pull one paired sample; plot mel + 4 evenly-spaced lip frames."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paired_dataset import PairedAVDataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUT_PNG = os.path.join(OUT_DIR, "02_sample.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(ROOT, "processed", "dataset_av.pt")):
        sys.exit("missing processed/dataset_av.pt — see top-level README "
                 "(run `python paired_dataset.py` first)")

    # t_stride=2 matches the AV training default: T=100 cache → T=50 input.
    ds = PairedAVDataset(t_stride=2)
    print(f"Dataset: {len(ds)} paired clips, {len(ds.label_to_idx)} word classes")

    # Pick a sample. Anything in [0, len(ds)) works; idx=1234 is arbitrary.
    idx = 1234
    mel, video, label = ds[idx]
    word = ds.idx_to_label[int(label)]
    speaker = ds.speakers[idx] if idx < len(ds.speakers) else "?"
    print(f"\nSample {idx}:")
    print(f"  word        = {word!r}")
    print(f"  speaker     = {speaker}")
    print(f"  mel shape   = {tuple(mel.shape)}  dtype={mel.dtype}")
    print(f"  video shape = {tuple(video.shape)} dtype={video.dtype}")
    print(f"  mel range   = [{mel.min():.2f}, {mel.max():.2f}]  (log-mel)")

    T = video.shape[1]
    frame_idxs = [int(round(t)) for t in [0.10 * T, 0.35 * T, 0.60 * T, 0.85 * T]]

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1.0])

    ax_mel = fig.add_subplot(gs[0, :])
    ax_mel.imshow(mel.numpy(), aspect="auto", origin="lower", cmap="magma")
    ax_mel.set_xlabel("frame (10 ms)")
    ax_mel.set_ylabel("mel band")
    ax_mel.set_title(f"log-mel — word={word!r}  speaker={speaker}")

    for col, t in enumerate(frame_idxs):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(video[0, t].numpy(), cmap="gray")
        ax.set_title(f"t={t}/{T}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"\nSaved figure to: {OUT_PNG}")


if __name__ == "__main__":
    main()
