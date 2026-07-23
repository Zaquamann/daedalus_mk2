#!/usr/bin/env python3
"""Inspect the AV cross-modal gate on one sample: α, per-channel gain, map."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_av import AVWordResNet
from paired_dataset import PairedAVDataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUT_PNG = os.path.join(OUT_DIR, "06_gate.png")

# From Phase C lesions — these two channels carry most of the visual signal.
# Lesion ch22 = −42 pp val_acc; lesion ch12 = −38 pp. See
# analysis/AV_INTEGRATION_DEEP_DIVE_SYNTHESIS.md §3.5.
HIGHLIGHT_CH = [12, 22]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(ROOT, "models/av_fused.pt")):
        sys.exit("missing models/av_fused.pt — see top-level README quickstart")

    ds = PairedAVDataset(t_stride=2)
    splits = torch.load(os.path.join(ROOT, "processed", "splits.pt"),
                         weights_only=False)
    sample_idx = int(splits["val_idx"][0])
    mel, video, label = ds[sample_idx]
    word = ds.idx_to_label[int(label)]

    ckpt = torch.load(os.path.join(ROOT, "models/av_fused.pt"),
                       map_location="cpu", weights_only=False)
    model = AVWordResNet(len(ckpt["label_to_idx"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    alpha = model.gate.alpha.item()
    print(f"Trained α (learnable gain scalar) = {alpha:.4f}")
    print(f"  initialised at 0.2; this value reflects how strongly the model")
    print(f"  learned to lean on the visual stream.\n")

    mel_in = mel.unsqueeze(0).unsqueeze(0)        # (1, 1, 80, 99)
    vid_in = video.unsqueeze(0)                    # (1, 1, 50, 88, 88)
    with torch.no_grad():
        a_mid = model.audio_block1(mel_in)         # (1, 64, 40, 50)
        v_mid = model.visual(vid_in)               # (1, 64, 40, 50)
        # Reproduce the gate forward so we can pull g and α·g out separately.
        g = torch.sigmoid(model.gate.Wa(a_mid) + model.gate.Wv(v_mid))   # (1, 64, 40, 50)
        gain = alpha * g                                                  # the "boost"

    # Per-channel mean gain (averaged over spatial-temporal). This tells you
    # which of the 64 channels are getting modulated most strongly.
    per_ch_gain = gain.mean(dim=(0, 2, 3)).numpy()                       # (64,)
    # 2D map averaged over channels, for a spatial-temporal heatmap.
    gate_map = g.mean(dim=1)[0].numpy()                                  # (40, 50)

    print(f"Sample idx={sample_idx} — word: {word!r}")
    print(f"  mean per-channel gain (α·g):  {per_ch_gain.mean():.3f}")
    print(f"  max  per-channel gain (α·g):  {per_ch_gain.max():.3f}   "
          f"(ch {per_ch_gain.argmax()})")
    print(f"  ch12 gain = {per_ch_gain[12]:.3f}   "
          f"ch22 gain = {per_ch_gain[22]:.3f}   "
          f"(integration channels)")

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          width_ratios=[1.3, 1.0])

    # Top-left: mel input
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(mel.numpy(), aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"audio mel — word {word!r}")
    ax.set_xlabel("time frame")
    ax.set_ylabel("mel band")

    # Top-right: lip frame
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(video[0, 25].numpy(), cmap="gray")
    ax.set_title("lip ROI (mid-clip)")
    ax.axis("off")

    # Bottom-left: 64-channel gain bar chart (mean over time)
    ax = fig.add_subplot(gs[1, 0])
    colors = ["#cc6677" if c in HIGHLIGHT_CH else "#888888"
              for c in range(len(per_ch_gain))]
    ax.bar(range(len(per_ch_gain)), per_ch_gain, color=colors,
           edgecolor="none")
    ax.set_xlabel("audio channel (0–63)")
    ax.set_ylabel("mean α·g over space-time")
    ax.set_title("per-channel gate gain (red = integration channels)")
    for c in HIGHLIGHT_CH:
        ax.annotate(f"ch{c}", xy=(c, per_ch_gain[c]),
                    xytext=(c, per_ch_gain[c] + 0.15),
                    ha="center", fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="#cc6677", lw=0.8))

    # Bottom-right: spatial-temporal gate heatmap
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(gate_map, aspect="auto", origin="lower",
                    cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"gate map σ(Wa·a + Wv·v) mean over 64 ch\n"
                  f"α·g range: [{alpha * gate_map.min():.2f}, "
                  f"{alpha * gate_map.max():.2f}]")
    ax.set_xlabel("time")
    ax.set_ylabel("mel band (post-block1)")
    fig.colorbar(im, ax=ax, fraction=0.05)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"\nSaved figure to: {OUT_PNG}")


if __name__ == "__main__":
    main()
