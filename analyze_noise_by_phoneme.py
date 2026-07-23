#!/usr/bin/env python3
"""Noise robustness broken down by phonetic class. Evaluates baseline +
noise-trained models on noisy val data across several std levels and
aggregates per-word accuracies by onset, syllable count, length, vowel."""

import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train import RANDOM_SEED, TEST_SIZE, WordResNet, stratified_split
from analyze_phoneme_accuracy import (
    get_length_group,
    get_onset,
    get_syllable_group,
    get_vowel_group,
)

DATA_PATH = os.path.join(SCRIPT_DIR, "processed", "dataset.pt")
BASELINE_PATH = os.path.join(SCRIPT_DIR, "processed", "model.pt")
NOISY_PATH = os.path.join(SCRIPT_DIR, "processed", "model_noisy.pt")
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.8]
NOISE_SEED = 42
BATCH_SIZE = 64

# Orderings (fixed, so plots are consistent across models)
SYLLABLE_ORDER = ["1", "2", "3", "4+"]
LENGTH_ORDER = ["Short (<=4)", "Medium (5-7)", "Long (8+)"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Inference

def load_model(path, num_classes, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = WordResNet(num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def eval_per_class(model, X_val, y_val, noise_std, device, gen):
    """Return dict {class_idx: (correct, total)} on noisy val data."""
    correct = defaultdict(int)
    total = defaultdict(int)
    ds = TensorDataset(X_val, y_val)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if noise_std > 0:
                noise = torch.empty_like(X_batch).normal_(
                    mean=0.0, std=float(noise_std), generator=gen,
                )
                X_in = X_batch + noise
            else:
                X_in = X_batch
            preds = model(X_in).argmax(1)
            for t, p in zip(y_batch.tolist(), preds.tolist()):
                total[t] += 1
                if t == p:
                    correct[t] += 1
    return correct, total


# Aggregation

def aggregate_by_class(per_class_correct, per_class_total, idx_to_label, classifier):
    """Sum per-word counts into class buckets. Returns {class: (correct, total)}."""
    agg_correct = defaultdict(int)
    agg_total = defaultdict(int)
    for cls_idx, n_total in per_class_total.items():
        word = idx_to_label[cls_idx]
        grp = classifier(word)
        agg_correct[grp] += per_class_correct.get(cls_idx, 0)
        agg_total[grp] += n_total
    return {g: (agg_correct[g], agg_total[g]) for g in agg_total}


def accs_from(agg):
    return {g: (c / t if t > 0 else 0.0) for g, (c, t) in agg.items()}


# Plotting helpers

def grouped_bar(labels, level_to_accs, levels, title, save_path,
                cmap="viridis", fig_size=(12, 6), sort_key=None):
    """Bars grouped by class, 1 bar per noise level.

    level_to_accs: {noise_std: {label: acc}}
    """
    if sort_key is not None:
        labels = sorted(labels, key=sort_key)

    n_classes = len(labels)
    n_levels = len(levels)
    bar_width = 0.8 / n_levels
    x = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=fig_size)
    cmap_obj = plt.get_cmap(cmap)
    for i, std in enumerate(levels):
        accs = [level_to_accs[std].get(lbl, 0.0) * 100 for lbl in labels]
        offset = (i - (n_levels - 1) / 2) * bar_width
        color = cmap_obj(0.15 + 0.7 * (i / max(1, n_levels - 1)))
        ax.bar(x + offset, accs, bar_width, label=f"std={std:.1f}",
               color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", ncol=n_levels)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def line_sweep(level_to_accs, labels, title, save_path, fig_size=(10, 6),
               cmap="tab10"):
    """One line per class across noise levels."""
    fig, ax = plt.subplots(figsize=fig_size)
    cmap_obj = plt.get_cmap(cmap)
    xs = sorted(level_to_accs.keys())
    for i, lbl in enumerate(labels):
        ys = [level_to_accs[std].get(lbl, 0.0) * 100 for std in xs]
        ax.plot(xs, ys, "o-", linewidth=2.0, markersize=8,
                color=cmap_obj(i % 10), label=lbl)
    ax.set_xlabel("Gaussian noise std")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def drop_bars(level_to_accs, labels, clean_std, noisy_std, title, save_path,
              fig_size=(10, 6), color="#d62728"):
    """Horizontal bars showing (clean - noisy) per class, sorted descending."""
    drops = [
        (lbl,
         (level_to_accs[clean_std].get(lbl, 0.0) - level_to_accs[noisy_std].get(lbl, 0.0)) * 100,
         level_to_accs[clean_std].get(lbl, 0.0) * 100,
         level_to_accs[noisy_std].get(lbl, 0.0) * 100)
        for lbl in labels
    ]
    drops.sort(key=lambda r: -r[1])
    names = [r[0] for r in drops][::-1]
    dvals = [r[1] for r in drops][::-1]
    cleans = [r[2] for r in drops][::-1]
    noisys = [r[3] for r in drops][::-1]

    fig, ax = plt.subplots(figsize=fig_size)
    bars = ax.barh(names, dvals, color=color, alpha=0.85)
    for bar, c, n in zip(bars, cleans, noisys):
        w = bar.get_width()
        x_text = w + (0.3 if w >= 0 else -0.3)
        ha = "left" if w >= 0 else "right"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f"{c:.0f}% → {n:.0f}%", va="center", ha=ha, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Accuracy drop (pp), clean → std={noisy_std}")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    spectrograms = data["spectrograms"]
    labels = data["labels"]
    idx_to_label = data["idx_to_label"]
    label_to_idx = data["label_to_idx"]
    num_classes = len(label_to_idx)

    _, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)
    X_val = spectrograms[val_idx].unsqueeze(1)
    y_val = labels[val_idx]
    print(f"Val samples: {len(y_val)}, classes: {num_classes}")

    baseline_model, _ = load_model(BASELINE_PATH, num_classes, device)
    noisy_model, _ = load_model(NOISY_PATH, num_classes, device)

    # Run evaluations
    # per_word[model][std] = {class_idx: (correct, total)}
    per_word = {"baseline": {}, "noisy": {}}
    for std in NOISE_LEVELS:
        gen_a = torch.Generator(device=device).manual_seed(NOISE_SEED + int(std * 100))
        gen_b = torch.Generator(device=device).manual_seed(NOISE_SEED + int(std * 100))
        c_b, t_b = eval_per_class(baseline_model, X_val, y_val, std, device, gen_a)
        c_n, t_n = eval_per_class(noisy_model, X_val, y_val, std, device, gen_b)
        per_word["baseline"][std] = {k: (c_b.get(k, 0), v) for k, v in t_b.items()}
        per_word["noisy"][std] = {k: (c_n.get(k, 0), v) for k, v in t_n.items()}
        acc_b = sum(c_b.values()) / sum(t_b.values())
        acc_n = sum(c_n.values()) / sum(t_n.values())
        print(f"  std={std:.2f}  baseline={acc_b:.4f}  noise-trained={acc_n:.4f}")

    # Aggregate by phoneme classes
    def per_class_correct_map(std_results):
        correct = {k: v[0] for k, v in std_results.items()}
        total = {k: v[1] for k, v in std_results.items()}
        return correct, total

    # Build {dim: {model: {std: {class: acc}}}}
    dimensions = {
        "onset": get_onset,
        "syllables": get_syllable_group,
        "length": get_length_group,
        "vowel": get_vowel_group,
    }

    agg = {
        dim: {"baseline": {}, "noisy": {}} for dim in dimensions
    }
    class_totals = {dim: {} for dim in dimensions}  # for N annotations

    for dim, classifier in dimensions.items():
        for model_name in ("baseline", "noisy"):
            for std in NOISE_LEVELS:
                c_map, t_map = per_class_correct_map(per_word[model_name][std])
                aggr = aggregate_by_class(c_map, t_map, idx_to_label, classifier)
                agg[dim][model_name][std] = accs_from(aggr)
                # Capture class totals once at clean level for labels
                if model_name == "baseline" and std == 0.0:
                    class_totals[dim] = {g: v[1] for g, v in aggr.items()}

    # CSV
    csv_path = os.path.join(ANALYSIS_DIR, "noise_by_phoneme.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_type", "class_name", "model",
            "clean_acc", "noisy_01", "noisy_03", "noisy_05", "noisy_08",
            "drop_at_03", "drop_at_08", "n_val",
        ])
        for dim in dimensions:
            classes = sorted(agg[dim]["baseline"][0.0].keys())
            for cls in classes:
                for model_name in ("baseline", "noisy"):
                    accs = agg[dim][model_name]
                    a0 = accs[0.0].get(cls, 0.0)
                    a01 = accs[0.1].get(cls, 0.0)
                    a03 = accs[0.3].get(cls, 0.0)
                    a05 = accs[0.5].get(cls, 0.0)
                    a08 = accs[0.8].get(cls, 0.0)
                    writer.writerow([
                        dim, cls, model_name,
                        f"{a0:.4f}", f"{a01:.4f}", f"{a03:.4f}",
                        f"{a05:.4f}", f"{a08:.4f}",
                        f"{a0 - a03:.4f}", f"{a0 - a08:.4f}",
                        class_totals[dim].get(cls, 0),
                    ])
    print(f"Wrote {csv_path}")

    # A. By onset consonant
    onset_classes = sorted(agg["onset"]["baseline"][0.0].keys())

    # A1: grouped bars — baseline model, std=[0.0, 0.3, 0.5, 0.8], sort by drop desc
    def sort_by_drop(cls, dim="onset", model="baseline"):
        a0 = agg[dim][model][0.0].get(cls, 0.0)
        a03 = agg[dim][model][0.3].get(cls, 0.0)
        return -(a0 - a03)  # biggest drop first

    a1_levels = [0.0, 0.3, 0.5, 0.8]
    grouped_bar(
        onset_classes,
        {std: agg["onset"]["baseline"][std] for std in a1_levels},
        a1_levels,
        "Baseline accuracy by onset consonant across noise levels\n(sorted by drop at std=0.3 — biggest drop left)",
        os.path.join(ANALYSIS_DIR, "noise_by_onset_bars.png"),
        cmap="viridis", fig_size=(13, 6),
        sort_key=sort_by_drop,
    )

    # A2: line sweep per onset class
    line_sweep(
        agg["onset"]["baseline"],
        sorted(onset_classes,
               key=lambda c: -agg["onset"]["baseline"][0.8].get(c, 0.0)),
        "Baseline: noise sweep by onset consonant",
        os.path.join(ANALYSIS_DIR, "noise_by_onset_sweep.png"),
        fig_size=(11, 6), cmap="tab20",
    )

    # A3: drop bar (clean → std=0.3)
    drop_bars(
        agg["onset"]["baseline"], onset_classes, 0.0, 0.3,
        "Baseline accuracy drop by onset consonant (clean → std=0.3)",
        os.path.join(ANALYSIS_DIR, "noise_by_onset_drop.png"),
        fig_size=(10, 7),
    )

    # B. By syllable count
    syl_classes = [c for c in SYLLABLE_ORDER if c in agg["syllables"]["baseline"][0.0]]

    # B1: noise sweep, baseline + noise-trained (8 lines)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap_obj = plt.get_cmap("tab10")
    xs = NOISE_LEVELS
    for i, cls in enumerate(syl_classes):
        color = cmap_obj(i)
        ys_b = [agg["syllables"]["baseline"][s].get(cls, 0.0) * 100 for s in xs]
        ys_n = [agg["syllables"]["noisy"][s].get(cls, 0.0) * 100 for s in xs]
        ax.plot(xs, ys_b, "o-", color=color, linewidth=2.0, markersize=8,
                label=f"{cls} syl · baseline")
        ax.plot(xs, ys_n, "s--", color=color, linewidth=2.0, markersize=8,
                label=f"{cls} syl · noise-trained")
    ax.set_xlabel("Gaussian noise std")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Noise sweep by syllable count (solid=baseline, dashed=noise-trained)")
    ax.set_xticks(xs)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "noise_by_syllables_sweep.png"))
    plt.close(fig)

    # B2: drop bars, baseline vs noise-trained at std=0.3
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(syl_classes))
    base_drops = [
        (agg["syllables"]["baseline"][0.0].get(c, 0.0)
         - agg["syllables"]["baseline"][0.3].get(c, 0.0)) * 100
        for c in syl_classes
    ]
    noisy_drops = [
        (agg["syllables"]["noisy"][0.0].get(c, 0.0)
         - agg["syllables"]["noisy"][0.3].get(c, 0.0)) * 100
        for c in syl_classes
    ]
    ax.bar(x - 0.2, base_drops, 0.4, label="Baseline", color="#1f77b4")
    ax.bar(x + 0.2, noisy_drops, 0.4, label="Noise-trained", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} syl" for c in syl_classes])
    ax.set_ylabel("Accuracy drop (pp), clean → std=0.3")
    ax.set_title("Syllable count: drop at std=0.3 (baseline vs noise-trained)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "noise_by_syllables_drop.png"))
    plt.close(fig)

    # C. By word length
    len_classes = [c for c in LENGTH_ORDER if c in agg["length"]["baseline"][0.0]]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap_obj = plt.get_cmap("tab10")
    for i, cls in enumerate(len_classes):
        color = cmap_obj(i)
        ys_b = [agg["length"]["baseline"][s].get(cls, 0.0) * 100 for s in NOISE_LEVELS]
        ys_n = [agg["length"]["noisy"][s].get(cls, 0.0) * 100 for s in NOISE_LEVELS]
        ax.plot(NOISE_LEVELS, ys_b, "o-", color=color, linewidth=2.0, markersize=8,
                label=f"{cls} · baseline")
        ax.plot(NOISE_LEVELS, ys_n, "s--", color=color, linewidth=2.0, markersize=8,
                label=f"{cls} · noise-trained")
    ax.set_xlabel("Gaussian noise std")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Noise sweep by word length (solid=baseline, dashed=noise-trained)")
    ax.set_xticks(NOISE_LEVELS)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "noise_by_length_sweep.png"))
    plt.close(fig)

    # D. By vowel sound
    vowel_classes = sorted(agg["vowel"]["baseline"][0.0].keys())

    grouped_bar(
        vowel_classes,
        {std: agg["vowel"]["baseline"][std] for std in a1_levels},
        a1_levels,
        "Baseline accuracy by vowel sound across noise levels\n(sorted by drop at std=0.3 — biggest drop left)",
        os.path.join(ANALYSIS_DIR, "noise_by_vowel_bars.png"),
        cmap="plasma", fig_size=(11, 6),
        sort_key=lambda c: sort_by_drop(c, dim="vowel"),
    )

    # E. Mega summary figure
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

    # (0,0) onset sweep
    ax = fig.add_subplot(gs[0, 0])
    cmap_obj = plt.get_cmap("tab20")
    ordered = sorted(onset_classes,
                     key=lambda c: -agg["onset"]["baseline"][0.8].get(c, 0.0))
    for i, cls in enumerate(ordered):
        ys = [agg["onset"]["baseline"][s].get(cls, 0.0) * 100 for s in NOISE_LEVELS]
        ax.plot(NOISE_LEVELS, ys, "o-", color=cmap_obj(i % 20),
                linewidth=1.6, markersize=6, label=cls)
    ax.set_xlabel("Noise std"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("A. Onset consonant sweep (baseline)")
    ax.set_xticks(NOISE_LEVELS); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=7)

    # (0,1) onset drop bars
    ax = fig.add_subplot(gs[0, 1])
    drops = [
        (c,
         (agg["onset"]["baseline"][0.0].get(c, 0.0)
          - agg["onset"]["baseline"][0.3].get(c, 0.0)) * 100)
        for c in onset_classes
    ]
    drops.sort(key=lambda r: -r[1])
    names = [r[0] for r in drops][::-1]
    vals = [r[1] for r in drops][::-1]
    ax.barh(names, vals, color="#d62728", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Drop (pp), clean → std=0.3")
    ax.set_title("A. Onset drop ranking (baseline)")
    ax.grid(True, axis="x", alpha=0.3)

    # (1,0) syllable sweep (both models)
    ax = fig.add_subplot(gs[1, 0])
    for i, cls in enumerate(syl_classes):
        color = plt.get_cmap("tab10")(i)
        ys_b = [agg["syllables"]["baseline"][s].get(cls, 0.0) * 100 for s in NOISE_LEVELS]
        ys_n = [agg["syllables"]["noisy"][s].get(cls, 0.0) * 100 for s in NOISE_LEVELS]
        ax.plot(NOISE_LEVELS, ys_b, "o-", color=color, linewidth=1.8, label=f"{cls} syl · base")
        ax.plot(NOISE_LEVELS, ys_n, "s--", color=color, linewidth=1.8, label=f"{cls} syl · noisy")
    ax.set_xlabel("Noise std"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("B. Syllable count sweep")
    ax.set_xticks(NOISE_LEVELS); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left", ncol=2, fontsize=7)

    # (1,1) length sweep (both models)
    ax = fig.add_subplot(gs[1, 1])
    for i, cls in enumerate(len_classes):
        color = plt.get_cmap("tab10")(i)
        ys_b = [agg["length"]["baseline"][s].get(cls, 0.0) * 100 for s in NOISE_LEVELS]
        ys_n = [agg["length"]["noisy"][s].get(cls, 0.0) * 100 for s in NOISE_LEVELS]
        ax.plot(NOISE_LEVELS, ys_b, "o-", color=color, linewidth=1.8, label=f"{cls} · base")
        ax.plot(NOISE_LEVELS, ys_n, "s--", color=color, linewidth=1.8, label=f"{cls} · noisy")
    ax.set_xlabel("Noise std"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("C. Word length sweep")
    ax.set_xticks(NOISE_LEVELS); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left", ncol=2, fontsize=7)

    # (2,0) vowel sweep
    ax = fig.add_subplot(gs[2, 0])
    cmap_obj = plt.get_cmap("tab10")
    for i, cls in enumerate(vowel_classes):
        ys = [agg["vowel"]["baseline"][s].get(cls, 0.0) * 100 for s in NOISE_LEVELS]
        ax.plot(NOISE_LEVELS, ys, "o-", color=cmap_obj(i % 10),
                linewidth=1.8, markersize=7, label=cls)
    ax.set_xlabel("Noise std"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("D. Vowel sound sweep (baseline)")
    ax.set_xticks(NOISE_LEVELS); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left", ncol=2, fontsize=7)

    # (2,1) vowel drop bars
    ax = fig.add_subplot(gs[2, 1])
    drops = [
        (c,
         (agg["vowel"]["baseline"][0.0].get(c, 0.0)
          - agg["vowel"]["baseline"][0.3].get(c, 0.0)) * 100)
        for c in vowel_classes
    ]
    drops.sort(key=lambda r: -r[1])
    names = [r[0] for r in drops][::-1]
    vals = [r[1] for r in drops][::-1]
    ax.barh(names, vals, color="#d62728", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Drop (pp), clean → std=0.3")
    ax.set_title("D. Vowel drop ranking (baseline)")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Noise robustness by phonetic class",
                 fontsize=16, fontweight="bold", y=1.00)
    fig.savefig(os.path.join(ANALYSIS_DIR, "noise_by_phoneme_summary.png"),
                bbox_inches="tight")
    plt.close(fig)

    # Text summary
    def rank_drops(dim, model="baseline", at=0.3):
        entries = []
        for cls in agg[dim][model][0.0]:
            d = (agg[dim][model][0.0][cls] - agg[dim][model][at].get(cls, 0.0)) * 100
            entries.append((cls, d,
                            agg[dim][model][0.0][cls] * 100,
                            agg[dim][model][at].get(cls, 0.0) * 100,
                            class_totals[dim].get(cls, 0)))
        entries.sort(key=lambda r: -r[1])
        return entries

    print("\n=== A. ONSET CONSONANT (baseline, clean → std=0.3) ===")
    print(f"{'onset':>8} {'clean':>8} {'noisy':>8} {'drop':>8} {'N':>6}")
    for cls, d, c, n, N in rank_drops("onset"):
        print(f"{cls:>8} {c:>7.1f}% {n:>7.1f}% {d:>7.1f}pp {N:>6}")

    print("\n=== A. ONSET at std=0.8 (baseline) ===")
    for cls, d, c, n, N in rank_drops("onset", at=0.8):
        print(f"{cls:>8} {c:>7.1f}% {n:>7.1f}% {d:>7.1f}pp {N:>6}")

    print("\n=== B. SYLLABLE COUNT (baseline, clean → std=0.3) ===")
    for cls, d, c, n, N in rank_drops("syllables"):
        print(f"  {cls:>4} syl: clean={c:5.1f}%  noisy={n:5.1f}%  drop={d:5.1f}pp  (N={N})")

    print("\n=== B. SYLLABLE COUNT (noise-trained, clean → std=0.3) ===")
    for cls, d, c, n, N in rank_drops("syllables", model="noisy"):
        print(f"  {cls:>4} syl: clean={c:5.1f}%  noisy={n:5.1f}%  drop={d:5.1f}pp")

    print("\n=== C. WORD LENGTH (baseline, clean → std=0.3) ===")
    for cls, d, c, n, N in rank_drops("length"):
        print(f"  {cls:>14}: clean={c:5.1f}%  noisy={n:5.1f}%  drop={d:5.1f}pp  (N={N})")

    print("\n=== D. VOWEL SOUND (baseline, clean → std=0.3) ===")
    for cls, d, c, n, N in rank_drops("vowel"):
        print(f"  {cls:>6}: clean={c:5.1f}%  noisy={n:5.1f}%  drop={d:5.1f}pp  (N={N})")

    print("\n=== D. VOWEL SOUND at std=0.8 (baseline) ===")
    for cls, d, c, n, N in rank_drops("vowel", at=0.8):
        print(f"  {cls:>6}: clean={c:5.1f}%  noisy={n:5.1f}%  drop={d:5.1f}pp")


if __name__ == "__main__":
    main()
