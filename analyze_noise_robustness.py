#!/usr/bin/env python3
"""Noise robustness analysis: baseline vs noise-trained model.

Evaluates both models on val data corrupted with Gaussian noise at several std
levels. Produces per-word CSV and 6 publication-quality plots.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train import RANDOM_SEED, TEST_SIZE, WordResNet, stratified_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "processed", "dataset.pt")
BASELINE_PATH = os.path.join(SCRIPT_DIR, "processed", "model.pt")
NOISY_PATH = os.path.join(SCRIPT_DIR, "processed", "model_noisy.pt")
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "analysis")

NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.8]
FOCUS_NOISE = 0.3  # the level used for per-word comparisons
NOISE_SEED = 42
BATCH_SIZE = 64

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_model(path, num_classes, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = WordResNet(num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def evaluate(model, X_val, y_val, noise_std, device, rng):
    """Return (overall_acc, per_class_correct, per_class_total) on X_val + noise."""
    per_correct = defaultdict(int)
    per_total = defaultdict(int)
    correct = 0
    total = 0

    # Generate noise once for this eval pass so both models see the same noise
    # when called with the same seed.
    ds = TensorDataset(X_val, y_val)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if noise_std > 0:
                # Deterministic noise per batch via torch's rng fed from `rng`
                noise = torch.empty_like(X_batch).normal_(
                    mean=0.0, std=float(noise_std), generator=rng,
                )
                X_in = X_batch + noise
            else:
                X_in = X_batch

            logits = model(X_in)
            preds = logits.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            for t, p in zip(y_batch.tolist(), preds.tolist()):
                per_total[t] += 1
                if t == p:
                    per_correct[t] += 1

    return correct / total, per_correct, per_total


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    spectrograms = data["spectrograms"]  # (N, F, T)
    labels = data["labels"]
    idx_to_label = data["idx_to_label"]
    label_to_idx = data["label_to_idx"]
    num_classes = len(label_to_idx)

    _, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)
    X_val = spectrograms[val_idx].unsqueeze(1)  # (N, 1, F, T)
    y_val = labels[val_idx]
    print(f"Val samples: {len(y_val)}, classes: {num_classes}")

    # Models
    baseline_model, baseline_ckpt = load_model(BASELINE_PATH, num_classes, device)
    noisy_model, noisy_ckpt = load_model(NOISY_PATH, num_classes, device)
    print(f"Baseline ckpt best_val_acc (clean): {baseline_ckpt.get('best_val_acc')}")
    print(f"Noise-trained ckpt best_val_acc (clean): {noisy_ckpt.get('best_val_acc')}")

    # Sweep
    overall_baseline = {}
    overall_noisy = {}
    perword_baseline = {}  # noise_std -> {class: acc}
    perword_noisy = {}

    for std in NOISE_LEVELS:
        # fresh generator per noise level so results are deterministic
        gen_a = torch.Generator(device=device).manual_seed(NOISE_SEED + int(std * 100))
        gen_b = torch.Generator(device=device).manual_seed(NOISE_SEED + int(std * 100))
        acc_b, pc_b, pt_b = evaluate(baseline_model, X_val, y_val, std, device, gen_a)
        acc_n, pc_n, pt_n = evaluate(noisy_model, X_val, y_val, std, device, gen_b)
        overall_baseline[std] = acc_b
        overall_noisy[std] = acc_n
        perword_baseline[std] = {
            c: (pc_b.get(c, 0) / pt_b[c] if pt_b[c] > 0 else 0.0)
            for c in pt_b
        }
        perword_noisy[std] = {
            c: (pc_n.get(c, 0) / pt_n[c] if pt_n[c] > 0 else 0.0)
            for c in pt_n
        }
        print(f"  std={std:.2f}  baseline={acc_b:.4f}  noise-trained={acc_n:.4f}")

    # CSV
    csv_path = os.path.join(ANALYSIS_DIR, "noise_robustness.csv")
    rows = []
    for cls in range(num_classes):
        b_clean = perword_baseline[0.0].get(cls, 0.0)
        b_noisy = perword_baseline[FOCUS_NOISE].get(cls, 0.0)
        n_clean = perword_noisy[0.0].get(cls, 0.0)
        n_noisy = perword_noisy[FOCUS_NOISE].get(cls, 0.0)
        drop = b_clean - b_noisy
        recovery = n_noisy - b_noisy
        rows.append({
            "word": idx_to_label[cls],
            "baseline_clean": b_clean,
            "baseline_noisy_03": b_noisy,
            "noise_trained_clean": n_clean,
            "noise_trained_noisy_03": n_noisy,
            "baseline_drop": drop,
            "recovery": recovery,
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "word", "baseline_clean", "baseline_noisy_03",
                "noise_trained_clean", "noise_trained_noisy_03",
                "baseline_drop", "recovery",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "word": r["word"],
                "baseline_clean": f"{r['baseline_clean']:.4f}",
                "baseline_noisy_03": f"{r['baseline_noisy_03']:.4f}",
                "noise_trained_clean": f"{r['noise_trained_clean']:.4f}",
                "noise_trained_noisy_03": f"{r['noise_trained_noisy_03']:.4f}",
                "baseline_drop": f"{r['baseline_drop']:.4f}",
                "recovery": f"{r['recovery']:.4f}",
            })
    print(f"Wrote {csv_path}")

    # Plot 1: noise sweep
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = NOISE_LEVELS
    ys_b = [overall_baseline[s] * 100 for s in xs]
    ys_n = [overall_noisy[s] * 100 for s in xs]
    ax.plot(xs, ys_b, "o-", color="#1f77b4", linewidth=2.5,
            markersize=10, label="Baseline model")
    ax.plot(xs, ys_n, "s-", color="#ff7f0e", linewidth=2.5,
            markersize=10, label="Noise-trained model")
    for x, y in zip(xs, ys_b):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", color="#1f77b4", fontsize=9)
    for x, y in zip(xs, ys_n):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, -16), ha="center", color="#ff7f0e", fontsize=9)
    ax.set_xlabel("Gaussian noise std added to val input")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Noise robustness: baseline vs noise-trained model")
    ax.set_xticks(xs)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "noise_sweep.png"))
    plt.close(fig)

    # Plot 2: per-word degradation scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    clean_accs = np.array([r["baseline_clean"] for r in rows])
    noisy_accs = np.array([r["baseline_noisy_03"] for r in rows])
    words = [r["word"] for r in rows]

    ax.scatter(clean_accs, noisy_accs, s=40, alpha=0.6, color="#1f77b4",
               edgecolor="white", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5, label="y = x (no degradation)")
    # Distance below diagonal
    dist_below = clean_accs - noisy_accs
    order = np.argsort(-dist_below)  # largest drop first
    for i in order[:10]:
        ax.annotate(words[i], (clean_accs[i], noisy_accs[i]),
                    textcoords="offset points", xytext=(5, -5),
                    fontsize=8, color="#d62728")
    # Most robust = smallest drop among words that have reasonable clean acc
    # so we don't just surface perfect words that stayed perfect.
    candidate = [i for i in range(len(words)) if clean_accs[i] >= 0.9]
    robust_order = sorted(candidate, key=lambda i: dist_below[i])
    for i in robust_order[:10]:
        ax.annotate(words[i], (clean_accs[i], noisy_accs[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="#2ca02c")
    ax.set_xlabel(f"Baseline clean accuracy")
    ax.set_ylabel(f"Baseline noisy accuracy (std={FOCUS_NOISE})")
    ax.set_title("Per-word degradation from noise (baseline model)\n"
                 "red = most degraded · green = most robust")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "per_word_degradation_scatter.png"))
    plt.close(fig)

    # Plot 3: top 20 vulnerable
    rows_sorted_drop = sorted(rows, key=lambda r: -r["baseline_drop"])
    top_vuln = rows_sorted_drop[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    labels_ = [r["word"] for r in top_vuln][::-1]
    drops = [r["baseline_drop"] * 100 for r in top_vuln][::-1]
    cleans = [r["baseline_clean"] * 100 for r in top_vuln][::-1]
    noisys = [r["baseline_noisy_03"] * 100 for r in top_vuln][::-1]
    bars = ax.barh(labels_, drops, color="#d62728", alpha=0.85)
    for bar, c, n in zip(bars, cleans, noisys):
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{c:.0f}% → {n:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("Accuracy drop (percentage points)")
    ax.set_title(f"Top 20 most vulnerable words (baseline, std={FOCUS_NOISE})")
    ax.set_xlim(0, max(drops) * 1.25 + 5)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "top20_vulnerable.png"))
    plt.close(fig)

    # Plot 4: top 20 robust
    rows_sorted_robust = sorted(rows, key=lambda r: r["baseline_drop"])
    top_robust = rows_sorted_robust[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    labels_ = [r["word"] for r in top_robust][::-1]
    drops = [r["baseline_drop"] * 100 for r in top_robust][::-1]
    cleans = [r["baseline_clean"] * 100 for r in top_robust][::-1]
    noisys = [r["baseline_noisy_03"] * 100 for r in top_robust][::-1]
    # Drops may be negative (noise helped); shift for visualization
    bars = ax.barh(labels_, drops, color="#2ca02c", alpha=0.85)
    for bar, c, n in zip(bars, cleans, noisys):
        w = bar.get_width()
        x_text = w + (0.3 if w >= 0 else -0.3)
        ha = "left" if w >= 0 else "right"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f"{c:.0f}% → {n:.0f}%", va="center", ha=ha, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy drop (percentage points)")
    ax.set_title(f"Top 20 most robust words (baseline, std={FOCUS_NOISE})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "top20_robust.png"))
    plt.close(fig)

    # Plot 5: noise training recovery
    rows_sorted_recovery = sorted(rows, key=lambda r: -r["recovery"])
    top_recovery = rows_sorted_recovery[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    labels_ = [r["word"] for r in top_recovery][::-1]
    base_noisy = [r["baseline_noisy_03"] * 100 for r in top_recovery][::-1]
    nt_noisy = [r["noise_trained_noisy_03"] * 100 for r in top_recovery][::-1]
    improvements = [r["recovery"] * 100 for r in top_recovery][::-1]

    y = np.arange(len(labels_))
    ax.barh(y - 0.2, base_noisy, height=0.4, color="#1f77b4",
            alpha=0.85, label=f"Baseline (std={FOCUS_NOISE})")
    ax.barh(y + 0.2, nt_noisy, height=0.4, color="#ff7f0e",
            alpha=0.85, label=f"Noise-trained (std={FOCUS_NOISE})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels_)
    for yi, imp in zip(y, improvements):
        ax.text(101, yi, f"+{imp:.0f}pp", va="center", fontsize=9,
                color="#2ca02c", fontweight="bold")
    ax.set_xlabel("Noisy accuracy (%)")
    ax.set_xlim(0, 120)
    ax.set_title(f"Top 20 words where noise training helps most (std={FOCUS_NOISE})")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "noise_training_recovery.png"))
    plt.close(fig)

    # Plot 6: per-category noise sweep
    # Split words into hand-crafted categories by inspection of labels
    def categorize(w):
        number_words = {
            "ZERO","ONE","TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE","TEN",
            "ELEVEN","TWELVE","THIRTEEN","FOURTEEN","FIFTEEN","SIXTEEN","SEVENTEEN",
            "EIGHTEEN","NINETEEN","TWENTY","THIRTY","FORTY","FIFTY","SIXTY","SEVENTY",
            "EIGHTY","NINETY","HUNDRED","THOUSAND","MILLION",
        }
        days = {"MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY",
                "TODAY","TOMORROW","YESTERDAY"}
        if w in number_words:
            return "numbers"
        if w in days:
            return "days/time"
        if len(w) <= 3:
            return "short (<=3 chars)"
        return "other"

    cat_to_classes = defaultdict(list)
    for cls in range(num_classes):
        cat_to_classes[categorize(idx_to_label[cls])].append(cls)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"numbers": "#d62728", "days/time": "#2ca02c",
              "short (<=3 chars)": "#9467bd", "other": "#1f77b4"}
    for cat, classes in cat_to_classes.items():
        cat_accs = []
        for std in NOISE_LEVELS:
            per = perword_baseline[std]
            vals = [per.get(c, 0.0) for c in classes]
            cat_accs.append(np.mean(vals) * 100 if vals else 0.0)
        ax.plot(NOISE_LEVELS, cat_accs, "o-", linewidth=2.2, markersize=9,
                color=colors.get(cat, "#7f7f7f"),
                label=f"{cat} (n={len(classes)})")
    ax.set_xlabel("Gaussian noise std")
    ax.set_ylabel("Mean per-word accuracy (%) — baseline model")
    ax.set_title("Per-category noise sweep (baseline model)")
    ax.set_xticks(NOISE_LEVELS)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "noise_sweep_per_category.png"))
    plt.close(fig)

    # Summary
    print("\n=== OVERALL ACCURACY BY NOISE LEVEL ===")
    print(f"{'std':>6} {'baseline':>12} {'noise-trained':>16}")
    for std in NOISE_LEVELS:
        print(f"{std:>6.2f} {overall_baseline[std]*100:>11.2f}% "
              f"{overall_noisy[std]*100:>15.2f}%")

    print("\n=== TOP 10 MOST VULNERABLE (baseline, std=0.3) ===")
    for r in rows_sorted_drop[:10]:
        print(f"  {r['word']:<18} clean={r['baseline_clean']*100:5.1f}%  "
              f"noisy={r['baseline_noisy_03']*100:5.1f}%  "
              f"drop={r['baseline_drop']*100:5.1f}pp")

    print("\n=== TOP 10 MOST ROBUST (baseline, std=0.3) ===")
    for r in rows_sorted_robust[:10]:
        print(f"  {r['word']:<18} clean={r['baseline_clean']*100:5.1f}%  "
              f"noisy={r['baseline_noisy_03']*100:5.1f}%  "
              f"drop={r['baseline_drop']*100:5.1f}pp")

    print("\n=== TOP 10 NOISE-TRAINING WINS (std=0.3) ===")
    for r in rows_sorted_recovery[:10]:
        print(f"  {r['word']:<18} base={r['baseline_noisy_03']*100:5.1f}%  "
              f"noise-trained={r['noise_trained_noisy_03']*100:5.1f}%  "
              f"+{r['recovery']*100:5.1f}pp")


if __name__ == "__main__":
    main()
