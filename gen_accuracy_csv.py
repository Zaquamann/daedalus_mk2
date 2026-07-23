#!/usr/bin/env python3
"""Generate per-word accuracy CSV from trained model on the val split."""

import csv
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from train import (
    RANDOM_SEED,
    TEST_SIZE,
    WordDataset,
    WordResNet,
    stratified_split,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "processed", "dataset.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "processed", "model.pt")
OUT_CSV = os.path.join(SCRIPT_DIR, "analysis", "per_word_accuracy.csv")


def main():
    torch.manual_seed(RANDOM_SEED)

    # Load dataset
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    spectrograms = data["spectrograms"]
    labels = data["labels"]
    label_to_idx = data["label_to_idx"]
    idx_to_label = data["idx_to_label"]
    num_classes = len(label_to_idx)

    # Stratified split — same as training
    _, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)

    val_dataset = WordDataset(
        spectrograms[val_idx], labels[val_idx], augment=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # Load model
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = WordResNet(num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run inference, collect predictions per true class
    #   confusion[true_idx][pred_idx] -> count
    confusion = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    correct = defaultdict(int)

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            preds = logits.argmax(1)
            for t, p in zip(y_batch.tolist(), preds.tolist()):
                totals[t] += 1
                confusion[t][p] += 1
                if t == p:
                    correct[t] += 1

    # Build per-word rows
    rows = []
    for cls_idx in range(num_classes):
        word = idx_to_label[cls_idx]
        total = totals.get(cls_idx, 0)
        n_correct = correct.get(cls_idx, 0)
        accuracy = (n_correct / total) if total > 0 else 0.0

        # Top confusion = most frequent WRONG prediction
        wrong_counts = {
            p: c for p, c in confusion[cls_idx].items() if p != cls_idx
        }
        if wrong_counts:
            top_pred = max(wrong_counts.items(), key=lambda kv: kv[1])
            top_confusion_word = idx_to_label[top_pred[0]]
            top_confusion_count = top_pred[1]
        else:
            top_confusion_word = ""
            top_confusion_count = 0

        rows.append({
            "word": word,
            "accuracy": accuracy,
            "correct": n_correct,
            "total": total,
            "top_confusion": top_confusion_word,
            "top_confusion_count": top_confusion_count,
        })

    # Sort ascending by accuracy (worst first)
    rows.sort(key=lambda r: (r["accuracy"], r["word"]))

    # Write CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "word",
                "accuracy",
                "correct",
                "total",
                "top_confusion",
                "top_confusion_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "word": row["word"],
                "accuracy": f"{row['accuracy']:.4f}",
                "correct": row["correct"],
                "total": row["total"],
                "top_confusion": row["top_confusion"],
                "top_confusion_count": row["top_confusion_count"],
            })

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print(f"Total val samples: {sum(r['total'] for r in rows)}")
    overall_correct = sum(r["correct"] for r in rows)
    overall_total = sum(r["total"] for r in rows)
    print(f"Overall val accuracy: {overall_correct / overall_total:.4f} "
          f"({overall_correct}/{overall_total})")

    print("\nTop 10 HARDEST words (lowest accuracy):")
    print(f"  {'word':<20} {'acc':>6} {'c/t':>8} {'top_confusion':<20} {'n':>4}")
    for r in rows[:10]:
        print(f"  {r['word']:<20} {r['accuracy']:>6.3f} "
              f"{r['correct']:>3}/{r['total']:<4} "
              f"{r['top_confusion']:<20} {r['top_confusion_count']:>4}")

    # Bottom 10 = last 10 in ascending sort = highest accuracy (perfect)
    perfect = [r for r in rows if r["accuracy"] >= 1.0]
    print(f"\nWords with perfect accuracy (1.000): {len(perfect)}")
    print("Last 10 entries (highest accuracy):")
    print(f"  {'word':<20} {'acc':>6} {'c/t':>8}")
    for r in rows[-10:]:
        print(f"  {r['word']:<20} {r['accuracy']:>6.3f} "
              f"{r['correct']:>3}/{r['total']:<4}")


if __name__ == "__main__":
    main()
