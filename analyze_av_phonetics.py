#!/usr/bin/env python3
"""Phonetic clustering across the four trained models (A-clean, A-noisy,
AV-clean, AV-noisy) on the shared val partition. Computes per-onset / per-
syllable / per-length / per-vowel breakdowns, AV confusion shifts, UMAPs +
silhouette + viseme linear probe, and Ward dendrograms. Outputs in
`analysis/phonetic_clustering_av/`."""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from analyze_phoneme_accuracy import (
    get_length_group,
    get_onset,
    get_syllable_group,
    get_vowel_group,
)
from model_av import AVWordResNet
from paired_dataset import PairedAVDataset
from train import WordResNet


# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "phonetic_clustering_av")
os.makedirs(OUT_DIR, exist_ok=True)

CKPTS = [
    ("A_clean",  "models/audio_only_filtered.pt",       "audio"),
    ("A_noisy",  "models/audio_only_noisy_filtered.pt", "audio"),
    ("AV_clean", "models/av_fused.pt",                  "av"),
    ("AV_noisy", "models/av_fused_noisy.pt",            "av"),
]

BATCH_SIZE = 64
T_STRIDE = 2

# Onset → viseme class. The lead's groups: /f v/, /b p m/, /w/, lingual, glottal.
# "lingual" lumps the alveolar+velar tongue-only articulations; "glottal" = /h/.
VISEME_MAP = {
    "/f/": "labiodental_fv",     # /f v/
    "/b/": "bilabial_bpm",       # /b p m/
    "/p/": "bilabial_bpm",
    "/m/": "bilabial_bpm",
    "/w/": "labiovelar_w",       # /w/
    "/t/": "lingual",
    "/d/": "lingual",
    "/n/": "lingual",
    "/s/": "lingual",
    "/r/": "lingual",
    "/k/": "lingual",
    "/h/": "glottal_h",
    "vowel": "vowel_initial",
    "other": "other",
}


def viseme_class(label: str) -> str:
    return VISEME_MAP.get(get_onset(label), "other")


# Data + model loading

class _ValView(torch.utils.data.Dataset):
    """Wraps PairedAVDataset to yield only validation indices, no augmentation."""

    def __init__(self, base: PairedAVDataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = int(self.indices[k])
        mel, video, label = self.base[idx]
        return mel, video, label


def _load_state(ckpt_path: str, device: torch.device, kind: str):
    """Load checkpoint and rebuild the matching model class."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    n_classes = len(ckpt["label_to_idx"])
    if kind == "audio":
        m = WordResNet(n_classes)
    elif kind == "av":
        m = AVWordResNet(n_classes)
    else:
        raise ValueError(kind)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()
    return m, ckpt


@torch.no_grad()
def _eval_audio(m: WordResNet, loader, device):
    preds, feats, labels = [], [], []
    for mel, _video, y in loader:
        x = mel.unsqueeze(1).to(device, non_blocking=True)
        x = m.block1(x)
        x = m.block2(x)
        x = m.gap(x).flatten(1)               # (B, 128) penultimate
        feats.append(x.cpu().numpy())
        preds.append(m.fc(x).argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return (np.concatenate(preds), np.concatenate(feats),
            np.concatenate(labels))


@torch.no_grad()
def _eval_av(m: AVWordResNet, loader, device):
    preds, feats, labels = [], [], []
    for mel, video, y in loader:
        mel_in = mel.unsqueeze(1).to(device, non_blocking=True)
        v_in = video.to(device, non_blocking=True)
        a_mid = m.audio_block1(mel_in)
        v_mid = m.visual(v_in)
        a_fused = m.gate(a_mid, v_mid)
        x = m.audio_block2(a_fused)
        x = m.gap(x).flatten(1)               # (B, 128) penultimate
        feats.append(x.cpu().numpy())
        preds.append(m.fc(x).argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return (np.concatenate(preds), np.concatenate(feats),
            np.concatenate(labels))


def gather_predictions(idx_to_label):
    """Returns dict[name] = {preds, feats, labels} on the val partition."""
    base = PairedAVDataset(t_stride=T_STRIDE)
    s = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                   weights_only=False)
    val_ds = _ValView(base, s["val_idx"])
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for name, path, kind in CKPTS:
        m, ckpt = _load_state(os.path.join(SCRIPT_DIR, path), device, kind)
        if kind == "audio":
            preds, feats, labels = _eval_audio(m, loader, device)
        else:
            preds, feats, labels = _eval_av(m, loader, device)
        acc = float((preds == labels).mean())
        print(f"  {name:>9s}: val_acc {acc:.4%}, feats {feats.shape}")
        out[name] = {
            "preds": preds, "feats": feats, "labels": labels,
            "ckpt_idx_to_label": ckpt["idx_to_label"],
            "acc": acc,
        }
        del m
        torch.cuda.empty_cache()
    return out


# Per-group accuracy

def _per_group_accuracy(preds, labels, idx_to_label, key_fn):
    correct = defaultdict(int)
    total = defaultdict(int)
    for p, t in zip(preds, labels):
        word = idx_to_label[int(t)]
        g = key_fn(word)
        total[g] += 1
        if p == t:
            correct[g] += 1
    return {g: (correct[g] / total[g], total[g]) for g in total}


def _write_group_compare_csv(name, groups_a, groups_av, path):
    rows = []
    for g in sorted(set(groups_a) | set(groups_av)):
        a_acc, a_n = groups_a.get(g, (None, 0))
        av_acc, _ = groups_av.get(g, (None, 0))
        delta = (av_acc - a_acc) if (a_acc is not None and av_acc is not None) else None
        rows.append((g, a_n, a_acc, av_acc, delta))
    rows.sort(key=lambda r: -(r[4] if r[4] is not None else -np.inf))
    with open(path, "w") as f:
        w = csv.writer(f)
        w.writerow([name, "n_samples", "A_acc", "AV_acc", "delta"])
        for r in rows:
            w.writerow([
                r[0], r[1],
                f"{r[2]:.4f}" if r[2] is not None else "",
                f"{r[3]:.4f}" if r[3] is not None else "",
                f"{r[4]:.4f}" if r[4] is not None else "",
            ])
    return rows


def _bar_plot_compare(rows, group_label, title, save_path):
    keep = [(r[0], r[2], r[3], r[4], r[1]) for r in rows
            if r[2] is not None and r[3] is not None]
    keep.sort(key=lambda r: -(r[3] - r[1]))  # by delta? r[3]=AV, r[1]=A; we want delta high
    g = [r[0] for r in keep]
    a = [r[1] for r in keep]
    av = [r[2] for r in keep]
    n = [r[4] for r in keep]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(g) + 1)))
    width = 0.4
    ax.barh(y - width / 2, a, height=width, color="#4477aa", label="A-only")
    ax.barh(y + width / 2, av, height=width, color="#cc6677", label="AV-fused")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label} (n={n_})" for label, n_ in zip(g, n)])
    ax.set_xlabel("val accuracy")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# Confusion rescue

def confusion_rescue_pairs(preds_a, preds_av, labels, idx_to_label, top_n=30):
    """Return list of (label_i, label_j, A_pair_count, AV_pair_count, rescue)
    sorted by rescue = A_pair_count - AV_pair_count, descending."""
    pair = defaultdict(lambda: [0, 0])
    for p, t in zip(preds_a, labels):
        if p != t:
            i, j = sorted([int(p), int(t)])
            pair[(i, j)][0] += 1
    for p, t in zip(preds_av, labels):
        if p != t:
            i, j = sorted([int(p), int(t)])
            pair[(i, j)][1] += 1
    rows = []
    for (i, j), (a, av) in pair.items():
        rows.append((idx_to_label[i], idx_to_label[j], a, av, a - av))
    rows.sort(key=lambda r: (-r[4], -r[2]))
    return rows[:top_n]


# UMAP + silhouette + linear probe + Ward

def umap_embed(feats: np.ndarray, seed: int = 42) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean",
                        random_state=seed, n_components=2)
    return reducer.fit_transform(feats.astype(np.float32))


def plot_umap_grid(coords_per_model, color_per_scheme, title_per_scheme,
                   save_path):
    """coords_per_model: dict[name] = (N, 2). Schemes: dict[name] = (label_array, palette)."""
    n_models = len(coords_per_model)
    n_schemes = len(color_per_scheme)
    fig, axes = plt.subplots(n_schemes, n_models,
                             figsize=(4 * n_models, 4 * n_schemes),
                             squeeze=False)
    for col, (mname, xy) in enumerate(coords_per_model.items()):
        for row, (sname, (labels_arr, palette)) in enumerate(color_per_scheme.items()):
            ax = axes[row][col]
            uniq = sorted(set(labels_arr))
            for cls in uniq:
                mask = labels_arr == cls
                color = palette.get(cls, "lightgray")
                ax.scatter(xy[mask, 0], xy[mask, 1], s=4, alpha=0.55,
                           c=color, label=cls if row == 0 else None,
                           linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(mname)
            if col == 0:
                ax.set_ylabel(title_per_scheme[sname], fontsize=10)
            if row == 0 and col == 0 and len(uniq) <= 12:
                ax.legend(loc="best", fontsize=6, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def silhouette_for_viseme(feats, viseme_labels):
    from sklearn.metrics import silhouette_score
    keep = [v not in {"other"} for v in viseme_labels]  # ignore "other" only
    f = feats[np.array(keep)]
    v = np.array(viseme_labels)[np.array(keep)]
    if len(set(v)) < 2:
        return float("nan")
    return float(silhouette_score(f, v, metric="euclidean"))


def linear_probe_viseme(feats, viseme_labels, seed=42):
    """Logistic regression on penultimate features to classify viseme.
    80/20 split for fair generalization."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    keep = np.array([v != "other" for v in viseme_labels])
    f = feats[keep]
    v = np.array(viseme_labels)[keep]
    Xtr, Xte, ytr, yte = train_test_split(f, v, test_size=0.2,
                                          random_state=seed, stratify=v)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        "acc": float(accuracy_score(yte, pred)),
        "balanced_acc": float(balanced_accuracy_score(yte, pred)),
        "n_classes": int(len(set(v))),
    }


def class_mean_embeddings(feats, labels, n_classes):
    means = np.zeros((n_classes, feats.shape[1]), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int64)
    for f, l in zip(feats, labels):
        means[int(l)] += f
        counts[int(l)] += 1
    nz = counts > 0
    means[nz] = means[nz] / counts[nz, None]
    return means, counts


def ward_dendrogram(class_means, idx_to_label, save_path,
                    title="Ward linkage on class-mean penult"):
    from scipy.cluster.hierarchy import dendrogram, linkage
    leaves = sorted(idx_to_label.keys())
    matrix = class_means[leaves]
    labels = [idx_to_label[i] for i in leaves]
    Z = linkage(matrix, method="ward")
    fig, ax = plt.subplots(figsize=(20, 6))
    dendrogram(Z, labels=labels, ax=ax, leaf_font_size=6,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# Main

def main() -> None:
    print("Loading & evaluating all 4 models on val partition...")
    base = PairedAVDataset(t_stride=T_STRIDE)
    idx_to_label = base.idx_to_label
    label_to_idx = base.label_to_idx
    n_classes = len(label_to_idx)

    out = gather_predictions(idx_to_label)

    # Sanity: all 4 models share idx_to_label byte-for-byte
    ref = out["A_clean"]["ckpt_idx_to_label"]
    for name in out:
        assert out[name]["ckpt_idx_to_label"] == ref, f"label maps differ for {name}"
    labels = out["A_clean"]["labels"]   # same val partition for all
    word_of = {i: idx_to_label[i] for i in range(n_classes)}

    # 1. Per-onset accuracy
    print("\n[1/7] Per-onset accuracy comparison...")
    onset_a  = _per_group_accuracy(out["A_clean"]["preds"],  labels, idx_to_label, get_onset)
    onset_av = _per_group_accuracy(out["AV_clean"]["preds"], labels, idx_to_label, get_onset)
    rows = _write_group_compare_csv(
        "onset", onset_a, onset_av,
        os.path.join(OUT_DIR, "per_onset_accuracy_compare.csv"),
    )
    _bar_plot_compare(
        rows, "onset",
        "Per-onset val accuracy: A-only-clean vs AV-clean",
        os.path.join(OUT_DIR, "per_onset_accuracy_compare.png"),
    )

    # Viseme grouping (cleaner story)
    visc_a  = _per_group_accuracy(out["A_clean"]["preds"],  labels, idx_to_label, viseme_class)
    visc_av = _per_group_accuracy(out["AV_clean"]["preds"], labels, idx_to_label, viseme_class)
    rows_v = _write_group_compare_csv(
        "viseme_class", visc_a, visc_av,
        os.path.join(OUT_DIR, "viseme_class_accuracy.csv"),
    )
    _bar_plot_compare(
        rows_v, "viseme_class",
        "Per-viseme val accuracy: A-only-clean vs AV-clean",
        os.path.join(OUT_DIR, "viseme_class_accuracy.png"),
    )

    # 2. Per-syllable / length / vowel
    print("[2/7] Per-syllable / length / vowel breakdowns...")
    for keyfn, fname in [
        (get_syllable_group, "per_syllable_accuracy_compare.csv"),
        (get_length_group,   "per_length_accuracy_compare.csv"),
        (get_vowel_group,    "per_vowel_accuracy_compare.csv"),
    ]:
        ga  = _per_group_accuracy(out["A_clean"]["preds"],  labels, idx_to_label, keyfn)
        gav = _per_group_accuracy(out["AV_clean"]["preds"], labels, idx_to_label, keyfn)
        _write_group_compare_csv(fname.split(".")[0], ga, gav,
                                 os.path.join(OUT_DIR, fname))

    # 3. Confusion rescue pairs
    print("[3/7] Confusion-rescue pairs (A→AV)...")
    rescues = confusion_rescue_pairs(
        out["A_clean"]["preds"], out["AV_clean"]["preds"],
        labels, idx_to_label, top_n=30,
    )
    with open(os.path.join(OUT_DIR, "confusion_rescue_pairs.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["word_i", "onset_i", "viseme_i",
                    "word_j", "onset_j", "viseme_j",
                    "A_pair_confusions", "AV_pair_confusions", "rescued"])
        for li, lj, a, av, rescue in rescues:
            w.writerow([li, get_onset(li), viseme_class(li),
                        lj, get_onset(lj), viseme_class(lj),
                        a, av, rescue])

    # 4. UMAP grids
    print("[4/7] UMAP embeddings (this is the slow step)...")
    coords = {}
    for name in ("A_clean", "AV_clean"):
        coords[name] = umap_embed(out[name]["feats"])

    # color schemes
    viseme_labels = np.array([viseme_class(idx_to_label[int(t)]) for t in labels])
    onset_labels = np.array([get_onset(idx_to_label[int(t)]) for t in labels])

    # palettes
    cmap_v = plt.get_cmap("tab10")
    visemes_uniq = sorted(set(viseme_labels))
    viseme_palette = {v: cmap_v(i % 10) for i, v in enumerate(visemes_uniq)}
    cmap_o = plt.get_cmap("tab20")
    onsets_uniq = sorted(set(onset_labels))
    onset_palette = {o: cmap_o(i % 20) for i, o in enumerate(onsets_uniq)}
    cmap_c = plt.get_cmap("hsv")
    class_palette = {idx_to_label[i]: cmap_c(i / max(1, n_classes - 1))
                     for i in range(n_classes)}
    class_labels_str = np.array([idx_to_label[int(t)] for t in labels])

    plot_umap_grid(
        coords,
        {"viseme": (viseme_labels, viseme_palette),
         "onset":  (onset_labels, onset_palette),
         "word":   (class_labels_str, class_palette)},
        {"viseme": "by viseme class", "onset": "by onset", "word": "by word (180 classes)"},
        os.path.join(OUT_DIR, "umap_a_vs_av.png"),
    )

    # 5. Silhouette score
    print("[5/7] Silhouette scores...")
    sil_rows = []
    for name in out:
        sil = silhouette_for_viseme(out[name]["feats"], viseme_labels)
        sil_rows.append((name, "viseme", sil))
        print(f"  {name:>9s}: silhouette(viseme) = {sil:.4f}")
    with open(os.path.join(OUT_DIR, "viseme_silhouette.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "grouping", "silhouette_score"])
        for r in sil_rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}"])

    # 6. Linear probe
    print("[6/7] Linear probe for viseme class...")
    probe_rows = []
    for name in out:
        res = linear_probe_viseme(out[name]["feats"], viseme_labels)
        probe_rows.append((name, res["n_classes"], res["acc"], res["balanced_acc"]))
        print(f"  {name:>9s}: probe acc = {res['acc']:.4f}, "
              f"balanced = {res['balanced_acc']:.4f}")
    with open(os.path.join(OUT_DIR, "linear_probe_viseme.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_viseme_classes", "test_acc", "balanced_test_acc"])
        for r in probe_rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}"])

    # 7. Ward dendrogram on class means
    print("[7/7] Ward dendrograms (A-only-clean vs AV-clean)...")
    for name in ("A_clean", "AV_clean"):
        means, counts = class_mean_embeddings(
            out[name]["feats"], out[name]["labels"], n_classes,
        )
        ward_dendrogram(
            means, idx_to_label,
            os.path.join(OUT_DIR, f"hierarchical_clusters_{name}.png"),
            title=f"Ward linkage on class-mean penult — {name}",
        )

    # Summary file
    summary_path = os.path.join(OUT_DIR, "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Phonetic clustering analysis — model val_acc summary\n")
        f.write("=" * 56 + "\n")
        for name in out:
            f.write(f"  {name:>9s}: {out[name]['acc']:.4%}\n")
        f.write("\nViseme silhouette (higher = tighter visemic clusters):\n")
        for r in sil_rows:
            f.write(f"  {r[0]:>9s}: {r[2]:.4f}\n")
        f.write("\nViseme linear probe (test acc):\n")
        for r in probe_rows:
            f.write(f"  {r[0]:>9s}: acc={r[2]:.4f}, balanced={r[3]:.4f}\n")
        f.write("\nTop 10 confusion rescues (A_clean → AV_clean):\n")
        for li, lj, a, av, rescue in rescues[:10]:
            f.write(f"  {li:>14s} ↔ {lj:<14s}: {a} → {av}  (rescued {rescue})\n")
    print(f"\nAll outputs in: {OUT_DIR}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
