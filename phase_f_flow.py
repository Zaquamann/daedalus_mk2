#!/usr/bin/env python3
"""Phase F — layer-wise information flow: linear probes per layer (word /
onset / viseme), within-AV and cross-model CKA, RDM-similarity vs depth.
Reads `processed/deepdive_act_cache.pt` from Phase A.
Run: `python phase_f_flow.py`."""

from __future__ import annotations

import csv
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from analyze_av_phonetics import viseme_class as _viseme_from_label
from analyze_phoneme_accuracy import get_onset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "deepdive")
CACHE_PATH = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
os.makedirs(OUT_DIR, exist_ok=True)


# Helpers

def _probe_5fold(X, y, max_iter: int = 1500, C: float = 1.0,
                  seed: int = 0) -> tuple[float, float]:
    """Return (mean acc, mean balanced acc) on 5-fold stratified CV.

    Features are z-scored per fold (LR converges much faster on
    standardized inputs — see sklearn ConvergenceWarning notes).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, bal_accs = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        clf.fit(X_tr, y[tr])
        pred = clf.predict(X_te)
        accs.append(accuracy_score(y[te], pred))
        bal_accs.append(balanced_accuracy_score(y[te], pred))
    return float(np.mean(accs)), float(np.mean(bal_accs))


def _linear_cka(X, Y) -> float:
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = (X.T @ Y).reshape(-1)
    num = float(np.dot(num, num))
    den_x = float(np.linalg.norm(X.T @ X, "fro"))
    den_y = float(np.linalg.norm(Y.T @ Y, "fro"))
    return num / (den_x * den_y + 1e-12)


def _class_mean(feats, labels, n_classes):
    means = np.zeros((n_classes, feats.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for f, l in zip(feats, labels):
        means[int(l)] += f
        counts[int(l)] += 1
    nz = counts > 0
    means[nz] /= counts[nz, None]
    return means.astype(np.float32)


# D5.1 — Layer-wise linear probes (word / onset / viseme)

LAYER_SITES = {
    "A_only":  ["block1_gap", "block2_gap", "penult"],
    "V_fair":  ["visual_gap", "block2_gap", "penult"],
    "AV_full": ["a_mid_gap", "v_mid_gap", "gate_out_gap",
                "block2_gap", "penult"],
}


def D5_1_layer_probe(cache, idx_to_label):
    print("\n  D5.1 — Layer-wise linear probes (word, onset, viseme)")
    labels = cache["labels"]
    onsets = np.asarray([get_onset(idx_to_label[int(l)]) for l in labels])
    visemes = np.asarray([_viseme_from_label(idx_to_label[int(l)])
                            for l in labels])
    keep_v = visemes != "other"
    # Onset keep: drop "vowel" + "other" to focus on consonant-initial words
    keep_o = (onsets != "vowel") & (onsets != "other")

    targets = [
        ("word",   labels,             slice(None)),
        ("onset",  onsets,             keep_o),
        ("viseme", visemes,            keep_v),
    ]

    rows = []        # for combined CSV
    plot_data = {}   # for the composite line plot

    for model_key, src_key in [("A_only", "A_only"),
                                 ("V_fair", "V_fair"),
                                 ("AV_full", "AV_clean_full")]:
        for site in LAYER_SITES[model_key]:
            X = cache[src_key][site]
            for tname, y, keep in targets:
                if keep is slice(None):
                    Xk, yk = X, y
                else:
                    Xk, yk = X[keep], y[keep]
                acc, bal = _probe_5fold(Xk, yk)
                print(f"    {model_key:>8s} | {site:<14s} | "
                      f"{tname:>7s} | acc={acc*100:5.2f}% bal={bal*100:5.2f}%")
                rows.append((model_key, site, tname, acc, bal))
                plot_data.setdefault((model_key, tname), []).append(
                    (site, acc))

    # Per-target CSVs
    for tname, _, _ in targets:
        out = os.path.join(OUT_DIR, f"D5_layer_decodability_{tname}.csv")
        with open(out, "w") as f:
            w = csv.writer(f)
            w.writerow(["model", "layer", "acc_5fold", "bal_acc_5fold"])
            for m, site, t, a, b in rows:
                if t == tname:
                    w.writerow([m, site, f"{a:.6f}", f"{b:.6f}"])
        print(f"  wrote {out}")

    # Composite line plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (tname, _, _) in zip(axes, targets):
        for (m, t), data in plot_data.items():
            if t != tname:
                continue
            xs = [s for s, _ in data]
            ys = [a for _, a in data]
            ax.plot(range(len(xs)), ys, "o-",
                     label=m, linewidth=2)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, rotation=20, ha="right",
                                 fontsize=8)
        ax.set_ylabel("5-fold acc")
        ax.set_title(f"{tname} decodability")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "D5_layer_decodability.png")
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  wrote {out_png}")


# D5.2 — Within-AV CKA across layers

def D5_2_within_av_cka(cache):
    print("\n  D5.2 — Within-AV linear CKA across layers:")
    av = cache["AV_clean_full"]
    sites = ["a_mid_gap", "v_mid_gap", "gate_out_gap",
              "block2_gap", "penult"]
    M = np.zeros((len(sites), len(sites)))
    for i, s1 in enumerate(sites):
        for j, s2 in enumerate(sites):
            if i <= j:
                M[i, j] = _linear_cka(av[s1], av[s2])
                M[j, i] = M[i, j]
    print("    sites: " + " ".join(sites))
    for i, s in enumerate(sites):
        row = " ".join(f"{M[i,j]:.3f}" for j in range(len(sites)))
        print(f"    {s:>14s}: {row}")
    out_csv = os.path.join(OUT_DIR, "D5_cka_within_av.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["site_A", "site_B", "linear_CKA"])
        for i, s1 in enumerate(sites):
            for j, s2 in enumerate(sites):
                w.writerow([s1, s2, f"{M[i,j]:.6f}"])
    print(f"  wrote {out_csv}")


# D5.3 — Information-plane (KSG MI estimator, LOOSE)

def _mi_continuous_discrete_binned(X, y, n_bins: int = 16) -> float:
    """Mutual information estimator (continuous X, discrete y) via histogramming.

    Cheap, biased but stable across runs. LOOSE per plan §6.F.
    """
    # Use just the first 8 PCA components to bound bin count.
    from sklearn.decomposition import PCA
    Xp = PCA(n_components=min(8, X.shape[1]), random_state=0).fit_transform(X)
    # Bin each column into n_bins, encode as a single int via base-n_bins
    bins = []
    for c in range(Xp.shape[1]):
        edges = np.quantile(Xp[:, c], np.linspace(0, 1, n_bins + 1))
        bins.append(np.clip(np.digitize(Xp[:, c], edges[1:-1]),
                             0, n_bins - 1))
    bins = np.stack(bins, axis=1)
    code = np.zeros(len(Xp), dtype=np.int64)
    for c in range(bins.shape[1]):
        code = code * n_bins + bins[:, c]
    # Joint histogram
    uniq_codes, code_inv = np.unique(code, return_inverse=True)
    uniq_y, y_inv = np.unique(y, return_inverse=True)
    joint = np.zeros((len(uniq_codes), len(uniq_y)))
    for i in range(len(code)):
        joint[code_inv[i], y_inv[i]] += 1
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py + 1e-12)
        log_ratio = np.where(ratio > 0, np.log(ratio + 1e-12), 0.0)
    mi = (joint * log_ratio).sum()
    return float(mi)


def D5_3_info_plane(cache):
    print("\n  D5.3 — Information plane (binned MI, LOOSE):")
    av = cache["AV_clean_full"]
    labels = cache["labels"]
    sites = ["a_mid_gap", "v_mid_gap", "gate_out_gap",
              "block2_gap", "penult"]
    out_csv = os.path.join(OUT_DIR, "D5_info_plane.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["site", "I_layer_label_nats", "note"])
        for s in sites:
            mi = _mi_continuous_discrete_binned(av[s], labels)
            print(f"    I({s}; Y) ≈ {mi:.3f} nats")
            w.writerow([s, f"{mi:.6f}",
                        "binned PCA-8 MI, n_bins=16 — LOOSE"])
    print(f"  wrote {out_csv}")


# D5.9 — Cross-model layer CKA (AV layers × A layers)

def D5_9_cross_model_cka(cache):
    print("\n  D5.9 — Cross-model AV × A_only CKA:")
    av_sites = ["a_mid_gap", "v_mid_gap", "gate_out_gap",
                 "block2_gap", "penult"]
    a_sites = ["block1_gap", "block2_gap", "penult"]
    out_csv = os.path.join(OUT_DIR, "D5_cka_cross_model.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["AV_site", "A_site", "linear_CKA"])
        print(f"    {'AV \\ A':>14s} | " +
              " | ".join(f"{a:>11s}" for a in a_sites))
        for s_av in av_sites:
            cells = []
            for s_a in a_sites:
                v = _linear_cka(cache["AV_clean_full"][s_av],
                                 cache["A_only"][s_a])
                w.writerow([s_av, s_a, f"{v:.6f}"])
                cells.append(f"{v:.3f}")
            print(f"    {s_av:>14s} | " +
                  " | ".join(f"{c:>11s}" for c in cells))
    print(f"  wrote {out_csv}")


# D5.11 — Per-block2-channel R² on v_mid features

def D5_11_v_modulation(cache):
    print("\n  D5.11 — Per-block2-channel R² on v_mid:")
    av = cache["AV_clean_full"]
    X = av["v_mid_gap"]       # (N, 64)
    Y = av["block2_gap"]      # (N, 128)
    out_csv = os.path.join(OUT_DIR, "D5_block2_v_modulation.csv")
    r2_per_ch = []
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["block2_channel", "r2_v_mid"])
        for c in range(Y.shape[1]):
            y = Y[:, c]
            reg = Ridge(alpha=1.0)
            reg.fit(X, y)
            yp = reg.predict(X)
            ss_res = float(((y - yp) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            r2_per_ch.append(r2)
            w.writerow([c, f"{r2:.6f}"])
    r2_arr = np.asarray(r2_per_ch)
    n_modulated = int((r2_arr > 0.3).sum())
    print(f"    R² > 0.30 channels: {n_modulated}/{len(r2_arr)} "
          f"({n_modulated/len(r2_arr)*100:.1f}%)")
    print(f"    R² distribution: median={np.median(r2_arr):.3f}, "
          f"p90={np.percentile(r2_arr,90):.3f}, "
          f"max={r2_arr.max():.3f}")
    print(f"  wrote {out_csv}")


# D5.12 — RSA layer trajectory: layer-RDM ↔ {word, onset, viseme} categorical RDM

def _categorical_rdm(y) -> np.ndarray:
    """1 if labels differ, 0 if same. Returns condensed pdist-style array."""
    n = len(y)
    out = np.zeros((n * (n - 1)) // 2)
    k = 0
    for i in range(n - 1):
        out[k:k+n-i-1] = (y[i+1:] != y[i]).astype(np.float32)
        k += n - i - 1
    return out


def _class_mean_rdm(feats, labels, n_classes):
    means = _class_mean(feats, labels, n_classes)
    return pdist(means, metric="cosine")


def D5_12_rsa_trajectory(cache, idx_to_label):
    print("\n  D5.12 — RSA layer trajectory")
    labels = cache["labels"]
    n_classes = int(labels.max()) + 1
    onsets = np.asarray([get_onset(idx_to_label[int(l)]) for l in labels])
    visemes = np.asarray([_viseme_from_label(idx_to_label[int(l)])
                            for l in labels])

    # Build per-class onset / viseme arrays (one value per class) for
    # category RDMs:
    cls_onset = np.empty(n_classes, dtype="<U16")
    cls_viseme = np.empty(n_classes, dtype="<U24")
    for l in range(n_classes):
        m = labels == l
        if m.any():
            cls_onset[l] = onsets[np.where(m)[0][0]]
            cls_viseme[l] = visemes[np.where(m)[0][0]]
    rdm_onset_cls = _categorical_rdm(cls_onset)
    rdm_viseme_cls = _categorical_rdm(cls_viseme)

    sources = {
        "A_only":  ("A_only",          LAYER_SITES["A_only"]),
        "V_fair":  ("V_fair",          LAYER_SITES["V_fair"]),
        "AV_full": ("AV_clean_full",   LAYER_SITES["AV_full"]),
    }
    rows = []
    for model, (src, sites) in sources.items():
        for site in sites:
            feats = cache[src][site]
            rdm = _class_mean_rdm(feats, labels, n_classes)
            rho_onset, _ = spearmanr(rdm, rdm_onset_cls)
            rho_viseme, _ = spearmanr(rdm, rdm_viseme_cls)
            print(f"    {model:>8s} {site:>14s}: "
                  f"ρ_onset={rho_onset:.3f}  ρ_viseme={rho_viseme:.3f}")
            rows.append((model, site, rho_onset, rho_viseme))
    out_csv = os.path.join(OUT_DIR, "D5_rsa_layer_trajectory.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "spearman_rho_onset_rdm",
                    "spearman_rho_viseme_rdm"])
        for m, s, ro, rv in rows:
            w.writerow([m, s, f"{ro:.6f}", f"{rv:.6f}"])
    print(f"  wrote {out_csv}")

    # PNG: line plot per model, per categorical target
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, target_idx, title in [(axes[0], 2, "Onset RDM"),
                                    (axes[1], 3, "Viseme RDM")]:
        for model in sources:
            sub = [r for r in rows if r[0] == model]
            xs = [r[1] for r in sub]
            ys = [r[target_idx] for r in sub]
            ax.plot(range(len(xs)), ys, "o-", linewidth=2, label=model)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Spearman ρ with category RDM")
        ax.set_title(f"D5.12 — {title}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "D5_rsa_layer_trajectory.png")
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  wrote {out_png}")


# Main

def main() -> None:
    torch.manual_seed(0); np.random.seed(0)
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"missing activation cache: run phase_a_deepdive.py first")
    print(f"Loading cache: {CACHE_PATH}")
    cache = torch.load(CACHE_PATH, weights_only=False)

    ckpt = torch.load(os.path.join(SCRIPT_DIR, "models", "av_fused.pt"),
                       weights_only=False)
    idx_to_label = ckpt["idx_to_label"]

    D5_1_layer_probe(cache, idx_to_label)
    D5_2_within_av_cka(cache)
    D5_3_info_plane(cache)
    D5_9_cross_model_cka(cache)
    D5_11_v_modulation(cache)
    D5_12_rsa_trajectory(cache, idx_to_label)

    print("\nPhase F done.")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("D5_") and (f.endswith(".csv") or f.endswith(".png")):
            print(f"  {f}")


if __name__ == "__main__":
    main()
