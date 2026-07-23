#!/usr/bin/env python3
"""Phase E — geometry: UMAPs, integration-driver regression, dendrograms
(D2.1–D2.7). Reads `processed/deepdive_act_cache.pt` from Phase A.
Run: `python phase_e_geometry.py`."""

from __future__ import annotations

import csv
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_av_phonetics import viseme_class as _viseme_from_label


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "deepdive")
CACHE_PATH = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
os.makedirs(OUT_DIR, exist_ok=True)

VISEME_COLORS = {
    "bilabial_bpm":   "#cc6677",
    "labiodental_fv": "#332288",
    "labiovelar_w":   "#117733",
    "lingual":        "#999933",
    "glottal_h":      "#882255",
    "vowel_initial":  "#44aa99",
    "other":          "#cccccc",
}


def _safe_umap(X: np.ndarray, n_neighbors=15, random_state=0):
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors,
                             n_components=2,
                             random_state=random_state,
                             metric="cosine")
        return reducer.fit_transform(X.astype(np.float32))
    except Exception as e:
        warnings.warn(f"UMAP failed ({e}); falling back to PCA")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=random_state).fit_transform(X)


# D2.1 — Three-condition UMAP from AV-only

def D2_1_three_cond_umap(cache, idx_to_label):
    print("\n  D2.1 — Three-condition UMAP (AV-only)")
    labels = cache["labels"]
    vis = np.asarray([_viseme_from_label(idx_to_label[int(l)])
                       for l in labels])
    full = cache["AV_clean_full"]["penult"]
    vz = cache["AV_clean_v_zero"]["penult"]
    az = cache["AV_clean_audio_zero"]["penult"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, X) in zip(axes,
                              [("AV full", full),
                               ("audio only (v_zero)", vz),
                               ("video only (audio_zero)", az)]):
        emb = _safe_umap(X)
        colors = [VISEME_COLORS.get(v, "#cccccc") for v in vis]
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4, alpha=0.6)
        ax.set_title(f"{name} (n={len(X)})")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.grid(alpha=0.2)
    # Legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=c, markersize=8, label=k)
                for k, c in VISEME_COLORS.items() if k != "other"]
    axes[0].legend(handles=handles, loc="lower left", fontsize=7,
                    bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "D2_umap_3cond.png")
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  wrote {out_png}")


# D2.2 — Joint UMAP across 3 models

def D2_2_joint_umap(cache, idx_to_label):
    print("\n  D2.2 — Joint UMAP across {A_only, V_fair, AV_full}")
    labels = cache["labels"]
    a = cache["A_only"]["penult"]
    v = cache["V_fair"]["penult"]
    av = cache["AV_clean_full"]["penult"]
    # Pad V to 128 dims if necessary
    if v.shape[1] != a.shape[1]:
        pad = np.zeros((v.shape[0], a.shape[1] - v.shape[1]),
                        dtype=v.dtype)
        v = np.concatenate([v, pad], axis=1)
    X = np.concatenate([a, v, av], axis=0)
    src = (["A_only"] * len(a) + ["V_fair"] * len(v)
           + ["AV_full"] * len(av))
    print(f"    embedding {X.shape}...")
    emb = _safe_umap(X)

    model_colors = {"A_only": "#4477aa", "V_fair": "#117733",
                     "AV_full": "#cc6677"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, color in model_colors.items():
        sel = np.asarray([s == name for s in src])
        ax.scatter(emb[sel, 0], emb[sel, 1], c=color, s=4,
                    alpha=0.5, label=name)
    ax.set_title("D2.2 — Joint UMAP: A vs V vs AV penultimates")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "D2_umap_joint_3model.png")
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  wrote {out_png}")


# D2.3 — Integration-driver regression (Random-Forest feature importance)

def D2_3_drivers(cache, idx_to_label):
    print("\n  D2.3 — Integration-driver regression")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    labels = cache["labels"]
    a_logits = cache["A_only"]["logits"]
    av_logits = cache["AV_clean_full"]["logits"]
    a_pred = a_logits.argmax(1)
    av_pred = av_logits.argmax(1)

    a_correct = (a_pred == labels).astype(np.int64)
    av_correct = (av_pred == labels).astype(np.int64)
    delta_flip = av_correct - a_correct      # {-1, 0, +1}

    # Log-confidence on true class
    def _logp(logits, lab):
        # log softmax then index true class
        x = logits - logits.max(axis=1, keepdims=True)
        lse = np.log(np.exp(x).sum(axis=1, keepdims=True))
        return (x - lse)[np.arange(len(lab)), lab]
    delta_logconf = _logp(av_logits, labels) - _logp(a_logits, labels)

    # Margin-to-second-class
    def _margin(logits):
        top2 = np.sort(logits, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]
    delta_margin = _margin(av_logits) - _margin(a_logits)

    # Features: onset (one-hot), viseme (one-hot), word-length-in-chars,
    # vowel-initial flag, syllable count proxy (=#vowels). Speaker/gender
    # would require parsing audio paths; skip for now (mid-quality proxy).
    onsets = [_viseme_from_label(idx_to_label[int(l)]) for l in labels]
    words = [idx_to_label[int(l)] for l in labels]
    onset_classes = sorted(set(onsets))

    n = len(labels)
    feat_names = []
    cols = []
    for o in onset_classes:
        col = np.asarray([1.0 if x == o else 0.0 for x in onsets])
        cols.append(col); feat_names.append(f"viseme_{o}")
    word_len = np.asarray([len(w) for w in words], dtype=np.float32)
    cols.append(word_len); feat_names.append("word_len")
    n_vowels = np.asarray([sum(1 for c in w if c.lower() in "aeiou")
                            for w in words], dtype=np.float32)
    cols.append(n_vowels); feat_names.append("n_vowels")
    vowel_init = np.asarray([1.0 if w and w[0].lower() in "aeiou" else 0.0
                              for w in words])
    cols.append(vowel_init); feat_names.append("vowel_initial")
    X = np.stack(cols, axis=1)
    print(f"    features: {X.shape}, names: {feat_names}")

    results = {}
    for target_name, y_t in (("delta_flip", delta_flip),
                               ("delta_logconf", delta_logconf),
                               ("delta_margin", delta_margin)):
        if target_name == "delta_flip":
            clf = RandomForestClassifier(
                n_estimators=200, random_state=0, n_jobs=-1)
            clf.fit(X, y_t)
            importances = clf.feature_importances_
        else:
            reg = RandomForestRegressor(
                n_estimators=200, random_state=0, n_jobs=-1)
            reg.fit(X, y_t)
            importances = reg.feature_importances_
        results[target_name] = importances

    # Save + Spearman rank stability across the 3 target choices
    from scipy.stats import spearmanr
    targets = list(results.keys())
    print("\n    Spearman ρ of feature ranks across targets:")
    for i, t1 in enumerate(targets):
        for t2 in targets[i+1:]:
            rho, _ = spearmanr(results[t1], results[t2])
            print(f"      {t1} ↔ {t2}: ρ = {rho:.3f}")

    out_csv = os.path.join(OUT_DIR, "D2_integration_drivers.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["feature", "rf_importance_delta_flip",
                    "rf_importance_delta_logconf",
                    "rf_importance_delta_margin"])
        for i, name in enumerate(feat_names):
            w.writerow([name,
                        f"{results['delta_flip'][i]:.6f}",
                        f"{results['delta_logconf'][i]:.6f}",
                        f"{results['delta_margin'][i]:.6f}"])
    print(f"  wrote {out_csv}")
    top_flip = sorted(zip(feat_names, results["delta_flip"]),
                       key=lambda x: -x[1])[:6]
    print("    top-6 features (Δ-flip):")
    for n, v in top_flip:
        print(f"      {n:<20s} {v:.4f}")


# D2.5 — Three-condition viseme probe (already covered in Phase A class probe
# but viseme-specific)

def D2_5_three_cond_viseme(cache, idx_to_label):
    print("\n  D2.5 — Three-condition viseme probe")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    labels = cache["labels"]
    visemes = np.asarray([_viseme_from_label(idx_to_label[int(l)])
                           for l in labels])
    keep = visemes != "other"

    out_csv = os.path.join(OUT_DIR, "D2_three_cond_probe.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["condition", "acc", "balanced_acc"])
        for name, feats in [
            ("AV_full",      cache["AV_clean_full"]["penult"]),
            ("AV_v_zero",    cache["AV_clean_v_zero"]["penult"]),
            ("AV_audio_zero",cache["AV_clean_audio_zero"]["penult"]),
            ("A_only",       cache["A_only"]["penult"]),
            ("V_fair",       cache["V_fair"]["penult"]),
        ]:
            X = feats[keep]
            y = visemes[keep]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            accs, bal_accs = [], []
            for tr, te in skf.split(X, y):
                clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                accs.append(accuracy_score(y[te], pred))
                bal_accs.append(balanced_accuracy_score(y[te], pred))
            a = float(np.mean(accs)); ba = float(np.mean(bal_accs))
            print(f"    {name:>14s}: acc={a:.4%}  bal_acc={ba:.4%}")
            w.writerow([name, f"{a:.6f}", f"{ba:.6f}"])
    print(f"  wrote {out_csv}")


# D2.7 — Class-mean dendrograms (3 models side-by-side)

def D2_7_dendrograms(cache, idx_to_label):
    print("\n  D2.7 — Class-mean dendrograms (Ward, cosine)")
    from scipy.cluster.hierarchy import dendrogram, linkage

    labels = cache["labels"]
    n_classes = int(labels.max()) + 1

    def _class_means(feats):
        means = np.zeros((n_classes, feats.shape[1]), dtype=np.float64)
        counts = np.zeros(n_classes, dtype=np.int64)
        for f, l in zip(feats, labels):
            means[int(l)] += f
            counts[int(l)] += 1
        nz = counts > 0
        means[nz] /= counts[nz, None]
        return means.astype(np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    for ax, (name, feats) in zip(axes, [
        ("A_only", cache["A_only"]["penult"]),
        ("V_fair", cache["V_fair"]["penult"]),
        ("AV_full", cache["AV_clean_full"]["penult"]),
    ]):
        means = _class_means(feats)
        Z = linkage(means, method="ward")
        leaves = [idx_to_label[i] for i in range(n_classes)]
        dendrogram(Z, labels=leaves, ax=ax, leaf_font_size=5,
                    color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title(f"D2.7 — Ward dendrogram ({name})")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "D2_dendrograms_3model.png")
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
    print(f"  labels: {cache['labels'].shape}, n_classes={cache['n_classes']}")

    # Build idx_to_label from one of the model checkpoints
    ckpt = torch.load(os.path.join(SCRIPT_DIR, "models", "av_fused.pt"),
                       weights_only=False)
    idx_to_label = ckpt["idx_to_label"]

    D2_1_three_cond_umap(cache, idx_to_label)
    D2_2_joint_umap(cache, idx_to_label)
    D2_3_drivers(cache, idx_to_label)
    D2_5_three_cond_viseme(cache, idx_to_label)
    D2_7_dendrograms(cache, idx_to_label)

    print("\nPhase E done.")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("D2_") and (f.endswith(".csv") or f.endswith(".png")):
            print(f"  {f}")


if __name__ == "__main__":
    main()
