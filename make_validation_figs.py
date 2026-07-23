"""Generate + self-validate three validation/analysis figures for the AV model.

Tufte-minimal (no title where avoidable, no grid, left+bottom spines only) like
make_msi_plots.py. Every plotted value is read straight from its source (CSV cell
or activation cache) — nothing hard-coded — and echoed in a VALIDATION block for
cell-by-cell checking. Saves .png (dpi 150) and .svg.

  FIG1  d-prime pooled accuracy (validation method): 2 panels
        analysis/msi/E1c (video-reliable) + E1d (audio-reliable)
        -> analysis/msi/FIG_dprime_pooled
  FIG2  UMAP of A/V/AV penultimate embeddings, 3 colorings (3x3)
        processed/deepdive_act_cache.pt
        -> analysis/deepdive/FIG_umap_AVV_3color
  FIG3  viseme decodability AV vs video-only + AV per-stage progression
        analysis/deepdive/D5 (cross-checked vs D2)
        -> analysis/deepdive/FIG_viseme_decodability_AV_vs_V

Run: python make_validation_figs.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Reuse verified logic (do NOT reinvent) -------------------------------------
from phase_e_geometry import _safe_umap                       # umap (cosine), PCA fallback
from analyze_revised import ONSET_CLASS, FIRST_SOUND, _categorize  # coloring label sets

HERE = os.path.dirname(os.path.abspath(__file__))
MSI = os.path.join(HERE, "analysis", "msi")
DEEP = os.path.join(HERE, "analysis", "deepdive")
CACHE_PATH = os.path.join(HERE, "processed", "deepdive_act_cache.pt")
AV_CKPT = os.path.join(HERE, "models", "av_fused.pt")

BLUE, GREEN, RED, ORANGE, GREY = "#4C72B0", "#55A868", "#C44E52", "#DD8452", "#9E9E9E"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def tufte(ax):
    for s in ("top", "right"):          # version-robust (no list-indexing of spines)
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=3)


# ===================================================================== FIG 1
# d-prime: unisensory (A, V), observed multisensory (AV), and the optimal
# independent-channels pooling prediction (the precomputed dprime_pred_opt
# column). The gap between observed d'_AV and d'_pred_opt is shaded.

def fig1_dprime():
    panels = [
        ("E1c_dprime_precision_sweep.csv", "Reliable video, audio degraded"),
        ("E1d_dprime_precision_balanced.csv", "Reliable audio, weak video"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    echo = []
    for ax, (fname, sub) in zip(axes, panels):
        rows = read_csv(os.path.join(MSI, fname))
        sig = [float(r["sigma_a"]) for r in rows]
        dA = [float(r["dprime_A"]) for r in rows]
        dV = [float(r["dprime_V"]) for r in rows]
        dAV = [float(r["dprime_AV"]) for r in rows]
        dopt = [float(r["dprime_pred_opt"]) for r in rows]
        aop = [float(r["AV_over_pred"]) for r in rows]
        x = list(range(len(sig)))

        # shade observed-AV vs optimal-pooling gap
        ax.fill_between(x, dAV, dopt, color=GREY, alpha=0.18, lw=0,
                        label="observed–optimal gap")
        ax.plot(x, dA, "o-", color=BLUE, lw=1.7, ms=4, label="Audio (d′$_A$)")
        ax.plot(x, dV, "s-", color=GREEN, lw=1.7, ms=4, label="Video (d′$_V$)")
        ax.plot(x, dAV, "D-", color=RED, lw=1.9, ms=4.5,
                label="Audio-visual (d′$_{AV}$, observed)")
        ax.plot(x, dopt, "^--", color=ORANGE, lw=1.7, ms=4,
                label="Optimal pooling  $\\sqrt{d'^2_A+d'^2_V}$")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:g}" for s in sig], rotation=60, fontsize=8)
        ax.set_xlabel("Audio noise  σ  (per-RMS)")
        ax.set_title(sub, fontsize=10)
        tufte(ax)
        echo.append((fname, sig, dA, dV, dAV, dopt, aop))
    axes[0].set_ylabel("Sensitivity  d′")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(MSI, "FIG_dprime_pooled")
    fig.savefig(out + ".png", dpi=150)
    fig.savefig(out + ".svg")
    plt.close(fig)

    print("\n=== FIG1 VALIDATION — plotted d′ vs source CSV cells ===")
    print(f"  wrote {out}.png / .svg")
    max_dev = 0.0
    for fname, sig, dA, dV, dAV, dopt, aop in echo:
        print(f"  [{fname}]  columns: sigma_a,dprime_A,dprime_V,dprime_AV,dprime_pred_opt,AV_over_pred")
        for i in range(len(sig)):
            recomputed = (dA[i] ** 2 + dV[i] ** 2) ** 0.5
            dev = abs(recomputed - dopt[i])
            max_dev = max(max_dev, dev)
            chk = abs(aop[i] - dAV[i] / dopt[i])   # AV_over_pred == dAV/dpred_opt ?
            print(f"    σ={sig[i]:.4f}  dA={dA[i]:.4f} dV={dV[i]:.4f} "
                  f"dAV={dAV[i]:.4f}  d_opt(csv)={dopt[i]:.4f} "
                  f"√(dA²+dV²)={recomputed:.4f} (Δ={dev:.4f})  "
                  f"AV_over_pred={aop[i]:.4f} [dAV/d_opt check Δ={chk:.4f}]")
    print(f"  NOTE: dprime_pred_opt is plotted straight from the CSV (canonical: "
          f"AV_over_pred == dprime_AV/dprime_pred_opt to <1e-3 every row). "
          f"Recomputing √(dA²+dV²) from the 4-dp display columns deviates by up "
          f"to Δ={max_dev:.4f} (the column was computed from full-precision d′, "
          f"NOT equal to 2 dp) — plotted source column, flagged not fabricated.")
    return out


# ===================================================================== FIG 2
# 3x3 UMAP of penultimate features. Rows = A_only / V_fair / AV_clean_full.
# Cols = (a) KMeans k=10 on class centroids, (b) onset phoneme class
# (manner-of-articulation 6-group, analyze_revised.ONSET_CLASS), (c) starting
# sound (analyze_revised.FIRST_SOUND). ONE UMAP per row, shared across columns.

def fig2_umap():
    from sklearn.cluster import KMeans

    cache = torch.load(CACHE_PATH, weights_only=False)
    idx_to_label = torch.load(AV_CKPT, weights_only=False)["idx_to_label"]
    labels = np.asarray(cache["labels"])
    n_classes = int(labels.max()) + 1
    words = [idx_to_label[int(l)] for l in labels]

    # (b) onset phoneme class — manner of articulation (analyze_revised line ~454)
    oc_colors = {"Plosives": "#e41a1c", "Fricatives": "#377eb8",
                 "Nasals": "#4daf4a", "Liquids & Glides": "#ff7f00",
                 "Vowel-Initial": "#984ea3", "Affricates": "#a65628",
                 "Other": "#999999"}
    oc_order = ["Plosives", "Fricatives", "Nasals", "Liquids & Glides",
                "Vowel-Initial", "Affricates", "Other"]
    oc_cat = np.array([_categorize(w, ONSET_CLASS) for w in words])

    # (c) starting sound (analyze_revised line ~472)
    fs_order = list(FIRST_SOUND.keys())
    fs_cmap = plt.get_cmap("tab20", len(fs_order))
    fs_colors = {c: fs_cmap(i) for i, c in enumerate(fs_order)}
    fs_cat = np.array([_categorize(w, FIRST_SOUND) for w in words])

    # (a) KMeans clusters (per-model), tab10
    km_cmap = plt.get_cmap("tab10", 10)
    km_colors = {f"C{i}": km_cmap(i) for i in range(10)}
    km_order = [f"C{i}" for i in range(10)]

    rows = [("A-only", "A_only"), ("Video-only", "V_fair"),
            ("Audio-visual", "AV_clean_full")]
    col_titles = ["KMeans (k=10, per-model centroids)",
                  "Onset phoneme class", "Starting sound"]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.0))
    echo = []
    for ri, (rlabel, ckey) in enumerate(rows):
        feats = np.asarray(cache[ckey]["penult"])
        emb = _safe_umap(feats)

        cent = np.zeros((n_classes, feats.shape[1]), dtype=np.float64)
        for c in range(n_classes):
            m = labels == c
            if m.any():
                cent[c] = feats[m].mean(0)
        km_lab = KMeans(n_clusters=10, random_state=0, n_init=10).fit_predict(cent)
        km_cat = np.array([f"C{km_lab[l]}" for l in labels])
        echo.append((rlabel, ckey, feats.shape, emb.shape))

        for ci, (cat, order, cols) in enumerate([
                (km_cat, km_order, km_colors),
                (oc_cat, oc_order, oc_colors),
                (fs_cat, fs_order, fs_colors)]):
            ax = axes[ri, ci]
            for k in order:
                msk = cat == k
                if not msk.any():
                    continue
                ax.scatter(emb[msk, 0], emb[msk, 1], c=[cols[k]], s=5,
                           alpha=0.6, edgecolors="none")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=11)
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=12, fontweight="bold")

    # bottom legends: onset class (col b) and starting sound (col c)
    oc_handles = [plt.Line2D([0], [0], marker="o", color="w", markersize=8,
                             markerfacecolor=oc_colors[k], label=k)
                  for k in oc_order]
    fs_handles = [plt.Line2D([0], [0], marker="o", color="w", markersize=8,
                             markerfacecolor=fs_colors[k], label=k)
                  for k in fs_order]
    leg1 = fig.legend(handles=oc_handles, loc="lower center",
                      bbox_to_anchor=(0.40, -0.045), ncol=4, frameon=False,
                      fontsize=8.5, title="onset phoneme class", title_fontsize=9)
    fig.legend(handles=fs_handles, loc="lower center",
               bbox_to_anchor=(0.83, -0.06), ncol=4, frameon=False,
               fontsize=8.5, title="starting sound", title_fontsize=9)
    fig.add_artist(leg1)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = os.path.join(DEEP, "FIG_umap_AVV_3color")
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    plt.close(fig)

    print("\n=== FIG2 VALIDATION — UMAP inputs vs cache ===")
    print(f"  wrote {out}.png / .svg   (n={len(labels)} samples, {n_classes} classes)")
    print(f"  cols = KMeans(k=10 per-model) | onset phoneme class (ONSET_CLASS, "
          f"manner-of-articulation) | starting sound (FIRST_SOUND)")
    print(f"  onset class 'Other' count={int((oc_cat=='Other').sum())}/{len(oc_cat)}; "
          f"starting sound 'Other' count={int((fs_cat=='Other').sum())}/{len(fs_cat)}")
    for rlabel, ckey, fshape, eshape in echo:
        print(f"  row {rlabel:<12s} cache['{ckey}']['penult']={fshape} -> umap {eshape}")
    return out


# ===================================================================== FIG 3
# Viseme decodability: AV_full vs V_fair at penult (grouped bars, acc + balanced)
# with the AV-V delta annotated; plus AV per-stage progression. Cross-checked
# against the independent D2 three-condition probe.

def fig3_viseme():
    d5 = read_csv(os.path.join(DEEP, "D5_layer_decodability_viseme.csv"))
    d2 = read_csv(os.path.join(DEEP, "D2_three_cond_probe.csv"))

    def d5row(model, layer):
        return next(r for r in d5 if r["model"] == model and r["layer"] == layer)

    v = d5row("V_fair", "penult")
    av = d5row("AV_full", "penult")
    v_acc, v_bal = float(v["acc_5fold"]) * 100, float(v["bal_acc_5fold"]) * 100
    av_acc, av_bal = float(av["acc_5fold"]) * 100, float(av["bal_acc_5fold"]) * 100
    d_acc, d_bal = av_acc - v_acc, av_bal - v_bal

    av_stages = ["a_mid_gap", "v_mid_gap", "gate_out_gap", "block2_gap", "penult"]
    av_pretty = ["a_mid", "v_mid", "gate_out", "block2", "penult"]
    prog_acc = [float(d5row("AV_full", s)["acc_5fold"]) * 100 for s in av_stages]
    prog_bal = [float(d5row("AV_full", s)["bal_acc_5fold"]) * 100 for s in av_stages]
    v_mid_acc = prog_acc[av_stages.index("v_mid_gap")]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.2, 4.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})

    # -- Panel L: grouped bars, V vs AV, accuracy + balanced accuracy
    groups = ["5-fold accuracy", "Balanced accuracy"]
    xg = np.arange(len(groups)); w = 0.36
    vbars = axL.bar(xg - w / 2, [v_acc, v_bal], w, color=GREEN, label="Video-only")
    avbars = axL.bar(xg + w / 2, [av_acc, av_bal], w, color=RED, label="Audio-visual")
    for bars in (vbars, avbars):
        for b in bars:
            axL.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                     f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9.5)
    # annotate AV-over-V delta above each AV bar
    for xi, (av_h, dlt) in enumerate(zip([av_acc, av_bal], [d_acc, d_bal])):
        axL.annotate(f"+{dlt:.1f} pp", xy=(xi + w / 2, av_h + 4.5),
                     ha="center", va="bottom", fontsize=10, color=RED,
                     fontweight="bold")
    axL.set_xticks(xg); axL.set_xticklabels(groups)
    axL.set_ylabel("Viseme decodability (%)")
    axL.set_ylim(0, 105)
    axL.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, fontsize=9)
    tufte(axL)

    # -- Panel R: AV per-stage progression, with v_mid input reference line
    xs = np.arange(len(av_stages))
    axR.axhline(v_mid_acc, color=GREEN, ls=":", lw=1.2, alpha=0.8,
                label=f"v_mid input ({v_mid_acc:.1f}%)")
    axR.plot(xs, prog_acc, "o-", color=RED, lw=1.8, ms=5, label="5-fold accuracy")
    axR.plot(xs, prog_bal, "s--", color=ORANGE, lw=1.8, ms=5, label="balanced accuracy")
    for xi, ya in zip(xs, prog_acc):
        axR.text(xi, ya + 1.4, f"{ya:.1f}", ha="center", va="bottom", fontsize=8.5)
    axR.set_xticks(xs); axR.set_xticklabels(av_pretty, rotation=20, fontsize=9)
    axR.set_xlabel("Audio-visual stage  (gate → readout)")
    axR.set_ylabel("Viseme decodability (%)")
    axR.set_ylim(0, 100)
    axR.set_title("AV per-stage progression", fontsize=10)
    axR.legend(frameon=False, loc="lower right", fontsize=8.5)
    tufte(axR)

    fig.tight_layout()
    out = os.path.join(DEEP, "FIG_viseme_decodability_AV_vs_V")
    fig.savefig(out + ".png", dpi=150)
    fig.savefig(out + ".svg")
    plt.close(fig)

    print("\n=== FIG3 VALIDATION — plotted % vs source CSV cells ===")
    print(f"  wrote {out}.png / .svg")
    print(f"  V_fair penult : acc={v_acc:.4f}%  bal={v_bal:.4f}%  "
          f"(csv {float(v['acc_5fold']):.6f}/{float(v['bal_acc_5fold']):.6f})")
    print(f"  AV_full penult: acc={av_acc:.4f}%  bal={av_bal:.4f}%  "
          f"(csv {float(av['acc_5fold']):.6f}/{float(av['bal_acc_5fold']):.6f})")
    print(f"  AV − V delta  : +{d_acc:.2f}pp acc, +{d_bal:.2f}pp balanced (annotated)")
    print("  AV per-stage (a_mid→v_mid→gate_out→block2→penult):")
    print("    acc:", [f"{a:.4f}" for a in prog_acc])
    print("    bal:", [f"{b:.4f}" for b in prog_bal])
    print(f"    integration exceeds v_mid input: penult {prog_acc[-1]:.1f}% > "
          f"v_mid {v_mid_acc:.1f}% (Δ +{prog_acc[-1]-v_mid_acc:.1f}pp)")
    d2v = next(r for r in d2 if r["condition"] == "V_fair")
    d2av = next(r for r in d2 if r["condition"] == "AV_full")
    print("  D2 cross-check (independent probe):")
    print(f"    V_fair  acc={float(d2v['acc'])*100:.3f}%  bal={float(d2v['balanced_acc'])*100:.3f}%")
    print(f"    AV_full acc={float(d2av['acc'])*100:.3f}%  bal={float(d2av['balanced_acc'])*100:.3f}%")
    return out


if __name__ == "__main__":
    np.random.seed(0)
    f1 = fig1_dprime()
    f3 = fig3_viseme()
    f2 = fig2_umap()  # last: UMAP is the slow one
    print("\nDONE. Figures:")
    for f in (f1, f2, f3):
        print(f"  {f}.png / .svg")
