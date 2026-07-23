#!/usr/bin/env python3
"""Build the Direction-1 figures (3×3 matrix, σ_v / frame-drop curves,
iso-perf contour, heterogeneous-noise comparison) from the deepdive CSVs."""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "deepdive")


def _read_csv(path):
    return list(csv.DictReader(open(path)))


# Fig 1 — 3×3 heatmap

def fig_3x3() -> None:
    rows = _read_csv(os.path.join(OUT_DIR, "D1_3x3_clean.csv"))
    models_ = ["A_trained", "V_trained", "AV_trained"]
    inputs = ["input_A_only", "input_V_only", "input_AV"]
    grid = np.full((3, 3), np.nan)
    for i, m in enumerate(models_):
        r = next(rr for rr in rows if rr["trained_model"] == m)
        for j, c in enumerate(inputs):
            v = r[c]
            if not v.startswith("n/a"):
                grid[i, j] = float(v)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["A-only input", "V-only input", "AV input"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["A-trained", "V-trained", "AV-trained"])
    for i in range(3):
        for j in range(3):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                         color="gray", fontsize=10)
            else:
                ax.text(j, i, f"{v*100:.2f}%", ha="center", va="center",
                         color="white" if v < 0.6 else "black", fontsize=11)
    ax.set_title("D1.1 — 3×3 accuracy matrix (clean inputs)")
    fig.colorbar(im, ax=ax, label="val_acc")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "D1_3x3_matrix.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


# Fig 2 — σ_v curves

def fig_sigma_v() -> None:
    rows = _read_csv(os.path.join(OUT_DIR, "D1_sigma_v_sweep.csv"))
    sv = np.asarray([float(r["sigma_v_per_pixstd"]) for r in rows])
    yv = np.asarray([float(r["V_only_acc"]) for r in rows])
    yav = np.asarray([float(r["AV_acc"]) for r in rows])
    y_raw = None
    if rows and "AV_rawnoise_acc" in rows[0]:
        y_raw = np.asarray([float(r["AV_rawnoise_acc"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(sv, yv,  "o-", color="#4477aa", label="V-only-fair", linewidth=2)
    ax.plot(sv, yav, "s-", color="#cc6677", label="AV-clean",   linewidth=2)
    if y_raw is not None:
        ax.plot(sv, y_raw, "^--", color="#883344",
                 label="AV-rawnoise", linewidth=2)
    ax.axhline(1 / 180, color="gray", linewidth=1, alpha=0.5,
                label="chance = 1/180")
    ax.set_xlabel("σ_v / per-clip pixel std")
    ax.set_ylabel("val_acc")
    ax.set_title("D1.2 — Visual-noise robustness (per-pixel Gaussian)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "D1_sigma_v_curves.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


# Fig 3 — frame-drop curves

def fig_frame_drop() -> None:
    rows = _read_csv(os.path.join(OUT_DIR, "D1_frame_drop.csv"))
    n = np.asarray([int(r["n_frames_dropped"]) for r in rows])
    yv = np.asarray([float(r["V_only_acc"]) for r in rows])
    yav = np.asarray([float(r["AV_acc"]) for r in rows])
    y_raw = None
    if rows and "AV_rawnoise_acc" in rows[0]:
        y_raw = np.asarray([float(r["AV_rawnoise_acc"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(n, yv,  "o-", color="#4477aa", label="V-only-fair", linewidth=2)
    ax.plot(n, yav, "s-", color="#cc6677", label="AV-clean",   linewidth=2)
    if y_raw is not None:
        ax.plot(n, y_raw, "^--", color="#883344",
                 label="AV-rawnoise", linewidth=2)
    ax.axhline(1 / 180, color="gray", linewidth=1, alpha=0.5,
                label="chance = 1/180")
    ax.set_xlabel("# frames zeroed (out of 50)")
    ax.set_ylabel("val_acc")
    ax.set_title("D1.3 — Temporal occlusion robustness")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "D1_frame_drop_curves.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


# Fig 4 — iso-perf contour

def fig_iso_contour() -> None:
    rows = _read_csv(os.path.join(OUT_DIR, "D1_iso_perf_grid.csv"))

    def _grid_for(name):
        cells = {}
        sas, svs = set(), set()
        for r in rows:
            if r["model"] != name:
                continue
            sa = float(r["sigma_a_per_rms"])
            sv = float(r["sigma_v_per_pixstd"])
            cells[(sa, sv)] = float(r["AV_acc"])
            sas.add(sa); svs.add(sv)
        sas = sorted(sas); svs = sorted(svs)
        if not cells:
            return None
        g = np.full((len(sas), len(svs)), np.nan)
        for i, sa in enumerate(sas):
            for j, sv in enumerate(svs):
                if (sa, sv) in cells:
                    g[i, j] = cells[(sa, sv)]
        return np.asarray(sas), np.asarray(svs), g

    av_grid = _grid_for("AV_clean")
    av_raw_grid = _grid_for("AV_rawnoise")
    grids = [("AV-clean", av_grid)]
    if av_raw_grid is not None:
        grids.append(("AV-rawnoise", av_raw_grid))

    fig, axes = plt.subplots(1, len(grids), figsize=(6 * len(grids), 4.6),
                              squeeze=False)
    for ax, (name, g) in zip(axes[0], grids):
        sas, svs, accs = g
        # Use σ_a on Y, σ_v on X to match the printed table.
        im = ax.imshow(accs, origin="lower", cmap="viridis",
                        vmin=0, vmax=1, aspect="auto",
                        extent=(svs.min(), svs.max(), sas.min(), sas.max()))
        # Overlay numeric labels
        for i, sa in enumerate(sas):
            for j, sv in enumerate(svs):
                v = accs[i, j]
                if not np.isnan(v):
                    ax.text(sv, sa, f"{int(v*100)}",
                             ha="center", va="center",
                             color="white" if v < 0.6 else "black",
                             fontsize=7.5)
        # Iso-perf contour lines
        try:
            cs = ax.contour(svs, sas, accs,
                             levels=[0.85, 0.75, 0.65, 0.50],
                             colors="white", linewidths=1.3)
            ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.2f")
        except Exception:
            pass
        ax.set_xlabel("σ_v / per-clip pixel std")
        ax.set_ylabel("σ_a / audio rms")
        ax.set_title(f"D1.4 — {name} iso-perf grid")
        fig.colorbar(im, ax=ax, label="val_acc")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "D1_iso_perf_contour.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


# Fig 5 — heterogeneous noise (σ_a-only vs σ_v-only vs symmetric)

def fig_heterogeneous() -> None:
    rows = _read_csv(os.path.join(OUT_DIR, "D1_heterogeneous_noise.csv"))
    sa, ya = [], []
    sv, yv = [], []
    sym_idx, ym_sym = [], []
    sym_pairs = []
    for r in rows:
        regime = r["regime"]
        acc = float(r["AV_acc"])
        if regime == "sigma_a_only":
            sa.append(float(r["sigma_a_per_rms"])); ya.append(acc)
        elif regime == "sigma_v_only":
            sv.append(float(r["sigma_v_per_pixstd"])); yv.append(acc)
        elif regime == "both_symmetric":
            sym_idx.append(len(sym_idx))
            ym_sym.append(acc)
            sym_pairs.append((float(r["sigma_a_per_rms"]),
                               float(r["sigma_v_per_pixstd"])))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.plot(sa, ya, "o-", color="#225588", label="σ_a alone (σ_v=0)",
             linewidth=2)
    ax.plot(sv, yv, "s-", color="#cc6677", label="σ_v alone (σ_a=0)",
             linewidth=2)
    ax.axhline(1 / 180, color="gray", linewidth=1, alpha=0.5,
                label="chance = 1/180")
    ax.set_xlabel("σ (per modality unit)")
    ax.set_ylabel("AV val_acc")
    ax.set_title("D1.7 — σ_a-only vs σ_v-only on AV")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[1]
    labels = [f"({a:.3f}, {v:.2f})" for a, v in sym_pairs]
    ax.plot(sym_idx, ym_sym, "d-", color="#883344",
             linewidth=2, label="both symmetric")
    ax.axhline(1 / 180, color="gray", linewidth=1, alpha=0.5,
                label="chance = 1/180")
    ax.set_xticks(sym_idx)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("(σ_a, σ_v)")
    ax.set_ylabel("AV val_acc")
    ax.set_title("D1.7 — both-symmetric noise")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "D1_heterogeneous.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    print("Building Direction-1 figures...")
    fig_3x3()
    if os.path.exists(os.path.join(OUT_DIR, "D1_sigma_v_sweep.csv")):
        fig_sigma_v()
    if os.path.exists(os.path.join(OUT_DIR, "D1_frame_drop.csv")):
        fig_frame_drop()
    if os.path.exists(os.path.join(OUT_DIR, "D1_iso_perf_grid.csv")):
        fig_iso_contour()
    if os.path.exists(os.path.join(OUT_DIR, "D1_heterogeneous_noise.csv")):
        fig_heterogeneous()
    print("Done.")


if __name__ == "__main__":
    main()
