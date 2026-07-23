#!/usr/bin/env python3
"""Build the two follow-on MSI figures requested by the lead.

Figure 1 — 4-curve E1 (A-clean / A-rawnoise / AV-clean / AV-rawnoise) σ-sweep.
Figure 2 — gate evolution: α trajectory + modality-dropout bar chart.
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "msi")


def _read_2col(path, x_col, y_col):
    rows = list(csv.DictReader(open(path)))
    x = np.asarray([float(r[x_col]) for r in rows])
    y = np.asarray([float(r[y_col]) for r in rows])
    return x, y


# Figure 1 — 4-curve E1
def fig1_e1_4model() -> None:
    sa_clean,  ya_clean  = _read_2col(
        os.path.join(SCRIPT_DIR, "models", "audio_only_filtered_noise_sweep.csv"),
        "sigma_per_rms", "val_acc")
    sa_raw,    ya_raw    = _read_2col(
        os.path.join(SCRIPT_DIR, "models", "audio_only_rawnoise_filtered_noise_sweep.csv"),
        "sigma_per_rms", "val_acc")
    sav_clean, yav_clean = _read_2col(
        os.path.join(SCRIPT_DIR, "models", "av_fused_av_noise_sweep.csv"),
        "sigma_per_rms", "AV_acc")
    sav_raw,   yav_raw   = _read_2col(
        os.path.join(SCRIPT_DIR, "models", "av_fused_rawnoise_av_noise_sweep.csv"),
        "sigma_per_rms", "AV_acc")

    out_csv = os.path.join(OUT_DIR, "E1_inverse_effectiveness_4model.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["sigma_per_rms", "A_clean", "A_rawnoise",
                    "AV_clean", "AV_rawnoise"])
        # Use the union of σ levels (they should match, but be safe)
        all_sigmas = sorted(set(sa_clean.tolist())
                            | set(sa_raw.tolist())
                            | set(sav_clean.tolist())
                            | set(sav_raw.tolist()))

        def _at(xs, ys, target):
            for x, y in zip(xs, ys):
                if abs(x - target) < 1e-9:
                    return y
            return float("nan")

        for s in all_sigmas:
            w.writerow([
                f"{s:.4f}",
                f"{_at(sa_clean,  ya_clean,  s):.6f}",
                f"{_at(sa_raw,    ya_raw,    s):.6f}",
                f"{_at(sav_clean, yav_clean, s):.6f}",
                f"{_at(sav_raw,   yav_raw,   s):.6f}",
            ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    style = dict(marker="o", linewidth=2.0)
    axes[0].plot(sa_clean,  ya_clean,  color="#4477aa", label="A-clean",     **style)
    axes[0].plot(sa_raw,    ya_raw,    color="#225588", linestyle="--",
                 label="A-rawnoise", **style)
    axes[0].plot(sav_clean, yav_clean, color="#cc6677", label="AV-clean",    **style)
    axes[0].plot(sav_raw,   yav_raw,   color="#883344", linestyle="--",
                 label="AV-rawnoise", **style)
    axes[0].set_xscale("symlog", linthresh=0.001)
    axes[0].set_xlabel("σ_a / audio_rms")
    axes[0].set_ylabel("clean-class val acc (noisy audio at inference)")
    axes[0].set_title("Inverse effectiveness, 4-model")
    axes[0].axhline(1 / 180, color="gray", linewidth=1, alpha=0.4,
                     label=f"chance = 1/180 ≈ 0.56 %")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    # AV − A gaps — align by exact σ value (A-only sweeps and AV sweeps were
    # run with slightly different σ grids).
    def _gap(sa, ya, sav, yav):
        gx, gy = [], []
        for s, v in zip(sav, yav):
            for sa_, ya_ in zip(sa, ya):
                if abs(sa_ - s) < 1e-9:
                    gx.append(s)
                    gy.append(v - ya_)
                    break
        return np.asarray(gx), np.asarray(gy)
    gx_c, gy_c = _gap(sa_clean, ya_clean, sav_clean, yav_clean)
    gx_r, gy_r = _gap(sa_raw,   ya_raw,   sav_raw,   yav_raw)
    axes[1].plot(gx_c, gy_c, color="#cc6677",
                 label="AV-clean − A-clean", **style)
    axes[1].plot(gx_r, gy_r, color="#883344",
                 linestyle="--", label="AV-rawnoise − A-rawnoise", **style)
    axes[1].axhline(0, color="gray", linewidth=1, alpha=0.5)
    axes[1].set_xscale("symlog", linthresh=0.001)
    axes[1].set_xlabel("σ_a / audio_rms")
    axes[1].set_ylabel("AV − A (gap, pp)")
    axes[1].set_title("Multisensory enhancement vs noise")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "E1_inverse_effectiveness_4model.png"), dpi=140)
    plt.close(fig)
    print(f"  wrote {os.path.join(OUT_DIR, 'E1_inverse_effectiveness_4model.png')}")


# Figure 2 — gate / α evolution + modality-dropout bars
def fig2_gate_evolution() -> None:
    # Read α trajectories
    def _alpha_curve(path):
        rows = list(csv.DictReader(open(path)))
        ep = np.asarray([int(r["epoch"]) for r in rows])
        a = np.asarray([float(r["alpha"]) for r in rows])
        return ep, a

    ep_clean, a_clean = _alpha_curve(os.path.join(
        SCRIPT_DIR, "analysis", "av_fused_curves.csv"))
    ep_noisy, a_noisy = _alpha_curve(os.path.join(
        SCRIPT_DIR, "analysis", "av_fused_noisy_curves.csv"))
    ep_raw, a_raw = _alpha_curve(os.path.join(
        SCRIPT_DIR, "analysis", "av_fused_rawnoise_curves.csv"))

    # Read modality-dropout
    def _dropout(path):
        rows = list(csv.DictReader(open(path)))
        return {r["condition"]: float(r["AV_acc"]) for r in rows}

    do_clean = _dropout(os.path.join(SCRIPT_DIR, "models",
                                      "av_fused_modality_dropout.csv"))
    do_noisy = _dropout(os.path.join(SCRIPT_DIR, "models",
                                      "av_fused_noisy_modality_dropout.csv"))
    do_raw = _dropout(os.path.join(SCRIPT_DIR, "models",
                                    "av_fused_rawnoise_modality_dropout.csv"))

    # CSV
    out_csv = os.path.join(OUT_DIR, "gate_evolution.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "alpha_final", "AV_full",
                    "audio_only_video_zero", "video_only_audio_zero", "both_zero"])
        for name, a, do in [
            ("AV_clean",     a_clean[-1],  do_clean),
            ("AV_mel_noisy", a_noisy[-1], do_noisy),
            ("AV_rawnoise",  a_raw[-1],   do_raw),
        ]:
            w.writerow([
                name, f"{a:.4f}",
                f"{do['AV_full']:.6f}",
                f"{do['audio_only_video_zero']:.6f}",
                f"{do['video_only_audio_zero']:.6f}",
                f"{do['both_zero']:.6f}",
            ])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top — α trajectories
    ax = axes[0]
    ax.plot(ep_clean, a_clean, color="#cc6677", label=f"AV-clean (final α={a_clean[-1]:.2f})",
            linewidth=2)
    ax.plot(ep_noisy, a_noisy, color="#888844", label=f"AV mel-noisy (final α={a_noisy[-1]:.2f})",
            linewidth=2, alpha=0.85)
    ax.plot(ep_raw, a_raw, color="#883344", label=f"AV-rawnoise (final α={a_raw[-1]:.2f})",
            linewidth=2)
    ax.axhline(0.2, color="gray", linewidth=1, alpha=0.5,
                label="init α=0.2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learned α (gate gain)")
    ax.set_title("Gate gain α grows with audio-noise training")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom — modality-dropout bars
    ax = axes[1]
    conditions = ["AV_full", "audio_only_video_zero", "video_only_audio_zero", "both_zero"]
    cond_labels = ["AV (both)", "audio only\n(video=0)", "video only\n(audio=0)", "both=0"]
    models_ = [
        ("AV-clean",     "#cc6677", do_clean),
        ("AV mel-noisy", "#888844", do_noisy),
        ("AV-rawnoise",  "#883344", do_raw),
    ]
    n_cond = len(conditions)
    n_mod = len(models_)
    x = np.arange(n_cond)
    width = 0.25
    for i, (name, color, do) in enumerate(models_):
        offset = (i - (n_mod - 1) / 2) * width
        bars = ax.bar(x + offset,
                       [do[c] for c in conditions],
                       width=width, label=name, color=color)
        for j, bar in enumerate(bars):
            v = do[conditions[j]]
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                     f"{v*100:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("clean val_acc")
    ax.set_title("Modality-dropout sanity (clean-input inference)")
    ax.axhline(1 / 180, color="gray", linewidth=1, alpha=0.5,
                label="chance ≈ 0.56 %")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "E_gate_evolution.png"), dpi=140)
    plt.close(fig)
    print(f"  wrote {os.path.join(OUT_DIR, 'E_gate_evolution.png')}")
    print(f"  wrote {out_csv}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Figure 1 — 4-curve E1...")
    fig1_e1_4model()
    print("Figure 2 — gate evolution + modality dropout...")
    fig2_gate_evolution()
    print("Done.")


if __name__ == "__main__":
    main()
