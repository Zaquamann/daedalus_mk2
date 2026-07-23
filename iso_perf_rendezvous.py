#!/usr/bin/env python3
"""D1.5 — iso-performance rendezvous post-processor. For each target acc
in {0.85, 0.75, 0.65, 0.50}, finds the σ_a or σ_v at which each model
hits the target; reports AV's additional noise budget over A-only / V-only.
Writes `analysis/deepdive/D1_iso_perf_lookup.csv`."""

from __future__ import annotations

import csv
import os
from typing import Optional, Sequence

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "deepdive")

GRID_CSV = os.path.join(OUT_DIR, "D1_iso_perf_grid.csv")
SIGMA_V_SWEEP_CSV = os.path.join(OUT_DIR, "D1_sigma_v_sweep.csv")
A_SWEEP_CSV = os.path.join(SCRIPT_DIR, "models",
                           "audio_only_filtered_noise_sweep.csv")
A_RAW_SWEEP_CSV = os.path.join(SCRIPT_DIR, "models",
                                "audio_only_rawnoise_filtered_noise_sweep.csv")

TARGETS = (0.85, 0.75, 0.65, 0.50)


def _read_curve(path: str, x_col: str, y_col: str):
    rows = list(csv.DictReader(open(path)))
    x = np.asarray([float(r[x_col]) for r in rows])
    y = np.asarray([float(r[y_col]) for r in rows])
    order = np.argsort(x)
    return x[order], y[order]


def _read_grid(path: str, model_filter: str):
    """Return dict {(sa, sv): acc} for a single model row in the grid CSV."""
    rows = list(csv.DictReader(open(path)))
    out = {}
    sas, svs = set(), set()
    for r in rows:
        if r["model"] != model_filter:
            continue
        sa = float(r["sigma_a_per_rms"])
        sv = float(r["sigma_v_per_pixstd"])
        out[(sa, sv)] = float(r["AV_acc"])
        sas.add(sa); svs.add(sv)
    return out, sorted(sas), sorted(svs)


def _interp_x_at_y(x: np.ndarray, y: np.ndarray, target: float) -> float:
    """Find x on a monotone-decreasing curve where y crosses `target`.

    Returns NaN if the curve never crosses; otherwise linear interp.
    """
    if len(x) == 0:
        return float("nan")
    # Monotone-decreasing assumption: walk in ascending x.
    # If y[0] < target → never reached.
    if y[0] < target:
        return float("nan")
    # If y[-1] >= target → never fell below.
    if y[-1] >= target:
        return float(x[-1])
    for i in range(len(x) - 1):
        if y[i] >= target and y[i + 1] < target:
            # Linear interp on (x, y).
            t = (target - y[i + 1]) / max(1e-12, y[i] - y[i + 1])
            return float(x[i + 1] - t * (x[i + 1] - x[i]))
    return float("nan")


def _grid_axis_curve(grid, sa_list, sv_list, axis: str):
    """Extract a 1D curve from the 2D grid along σ_a or σ_v at the orthogonal=0."""
    if axis == "sigma_a":
        xs = np.asarray(sa_list)
        ys = np.asarray([grid[(s, 0.0)] for s in sa_list if (s, 0.0) in grid])
        return xs[: len(ys)], ys
    elif axis == "sigma_v":
        xs = np.asarray(sv_list)
        ys = np.asarray([grid[(0.0, s)] for s in sv_list if (0.0, s) in grid])
        return xs[: len(ys)], ys
    elif axis == "diagonal":
        # Pair (sa, sv) by *index* order — i.e. assume both axes have the
        # same length and interpret as a synthetic "symmetric noise" sweep.
        pairs = [(sa, sv) for sa, sv in zip(sa_list, sv_list)
                  if (sa, sv) in grid]
        xs = np.arange(len(pairs))
        ys = np.asarray([grid[p] for p in pairs])
        return xs, ys, pairs
    raise ValueError(axis)


def main() -> None:
    print("Reading inputs...")
    sa_aonly, ya_aonly = _read_curve(A_SWEEP_CSV, "sigma_per_rms", "val_acc")
    print(f"  A-only sweep: {len(sa_aonly)} σ_a levels, "
          f"acc {ya_aonly[0]:.4f} → {ya_aonly[-1]:.4f}")

    sv_v, yv_v = _read_curve(SIGMA_V_SWEEP_CSV,
                              "sigma_v_per_pixstd", "V_only_acc")
    print(f"  V-only sweep: {len(sv_v)} σ_v levels, "
          f"acc {yv_v[0]:.4f} → {yv_v[-1]:.4f}")

    sa_av, ya_av_along_a = _read_curve(SIGMA_V_SWEEP_CSV,
                                        "sigma_v_per_pixstd", "AV_acc")
    print(f"  AV σ_v sweep (from D1.2): {len(sv_v)} levels, "
          f"acc {ya_av_along_a[0]:.4f} → {ya_av_along_a[-1]:.4f}")

    av_grid, sa_g, sv_g = _read_grid(GRID_CSV, "AV_clean")
    print(f"  AV iso-perf grid: |σ_a|={len(sa_g)}, |σ_v|={len(sv_g)}, "
          f"|cells|={len(av_grid)}")

    av_raw_grid, _, _ = _read_grid(GRID_CSV, "AV_rawnoise")
    has_raw = len(av_raw_grid) > 0
    if has_raw:
        print(f"  AV-rawnoise iso-perf grid: {len(av_raw_grid)} cells")

    if os.path.exists(A_RAW_SWEEP_CSV):
        sa_araw, ya_araw = _read_curve(A_RAW_SWEEP_CSV,
                                        "sigma_per_rms", "val_acc")
    else:
        sa_araw, ya_araw = None, None

    # compute crossings
    rows = []
    print("\n  Iso-performance crossings:")
    print(f"  {'target':>7} | {'A_only σ_a':>12} | {'V_only σ_v':>12} | "
          f"{'AV-clean σ_a':>13} | {'AV-clean σ_v':>13} | "
          f"{'A_raw σ_a':>12}")
    for t in TARGETS:
        x_a = _interp_x_at_y(sa_aonly, ya_aonly, t)
        x_v = _interp_x_at_y(sv_v, yv_v, t)

        # AV σ_a at σ_v=0
        xs_av_a, ys_av_a = _grid_axis_curve(av_grid, sa_g, sv_g, "sigma_a")
        x_av_a = _interp_x_at_y(xs_av_a, ys_av_a, t)

        # AV σ_v at σ_a=0
        xs_av_v, ys_av_v = _grid_axis_curve(av_grid, sa_g, sv_g, "sigma_v")
        x_av_v = _interp_x_at_y(xs_av_v, ys_av_v, t)

        x_a_raw = float("nan")
        if sa_araw is not None:
            x_a_raw = _interp_x_at_y(sa_araw, ya_araw, t)

        rows.append(dict(
            target=t, A_only_sigma_a=x_a, V_only_sigma_v=x_v,
            AV_clean_sigma_a=x_av_a, AV_clean_sigma_v=x_av_v,
            A_rawnoise_sigma_a=x_a_raw,
        ))
        def _fmt(v):
            return "n/a" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        print(f"  {t*100:>5.0f}%  | {_fmt(x_a):>12} | {_fmt(x_v):>12} | "
              f"{_fmt(x_av_a):>13} | {_fmt(x_av_v):>13} | {_fmt(x_a_raw):>12}")

    # integration premium = AV's extra noise budget at iso-performance
    print("\n  Integration premium at iso-performance (AV's extra noise budget):")
    print(f"  {'target':>7} | {'σ_a premium':>14} | {'σ_v premium':>14}")
    prem_rows = []
    for r in rows:
        # Premium = (AV_x − UNI_x), in same unit.
        prem_sa = (r["AV_clean_sigma_a"] - r["A_only_sigma_a"])
        prem_sv = (r["AV_clean_sigma_v"] - r["V_only_sigma_v"])
        prem_rows.append((r["target"], prem_sa, prem_sv))
        print(f"  {r['target']*100:>5.0f}%  | {prem_sa:>14.4f} | {prem_sv:>14.4f}")

    out_csv = os.path.join(OUT_DIR, "D1_iso_perf_lookup.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow([
            "target_acc",
            "A_only_sigma_a", "V_only_sigma_v",
            "AV_clean_sigma_a_at_sv0", "AV_clean_sigma_v_at_sa0",
            "A_rawnoise_sigma_a",
            "sigma_a_premium_AV_vs_A", "sigma_v_premium_AV_vs_V",
        ])
        for r, (tt, psa, psv) in zip(rows, prem_rows):
            w.writerow([
                f"{r['target']:.4f}",
                f"{r['A_only_sigma_a']:.6f}",
                f"{r['V_only_sigma_v']:.6f}",
                f"{r['AV_clean_sigma_a']:.6f}",
                f"{r['AV_clean_sigma_v']:.6f}",
                f"{r['A_rawnoise_sigma_a']:.6f}",
                f"{psa:.6f}",
                f"{psv:.6f}",
            ])
    print(f"\n  wrote {out_csv}")


if __name__ == "__main__":
    main()
