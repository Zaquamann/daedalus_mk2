#!/usr/bin/env python3
"""Orchestrator for the full Tier-0 AV-integration deep-dive.

Runs phases in order, gates on per-phase sanity checks, prints a
phase-complete summary. Each phase is implemented in its own module so we
can also run them individually.

Usage:
    python run_deepdive_tier0.py            # all phases
    python run_deepdive_tier0.py --phase A  # just Phase A
    python run_deepdive_tier0.py --skip B   # skip Phase B (already done)
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time


PHASES = {
    "B-pre":  ("analyze_av_deepdive",  "main",
                "D1.1, D1.6, D1.7 (matrix + heterogeneous noise)"),
    "B-mid":  ("eval_av_visual_noise", "main",
                "D1.2, D1.3, D1.4 (σ_v + frame-drop + iso-perf grid)"),
    "B-post": ("iso_perf_rendezvous",  "main",
                "D1.5 (iso-perf rendezvous lookup)"),
    "B-fig":  ("build_d1_figures",     "main",
                "D1 figures"),
    "A":      ("phase_a_deepdive",     "main",
                "D4.1, D4.2, D4.3, D4.5 + activation cache"),
    "C":      ("phase_c_lesions",      "main",
                "D3.5–D3.9 lesions + α-sweep"),
    "D":      ("phase_d_saliency",     "main",
                "D5.4, D5.5, D5.6, D5.7, D5.8 saliency + lesions"),
    "E":      ("phase_e_geometry",     "main",
                "D2.1–D2.7 UMAPs + drivers + dendrograms"),
    "F":      ("phase_f_flow",         "main",
                "D5.1, D5.2, D5.3, D5.9, D5.11, D5.12 layer flow"),
}

# Execution order: Phase B parts run first (already started); A is the
# activation-cache phase that gates Phase D/E/F.
DEFAULT_ORDER = ["B-pre", "B-mid", "B-post", "B-fig",
                  "A", "C", "D", "E", "F"]


def _run_phase(key: str) -> bool:
    module, fn, desc = PHASES[key]
    print(f"\n{'=' * 72}\nPhase {key} — {desc}\n{'=' * 72}")
    t0 = time.time()
    try:
        mod = importlib.import_module(module)
        getattr(mod, fn)()
    except SystemExit as e:
        if e.code:
            print(f"[WARN] Phase {key} exited with code {e.code}")
            return False
    except Exception as e:
        print(f"[WARN] Phase {key} raised {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    dt = time.time() - t0
    print(f"\nPhase {key} done ({dt:.0f}s).")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", action="append", default=None,
                     help="run only these phases (repeatable)")
    ap.add_argument("--skip", action="append", default=None,
                     help="skip these phases")
    args = ap.parse_args()

    order = list(DEFAULT_ORDER)
    if args.phase:
        order = [p for p in order if p in args.phase]
    if args.skip:
        order = [p for p in order if p not in args.skip]

    print("Phases to run (in order):")
    for k in order:
        _, _, desc = PHASES[k]
        print(f"  {k:>6}: {desc}")

    overall_ok = True
    for k in order:
        ok = _run_phase(k)
        if not ok:
            overall_ok = False
            print(f"\n[FAIL] Phase {k} failed. Stopping orchestrator.")
            break

    print("\n" + ("[OK] All requested phases completed." if overall_ok
                    else "[WARN] Stopped early."))


if __name__ == "__main__":
    main()
