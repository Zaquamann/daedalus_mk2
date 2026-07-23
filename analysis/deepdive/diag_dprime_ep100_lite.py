#!/usr/bin/env python3
"""TEMP DEBUG INSTRUMENT (debugger, task #15) — drive the EXACT dprime_latefusion
harness with FEWER dataloader workers and a REDUCED sigma grid, so the ep100
mid-read d' can be measured WITHOUT thrashing the CPU against the concurrent GPU0
trainer. The full harness hard-codes NW=32; with the trainer's 16 workers also
running, 48 workers starve the CPU and a 20-min run produced ZERO rows.

Methodology is the harness VERBATIM — d' formula, pair selection (uses only the
clean sigma=0 loader, so pair_ids / dV_pair are IDENTICAL to the canonical run,
keeping every d' directly comparable to the v2 CSVs), model loading, ablation
readouts. ONLY overridden: NW (32->6) and the two sigma grids (kept the
diagnostic points: clean + degradation for R1/E1c; the dA~dV crossover region for
R3/E1d). For the canonical full-grid numbers I re-run the unmodified harness on
the FINAL ep185 model when the trainer is done and the CPU is free.

Run: CUDA_VISIBLE_DEVICES=1 LATE_CKPT=models/av_fused_latefusion_ep100.pt \
     python analysis/deepdive/diag_dprime_ep100_lite.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dprime_latefusion as dlf  # noqa: E402

# --- single-variable overrides: workers + grid only, all d' logic unchanged ---
dlf.NW = 6
dlf.SIGMA_E1C = [0.0, 0.005, 0.02, 0.05, 0.08]
dlf.SIGMA_E1D = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.13, 0.22]
dlf.LATE_CKPT = os.environ.get("LATE_CKPT", dlf.LATE_CKPT)

if __name__ == "__main__":
    print(f"[lite] NW={dlf.NW}  E1C={dlf.SIGMA_E1C}  E1D={dlf.SIGMA_E1D}",
          flush=True)
    print(f"[lite] LATE_CKPT={dlf.LATE_CKPT}", flush=True)
    dlf.main()
