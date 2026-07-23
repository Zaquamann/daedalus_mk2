#!/usr/bin/env python3
"""TEMP DEBUG (debugger task #15) — complete the CANONICAL E1d full-grid d' on the
FINAL ep165 model. The combined both-designs canonical run exceeded a 30-min
timeout mid-E1d (E1c completed + wrote its official CSV; E1d was cut at sigma0.04
before run_design writes its CSV). This reruns ONLY run_design('e1d') with the
UNMODIFIED harness code (NW=32, full 16-sigma SIGMA_E1D, no overrides) to write the
official D310_e1d_latefusion.csv. No contention (trainer done, GPUs free).

Run: CUDA_VISIBLE_DEVICES=0 python analysis/deepdive/diag_dprime_e1d_only.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import dprime_latefusion as dlf  # noqa: E402

if __name__ == "__main__":
    device = torch.device("cuda")
    base = dlf.RawNoisyAVDataset(noise=False, t_stride=dlf.T_STRIDE,
                                 return_video=True)
    val_idx = torch.load(os.path.join(dlf.SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    models = dlf._load_models(device)
    ck = torch.load(dlf.LATE_CKPT, weights_only=False)
    AVl = dlf.AVLateFusionReliabilityWordResNet(
        len(ck["label_to_idx"]), use_mid_gate=ck.get("use_mid_gate", False))
    AVl.load_state_dict(ck["model_state_dict"])
    AVl = AVl.to(device).eval()
    print(f"AV_late ckpt={os.path.basename(dlf.LATE_CKPT)} "
          f"acc={ck.get('best_val_acc')} NW={dlf.NW} "
          f"E1D_grid={dlf.SIGMA_E1D}", flush=True)
    dlf.run_design("e1d", models, AVl, base, val_idx, device)
