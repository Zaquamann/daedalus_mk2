#!/usr/bin/env python3
"""Re-run D5.7 (GradCAM) + D5.8 (block2 lesion) after the original Phase D
errored on D5.7. D5.4–D5.6 artifacts already on disk."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import BATCH_SIZE, T_STRIDE, _ValAVView, _accuracy, _load_models
from dataset_raw_noisy import RawNoisyAVDataset
from phase_d_saliency import (
    _cache_a_v_mid, _forward_AV_from_cache,
    D5_7_gradcam_visual, D5_8_block2_lesion,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)
    val_idx = splits["val_idx"]
    if hasattr(val_idx, "numpy"):
        val_idx = val_idx.numpy()
    models = _load_models(device)
    AV = models["AV"][0]
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

    print("Building activation cache for D5.8...")
    t0 = time.time()
    a_cache, v_cache, labels_cache = _cache_a_v_mid(AV, loader, device)
    print(f"  cached in {time.time()-t0:.1f}s")
    out = _forward_AV_from_cache(AV, a_cache, v_cache, labels_cache, device)
    baseline = _accuracy(out["preds"], out["labels"])
    print(f"  baseline = {baseline:.4%}")

    D5_7_gradcam_visual(AV, loader, device)
    D5_8_block2_lesion(AV, a_cache, v_cache, labels_cache, device, baseline)

    print("\nResume done.")


if __name__ == "__main__":
    main()
