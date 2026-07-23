#!/usr/bin/env python3
"""VALIDATOR — independent gate-lesion re-derivation for av_fused.pt.

Re-derives the high-leverage Q15/Q8/Q6 lesion numbers WITHOUT importing
phase_c_lesions.py:
  * Wv per-channel single lesions -> ch22 (+42.16pp), ch12 (+37.70pp)
  * Wv top-21 tertile lesion (+94.53pp) and all-Wv-zero (+94.83pp)
  * Wa per-channel single lesions -> ch10 (+6.10pp)  [audio-side asymmetry]

Lesion definition (from the gate math in model_av.py:85-87, reimplemented here):
  g       = sigmoid(Wa(a_mid) + Wv(v_mid))
  a_fused = a_mid * (1 + alpha * g)
A "lesion" of Wv channel c zeroes the c-th output channel of Wv(v_mid) (which,
since Wv is a 1x1 conv with bias=False, is identical to zeroing Wv.weight[c]).
impact_pp = (baseline_acc - lesioned_acc) * 100.

Independence: reuses ONLY the trained submodules of AVWordResNet (audio_block1,
visual, gate.Wa, gate.Wv, gate.alpha, audio_block2, gap, fc) and the cached
input representation (dataset_av.pt mel + uint8 video memmap). The caching +
masked-forward loop is reimplemented here; phase_c_lesions.py is NOT imported.

Correctness anchor (the role the sigma=0 self-check played): the lesion-free
cached forward must reproduce the AV fp32 clean-val accuracy 0.956712 (already
independently confirmed by validator_indep_av_eval.py). If it doesn't, the
cache/forward path is wrong and the lesion deltas are meaningless.

fp32, no autocast — matches the regime phase_c_lesions.py computed the CSVs in.

Run on dev-codex:
    python validator_indep_lesion.py --root /scratch/daedelus \
        --ckpt /scratch/daedelus/models/av_fused.pt \
        --out  /scratch/daedelus/analysis/validator_indep_lesion_av_fused.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
BASELINE_FP32_REF = 0.956712   # AV clean-val fp32, independently confirmed earlier


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--t-stride", type=int, default=2)
    ap.add_argument("--video-cache", default=None)
    ap.add_argument("--expect-sha", default=EXPECT_SHA)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cache-batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.root)
    from model_av import AVWordResNet  # noqa: E402

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}")
    if args.expect_sha and val_sha != args.expect_sha:
        print(f"[FATAL] val sha {val_sha} != expected {args.expect_sha}; STOP.")
        sys.exit(2)

    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels = dav["spectrograms"]
    mels_np = mels.numpy() if hasattr(mels, "numpy") else np.asarray(mels)
    labels = np.asarray(dav["labels"]).astype(np.int64)
    n_all = len(labels)
    T_FRAMES, H, W = dav["video_shape"]
    cache_path = args.video_cache or dav.get("video_cache_path")
    cache_name = dav.get("video_cache_name", "videos_88_100.uint8")
    if not cache_path or not os.path.exists(cache_path):
        alt = os.path.join(args.root, "data", "visual", "cache", cache_name)
        cache_path = alt if os.path.exists(alt) else cache_path
    if not cache_path or not os.path.exists(cache_path):
        print(f"[FATAL] video memmap not found ({cache_path}); STOP.")
        sys.exit(3)
    videos = np.memmap(cache_path, dtype=np.uint8, mode="r",
                       shape=(n_all, T_FRAMES, H, W))

    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    if ckpt.get("val_idx_sha256") and ckpt["val_idx_sha256"] != val_sha:
        print("[FATAL] ckpt val sha != splits sha; STOP."); sys.exit(2)
    n_classes = len(ckpt["label_to_idx"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AVWordResNet(n_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    alpha = model.gate.alpha.detach()
    print(f"[ckpt] {args.ckpt}  alpha={float(alpha):.4f}  device={device}")
    stride = max(1, int(args.t_stride))

    class V(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k])
            mel = torch.from_numpy(mels_np[g]).unsqueeze(0)
            v = np.array(videos[g])
            if stride > 1: v = v[::stride]
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, vid, int(labels[g])

    dl = DataLoader(V(), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    # ---- cache a_mid, v_mid (one pass, fp32) ----
    @torch.no_grad()
    def build_cache():
        A, Vv, Y = [], [], []
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            A.append(model.audio_block1(mel).cpu())
            Vv.append(model.visual(vid).cpu())
            Y.append(y)
        return torch.cat(A), torch.cat(Vv), torch.cat(Y)

    print("[cache] forwarding audio_block1 + visual over val ...")
    a_cache, v_cache, y_cache = build_cache()
    print(f"[cache] a_mid={tuple(a_cache.shape)} v_mid={tuple(v_cache.shape)}")

    @torch.no_grad()
    def fwd(wv_mask=None, wa_mask=None):
        cb = args.cache_batch
        correct = total = 0
        for i in range(0, a_cache.shape[0], cb):
            a = a_cache[i:i+cb].to(device, non_blocking=True)
            v = v_cache[i:i+cb].to(device, non_blocking=True)
            y = y_cache[i:i+cb].to(device, non_blocking=True)
            Wa = model.gate.Wa(a)
            Wv = model.gate.Wv(v)
            if wv_mask is not None: Wv = Wv * wv_mask.view(1, -1, 1, 1)
            if wa_mask is not None: Wa = Wa * wa_mask.view(1, -1, 1, 1)
            g = torch.sigmoid(Wa + Wv)
            af = a * (1.0 + alpha * g)
            x = model.audio_block2(af)
            pen = model.gap(x).flatten(1)
            logits = model.fc(model.dropout(pen))
            correct += (logits.argmax(1) == y).sum().item()
            total += int(y.numel())
        return correct / total

    baseline = fwd()
    print(f"[baseline] cached lesion-free acc = {baseline:.6f}  "
          f"(fp32 ref {BASELINE_FP32_REF})  delta={baseline-BASELINE_FP32_REF:+.6f}")
    if abs(baseline - BASELINE_FP32_REF) > 0.002:
        print("[WARN] baseline deviates from the independently-confirmed AV fp32 "
              "clean-val — cache/forward path suspect; lesion deltas NOT trustworthy.")

    nch = model.gate.Wv.weight.shape[0]   # 64
    ones = torch.ones(nch, device=device)

    # ---- Wv single-channel lesions ----
    wv_imp = np.zeros(nch)
    for c in range(nch):
        m = ones.clone(); m[c] = 0.0
        acc = fwd(wv_mask=m)
        wv_imp[c] = (baseline - acc) * 100.0
    order = np.argsort(-wv_imp)
    print("[Wv] top-8 by impact (pp):")
    for c in order[:8]:
        print(f"     ch{int(c):>2d}: {wv_imp[c]:+.4f}")

    # ---- Wv tertile (top-21) + all-zero ----
    k = nch // 3   # 21
    top_idx = order[:k]
    m = ones.clone(); m[top_idx] = 0.0
    acc_tert = fwd(wv_mask=m)
    tert_pp = (baseline - acc_tert) * 100.0
    m0 = torch.zeros(nch, device=device)
    acc_allzero = fwd(wv_mask=m0)
    allzero_pp = (baseline - acc_allzero) * 100.0
    print(f"[Wv] top-{k} tertile: acc={acc_tert:.6f} delta={tert_pp:+.4f}pp "
          f"channels={sorted(int(x) for x in top_idx)}")
    print(f"[Wv] all-zero: acc={acc_allzero:.6f} delta={allzero_pp:+.4f}pp")

    # ---- Wa single-channel lesions (audio-side asymmetry) ----
    wa_imp = np.zeros(nch)
    for c in range(nch):
        m = ones.clone(); m[c] = 0.0
        acc = fwd(wa_mask=m)
        wa_imp[c] = (baseline - acc) * 100.0
    wa_order = np.argsort(-wa_imp)
    print("[Wa] top-5 by impact (pp):")
    for c in wa_order[:5]:
        print(f"     ch{int(c):>2d}: {wa_imp[c]:+.4f}")

    out = args.out or os.path.join(args.root, "analysis",
                                   "validator_indep_lesion_av_fused.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["kind", "channel_or_set", "baseline_acc", "lesioned_acc", "delta_pp"])
        for c in range(nch):
            w.writerow(["Wv_single", c, f"{baseline:.6f}",
                        f"{baseline - wv_imp[c]/100.0:.6f}", f"{wv_imp[c]:+.4f}"])
        w.writerow(["Wv_top21_tertile", ";".join(map(str, sorted(int(x) for x in top_idx))),
                    f"{baseline:.6f}", f"{acc_tert:.6f}", f"{tert_pp:+.4f}"])
        w.writerow(["Wv_all_zero", "all", f"{baseline:.6f}",
                    f"{acc_allzero:.6f}", f"{allzero_pp:+.4f}"])
        for c in range(nch):
            w.writerow(["Wa_single", c, f"{baseline:.6f}",
                        f"{baseline - wa_imp[c]/100.0:.6f}", f"{wa_imp[c]:+.4f}"])
    print(f"[out] wrote {out}")


if __name__ == "__main__":
    main()
