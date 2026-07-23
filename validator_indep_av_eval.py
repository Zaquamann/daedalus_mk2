#!/usr/bin/env python3
"""VALIDATOR — independent V/AV clean-val forward-pass harness.

Purpose: independently reproduce the CLEAN-val accuracy anchors that the A-only
harness (validator_indep_eval.py) does not cover —
    V-trained  video_only_fair.pt  (video pipeline)        → 86.56%
    AV-trained av_fused.pt          (gated fusion, α≈5.20)  → 95.80%
    [also handles av_fused_rawnoise.pt → 95.96%; same arch]
on the pinned val partition (sha 03c5a87a, N=5244, 180 classes).

This is the V/AV analogue of the σ=0 self-check that anchored the noise work:
if an independent reimplementation of the eval reproduces the checkpoint's
best_val_acc, the "the project eval script manufactures the number" hypothesis
is refuted for these anchors.

INDEPENDENCE — this script does NOT import or call any project EVAL/TRAIN loop
(train_av.py, train_v_only_fair.py, analyze_av_msi.py, eval_av_*.py). It
reimplements here: val-partition selection + P5 re-hash, the clean-val dataset
view (cached-mel + strided uint8 video memmap → /255), batching, forward call,
argmax top-1, accuracy.

REUSED ON PURPOSE (reimplementing them would feed the net out-of-distribution
inputs / wrong weights, which is NOT the locus of a "manufacturing" bug):
  * the model ARCHITECTURE classes — AVWordResNet (model_av.py),
    VOnlyFairWordResNet (model_v_only_fair.py); the checkpoint weights bind to
    these exact modules.
  * the canonical INPUT REPRESENTATION — the precomputed log-mel cache and the
    uint8 lip-ROI video memmap in processed/dataset_av.pt, indexed exactly as
    PairedAVDataset does (mel[idx]; video[idx][::t_stride] / 255).

Measurement regime: the 86.56 / 95.80 targets were produced under bf16 autocast
(+ torch.compile) in the training scripts. We therefore run BOTH bf16-autocast
(the apples-to-apples regime; the self-check keys on this) AND plain fp32 (a
true-precision sanity number). torch.compile is intentionally NOT used (eager is
more independent; argmax accuracy is robust to the compile-vs-eager fp rounding).

Run on dev-codex (project at /scratch/daedelus), READ-ONLY:
    python validator_indep_av_eval.py \
        --root /scratch/daedelus \
        --ckpt /scratch/daedelus/models/av_fused.pt \
        --out  /scratch/daedelus/analysis/validator_indep_cleanval_av_fused.csv
    python validator_indep_av_eval.py \
        --root /scratch/daedelus \
        --ckpt /scratch/daedelus/models/video_only_fair.pt \
        --out  /scratch/daedelus/analysis/validator_indep_cleanval_video_only_fair.csv
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
# Self-check band: bf16-autocast-eager vs the (bf16-autocast + torch.compile)
# training measurement should agree to a few borderline samples. Flag wider gaps
# for a debugger handoff rather than papering over them.
SELFCHECK_TOL = 0.005   # 0.5% ≈ 26 / 5244 samples


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _detect_model(state_dict: dict) -> str:
    """'av' if the cross-modal gate is present, else 'v_fair'."""
    mods = {k.split(".")[0] for k in state_dict}
    if "gate" in mods and "audio_block1" in mods:
        return "av"
    if "visual" in mods and "block2" in mods and "audio_block1" not in mods:
        return "v_fair"
    raise SystemExit(f"[FATAL] cannot classify checkpoint from modules {sorted(mods)} "
                     f"(expected AV {{audio_block1,audio_block2,gate,visual,fc}} or "
                     f"V-fair {{visual,block2,fc}}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="project root (e.g. /scratch/daedelus)")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt to evaluate")
    ap.add_argument("--model", choices=["auto", "av", "v_fair"], default="auto")
    ap.add_argument("--t-stride", type=int, default=2,
                    help="video temporal stride (training used 2: T=100 cache → 50)")
    ap.add_argument("--video-cache", default=None,
                    help="override path to the videos_88_100.uint8 memmap")
    ap.add_argument("--expect-sha", default=EXPECT_SHA,
                    help="expected val_idx sha256 (P5 integrity gate); empty to skip")
    ap.add_argument("--precision", choices=["both", "bf16", "fp32"], default="both")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Import project modules from the (possibly relocated) project copy.
    sys.path.insert(0, args.root)

    proc = os.path.join(args.root, "processed")
    for need in ("splits.pt", "dataset_av.pt"):
        p = os.path.join(proc, need)
        if not os.path.exists(p):
            print(f"[FATAL] missing {p} — env not fully staged (need Phase-2 staged signal).")
            sys.exit(4)

    # --- independent val selection + P5 re-hash --------------------------------
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)}  sha256={val_sha}")
    if args.expect_sha and val_sha != args.expect_sha:
        print(f"[FATAL] val sha {val_sha} != expected {args.expect_sha}")
        print("        data-integrity failure (corrupt/truncated transfer?). STOP — not a result.")
        sys.exit(2)
    if args.expect_sha:
        print("[val] sha matches expected — partition integrity OK")

    # --- canonical input representation: cached mel + uint8 video memmap -------
    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels = dav["spectrograms"]                       # (N, 80, 99) float32 tensor
    labels = np.asarray(dav["labels"]).astype(np.int64)
    n_all = len(labels)
    T_FRAMES, H, W = dav["video_shape"]
    if hasattr(mels, "numpy"):
        mels_np = mels.numpy()
    else:
        mels_np = np.asarray(mels)

    # Resolve the video memmap path (absolute in the .pt, with fallbacks).
    cache_path = args.video_cache or dav.get("video_cache_path")
    cache_name = dav.get("video_cache_name", "videos_88_100.uint8")
    if not cache_path or not os.path.exists(cache_path):
        alt = os.path.join(args.root, "data", "visual", "cache", cache_name)
        if os.path.exists(alt):
            cache_path = alt
        else:
            print(f"[FATAL] video memmap not found.")
            print(f"        tried: {args.video_cache or dav.get('video_cache_path')}")
            print(f"        tried: {alt}")
            print("        the ~13G uint8 video cache must be staged on the pod. STOP.")
            sys.exit(3)
    expect_bytes = n_all * T_FRAMES * H * W
    actual_bytes = os.path.getsize(cache_path)
    print(f"[video] memmap {cache_path}")
    print(f"        shape=({n_all},{T_FRAMES},{H},{W}) uint8  size={actual_bytes/1024**3:.2f} GiB")
    if actual_bytes != expect_bytes:
        print(f"[FATAL] memmap size {actual_bytes} != expected {expect_bytes} "
              f"(N*T*H*W). Truncated/incomplete stage. STOP.")
        sys.exit(3)
    videos = np.memmap(cache_path, dtype=np.uint8, mode="r",
                       shape=(n_all, T_FRAMES, H, W))

    # --- model ----------------------------------------------------------------
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    n_classes = len(ckpt["label_to_idx"])
    clean_ref = float(ckpt.get("best_val_acc", float("nan")))
    kind = args.model if args.model != "auto" else _detect_model(sd)

    # Cross-check the checkpoint's own pinned partition against splits.pt.
    ck_sha = ckpt.get("val_idx_sha256")
    if ck_sha and ck_sha != val_sha:
        print(f"[FATAL] checkpoint val_idx_sha256 {ck_sha} != splits.pt val sha {val_sha}")
        print("        checkpoint and pinned partition disagree. STOP — not a result.")
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if kind == "av":
        from model_av import AVWordResNet
        model = AVWordResNet(n_classes).to(device)
        model.load_state_dict(sd)
        a_loaded = float(model.gate.alpha.detach().cpu())
        a_ref = ckpt.get("alpha_at_best")
        print(f"[ckpt] {args.ckpt}  [model=AVWordResNet]")
        print(f"       best_val_acc(train)={clean_ref:.4%}  n_classes={n_classes}")
        print(f"       gate.alpha(loaded)={a_loaded:.4f}  alpha_at_best(ckpt)={a_ref}")
        if a_ref is not None and abs(a_loaded - float(a_ref)) > 1e-3:
            print(f"[WARN] loaded α != stored α — weight load suspect.")
    else:
        from model_v_only_fair import VOnlyFairWordResNet
        model = VOnlyFairWordResNet(n_classes).to(device)
        model.load_state_dict(sd)
        print(f"[ckpt] {args.ckpt}  [model=VOnlyFairWordResNet]")
        print(f"       best_val_acc(train)={clean_ref:.4%}  n_classes={n_classes}")
    model.eval()

    stride = max(1, int(args.t_stride))

    class IndepAVView(Dataset):
        """Clean-val view: cached mel + strided uint8 video → /255. No augment.
        Reimplements PairedAVDataset's indexing independently."""

        def __len__(self) -> int:
            return len(val_idx)

        def __getitem__(self, k: int):
            gidx = int(val_idx[k])
            mel = torch.from_numpy(mels_np[gidx]).unsqueeze(0)        # (1, 80, 99)
            v = np.array(videos[gidx])                               # (T, H, W) uint8 copy
            if stride > 1:
                v = v[::stride]                                      # (T', H, W)
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0    # (1, T', H, W)
            return mel, vid, int(labels[gidx])

    @torch.no_grad()
    def run(precision: str) -> tuple[int, int, float]:
        use_ac = (precision == "bf16") and device.type == "cuda"
        ds = IndepAVView()
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        correct = total = 0
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_ac):
                logits = model(vid) if kind == "v_fair" else model(mel, vid)
            correct += (logits.argmax(1) == y).sum().item()
            total += int(y.size(0))
        return correct, total, correct / total

    precisions = ["bf16", "fp32"] if args.precision == "both" else [args.precision]
    results: dict[str, tuple[int, int, float]] = {}
    for prec in precisions:
        c, t, a = run(prec)
        results[prec] = (c, t, a)
        print(f"[clean-val {prec:>4}] acc={a:.4%}  ({c}/{t})")

    # Self-check keyed on the bf16 regime that produced the target (fall back to
    # whichever precision was run).
    key = "bf16" if "bf16" in results else precisions[0]
    acc_key = results[key][2]
    delta = acc_key - clean_ref
    passed = abs(delta) <= SELFCHECK_TOL
    print(f"[self-check] {key} acc={acc_key:.4%}  ref(best_val_acc)={clean_ref:.4%}  "
          f"delta={delta:+.4%}  tol=±{SELFCHECK_TOL:.2%} -> {'PASS' if passed else 'REVIEW'}")
    if not passed:
        print("[REVIEW] independent clean-val deviates beyond tolerance from the "
              "checkpoint's best_val_acc — debugger handoff, do NOT reconcile here.")
    print("[note] final GO/NO-GO is the validator's call, not this script's.")

    out = args.out or os.path.join(
        args.root, "analysis",
        f"validator_indep_cleanval_{os.path.splitext(os.path.basename(args.ckpt))[0]}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "ckpt", "precision", "n", "correct", "val_acc",
                    "best_val_acc_ref", "delta", "val_sha"])
        for prec in precisions:
            c, t, a = results[prec]
            w.writerow([kind, os.path.basename(args.ckpt), prec, t, c,
                        f"{a:.6f}", f"{clean_ref:.6f}", f"{a - clean_ref:+.6f}", val_sha])
    print(f"[out] wrote {out}")


if __name__ == "__main__":
    main()
