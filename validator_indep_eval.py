#!/usr/bin/env python3
"""VALIDATOR — independent re-implementation of the raw-noise eval at a single
sigma. Purpose: refute the hypothesis "eval_rawnoise_sweep.py itself manufactures
the 70s number." This script does NOT import or call _FixedSigmaView /
eval_rawnoise_sweep.py.

Independent here:
  * val-partition selection — loaded from processed/splits.pt and RE-HASHED
    (the locked P5 integrity gate); a mismatch vs the expected sha is a
    data-integrity failure, not a result.
  * raw-waveform Gaussian injection — reimplemented from the documented recipe:
    sigma = sigma_mult * audio_rms, rng = default_rng(NOISE_SEED + GLOBAL_idx),
    noise added to the raw waveform PRE-pad / PRE-STFT.
  * batched forward + top-1 — computed here.

Reused ON PURPOSE (the canonical INPUT REPRESENTATION the model was trained on;
reimplementing them would feed the net out-of-distribution inputs and is NOT the
locus of a "manufacturing" bug):
  * _read_wav, _pad_audio, _wav_to_log_mel  (paired_dataset.py)
  * WordResNet                              (train.py)

Built-in correctness gate: a sigma=0 self-check must reproduce the checkpoint's
clean val accuracy. If it doesn't, the representation/forward/top-1 path is
wrong and the sigma>0 number is meaningless — so this runs FIRST and cheaply
catches a broken harness before the full seed sweep.

Run on dev-codex (project at /scratch/daedelus):
    python validator_indep_eval.py \
        --root /scratch/daedelus \
        --ckpt /scratch/daedelus/models/audio_only_rawnoise_filtered.pt \
        --sigma 0.2 --seeds 0,1,2 \
        --out /scratch/daedelus/analysis/validator_indep_sigma02_rawnoise.csv
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


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="project root (e.g. /scratch/daedelus)")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt to evaluate")
    ap.add_argument("--sigma", type=float, default=0.2, help="sigma_a / audio_rms")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated NOISE_SEED values")
    ap.add_argument("--expect-sha", default=EXPECT_SHA,
                    help="expected val_idx sha256 (P5 integrity gate); empty to skip")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Import project modules from the (possibly relocated) project copy.
    sys.path.insert(0, args.root)
    from paired_dataset import _pad_audio, _read_wav, _wav_to_log_mel  # noqa: E402
    from train import WordResNet  # noqa: E402

    proc = os.path.join(args.root, "processed")
    for need in ("splits.pt", "dataset_av.pt", "pad_offsets.pt"):
        p = os.path.join(proc, need)
        if not os.path.exists(p):
            print(f"[FATAL] missing {p} — env not fully migrated (need #2 env-ready).")
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
        print(f"[val] sha matches expected — partition integrity OK")

    # --- canonical audio paths / labels / pad offsets --------------------------
    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    audio_paths = dav["audio_paths"]
    labels = np.asarray(dav["labels"]).astype(np.int64)
    po = torch.load(os.path.join(proc, "pad_offsets.pt"), weights_only=False)
    pad_offsets = po["pad_left_frames"].numpy().astype(np.int64)

    # audio reachable at the stored absolute paths? (migration gotcha)
    probe = [int(i) for i in val_idx[:8]]
    missing = [audio_paths[i] for i in probe if not os.path.exists(audio_paths[i])]
    if missing:
        print(f"[FATAL] raw .wav not found at stored path, e.g.\n        {missing[0]}")
        print("        dataset_av.pt holds ABSOLUTE audio paths (~/Downloads/audio/...).")
        print("        On the pod the raw wavs must be reachable there (symlink or copy).")
        sys.exit(3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    class IndepView(Dataset):
        """My own injection (NOT _FixedSigmaView). Seed uses the GLOBAL index."""

        def __init__(self, sigma_mult: float, seed: int):
            self.sigma_mult = float(sigma_mult)
            self.seed = int(seed)

        def __len__(self) -> int:
            return len(val_idx)

        def __getitem__(self, k: int):
            gidx = int(val_idx[k])
            audio = _read_wav(audio_paths[gidx])          # float32 in [-1, 1]
            if self.sigma_mult > 0:
                rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
                sigma = self.sigma_mult * rms
                rng = np.random.default_rng(self.seed + gidx)
                noise = rng.standard_normal(len(audio)).astype(np.float32) * sigma
                audio = audio + noise
            audio_p = _pad_audio(audio, int(pad_offsets[gidx]))
            mel = _wav_to_log_mel(audio_p).astype(np.float32)
            return torch.from_numpy(mel).unsqueeze(0), int(labels[gidx])

    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    n_classes = len(ckpt["label_to_idx"])
    clean_ref = float(ckpt.get("best_val_acc", float("nan")))
    model = WordResNet(n_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[ckpt] {args.ckpt}")
    print(f"       best_val_acc(train)={clean_ref:.4f}  n_classes={n_classes}  "
          f"noise_kind={ckpt.get('noise_kind', '-')}  noise_range={ckpt.get('noise_range', '-')}")

    @torch.no_grad()
    def run(sigma_mult: float, seed: int) -> float:
        ds = IndepView(sigma_mult, seed)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        correct = total = 0
        for X, y in dl:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            correct += (model(X).argmax(1) == y).sum().item()
            total += int(y.size(0))
        return correct / total

    rows: list[tuple[str, int, float]] = []

    # --- correctness gate: sigma=0 must reproduce clean val acc ----------------
    acc0 = run(0.0, 0)
    delta = acc0 - clean_ref
    print(f"[self-check] sigma=0 acc={acc0:.4%}  ref(best_val_acc)={clean_ref:.4%}  "
          f"delta={delta:+.4%}")
    if abs(delta) > 0.01:
        print("[WARN] sigma=0 self-check deviates >1% from clean val — "
              "harness/representation suspect; sigma>0 numbers NOT trustworthy.")
    else:
        print("[self-check] OK — representation/forward/top-1 path verified.")
    rows.append(("selfcheck_sigma0", 0, acc0))

    # --- the claim: sigma over noise seeds -------------------------------------
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    accs = []
    for sd in seeds:
        a = run(args.sigma, sd)
        accs.append(a)
        print(f"[sigma={args.sigma}] noise_seed={sd}  acc={a:.4%}")
        rows.append((f"sigma{args.sigma:g}", sd, a))
    accs = np.asarray(accs)
    mean, sdv = float(accs.mean()), float(accs.std(ddof=0))
    print(f"[sigma={args.sigma}] mean={mean:.4%}  sd={sdv:.4%}  over {len(seeds)} noise seeds")
    p1 = 0.70 <= mean < 0.80
    print(f"[P1 gate] 3-seed MEAN in [70,80)% -> {'PASS' if p1 else 'FAIL'}")
    print("[note] final GO/NO-GO is the validator's call, not this script's; "
          "P1 also requires the baseline-contrast (P3) check, run separately.")

    out = args.out or os.path.join(
        args.root, "analysis",
        f"validator_indep_sigma{args.sigma:g}_"
        f"{os.path.splitext(os.path.basename(args.ckpt))[0]}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["condition", "noise_seed", "val_acc"])
        for c, sd, a in rows:
            w.writerow([c, sd, f"{a:.6f}"])
    print(f"[out] wrote {out}")


if __name__ == "__main__":
    main()
