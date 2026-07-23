#!/usr/bin/env python3
"""Q14 — recurrent-vs-feedforward comparison battery (4 axes, 3 models), ONE artifact:
analysis/deepdive/Q14_recurrent_vs_feedforward.csv.

Three models, ALL on the pinned clean val (sha 03c5a87a, N=5244), eager fp32, .eval():
  recurrent     models/av_fused_recurrent.pt          (GRU over v_mid; seed-0, eager)
  ff_matched    models/av_fused_ff_baseline_seed0.pt  (recurrent minus the GRU; seed-0, eager)
  ff_canonical  models/av_fused.pt                    (seed-42, torch.compile — prior report)

The three head-to-head deltas:
  recurrent − ff_matched     = the PURE recurrence effect (single-variable: only the GRU
                               differs; same seed-0/eager recipe + split). Causal attribution.
  ff_matched − ff_canonical  = run-to-run variance (same AVWordResNet architecture, a
                               different seed/training-path draw). Calibrates how big a
                               clean-val delta must be to MEAN anything.
  recurrent − ff_canonical   = the uncontrolled comparison, for continuity with the rest
                               of the report (CONFOUNDED by seed+compile — read with the
                               ff_matched−ff_canonical variance band in mind).

Four axes:
  (1) clean_val            — top-line AV accuracy.
  (2) rawnoise_sweep       — additive-Gaussian σ_a/rms sweep, clean video.
  (3) E11_temporal         — Δt video-shift temporal-congruence sweep (peak at Δt=0).
  (4) D5_temporal_saliency — zero a sliding 10-frame (200 ms) video window; the accuracy
                             drop profile = WHERE in time the visual stream is load-bearing.
The temporal axes (E11/D5) are the scientific payload of Q14: recurrence is an
architecture change, so the real question is whether it shifts WHEN the model integrates,
robust to run-to-run variance even if top-line acc lands similar.

CRITICAL — drive every model through its NATIVE forward(audio, video). The recurrent model
computes v_mid_ctx = GRU(v_mid) INSIDE forward(); the existing manual-unroll eval helpers
(analyze_av_msi._forward_AV, phase_d_saliency._forward_with_temporal_mask) call
model.gate(a_mid, v_mid) directly and would BYPASS the GRU. This harness reuses only the
model-AGNOSTIC views (_ValAVView, _NoisyAudioView, _TemporalShiftView) verbatim and swaps
the forward. For AVWordResNet, forward() IS the manual unroll, so ff_canonical reproduces
the published anchors bit-close.

USAGE
  # 1) HARNESS SELF-TEST (run NOW, before any Q14 number is trusted): load ONLY the
  #    canonical av_fused.pt and reproduce EVERY published anchor (clean + rawnoise + E11
  #    + D5). PASS iff every anchored point is within --tol of its anchor.
  python q14_recurrent_vs_feedforward.py --self-test
  # 2) FULL COMPARISON (run when both new ckpts exist): point it at the three ckpts.
  python q14_recurrent_vs_feedforward.py \
      --recurrent-ckpt   models/av_fused_recurrent.pt \
      --ff-matched-ckpt  models/av_fused_ff_baseline_seed0.pt \
      --ff-canonical-ckpt models/av_fused.pt
The checkpoint paths are ARGS so the full run is just a swap-in once both .done fire.
"""
import argparse
import csv
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (
    BATCH_SIZE,
    T_STRIDE,
    _NoisyAudioView,
    _TemporalShiftView,
    _ValAVView,
    _accuracy,
)
from dataset_raw_noisy import RawNoisyAVDataset
from model_av import AVWordResNet
from model_av_recurrent import AVRecurrentWordResNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPDIVE = os.path.join(SCRIPT_DIR, "analysis", "deepdive")
OUT = os.path.join(DEEPDIVE, "Q14_recurrent_vs_feedforward.csv")
SELFTEST_OUT = os.path.join(DEEPDIVE, "Q14_selftest_canonical.csv")
os.makedirs(DEEPDIVE, exist_ok=True)
PIN = "03c5a87a"

REC_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused_recurrent.pt")
FFM_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused_ff_baseline_seed0.pt")
FFC_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused.pt")

SIGMA_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
E11_DELTA_MS = (-200, -100, -60, -40, -20, 0, 20, 40, 60, 100, 200)
D5_WINDOW = 10
D5_NFRAMES = 50
D5_STEP = 5

# Published ff_canonical (av_fused.pt) anchors, eager fp32, for harness self-validation.
#   clean/rawnoise: models/av_fused_av_noise_sweep.csv (eval_av_rawnoise_sweep.py)
#   E11:            analysis/msi/E11_temporal_window.csv (analyze_av_msi.E11_temporal)
#   D5:             analysis/deepdive/D5_temporal_saliency.csv (phase_d_saliency.D5_5)
FF_ANCHOR = {
    ("clean_val", "clean"): 0.956712,
    ("rawnoise_sweep", "0.0000"): 0.956712,
    ("rawnoise_sweep", "0.0010"): 0.956712,
    ("rawnoise_sweep", "0.0050"): 0.948894,
    ("rawnoise_sweep", "0.0100"): 0.927536,
    ("rawnoise_sweep", "0.0200"): 0.851831,
    ("rawnoise_sweep", "0.0500"): 0.609268,
    ("rawnoise_sweep", "0.1000"): 0.376049,
    ("rawnoise_sweep", "0.2000"): 0.217201,
    ("rawnoise_sweep", "0.5000"): 0.120519,
    ("E11_temporal", "delta_ms=-200"): 0.152555,
    ("E11_temporal", "delta_ms=-100"): 0.475400,
    ("E11_temporal", "delta_ms=-60"): 0.741228,
    ("E11_temporal", "delta_ms=-40"): 0.849352,
    ("E11_temporal", "delta_ms=-20"): 0.939550,
    ("E11_temporal", "delta_ms=+0"): 0.956712,
    ("E11_temporal", "delta_ms=+20"): 0.946796,
    ("E11_temporal", "delta_ms=+40"): 0.907132,
    ("E11_temporal", "delta_ms=+60"): 0.838101,
    ("E11_temporal", "delta_ms=+100"): 0.597063,
    ("E11_temporal", "delta_ms=+200"): 0.193745,
    ("D5_temporal_saliency", "win=[0,10)"): 0.765446,
    ("D5_temporal_saliency", "win=[5,15)"): 0.539664,
    ("D5_temporal_saliency", "win=[10,20)"): 0.346491,
    ("D5_temporal_saliency", "win=[15,25)"): 0.232265,
    ("D5_temporal_saliency", "win=[20,30)"): 0.213768,
    ("D5_temporal_saliency", "win=[25,35)"): 0.281655,
    ("D5_temporal_saliency", "win=[30,40)"): 0.451564,
    ("D5_temporal_saliency", "win=[35,45)"): 0.663425,
    ("D5_temporal_saliency", "win=[40,50)"): 0.859649,
}
HARD_TOL = 1e-3      # critical anchors (clean, σ=0.05) in full mode must reproduce within this
SOFT_TOL = 5e-3      # other ff_canonical points flagged (full mode) if |Δ| exceeds this
SELFTEST_TOL = 1.5e-3  # self-test PASS: EVERY anchored ff_canonical point within this (~8 samples)


def _val_sha(val_idx) -> str:
    arr = val_idx.numpy() if hasattr(val_idx, "numpy") else np.asarray(val_idx)
    return hashlib.sha256(bytes(arr.astype("int64").tobytes())).hexdigest()


def _load_model(ckpt_path, cls, device):
    """Load a checkpoint into `cls`, inferring num_classes / gru_hidden from the
    state_dict so we never mis-size a layer. Returns (model.eval(), ckpt)."""
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = ck["model_state_dict"]
    n_classes = int(sd["fc.weight"].shape[0])
    if cls is AVRecurrentWordResNet:
        gru_hidden = int(sd["vproj.weight"].shape[1])   # vproj: (64, gru_hidden)
        model = cls(n_classes, gru_hidden=gru_hidden)
    else:
        model = cls(n_classes)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, ck


@torch.no_grad()
def _native_av_acc(model, loader, device, t_zero=None):
    """AV accuracy via NATIVE forward(audio, video). t_zero=(t0,t1) blanks a video frame
    window (D5). Works for BOTH classes — the recurrent GRU runs inside forward()."""
    preds, labels = [], []
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)     # (B,1,80,99)
        vid = vid.to(device, non_blocking=True)
        if t_zero is not None:
            vid = vid.clone()
            vid[:, :, t_zero[0]:t_zero[1], :, :] = 0.0
        logits = model(mel, vid)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return _accuracy(np.concatenate(preds), np.concatenate(labels))


def _loader(view):
    return DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)


def _axis_specs(base, val_np):
    """The 4 axes as a flat list of (axis, condition, view, t_zero) — identical
    iteration in both self-test and full-comparison modes."""
    specs = [("clean_val", "clean", _ValAVView(base, val_np), None)]
    for sigma in SIGMA_LEVELS:
        specs.append(("rawnoise_sweep", f"{sigma:.4f}",
                      _NoisyAudioView(base, val_np, sigma_mult=sigma, seed=0), None))
    for dms in E11_DELTA_MS:
        specs.append(("E11_temporal", f"delta_ms={dms:+d}",
                      _TemporalShiftView(base, val_np, delta_frames=round(dms / 20.0)), None))
    for t in range(0, D5_NFRAMES - D5_WINDOW + 1, D5_STEP):
        specs.append(("D5_temporal_saliency", f"win=[{t},{t + D5_WINDOW})",
                      _ValAVView(base, val_np), (t, t + D5_WINDOW)))
    return specs


def _evaluate(models, specs, device, verbose=True):
    """models: list of (name, model). Returns {(axis, condition): {name: acc}}."""
    results = {}
    for axis, cond, view, t_zero in specs:
        ld = _loader(view)
        results[(axis, cond)] = {name: _native_av_acc(m, ld, device, t_zero=t_zero)
                                 for name, m in models}
        if verbose:
            r = results[(axis, cond)]
            anc = FF_ANCHOR.get((axis, cond))
            ancs = ("" if anc is None or "ff_canonical" not in r
                    else f" | ffc_anchorΔ={(r['ff_canonical'] - anc) * 100:+.3f}pp")
            cells = " ".join(f"{n}={a:.6f}" for n, a in r.items())
            print(f"  [{axis}] {cond:>14s}: {cells}{ancs}")
    return results


def _prep_val(device):
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    splits = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                        weights_only=False)
    val_idx = splits["val_idx"]
    sha = _val_sha(val_idx)
    assert sha.startswith(PIN), f"VAL PIN MISMATCH: {sha[:16]}"
    val_np = val_idx.numpy() if hasattr(val_idx, "numpy") else np.asarray(val_idx)
    assert len(val_np) == 5244, len(val_np)
    print(f"  val sha[:16]={sha[:16]} (OK) N={len(val_np)}")
    return base, val_np


def run_self_test(args, device):
    """Load ONLY ff_canonical; reproduce EVERY published anchor. PASS iff all within tol."""
    print(f"\n=== SELF-TEST: ff_canonical reproduces published anchors ===")
    print(f"  ckpt: {args.ff_canonical_ckpt}")
    base, val_np = _prep_val(device)
    model, ck = _load_model(args.ff_canonical_ckpt, AVWordResNet, device)
    print(f"  loaded best_val_acc={ck.get('best_val_acc', float('nan')):.4%} "
          f"val_sha={str(ck.get('val_idx_sha256', ''))[:16]}")
    specs = _axis_specs(base, val_np)
    results = _evaluate([("ff_canonical", model)], specs, device, verbose=False)

    rows, max_abs, n_anchored, n_fail = [], 0.0, 0, 0
    print(f"\n  {'axis':>22s} {'condition':>14s}  {'measured':>9s} {'anchor':>9s} "
          f"{'Δpp':>9s}  status")
    for axis, cond, _v, _tz in specs:
        anc = FF_ANCHOR.get((axis, cond))
        if anc is None:
            continue
        got = results[(axis, cond)]["ff_canonical"]
        d = got - anc
        n_anchored += 1
        max_abs = max(max_abs, abs(d))
        status = "OK" if abs(d) <= args.tol else "FAIL"
        n_fail += status == "FAIL"
        rows.append((axis, cond, got, anc, d, status))
        print(f"  {axis:>22s} {cond:>14s}  {got:9.6f} {anc:9.6f} {d * 100:+9.4f}  {status}")

    with open(SELFTEST_OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["axis", "condition", "measured_acc", "anchor_acc", "delta_pp", "status"])
        for axis, cond, got, anc, d, status in rows:
            w.writerow([axis, cond, f"{got:.6f}", f"{anc:.6f}", f"{d * 100:+.4f}", status])

    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"\nSELF-TEST {verdict}: {n_anchored - n_fail}/{n_anchored} anchors within "
          f"{args.tol * 100:.3f}pp | max|Δ|={max_abs * 100:.4f}pp")
    n_bitclose = sum(1 for *_, d, _s in rows if abs(d) <= 2e-4)
    print(f"  ({n_bitclose}/{n_anchored} reproduce within 0.02pp = bit-close)")
    print(f"wrote {SELFTEST_OUT}")
    print("DONE")
    if n_fail:
        raise SystemExit(2)


def run_full(args, device):
    base, val_np = _prep_val(device)
    specs = _axis_specs(base, val_np)
    model_specs = [
        ("recurrent", args.recurrent_ckpt, AVRecurrentWordResNet),
        ("ff_matched", args.ff_matched_ckpt, AVWordResNet),
        ("ff_canonical", args.ff_canonical_ckpt, AVWordResNet),
    ]
    models = []
    for name, path, cls in model_specs:
        m, ck = _load_model(path, cls, device)
        models.append((name, m))
        print(f"  {name:>13s}: best_val_acc={ck.get('best_val_acc', float('nan')):.4%} "
              f"seed={ck.get('seed', '?')} arch={ck.get('arch', cls.__name__)}")

    print("\n[evaluating 4 axes x 3 models]")
    results = _evaluate(models, specs, device, verbose=True)

    # harness self-validation (ff_canonical must reproduce anchors)
    print("\n[self-check] ff_canonical arm vs published anchors")
    ffc_clean = results[("clean_val", "clean")]["ff_canonical"]
    ffc_s05 = results[("rawnoise_sweep", "0.0500")]["ff_canonical"]
    assert abs(ffc_clean - 0.956712) < HARD_TOL, f"ff_canonical clean {ffc_clean} != anchor"
    assert abs(ffc_s05 - 0.609268) < HARD_TOL, f"ff_canonical σ0.05 {ffc_s05} != anchor"
    flagged = [(ax, cd, results[(ax, cd)]["ff_canonical"], anc)
               for (ax, cd), anc in FF_ANCHOR.items()
               if abs(results[(ax, cd)]["ff_canonical"] - anc) > SOFT_TOL]
    if flagged:
        print(f"  [WARN] {len(flagged)} ff_canonical points exceed {SOFT_TOL * 100:.1f}pp:")
        for ax, cd, got, anc in flagged:
            print(f"      {ax} {cd}: {got:.6f} vs {anc:.6f} ({(got - anc) * 100:+.3f}pp)")
    else:
        print(f"  all {len(FF_ANCHOR)} anchored points within {SOFT_TOL * 100:.1f}pp [OK]")

    order = ([("clean_val", "clean")]
             + [("rawnoise_sweep", f"{s:.4f}") for s in SIGMA_LEVELS]
             + [("E11_temporal", f"delta_ms={d:+d}") for d in E11_DELTA_MS]
             + [("D5_temporal_saliency", f"win=[{t},{t + D5_WINDOW})")
                for t in range(0, D5_NFRAMES - D5_WINDOW + 1, D5_STEP)])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["axis", "condition", "recurrent_acc", "ff_matched_acc",
                    "ff_canonical_acc", "rec_minus_ffmatched_pp",
                    "ffmatched_minus_ffcanonical_pp", "rec_minus_ffcanonical_pp",
                    "ff_canonical_anchor", "ff_canonical_anchor_delta_pp", "note"])
        for axis, cond in order:
            r = results[(axis, cond)]
            rec, ffm, ffc = r["recurrent"], r["ff_matched"], r["ff_canonical"]
            anc = FF_ANCHOR.get((axis, cond))
            ancs = "" if anc is None else f"{anc:.6f}"
            ancd = "" if anc is None else f"{(ffc - anc) * 100:+.4f}"
            note = ""
            if axis in ("D5_temporal_saliency", "rawnoise_sweep"):
                note = "saliency/robustness — read per-model vs that model's clean_val row"
            elif axis == "E11_temporal":
                note = "congruence — read per-model vs that model's delta_ms=+0 row"
            w.writerow([axis, cond, f"{rec:.6f}", f"{ffm:.6f}", f"{ffc:.6f}",
                        f"{(rec - ffm) * 100:+.4f}", f"{(ffm - ffc) * 100:+.4f}",
                        f"{(rec - ffc) * 100:+.4f}", ancs, ancd, note])
    print(f"\nwrote {args.out}")

    rc = results[("clean_val", "clean")]
    print(f"\nSUMMARY clean: rec={rc['recurrent']:.4%} ffm={rc['ff_matched']:.4%} "
          f"ffc={rc['ff_canonical']:.4%}")
    print(f"  PURE recurrence (rec-ffm) = {(rc['recurrent'] - rc['ff_matched']) * 100:+.2f}pp")
    print(f"  run-to-run variance (ffm-ffc) = "
          f"{(rc['ff_matched'] - rc['ff_canonical']) * 100:+.2f}pp")
    print("DONE")


def main():
    ap = argparse.ArgumentParser(description="Q14 recurrent-vs-feedforward battery")
    ap.add_argument("--recurrent-ckpt", default=REC_CKPT)
    ap.add_argument("--ff-matched-ckpt", default=FFM_CKPT)
    ap.add_argument("--ff-canonical-ckpt", default=FFC_CKPT)
    ap.add_argument("--self-test", action="store_true",
                    help="load ONLY ff_canonical and reproduce ALL published anchors")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tol", type=float, default=SELFTEST_TOL,
                    help="self-test PASS tolerance per anchored point")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.self_test:
        run_self_test(args, device)
    else:
        run_full(args, device)


if __name__ == "__main__":
    main()
