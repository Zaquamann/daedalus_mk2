#!/usr/bin/env python3
"""Q18 — per-layer audio-only (v_mid=0) word decodability staircase of av_fused.pt.
ONE artifact: analysis/deepdive/Q18_perlayer_audio_only_decodability.csv.

Existing D5 has only the FULL-AV per-layer staircase (a_mid 0.280 -> gate_out 0.760 ->
block2 0.943); D4 has the v_mid=0 condition ONLY at penult (0.526). This fills the gap:
the per-LAYER v_mid=0 staircase, isolating whether the early-depth audio regression is
recoverable rotation vs true loss, and quantifying the v_mid=0 cliff per layer.

EAGER fp32, inference-only: works on the cached eager-fp32 activations
(processed/deepdive_act_cache.pt). The cache holds the av_fused.pt VIDEO-ZEROED forward
(AV_clean_v_zero) at every layer, the full-AV forward (AV_clean_full), and the A-only
forward (A_only) on the pinned val (sha 03c5a87a, N=5244, 180 classes). No forward, no GPU.

Probe: phase_f_flow._probe_5fold (copied verbatim) — 5-fold stratified, per-fold
z-score, LogisticRegression(max_iter=1500, C=1.0), word target (180 classes).

GUARDRAILS are built on DETERMINISTIC quantities (a_mid is identical with video zeroed;
the staircase is monotone) rather than bit-exact LR reproduction: the iterative lbfgs
probe reproduces published anchors only to ~0.2-0.3pp across sklearn versions (pod has
1.9.0; the D5/D4 CSVs were generated under an older sklearn — see the Q16 finding). The
published anchors are therefore reported as CROSS-REFERENCE columns (reproduced value,
published value, delta) for transparent verification, not hard-asserted. The SCIENCE is a
WITHIN-RUN comparison (full-AV vs v_mid=0 vs A-only, identical probe/version) — version-
independent in its conclusion.
"""
import csv
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
OUT = os.path.join(SCRIPT_DIR, "analysis", "deepdive",
                   "Q18_perlayer_audio_only_decodability.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
PIN = "03c5a87a"
# published word-probe anchors (D5_layer_decodability_word.csv / D4_linprobe_class.csv)
ANCHOR = {
    ("A_only", "block1_gap"): 0.426774, ("A_only", "block2_gap"): 0.902745,
    ("A_only", "penult"): 0.902745,
    ("AV_clean_full", "a_mid_gap"): 0.279558, ("AV_clean_full", "gate_out_gap"): 0.759915,
    ("AV_clean_full", "block2_gap"): 0.943173, ("AV_clean_full", "penult"): 0.945271,
    ("AV_clean_v_zero", "penult"): 0.526320,   # the v_mid=0 cliff (D4_linprobe_class)
}


def _probe_5fold(X, y, max_iter=1500, C=1.0, seed=0):
    """phase_f_flow._probe_5fold, verbatim — (mean acc, mean balanced acc), 5-fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, bal = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        bal.append(balanced_accuracy_score(y[te], pred))
    return float(np.mean(accs)), float(np.mean(bal))


def main():
    c = torch.load(CACHE, weights_only=False)
    assert c.get("val_idx_sha256", "").startswith(PIN), "VAL PIN MISMATCH"
    y = np.asarray(c["labels"]).astype(np.int64)
    assert len(y) == 5244, len(y)

    # deterministic guardrail: a_mid is video-independent (gate is AFTER a_mid)
    amid_full = np.asarray(c["AV_clean_full"]["a_mid_gap"])
    amid_vz = np.asarray(c["AV_clean_v_zero"]["a_mid_gap"])
    max_d = float(np.abs(amid_vz - amid_full).max())
    print(f"[guardrail] a_mid(v_zero) vs a_mid(full) max|Δ|={max_d:.3e}")
    assert max_d < 1e-5, f"a_mid NOT video-independent ({max_d})"

    # the staircases to probe (condition, [layers])
    plan = [
        ("A_only", ["block1_gap", "block2_gap", "penult"]),
        ("AV_clean_full", ["a_mid_gap", "gate_out_gap", "block2_gap", "penult"]),
        ("AV_clean_v_zero", ["a_mid_gap", "gate_out_gap", "block2_gap", "penult"]),
    ]
    acc = {}      # (cond, layer) -> (acc, bal)
    print(f"\n{'condition':>16s} {'layer':>13s} {'acc':>7s} {'bal':>7s} "
          f"{'anchor':>8s} {'Δpp':>6s}")
    for cond, layers in plan:
        for layer in layers:
            # reuse: v_zero a_mid == full a_mid (identical array, deterministic)
            if cond == "AV_clean_v_zero" and layer == "a_mid_gap":
                acc[(cond, layer)] = acc[("AV_clean_full", "a_mid_gap")]
            else:
                acc[(cond, layer)] = _probe_5fold(np.asarray(c[cond][layer]), y)
            a, b = acc[(cond, layer)]
            anc = ANCHOR.get((cond, layer))
            dpp = "" if anc is None else f"{(a - anc) * 100:+.2f}"
            ancs = "" if anc is None else f"{anc:.4f}"
            print(f"{cond:>16s} {layer:>13s} {a*100:6.2f}% {b*100:6.2f}% {ancs:>8s} {dpp:>6s}")

    # deterministic guardrail: monotone full-AV staircase (probe is working)
    fa, fg, fb = (acc[("AV_clean_full", s)][0] for s in
                  ("a_mid_gap", "gate_out_gap", "block2_gap"))
    assert fa < fg < fb, f"full-AV staircase not monotone: {fa:.3f} {fg:.3f} {fb:.3f}"

    # cross-reference deltas vs published anchors (REPORTED, not asserted)
    print("\n[cross-reference vs published (LR reproduces ~0.2-0.3pp across sklearn vers.)]")
    for k, anc in ANCHOR.items():
        d = (acc[k][0] - anc) * 100
        print(f"    {k[0]:>16s} {k[1]:>13s}: {acc[k][0]:.6f} vs {anc:.6f}  Δ={d:+.3f}pp")

    # the v_mid=0 cliff per layer = full-AV minus v_zero at each shared layer
    print("\n[v_mid=0 cliff per layer: full-AV acc - v_zero acc]")
    for layer in ("a_mid_gap", "gate_out_gap", "block2_gap", "penult"):
        cliff = (acc[("AV_clean_full", layer)][0] - acc[("AV_clean_v_zero", layer)][0]) * 100
        print(f"    {layer:>13s}: {cliff:+.2f}pp")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["condition", "layer", "acc_5fold", "bal_acc_5fold",
                    "published_anchor", "delta_vs_anchor_pp", "note"])
        for cond, layers in plan:
            for layer in layers:
                a, b = acc[(cond, layer)]
                anc = ANCHOR.get((cond, layer))
                ancs = "" if anc is None else f"{anc:.6f}"
                dpp = "" if anc is None else f"{(a - anc) * 100:.4f}"
                note = ""
                if cond == "AV_clean_v_zero" and layer != "a_mid_gap":
                    note = "audio-only (v_mid=0) — NEW per-layer measurement"
                elif cond == "AV_clean_v_zero":
                    note = "a_mid identical to full-AV (video-independent)"
                w.writerow([cond, layer, f"{a:.6f}", f"{b:.6f}", ancs, dpp, note])
        # v_mid=0 cliff rows
        for layer in ("a_mid_gap", "gate_out_gap", "block2_gap", "penult"):
            cliff = acc[("AV_clean_full", layer)][0] - acc[("AV_clean_v_zero", layer)][0]
            w.writerow(["cliff_full_minus_vzero", layer, f"{cliff:.6f}", "", "", "",
                        "full-AV word-decodability lost when v_mid=0"])
    print(f"\nwrote {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
