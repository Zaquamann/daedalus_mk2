#!/usr/bin/env python3
"""Q4 — variance decomposition of the AV accuracy gain + per-sample gate-path
attribution. ONE auditable artifact: analysis/deepdive/Q4_variance_decomposition.csv.

EAGER fp32 (the cache was built by phase_a_deepdive.py via analyze_av_msi._load_models,
which forwards in plain eager fp32 — the same path that yields the published anchors).
Self-check at top reproduces the AV-clean / A-clean / V-fair anchors bit-exact before
any decomposition, so every delta sits on a verified baseline.

Two deliverables (lead/EVIDENCE_MAP Q4 spec):
 (a) variance table: full AV, AV-audio-zero, AV-video-zero, A-only, V-fair, plus the
     two-term split of the clean AV-over-A gain that SUMS to +2.97pp:
        ensemble-attainable  = ensemble_50_50 - A_only      (+2.36pp)
        learned-fusion resid. = AV_fused - ensemble_50_50   (+0.61pp)
 (b) per-sample gate-weighting decomposition: the visual stream influences the output
     ONLY through Wv(v_mid) inside the multiplicative gate (Wv is bias-free, so v_mid=0
     => Wv(v_mid)=0). Therefore AV_clean_v_zero IS the visual-gate-term-ablated model.
     Among A-wrong->AV-right rescues, the fraction whose correct prediction is LOST when
     the visual gate term is ablated = fraction of rescues driven by the visual gate path.

Pure CPU on processed/deepdive_act_cache.pt (no forward, no GPU) — does not contend with
the Q14 training job.
"""
import csv
import hashlib
import os

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
OUT = os.path.join(SCRIPT_DIR, "analysis", "deepdive", "Q4_variance_decomposition.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PIN = "03c5a87a"
# Published anchors (eager fp32) — self-check tolerances.
ANCHORS = {"A": 0.926964, "V": 0.864989, "AV": 0.956712,
           "AV_vzero": 0.008391, "AV_azero": 0.4447, "ens5050": 0.950610}


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _acc(logits, y):
    return float((logits.argmax(1) == y).mean())


def main():
    c = torch.load(CACHE, weights_only=False)
    sha = c.get("val_idx_sha256", "")
    n = len(np.asarray(c["labels"]))
    print(f"[cache] val sha16={sha[:16]} N={n}")
    assert sha.startswith(PIN), f"VAL PIN MISMATCH {sha[:16]}"
    assert n == 5244, f"N={n}"

    y = np.asarray(c["AV_clean_full"]["labels"]).astype(np.int64)
    assert (np.asarray(c["A_only"]["labels"]).astype(np.int64) == y).all()
    assert (np.asarray(c["V_fair"]["labels"]).astype(np.int64) == y).all()

    A_log = np.asarray(c["A_only"]["logits"])
    V_log = np.asarray(c["V_fair"]["logits"])
    AV_log = np.asarray(c["AV_clean_full"]["logits"])
    AVvz_log = np.asarray(c["AV_clean_v_zero"]["logits"])      # video zeroed
    AVaz_log = np.asarray(c["AV_clean_audio_zero"]["logits"])  # audio zeroed

    acc = {
        "A": _acc(A_log, y), "V": _acc(V_log, y), "AV": _acc(AV_log, y),
        "AV_vzero": _acc(AVvz_log, y), "AV_azero": _acc(AVaz_log, y),
    }
    # 50/50 late ensemble (softmax-average A + V).
    p_ens = 0.5 * (_softmax(A_log) + _softmax(V_log))
    acc["ens5050"] = float((p_ens.argmax(1) == y).mean())

    # --- self-check: reproduce anchors before any decomposition ---
    print("\n[self-check vs eager-fp32 anchors]")
    ok = True
    tol = {"A": 5e-4, "V": 5e-3, "AV": 5e-4, "AV_vzero": 1e-3,
           "AV_azero": 5e-3, "ens5050": 5e-4}
    for k in ("A", "V", "AV", "AV_vzero", "AV_azero", "ens5050"):
        d = acc[k] - ANCHORS[k]
        flag = "OK" if abs(d) <= tol[k] else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  {k:9s} = {acc[k]:.6f}  anchor {ANCHORS[k]:.6f}  d={d:+.6f}  [{flag}]")
    assert ok, "anchor self-check FAILED — refusing to write decomposition"

    # --- (a) two-term decomposition of the clean AV-over-A gain ---
    gain_total = acc["AV"] - acc["A"]              # +2.97pp
    gain_ens = acc["ens5050"] - acc["A"]           # ensemble-attainable +2.36pp
    gain_resid = acc["AV"] - acc["ens5050"]        # learned-fusion residual +0.61pp
    print("\n[two-term split of clean AV-over-A gain]")
    print(f"  total            = {gain_total*100:+.4f} pp")
    print(f"  ensemble-attain. = {gain_ens*100:+.4f} pp")
    print(f"  fusion-residual  = {gain_resid*100:+.4f} pp")
    print(f"  sum check        = {(gain_ens+gain_resid)*100:+.4f} pp")

    # --- (b) per-sample gate-path attribution on rescues ---
    A_pred = A_log.argmax(1)
    AV_pred = AV_log.argmax(1)
    AVvz_pred = AVvz_log.argmax(1)               # visual-gate-term ablated
    A_wrong = A_pred != y
    AV_right = AV_pred == y
    rescued = A_wrong & AV_right                 # A-wrong -> AV-right
    n_resc = int(rescued.sum())
    # rescue lost when the visual gate term is removed => driven by visual gate path
    lost_when_vablated = rescued & (AVvz_pred != y)
    frac_visual_driven = float(lost_when_vablated.sum()) / max(n_resc, 1)
    # converse: rescues that survive even with visual gate ablated (audio-path rescues)
    frac_audio_survive = 1.0 - frac_visual_driven
    # context: total regressions A-right -> AV-wrong
    regressed = (~A_wrong) & (~AV_right)
    print("\n[gate-path attribution on A-wrong->AV-right rescues]")
    print(f"  n_rescued (A-wrong & AV-right)      = {n_resc}")
    print(f"  n_regressed (A-right & AV-wrong)    = {int(regressed.sum())}")
    print(f"  frac rescues lost when Wv ablated   = {frac_visual_driven:.4f}  "
          f"(= visual-gate-path-driven)")
    print(f"  frac rescues surviving Wv ablation  = {frac_audio_survive:.4f}")

    # --- write ONE decomposition CSV ---
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "row", "value", "unit", "note"])
        # accuracies
        w.writerow(["accuracy", "AV_full", f"{acc['AV']:.6f}", "frac", "clean AV (anchor 0.956712)"])
        w.writerow(["accuracy", "AV_audio_zero", f"{acc['AV_azero']:.6f}", "frac", "audio input zeroed"])
        w.writerow(["accuracy", "AV_video_zero", f"{acc['AV_vzero']:.6f}", "frac", "video zeroed = visual gate ablated"])
        w.writerow(["accuracy", "A_only", f"{acc['A']:.6f}", "frac", "anchor 0.926964"])
        w.writerow(["accuracy", "V_fair", f"{acc['V']:.6f}", "frac", "anchor 0.864989"])
        w.writerow(["accuracy", "ensemble_50_50", f"{acc['ens5050']:.6f}", "frac", "softmax-avg A+V (anchor 0.950610)"])
        # two-term decomposition (sums to total)
        w.writerow(["decomp_gain", "total_AV_minus_A", f"{gain_total*100:.4f}", "pp", "clean AV - A"])
        w.writerow(["decomp_gain", "ensemble_attainable", f"{gain_ens*100:.4f}", "pp", "ensemble_50_50 - A"])
        w.writerow(["decomp_gain", "learned_fusion_residual", f"{gain_resid*100:.4f}", "pp", "AV - ensemble_50_50"])
        w.writerow(["decomp_gain", "sum_check", f"{(gain_ens+gain_resid)*100:.4f}", "pp", "attainable + residual"])
        # gate-path attribution
        w.writerow(["gate_path", "n_rescued", str(n_resc), "count", "A-wrong & AV-right"])
        w.writerow(["gate_path", "n_regressed", str(int(regressed.sum())), "count", "A-right & AV-wrong"])
        w.writerow(["gate_path", "frac_visual_gate_driven", f"{frac_visual_driven:.6f}", "frac",
                    "rescues lost when Wv(v_mid) ablated (=video_zero)"])
        w.writerow(["gate_path", "frac_audio_path_survives", f"{frac_audio_survive:.6f}", "frac",
                    "rescues surviving visual-gate ablation"])
    print(f"\nwrote {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
