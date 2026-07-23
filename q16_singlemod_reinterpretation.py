#!/usr/bin/env python3
"""Q16 TEST A — does access to both modalities change how the SINGLE (audio) modality
is represented? Compare the linear-DECODABILITY (not accuracy) of the AV net's audio
mid-rep (a_mid, video zeroed) vs the A-only net's first block on IDENTICAL audio.
ONE artifact: analysis/deepdive/Q16_singlemod_reinterpretation.csv.

EAGER fp32, inference-only: works on the cached eager-fp32 activations
(processed/deepdive_act_cache.pt), the same path that yields the published anchors.
The cache already holds the av_fused.pt VIDEO-ZEROED forward (AV_clean_v_zero) and the
audio_only_filtered.pt forward (A_only) on the pinned val (sha 03c5a87a, N=5244).

CRITICAL (per EVIDENCE_MAP): the reinterpretation metric is probe-DECODABILITY on
identical input — NOT AV-audio-only ACCURACY (the AV net collapses to 0.84% with
audio-only input; its audio path is not a standalone reader).

What this nails:
  1. AIRTIGHT IDENTICAL-AUDIO: a_mid is bit-identical whether video is real or zeroed
     (gate is AFTER a_mid) => the representational difference vs A-only is purely
     joint-training-induced, not video leaking into the audio branch.
  2. DECODABILITY (word/onset/viseme), 5-fold linear probe (phase_f_flow._probe_5fold,
     copied verbatim): the SAME audio is LESS word/viseme-decodable in the AV audio
     branch than in the A-only block1.
  3. GEOMETRY/FUNCTION DISSOCIATION: linear CKA(A-only block1, AV a_mid) ~0.976 — the
     subspace is near-identical yet carries less linearly-decodable word info; the
     interpretation shift is invisible to CKA.

SELF-CHECKS reproduce the published D5/D4 numbers before any new claim:
  AV a_mid word 0.279558 / A_only block1 word 0.426774 (D5_layer_decodability_word.csv);
  AV a_mid viseme 0.569777 / A_only block1 viseme 0.631643 (D5_..._viseme.csv);
  linear CKA 0.976481 (D4_cka_matrix.csv).
The deterministic CKA and the word/AV + viseme probes reproduce bit-exact; the iterative
word/A-block1 lbfgs probe lands ~0.2% low on pod sklearn 1.9.0 (the local anchors were
generated under 1.8.0) and is held to the project's ≤0.5% LR-probe guardrail. Every
selfcheck row records the actual value and its signed delta vs the anchor, undiagnosed.

Pure CPU on the cache (no forward, no GPU) — does not contend with the Q14 job.
"""
import csv
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from analyze_av_phonetics import viseme_class as _viseme
from analyze_phoneme_accuracy import get_onset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
OUT = os.path.join(SCRIPT_DIR, "analysis", "deepdive", "Q16_singlemod_reinterpretation.csv")
AV_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused.pt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
PIN = "03c5a87a"
# published anchors for self-check (D5_layer_decodability_*.csv, D4_cka_matrix.csv)
PUB = {("word", "AV"): 0.279558, ("word", "A"): 0.426774,
       ("viseme", "AV"): 0.569777, ("viseme", "A"): 0.631643, "cka": 0.976481}


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


def _linear_cka(X, Y):
    """phase_f_flow._linear_cka, verbatim."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = (X.T @ Y).reshape(-1)
    num = float(np.dot(num, num))
    den_x = float(np.linalg.norm(X.T @ X, "fro"))
    den_y = float(np.linalg.norm(Y.T @ Y, "fro"))
    return num / (den_x * den_y + 1e-12)


def main():
    c = torch.load(CACHE, weights_only=False)
    assert c.get("val_idx_sha256", "").startswith(PIN), "VAL PIN MISMATCH"
    labels = np.asarray(c["labels"]).astype(np.int64)
    assert len(labels) == 5244, len(labels)
    idx_to_label = torch.load(AV_CKPT, weights_only=False)["idx_to_label"]

    a_block1 = np.asarray(c["A_only"]["block1_gap"])          # A-only net, first block
    av_amid_full = np.asarray(c["AV_clean_full"]["a_mid_gap"])  # AV audio branch, real video
    av_amid_vz = np.asarray(c["AV_clean_v_zero"]["a_mid_gap"])  # AV audio branch, video ZEROED

    # ---- 1) airtight identical-audio: a_mid is video-independent ----
    diff = np.abs(av_amid_vz - av_amid_full)
    max_d, mean_d = float(diff.max()), float(diff.mean())
    print(f"[identical-audio] a_mid(video-zeroed) vs a_mid(full): "
          f"max|Δ|={max_d:.3e} mean|Δ|={mean_d:.3e}")
    assert max_d < 1e-5, f"a_mid NOT video-independent (max|Δ|={max_d})"

    # readout label vectors + keep masks (match D5_1_layer_probe exactly)
    onsets = np.asarray([get_onset(idx_to_label[int(l)]) for l in labels])
    visemes = np.asarray([_viseme(idx_to_label[int(l)]) for l in labels])
    keep_v = visemes != "other"
    keep_o = (onsets != "vowel") & (onsets != "other")
    readouts = [("word", labels, slice(None)),
                ("onset", onsets, keep_o),
                ("viseme", visemes, keep_v)]

    rows = []

    # ---- 2) decodability on IDENTICAL audio: A-only block1 vs AV a_mid (video-zeroed) ----
    print("[decodability] 5-fold linear probe on identical audio")
    dec = {}
    for tname, y, keep in readouts:
        Xa = a_block1 if keep is slice(None) else a_block1[keep]
        Xv = av_amid_vz if keep is slice(None) else av_amid_vz[keep]
        yk = y if keep is slice(None) else y[keep]
        a_acc, a_bal = _probe_5fold(Xa, yk)
        v_acc, v_bal = _probe_5fold(Xv, yk)
        dec[tname] = (a_acc, a_bal, v_acc, v_bal)
        gap = (a_acc - v_acc) * 100
        print(f"    {tname:>7s} | A_block1 acc={a_acc*100:5.2f}% | "
              f"AV_a_mid acc={v_acc*100:5.2f}% | gap={gap:+5.2f}pp")
        rows += [
            ("decodability", tname, "A_only_block1", "acc_5fold", f"{a_acc:.6f}", ""),
            ("decodability", tname, "A_only_block1", "bal_acc_5fold", f"{a_bal:.6f}", ""),
            ("decodability", tname, "AV_a_mid_vzero", "acc_5fold", f"{v_acc:.6f}",
             "video-zeroed AV audio branch"),
            ("decodability", tname, "AV_a_mid_vzero", "bal_acc_5fold", f"{v_bal:.6f}", ""),
            ("decodability", tname, "block1_minus_a_mid", "acc_gap_pp", f"{gap:.4f}",
             "A-only block1 more decodable on identical audio"),
        ]

    # ---- 3) geometry/function dissociation: linear CKA(A block1, AV a_mid) ----
    cka = _linear_cka(a_block1, av_amid_full)
    print(f"[geometry] linear CKA(A_block1, AV_a_mid) = {cka:.6f} (pub {PUB['cka']:.6f})")
    rows.append(("geometry", "a_mid_vs_block1", "A_x_AV", "linear_CKA", f"{cka:.6f}",
                 "near-identical subspace despite the decodability gap"))

    # ---- self-checks (reproduce published D5/D4 bit-exact) ----
    av_word_full = _probe_5fold(av_amid_full, labels)[0]   # full-video a_mid == vzero
    # Per-check tolerance (lead ruling 2026-06-15): the iterative lbfgs word/A-block1
    # probe reproduces to ~0.2% across sklearn versions (pod 1.9.0 vs the 1.8.0 that
    # generated the local D5 anchors), so it is held to the PROJECT'S established
    # ≤0.5% LR-probe reproduction guardrail — the same standard the validator uses to
    # clear every Q6/Q8/Q10/Q11 lbfgs probe — NOT a private exception. The deterministic
    # CKA and the lbfgs checks that DID reproduce within 5e-4 (word/AV, viseme/*) stay
    # hard-gated. The actual pod value AND its signed delta vs the local anchor are
    # recorded VISIBLY in every selfcheck row (cause left undiagnosed, per ruling).
    GUARDRAIL = 5e-3   # project ≤0.5% lbfgs-LR reproduction guardrail
    TIGHT = 5e-4       # bit-exact gate for checks that reproduce exactly
    checks = [
        (("word", "AV"), dec["word"][2], TIGHT),
        (("word", "A"),  dec["word"][0], GUARDRAIL),   # cross-version-drifting probe
        (("viseme", "AV"), dec["viseme"][2], TIGHT),
        (("viseme", "A"),  dec["viseme"][0], TIGHT),
        ("cka", cka, TIGHT),                            # deterministic — hard-gated
    ]
    print("[self-check vs published]")
    ok = True
    sc_delta = {}   # lbl -> signed delta-vs-anchor (%), surfaced as its OWN CSV column
    for name, ds, tol in checks:
        anchor = PUB[name]
        delta = ds - anchor
        flag = "OK" if abs(delta) <= tol else "FAIL"
        if flag == "FAIL":
            ok = False
        lbl = name if isinstance(name, str) else f"{name[0]}/{name[1]}"
        sc_delta[lbl] = f"{delta*100:+.3f}%"
        print(f"    {lbl:>12s} = {ds:.6f}  anchor {anchor:.6f}  Δ={delta*100:+.3f}%  "
              f"[{flag}, tol={tol:g}]")
        rows.append(("selfcheck", lbl, "reproduce_published", "value", f"{ds:.6f}",
                     f"anchor {anchor:.6f} delta {delta*100:+.3f}% tol {tol:g} [{flag}]"))
    # video-independence rows
    rows += [
        ("identical_audio", "a_mid", "AV_vzero_vs_full", "max_abs_diff", f"{max_d:.3e}",
         "a_mid identical with video zeroed => audio-only determined"),
        ("identical_audio", "a_mid", "AV_vzero_vs_full", "mean_abs_diff", f"{mean_d:.3e}", ""),
    ]
    # cross-check that the full-video a_mid word probe also reproduces 0.279558
    assert abs(av_word_full - PUB[("word", "AV")]) < 5e-4, f"a_mid(full) word {av_word_full}"
    assert ok, "published-anchor self-check FAILED — refusing to write"

    # delta-vs-anchor is a FLAGGED first-class column (not buried in the note) for the
    # selfcheck rows; blank for the science rows. word/A reads -0.210% at tol 0.5%.
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "readout", "model", "metric", "value", "note",
                    "delta_pct_vs_anchor"])
        for r in rows:
            dcol = sc_delta.get(r[1], "") if r[0] == "selfcheck" else ""
            w.writerow(list(r) + [dcol])
    print(f"\nwrote {OUT}")
    print("[SUMMARY] identical audio -> A_only block1 word-decodability "
          f"{dec['word'][0]*100:.1f}% vs AV a_mid {dec['word'][2]*100:.1f}% "
          f"(gap {(dec['word'][0]-dec['word'][2])*100:+.1f}pp) at CKA {cka:.3f}: "
          "same subspace, differently-structured (joint-training reinterpretation).")
    print("DONE")


if __name__ == "__main__":
    main()
