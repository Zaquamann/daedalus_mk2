#!/usr/bin/env python3
"""Q17 (optional extend) — A-vs-AV prediction overlap on the CLEAN val set.
ONE artifact: analysis/deepdive/Q17_a_vs_av_prediction_overlap.csv.

Q17 ("how does an A model handle an AV stimulus vs how does an AV model handle it")
has a literal reading that is ARCHITECTURALLY IMPOSSIBLE — audio_only_filtered.pt
(WordResNet) has no video input port, so you cannot feed it an AV tensor; that would
need a new architecture + retrain (NOT built here). The McGurk-conflict reading is
already answered by E3 (A follows audio 92.73%, AV fuses to a third word 82.73% on
n=1500 viseme-distinct pairs). This script builds the cheap clarifying extend the
EVIDENCE_MAP names: generalize E3 from conflict pairs to MATCHED-content clean audio by
reporting, per stimulus, the prediction overlap between A-only and AV-fused.

EAGER fp32, inference-only, cache-based: A-only and AV-fused predictions are argmaxes of
the cached eager-fp32 logits (processed/deepdive_act_cache.pt) on the pinned val
(sha 03c5a87a, N=5244, 180 classes) — the same path that yields the published anchors.
Fully DETERMINISTIC (argmax + counting); no probe, no forward, no GPU.

Metrics: (a) % clips where A and AV agree; (b) AV accuracy on the subset A gets wrong
(rescue set); (c) A accuracy on the subset AV gets wrong (regression set); plus the full
agreement contingency and who-is-right-when-they-disagree.
"""
import csv
import os

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
OUT = os.path.join(SCRIPT_DIR, "analysis", "deepdive", "Q17_a_vs_av_prediction_overlap.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
PIN = "03c5a87a"


def main():
    c = torch.load(CACHE, weights_only=False)
    assert c.get("val_idx_sha256", "").startswith(PIN), "VAL PIN MISMATCH"
    y = np.asarray(c["labels"]).astype(np.int64)
    N = len(y)
    assert N == 5244, N

    a_pred = np.asarray(c["A_only"]["logits"]).argmax(1)
    av_pred = np.asarray(c["AV_clean_full"]["logits"]).argmax(1)
    a_ok = a_pred == y
    av_ok = av_pred == y
    a_acc, av_acc = float(a_ok.mean()), float(av_ok.mean())

    # self-check anchors (deterministic argmax)
    print(f"[self-check] A_acc={a_acc:.6f} (0.926964)  AV_acc={av_acc:.6f} (0.956712)")
    assert abs(a_acc - 0.926964) < 5e-4 and abs(av_acc - 0.956712) < 5e-4, "anchor mismatch"

    agree = a_pred == av_pred
    both_ok = a_ok & av_ok
    both_wrong = (~a_ok) & (~av_ok)
    regression = a_ok & (~av_ok)          # A right, AV wrong
    rescue = (~a_ok) & av_ok               # A wrong, AV right
    shared_err = agree & (~a_ok)           # same prediction AND it's wrong
    disagree = ~agree
    # among disagreements, who is right
    dis_a_right = disagree & a_ok
    dis_av_right = disagree & av_ok
    dis_neither = disagree & (~a_ok) & (~av_ok)

    n_a_wrong = int((~a_ok).sum())
    n_av_wrong = int((~av_ok).sum())
    av_acc_on_a_wrong = float(av_ok[~a_ok].mean())   # rescue rate among A-errors
    a_acc_on_av_wrong = float(a_ok[~av_ok].mean())    # A-right rate among AV-errors
    n_dis = int(disagree.sum())

    # cross-check rescue/regression counts vs Q4 (n_rescued=261, n_regressed=105)
    print(f"[xcheck Q4] rescue n={int(rescue.sum())} (261)  "
          f"regression n={int(regression.sum())} (105)")
    assert int(rescue.sum()) == 261 and int(regression.sum()) == 105, "Q4 count mismatch"

    rows = [
        ("selfcheck", "A_only_acc", a_acc, N, "anchor 0.926964"),
        ("selfcheck", "AV_fused_acc", av_acc, N, "anchor 0.956712"),
        ("overlap", "agreement_rate", float(agree.mean()), int(agree.sum()),
         "A_pred == AV_pred (any label)"),
        ("overlap", "disagreement_rate", float(disagree.mean()), n_dis, ""),
        ("contingency", "both_correct", float(both_ok.mean()), int(both_ok.sum()), ""),
        ("contingency", "both_wrong", float(both_wrong.mean()), int(both_wrong.sum()), ""),
        ("contingency", "A_right_AV_wrong_regression", float(regression.mean()),
         int(regression.sum()), "AV loses what A had"),
        ("contingency", "A_wrong_AV_right_rescue", float(rescue.mean()),
         int(rescue.sum()), "AV rescues what A missed"),
        ("contingency", "shared_error_same_label", float(shared_err.mean()),
         int(shared_err.sum()), "agree on the SAME wrong word"),
        ("conditional", "AV_acc_on_A_wrong_set", av_acc_on_a_wrong, n_a_wrong,
         "rescue rate among A-errors = 261/383"),
        ("conditional", "A_acc_on_AV_wrong_set", a_acc_on_av_wrong, n_av_wrong,
         "A-right rate among AV-errors = 105/227"),
        ("disagree", "frac_disagree_A_right", float(dis_a_right.sum()) / max(n_dis, 1),
         int(dis_a_right.sum()), "of disagreements, A is the correct one"),
        ("disagree", "frac_disagree_AV_right", float(dis_av_right.sum()) / max(n_dis, 1),
         int(dis_av_right.sum()), "of disagreements, AV is the correct one"),
        ("disagree", "frac_disagree_neither_right",
         float(dis_neither.sum()) / max(n_dis, 1), int(dis_neither.sum()),
         "both wrong but different labels"),
        ("mcgurk_ref", "E3_A_audio_capture_distinct", 0.9273, 1500,
         "E3 conflict pairs (existing) — A follows audio"),
        ("mcgurk_ref", "E3_AV_third_word_distinct", 0.8273, 1500,
         "E3 conflict pairs (existing) — AV fuses to a third word"),
    ]

    print(f"\n  agree={agree.mean()*100:.2f}%  both_correct={both_ok.mean()*100:.2f}%  "
          f"rescue={int(rescue.sum())}  regression={int(regression.sum())}  "
          f"shared_err={int(shared_err.sum())}")
    print(f"  AV acc on A-wrong set = {av_acc_on_a_wrong*100:.2f}% (n={n_a_wrong}); "
          f"A acc on AV-wrong set = {a_acc_on_av_wrong*100:.2f}% (n={n_av_wrong})")
    print(f"  when they DISAGREE (n={n_dis}): A-right {dis_a_right.sum()}, "
          f"AV-right {dis_av_right.sum()}, neither {dis_neither.sum()}")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "metric", "value", "count", "note"])
        for sec, metric, val, cnt, note in rows:
            w.writerow([sec, metric, f"{val:.6f}", cnt, note])
    print(f"\nwrote {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
