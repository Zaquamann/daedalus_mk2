#!/usr/bin/env python3
"""Q12 — integration drivers with the A-only baseline regressed out, + bootstrap
95% CIs on the small-n phonetic bins. ONE artifact:
analysis/deepdive/Q12_drivers_controlled.csv.

Background: the D2.3 RF (phase_e_geometry.py:123-216) predicts per-sample
delta_flip / delta_logconf / delta_margin from viseme one-hots + word_len +
n_vowels + vowel_initial, and word_len dominates (0.507734/0.415869/0.324457). BUT
the A-only baseline is NOT a covariate, and longer words have lower A-only accuracy
(more rescue headroom) — so word_len may just index opportunity-to-rescue.

EAGER fp32: works on the cached eager-fp32 logits (processed/deepdive_act_cache.pt),
the same path that yields the published anchors (A 0.926964, AV 0.956712). No forward.

Deliverables (EVIDENCE_MAP Q12 'NEW TEST NEEDED'):
 1. SELF-CHECK: reproduce the published per-sample RF importances (word_len 0.5077 etc.).
 2. Per-sample RF/regressor refit with a_correct AND a_logp (A log-prob on true class)
    as covariates — does word_len drop / do visemes rise once headroom is in the model?
 3. GROUP-AWARE permutation importance (GroupKFold by WORD class) on the +covariate
    model: honest, generalizable importance free of the impurity cardinality bias AND
    of per-word leakage (constant features within a word).
 4. Per-WORD ridge (180 word rows): standardized coefs of baseline-controlled rescue
    (mean av_correct - mean a_correct per word) on features, WITH vs WITHOUT the
    per-word A-baseline as a covariate — the decisive viseme-vs-word_len comparison.
 5. Paired bootstrap 95% CIs on the per-bin AV-minus-A deltas for all five phonetic
    categorizations (onset/viseme/syllable/length/vowel) — error bars the published
    CSVs lack on small bins (4+ syll n=115, /b/ n=115, glottal_h n=161).

Pure CPU; capped threads/jobs so it stays a good neighbour to the Q14 GPU job.
"""
import csv
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analyze_av_phonetics import viseme_class as _viseme
from analyze_phoneme_accuracy import (get_length_group, get_onset,
                                      get_syllable_group, get_vowel_group)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
OUT = os.path.join(SCRIPT_DIR, "analysis", "deepdive", "Q12_drivers_controlled.csv")
AV_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused.pt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
PIN = "03c5a87a"
NJOBS = 2  # be a good neighbour to the Q14 GPU job
# published per-sample RF importances (D2_integration_drivers.csv) for self-check
PUB_WL = {"delta_flip": 0.507734, "delta_logconf": 0.415869, "delta_margin": 0.324457}
# published per-bin (n, delta=AV-A) anchors for bin-reproduction self-check
PUB_BINS = {
    ("viseme", "glottal_h"): (161, 0.0248), ("viseme", "bilabial_bpm"): (777, 0.0309),
    ("viseme", "lingual"): (2587, 0.0267), ("viseme", "other"): (314, 0.0446),
    ("onset", "/b/"): (115, 0.0435),
    ("syllable", "4+"): (115, 0.0087), ("syllable", "2"): (1968, 0.0356),
    ("syllable", "1"): (2520, 0.0270),
    ("length", "Medium (5-7)"): (2348, 0.0362), ("length", "Long (8+)"): (756, 0.0238),
    ("vowel", "/ɔː/"): (412, 0.0388), ("vowel", "/æ/"): (527, 0.0133),
}


def _logp(logits, lab):
    x = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(x).sum(axis=1, keepdims=True))
    return (x - lse)[np.arange(len(lab)), lab]


def _margin(logits):
    top2 = np.sort(logits, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def _build_features(labels, idx_to_label):
    """Replicate phase_e_geometry.D2_3_drivers feature matrix EXACTLY."""
    visemes = [_viseme(idx_to_label[int(l)]) for l in labels]
    words = [idx_to_label[int(l)] for l in labels]
    viseme_classes = sorted(set(visemes))
    names, cols = [], []
    for o in viseme_classes:
        cols.append(np.asarray([1.0 if x == o else 0.0 for x in visemes]))
        names.append(f"viseme_{o}")
    word_len = np.asarray([len(w) for w in words], dtype=np.float32)
    cols.append(word_len); names.append("word_len")
    n_vowels = np.asarray([sum(1 for c in w if c.lower() in "aeiou") for w in words],
                          dtype=np.float32)
    cols.append(n_vowels); names.append("n_vowels")
    vowel_init = np.asarray([1.0 if w and w[0].lower() in "aeiou" else 0.0 for w in words])
    cols.append(vowel_init); names.append("vowel_initial")
    return np.stack(cols, axis=1), names, np.asarray(visemes), words


def main():
    c = torch.load(CACHE, weights_only=False)
    assert c.get("val_idx_sha256", "").startswith(PIN), "VAL PIN MISMATCH"
    labels = np.asarray(c["labels"]).astype(np.int64)
    assert len(labels) == 5244, len(labels)
    idx_to_label = torch.load(AV_CKPT, weights_only=False)["idx_to_label"]

    a_logits = np.asarray(c["A_only"]["logits"])
    av_logits = np.asarray(c["AV_clean_full"]["logits"])
    a_pred = a_logits.argmax(1); av_pred = av_logits.argmax(1)
    a_correct = (a_pred == labels).astype(np.int64)
    av_correct = (av_pred == labels).astype(np.int64)
    a_logp = _logp(a_logits, labels)
    assert abs(a_correct.mean() - 0.926964) < 5e-4, a_correct.mean()
    assert abs(av_correct.mean() - 0.956712) < 5e-4, av_correct.mean()

    targets = {
        "delta_flip": (av_correct - a_correct).astype(np.int64),
        "delta_logconf": _logp(av_logits, labels) - a_logp,
        "delta_margin": _margin(av_logits) - _margin(a_logits),
    }
    X, names, visemes, words = _build_features(labels, idx_to_label)
    wl_i = names.index("word_len")

    def _est(t):
        return (RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=NJOBS)
                if t == "delta_flip" else
                RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=NJOBS))

    rows = []

    # 1) per-sample baseline RF (self-check) -------------------------------------
    print("[1] per-sample baseline RF importances")
    base_imp = {}
    for t, y in targets.items():
        est = _est(t).fit(X, y)
        imp = dict(zip(names, est.feature_importances_))
        base_imp[t] = imp
        print(f"    {t:13s} word_len={imp['word_len']:.6f} (pub {PUB_WL[t]:.6f})")
        assert abs(imp["word_len"] - PUB_WL[t]) < 0.02, f"baseline RF {t} not reproduced"
        for nm in names:
            rows.append(("rf_persample_baseline", nm, t, f"{imp[nm]:.6f}", "", "", "", ""))

    # 2) per-sample RF + a_correct + a_logp covariates ---------------------------
    print("[2] per-sample RF + (a_correct, a_logp) covariates")
    Xc = np.concatenate([X, a_correct[:, None].astype(float), a_logp[:, None]], axis=1)
    names_c = names + ["A_correct", "A_logp"]
    cov_est = {}
    for t, y in targets.items():
        est = _est(t).fit(Xc, y)
        cov_est[t] = est
        imp = dict(zip(names_c, est.feature_importances_))
        drop = 100 * (1 - imp["word_len"] / base_imp[t]["word_len"])
        print(f"    {t:13s} word_len={imp['word_len']:.4f} (-{drop:.0f}%)  "
              f"A_correct={imp['A_correct']:.4f}  A_logp={imp['A_logp']:.4f}")
        for nm in names_c:
            rows.append(("rf_persample_with_Abase", nm, t, f"{imp[nm]:.6f}", "", "", "", ""))

    # 3) GROUP-AWARE permutation importance (GroupKFold by word) -----------------
    print("[3] group-aware permutation importance (+covariates, GroupKFold by word)")
    gkf = GroupKFold(n_splits=5)
    for t, y in targets.items():
        fold_imp = []
        for tr, te in gkf.split(Xc, y, groups=labels):
            est = _est(t).fit(Xc[tr], y[tr])
            r = permutation_importance(est, Xc[te], y[te], n_repeats=10,
                                       random_state=0, n_jobs=NJOBS)
            fold_imp.append(r.importances_mean)
        pm = np.mean(fold_imp, axis=0); ps = np.std(fold_imp, axis=0)
        order = np.argsort(-pm)[:4]
        print(f"    {t:13s} top: " +
              ", ".join(f"{names_c[i]}={pm[i]:.4f}" for i in order))
        for i, nm in enumerate(names_c):
            rows.append(("perm_persample_with_Abase", nm, t, f"{pm[i]:.6f}",
                         f"{ps[i]:.6f}", "", "", "permutation_importance test-fold"))

    # 4) per-WORD ridge: baseline-controlled rescue ~ features -------------------
    print("[4] per-word ridge (rescue ~ features) with/without A-baseline")
    uw = np.unique(labels)
    aw = np.array([a_correct[labels == w].mean() for w in uw])      # per-word A baseline
    rescue_w = np.array([av_correct[labels == w].mean() for w in uw]) - aw
    wl_w = np.array([len(idx_to_label[int(w)]) for w in uw], dtype=float)
    nv_w = np.array([sum(ch.lower() in "aeiou" for ch in idx_to_label[int(w)]) for w in uw],
                    dtype=float)
    vi_w = np.array([1.0 if idx_to_label[int(w)][0].lower() in "aeiou" else 0.0 for w in uw])
    vc_w = [_viseme(idx_to_label[int(w)]) for w in uw]
    vcl = sorted(set(vc_w))
    vis_oh = np.stack([[1.0 if x == o else 0.0 for x in vc_w] for o in vcl], axis=1)
    feat_w = np.concatenate([wl_w[:, None], nv_w[:, None], vi_w[:, None], vis_oh], axis=1)
    names_w = ["word_len", "n_vowels", "vowel_initial"] + [f"viseme_{o}" for o in vcl]

    def _ridge_betas(Xw, yw, nm):
        Xs = StandardScaler().fit_transform(Xw)
        ys = (yw - yw.mean()) / yw.std()
        coef = Ridge(alpha=1.0, random_state=0).fit(Xs, ys).coef_
        return dict(zip(nm, coef))

    b_no = _ridge_betas(feat_w, rescue_w, names_w)
    b_yes = _ridge_betas(np.concatenate([aw[:, None], feat_w], axis=1), rescue_w,
                         ["A_baseline"] + names_w)
    print(f"    no-baseline:  word_len={b_no['word_len']:+.3f}  "
          f"max_viseme={max(abs(v) for k, v in b_no.items() if k.startswith('viseme')):.3f}")
    print(f"    +baseline:    word_len={b_yes['word_len']:+.3f}  A_baseline={b_yes['A_baseline']:+.3f}  "
          f"max_viseme={max(abs(v) for k, v in b_yes.items() if k.startswith('viseme')):.3f}")
    for nm in names_w:
        rows.append(("perword_ridge_no_Abase", nm, "rescue", f"{b_no[nm]:.6f}", "", "", "", "std beta"))
    for nm in ["A_baseline"] + names_w:
        rows.append(("perword_ridge_with_Abase", nm, "rescue", f"{b_yes[nm]:.6f}", "", "", "", "std beta"))

    # 5) paired bootstrap 95% CIs on per-bin AV-A deltas -------------------------
    print("[5] bootstrap 95% CIs on per-bin AV-minus-A deltas")
    rng = np.random.default_rng(0)
    B = 10000
    word_of = np.array([idx_to_label[int(l)] for l in labels])
    cats = {"onset": get_onset, "viseme": _viseme, "syllable": get_syllable_group,
            "length": get_length_group, "vowel": get_vowel_group}
    selfcheck_fail = []
    for cat, fn in cats.items():
        binlab = np.array([fn(w) for w in word_of])
        for b in sorted(set(binlab)):
            m = binlab == b
            n = int(m.sum())
            a_acc = float(a_correct[m].mean()); av_acc = float(av_correct[m].mean())
            delta = av_acc - a_acc
            # paired bootstrap: resample sample indices within the bin
            av_b = av_correct[m]; a_b = a_correct[m]
            idx = rng.integers(0, n, size=(B, n), dtype=np.int32)
            db = av_b[idx].mean(1) - a_b[idx].mean(1)
            lo, hi = np.percentile(db, [2.5, 97.5])
            excl0 = "yes" if (lo > 0 or hi < 0) else "no"
            rows.append((f"bin_ci_{cat}", b, "delta_AV_minus_A", f"{delta:.6f}",
                         f"{lo:.6f}", f"{hi:.6f}", str(n),
                         f"A={a_acc:.4f} AV={av_acc:.4f} excl0={excl0}"))
            # self-check vs published anchors
            if (cat, b) in PUB_BINS:
                pn, pd = PUB_BINS[(cat, b)]
                if n != pn or abs(delta - pd) > 6e-4:
                    selfcheck_fail.append((cat, b, n, pn, round(delta, 4), pd))
            del idx, db
    print(f"    bin self-check failures: {selfcheck_fail if selfcheck_fail else 'NONE'}")
    assert not selfcheck_fail, f"bin reproduction mismatch: {selfcheck_fail}"

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "feature_or_bin", "target_or_metric", "value",
                    "ci_lo", "ci_hi", "n", "note"])
        w.writerows(rows)
    print(f"\nwrote {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
