#!/usr/bin/env python3
"""Q5 — is the word_len 'driver' of AV rescue a genuine stimulus feature, or just
a proxy for A-only rescue headroom?

The existing D2.3 RF (phase_e_geometry.py:123) predicts per-sample delta_flip
(av_correct - a_correct, in {-1,0,+1}) from viseme one-hots + word_len + n_vowels +
vowel_initial, and word_len dominates (importance 0.5077 on delta_flip). BUT the
A-only baseline is NOT a predictor, and a word already correct under A-only CANNOT be
positively flipped — so word_len may simply index how much room there is to rescue.

This script (CPU sklearn on the cached eager-fp32 logits):
  1. SELF-CHECK: reproduce the published D2.3 baseline RF importances (word_len 0.5077).
  2. Add the per-class A-only baseline as a covariate (leave-one-out per class, so a
     sample's own a_correct does not leak into its own headroom predictor) and refit;
     report whether word_len importance survives once headroom is in the model.
  3. Residualize word_len on the A-only baseline (word_len - OLS(word_len ~ A_base))
     and refit; report the residual word_len's importance + corr(word_len, A_base).
  4. Logistic check: standardized coef of word_len for the binary rescue target
     (delta_flip==+1) WITH vs WITHOUT the A-baseline covariate.
ONE artifact: analysis/deepdive/Q5_rescue_confound.csv.
"""
import csv
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze_av_phonetics import viseme_class as _viseme

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
OUT = os.path.join(SCRIPT_DIR, "analysis", "deepdive", "Q5_rescue_confound.csv")
AV_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused.pt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
PIN = "03c5a87a"
RF_KW = dict(n_estimators=200, random_state=0, n_jobs=-1)


def _build_base_features(labels, idx_to_label):
    """Replicate phase_e_geometry.D2_3_drivers feature matrix EXACTLY."""
    visemes = [_viseme(idx_to_label[int(l)]) for l in labels]
    words = [idx_to_label[int(l)] for l in labels]
    viseme_classes = sorted(set(visemes))
    feat_names, cols = [], []
    for o in viseme_classes:
        cols.append(np.asarray([1.0 if x == o else 0.0 for x in visemes]))
        feat_names.append(f"viseme_{o}")
    word_len = np.asarray([len(w) for w in words], dtype=np.float32)
    cols.append(word_len); feat_names.append("word_len")
    n_vowels = np.asarray([sum(1 for c in w if c.lower() in "aeiou") for w in words],
                          dtype=np.float32)
    cols.append(n_vowels); feat_names.append("n_vowels")
    vowel_init = np.asarray([1.0 if w and w[0].lower() in "aeiou" else 0.0 for w in words])
    cols.append(vowel_init); feat_names.append("vowel_initial")
    X = np.stack(cols, axis=1)
    return X, feat_names, word_len


def main():
    c = torch.load(CACHE, weights_only=False)
    assert c.get("val_idx_sha256", "").startswith(PIN), "VAL PIN MISMATCH"
    labels = np.asarray(c["labels"]).astype(np.int64)
    assert len(labels) == 5244
    idx_to_label = torch.load(AV_CKPT, weights_only=False)["idx_to_label"]
    n_classes = len(idx_to_label)

    a_pred = np.asarray(c["A_only"]["logits"]).argmax(1)
    av_pred = np.asarray(c["AV_clean_full"]["logits"]).argmax(1)
    a_correct = (a_pred == labels).astype(np.float64)
    av_correct = (av_pred == labels).astype(np.float64)
    delta_flip = (av_correct - a_correct).astype(np.int64)   # {-1,0,+1}

    X_base, feat_names, word_len = _build_base_features(labels, idx_to_label)
    wl_i = feat_names.index("word_len")

    # 1) SELF-CHECK: reproduce published baseline RF importances.
    clf0 = RandomForestClassifier(**RF_KW).fit(X_base, delta_flip)
    imp0 = dict(zip(feat_names, clf0.feature_importances_))
    print(f"[self-check] baseline RF word_len importance = {imp0['word_len']:.6f} "
          f"(published 0.507734)")
    assert abs(imp0["word_len"] - 0.507734) < 0.02, "baseline RF did not reproduce"

    # 2) per-class A-only baseline (leave-one-out) as covariate.
    class_sum = np.zeros(n_classes); class_cnt = np.zeros(n_classes)
    np.add.at(class_sum, labels, a_correct)
    np.add.at(class_cnt, labels, 1.0)
    a_base_full = class_sum[labels] / np.maximum(class_cnt[labels], 1)
    a_base_loo = (class_sum[labels] - a_correct) / np.maximum(class_cnt[labels] - 1, 1)

    X_cov = np.concatenate([X_base, a_base_loo[:, None]], axis=1)
    names_cov = feat_names + ["A_baseline_loo"]
    clf1 = RandomForestClassifier(**RF_KW).fit(X_cov, delta_flip)
    imp1 = dict(zip(names_cov, clf1.feature_importances_))
    print(f"[+A-baseline] word_len={imp1['word_len']:.6f}  "
          f"A_baseline_loo={imp1['A_baseline_loo']:.6f}  "
          f"(word_len drop {100*(1-imp1['word_len']/imp0['word_len']):.1f}%)")

    # 3) residualize word_len on A-baseline; PARTIAL correlation with rescue.
    # (RF impurity-importance on a residualized continuous word_len is inflated by
    # the high-cardinality bias — integer word_len becomes continuous — so we use
    # the cardinality-free partial correlation instead.)
    df_f = delta_flip.astype(np.float64)
    s1, i1 = np.polyfit(a_base_full, word_len, 1)
    wl_resid = word_len - (s1 * a_base_full + i1)
    s2, i2 = np.polyfit(a_base_full, df_f, 1)
    df_resid = df_f - (s2 * a_base_full + i2)
    corr_wl_ab = float(np.corrcoef(word_len, a_base_full)[0, 1])
    raw_corr = float(np.corrcoef(word_len, df_f)[0, 1])
    partial_corr = float(np.corrcoef(wl_resid, df_resid)[0, 1])
    print(f"[residualize] corr(word_len,A_base)={corr_wl_ab:+.4f}  "
          f"raw corr(word_len,delta_flip)={raw_corr:+.4f}  "
          f"partial corr|A_base={partial_corr:+.4f}")

    # 4) logistic on binary rescue (delta_flip==+1), standardized coefs.
    y_resc = (delta_flip == 1).astype(np.int64)
    def _logit_coef(feature_block, names):
        Xs = StandardScaler().fit_transform(feature_block)
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y_resc)
        return dict(zip(names, lr.coef_[0]))
    co_no = _logit_coef(np.stack([word_len, X_base[:, feat_names.index("n_vowels")]], 1),
                        ["word_len", "n_vowels"])
    co_yes = _logit_coef(np.stack([word_len, X_base[:, feat_names.index("n_vowels")],
                                   a_base_full], 1),
                         ["word_len", "n_vowels", "A_baseline"])
    print(f"[logistic rescue] word_len coef  no-Abase={co_no['word_len']:+.4f}  "
          f"with-Abase={co_yes['word_len']:+.4f}  A_base coef={co_yes['A_baseline']:+.4f}")

    survives = imp1["word_len"] >= imp1["A_baseline_loo"]
    print(f"\n[VERDICT] word_len still top driver after headroom control? {survives}")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "feature", "value", "note"])
        # RF importances: baseline vs +A-baseline
        for nm in feat_names:
            w.writerow(["rf_imp_baseline", nm, f"{imp0[nm]:.6f}", "no A-baseline covariate"])
        for nm in names_cov:
            w.writerow(["rf_imp_with_Abaseline", nm, f"{imp1[nm]:.6f}", "LOO A-baseline added"])
        # residualization (cardinality-free partial correlation)
        w.writerow(["residualize", "corr_word_len_Abaseline", f"{corr_wl_ab:.6f}", "Pearson"])
        w.writerow(["residualize", "raw_corr_word_len_deltaflip", f"{raw_corr:.6f}", "Pearson"])
        w.writerow(["residualize", "partial_corr_word_len_deltaflip_given_Abaseline",
                    f"{partial_corr:.6f}", "word_len & delta_flip both residualized on A-baseline"])
        # logistic coefs
        w.writerow(["logistic_no_Abaseline", "word_len", f"{co_no['word_len']:.6f}", "std coef, rescue=delta_flip==+1"])
        w.writerow(["logistic_no_Abaseline", "n_vowels", f"{co_no['n_vowels']:.6f}", "std coef"])
        w.writerow(["logistic_with_Abaseline", "word_len", f"{co_yes['word_len']:.6f}", "std coef"])
        w.writerow(["logistic_with_Abaseline", "n_vowels", f"{co_yes['n_vowels']:.6f}", "std coef"])
        w.writerow(["logistic_with_Abaseline", "A_baseline", f"{co_yes['A_baseline']:.6f}", "std coef"])
        w.writerow(["verdict", "word_len_survives_headroom_control", str(survives),
                    "word_len importance >= A_baseline importance after covariate"])
    print(f"wrote {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
