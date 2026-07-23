#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of Q12 (integration drivers, A-baseline
controlled + per-bin bootstrap CIs). Q12_drivers_controlled.csv.

My own forward for A+AV LOGITS (deepdive_act_cache NOT loaded); reimplement all
five sections (q12_integration_drivers_controlled.py / phase_e_geometry NOT
imported). Allowed reuse: the label->category taxonomers (viseme_class, get_onset,
get_syllable_group, get_length_group, get_vowel_group).

Lead's load-bearing subset:
  baseline RF delta_flip word_len 0.507734 (== D2.3, == my verified Q5 baseline)
  +A-baseline:  word_len 0.507734->0.048851 ; A_logp 0.597530 dominant
  GroupKFold-by-word permutation importance: word_len -0.005396 (~0/neg);
       A_correct 0.069323 / A_logp 0.018844 positive
  per-bin bootstrap 95% CIs: small bins (/b/ n=115, glottal_h n=161, 4+ syll n=115)
       INCLUDE 0 ; larger bins EXCLUDE 0
Bootstrap is bit-reproducible via the documented seed np.random.default_rng(0),
B=10000, cats in insertion order, bins sorted.

Self-check: A 0.926964 / AV 0.956712. fp32, no autocast. n_jobs=2 (good GPU neighbour).

Run on dev-codex:
    python validator_indep_q12.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q12.csv
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
REF = {"A": 0.926964, "AV": 0.956712}
NJOBS = 2
# load-bearing subset to assert (section, feature, target) -> report value
REPORT = {
    ("rf_persample_baseline", "word_len", "delta_flip"): 0.507734,
    ("rf_persample_baseline", "n_vowels", "delta_flip"): 0.231139,
    ("rf_persample_baseline", "viseme_lingual", "delta_flip"): 0.061253,
    ("rf_persample_baseline", "word_len", "delta_logconf"): 0.415869,
    ("rf_persample_baseline", "word_len", "delta_margin"): 0.324457,
    ("rf_persample_with_Abase", "word_len", "delta_flip"): 0.048851,
    ("rf_persample_with_Abase", "A_logp", "delta_flip"): 0.597530,
    ("rf_persample_with_Abase", "A_correct", "delta_flip"): 0.296220,
    ("rf_persample_with_Abase", "n_vowels", "delta_flip"): 0.025223,
    ("rf_persample_with_Abase", "word_len", "delta_logconf"): 0.077866,
    ("rf_persample_with_Abase", "A_logp", "delta_logconf"): 0.811817,
    ("rf_persample_with_Abase", "word_len", "delta_margin"): 0.103228,
    ("rf_persample_with_Abase", "A_logp", "delta_margin"): 0.703198,
    ("perm_persample_with_Abase", "word_len", "delta_flip"): -0.005396,
    ("perm_persample_with_Abase", "A_correct", "delta_flip"): 0.069323,
    ("perm_persample_with_Abase", "A_logp", "delta_flip"): 0.018844,
    ("perword_ridge_no_Abase", "word_len", "rescue"): 0.188265,
    ("perword_ridge_no_Abase", "n_vowels", "rescue"): -0.187485,
    ("perword_ridge_with_Abase", "A_baseline", "rescue"): -0.681227,
    ("perword_ridge_with_Abase", "word_len", "rescue"): 0.259026,
}
# per-bin: (cat,bin) -> (delta, lo, hi, n, excl0)
REPORT_BIN = {
    ("onset", "/b/"): (0.043478, -0.008696, 0.095652, 115, "no"),
    ("viseme", "glottal_h"): (0.024845, -0.006211, 0.062112, 161, "no"),
    ("viseme", "lingual"): (0.026672, 0.016622, 0.037109, 2587, "yes"),
    ("viseme", "bilabial_bpm"): (0.030888, 0.012870, 0.048906, 777, "yes"),
    ("syllable", "4+"): (0.008696, 0.000000, 0.026087, 115, "no"),
    ("syllable", "2"): (0.035569, 0.023882, 0.047256, 1968, "yes"),
    ("syllable", "1"): (0.026984, 0.016270, 0.037698, 2520, "yes"),
    ("length", "Medium (5-7)"): (0.036201, 0.026405, 0.046422, 2348, "yes"),
    ("vowel", "/ɔː/"): (0.038835, 0.012136, 0.065534, 412, "yes"),
}


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _logp(logits, lab):
    x = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(x).sum(axis=1, keepdims=True))
    return (x - lse)[np.arange(len(lab)), lab]


def _margin(logits):
    top2 = np.sort(logits, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--t-stride", type=int, default=2)
    ap.add_argument("--expect-sha", default=EXPECT_SHA)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    sys.path.insert(0, args.root)
    from train import WordResNet
    from model_av import AVWordResNet
    from analyze_av_phonetics import viseme_class as _viseme
    from analyze_phoneme_accuracy import (get_length_group, get_onset,
                                          get_syllable_group, get_vowel_group)

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print("[FATAL] val sha != expected; STOP."); sys.exit(2)

    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels_np = dav["spectrograms"]
    mels_np = mels_np.numpy() if hasattr(mels_np, "numpy") else np.asarray(mels_np)
    labels_all = np.asarray(dav["labels"]).astype(np.int64)
    n_all = len(labels_all)
    T_FRAMES, H, W = dav["video_shape"]
    cache_path = dav.get("video_cache_path")
    if not cache_path or not os.path.exists(cache_path):
        cache_path = os.path.join(args.root, "data", "visual", "cache",
                                  dav.get("video_cache_name", "videos_88_100.uint8"))
    videos = np.memmap(cache_path, dtype=np.uint8, mode="r", shape=(n_all, T_FRAMES, H, W))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = os.path.join(args.root, "models")

    def _load(cls, name):
        ck = torch.load(os.path.join(mdir, name), weights_only=False, map_location="cpu")
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        return m.to(device).eval(), ck

    A, _ = _load(WordResNet, "audio_only_filtered.pt")
    AV, av_ck = _load(AVWordResNet, "av_fused.pt")
    idx_to_label = av_ck["idx_to_label"]
    stride = max(1, int(args.t_stride))

    class Vw(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k])
            mel = torch.from_numpy(mels_np[g]).unsqueeze(0)
            v = np.array(videos[g])
            if stride > 1: v = v[::stride]
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, vid, int(labels_all[g])

    dl = DataLoader(Vw(), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    la_all, lav_all, ys = [], [], []
    print("[fwd] A + AV logits over val ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            la_all.append(A(mel).float().cpu().numpy())
            a_mid = AV.audio_block1(mel); v_mid = AV.visual(vid)
            lf = AV.fc(AV.dropout(AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)))
            lav_all.append(lf.float().cpu().numpy())
            ys.append(y.numpy())
    labels = np.concatenate(ys)
    a_logits = np.concatenate(la_all); av_logits = np.concatenate(lav_all)
    a_pred = a_logits.argmax(1); av_pred = av_logits.argmax(1)
    a_correct = (a_pred == labels).astype(np.int64)
    av_correct = (av_pred == labels).astype(np.int64)
    accA = float(a_correct.mean()); accAV = float(av_correct.mean())
    print(f"[self-check] A={accA:.6f} (ref {REF['A']}) | AV={accAV:.6f} (ref {REF['AV']})", flush=True)
    a_logp = _logp(a_logits, labels)

    targets = {
        "delta_flip": (av_correct - a_correct).astype(np.int64),
        "delta_logconf": _logp(av_logits, labels) - a_logp,
        "delta_margin": _margin(av_logits) - _margin(a_logits),
    }

    # features (== Q5 / D2.3)
    visemes = [_viseme(idx_to_label[int(l)]) for l in labels]
    words = [idx_to_label[int(l)] for l in labels]
    viseme_classes = sorted(set(visemes))
    names, cols = [], []
    for o in viseme_classes:
        cols.append(np.asarray([1.0 if x == o else 0.0 for x in visemes])); names.append(f"viseme_{o}")
    word_len = np.asarray([len(w) for w in words], dtype=np.float32); cols.append(word_len); names.append("word_len")
    n_vowels = np.asarray([sum(1 for c in w if c.lower() in "aeiou") for w in words], dtype=np.float32)
    cols.append(n_vowels); names.append("n_vowels")
    vowel_init = np.asarray([1.0 if w and w[0].lower() in "aeiou" else 0.0 for w in words])
    cols.append(vowel_init); names.append("vowel_initial")
    X = np.stack(cols, axis=1)

    def _est(t):
        return (RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=NJOBS)
                if t == "delta_flip" else
                RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=NJOBS))

    rows, flags = [], []

    def _chk(section, feat, target, mine, tol=0.02):
        rep = REPORT.get((section, feat, target))
        d = "" if rep is None else f"{mine-rep:+.6f}"
        tag = ""
        if rep is not None and abs(mine - rep) > tol:
            tag = f"  ** FLAG >|{tol}|"; flags.append((section, feat, target, mine, rep))
        if rep is not None:
            print(f"  [{section}/{target}] {feat}: mine={mine:.6f} report={rep} d={d}{tag}", flush=True)
        rows.append([section, feat, target, f"{mine:.6f}", "", "", "", "" if rep is None else f"rep={rep}"])
        return tag

    # 1) baseline RF
    print("\n[1] per-sample baseline RF", flush=True)
    base_imp = {}
    for t, yv in targets.items():
        est = _est(t).fit(X, yv)
        imp = dict(zip(names, est.feature_importances_)); base_imp[t] = imp
        for nm in names:
            _chk("rf_persample_baseline", nm, t, imp[nm])

    # 2) + A_correct + A_logp
    print("\n[2] per-sample RF + (A_correct, A_logp)", flush=True)
    Xc = np.concatenate([X, a_correct[:, None].astype(float), a_logp[:, None]], axis=1)
    names_c = names + ["A_correct", "A_logp"]
    for t, yv in targets.items():
        est = _est(t).fit(Xc, yv)
        imp = dict(zip(names_c, est.feature_importances_))
        for nm in names_c:
            _chk("rf_persample_with_Abase", nm, t, imp[nm])

    # 3) GroupKFold-by-word permutation importance
    print("\n[3] group-aware permutation importance (GroupKFold by word)", flush=True)
    gkf = GroupKFold(n_splits=5)
    for t, yv in targets.items():
        fold_imp = []
        for tr, te in gkf.split(Xc, yv, groups=labels):
            est = _est(t).fit(Xc[tr], yv[tr])
            r = permutation_importance(est, Xc[te], yv[te], n_repeats=10,
                                       random_state=0, n_jobs=NJOBS)
            fold_imp.append(r.importances_mean)
        pm = np.mean(fold_imp, axis=0); ps = np.std(fold_imp, axis=0)
        for i, nm in enumerate(names_c):
            rep = REPORT.get(("perm_persample_with_Abase", nm, t))
            tag = ""
            if rep is not None and abs(pm[i] - rep) > 0.02:
                tag = "  ** FLAG >|0.02|"; flags.append(("perm", nm, t, float(pm[i]), rep))
            if rep is not None:
                print(f"  [perm/{t}] {nm}: mine={pm[i]:+.6f} report={rep} d={pm[i]-rep:+.6f}{tag}", flush=True)
            rows.append(["perm_persample_with_Abase", nm, t, f"{pm[i]:.6f}", f"{ps[i]:.6f}", "", "", "" if rep is None else f"rep={rep}"])

    # 4) per-word ridge
    print("\n[4] per-word ridge (rescue ~ features) with/without A-baseline", flush=True)
    uw = np.unique(labels)
    aw = np.array([a_correct[labels == w].mean() for w in uw])
    rescue_w = np.array([av_correct[labels == w].mean() for w in uw]) - aw
    wl_w = np.array([len(idx_to_label[int(w)]) for w in uw], dtype=float)
    nv_w = np.array([sum(ch.lower() in "aeiou" for ch in idx_to_label[int(w)]) for w in uw], dtype=float)
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
    b_yes = _ridge_betas(np.concatenate([aw[:, None], feat_w], axis=1), rescue_w, ["A_baseline"] + names_w)
    for nm in names_w:
        _chk("perword_ridge_no_Abase", nm, "rescue", b_no[nm], tol=0.01)
    for nm in ["A_baseline"] + names_w:
        _chk("perword_ridge_with_Abase", nm, "rescue", b_yes[nm], tol=0.01)

    # 5) paired bootstrap 95% CIs per bin (EXACT rng order)
    print("\n[5] bootstrap 95% CIs on per-bin AV-minus-A deltas", flush=True)
    rng = np.random.default_rng(0)
    B = 10000
    word_of = np.array([idx_to_label[int(l)] for l in labels])
    cats = {"onset": get_onset, "viseme": _viseme, "syllable": get_syllable_group,
            "length": get_length_group, "vowel": get_vowel_group}
    bin_flags = []
    for cat, fn in cats.items():
        binlab = np.array([fn(w) for w in word_of])
        for b in sorted(set(binlab)):
            m = binlab == b
            n = int(m.sum())
            a_acc = float(a_correct[m].mean()); av_acc = float(av_correct[m].mean())
            delta = av_acc - a_acc
            av_b = av_correct[m]; a_b = a_correct[m]
            idx = rng.integers(0, n, size=(B, n), dtype=np.int32)
            db = av_b[idx].mean(1) - a_b[idx].mean(1)
            lo, hi = np.percentile(db, [2.5, 97.5])
            excl0 = "yes" if (lo > 0 or hi < 0) else "no"
            rows.append([f"bin_ci_{cat}", b, "delta_AV_minus_A", f"{delta:.6f}",
                         f"{lo:.6f}", f"{hi:.6f}", str(n), f"A={a_acc:.4f} AV={av_acc:.4f} excl0={excl0}"])
            rep = REPORT_BIN.get((cat, b))
            if rep is not None:
                rd, rlo, rhi, rn, rex = rep
                ok = (n == rn and abs(delta - rd) < 6e-4 and excl0 == rex
                      and abs(lo - rlo) < 2e-3 and abs(hi - rhi) < 2e-3)
                tag = "" if ok else "  ** FLAG"
                if not ok:
                    bin_flags.append((cat, b, n, delta, lo, hi, excl0, rep))
                print(f"  [{cat}/{b}] n={n} delta={delta:.6f}(rep {rd}) "
                      f"CI[{lo:.6f},{hi:.6f}](rep[{rlo},{rhi}]) excl0={excl0}(rep {rex}){tag}", flush=True)
            del idx, db

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q12.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "feature_or_bin", "target_or_metric", "value", "ci_lo", "ci_hi", "n", "note"])
        w.writerows(rows)
    print(f"\n[out] wrote {out}", flush=True)
    allflags = flags + bin_flags
    if allflags:
        print(f"[FLAGS] {allflags}", flush=True)
    else:
        print("[PASS] load-bearing RF/perm/ridge within tol; bin deltas+CIs+excl0 reproduced.", flush=True)


if __name__ == "__main__":
    main()
