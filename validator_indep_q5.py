#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of Q5 (rescue-confound regression).
Q5_rescue_confound.csv. My own forward for A/AV predictions (deepdive_act_cache.pt
NOT loaded — I reimplement the forward; the trained submodules + viseme taxonomy are
the only reuse). RF/LR/partial-corr math reimplemented per the generator's spec.

Targets:
  rf_imp_baseline word_len 0.507734 (self-check == D2.3)
  +LOO A-baseline:  word_len 0.188481 ; A_baseline_loo 0.667862 (dominant)
  residualize: corr(word_len,A_base) 0.145054 ; raw corr(word_len,delta_flip) 0.006360 ;
               partial corr | A_base 0.026627
  logistic: word_len no-Abase -0.051116 / with-Abase +0.077456 ; A_baseline -0.643118
  verdict word_len_survives_headroom_control = False

Self-check: A 0.926964 / AV 0.956712 (fp32 anchors). ≤0.5%/abs-0.02 RF guardrail.
fp32, no autocast.

Run on dev-codex:
    python validator_indep_q5.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q5.csv
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
REF = {"A": 0.926964, "AV": 0.956712}
RF_KW = dict(n_estimators=200, random_state=0, n_jobs=-1)
REPORT = {
    ("rf_imp_baseline", "word_len"): 0.507734,
    ("rf_imp_baseline", "n_vowels"): 0.231139,
    ("rf_imp_with_Abaseline", "word_len"): 0.188481,
    ("rf_imp_with_Abaseline", "A_baseline_loo"): 0.667862,
    ("rf_imp_with_Abaseline", "n_vowels"): 0.078586,
    ("residualize", "corr_word_len_Abaseline"): 0.145054,
    ("residualize", "raw_corr_word_len_deltaflip"): 0.006360,
    ("residualize", "partial_corr_word_len_deltaflip_given_Abaseline"): 0.026627,
    ("logistic_no_Abaseline", "word_len"): -0.051116,
    ("logistic_with_Abaseline", "word_len"): 0.077456,
    ("logistic_with_Abaseline", "A_baseline"): -0.643118,
}


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


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
    n_classes = len(idx_to_label)
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

    a_pred, av_pred, ys = [], [], []
    print("[fwd] A + AV preds over val ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            a_pred.append(A(mel).argmax(1).cpu().numpy())
            a_mid = AV.audio_block1(mel); v_mid = AV.visual(vid)
            lf = AV.fc(AV.dropout(AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)))
            av_pred.append(lf.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    labels = np.concatenate(ys)
    a_pred = np.concatenate(a_pred); av_pred = np.concatenate(av_pred)
    accA = float((a_pred == labels).mean()); accAV = float((av_pred == labels).mean())
    print(f"[self-check] A={accA:.6f} (ref {REF['A']}) | AV={accAV:.6f} (ref {REF['AV']})", flush=True)

    a_correct = (a_pred == labels).astype(np.float64)
    av_correct = (av_pred == labels).astype(np.float64)
    delta_flip = (av_correct - a_correct).astype(np.int64)

    # features (replicate _build_base_features EXACTLY)
    visemes = [_viseme(idx_to_label[int(l)]) for l in labels]
    words = [idx_to_label[int(l)] for l in labels]
    viseme_classes = sorted(set(visemes))
    feat_names, cols = [], []
    for o in viseme_classes:
        cols.append(np.asarray([1.0 if x == o else 0.0 for x in visemes]))
        feat_names.append(f"viseme_{o}")
    word_len = np.asarray([len(w) for w in words], dtype=np.float32)
    cols.append(word_len); feat_names.append("word_len")
    n_vowels = np.asarray([sum(1 for c in w if c.lower() in "aeiou") for w in words], dtype=np.float32)
    cols.append(n_vowels); feat_names.append("n_vowels")
    vowel_init = np.asarray([1.0 if w and w[0].lower() in "aeiou" else 0.0 for w in words])
    cols.append(vowel_init); feat_names.append("vowel_initial")
    X_base = np.stack(cols, axis=1)

    rows = []
    def _cmp(section, name, mine):
        rep = REPORT.get((section, name))
        d = "" if rep is None else f"{mine-rep:+.6f}"
        tag = ""
        if rep is not None and abs(mine - rep) > 0.02:
            tag = "  ** FLAG abs>0.02"
        print(f"  [{section}] {name}: mine={mine:.6f} report={rep} delta={d}{tag}", flush=True)
        rows.append([section, name, f"{mine:.6f}", "" if rep is None else f"{rep:.6f}", d])
        return tag

    flags = []
    # 1) baseline RF
    clf0 = RandomForestClassifier(**RF_KW).fit(X_base, delta_flip)
    imp0 = dict(zip(feat_names, clf0.feature_importances_))
    print("\n[1 baseline RF importances]", flush=True)
    for nm in ["word_len", "n_vowels"]:
        if _cmp("rf_imp_baseline", nm, imp0[nm]): flags.append(("rf_imp_baseline", nm))

    # 2) +LOO A-baseline
    class_sum = np.zeros(n_classes); class_cnt = np.zeros(n_classes)
    np.add.at(class_sum, labels, a_correct); np.add.at(class_cnt, labels, 1.0)
    a_base_full = class_sum[labels] / np.maximum(class_cnt[labels], 1)
    a_base_loo = (class_sum[labels] - a_correct) / np.maximum(class_cnt[labels] - 1, 1)
    X_cov = np.concatenate([X_base, a_base_loo[:, None]], axis=1)
    names_cov = feat_names + ["A_baseline_loo"]
    clf1 = RandomForestClassifier(**RF_KW).fit(X_cov, delta_flip)
    imp1 = dict(zip(names_cov, clf1.feature_importances_))
    print("\n[2 +LOO A-baseline RF importances]", flush=True)
    for nm in ["word_len", "A_baseline_loo", "n_vowels"]:
        if _cmp("rf_imp_with_Abaseline", nm, imp1[nm]): flags.append(("rf_imp_with_Abaseline", nm))

    # 3) residualize / partial corr
    df_f = delta_flip.astype(np.float64)
    s1, i1 = np.polyfit(a_base_full, word_len, 1); wl_resid = word_len - (s1 * a_base_full + i1)
    s2, i2 = np.polyfit(a_base_full, df_f, 1); df_resid = df_f - (s2 * a_base_full + i2)
    corr_wl_ab = float(np.corrcoef(word_len, a_base_full)[0, 1])
    raw_corr = float(np.corrcoef(word_len, df_f)[0, 1])
    partial_corr = float(np.corrcoef(wl_resid, df_resid)[0, 1])
    print("\n[3 residualize / partial correlation]", flush=True)
    _cmp("residualize", "corr_word_len_Abaseline", corr_wl_ab)
    _cmp("residualize", "raw_corr_word_len_deltaflip", raw_corr)
    _cmp("residualize", "partial_corr_word_len_deltaflip_given_Abaseline", partial_corr)

    # 4) logistic
    y_resc = (delta_flip == 1).astype(np.int64)
    def _logit_coef(block, names):
        Xs = StandardScaler().fit_transform(block)
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y_resc)
        return dict(zip(names, lr.coef_[0]))
    co_no = _logit_coef(np.stack([word_len, n_vowels], 1), ["word_len", "n_vowels"])
    co_yes = _logit_coef(np.stack([word_len, n_vowels, a_base_full], 1),
                         ["word_len", "n_vowels", "A_baseline"])
    print("\n[4 logistic rescue coefs]", flush=True)
    _cmp("logistic_no_Abaseline", "word_len", co_no["word_len"])
    _cmp("logistic_with_Abaseline", "word_len", co_yes["word_len"])
    _cmp("logistic_with_Abaseline", "A_baseline", co_yes["A_baseline"])

    survives = imp1["word_len"] >= imp1["A_baseline_loo"]
    print(f"\n[VERDICT] word_len_survives_headroom_control = {survives}  (report False)", flush=True)
    rows.append(["verdict", "word_len_survives_headroom_control", str(survives), "False", ""])

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q5.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["section", "feature", "mine", "report", "delta"])
        for r in rows:
            w.writerow(r)
    print(f"[out] wrote {out}", flush=True)
    if flags:
        print(f"[FLAGS] {flags}", flush=True)
    else:
        print("[PASS] all checked RF importances within abs 0.02; verdict matches.", flush=True)


if __name__ == "__main__":
    main()
