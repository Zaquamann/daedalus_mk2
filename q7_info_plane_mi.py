#!/usr/bin/env python3
"""Q7 — non-degenerate information-plane MI, replacing the binned estimator
(phase_f_flow.py:188).

The old estimator bins each layer's activations into 8 PCA dims x 16 quantile bins
=> ~16^8 codes for 5244 samples => almost every sample gets a unique code => X
trivially determines Y in-sample => I(X;Y) saturates at H(Y)=5.028 nats at EVERY
site (degenerate). We replace it with two principled estimators that are bounded
above by H(Y) and so cannot re-pin:

  infonce_*  : variational / InfoNCE-MINE lower bound for discrete Y,
               I(X;Y) >= H(Y) - CE_heldout, where CE is the held-out cross-entropy
               of a critic q(y|x) (Barber-Agakov / InfoNCE form for classification).
               We report both a neural MLP critic (primary, 'infonce_mlp') and a
               linear critic ('infonce_linear', a more conservative lower bound).
  ksg        : Ross (2014) k-NN joint MI estimator for continuous-vector X and
               discrete Y (the same estimator sklearn.mutual_info_classif uses
               per-feature, extended to the joint vector), on PCA-reduced X.

5 AV sites on processed/deepdive_act_cache.pt (NO forward). Overwrites
analysis/deepdive/D5_info_plane.csv with an 'estimator' column. Also re-runs the
OLD binned estimator on each site for explicit contrast (estimator='binned_pca8_OLD').

PRE-REGISTERED KILL: if any infonce_*/ksg site estimate >= H(Y) - 0.05 nats it has
re-pinned (degenerate) -> exit nonzero. Expect all < 5.028 and ordered
a_mid < gate_out < block2 for the primary estimator.
"""
import csv
import os
import sys

import numpy as np
import torch
from scipy.special import digamma
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
OUT = os.path.join(SCRIPT_DIR, "analysis", "deepdive", "D5_info_plane.csv")
PIN = "03c5a87a"
SITES = ["a_mid_gap", "v_mid_gap", "gate_out_gap", "block2_gap", "penult"]


def _entropy_nats(y):
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum())


def _ce_lower_bound(X, y, HY, n_classes, critic, seed=0):
    """I(X;Y) >= H(Y) - heldout_CE for critic q(y|x). Returns (MI, CE).

    CE is computed manually in nats with a probability floor, and the per-fold
    probabilities are re-indexed into the global class space — robust to a class
    being absent from a given train fold (its floored prob -> finite loss, never a
    crash)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_tot, ce_sum = 0, 0.0
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = critic()
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)                      # (n_te, len(classes_))
        full = np.full((len(te), n_classes), 1e-12)
        full[:, clf.classes_] = proba
        p_true = full[np.arange(len(te)), y[te]]
        ce_sum += float(-np.log(np.clip(p_true, 1e-12, 1.0)).sum())  # natural log => nats
        n_tot += len(te)
    ce = ce_sum / n_tot
    return max(HY - ce, 0.0), ce


def _ksg_mi_cd(X, y, k=3, pca_dim=16, seed=0):
    """Ross 2014 k-NN MI for continuous vector X and discrete y, on PCA-reduced X."""
    d = min(pca_dim, X.shape[1])
    Xp = PCA(n_components=d, random_state=seed).fit_transform(X).astype(np.float64)
    N = len(Xp)
    classes, y_idx = np.unique(y, return_inverse=True)
    Nx = np.empty(N)
    d_k = np.empty(N)
    for ci in range(len(classes)):
        idx = np.where(y_idx == ci)[0]
        nc = len(idx)
        Nx[idx] = nc
        kk = min(k, nc - 1)
        if kk < 1:
            d_k[idx] = 0.0
            continue
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xp[idx])
        dist, _ = nn.kneighbors(Xp[idx])
        d_k[idx] = dist[:, kk]
    nn_full = NearestNeighbors().fit(Xp)
    m = np.empty(N)
    for i in range(N):
        if d_k[i] <= 0:
            m[i] = 0
            continue
        ind = nn_full.radius_neighbors(Xp[i:i + 1], radius=d_k[i] - 1e-12,
                                       return_distance=False)[0]
        m[i] = max(len(ind) - 1, 0)
    mi = (digamma(N) + digamma(k)
          - np.mean(digamma(Nx)) - np.mean(digamma(m + 1.0)))
    return max(float(mi), 0.0)


def _binned_old(X, y, n_bins=16):
    """The degenerate estimator being replaced (phase_f_flow.py:188), for contrast."""
    Xp = PCA(n_components=min(8, X.shape[1]), random_state=0).fit_transform(X)
    bins = []
    for c in range(Xp.shape[1]):
        edges = np.quantile(Xp[:, c], np.linspace(0, 1, n_bins + 1))
        bins.append(np.clip(np.digitize(Xp[:, c], edges[1:-1]), 0, n_bins - 1))
    bins = np.stack(bins, axis=1)
    code = np.zeros(len(Xp), dtype=np.int64)
    for c in range(bins.shape[1]):
        code = code * n_bins + bins[:, c]
    uc, ci = np.unique(code, return_inverse=True)
    uy, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(uc), len(uy)))
    for i in range(len(code)):
        joint[ci[i], yi[i]] += 1
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True); py = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py + 1e-12)
        lr = np.where(ratio > 0, np.log(ratio + 1e-12), 0.0)
    return float((joint * lr).sum())


def main():
    c = torch.load(CACHE, weights_only=False)
    assert c.get("val_idx_sha256", "").startswith(PIN), "VAL PIN MISMATCH"
    av = c["AV_clean_full"]
    y = np.asarray(c["labels"]).astype(np.int64)
    n_classes = int(y.max()) + 1
    HY = _entropy_nats(y)
    print(f"[H(Y)] empirical label entropy = {HY:.4f} nats (doc 5.028, "
          f"n_classes={n_classes}, log180={np.log(180):.4f})")
    assert abs(HY - 5.028) < 0.05, f"H(Y) mismatch {HY}"

    def _mlp():
        return MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                             alpha=1e-3, max_iter=300, early_stopping=True,
                             random_state=0)

    def _logit():
        return LogisticRegression(max_iter=2000, C=1.0)

    rows = [("_reference", "H(Y)", HY, "empirical label entropy (nats)")]
    results = {}  # estimator -> {site: mi}
    for est in ("infonce_mlp", "infonce_linear", "ksg", "binned_pca8_OLD"):
        results[est] = {}
    print(f"\n{'site':>14s} | {'infonce_mlp':>11s} {'infonce_lin':>11s} "
          f"{'ksg':>7s} {'binned_OLD':>10s}")
    for s in SITES:
        X = np.asarray(av[s])
        mi_mlp, ce_mlp = _ce_lower_bound(X, y, HY, n_classes, _mlp)
        mi_lin, ce_lin = _ce_lower_bound(X, y, HY, n_classes, _logit)
        mi_ksg = _ksg_mi_cd(X, y)
        mi_old = _binned_old(X, y)
        results["infonce_mlp"][s] = mi_mlp
        results["infonce_linear"][s] = mi_lin
        results["ksg"][s] = mi_ksg
        results["binned_pca8_OLD"][s] = mi_old
        print(f"{s:>14s} | {mi_mlp:11.4f} {mi_lin:11.4f} {mi_ksg:7.4f} {mi_old:10.4f}")
        rows += [
            (s, "infonce_mlp", mi_mlp, f"H(Y)-heldout_CE, MLP critic (CE={ce_mlp:.4f})"),
            (s, "infonce_linear", mi_lin, f"H(Y)-heldout_CE, linear critic (CE={ce_lin:.4f})"),
            (s, "ksg", mi_ksg, "Ross k-NN joint MI, PCA-16, k=3"),
            (s, "binned_pca8_OLD", mi_old, "degenerate estimator being replaced"),
        ]

    # KILL check: principled estimators must NOT re-pin near H(Y).
    repinned = []
    for est in ("infonce_mlp", "infonce_linear", "ksg"):
        for s in SITES:
            if results[est][s] >= HY - 0.05:
                repinned.append((est, s, results[est][s]))

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["site", "estimator", "I_layer_label_nats", "note"])
        for s, est, mi, note in rows:
            w.writerow([s, est, f"{mi:.6f}", note])
    print(f"\nwrote {OUT}")

    # ordering check on primary estimator
    p = results["infonce_mlp"]
    order_ok = p["a_mid_gap"] < p["gate_out_gap"] < p["block2_gap"]
    print(f"[ordering] infonce_mlp a_mid({p['a_mid_gap']:.3f}) < "
          f"gate_out({p['gate_out_gap']:.3f}) < block2({p['block2_gap']:.3f}) : {order_ok}")
    print(f"[contrast] binned_OLD a_mid={results['binned_pca8_OLD']['a_mid_gap']:.4f} "
          f"(degenerate ~H(Y))")

    if repinned:
        print(f"[KILL] estimator(s) re-pinned at H(Y): {repinned}")
        sys.exit(2)
    print("[OK] no principled estimator re-pinned at 5.028")
    print("DONE")


if __name__ == "__main__":
    main()
