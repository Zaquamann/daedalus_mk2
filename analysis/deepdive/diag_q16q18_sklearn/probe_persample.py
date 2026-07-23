#!/usr/bin/env python3
"""Per-sample out-of-fold predictions for the A-block1 word probe, to prove the
dichotomy: thread-pinned 1.9.0 reproduces 1.8.0's EXACT predictions (=> reduction-order
nondeterminism, not an algorithm change). Saves the 5244-vector keyed by TAG."""
import os, hashlib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings; from sklearn.exceptions import ConvergenceWarning


def oof_preds(X, y, max_iter=1500, C=1.0, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.full(len(y), -1, dtype=np.int64)
    for tr, te in skf.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(Xtr, y[tr])
        pred[te] = clf.predict(Xte)
    return pred


def main():
    X = np.load("/tmp/Xb1.npy"); y = np.load("/tmp/yb1.npy")
    tag = os.environ["TAG"]
    p = oof_preds(X, y)
    acc = float((p == y).mean())
    np.save(f"/tmp/oof_{tag}.npy", p)
    print(f"TAG={tag} sklearn {sklearn.__version__} OMP={os.environ.get('OMP_NUM_THREADS','def')} "
          f"acc={acc:.6f} sha={hashlib.sha256(p.tobytes()).hexdigest()[:12]} saved /tmp/oof_{tag}.npy")


if __name__ == "__main__":
    main()
