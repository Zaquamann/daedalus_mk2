#!/usr/bin/env python3
"""v2 — mechanism probe: report lbfgs n_iter_ / convergence per fold, and test
BLAS/thread sensitivity. If lbfgs hits max_iter (non-converged), the solution is
path-dependent => sensitive to solver version AND reduction order (the residual)."""
import os, hashlib, sys, warnings
import numpy as np
import sklearn, scipy
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def probe(X, y, max_iter=1500, C=1.0, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, niters, maxed = [], [], 0
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(Xtr, y[tr])
        ni = int(np.max(clf.n_iter_))
        niters.append(ni); maxed += int(ni >= max_iter)
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(accs)), niters, maxed


def main():
    X = np.load("/tmp/Xb1.npy"); y = np.load("/tmp/yb1.npy")
    nthreads = os.environ.get("OMP_NUM_THREADS", "default")
    print(f"ENV sklearn {sklearn.__version__} numpy {np.__version__} scipy {scipy.__version__} "
          f"| OMP_NUM_THREADS={nthreads}")
    print(f"X sha={hashlib.sha256(X.tobytes()).hexdigest()[:12]}")
    acc, niters, maxed = probe(X, y)
    print(f"acc={acc:.6f}  per-fold n_iter={niters}  folds_at_max_iter(1500)={maxed}/5")


if __name__ == "__main__":
    main()
