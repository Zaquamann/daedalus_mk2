#!/usr/bin/env python3
"""v3 — causal test of the mechanism: the inter-environment spread is finite-tolerance
lbfgs path noise. Sweep tol; if tighter tol collapses the version/thread spread toward
a common value, the discrepancy is proven to be reduction-order noise on a tol-ball,
not a genuine version-dependent model."""
import os, hashlib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings; from sklearn.exceptions import ConvergenceWarning


def probe(X, y, tol, max_iter=20000, C=1.0, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, niters = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C, tol=tol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(Xtr, y[tr])
        niters.append(int(np.max(clf.n_iter_)))
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(accs)), int(np.mean(niters))


def main():
    X = np.load("/tmp/Xb1.npy"); y = np.load("/tmp/yb1.npy")
    tol = float(os.environ["TOL"])
    acc, ni = probe(X, y, tol)
    print(f"sklearn {sklearn.__version__} OMP={os.environ.get('OMP_NUM_THREADS','def')} "
          f"tol={tol:.0e} -> acc={acc:.6f} mean_n_iter={ni}")


if __name__ == "__main__":
    main()
