#!/usr/bin/env python3
"""Single-variable sklearn-version test for the Q16/Q18 A-block1 word probe.
Loads a FROZEN activation array (identical bytes across envs) and runs the
verbatim phase_f_flow._probe_5fold. Only the sklearn version differs between runs.
"""
import hashlib, sys
import numpy as np
import sklearn, scipy
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def _probe_5fold(X, y, max_iter=1500, C=1.0, seed=0):
    """phase_f_flow._probe_5fold, verbatim."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, bal = [], []
    perfold = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        a = accuracy_score(y[te], pred)
        accs.append(a); bal.append(balanced_accuracy_score(y[te], pred))
        perfold.append(a)
    return float(np.mean(accs)), float(np.mean(bal)), perfold


def main():
    X = np.load("/tmp/Xb1.npy"); y = np.load("/tmp/yb1.npy")
    print(f"ENV: python {sys.version.split()[0]} | sklearn {sklearn.__version__} | "
          f"numpy {np.__version__} | scipy {scipy.__version__}")
    print(f"X sha256={hashlib.sha256(X.tobytes()).hexdigest()[:16]} shape={X.shape} "
          f"dtype={X.dtype} | y sha256={hashlib.sha256(y.tobytes()).hexdigest()[:16]}")
    # run twice -> determinism check within the env
    a1, b1, pf1 = _probe_5fold(X, y)
    a2, b2, pf2 = _probe_5fold(X, y)
    print(f"run#1 acc={a1:.6f} bal={b1:.6f} perfold={[f'{p:.6f}' for p in pf1]}")
    print(f"run#2 acc={a2:.6f} bal={b2:.6f}  (run-to-run d_acc={a1-a2:+.2e})")
    print(f"ANCHOR=0.426774  POD(1.9.0)=0.424676  | this-env acc={a1:.6f} "
          f"d_anchor={a1-0.426774:+.6f} d_pod={a1-0.424676:+.6f}")


if __name__ == "__main__":
    main()
