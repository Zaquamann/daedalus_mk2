#!/usr/bin/env python3
"""Phase A — build activation cache + cross-model penultimate comparison
(D4.1 late-ensemble, D4.2 CKA, D4.3 RSA, D4.5 linear probe). Writes CSVs to
`analysis/deepdive/`. Run: `python phase_a_deepdive.py`."""

from __future__ import annotations

import csv
import hashlib
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from analyze_av_msi import (
    BATCH_SIZE, T_STRIDE, _ValAVView, _accuracy, _load_models,
)
from dataset_raw_noisy import RawNoisyAVDataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "deepdive")
CACHE_PATH = os.path.join(SCRIPT_DIR, "processed", "deepdive_act_cache.pt")
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def _extract_A_only(model, loader, device):
    """Forward A-only model, capture {block1_gap, block2_gap, penult, logits}."""
    block1_gap, block2_gap, pens, logits, labels = [], [], [], [], []
    for mel, _v, y in loader:
        x = mel.unsqueeze(1).to(device, non_blocking=True)
        b1 = model.block1(x)                   # (B, 64, 40, 50)
        b2 = model.block2(b1)                  # (B, 128, 20, 25)
        pen = model.gap(b2).flatten(1)         # (B, 128)
        lg = model.fc(model.dropout(pen))      # (B, num_classes)
        block1_gap.append(b1.mean(dim=(2, 3)).cpu().numpy())   # (B, 64)
        block2_gap.append(b2.mean(dim=(2, 3)).cpu().numpy())   # (B, 128)
        pens.append(pen.cpu().numpy())
        logits.append(lg.cpu().numpy())
        labels.append(y.numpy())
    return dict(
        block1_gap=np.concatenate(block1_gap),
        block2_gap=np.concatenate(block2_gap),
        penult=np.concatenate(pens),
        logits=np.concatenate(logits),
        labels=np.concatenate(labels),
    )


@torch.no_grad()
def _extract_V_fair(model, loader, device):
    """V-only-fair: {visual_gap, block2_gap, penult, logits}."""
    visual_gap, block2_gap, pens, logits, labels = [], [], [], [], []
    for _mel, v, y in loader:
        v = v.to(device, non_blocking=True)
        vfeat = model.visual(v)                # (B, 64, 40, 50)
        b2 = model.block2(vfeat)               # (B, 128, 20, 25)
        pen = model.gap(b2).flatten(1)         # (B, 128)
        lg = model.fc(model.dropout(pen))
        visual_gap.append(vfeat.mean(dim=(2, 3)).cpu().numpy())  # (B, 64)
        block2_gap.append(b2.mean(dim=(2, 3)).cpu().numpy())
        pens.append(pen.cpu().numpy())
        logits.append(lg.cpu().numpy())
        labels.append(y.numpy())
    return dict(
        visual_gap=np.concatenate(visual_gap),
        block2_gap=np.concatenate(block2_gap),
        penult=np.concatenate(pens),
        logits=np.concatenate(logits),
        labels=np.concatenate(labels),
    )


@torch.no_grad()
def _extract_AV(model, loader, device,
                 video_kind: str = "real", audio_kind: str = "real"):
    """AV model: {a_mid_gap, v_mid_gap, gate_out_gap, block2_gap, penult, logits}.

    Mirrors `analyze_av_msi._forward_AV`'s zero/scaled toggles.
    """
    a_mids, v_mids, gate_outs, b2s, pens, logits, labels = (
        [], [], [], [], [], [], [])
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        if audio_kind == "zero":
            mel = torch.zeros_like(mel)
        if video_kind == "zero":
            vid = torch.zeros_like(vid)

        a_mid = model.audio_block1(mel)                # (B, 64, 40, 50)
        v_mid = (torch.zeros_like(a_mid) if video_kind == "zero"
                 else model.visual(vid))                # (B, 64, 40, 50)
        a_fused = model.gate(a_mid, v_mid)              # (B, 64, 40, 50)
        b2 = model.audio_block2(a_fused)                # (B, 128, 20, 25)
        pen = model.gap(b2).flatten(1)                  # (B, 128)
        lg = model.fc(model.dropout(pen))

        a_mids.append(a_mid.mean(dim=(2, 3)).cpu().numpy())     # (B, 64)
        v_mids.append(v_mid.mean(dim=(2, 3)).cpu().numpy())     # (B, 64)
        gate_outs.append(a_fused.mean(dim=(2, 3)).cpu().numpy())# (B, 64)
        b2s.append(b2.mean(dim=(2, 3)).cpu().numpy())           # (B, 128)
        pens.append(pen.cpu().numpy())
        logits.append(lg.cpu().numpy())
        labels.append(y.numpy())
    return dict(
        a_mid_gap=np.concatenate(a_mids),
        v_mid_gap=np.concatenate(v_mids),
        gate_out_gap=np.concatenate(gate_outs),
        block2_gap=np.concatenate(b2s),
        penult=np.concatenate(pens),
        logits=np.concatenate(logits),
        labels=np.concatenate(labels),
    )


# Linear CKA (Kornblith 2019). On centered features.

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (N, d) feature matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # Frobenius / HSIC formulation
    num = (X.T @ Y).reshape(-1)
    num = float(np.dot(num, num))
    den_x = float(np.linalg.norm(X.T @ X, "fro"))
    den_y = float(np.linalg.norm(Y.T @ Y, "fro"))
    if den_x * den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


# Per-class RDM (180×180) for RSA

def _class_mean(feats: np.ndarray, labels: np.ndarray,
                n_classes: int) -> np.ndarray:
    means = np.zeros((n_classes, feats.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for f, l in zip(feats, labels):
        means[int(l)] += f
        counts[int(l)] += 1
    nz = counts > 0
    means[nz] = means[nz] / counts[nz, None]
    return means.astype(np.float32)


def _class_rdm(feats: np.ndarray, labels: np.ndarray, n_classes: int):
    means = _class_mean(feats, labels, n_classes)
    # Cosine distance pdist → condensed (n_classes*(n_classes-1)/2,)
    rdm = pdist(means, metric="cosine")
    return rdm


# Linear probe with 5-fold CV on val_idx penultimates

def _linprobe_5fold(feats: np.ndarray, labels: np.ndarray,
                     C: float = 1.0, seed: int = 0,
                     max_iter: int = 2000) -> dict:
    """5-fold stratified CV linear-probe accuracy and balanced accuracy."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, bal_accs = [], []
    for fold, (tr, te) in enumerate(skf.split(feats, labels)):
        clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=-1)
        clf.fit(feats[tr], labels[tr])
        pred = clf.predict(feats[te])
        accs.append(float(accuracy_score(labels[te], pred)))
        bal_accs.append(float(balanced_accuracy_score(labels[te], pred)))
    return dict(
        acc_mean=float(np.mean(accs)),
        acc_std=float(np.std(accs)),
        bal_acc_mean=float(np.mean(bal_accs)),
        bal_acc_std=float(np.std(bal_accs)),
        per_fold_acc=accs,
    )


# D4.1 — Late ensemble

def D4_1_late_ensemble(A, V, AV, labels) -> dict:
    """Softmax-average A + V; compare to AV-fused."""

    def _softmax(x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    p_a = _softmax(A["logits"])
    p_v = _softmax(V["logits"])
    p_av = _softmax(AV["logits"])
    p_ens = 0.5 * (p_a + p_v)

    acc_a = float((p_a.argmax(1) == labels).mean())
    acc_v = float((p_v.argmax(1) == labels).mean())
    acc_av = float((p_av.argmax(1) == labels).mean())
    acc_ens = float((p_ens.argmax(1) == labels).mean())

    # Try a few mixing weights for completeness.
    extras = []
    for w in (0.25, 0.5, 0.75):
        p = w * p_a + (1 - w) * p_v
        extras.append((w, float((p.argmax(1) == labels).mean())))

    print("\n  D4.1 — Late ensemble vs AV-fused:")
    print(f"    A-only acc      = {acc_a:.4%}")
    print(f"    V-fair acc      = {acc_v:.4%}")
    print(f"    AV-fused acc    = {acc_av:.4%}")
    print(f"    50/50 ensemble  = {acc_ens:.4%}")
    for w, a in extras:
        print(f"    {int(w*100)}/{int((1-w)*100):>2d} ensemble = {a:.4%}")
    print(f"    AV − 50/50 gap  = {(acc_av - acc_ens)*100:+.2f} pp")

    out_csv = os.path.join(OUT_DIR, "D4_late_ensemble.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["regime", "acc"])
        w.writerow(["A_only_alone", f"{acc_a:.6f}"])
        w.writerow(["V_fair_alone", f"{acc_v:.6f}"])
        w.writerow(["AV_fused", f"{acc_av:.6f}"])
        w.writerow(["ensemble_50_50", f"{acc_ens:.6f}"])
        for w_, a_ in extras:
            w.writerow([f"ensemble_{int(w_*100):02d}_{int((1-w_)*100):02d}",
                        f"{a_:.6f}"])
        w.writerow(["AV_minus_5050_pp", f"{(acc_av - acc_ens)*100:.4f}"])
    print(f"  wrote {out_csv}")
    return dict(
        acc_a=acc_a, acc_v=acc_v, acc_av=acc_av, acc_ens=acc_ens)


# D4.2 — Linear CKA layer-by-layer

def D4_2_cka(A, V, AV_full) -> None:
    """CKA between {A_only layers} × {AV layers}."""
    A_layers = [("block1_gap", A["block1_gap"]),
                ("block2_gap", A["block2_gap"]),
                ("penult",     A["penult"])]
    AV_layers = [("a_mid_gap",    AV_full["a_mid_gap"]),
                 ("gate_out_gap", AV_full["gate_out_gap"]),
                 ("block2_gap",   AV_full["block2_gap"]),
                 ("penult",       AV_full["penult"])]
    V_layers = [("visual_gap", V["visual_gap"]),
                ("block2_gap", V["block2_gap"]),
                ("penult",     V["penult"])]

    print("\n  D4.2 — Linear CKA matrices:")
    out_csv = os.path.join(OUT_DIR, "D4_cka_matrix.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "A_layer", "B_layer", "linear_CKA"])
        # A × AV
        print("    A_only × AV:")
        for an, af in A_layers:
            row_vals = []
            for bn, bf in AV_layers:
                v = _linear_cka(af, bf)
                w.writerow(["A_x_AV", an, bn, f"{v:.6f}"])
                row_vals.append(f"{v:.3f}")
            print(f"      {an:>12s}: " + " | ".join(
                f"{n}={vv}" for (n, _), vv in zip(AV_layers, row_vals)))
        # V × AV
        print("    V_fair × AV:")
        for an, af in V_layers:
            row_vals = []
            for bn, bf in AV_layers:
                v = _linear_cka(af, bf)
                w.writerow(["V_x_AV", an, bn, f"{v:.6f}"])
                row_vals.append(f"{v:.3f}")
            print(f"      {an:>12s}: " + " | ".join(
                f"{n}={vv}" for (n, _), vv in zip(AV_layers, row_vals)))
        # A × V (sanity)
        print("    A_only × V_fair:")
        for an, af in A_layers:
            row_vals = []
            for bn, bf in V_layers:
                v = _linear_cka(af, bf)
                w.writerow(["A_x_V", an, bn, f"{v:.6f}"])
                row_vals.append(f"{v:.3f}")
            print(f"      {an:>12s}: " + " | ".join(
                f"{n}={vv}" for (n, _), vv in zip(V_layers, row_vals)))
    print(f"  wrote {out_csv}")


# D4.3 — RSA per-class

def D4_3_rsa(A, V, AV_full, labels, n_classes: int) -> None:
    print("\n  D4.3 — RSA on per-class RDMs (cosine):")
    rdm_a = _class_rdm(A["penult"], labels, n_classes)
    rdm_v = _class_rdm(V["penult"], labels, n_classes)
    rdm_av = _class_rdm(AV_full["penult"], labels, n_classes)
    r_av_a, _ = spearmanr(rdm_av, rdm_a)
    r_av_v, _ = spearmanr(rdm_av, rdm_v)
    r_a_v, _ = spearmanr(rdm_a, rdm_v)
    print(f"    AV vs A    Spearman ρ = {r_av_a:.4f}")
    print(f"    AV vs V    Spearman ρ = {r_av_v:.4f}")
    print(f"    A  vs V    Spearman ρ = {r_a_v:.4f}")
    out_csv = os.path.join(OUT_DIR, "D4_rsa_corrcoef.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["pair", "spearman_rho",
                    "n_unique_pairs"])
        w.writerow(["AV_vs_A",  f"{r_av_a:.6f}", str(len(rdm_a))])
        w.writerow(["AV_vs_V",  f"{r_av_v:.6f}", str(len(rdm_a))])
        w.writerow(["A_vs_V",   f"{r_a_v:.6f}",  str(len(rdm_a))])
    print(f"  wrote {out_csv}")


# D4.5 — Linear probe class accuracy (5-fold CV)

def D4_5_class_probe(A, V, AV_full, AV_v_zero, AV_audio_zero, labels) -> None:
    print("\n  D4.5 — Class linear-probe (5-fold CV on val penultimates):")
    targets = [
        ("A_only",            A["penult"]),
        ("V_fair",            V["penult"]),
        ("AV_full",           AV_full["penult"]),
        ("AV_v_zero",         AV_v_zero["penult"]),
        ("AV_audio_zero",     AV_audio_zero["penult"]),
    ]
    out_csv = os.path.join(OUT_DIR, "D4_linprobe_class.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["model_or_condition", "acc_mean", "acc_std",
                    "bal_acc_mean", "bal_acc_std"])
        for name, feats in targets:
            t0 = time.time()
            res = _linprobe_5fold(feats, labels)
            dt = time.time() - t0
            print(f"    {name:>16s} acc={res['acc_mean']*100:5.2f}% "
                  f"±{res['acc_std']*100:.2f}, bal={res['bal_acc_mean']*100:5.2f}%, "
                  f"{dt:.1f}s")
            w.writerow([name,
                        f"{res['acc_mean']:.6f}",
                        f"{res['acc_std']:.6f}",
                        f"{res['bal_acc_mean']:.6f}",
                        f"{res['bal_acc_std']:.6f}"])
    print(f"  wrote {out_csv}")


# Main

def _hash_idx(arr) -> str:
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return hashlib.sha256(bytes(arr.astype("int64").tobytes())).hexdigest()


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Sanity: val_idx hash
    splits = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                        weights_only=False)
    val_idx = splits["val_idx"]
    if hasattr(val_idx, "numpy"):
        val_idx = val_idx.numpy()
    sha = _hash_idx(val_idx)
    assert sha.startswith("03c5a87a"), \
        f"val_idx sha mismatch: got {sha[:16]}, expected 03c5a87a…"
    print(f"  val_idx sha256[:16] = {sha[:16]} (OK)")
    print(f"  N val = {len(val_idx)}")

    # Models
    print("\nLoading models...")
    models = _load_models(device)
    n_classes = len(models["A"][1]["label_to_idx"])
    print(f"  A:  {models['A'][1].get('best_val_acc', 0)*100:.2f}%")
    print(f"  V:  {models['V'][1].get('best_val_acc', 0)*100:.2f}%  "
          f"({models['_V_path']})")
    print(f"  AV: {models['AV'][1].get('best_val_acc', 0)*100:.2f}%")
    print(f"  n_classes = {n_classes}")

    # Loader
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

    # Activation extraction (3 models × {AV: 3 conditions})
    print("\nExtracting activations...")
    t0 = time.time()
    A_act = _extract_A_only(models["A"][0], loader, device)
    print(f"  A_only      done ({time.time()-t0:.1f}s)")
    t0 = time.time()
    V_act = _extract_V_fair(models["V"][0], loader, device)
    print(f"  V_fair      done ({time.time()-t0:.1f}s)")
    t0 = time.time()
    AV_full = _extract_AV(models["AV"][0], loader, device,
                           video_kind="real", audio_kind="real")
    print(f"  AV full     done ({time.time()-t0:.1f}s)")
    t0 = time.time()
    AV_v_zero = _extract_AV(models["AV"][0], loader, device,
                             video_kind="zero", audio_kind="real")
    print(f"  AV v_zero   done ({time.time()-t0:.1f}s)")
    t0 = time.time()
    AV_audio_zero = _extract_AV(models["AV"][0], loader, device,
                                 video_kind="real", audio_kind="zero")
    print(f"  AV a_zero   done ({time.time()-t0:.1f}s)")

    labels = A_act["labels"]
    assert (V_act["labels"] == labels).all()
    assert (AV_full["labels"] == labels).all()

    # Sanity checks (plan §8)
    acc_a = float((A_act["logits"].argmax(1) == labels).mean())
    acc_v = float((V_act["logits"].argmax(1) == labels).mean())
    acc_av = float((AV_full["logits"].argmax(1) == labels).mean())
    print(f"\n  Sanity:")
    print(f"    A_only      = {acc_a:.4%} (expect 92.70 ± 0.05%)")
    print(f"    V_fair      = {acc_v:.4%} (expect 86.50 ± 0.5%)")
    print(f"    AV_full     = {acc_av:.4%} (expect 95.67–95.80%)")
    sanity = []
    sanity.append(("A_only_92.70_pm_0.05",
                    abs(acc_a - 0.9270) <= 0.0005))
    sanity.append(("V_fair_86.50_pm_0.5",
                    abs(acc_v - 0.8650) <= 0.005))
    sanity.append(("AV_full_in_0.9567_0.9580",
                    0.9560 <= acc_av <= 0.9590))
    fails = [n for n, ok in sanity if not ok]
    if fails:
        print(f"  [FAIL] sanity FAILED: {fails}")
    else:
        print(f"  [OK] all 3 sanity checks pass")

    # Save activation cache
    print(f"\nSaving cache → {CACHE_PATH}")
    torch.save({
        "val_idx": val_idx,
        "val_idx_sha256": sha,
        "labels": labels,
        "n_classes": n_classes,
        "A_only": A_act,
        "V_fair": V_act,
        "AV_clean_full": AV_full,
        "AV_clean_v_zero": AV_v_zero,
        "AV_clean_audio_zero": AV_audio_zero,
    }, CACHE_PATH)
    sz = os.path.getsize(CACHE_PATH) / 2**20
    print(f"  cache size: {sz:.1f} MB")

    # Sub-experiments
    D4_1_late_ensemble(A_act, V_act, AV_full, labels)
    D4_2_cka(A_act, V_act, AV_full)
    D4_3_rsa(A_act, V_act, AV_full, labels, n_classes)
    D4_5_class_probe(A_act, V_act, AV_full, AV_v_zero, AV_audio_zero, labels)

    print("\nPhase A done.")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("D4_") and f.endswith(".csv"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
