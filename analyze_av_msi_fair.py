#!/usr/bin/env python3
"""Re-run the V-only-touching MSI experiments (E6/E7, E8, E10) using the fair
V-only checkpoint `models/video_only_fair.pt` (#17, 86.59 % val_acc).

Outputs land alongside the original CSVs as `*_fair.csv` so the lean and fair
numbers can be compared.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (
    BATCH_SIZE, OUT_DIR, T_STRIDE,
    _ValAVView, _accuracy, _forward_A, _forward_AV, _forward_V,
)
from dataset_raw_noisy import RawNoisyAVDataset
from model_av import AVWordResNet
from model_v_only_fair import VOnlyFairWordResNet
from train import WordResNet


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
A_CKPT = os.path.join(SCRIPT_DIR, "models", "audio_only_filtered.pt")
V_FAIR_CKPT = os.path.join(SCRIPT_DIR, "models", "video_only_fair.pt")
AV_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused.pt")


def _load(device):
    a_ck = torch.load(A_CKPT, weights_only=False)
    a = WordResNet(len(a_ck["label_to_idx"]))
    a.load_state_dict(a_ck["model_state_dict"])
    v_ck = torch.load(V_FAIR_CKPT, weights_only=False)
    v = VOnlyFairWordResNet(len(v_ck["label_to_idx"]))
    v.load_state_dict(v_ck["model_state_dict"])
    av_ck = torch.load(AV_CKPT, weights_only=False)
    av = AVWordResNet(len(av_ck["label_to_idx"]))
    av.load_state_dict(av_ck["model_state_dict"])
    return {
        "A": (a.to(device).eval(), a_ck),
        "V": (v.to(device).eval(), v_ck),
        "AV": (av.to(device).eval(), av_ck),
    }


def E67_race_bound_fair(models, val_idx, base, device):
    print("\n[E6/E7 — fair V-only] race-bound on per-item correctness")
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    a_preds, _, labels = _forward_A(models["A"][0], loader, device)
    v_preds, _, _ = _forward_V(models["V"][0], loader, device)
    av_out = _forward_AV(models["AV"][0], loader, device)
    av_preds = av_out["preds"]

    p_a = (a_preds == labels).astype(np.int32)
    p_v = (v_preds == labels).astype(np.int32)
    p_av = (av_preds == labels).astype(np.int32)
    bound = (p_a | p_v).astype(np.int32)
    violations = (p_av > bound).astype(np.int32)

    rows = {
        "n_items": int(len(labels)),
        "P_A":  float(p_a.mean()),
        "P_V":  float(p_v.mean()),
        "P_AV": float(p_av.mean()),
        "P_A_or_V":         float(bound.mean()),
        "frac_violations":  float(violations.mean()),
        "n_violations":     int(violations.sum()),
        "frac_AV_correct_alone": float(((p_av == 1) & (bound == 0)).mean()),
    }
    out_csv = os.path.join(OUT_DIR, "E67_race_bound_fair.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(list(rows.keys()))
        w.writerow([f"{v:.6f}" if isinstance(v, float) else v
                    for v in rows.values()])
    print(f"  P(A)={rows['P_A']:.4%}, P(V_fair)={rows['P_V']:.4%}, "
          f"P(AV)={rows['P_AV']:.4%}, P(A∨V)={rows['P_A_or_V']:.4%}")
    print(f"  AV-only-correct: {rows['n_violations']}/{rows['n_items']} "
          f"({rows['frac_violations']:.4%})")
    return rows


def E8_cross_predict_fair(models, val_idx, base, device, av_acts):
    print("\n[E8 — fair V-only] cross-modal prediction a_mid ↔ v_mid")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    av_amid = av_acts["a_mid"].reshape(av_acts["a_mid"].shape[0], -1)
    av_vmid = av_acts["v_mid"].reshape(av_acts["v_mid"].shape[0], -1)

    # Independent unimodal mid features — A-only (block1), V-only-fair (visual encoder out)
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    a_only = []
    with torch.no_grad():
        for mel, _v, _y in loader:
            x = mel.unsqueeze(1).to(device)
            a_only.append(models["A"][0].block1(x).cpu().numpy())
    a_only = np.concatenate(a_only).reshape(av_amid.shape[0], -1)
    v_only = []
    with torch.no_grad():
        for _m, vid, _y in loader:
            v = vid.to(device)
            v_only.append(models["V"][0].visual(v).cpu().numpy())
    v_only = np.concatenate(v_only).reshape(av_vmid.shape[0], -1)

    def _project(x: np.ndarray, k: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((x.shape[1], k)).astype(np.float32) / np.sqrt(x.shape[1])
        return x.astype(np.float32) @ proj

    av_amid_p = _project(av_amid, seed=0)
    av_vmid_p = _project(av_vmid, seed=1)
    a_only_p = _project(a_only, seed=2)
    v_only_p = _project(v_only, seed=3)

    rows = []
    for label, X, Y in [
        ("AV_a→v", av_amid_p, av_vmid_p),
        ("AV_v→a", av_vmid_p, av_amid_p),
        ("UNI_fair_a→v", a_only_p, v_only_p),
        ("UNI_fair_v→a", v_only_p, a_only_p),
    ]:
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2,
                                              random_state=42)
        clf = Ridge(alpha=1.0)
        clf.fit(Xtr, Ytr)
        r2 = r2_score(Yte, clf.predict(Xte), multioutput="variance_weighted")
        rows.append((label, float(r2)))
        print(f"  {label:>14s}: R² = {r2:.4f}")

    out_csv = os.path.join(OUT_DIR, "E8_cross_predict_fair.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["probe", "r2"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.6f}"])
    return rows


def E10_bayes_fair(models, val_idx, base, device):
    print("\n[E10 — fair V-only] inverse-variance check on confidence")
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    _, prob_a, _ = _forward_A(models["A"][0], loader, device)
    _, prob_v, _ = _forward_V(models["V"][0], loader, device)
    av_out = _forward_AV(models["AV"][0], loader, device)
    prob_av = av_out["probs"]

    conf_a = prob_a.max(axis=1)
    conf_v = prob_v.max(axis=1)
    conf_av = prob_av.max(axis=1)

    var_a = float(conf_a.var())
    var_v = float(conf_v.var())
    var_av_obs = float(conf_av.var())
    pred_var_av = (var_a * var_v) / max(var_a + var_v, 1e-12)

    rows = {
        "mean_conf_A": float(conf_a.mean()),
        "mean_conf_V_fair": float(conf_v.mean()),
        "mean_conf_AV": float(conf_av.mean()),
        "var_conf_A": var_a,
        "var_conf_V_fair": var_v,
        "var_conf_AV_observed": var_av_obs,
        "var_conf_AV_optimal_pred": pred_var_av,
        "ratio_observed_over_optimal": var_av_obs / max(pred_var_av, 1e-12),
    }
    out_csv = os.path.join(OUT_DIR, "E10_bayes_check_fair.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(list(rows.keys()))
        w.writerow([f"{v:.6f}" for v in rows.values()])
    print(f"  conf mean: A={rows['mean_conf_A']:.3f}, V_fair={rows['mean_conf_V_fair']:.3f}, "
          f"AV={rows['mean_conf_AV']:.3f}")
    print(f"  conf var:  A={var_a:.4f}, V_fair={var_v:.4f}, "
          f"AV(obs)={var_av_obs:.4f}, AV(opt)={pred_var_av:.4f}")
    return rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    s = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                   weights_only=False)
    val_idx = s["val_idx"]
    models = _load(device)
    for name in ("A", "V", "AV"):
        m, ck = models[name]
        n = sum(p.numel() for p in m.parameters())
        ba = ck.get("best_val_acc", float("nan"))
        print(f"  {name:>2s}: params={n:,}, best_val_acc={ba:.4%}")

    e67 = E67_race_bound_fair(models, val_idx, base, device)

    # E8 needs av_acts. Re-derive a_mid + v_mid from AV.
    print("\n[prep] caching AV acts for E8...")
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    av_acts = _forward_AV(models["AV"][0], loader, device, return_acts=True)
    e8 = E8_cross_predict_fair(models, val_idx, base, device, av_acts)
    e10 = E10_bayes_fair(models, val_idx, base, device)

    print("\nDone. Outputs in analysis/msi/ as *_fair.csv")
    print("  E6/E7:", e67)
    print("  E8:", e8)
    print("  E10:", e10)


if __name__ == "__main__":
    main()
