#!/usr/bin/env python3
"""TEMP DEBUG (D321, task #28) — MID-FEATURE decodability PRE-TEST (refute-gate for MID_GATE).

Lead ACCEPTED the STRUCTURAL verdict (D318/D320): confidence-gated pure-LATE fusion is
signal-capped at E1d-clean — the gate's post-hoc inputs (rel_in: detached penults + per-head
[ent,maxp,margin]) separate the E1d-trust-audio vs E1c-trust-video regime at only AUC 0.764,
so LEARNABLE-on-rel_in gives E1d-only gL 0.892 / E1c-only 0.796 (<< ORACLE 1.000/0.968) and a
strengthened per-trial gate seesaws E1c (D320 T3).

REFUTE-GATE for the architectural fix (MID_GATE on -> point the gate at MID-LEVEL video features
instead of post-hoc confidence): do mid-level video-pathway features supply the regime signal the
post-hoc confidences lack, or do they inherit the same ceiling? READ-ONLY; do NOT retrain off this.

The MID_GATE (CrossModalGate, model_av_latefusion.py L146) taps v_mid = VisualEncoder output
(B,64,40,50), UPSTREAM of visual_block2 -> GAP -> v_pen(128-d, already in rel_in). Depth sweep of
the real video-pathway taps (shallow->deep), each reduced by GAP exactly as a gate head pools:
  res2  = VisualEncoder.res2 out (B,32,T,44,44)   [earliest]
  res3  = VisualEncoder.res3 out (B,64,T,44,44)
  v_mid = VisualEncoder out      (B,64,40,50)      [THE MID_GATE tap]
  v_mid_2x2 = v_mid adaptive-pooled 2x2 (256-d)    [richer, gives the hypothesis its best shot]
  v_blk2 = visual_block2 out     (B,128,20,25)     [GAP(v_blk2)=v_pen, the late anchor]
  rel_in = the ACTUAL late gate input (262-d)      [BASELINE — must reproduce D317-E2 0.764/0.892]

PRE-REGISTERED gate (lead, fixed before data):
  PASS (-> lead specs a MID_GATE seed-0 retrain): SOME tap reaches LEARNABLE(mid) E1d-only gL >= 0.98
    AND E1c-only gL >= 0.95 (approaches ORACLE on BOTH, no E1c crash).
  FAIL (-> accept the structural ceiling across fusion depths): NO tap clears E1d>=0.98 & E1c>=0.95
    (or AUC(mid) stays < ~0.85). Ceiling invariant to fusion depth = the result.

Leakage discipline = the whole test: replicate D317-E2 EXACTLY — leakage-safe cross_val_predict
(probe-fit and gL-eval folds disjoint), only-regime pairs (E1d n_pairs=10 / E1c n_pairs=11), the
SAME head logits la/lv and best_single; ONLY the gate INPUT changes per tap.

Run: LATE_CKPT=models/av_fused_latefusion_candd_ep100.pt CUDA_VISIBLE_DEVICES=1 \
     python analysis/deepdive/diag_mid_feature_pretest_D321.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.model_selection import cross_val_score, cross_val_predict  # noqa: E402

import dprime_latefusion as dlf  # noqa: E402
from analyze_av_msi import _NoisyAudioView, BATCH_SIZE  # noqa: E402

dlf.NW = 6
SEED = 0


def _smx(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _dp_pairs(prob, lab, pair_ids):
    lp = dlf._logp(prob)
    return np.array([abs(dlf._dprime_pair(lp, lab, i, j)) for i, j in pair_ids])


def _gL(wa_vec, la, lv, lab, pair_ids, best):
    fused = wa_vec[:, None] * la + (1.0 - wa_vec)[:, None] * lv
    d = float(np.nanmean(_dp_pairs(_smx(fused), lab, pair_ids)))
    return d / best


def auc(X, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X, y, cv=4, scoring="roc_auc").mean()


def main():
    device = torch.device("cuda")
    base = dlf.RawNoisyAVDataset(noise=False, t_stride=dlf.T_STRIDE,
                                 return_video=True)
    val_idx = torch.load(os.path.join(dlf.SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    models = dlf._load_models(device)
    A, V = models["A"][0], models["V"][0]
    ck = torch.load(dlf.LATE_CKPT, weights_only=False)
    AVl = dlf.AVLateFusionReliabilityWordResNet(
        len(ck["label_to_idx"]), use_mid_gate=ck.get("use_mid_gate", False))
    AVl.load_state_dict(ck["model_state_dict"])
    AVl = AVl.to(device).eval()
    print(f"ckpt={os.path.basename(dlf.LATE_CKPT)} acc={ck.get('best_val_acc')} "
          f"use_mid_gate(ckpt)={ck.get('use_mid_gate', False)}", flush=True)

    # ---- hooks: rel_in (late baseline) + the mid-level video taps, GAP-reduced ----
    store = {}
    AVl.rel_gate.register_forward_pre_hook(
        lambda m, inp: store.setdefault("rel_in", []).append(
            inp[0].detach().cpu().numpy()))

    def gap_hook(name):
        def h(m, inp, out):
            g = out.mean(dim=tuple(range(2, out.ndim)))   # (B, C)
            store.setdefault(name, []).append(g.detach().cpu().numpy())
        return h

    def pool2x2_hook(name):
        def h(m, inp, out):
            g = F.adaptive_avg_pool2d(out, 2).flatten(1)  # (B, C*4)
            store.setdefault(name, []).append(g.detach().cpu().numpy())
        return h

    AVl.visual.res2.register_forward_hook(gap_hook("res2"))
    AVl.visual.res3.register_forward_hook(gap_hook("res3"))
    AVl.visual.register_forward_hook(gap_hook("v_mid"))
    AVl.visual.register_forward_hook(pool2x2_hook("v_mid_2x2"))
    AVl.visual_block2.register_forward_hook(gap_hook("v_blk2"))

    pid_d, dV_d = dlf._select_pairs("e1d", A, V, base, val_idx, device)
    pid_c, dV_c = dlf._select_pairs("e1c", A, V, base, val_idx, device)
    cls_d = sorted({c for p in pid_d for c in p})
    cls_c = sorted({c for p in pid_c for c in p})

    loader = DataLoader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=SEED),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=dlf.NW, pin_memory=True)
    LA, LV, WA, PA, ys = [], [], [], [], []
    with torch.no_grad():
        for mel, vid, y in loader:
            m1 = mel.unsqueeze(1).to(device, non_blocking=True)
            vd = vid.to(device, non_blocking=True)
            _, la, lv, w = AVl(m1, vd, return_parts=True)
            LA.append(la.cpu().numpy()); LV.append(lv.cpu().numpy())
            WA.append(w[:, 0].cpu().numpy())
            PA.append(A(m1).softmax(1).cpu().numpy())
            ys.append(y.numpy())
    LA, LV = np.concatenate(LA), np.concatenate(LV)
    WA = np.concatenate(WA); PA = np.concatenate(PA); ys = np.concatenate(ys)
    FEATS = {k: np.concatenate(v) for k, v in store.items()}
    for k, v in FEATS.items():
        print(f"  feat {k:10s} dim={v.shape[1]}", flush=True)

    # ---- only-regime pairs + best_single (EXACT D317-E2) ----
    dset = sorted(set(cls_d) - set(cls_c))
    cset = sorted(set(cls_c) - set(cls_d))
    only_d = np.isin(ys, dset)
    only_c = np.isin(ys, cset)
    sub = only_d | only_c
    yreg = only_d[sub].astype(int)                 # 1 = E1d regime (want w_a high)
    dA_d = _dp_pairs(PA, ys, pid_d); dA_c = _dp_pairs(PA, ys, pid_c)
    dsetS, csetS = set(dset), set(cset)
    idx_d_o = [k for k, (i, j) in enumerate(pid_d) if i in dsetS and j in dsetS]
    idx_c_o = [k for k, (i, j) in enumerate(pid_c) if i in csetS and j in csetS]
    pid_d_o = [pid_d[k] for k in idx_d_o]
    pid_c_o = [pid_c[k] for k in idx_c_o]
    best_d_o = float(np.nanmean(np.maximum(dA_d[idx_d_o], np.asarray(dV_d)[idx_d_o])))
    best_c_o = float(np.nanmean(np.maximum(dA_c[idx_c_o], np.asarray(dV_c)[idx_c_o])))

    # anchors
    wa_oracle = WA.copy()
    wa_oracle[np.isin(ys, dset)] = 1.0
    wa_oracle[np.isin(ys, cset)] = 0.375
    cur_d = _gL(WA, LA, LV, ys, pid_d_o, best_d_o)
    cur_c = _gL(WA, LA, LV, ys, pid_c_o, best_c_o)
    ora_d = _gL(wa_oracle, LA, LV, ys, pid_d_o, best_d_o)
    ora_c = _gL(wa_oracle, LA, LV, ys, pid_c_o, best_c_o)

    print(f"\n===== D321 mid-feature pre-test (E1d n_pairs={len(pid_d_o)}, "
          f"E1c n_pairs={len(pid_c_o)}; only-regime, leakage-safe CV) =====", flush=True)
    print(f"  late ceiling baselines: AUC(rel_in)=0.764 conf6=0.727 | "
          f"LEARNABLE(rel_in) gL 0.892/0.796 | current {cur_d:.3f}/{cur_c:.3f} | "
          f"ORACLE {ora_d:.3f}/{ora_c:.3f}", flush=True)
    print(f"  {'tap':12s} {'AUC->regime':>11} {'E1d-only gL':>12} {'E1c-only gL':>12} "
          f"{'PASS?':>6}", flush=True)

    # rel_in first (self-check it reproduces D317-E2), then mid taps shallow->deep
    order = ["rel_in", "res2", "res3", "v_mid", "v_mid_2x2", "v_blk2"]
    any_pass = False
    for name in order:
        X = FEATS[name]
        a = auc(X[sub], yreg)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        p_e1d = cross_val_predict(clf, X[sub], yreg, cv=4,
                                  method="predict_proba")[:, 1]
        wa = WA.copy()
        wa[np.where(sub)[0]] = 0.375 + p_e1d * (1.0 - 0.375)
        gd = _gL(wa, LA, LV, ys, pid_d_o, best_d_o)
        gc = _gL(wa, LA, LV, ys, pid_c_o, best_c_o)
        ok = (gd >= 0.98) and (gc >= 0.95)
        any_pass = any_pass or (ok and name != "rel_in")
        print(f"  {name:12s} {a:11.3f} {gd:12.3f} {gc:12.3f} {('PASS' if ok else ''):>6}",
              flush=True)
    print(f"\n  PRE-REGISTERED VERDICT: "
          f"{'PASS -> MID_GATE retrain feasible' if any_pass else 'FAIL -> structural ceiling invariant to fusion depth'}",
          flush=True)
    print("D321_RC=0", flush=True)


if __name__ == "__main__":
    main()
