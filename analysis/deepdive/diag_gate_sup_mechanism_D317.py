#!/usr/bin/env python3
"""TEMP DEBUG (D317, task #28) — HALT confirmed at ep100: candidate-(d) E1d-clean gL
stuck ~0.935 (NOT rising), E1d w_a relaxing 0.839->0.786->0.686, per-trial within-E1d
delta weakly growing (+0.007->+0.015->+0.056) but OUTPACED by the global relaxation.
PROVE single-variable WHY per-trial conf_v->route discrimination isn't developing
despite conf_v->video-wrong AUC ~0.74.

The gate-supervision (train_av_latefusion.py L381-400) fires ONLY on BOTH-present
DECISIVE trials (exactly one head's 180-way argmax correct): target w_a=0.9 when
audio-right&video-wrong, 0.1 when video-right&audio-wrong; both-right/both-wrong get
NO push ("gate stays free to learn ~0.5"). HYPOTHESIS: on E1d-clean, video is
PAIR-confusable even when its 180-way argmax is correct, so the d' floor wants
w_a->1.0 on ALL E1d-clean trials — but the supervision only covers the DECISIVE
subset and leaves the both-right MAJORITY unsupervised, so the regime-average w_a
relaxes to the global mean and gL caps below 1.0.

Tests (all read-only, ep100 ckpt, clean val, GPU1):
 A — coverage: fraction of E1d-clean in each partition {route_aud, route_vid,
     both_right, both_wrong} and the gate-sup target each receives.
 B — w_a decomposition: actual w_a per partition. Prediction: route_aud HIGH
     (supervised->0.9, rules out GATE_SUP_W-too-weak C1); both_right LOW (~global
     mean, unsupervised drift = C4/structural-target).
 B2 — CAUSAL single-variable: take ACTUAL per-trial w_a, override ONLY the both-right
     E1d subset -> 1.0 (change nothing else), recompute E1d gL. If gL jumps toward 1.0
     it PROVES the unsupervised both-right drift is the binding cap. Compare vs
     overriding only the decisive subset.
 C — target-0.9 cap (C2): forced uniform w_a in {actual, 0.8, 0.9, 0.95, 1.0} -> E1d gL.
     Is 0.9 a binding ceiling or a minor secondary one (gate is at 0.686 << 0.9)?
 E — TUNABLE vs STRUCTURAL (lead's key question): AUC(rel_in -> E1d-clean vs E1c-clean
     regime) from the gate's ACTUAL input (262-d, via forward-pre-hook) and from the
     6-d [conf_a||conf_v] reliability signal. HIGH AUC => the gate CAN route by regime
     from its inputs => the failure is the supervision TARGET design (tunable, A' =
     regime-reliability target). LOW AUC => signal structurally too weak for the
     convex-combo gate (architecture limit) -> surface to lead.

Run: LATE_CKPT=models/av_fused_latefusion_candd_ep100.pt CUDA_VISIBLE_DEVICES=1 \
     python analysis/deepdive/diag_gate_sup_mechanism_D317.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
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


def _conf_np(logit):
    p = _smx(logit)
    ent = -(p * np.log(p + 1e-12)).sum(1, keepdims=True)
    srt = np.sort(p, 1)
    return np.concatenate([ent, srt[:, -1:], srt[:, -1:] - srt[:, -2:-1]], 1)


def _dp_pairs(prob, lab, pair_ids):
    lp = dlf._logp(prob)
    return np.array([abs(dlf._dprime_pair(lp, lab, i, j)) for i, j in pair_ids])


def _gL(wa_vec, la, lv, lab, pair_ids, best):
    fused = wa_vec[:, None] * la + (1.0 - wa_vec)[:, None] * lv
    d = float(np.nanmean(_dp_pairs(_smx(fused), lab, pair_ids)))
    return d, d / best


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
    print(f"ckpt={os.path.basename(dlf.LATE_CKPT)} acc={ck.get('best_val_acc')}",
          flush=True)

    # capture the ACTUAL 262-d rel_gate input via forward-pre-hook
    store = {}
    AVl.rel_gate.register_forward_pre_hook(
        lambda m, inp: store.__setitem__("x", inp[0].detach().cpu().numpy()))

    pid_d, dV_d = dlf._select_pairs("e1d", A, V, base, val_idx, device)
    pid_c, dV_c = dlf._select_pairs("e1c", A, V, base, val_idx, device)
    cls_d = sorted({c for p in pid_d for c in p})
    cls_c = sorted({c for p in pid_c for c in p})

    loader = DataLoader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=SEED),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=dlf.NW, pin_memory=True)
    LA, LV, WA, RIN, PA, ys = [], [], [], [], [], []
    with torch.no_grad():
        for mel, vid, y in loader:
            m1 = mel.unsqueeze(1).to(device, non_blocking=True)
            vd = vid.to(device, non_blocking=True)
            _, la, lv, w = AVl(m1, vd, return_parts=True)
            RIN.append(store["x"])
            LA.append(la.cpu().numpy()); LV.append(lv.cpu().numpy())
            WA.append(w[:, 0].cpu().numpy())
            PA.append(A(m1).softmax(1).cpu().numpy())
            ys.append(y.numpy())
    LA, LV = np.concatenate(LA), np.concatenate(LV)
    WA = np.concatenate(WA); RIN = np.concatenate(RIN)
    PA = np.concatenate(PA); ys = np.concatenate(ys)

    a_ok = (LA.argmax(1) == ys)
    v_ok = (LV.argmax(1) == ys)
    e1d = np.isin(ys, cls_d)

    # ---- best_single per design (for gL) ----
    dA_d = _dp_pairs(PA, ys, pid_d)
    best_d = float(np.nanmean(np.maximum(dA_d, dV_d)))

    # =========== partitions on E1d-clean (training mask logic) ===========
    route_aud = e1d & a_ok & (~v_ok)     # target 0.9 in training
    route_vid = e1d & v_ok & (~a_ok)     # target 0.1
    both_ok = e1d & a_ok & v_ok          # NO push (free ~0.5)
    both_no = e1d & (~a_ok) & (~v_ok)    # NO push
    parts = [("route_aud(A-rt,V-wr) tgt0.9", route_aud),
             ("route_vid(V-rt,A-wr) tgt0.1", route_vid),
             ("both_right         NO-push", both_ok),
             ("both_wrong         NO-push", both_no)]
    nE = int(e1d.sum())
    print(f"\n===== A+B: E1d-clean partition (n={nE}, video-head acc={v_ok[e1d].mean():.3f}) "
          f"| actual mean w_a(E1d)={WA[e1d].mean():.3f} =====", flush=True)
    print(f"{'partition':30s} {'frac':>6} {'mean_w_a':>9} {'std':>6}", flush=True)
    for name, m in parts:
        f = m.sum() / max(nE, 1)
        wm = WA[m].mean() if m.any() else float("nan")
        ws = WA[m].std() if m.any() else float("nan")
        print(f"{name:30s} {f:6.3f} {wm:9.3f} {ws:6.3f}", flush=True)
    supervised = route_aud | route_vid
    print(f"  -> supervised (decisive) frac of E1d-clean = {supervised[e1d].sum()/nE:.3f}; "
          f"UNsupervised (both) frac = {(both_ok|both_no)[e1d].sum()/nE:.3f}", flush=True)

    # =========== B2: CAUSAL per-trial override (single-variable) ===========
    print(f"\n===== B2: E1d gL, override ONLY a subset's w_a -> 1.0 (else ACTUAL) =====",
          flush=True)
    d0, g0 = _gL(WA.copy(), LA, LV, ys, pid_d, best_d)
    print(f"  ACTUAL w_a                         : d'={d0:.3f}  gL={g0:.3f}", flush=True)
    for name, m in [("both_right->1.0", both_ok),
                    ("both(right+wrong)->1.0", both_ok | both_no),
                    ("decisive(route_aud)->1.0", route_aud),
                    ("ALL E1d-clean->1.0 (ceiling)", e1d)]:
        wa2 = WA.copy(); wa2[m] = 1.0
        d, g = _gL(wa2, LA, LV, ys, pid_d, best_d)
        print(f"  +{name:32s}: d'={d:.3f}  gL={g:.3f}  (ΔgL={g-g0:+.3f})", flush=True)

    # =========== C: target-0.9 cap (uniform forced) ===========
    print(f"\n===== C: forced UNIFORM w_a -> E1d gL (target-0.9 cap check) =====",
          flush=True)
    for wf in [WA[e1d].mean(), 0.8, 0.9, 0.95, 1.0]:
        d, g = _gL(np.full(len(WA), wf), LA, LV, ys, pid_d, best_d)
        print(f"  w_a={wf:5.3f}: d'={d:.3f}  gL={g:.3f}", flush=True)

    # =========== E: TUNABLE vs STRUCTURAL — regime separability ===========
    # E1d-only vs E1c-only clean trials, from the gate's ACTUAL inputs
    only_d = np.isin(ys, sorted(set(cls_d) - set(cls_c)))
    only_c = np.isin(ys, sorted(set(cls_c) - set(cls_d)))
    sub = only_d | only_c
    yreg = only_d[sub].astype(int)   # 1 = E1d regime (want w_a high)
    CONF = np.concatenate([_conf_np(LA), _conf_np(LV)], 1)  # 6-d reliability signal
    print(f"\n===== E: regime separability (E1d-only n={int(only_d.sum())} vs "
          f"E1c-only n={int(only_c.sum())}) — TUNABLE vs STRUCTURAL =====", flush=True)
    print(f"  AUC(full rel_in 262-d -> E1d-vs-E1c regime) = {auc(RIN[sub], yreg):.3f}",
          flush=True)
    print(f"  AUC([conf_a||conf_v] 6-d -> regime)         = {auc(CONF[sub], yreg):.3f}",
          flush=True)
    print(f"  (HIGH => gate CAN route by regime from its inputs => failure is the "
          f"supervision TARGET (tunable). LOW => signal structurally weak.)", flush=True)
    # reliability contrast: conf_a-vs-conf_v margin by regime
    print(f"  mean max-prob: E1d-only conf_a={_conf_np(LA)[only_d,1].mean():.3f} "
          f"conf_v={_conf_np(LV)[only_d,1].mean():.3f} | "
          f"E1c-only conf_a={_conf_np(LA)[only_c,1].mean():.3f} "
          f"conf_v={_conf_np(LV)[only_c,1].mean():.3f}", flush=True)

    # =========== E2: best LEARNABLE readout gL vs ORACLE (DECISIVE tunable-vs-structural) ===========
    # Restrict to only-regime pairs (all their trials are in `sub`, so cross-val preds exist).
    dset, cset = set(sorted(set(cls_d) - set(cls_c))), set(sorted(set(cls_c) - set(cls_d)))
    idx_d_o = [k for k, (i, j) in enumerate(pid_d) if i in dset and j in dset]
    idx_c_o = [k for k, (i, j) in enumerate(pid_c) if i in cset and j in cset]
    pid_d_o = [pid_d[k] for k in idx_d_o]
    pid_c_o = [pid_c[k] for k in idx_c_o]
    dA_c = _dp_pairs(PA, ys, pid_c)
    best_d_o = float(np.nanmean(np.maximum(dA_d[idx_d_o], np.asarray(dV_d)[idx_d_o])))
    best_c_o = float(np.nanmean(np.maximum(dA_c[idx_c_o], np.asarray(dV_c)[idx_c_o])))
    # best LEARNABLE per-trial gate: LR(rel_in)->P(E1d), mapped to w_a in [0.375(E1c opt),1.0(E1d opt)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p_e1d = cross_val_predict(clf, RIN[sub], yreg, cv=4, method="predict_proba")[:, 1]
    wa_learn = WA.copy(); wa_learn[np.where(sub)[0]] = 0.375 + p_e1d * (1.0 - 0.375)
    wa_oracle = WA.copy()
    wa_oracle[np.isin(ys, sorted(dset))] = 1.0
    wa_oracle[np.isin(ys, sorted(cset))] = 0.375
    print(f"\n===== E2: per-trial gate gL on ONLY-regime pairs (E1d n_pairs={len(pid_d_o)}, "
          f"E1c n_pairs={len(pid_c_o)}) — best learnable vs oracle =====", flush=True)
    print(f"  {'gate':26s} {'E1d-only gL':>12} {'E1c-only gL':>12}", flush=True)
    for name, wav in [("current (actual gate)", WA),
                      ("LEARNABLE (LR on rel_in)", wa_learn),
                      ("ORACLE (true regime)", wa_oracle)]:
        _, gd = _gL(wav, LA, LV, ys, pid_d_o, best_d_o)
        _, gc = _gL(wav, LA, LV, ys, pid_c_o, best_c_o)
        print(f"  {name:26s} {gd:12.3f} {gc:12.3f}", flush=True)
    print("  -> LEARNABLE≈ORACLE & both≥~1.0 => TUNABLE (better target); "
          "LEARNABLE≪ORACLE => STRUCTURAL (AUC ceiling caps any gate).", flush=True)
    print("D317_RC=0", flush=True)


if __name__ == "__main__":
    main()
