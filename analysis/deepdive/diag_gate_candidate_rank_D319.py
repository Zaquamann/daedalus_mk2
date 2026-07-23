#!/usr/bin/env python3
"""TEMP DEBUG (D319, task #28 follow-on) — RANK the lead's 3 A' candidates single-variable.

Lead's reframing after the delta-GROWTH correction (+0.015->+0.056, conf_v usable):
"architecture structurally can't route" is DOWNGRADED to fallback; A' is most likely a
TUNABLE strengthening (make the per-trial gradient win the race vs the global relaxation).
Prove WHICH binds, rank by effect size (not mutually exclusive):
  Cand 1 = operating-point / task-CE dominating the per-trial gradient
  Cand 2 = GATE_SUP_W too weak
  Cand 3 = detached-argmax target-0.9 cap throttling the achievable swing
Keep STRUCTURAL (architecture) as the explicit fallback only.

All READ-ONLY on the ep100 candidate-(d) ckpt (frozen), clean val (sigma0), GPU1.

 T1 EFFECT-SIZE GRID (ranks Cand 2 vs Cand 1 by realizable gL gain): force
     w_a on the DECISIVE (route_aud) subset in {actual,0.9,1.0} x the NON-DECISIVE
     (both) subset in {actual,0.8,0.9,1.0}; E1d gL each. Decisive axis = Cand-2 lever
     (per-trial strength), non-decisive axis = Cand-1 lever (operating point).
 T1b CAP SLICE (Cand 3): decisive route_aud -> 0.9 vs 1.0 (rest actual); the gL gap = the
     target-0.9 cap cost. Plus uniform 0.9 vs 1.0 (already C in D317).
 T2 GRADIENT DOMINANCE (Cand 1, direct): on the gate's rel_gate params, compare
     ||grad(L_task_CE)|| vs GATE_SUP_W*||grad(L_gate_sup)|| over the clean val set, and
     the PER-TRIAL force on w_a (sign/magnitude) for route_aud vs both_right. If the
     task-CE gradient dominates the gate update AND pulls route_aud w_a back toward the
     global mean, Cand 1 is the active suppressor.
 T3 REALISTIC PER-TRIAL CEILING (tunable vs structural fallback): emulate the BEST a
     strengthened gate-sup could learn = fit LR on rel_in to the per-trial DECISIVENESS
     target (route_aud=1 vs route_vid=0, the actual gate-sup signal, NOT the regime
     label), cross-val-predict, map to w_a, measure E1d gL AND E1c gL (seesaw check).
     LEARNABLE~=ORACLE & E1c holds => TUNABLE (strengthen per-trial). LEARNABLE<<ORACLE
     or E1c crashes when E1d rises => STRUCTURAL fallback.

Run: LATE_CKPT=models/av_fused_latefusion_candd_ep100.pt CUDA_VISIBLE_DEVICES=1 \
     python analysis/deepdive/diag_gate_candidate_rank_D319.py
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
from sklearn.model_selection import cross_val_predict  # noqa: E402

import dprime_latefusion as dlf  # noqa: E402
from analyze_av_msi import _NoisyAudioView, BATCH_SIZE  # noqa: E402

dlf.NW = 6
SEED = 0
GATE_SUP_W = float(os.environ.get("GATE_SUP_W", "1.0"))  # the trained value


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
    return d, d / best


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
          f"GATE_SUP_W(assumed train)={GATE_SUP_W}", flush=True)

    store = {}
    AVl.rel_gate.register_forward_pre_hook(
        lambda m, inp: store.__setitem__("x", inp[0].detach()))

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
            RIN.append(store["x"].cpu().numpy())
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
    dA_d = _dp_pairs(PA, ys, pid_d)
    best_d = float(np.nanmean(np.maximum(dA_d, dV_d)))

    route_aud = e1d & a_ok & (~v_ok)
    route_vid = e1d & v_ok & (~a_ok)
    both_ok = e1d & a_ok & v_ok
    both_no = e1d & (~a_ok) & (~v_ok)
    nondec = both_ok | both_no
    d0, g0 = _gL(WA.copy(), LA, LV, ys, pid_d, best_d)
    print(f"\nE1d-clean n={int(e1d.sum())}  ACTUAL gL={g0:.3f}  "
          f"mean w_a: route_aud={WA[route_aud].mean():.3f} "
          f"nondec={WA[nondec].mean():.3f}", flush=True)

    # ===== T1 EFFECT-SIZE GRID: decisive (Cand2) x non-decisive (Cand1) =====
    print("\n===== T1: E1d gL grid — decisive(route_aud) w_a  x  non-decisive(both) w_a "
          "=====", flush=True)
    dvals = [("actual", None), ("0.9", 0.9), ("1.0", 1.0)]
    nvals = [("actual", None), ("0.8", 0.8), ("0.9", 0.9), ("1.0", 1.0)]
    hdr = "  decisive\\nondec  " + "".join(f"{nn:>10}" for nn, _ in nvals)
    print(hdr, flush=True)
    for dn, dv in dvals:
        row = f"  {dn:>14}  "
        for nn, nv in nvals:
            wa = WA.copy()
            if dv is not None:
                wa[route_aud] = dv
            if nv is not None:
                wa[nondec] = nv
            _, g = _gL(wa, LA, LV, ys, pid_d, best_d)
            row += f"{g:>10.3f}"
        print(row, flush=True)
    print("  (decisive axis = Cand-2 per-trial-strength lever; "
          "non-decisive axis = Cand-1 operating-point lever)", flush=True)

    # ===== T1b CAP SLICE (Cand 3): route_aud->0.9 vs 1.0, rest actual =====
    print("\n===== T1b: target-0.9 cap cost (Cand 3) =====", flush=True)
    for tv in [0.9, 1.0]:
        wa = WA.copy(); wa[route_aud] = tv
        _, g = _gL(wa, LA, LV, ys, pid_d, best_d)
        print(f"  route_aud->{tv}: gL={g:.3f}", flush=True)

    # ===== T2 GRADIENT DOMINANCE (Cand 1) =====
    print("\n===== T2: gradient dominance on the gate (Cand 1) =====", flush=True)
    rin_t = torch.tensor(RIN, dtype=torch.float32, device=device)
    la_t = torch.tensor(LA, dtype=torch.float32, device=device)
    lv_t = torch.tensor(LV, dtype=torch.float32, device=device)
    y_t = torch.tensor(ys, dtype=torch.long, device=device)
    # param-space: which loss dominates the rel_gate update over the full clean val set
    for p in AVl.rel_gate.parameters():
        p.requires_grad_(True)
    w = torch.softmax(AVl.rel_gate(rin_t), dim=1)
    wa = w[:, 0]
    fused = wa[:, None] * la_t + (1.0 - wa[:, None]) * lv_t
    L_task = F.cross_entropy(fused, y_t)
    params = [p for p in AVl.rel_gate.parameters()]
    g_task = torch.autograd.grad(L_task, params, retain_graph=True)
    nrm_task = float(torch.sqrt(sum((g * g).sum() for g in g_task)))
    sel = torch.tensor(route_aud | route_vid, device=device)
    tgt = torch.full_like(wa, 0.1)
    tgt[torch.tensor(route_aud, device=device)] = 0.9
    L_gs = F.binary_cross_entropy(wa[sel].clamp(1e-6, 1 - 1e-6), tgt[sel])
    g_gs = torch.autograd.grad(L_gs, params, retain_graph=False)
    nrm_gs = float(torch.sqrt(sum((g * g).sum() for g in g_gs)))
    print(f"  ||grad rel_gate||: task-CE={nrm_task:.4e}  "
          f"GATE_SUP_W*gate-sup={GATE_SUP_W * nrm_gs:.4e}  "
          f"ratio task/gatesup={nrm_task / (GATE_SUP_W * nrm_gs + 1e-12):.2f}", flush=True)
    # per-trial w_a force direction (negative grad = pushes w_a UP)
    wa_leaf = torch.tensor(WA, dtype=torch.float32, device=device, requires_grad=True)
    fused2 = wa_leaf[:, None] * la_t + (1.0 - wa_leaf[:, None]) * lv_t
    gt = torch.autograd.grad(F.cross_entropy(fused2, y_t, reduction="sum"), wa_leaf)[0]
    gt = gt.cpu().numpy()  # dL_task/dw_a per trial (sum reduction => per-trial force)
    print(f"  per-trial dL_task/dw_a (mean): route_aud={gt[route_aud].mean():+.3f} "
          f"(neg=wants w_a UP)  both_right={gt[both_ok].mean():+.3f} "
          f"(pos=wants w_a DOWN)", flush=True)
    print(f"  -> if task||grad|| dominates AND both_right pulls w_a DOWN while route_aud "
          f"is the minority, the operating point (Cand 1) suppresses the per-trial push.",
          flush=True)

    # ===== T3 REALISTIC PER-TRIAL CEILING (tunable vs structural fallback) =====
    print("\n===== T3: realistic per-trial gate from DECISIVENESS target (not regime) "
          "=====", flush=True)
    dec_all = (a_ok & (~v_ok)) | (v_ok & (~a_ok))  # decisive across BOTH regimes
    dec_aud = a_ok & (~v_ok)
    # fit LR on rel_in over decisive trials: route_aud(1) vs route_vid(0)
    Xtr = RIN[dec_all]
    ytr = dec_aud[dec_all].astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    # cross-val prob on decisive trials (leakage-safe); fit-all to score the both trials
    p_dec = cross_val_predict(clf, Xtr, ytr, cv=4, method="predict_proba")[:, 1]
    clf.fit(Xtr, ytr)
    p_all = clf.predict_proba(RIN)[:, 1]
    p_all[np.where(dec_all)[0]] = p_dec  # use CV preds where we have them
    # map route_aud-prob -> w_a in [0.375 (E1c/video opt), 1.0 (E1d/audio opt)]
    wa_real = 0.375 + p_all * (1.0 - 0.375)
    # E1c gL needs its own best_single
    dA_c = _dp_pairs(PA, ys, pid_c)
    best_c = float(np.nanmean(np.maximum(dA_c, dV_c)))
    wa_oracle = WA.copy()
    wa_oracle[e1d] = 1.0
    wa_oracle[np.isin(ys, cls_c)] = 0.375
    print(f"  AUC-decisive(rel_in->route_aud vs route_vid) inferred via gL below", flush=True)
    print(f"  {'gate':30s} {'E1d gL':>8} {'E1c gL':>8}", flush=True)
    for name, wav in [("current (actual)", WA),
                      ("REALISTIC (LR decisiveness)", wa_real),
                      ("ORACLE (E1d->1.0,E1c->0.375)", wa_oracle)]:
        _, gd = _gL(wav, LA, LV, ys, pid_d, best_d)
        _, gc = _gL(wav, LA, LV, ys, pid_c, best_c)
        print(f"  {name:30s} {gd:8.3f} {gc:8.3f}", flush=True)
    print("  -> REALISTIC~=ORACLE & E1c holds => TUNABLE (strengthen per-trial wins); "
          "E1c CRASHES as E1d rises => STRUCTURAL seesaw fallback.", flush=True)
    print("D319_RC=0", flush=True)


if __name__ == "__main__":
    main()
