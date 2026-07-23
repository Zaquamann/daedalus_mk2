#!/usr/bin/env python3
"""TEMP DEBUG (D322, task #28) — MULTI-TAP COMBINATION pre-test (the LAST gating loophole).

D321 showed NO single video-pathway tap supplies the E1d-vs-E1c regime signal (all AUC ≤0.712 <
late 0.764; best LEARNABLE gL 0.904/0.880 << 0.98/0.95 bar). One loophole remained: res2's E1c gL
(0.880) exceeds rel_in's (0.796) while rel_in's E1d (0.892) leads — so an ADDITIVE/CONCAT multi-tap
signal (combine the best-E1c mid tap with the best-E1d late tap) is not strictly ruled out.

This closes it. Replicates the D321/E2 harness EXACTLY (leakage-safe cross_val_predict, only-regime
pairs E1d n=10 / E1c n=11, same head logits la/lv + best_single; ONLY the gate INPUT changes) and
feeds the gate MULTI-TAP combinations:
  multi_concat = concat(res2, v_blk2, rel_in)                 [the lead's specified combo]
  multi_all    = concat(res2, res3, v_mid, v_blk2, rel_in)    [absolute best shot, all depths]
  ensemble_add = mean of the per-tap CV regime-probabilities of {res2, v_blk2, rel_in}  [ADDITIVE]
rel_in + the singles are re-run as the harness self-check / reference.

PRE-REGISTERED (lead, fixed before data):
  PASS = some COMBINATION reaches LEARNABLE gL E1d-only ≥0.98 AND E1c-only ≥0.95 (or AUC ≥~0.85)
         → a multi-tap signal EXISTS (a gate could route on it).
  FAIL (predicted by the pair-level mechanism) = no combination clears it → closes the last gating
         variant; structural airtight across ALL depths AND combinations.

Run: LATE_CKPT=models/av_fused_latefusion_candd_ep100.pt CUDA_VISIBLE_DEVICES=1 \
     python analysis/deepdive/diag_multitap_pretest_D322.py
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


def _pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def auc(X, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    return cross_val_score(_pipe(), X, y, cv=4, scoring="roc_auc").mean()


def cv_p(X, y):
    return cross_val_predict(_pipe(), X, y, cv=4, method="predict_proba")[:, 1]


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
    print(f"ckpt={os.path.basename(dlf.LATE_CKPT)} acc={ck.get('best_val_acc')}", flush=True)

    store = {}
    AVl.rel_gate.register_forward_pre_hook(
        lambda m, inp: store.setdefault("rel_in", []).append(
            inp[0].detach().cpu().numpy()))

    def gap_hook(name):
        def h(m, inp, out):
            store.setdefault(name, []).append(
                out.mean(dim=tuple(range(2, out.ndim))).detach().cpu().numpy())
        return h

    AVl.visual.res2.register_forward_hook(gap_hook("res2"))
    AVl.visual.res3.register_forward_hook(gap_hook("res3"))
    AVl.visual.register_forward_hook(gap_hook("v_mid"))
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
    F_ = {k: np.concatenate(v) for k, v in store.items()}
    F_["multi_concat"] = np.concatenate([F_["res2"], F_["v_blk2"], F_["rel_in"]], 1)
    F_["multi_all"] = np.concatenate(
        [F_["res2"], F_["res3"], F_["v_mid"], F_["v_blk2"], F_["rel_in"]], 1)

    # only-regime pairs + best_single (EXACT D321/E2)
    dset = sorted(set(cls_d) - set(cls_c)); cset = sorted(set(cls_c) - set(cls_d))
    only_d = np.isin(ys, dset); only_c = np.isin(ys, cset)
    sub = only_d | only_c
    yreg = only_d[sub].astype(int)
    dA_d = _dp_pairs(PA, ys, pid_d); dA_c = _dp_pairs(PA, ys, pid_c)
    dsetS, csetS = set(dset), set(cset)
    idx_d_o = [k for k, (i, j) in enumerate(pid_d) if i in dsetS and j in dsetS]
    idx_c_o = [k for k, (i, j) in enumerate(pid_c) if i in csetS and j in csetS]
    pid_d_o = [pid_d[k] for k in idx_d_o]; pid_c_o = [pid_c[k] for k in idx_c_o]
    best_d_o = float(np.nanmean(np.maximum(dA_d[idx_d_o], np.asarray(dV_d)[idx_d_o])))
    best_c_o = float(np.nanmean(np.maximum(dA_c[idx_c_o], np.asarray(dV_c)[idx_c_o])))
    subi = np.where(sub)[0]

    def learn_gL(p_e1d):
        wa = WA.copy(); wa[subi] = 0.375 + p_e1d * (1.0 - 0.375)
        return (_gL(wa, LA, LV, ys, pid_d_o, best_d_o),
                _gL(wa, LA, LV, ys, pid_c_o, best_c_o))

    wa_or = WA.copy(); wa_or[np.isin(ys, dset)] = 1.0; wa_or[np.isin(ys, cset)] = 0.375
    print(f"\n===== D322 multi-tap pre-test (E1d n_pairs={len(pid_d_o)}, "
          f"E1c n_pairs={len(pid_c_o)}; leakage-safe CV) =====", flush=True)
    print(f"  ORACLE gL E1d={_gL(wa_or,LA,LV,ys,pid_d_o,best_d_o):.3f} "
          f"E1c={_gL(wa_or,LA,LV,ys,pid_c_o,best_c_o):.3f} | "
          f"current {_gL(WA,LA,LV,ys,pid_d_o,best_d_o):.3f}/"
          f"{_gL(WA,LA,LV,ys,pid_c_o,best_c_o):.3f}", flush=True)
    print(f"  {'feature':16s} {'dim':>4} {'AUC':>6} {'E1d gL':>7} {'E1c gL':>7} {'PASS?':>6}",
          flush=True)
    any_pass = False
    rows = ["rel_in", "res2", "v_blk2", "multi_concat", "multi_all"]
    for name in rows:
        X = F_[name]
        a = auc(X[sub], yreg)
        gd, gc = learn_gL(cv_p(X[sub], yreg))
        ok = (gd >= 0.98) and (gc >= 0.95)
        any_pass = any_pass or (ok and name not in ("rel_in",))
        print(f"  {name:16s} {X.shape[1]:>4} {a:6.3f} {gd:7.3f} {gc:7.3f} "
              f"{('PASS' if ok else ''):>6}", flush=True)
    # ADDITIVE ensemble: mean of per-tap CV regime-probabilities
    p_ens = np.mean([cv_p(F_[t][sub], yreg) for t in ["res2", "v_blk2", "rel_in"]], 0)
    gd, gc = learn_gL(p_ens)
    ok = (gd >= 0.98) and (gc >= 0.95); any_pass = any_pass or ok
    a_ens = auc(np.column_stack([cv_p(F_[t][sub], yreg)
                                 for t in ["res2", "v_blk2", "rel_in"]]), yreg)
    print(f"  {'ensemble_add':16s} {3:>4} {a_ens:6.3f} {gd:7.3f} {gc:7.3f} "
          f"{('PASS' if ok else ''):>6}", flush=True)
    print(f"\n  PRE-REGISTERED VERDICT: "
          f"{'PASS -> multi-tap signal exists' if any_pass else 'FAIL -> last gating variant closed; structural airtight across depths AND combinations'}",
          flush=True)
    print("D322_RC=0", flush=True)


if __name__ == "__main__":
    main()
