#!/usr/bin/env python3
"""TEMP DEBUG (task #20, Q2 evidence) — which TRAIN-TIME video degradation best
reproduces the E1d-confusable video-unreliability signature IN THE SPACE THE GATE
READS (v_pen, the visual_gap output that rel_gate consumes as v_pen.detach())?

The gate routes off video only if it can DETECT, from v_pen, that this video is
unreliable. Candidate (a) trains it by degrading video. For the trained gate to
fire on the EVAL's E1d (clean-but-viseme-AMBIGUOUS video), the training
degradation must push v_pen into the SAME region E1d-confusable clean v_pens
occupy. This probe measures that overlap directly, per degradation:

  detector := LogisticRegression( clean-v_pen[reliable] vs degraded-v_pen )  on
              a TRAIN half of non-E1d trials  (a proxy for what the gate learns)
  transfer := detector's P(unreliable) on held-out CLEAN E1d-confusable v_pen
  control  := detector's P(unreliable) on held-out CLEAN non-E1d v_pen (false-pos)
  signal   := transfer - control   (high => training on this degradation makes the
              reliability detector fire on E1d => candidate (a) transfers)

Also reports:
  - degraded video-head top-1 acc (how much discriminability each degradation
    removes, so we compare at matched damage, not unfair strengths)
  - INTRINSIC separability: detector trained directly on clean-E1d(1) vs
    clean-nonE1d(0) -> is E1d-ambiguity even ENCODED in v_pen at all? (if not,
    NO degradation can transfer and candidate (a) is fundamentally limited.)

READ-ONLY / CHEAP. No training of the model, no production edits. Reuses the
canonical harness verbatim (val split, pair selection, model load).

Run: CUDA_VISIBLE_DEVICES=1 python analysis/deepdive/diag_video_degradation_transfer.py
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

import dprime_latefusion as dlf  # noqa: E402
from analyze_av_msi import _NoisyAudioView, BATCH_SIZE  # noqa: E402

dlf.NW = 6
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ---- candidate degradations (operate on the model-shape video tensor) ----
# pixel: additive Gaussian (the coder's literal audio-symmetric proposal)
# blur : spatial downsample-by-f then up (reduces visual DISCRIMINABILITY)
# drop : temporal frame zeroing (removes lip-MOTION discriminability)


def deg_pixel(v, u):
    rms = v.pow(2).mean().sqrt()
    return (v + torch.randn_like(v) * (u * rms)).clamp(0.0, 1.0)


def deg_blur(v, f):
    shp = v.shape
    H, W = shp[-2], shp[-1]
    x = v.reshape(-1, 1, H, W)
    x = F.interpolate(x, scale_factor=1.0 / f, mode="bilinear",
                      align_corners=False, recompute_scale_factor=False)
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return x.reshape(shp)


def deg_drop(v, p):
    # zero a fraction p of frames along the temporal axis. Robust to both
    # 5D (B,1,T,H,W) [T at dim2] and 4D (B,T,H,W) [T at dim1].
    g = torch.Generator(device=v.device).manual_seed(SEED)
    if v.dim() == 5:
        B, _C, T = v.shape[0], v.shape[1], v.shape[2]
        keep = (torch.rand(B, 1, T, 1, 1, device=v.device, generator=g) >= p)
    else:  # 4D
        B, T = v.shape[0], v.shape[1]
        keep = (torch.rand(B, T, 1, 1, device=v.device, generator=g) >= p)
    return v * keep.to(v.dtype)


# ---- CONTENT-confusability degradations (target the CLASS-AMBIGUITY axis, not
# input quality): blend / scramble the video so the video head is genuinely
# confused BETWEEN CLASSES, the same kind of unreliability E1d-confusable pairs
# have. Hypothesis: these move v_pen along the INTRINSIC E1d direction (AUC 0.813)
# where pixel/blur/drop did not. ----

def deg_mixup(v, lam):
    # blend each clip with another clip (roll by 1 along batch => different word)
    return lam * v + (1.0 - lam) * torch.roll(v, shifts=1, dims=0)


def deg_tshuffle(v):
    # permute the temporal frame order => destroys the lip-MOTION that
    # distinguishes similar visemes, while every frame stays sharp & in-gamut
    g = torch.Generator(device=v.device).manual_seed(SEED)
    if v.dim() == 5:
        T = v.shape[2]
        perm = torch.randperm(T, generator=g, device=v.device)
        return v[:, :, perm]
    T = v.shape[1]
    perm = torch.randperm(T, generator=g, device=v.device)
    return v[:, perm]


# Round 2: input-quality degradations (pixel/blur/drop) all had transfer signal
# <= 0 (see diag_q2_transfer.log). Test CONTENT-confusability degradations here,
# with blur_f4 retained as a consistency reference (should reproduce ~ -0.040).
DEGRADATIONS = [
    ("blur_f4_ref", lambda v: deg_blur(v, 4)),
    ("mixup_l0.50", lambda v: deg_mixup(v, 0.50)),
    ("mixup_l0.65", lambda v: deg_mixup(v, 0.65)),
    ("tshuffle",    lambda v: deg_tshuffle(v)),
]


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

    # E1d pair classes = the audio-strong / video-weak (confusable) word set
    pair_ids, _dV = dlf._select_pairs("e1d", A, V, base, val_idx, device)
    pair_classes = sorted({c for p in pair_ids for c in p})
    print(f"E1d n_pairs={len(pair_ids)} n_pair_classes={len(pair_classes)}",
          flush=True)

    # hook visual_gap -> exactly the v_pen the rel_gate reads (pre-dropout)
    store = {}
    AVl.visual_gap.register_forward_hook(
        lambda m, i, o: store.__setitem__("v", o.flatten(1).detach().cpu().numpy()))

    loader = DataLoader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=SEED),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=dlf.NW, pin_memory=True)

    @torch.no_grad()
    def collect(degfn):
        vps, ys, vcorrect, n = [], [], 0, 0
        shape_printed = False
        for mel, vid, y in loader:
            m1 = mel.unsqueeze(1).to(device, non_blocking=True)
            vd = vid.to(device, non_blocking=True)
            if not shape_printed:
                print(f"  [shapes] mel1={tuple(m1.shape)} vid={tuple(vd.shape)}",
                      flush=True)
                shape_printed = True
            if degfn is not None:
                vd = degfn(vd)
            _, _la, lv, _w = AVl(m1, vd, return_parts=True)
            vps.append(store["v"])
            vcorrect += int((lv.argmax(1).cpu().numpy() == y.numpy()).sum())
            n += len(y)
            ys.append(y.numpy())
        return np.concatenate(vps), np.concatenate(ys), vcorrect / n

    print("\n[clean baseline pass]", flush=True)
    vp_clean, ys, acc_clean = collect(None)
    mask = np.isin(ys, pair_classes)          # E1d-confusable trials
    nonm = ~mask
    print(f"  clean video-head top1 acc = {acc_clean:.3f}  "
          f"(n_E1d={int(mask.sum())} n_other={int(nonm.sum())})", flush=True)

    # held-out split of non-E1d trials: train detector on half, eval false-pos on half
    nonm_idx = np.where(nonm)[0]
    rng.shuffle(nonm_idx)
    half = len(nonm_idx) // 2
    tr_idx, te_idx = nonm_idx[:half], nonm_idx[half:]
    e1d_idx = np.where(mask)[0]

    def detector_signal(vp_deg):
        # train: clean(reliable,0) vs degraded(unreliable,1) on TRAIN half (non-E1d)
        Xtr = np.vstack([vp_clean[tr_idx], vp_deg[tr_idx]])
        ytr = np.concatenate([np.zeros(len(tr_idx)), np.ones(len(tr_idx))])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
        # transfer: P(unreliable) on held-out CLEAN E1d v_pen
        p_e1d = clf.predict_proba(sc.transform(vp_clean[e1d_idx]))[:, 1].mean()
        # control: P(unreliable) on held-out CLEAN non-E1d v_pen (false-pos)
        p_ctrl = clf.predict_proba(sc.transform(vp_clean[te_idx]))[:, 1].mean()
        # sanity: detector recall on its own degraded test half
        p_deg = clf.predict_proba(sc.transform(vp_deg[te_idx]))[:, 1].mean()
        return p_e1d, p_ctrl, p_deg

    # intrinsic: is E1d-ambiguity ENCODED in clean v_pen at all?
    Xi = np.vstack([vp_clean[te_idx], vp_clean[e1d_idx]])
    yi = np.concatenate([np.zeros(len(te_idx)), np.ones(len(e1d_idx))])
    sci = StandardScaler().fit(Xi)
    cli = LogisticRegression(max_iter=2000).fit(sci.transform(Xi), yi)
    from sklearn.model_selection import cross_val_score
    auc_intrinsic = cross_val_score(
        LogisticRegression(max_iter=2000), sci.transform(Xi), yi,
        cv=4, scoring="roc_auc").mean()
    print(f"\n[INTRINSIC] is clean-E1d v_pen separable from clean-nonE1d? "
          f"4-fold AUC = {auc_intrinsic:.3f}  (0.5=not encoded; >>0.5=gate CAN "
          f"see E1d-ambiguity in v_pen)", flush=True)

    print(f"\n{'degradation':>12} {'vhead_acc':>9} {'P_e1d':>7} {'P_ctrl':>7} "
          f"{'signal':>7} {'P_deg':>6}", flush=True)
    print(f"{'(clean)':>12} {acc_clean:9.3f} {'-':>7} {'-':>7} {'-':>7} {'-':>6}",
          flush=True)
    results = []
    for name, fn in DEGRADATIONS:
        vp_deg, _ys2, acc_deg = collect(fn)
        p_e1d, p_ctrl, p_deg = detector_signal(vp_deg)
        sig = p_e1d - p_ctrl
        results.append((name, acc_deg, sig))
        print(f"{name:>12} {acc_deg:9.3f} {p_e1d:7.3f} {p_ctrl:7.3f} "
              f"{sig:7.3f} {p_deg:6.3f}", flush=True)

    print("\n[VERDICT] higher 'signal' at a moderate vhead_acc drop = the "
          "degradation whose learned-unreliable region best overlaps E1d.",
          flush=True)
    results.sort(key=lambda r: r[2], reverse=True)
    print("  ranked by transfer signal:", flush=True)
    for name, acc, sig in results:
        print(f"    {name:>12}  signal={sig:+.3f}  vhead_acc={acc:.3f}", flush=True)
    print("PROBE_RC=0", flush=True)


if __name__ == "__main__":
    main()
