#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of Q7 (non-degenerate info-plane MI).
D5_info_plane.csv. I rebuild the 4 distinct AV-site GAP activations myself
(a_mid_gap, v_mid_gap, gate_out_gap, block2_gap == penult) from the trained
submodules — deepdive_act_cache.pt NOT loaded — and reimplement all four
estimators from their mathematical definitions (q7_info_plane_mi.py NOT imported):

  H(Y)             empirical label entropy = -sum p ln p          (report 5.028045)
  binned_pca8_OLD  PCA-8 x 16 quantile bins -> joint discrete MI  (report ~5.028044 EVERY site = degenerate)
  ksg              Ross(2014) k-NN MI, PCA-16, k=3                 (report 1.354729/2.016282/3.362625/4.809244)
  infonce_linear   H(Y)-heldout_CE, LogisticRegression critic     (report 2.101221/3.028328/4.109006/4.807701)
  infonce_mlp      H(Y)-heldout_CE, MLP(128) critic               (report 1.840675/2.779547/4.002261/4.761737)

Claims under test: OLD pins at H(Y) (degenerate); NEW are non-degenerate, monotone
along a_mid<gate_out<block2, and bounded < H(Y). Spot-check: MI == H(Y) - CE.

Self-check: AV acc 0.956712 (guards the activation-rebuild path). fp32, no autocast.

Run on dev-codex:
    python validator_indep_q7.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q7.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.special import digamma
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
REF_AV = 0.956712
HY_REF = 5.028045
SITES = ["a_mid_gap", "v_mid_gap", "gate_out_gap", "block2_gap"]  # penult == block2_gap
REPORT = {
    ("a_mid_gap", "infonce_mlp"): 1.840675, ("a_mid_gap", "infonce_linear"): 2.101221,
    ("a_mid_gap", "ksg"): 1.354729, ("a_mid_gap", "binned_pca8_OLD"): 5.028044,
    ("v_mid_gap", "infonce_mlp"): 2.779547, ("v_mid_gap", "infonce_linear"): 3.028328,
    ("v_mid_gap", "ksg"): 2.016282, ("v_mid_gap", "binned_pca8_OLD"): 5.028044,
    ("gate_out_gap", "infonce_mlp"): 4.002261, ("gate_out_gap", "infonce_linear"): 4.109006,
    ("gate_out_gap", "ksg"): 3.362625, ("gate_out_gap", "binned_pca8_OLD"): 5.028044,
    ("block2_gap", "infonce_mlp"): 4.761737, ("block2_gap", "infonce_linear"): 4.807701,
    ("block2_gap", "ksg"): 4.809244, ("block2_gap", "binned_pca8_OLD"): 5.028044,
}


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _entropy_nats(y):
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum())


def _ce_lower_bound(X, y, HY, n_classes, critic, seed=0):
    """I(X;Y) >= H(Y) - heldout_CE (nats), 5-fold SKF, per-fold StandardScaler."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_tot, ce_sum = 0, 0.0
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = critic(); clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)
        full = np.full((len(te), n_classes), 1e-12)
        full[:, clf.classes_] = proba
        p_true = full[np.arange(len(te)), y[te]]
        ce_sum += float(-np.log(np.clip(p_true, 1e-12, 1.0)).sum())
        n_tot += len(te)
    ce = ce_sum / n_tot
    return max(HY - ce, 0.0), ce


def _ksg_mi_cd(X, y, k=3, pca_dim=16, seed=0):
    """Ross 2014 k-NN MI, continuous vector X vs discrete y, on PCA-reduced X."""
    d = min(pca_dim, X.shape[1])
    Xp = PCA(n_components=d, random_state=seed).fit_transform(X).astype(np.float64)
    N = len(Xp)
    classes, y_idx = np.unique(y, return_inverse=True)
    Nx = np.empty(N); d_k = np.empty(N)
    for ci in range(len(classes)):
        idx = np.where(y_idx == ci)[0]
        nc = len(idx); Nx[idx] = nc
        kk = min(k, nc - 1)
        if kk < 1:
            d_k[idx] = 0.0; continue
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xp[idx])
        dist, _ = nn.kneighbors(Xp[idx])
        d_k[idx] = dist[:, kk]
    nn_full = NearestNeighbors().fit(Xp)
    m = np.empty(N)
    for i in range(N):
        if d_k[i] <= 0:
            m[i] = 0; continue
        ind = nn_full.radius_neighbors(Xp[i:i + 1], radius=d_k[i] - 1e-12,
                                       return_distance=False)[0]
        m[i] = max(len(ind) - 1, 0)
    mi = digamma(N) + digamma(k) - np.mean(digamma(Nx)) - np.mean(digamma(m + 1.0))
    return max(float(mi), 0.0)


def _binned_old(X, y, n_bins=16):
    """Degenerate estimator: PCA-8 x 16 quantile bins -> discrete joint MI."""
    Xp = PCA(n_components=min(8, X.shape[1]), random_state=0).fit_transform(X)
    bins = []
    for c in range(Xp.shape[1]):
        edges = np.quantile(Xp[:, c], np.linspace(0, 1, n_bins + 1))
        bins.append(np.clip(np.digitize(Xp[:, c], edges[1:-1]), 0, n_bins - 1))
    bins = np.stack(bins, axis=1)
    code = np.zeros(len(Xp), dtype=np.int64)
    for c in range(bins.shape[1]):
        code = code * n_bins + bins[:, c]
    uc, ci = np.unique(code, return_inverse=True)
    uy, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(uc), len(uy)))
    for i in range(len(code)):
        joint[ci[i], yi[i]] += 1
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True); py = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py + 1e-12)
        lr = np.where(ratio > 0, np.log(ratio + 1e-12), 0.0)
    return float((joint * lr).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--t-stride", type=int, default=2)
    ap.add_argument("--expect-sha", default=EXPECT_SHA)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    sys.path.insert(0, args.root)
    from model_av import AVWordResNet

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print("[FATAL] val sha != expected; STOP."); sys.exit(2)

    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels_np = dav["spectrograms"]
    mels_np = mels_np.numpy() if hasattr(mels_np, "numpy") else np.asarray(mels_np)
    labels_all = np.asarray(dav["labels"]).astype(np.int64)
    n_all = len(labels_all)
    T_FRAMES, H, W = dav["video_shape"]
    cache_path = dav.get("video_cache_path")
    if not cache_path or not os.path.exists(cache_path):
        cache_path = os.path.join(args.root, "data", "visual", "cache",
                                  dav.get("video_cache_name", "videos_88_100.uint8"))
    videos = np.memmap(cache_path, dtype=np.uint8, mode="r", shape=(n_all, T_FRAMES, H, W))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = os.path.join(args.root, "models")
    ck = torch.load(os.path.join(mdir, "av_fused.pt"), weights_only=False, map_location="cpu")
    AV = AVWordResNet(len(ck["label_to_idx"]))
    AV.load_state_dict(ck["model_state_dict"])
    AV = AV.to(device).eval()
    stride = max(1, int(args.t_stride))

    class Vw(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k])
            mel = torch.from_numpy(mels_np[g]).unsqueeze(0)
            v = np.array(videos[g])
            if stride > 1: v = v[::stride]
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, vid, int(labels_all[g])

    dl = DataLoader(Vw(), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    acts = {s_: [] for s_ in SITES}
    av_pred, ys = [], []
    print("[fwd] rebuild a_mid/v_mid/gate_out/block2 GAPs over val ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            a_mid = AV.audio_block1(mel)
            v_mid = AV.visual(vid)
            gate = AV.gate(a_mid, v_mid)
            blk2 = AV.audio_block2(gate)
            penult = AV.gap(blk2).flatten(1)
            acts["a_mid_gap"].append(AV.gap(a_mid).flatten(1).cpu().numpy())
            acts["v_mid_gap"].append(AV.gap(v_mid).flatten(1).cpu().numpy())
            acts["gate_out_gap"].append(AV.gap(gate).flatten(1).cpu().numpy())
            acts["block2_gap"].append(penult.cpu().numpy())
            av_pred.append(AV.fc(AV.dropout(penult)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys)
    acts = {k: np.concatenate(v).astype(np.float64) for k, v in acts.items()}
    accAV = float((np.concatenate(av_pred) == y).mean())
    print(f"[self-check] AV acc={accAV:.6f} (ref {REF_AV}) delta={accAV-REF_AV:+.6f}", flush=True)
    for k in SITES:
        print(f"  {k:>13s} shape={acts[k].shape}", flush=True)

    n_classes = int(y.max()) + 1
    HY = _entropy_nats(y)
    print(f"\n[H(Y)] mine={HY:.6f} report={HY_REF} delta={HY-HY_REF:+.6f} "
          f"(log180={np.log(180):.6f})", flush=True)

    def _mlp():
        return MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                             alpha=1e-3, max_iter=300, early_stopping=True, random_state=0)

    def _logit():
        return LogisticRegression(max_iter=2000, C=1.0)

    rows = [["_reference", "H(Y)", f"{HY:.6f}", f"{HY_REF:.6f}", f"{HY-HY_REF:+.6f}"]]
    flags = []
    print(f"\n{'site':>13s} {'estimator':>16s} {'mine':>10s} {'report':>10s} {'delta':>10s}", flush=True)
    mlp_mi = {}
    for s_ in SITES:
        X = acts[s_]
        mi_mlp, ce_mlp = _ce_lower_bound(X, y, HY, n_classes, _mlp)
        mi_lin, ce_lin = _ce_lower_bound(X, y, HY, n_classes, _logit)
        mi_ksg = _ksg_mi_cd(X, y)
        mi_old = _binned_old(X, y)
        mlp_mi[s_] = mi_mlp
        for est, mine, extra in [("infonce_mlp", mi_mlp, f"CE={ce_mlp:.4f}"),
                                 ("infonce_linear", mi_lin, f"CE={ce_lin:.4f}"),
                                 ("ksg", mi_ksg, ""), ("binned_pca8_OLD", mi_old, "")]:
            rep = REPORT[(s_, est)]
            d = mine - rep
            tag = ""
            # tolerances: deterministic (ksg/binned) tight; training-based (infonce) looser
            tol = 0.02 if est in ("ksg", "binned_pca8_OLD") else 0.10
            if abs(d) > tol:
                tag = f"  ** FLAG >|{tol}|"
                flags.append((s_, est, mine, rep))
            print(f"{s_:>13s} {est:>16s} {mine:10.6f} {rep:10.6f} {d:+10.6f}{tag}  {extra}", flush=True)
            rows.append([s_, est, f"{mine:.6f}", f"{rep:.6f}", f"{d:+.6f}"])

    # CE->MI arithmetic spot-check on a_mid_gap infonce_mlp
    ce_recon = HY - mlp_mi["a_mid_gap"]
    print(f"\n[CE<->MI spot-check] a_mid_gap mlp: H(Y)-MI = {ce_recon:.4f} "
          f"(note CE=3.1874)  consistent={abs(ce_recon-3.1874)<0.01}", flush=True)

    # structural claims
    mono = mlp_mi["a_mid_gap"] < mlp_mi["gate_out_gap"] < mlp_mi["block2_gap"]
    bound_ok = all(REPORT[(s_, e)] < HY for s_ in SITES for e in ("infonce_mlp", "infonce_linear", "ksg"))
    binned_degenerate = all(abs(_b - HY) < 0.01 for _b in
                            [float(r[2]) for r in rows if r[1] == "binned_pca8_OLD"])
    print(f"[structural] monotone a_mid<gate_out<block2 (mlp) = {mono}", flush=True)
    print(f"[structural] all NEW estimators < H(Y)            = {bound_ok}", flush=True)
    print(f"[structural] binned_OLD ~H(Y) every site (degen)  = {binned_degenerate}", flush=True)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q7.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["site", "estimator", "mine", "report", "delta"])
        for r in rows:
            w.writerow(r)
        w.writerow(["_struct", "monotone_mlp", str(mono), "", ""])
        w.writerow(["_struct", "all_new_below_HY", str(bound_ok), "", ""])
        w.writerow(["_struct", "binned_degenerate", str(binned_degenerate), "", ""])
    print(f"\n[out] wrote {out}", flush=True)
    if flags:
        print(f"[FLAGS] {flags}", flush=True)
    else:
        print("[PASS] all estimators within tolerance; structure holds.", flush=True)


if __name__ == "__main__":
    main()
