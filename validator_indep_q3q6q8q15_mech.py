#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the av_fused GATE-MECHANISM numbers
(Q3 / Q6 / Q8 / Q15) in MODALITY_INTEGRATION_REPORT.md. ONE forward pass over the
pinned clean val; analyze_av_msi / phase_d_saliency / phase_f_flow NOT imported —
reimplemented from the AV submodules (audio_block1, visual, gate, audio_block2, gap, fc)
+ sklearn primitives.

Covers:
  Q6/Q8  E4 MEI super-additivity: per site R=mean|act| over (N, spatial); SA frac =
         mean_channels(R_AV > R_A + R_V). R_AV=real/real, R_A=video-zero, R_V=audio-zero.
         Anchor: block2 frac_super_additive 0.2656, a_mid 0.0 (pre-gate control).
  Q8     E9 gate magnitude: gate activation g = a_fused - a_mid; gate_mean_abs =
         mean|g|. AV_full (real/real) 0.2745 ; audio_only (video-zero) 0.5377 ; α 5.202385.
  Q6     α=0 collapse: gate(α=0) => a_fused == a_mid analytically => penult =
         gap(audio_block2(a_mid)). Anchor AV acc 1.49%.
  Q15    D5.8 block2 lesion: zeroing block2 channel c before gap == zeroing penult dim c
         (gap is linear). Per-channel acc drop; anchor max +0.2479 (ch78).
  Q15    D5.11 v-modulation: per block2 channel, in-sample Ridge(alpha=1) R² of
         block2_gap(128) on v_mid_gap(64); count R²>0.30. Anchor 107/128.
  Q3     D4 no-scaler 5-fold word linprobe on AV penult with audio zeroed (mel:=0,
         video real). Anchor AV_audio_zero 0.673913.

Self-checks (bit-exact / guardrail): AV 0.956712, A 0.926964, V_fair 0.864989.

Run on dev-codex:
    python validator_indep_q3q6q8q15_mech.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q3q6q8q15_mech.csv
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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
SITES = ["a_mid", "v_mid", "gate", "block2", "penult"]
ANCHOR = {
    "mei_block2_sa": 0.2656, "mei_amid_sa": 0.0,
    "e9_av_full": 0.2745, "e9_audio_only": 0.5377, "alpha": 5.202385,
    "alpha0_acc": 0.0149, "block2_lesion_max_pp": 0.2479,
    "vmod_n_modulated": 107, "q3_audio_zero_word": 0.673913,
    "race_violations": 68, "race_frac": 0.012967,
    "AV": 0.956712, "A": 0.926964, "V": 0.864989,
}


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _linprobe_5fold(X, y, seed=0):
    """D4 protocol: StratifiedKFold(5, shuffle, rs=0), NO scaler, LR(max_iter=2000,C=1)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


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
    from train import WordResNet
    from model_av import AVWordResNet
    from model_v_only_fair import VOnlyFairWordResNet

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print("[FATAL] val sha mismatch"); sys.exit(2)

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
    stride = max(1, int(args.t_stride))

    def _load(cls, name):
        ck = torch.load(os.path.join(mdir, name), weights_only=False, map_location="cpu")
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        return m.to(device).eval()

    AV = _load(AVWordResNet, "av_fused.pt")
    A = _load(WordResNet, "audio_only_filtered.pt")
    V = _load(VOnlyFairWordResNet, "video_only_fair.pt")
    alpha = float(AV.gate.alpha.detach().item())
    print(f"[alpha] gate.alpha = {alpha:.6f} (anchor {ANCHOR['alpha']})", flush=True)

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

    # MEI accumulators: per condition (AV/A/V) per site -> sum|act| (per channel) + count
    sums = {c: {s: None for s in SITES} for c in ("AV", "A", "V")}
    counts = {c: {s: 0 for s in SITES} for c in ("AV", "A", "V")}
    # E9 gate-magnitude accumulators
    e9_sum = {"AV_full": 0.0, "audio_only": 0.0}
    e9_cnt = {"AV_full": 0, "audio_only": 0}

    pen_rr, vmid_gap, pen_az = [], [], []
    preds_av, preds_a0, preds_a, preds_v, ys = [], [], [], [], []

    def _acc_site(cond, site, act):
        a = np.abs(act).reshape(act.shape[0], act.shape[1], -1)  # (B,C,*)
        sm = a.sum(axis=(0, 2))                                  # (C,)
        sums[cond][site] = sm if sums[cond][site] is None else sums[cond][site] + sm
        counts[cond][site] += act.shape[0] * (a.shape[2])

    print("[fwd] single pass: AV(real/real, video-zero, audio-zero) + α0 + A + V_fair ...",
          flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)

            # ---- AV real/real ----
            a_mid = AV.audio_block1(mel)
            v_mid = AV.visual(vid)
            a_fused = AV.gate(a_mid, v_mid)
            gate_rr = a_fused - a_mid
            blk2 = AV.audio_block2(a_fused)
            pen = AV.gap(blk2).flatten(1)
            logits = AV.fc(pen)
            preds_av.append(logits.argmax(1).cpu().numpy())
            pen_rr.append(pen.cpu().numpy())
            vmid_gap.append(AV.gap(v_mid).flatten(1).cpu().numpy())
            for site, act in zip(SITES, (a_mid, v_mid, gate_rr, blk2, pen)):
                _acc_site("AV", site, act.cpu().numpy())
            g_rr = gate_rr.cpu().numpy()
            e9_sum["AV_full"] += float(np.abs(g_rr).sum()); e9_cnt["AV_full"] += g_rr.size

            # ---- α=0 (analytic: a_fused == a_mid) ----
            pen_a0 = AV.gap(AV.audio_block2(a_mid)).flatten(1)
            preds_a0.append(AV.fc(pen_a0).argmax(1).cpu().numpy())

            # ---- AV video-zero (v_mid := 0) ----
            v0 = torch.zeros_like(v_mid)
            a_fused0 = AV.gate(a_mid, v0)
            gate_vz = a_fused0 - a_mid
            blk2_0 = AV.audio_block2(a_fused0)
            pen0 = AV.gap(blk2_0).flatten(1)
            for site, act in zip(SITES, (a_mid, v0, gate_vz, blk2_0, pen0)):
                _acc_site("A", site, act.cpu().numpy())
            g_vz = gate_vz.cpu().numpy()
            e9_sum["audio_only"] += float(np.abs(g_vz).sum()); e9_cnt["audio_only"] += g_vz.size

            # ---- AV audio-zero (mel := 0) ----
            a_mid_z = AV.audio_block1(torch.zeros_like(mel))
            a_fused_z = AV.gate(a_mid_z, v_mid)
            gate_az = a_fused_z - a_mid_z
            blk2_z = AV.audio_block2(a_fused_z)
            pen_z = AV.gap(blk2_z).flatten(1)
            pen_az.append(pen_z.cpu().numpy())
            for site, act in zip(SITES, (a_mid_z, v_mid, gate_az, blk2_z, pen_z)):
                _acc_site("V", site, act.cpu().numpy())

            # ---- A specialist ----
            preds_a.append(A.fc(A.gap(A.block2(A.block1(mel))).flatten(1)).argmax(1).cpu().numpy())
            # ---- V_fair specialist ----
            preds_v.append(V.fc(V.gap(V.block2(V.visual(vid))).flatten(1)).argmax(1).cpu().numpy())
            ys.append(y.numpy())

    y = np.concatenate(ys).astype(np.int64)
    pen_rr = np.concatenate(pen_rr).astype(np.float64)
    pen_az = np.concatenate(pen_az).astype(np.float64)
    vmid_gap = np.concatenate(vmid_gap).astype(np.float64)
    preds_av = np.concatenate(preds_av); preds_a0 = np.concatenate(preds_a0)
    preds_a = np.concatenate(preds_a); preds_v = np.concatenate(preds_v)

    # ---- self-checks ----
    acc_av = float((preds_av == y).mean()); acc_a = float((preds_a == y).mean())
    acc_v = float((preds_v == y).mean()); acc_a0 = float((preds_a0 == y).mean())
    print(f"[self-check] AV={acc_av:.6f} A={acc_a:.6f} V={acc_v:.6f}", flush=True)
    sc_ok = (abs(acc_av - ANCHOR["AV"]) < 5e-4 and abs(acc_a - ANCHOR["A"]) < 5e-4
             and abs(acc_v - ANCHOR["V"]) < 5e-4)

    rows = []
    flags = []

    # ---- MEI super-additive ----
    R = {c: {s: sums[c][s] / max(counts[c][s], 1) for s in SITES} for c in ("AV", "A", "V")}
    print("\n[MEI] frac_super_additive (R_AV > R_A + R_V) per site:", flush=True)
    mei_sa = {}
    for site in SITES:
        sa = float((R["AV"][site] > (R["A"][site] + R["V"][site])).mean())
        mei_sa[site] = sa
        print(f"    {site:>7s} SA={sa:.4f}  (R_AV={R['AV'][site].mean():.3f} "
              f"R_A={R['A'][site].mean():.3f} R_V={R['V'][site].mean():.3f})", flush=True)
        rows.append(["MEI_frac_super_additive", site, f"{sa:.6f}",
                     f"{ANCHOR['mei_block2_sa']}" if site == "block2"
                     else (f"{ANCHOR['mei_amid_sa']}" if site == "a_mid" else "")])
    mei_b2_ok = abs(mei_sa["block2"] - ANCHOR["mei_block2_sa"]) <= 0.02
    mei_amid_ok = mei_sa["a_mid"] == 0.0
    if not mei_b2_ok: flags.append(("MEI block2 SA", mei_sa["block2"], ANCHOR["mei_block2_sa"]))
    if not mei_amid_ok: flags.append(("MEI a_mid SA", mei_sa["a_mid"], 0.0))

    # ---- E9 gate magnitude ----
    e9_av = e9_sum["AV_full"] / e9_cnt["AV_full"]
    e9_ao = e9_sum["audio_only"] / e9_cnt["audio_only"]
    print(f"\n[E9] gate_mean_abs AV_full={e9_av:.4f} (anchor {ANCHOR['e9_av_full']}) ; "
          f"audio_only={e9_ao:.4f} (anchor {ANCHOR['e9_audio_only']})", flush=True)
    e9_ok = (abs(e9_av - ANCHOR["e9_av_full"]) <= 0.01 and abs(e9_ao - ANCHOR["e9_audio_only"]) <= 0.01)
    if not e9_ok: flags.append(("E9 gate-mag", (e9_av, e9_ao), (ANCHOR["e9_av_full"], ANCHOR["e9_audio_only"])))
    rows += [["E9_gate_mean_abs", "AV_full", f"{e9_av:.6f}", f"{ANCHOR['e9_av_full']}"],
             ["E9_gate_mean_abs", "audio_only", f"{e9_ao:.6f}", f"{ANCHOR['e9_audio_only']}"],
             ["gate_alpha", "param", f"{alpha:.6f}", f"{ANCHOR['alpha']}"]]

    # ---- α=0 collapse ----
    print(f"\n[α=0] AV(α=0) acc={acc_a0:.6f} (anchor ~{ANCHOR['alpha0_acc']})", flush=True)
    a0_ok = abs(acc_a0 - ANCHOR["alpha0_acc"]) < 5e-3
    if not a0_ok: flags.append(("alpha0 acc", acc_a0, ANCHOR["alpha0_acc"]))
    rows.append(["alpha0_collapse", "AV_acc", f"{acc_a0:.6f}", f"{ANCHOR['alpha0_acc']}"])

    # ---- Q15 block2 lesion (zero penult dim c) ----
    fcW = AV.fc.weight.detach().cpu().numpy().astype(np.float64)  # (180,128)
    fcb = AV.fc.bias.detach().cpu().numpy().astype(np.float64)
    base_logits = pen_rr @ fcW.T + fcb
    base_acc = float((base_logits.argmax(1) == y).mean())
    deltas = np.zeros(pen_rr.shape[1])
    for c in range(pen_rr.shape[1]):
        col = pen_rr[:, c].copy()
        pen_rr[:, c] = 0.0
        acc_c = float(((pen_rr @ fcW.T + fcb).argmax(1) == y).mean())
        pen_rr[:, c] = col
        deltas[c] = (base_acc - acc_c) * 100.0
    ch_max = int(np.argmax(deltas)); les_max = float(deltas[ch_max])
    print(f"\n[Q15 block2-lesion] base_acc={base_acc:.6f} ; max drop ch{ch_max} "
          f"+{les_max:.4f}pp (anchor +{ANCHOR['block2_lesion_max_pp']} ch78)", flush=True)
    les_ok = abs(les_max - ANCHOR["block2_lesion_max_pp"]) <= 0.05
    if not les_ok: flags.append(("block2 lesion max", les_max, ANCHOR["block2_lesion_max_pp"]))
    rows.append(["block2_lesion_max", f"ch{ch_max}", f"{les_max:.4f}", f"{ANCHOR['block2_lesion_max_pp']}"])

    # ---- Q15 v-modulation ----
    r2 = np.zeros(pen_rr.shape[1])
    for c in range(pen_rr.shape[1]):
        yy = pen_rr[:, c]
        reg = Ridge(alpha=1.0).fit(vmid_gap, yy)
        yp = reg.predict(vmid_gap)
        ss_res = float(((yy - yp) ** 2).sum()); ss_tot = float(((yy - yy.mean()) ** 2).sum())
        r2[c] = 1.0 - ss_res / (ss_tot + 1e-12)
    n_mod = int((r2 > 0.30).sum())
    print(f"[Q15 v-modulation] R²>0.30 channels = {n_mod}/128 (anchor {ANCHOR['vmod_n_modulated']}); "
          f"median={np.median(r2):.3f} max={r2.max():.3f}", flush=True)
    vmod_ok = abs(n_mod - ANCHOR["vmod_n_modulated"]) <= 2
    if not vmod_ok: flags.append(("v-mod n_modulated", n_mod, ANCHOR["vmod_n_modulated"]))
    rows.append(["v_modulation", "n_R2_gt_0.30", str(n_mod), str(ANCHOR["vmod_n_modulated"])])

    # ---- Q8 race-bound ----
    p_a = (preds_a == y).astype(np.int32); p_v = (preds_v == y).astype(np.int32)
    p_av = (preds_av == y).astype(np.int32)
    bound = (p_a | p_v); viol = (p_av > bound).astype(np.int32)
    n_viol = int(viol.sum()); frac_viol = float(viol.mean())
    print(f"\n[Q8 race-bound] violations = {n_viol}/{len(y)} ({frac_viol:.6f}) "
          f"(anchor {ANCHOR['race_violations']} / {ANCHOR['race_frac']})", flush=True)
    race_ok = (n_viol == ANCHOR["race_violations"])
    if not race_ok: flags.append(("race violations", n_viol, ANCHOR["race_violations"]))
    rows.append(["race_bound", "n_violations", str(n_viol), str(ANCHOR["race_violations"])])
    rows.append(["race_bound", "frac_violations", f"{frac_viol:.6f}", f"{ANCHOR['race_frac']}"])

    # ---- Q3 audio-zero word linprobe ----
    print("\n[Q3] D4 no-scaler 5-fold word linprobe on AV penult (audio zeroed) ...", flush=True)
    q3 = _linprobe_5fold(pen_az, y, seed=0)
    rel = (q3 - ANCHOR["q3_audio_zero_word"]) / ANCHOR["q3_audio_zero_word"] * 100.0
    print(f"    AV_audio_zero word acc = {q3:.6f} (anchor {ANCHOR['q3_audio_zero_word']}, "
          f"rel {rel:+.3f}%)", flush=True)
    q3_ok = abs(rel) <= 0.5
    if not q3_ok: flags.append(("Q3 audio-zero word probe", q3, ANCHOR["q3_audio_zero_word"]))
    rows.append(["q3_audio_zero_word_probe", "AV", f"{q3:.6f}",
                 f"{ANCHOR['q3_audio_zero_word']} (rel {rel:+.3f}%)"])

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q3q6q8q15_mech.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["quantity", "key", "mine", "anchor"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)

    all_ok = (sc_ok and mei_b2_ok and mei_amid_ok and e9_ok and a0_ok and les_ok
              and vmod_ok and race_ok and q3_ok)
    print("\n[VERDICT]", flush=True)
    print(f"  self-check AV/A/V ........ {'OK' if sc_ok else 'FAIL'}", flush=True)
    print(f"  Q6/Q8 MEI block2 SA ...... {'OK' if mei_b2_ok else 'FLAG'} ({mei_sa['block2']:.4f})", flush=True)
    print(f"  Q6/Q8 MEI a_mid SA=0 ..... {'OK' if mei_amid_ok else 'FLAG'}", flush=True)
    print(f"  Q8 E9 gate magnitude ..... {'OK' if e9_ok else 'FLAG'}", flush=True)
    print(f"  Q6 α=0 collapse .......... {'OK' if a0_ok else 'FLAG'} ({acc_a0:.4f})", flush=True)
    print(f"  Q15 block2 lesion max .... {'OK' if les_ok else 'FLAG'} (+{les_max:.4f}pp ch{ch_max})", flush=True)
    print(f"  Q15 v-modulation ......... {'OK' if vmod_ok else 'FLAG'} ({n_mod}/128)", flush=True)
    print(f"  Q8 race-bound ............ {'OK' if race_ok else 'FLAG'} ({n_viol})", flush=True)
    print(f"  Q3 audio-zero word probe . {'OK' if q3_ok else 'FLAG'} ({q3:.4f})", flush=True)
    if all_ok:
        print("[GO] Q3/Q6/Q8/Q15 gate-mechanism numbers reproduced.", flush=True)
    else:
        print(f"[NO-GO/FLAG] flags={flags} -> report to lead (no self-reconcile).", flush=True)


if __name__ == "__main__":
    main()
