#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of Q17 (A-vs-AV prediction overlap on clean val)
+ independent re-derivation of the cited McGurk E3 capture rates.

Q17 core is DETERMINISTIC (argmax + counting) -> expect BIT-EXACT (the >=0.5% LR-probe
guardrail does NOT apply; any non-trivial count delta is a real discrepancy -> flag to
lead, no self-reconcile). I forward audio_only_filtered.pt (A) and av_fused.pt (AV)
MYSELF on the pinned val (sha 03c5a87a, N=5244), eager fp32, .eval() — deepdive_act_cache
NOT loaded — and rebuild the full contingency.

McGurk E3 (the report's hardcoded cross-ref, source analysis/msi/E3_mcgurk_capture_rates.csv,
generator analyze_av_msi.E3_mcgurk lines 339-466): I REIMPLEMENT the seeded pairing
(_build_mcgurk_pairs: default_rng(0), bucket by (speaker,group), per-bucket shuffle,
viseme-distinct partner, cap 1500) and the capture counts — NOT imported. Target distinct
n=1500: A_only_audio_capture 0.9273, AV_third_word 0.8273 (also AV_audio_capture 0.0780,
AV_visual_capture 0.0947). Reused taxonomer only: analyze_av_phonetics.viseme_class.

GATE: contingency counts bit-exact (abs delta 0); McGurk rates within 1e-3 of the E3 CSV
(re-derived independently; pairing is seed-deterministic so exact reproduction expected).
Self-check A 0.926964 / AV 0.956712.

Run on dev-codex:
    python validator_indep_q17.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q17.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
REF = {"A": 0.926964, "AV": 0.956712}

# Q17 CSV (the counts I must reproduce bit-exact)
REPORT = {
    "agreement_n": 4814, "disagreement_n": 430,
    "both_correct": 4756, "both_wrong": 122,
    "regression": 105, "rescue": 261, "shared_error": 58,
    "av_on_a_wrong_num": 261, "av_on_a_wrong_den": 383,
    "a_on_av_wrong_num": 105, "a_on_av_wrong_den": 227,
    "dis_a_right": 105, "dis_av_right": 261, "dis_neither": 64,
}
# McGurk E3 distinct-pair targets (E3_mcgurk_capture_rates.csv line 2)
MCG = {"n_pairs": 1500, "AV_audio_capture": 0.0780, "AV_visual_capture": 0.0947,
       "AV_third_word": 0.8273, "A_only_audio_capture": 0.9273}


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


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
    from analyze_av_phonetics import viseme_class

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
    speakers = dav.get("speakers", None)
    groups = dav.get("groups", None)
    cache_path = dav.get("video_cache_path")
    if not cache_path or not os.path.exists(cache_path):
        cache_path = os.path.join(args.root, "data", "visual", "cache",
                                  dav.get("video_cache_name", "videos_88_100.uint8"))
    videos = np.memmap(cache_path, dtype=np.uint8, mode="r", shape=(n_all, T_FRAMES, H, W))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = os.path.join(args.root, "models")

    def _load(cls, name):
        ck = torch.load(os.path.join(mdir, name), weights_only=False, map_location="cpu")
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        return m.to(device).eval(), ck

    A, _ = _load(WordResNet, "audio_only_filtered.pt")
    AV, av_ck = _load(AVWordResNet, "av_fused.pt")
    idx_to_label = av_ck["idx_to_label"]
    stride = max(1, int(args.t_stride))

    def _prep_video(g):
        v = np.array(videos[g])
        if stride > 1:
            v = v[::stride]
        return torch.from_numpy(v).unsqueeze(0).float() / 255.0

    def _prep_mel(g):
        return torch.from_numpy(mels_np[g]).unsqueeze(0)

    # ---- 1) clean-val contingency: forward A + AV on val_idx ----
    class Vw(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k])
            return _prep_mel(g), _prep_video(g), int(labels_all[g])

    dl = DataLoader(Vw(), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    a_pred, av_pred, ys = [], [], []
    print("[fwd] clean-val A + AV argmax ...", flush=True)
    with torch.no_grad():
        for mel, vid, y in dl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            a_pred.append(A(mel).argmax(1).cpu().numpy())
            a_mid = AV.audio_block1(mel); v_mid = AV.visual(vid)
            pen = AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)
            av_pred.append(AV.fc(AV.dropout(pen)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys).astype(np.int64)
    a_pred = np.concatenate(a_pred); av_pred = np.concatenate(av_pred)
    a_ok = a_pred == y; av_ok = av_pred == y
    accA = float(a_ok.mean()); accAV = float(av_ok.mean())
    print(f"[self-check] A={accA:.6f} (ref {REF['A']}) AV={accAV:.6f} (ref {REF['AV']})",
          flush=True)
    sc_ok = abs(accA - REF["A"]) < 5e-4 and abs(accAV - REF["AV"]) < 5e-4

    agree = a_pred == av_pred
    got = {
        "agreement_n": int(agree.sum()), "disagreement_n": int((~agree).sum()),
        "both_correct": int((a_ok & av_ok).sum()), "both_wrong": int((~a_ok & ~av_ok).sum()),
        "regression": int((a_ok & ~av_ok).sum()), "rescue": int((~a_ok & av_ok).sum()),
        "shared_error": int((agree & ~a_ok).sum()),
        "av_on_a_wrong_num": int(av_ok[~a_ok].sum()), "av_on_a_wrong_den": int((~a_ok).sum()),
        "a_on_av_wrong_num": int(a_ok[~av_ok].sum()), "a_on_av_wrong_den": int((~av_ok).sum()),
        "dis_a_right": int((~agree & a_ok).sum()), "dis_av_right": int((~agree & av_ok).sum()),
        "dis_neither": int((~agree & ~a_ok & ~av_ok).sum()),
    }
    rows, flags = [], []
    print("\n[contingency] independent forward vs Q17 CSV (bit-exact expected):", flush=True)
    for k in REPORT:
        d = got[k] - REPORT[k]
        tag = "" if d == 0 else "  ** FLAG"
        if d != 0:
            flags.append((k, got[k], REPORT[k]))
        print(f"  {k:>22s}: mine={got[k]:>6d}  report={REPORT[k]:>6d}  Δ={d:+d}{tag}",
              flush=True)
        rows.append(["contingency", k, str(got[k]), str(REPORT[k]), f"{d:+d}"])
    # sanity partition checks
    part1 = got["both_correct"] + got["both_wrong"] + got["regression"] + got["rescue"]
    part2 = got["agreement_n"] + got["disagreement_n"]
    print(f"[partition] bc+bw+reg+res={part1} (=5244? {part1 == 5244}); "
          f"agree+disagree={part2} (=5244? {part2 == 5244})", flush=True)

    # ---- 2) McGurk E3: reimplement the seeded pairing + capture ----
    print("\n[McGurk E3] independent reimpl of _build_mcgurk_pairs + capture", flush=True)
    rng = np.random.default_rng(0)
    spk = speakers if speakers is not None else [None] * n_all
    grp = groups if groups is not None else [None] * n_all
    by_sg = defaultdict(list)
    for i in val_idx:
        i = int(i)
        by_sg[(spk[i] if speakers is not None else "?",
               grp[i] if groups is not None else 0)].append(i)
    distinct = []
    for (_s, _g), items in by_sg.items():
        items = list(items)
        rng.shuffle(items)
        if len(items) < 2:
            continue
        for i in items:
            vis_i = viseme_class(idx_to_label[int(labels_all[i])])
            for j in items:
                if j == i:
                    continue
                vis_j = viseme_class(idx_to_label[int(labels_all[j])])
                if vis_j != vis_i and vis_j != "other" and vis_i != "other":
                    distinct.append((i, j))
                    break
    distinct = distinct[:1500]
    print(f"  distinct pairs built: {len(distinct)}", flush=True)

    class McGurkView(Dataset):
        def __len__(self): return len(distinct)
        def __getitem__(self, k):
            i, j = distinct[k]
            return _prep_mel(i), _prep_video(j), int(labels_all[i]), int(labels_all[j])

    mdl = DataLoader(McGurkView(), batch_size=args.batch, shuffle=False,
                     num_workers=args.workers, pin_memory=True)
    audio_caps = vis_caps = third = total = a_caps_a = 0
    with torch.no_grad():
        for mel, vid, yi, yj in mdl:
            mel = mel.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            yi = yi.numpy(); yj = yj.numpy()
            a_mid = AV.audio_block1(mel); v_mid = AV.visual(vid)
            pen = AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)
            pred = AV.fc(AV.dropout(pen)).argmax(1).cpu().numpy()
            pa = A(mel).argmax(1).cpu().numpy()
            for k in range(len(pred)):
                total += 1
                if pred[k] == yi[k]:
                    audio_caps += 1
                elif pred[k] == yj[k]:
                    vis_caps += 1
                else:
                    third += 1
                if pa[k] == yi[k]:
                    a_caps_a += 1
    tot = max(1, total)
    mcg_got = {"n_pairs": total, "AV_audio_capture": audio_caps / tot,
               "AV_visual_capture": vis_caps / tot, "AV_third_word": third / tot,
               "A_only_audio_capture": a_caps_a / tot}
    for k in ["n_pairs", "AV_audio_capture", "AV_visual_capture", "AV_third_word",
              "A_only_audio_capture"]:
        mine, ref = mcg_got[k], MCG[k]
        if k == "n_pairs":
            d = mine - ref; tag = "" if d == 0 else "  ** FLAG"; ds = f"{d:+d}"
            if d != 0: flags.append((k, mine, ref))
        else:
            d = mine - ref; tag = "" if abs(d) <= 1e-3 else "  ** FLAG(>1e-3)"
            ds = f"{d:+.4f}"
            if abs(d) > 1e-3: flags.append((k, round(mine, 4), ref))
        print(f"  {k:>22s}: mine={mine if k=='n_pairs' else f'{mine:.4f}'}  "
              f"E3={ref}  Δ={ds}{tag}", flush=True)
        rows.append(["mcgurk_E3", k, (str(mine) if k == "n_pairs" else f"{mine:.4f}"),
                     str(ref), ds])

    rows.append(["selfcheck", "A/AV_acc", f"{accA:.6f}/{accAV:.6f}",
                 f"{REF['A']}/{REF['AV']}", "OK" if sc_ok else "** FLAG"])

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q17.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "metric", "value", "report", "delta"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)

    cont_exact = all(got[k] == REPORT[k] for k in REPORT)
    print("\n[VERDICT]", flush=True)
    print(f"  self-check ............... {'OK' if sc_ok else 'FAIL'}", flush=True)
    print(f"  contingency bit-exact .... {cont_exact}", flush=True)
    print(f"  McGurk E3 re-derived ..... distinct n={total} "
          f"A_cap={mcg_got['A_only_audio_capture']:.4f} third={mcg_got['AV_third_word']:.4f}",
          flush=True)
    if cont_exact and sc_ok and not flags:
        print("[GO] Q17 contingency bit-exact; McGurk E3 independently reproduced.", flush=True)
    else:
        print(f"[NO-GO/FLAG] flags={flags} (sc_ok={sc_ok} cont_exact={cont_exact}) -> "
              f"report to lead (no self-reconcile).", flush=True)


if __name__ == "__main__":
    main()
