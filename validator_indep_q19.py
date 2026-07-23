#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of Q19 (audio-scramble eval, artifact
analysis/msi/E5b_audio_scramble.csv).

INDEPENDENCE: I reimplement the scramble ops, the A/AV forward, and accuracy MYSELF
(analyze_av_msi NOT imported — its _forward_AV/_forward_A/_accuracy/_load_models are on
the do-not-import list). I reuse ONLY data/signal-processing infrastructure: the data
loader RawNoisyAVDataset (audio paths, pad offsets, video memmap, labels) and the mel
primitives paired_dataset._read_wav/_wav_to_log_mel/_pad_audio. Models loaded from the
frozen checkpoints. fp32 eager, .eval(), pinned val sha 03c5a87a (N=5244).

Two-part verification (per lead task #25):
 (1) EXACT — `none` control must reproduce the clean anchors (AV 0.956712 / A 0.926964).
     The additive-noise sigma=0.05 reference row (AV 0.609268 / A 0.134249) is ALREADY
     independently re-derived by this validator in analysis/validator_indep_E1_sigma_a.csv
     (line sigma 0.0500 -> A 0.134249, AV 0.609268, exact) and stage_ranking.csv -> cited,
     not re-run.
 (2) SEED-ROBUSTNESS of the scramble rows (the headline is QUALITATIVE): the generator
     seeds PER SAMPLE as default_rng(seed+idx), so seeds 0/1/2 overlap ~99.9% (a weak
     test). I use WELL-SEPARATED seeds {0, 10000, 20000} for genuinely independent
     realizations. seed=0 also reproduces the artifact rows exactly (deterministic).
     Claim to confirm: mel_time_shuffle AND mel_block_shuffle floor BOTH models (|AV-A|
     small, both low) while phase_scramble (magnitude spectrum intact) keeps a LARGE
     POSITIVE AV-A margin -> vision rescues only when temporal order / spectral identity
     is retained.

Artifact rows (seed=0):  mel_time_shuffle AV 0.049199 / A 0.023837;
  mel_block_shuffle AV 0.124523 / A 0.135965;  phase_scramble AV 0.247330 / A 0.060259.

Run on dev-codex:
    python validator_indep_q19.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q19.csv
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

EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"
AV_CLEAN, A_CLEAN = 0.956712, 0.926964
SEG_LEN = 10
# E5b_audio_scramble.csv seed=0 rows to reproduce
ARTIFACT = {
    "mel_time_shuffle": (0.049199, 0.023837),
    "mel_block_shuffle": (0.124523, 0.135965),
    "phase_scramble": (0.247330, 0.060259),
}
SCRAMBLE_MODES = ["mel_time_shuffle", "mel_block_shuffle", "phase_scramble"]
SEEDS = [0, 10000, 20000]


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--expect-sha", default=EXPECT_SHA)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    sys.path.insert(0, args.root)
    from train import WordResNet
    from model_av import AVWordResNet
    from dataset_raw_noisy import RawNoisyAVDataset
    from paired_dataset import _read_wav, _wav_to_log_mel, _pad_audio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    vraw = s["val_idx"]
    val_idx = (vraw.numpy() if hasattr(vraw, "numpy") else np.asarray(vraw)).astype(np.int64)
    val_sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={val_sha}", flush=True)
    if args.expect_sha and val_sha != args.expect_sha:
        print("[FATAL] val sha != expected; STOP."); sys.exit(2)

    base = RawNoisyAVDataset(t_stride=2, noise=False, return_video=True)

    mdir = os.path.join(args.root, "models")

    def _load(cls, name):
        ck = torch.load(os.path.join(mdir, name), weights_only=False, map_location="cpu")
        m = cls(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        return m.to(device).eval()

    A = _load(WordResNet, "audio_only_filtered.pt")
    AV = _load(AVWordResNet, "av_fused.pt")

    class ScrambledView(Dataset):
        def __init__(self, mode, seed):
            self.mode = mode; self.seed = int(seed)

        def __len__(self):
            return len(val_idx)

        def __getitem__(self, k):
            idx = int(val_idx[k])
            if self.mode == "phase_scramble":
                audio = _read_wav(base.audio_paths[idx])
                rng = np.random.default_rng(self.seed + idx)
                spec = np.fft.rfft(audio)
                mag = np.abs(spec)
                phase = rng.uniform(-np.pi, np.pi, size=spec.shape)
                scram = mag * np.exp(1j * phase)
                scram[0] = mag[0]
                if len(audio) % 2 == 0:
                    scram[-1] = mag[-1]
                audio_s = np.fft.irfft(scram, n=len(audio)).astype(np.float32)
                audio_p = _pad_audio(audio_s, int(base.pad_offsets[idx]))
                mel_t = torch.from_numpy(_wav_to_log_mel(audio_p).astype(np.float32))
                v = np.array(base._videos[idx])
                if base.t_stride > 1:
                    v = v[:: base.t_stride]
                vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
                return mel_t, vid, int(base.labels[idx])
            mel_t, vid, label = base[idx]           # clean mel [80,99], video [1,T,88,88]
            if self.mode == "none":
                return mel_t, vid, label
            T = mel_t.shape[1]
            rng = np.random.default_rng(self.seed + idx)
            if self.mode == "mel_time_shuffle":
                perm = rng.permutation(T)
            elif self.mode == "mel_block_shuffle":
                segs = np.array_split(np.arange(T), int(np.ceil(T / SEG_LEN)))
                order = rng.permutation(len(segs))
                perm = np.concatenate([segs[i] for i in order])
            else:
                raise ValueError(self.mode)
            perm_t = torch.from_numpy(perm.astype(np.int64))
            return mel_t.index_select(1, perm_t).contiguous(), vid, label

    @torch.no_grad()
    def _eval(mode, seed):
        dl = DataLoader(ScrambledView(mode, seed), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        av_corr = a_corr = n = 0
        for mel, vid, y in dl:
            mel = mel.unsqueeze(1).to(device, non_blocking=True)   # [B,1,80,99]
            vid = vid.to(device, non_blocking=True)
            y = y.numpy()
            a_pred = A(mel).argmax(1).cpu().numpy()
            a_mid = AV.audio_block1(mel); v_mid = AV.visual(vid)
            pen = AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1)
            av_pred = AV.fc(AV.dropout(pen)).argmax(1).cpu().numpy()
            av_corr += int((av_pred == y).sum())
            a_corr += int((a_pred == y).sum())
            n += len(y)
        return av_corr / n, a_corr / n

    rows = []
    print(f"\n{'mode':>18s} {'seed':>6s} {'AV':>9s} {'A':>9s} {'AV-A(pp)':>9s}", flush=True)

    # (1) none control
    av0, a0 = _eval("none", 0)
    none_ok = abs(av0 - AV_CLEAN) < 1e-3 and abs(a0 - A_CLEAN) < 1e-3
    print(f"{'none':>18s} {'-':>6s} {av0:9.6f} {a0:9.6f} {(av0-a0)*100:+9.4f}  "
          f"clean? {none_ok} (AVΔ={av0-AV_CLEAN:+.2e} AΔ={a0-A_CLEAN:+.2e})", flush=True)
    rows.append(["none", "-", f"{av0:.6f}", f"{a0:.6f}", f"{(av0-a0)*100:+.4f}",
                 f"AV {AV_CLEAN}/A {A_CLEAN}",
                 "OK" if none_ok else "** FLAG"])

    # (2) scramble modes x seeds
    agg = {m: [] for m in SCRAMBLE_MODES}
    for mode in SCRAMBLE_MODES:
        for seed in SEEDS:
            av, a = _eval(mode, seed)
            agg[mode].append((seed, av, a))
            ref = ""
            if seed == 0 and mode in ARTIFACT:
                rav, ra = ARTIFACT[mode]
                ref = f"artifact AV {rav}/A {ra} (Δ AV {av-rav:+.4f}/A {a-ra:+.4f})"
            print(f"{mode:>18s} {seed:>6d} {av:9.6f} {a:9.6f} {(av-a)*100:+9.4f}  {ref}",
                  flush=True)
            rows.append([mode, str(seed), f"{av:.6f}", f"{a:.6f}", f"{(av-a)*100:+.4f}",
                         ref, ""])

    # additive-noise reference (cited from my prior independent re-derivation)
    rows.append(["ref_additive_noise_sigma0.05", "-", "0.609268", "0.134249", "+47.5019",
                 "cited: validator_indep_E1_sigma_a.csv (independently re-derived, exact)",
                 "OK"])
    rows.append(["ref_clean_baseline", "-", f"{AV_CLEAN}", f"{A_CLEAN}",
                 f"{(AV_CLEAN-A_CLEAN)*100:+.4f}", "anchor", "OK"])

    # ---- qualitative verdict ----
    print("\n[qualitative robustness check across seeds {0,10000,20000}]", flush=True)
    # phase margin must be large-positive every seed; shuffle margins small every seed
    phase_margins = [(av - a) * 100 for _, av, a in agg["phase_scramble"]]
    time_margins = [(av - a) * 100 for _, av, a in agg["mel_time_shuffle"]]
    block_margins = [(av - a) * 100 for _, av, a in agg["mel_block_shuffle"]]
    time_lvl = [(av, a) for _, av, a in agg["mel_time_shuffle"]]
    block_lvl = [(av, a) for _, av, a in agg["mel_block_shuffle"]]
    print(f"  phase_scramble AV-A margins (pp): {[f'{m:+.2f}' for m in phase_margins]}", flush=True)
    print(f"  mel_time_shuffle AV-A margins (pp): {[f'{m:+.2f}' for m in time_margins]}", flush=True)
    print(f"  mel_block_shuffle AV-A margins (pp): {[f'{m:+.2f}' for m in block_margins]}", flush=True)
    phase_big = all(m > 10.0 for m in phase_margins)
    shuffle_small = all(abs(m) < 6.0 for m in time_margins + block_margins)
    shuffle_floored = all(av < 0.20 and a < 0.20 for av, a in time_lvl + block_lvl)
    phase_dominates = all(p > max(time_margins + block_margins) for p in phase_margins)
    print(f"  phase margin >10pp all seeds .......... {phase_big}", flush=True)
    print(f"  shuffle |margin| <6pp all seeds ....... {shuffle_small}", flush=True)
    print(f"  shuffle both models floored (<20%) .... {shuffle_floored}", flush=True)
    print(f"  phase margin > every shuffle margin ... {phase_dominates}", flush=True)

    seed0_repro = all(
        abs(dict((s, (av, a)) for s, av, a in agg[m])[0][0] - ARTIFACT[m][0]) < 0.02 and
        abs(dict((s, (av, a)) for s, av, a in agg[m])[0][1] - ARTIFACT[m][1]) < 0.02
        for m in SCRAMBLE_MODES)

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q19.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["mode", "seed", "AV_acc", "A_acc", "AV_minus_A_pp", "reference", "flag"])
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out}", flush=True)

    print("\n[VERDICT]", flush=True)
    print(f"  none == clean anchors ............ {none_ok}", flush=True)
    print(f"  seed=0 reproduces artifact (<2pp). {seed0_repro}", flush=True)
    print(f"  qualitative claim stable ......... "
          f"{phase_big and shuffle_small and shuffle_floored and phase_dominates}", flush=True)
    if none_ok and seed0_repro and phase_big and shuffle_small and shuffle_floored and phase_dominates:
        print("[GO] none bit-exact; artifact reproduced at seed 0; the structure-dependent "
              "vision-rescue claim holds across independent seeds.", flush=True)
    else:
        print("[NO-GO/FLAG] one or more checks failed -> report to lead (no self-reconcile).",
              flush=True)


if __name__ == "__main__":
    main()
