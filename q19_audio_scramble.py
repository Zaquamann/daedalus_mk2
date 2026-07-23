#!/usr/bin/env python3
"""Q19 — audio-scramble eval (the missing audio analog of the E5 video-perturbation
battery). ONE artifact: analysis/msi/E5b_audio_scramble.csv (columns mode,AV_acc,A_acc).

The MSI E5 set scrambles VIDEO only (_PerturbedAVView.PERT = none/time_shuffle/
freeze_t0/block_shuffle/random_video/zero_video); the audio mel is passed untouched in
every branch. The only existing audio-degradation evidence is ADDITIVE Gaussian noise,
which preserves temporal/spectral ORDER — it cannot answer "does the model need
coherently-ordered audio". This fills that gap with a new _ScrambledAudioView and ≥3
deterministic scramble modes, contrasting AV (clean video present) vs A-only.

EVAL ONLY (no retrain). EAGER fp32 forward — imports the VERBATIM forward helpers
(_forward_AV/_forward_A/_load_models/_accuracy) from analyze_av_msi.py so the 'none'
control reproduces the E5 clean anchor (AV 0.956712, A 0.926964) through the exact same
code path that produced it. Runs a single GPU inference pass (no_grad, batch 64) per
mode; shares the H200 with the Q14 training job (inference adds ~1-2 GB; no contention).

Modes (deterministic per index, rng = default_rng(seed+idx), seed=0):
  none              clean control — must reproduce clean within ~0.1pp.
  mel_time_shuffle  permute the 99 log-mel time frames (direct audio analog of E5 VIDEO
                    time_shuffle, which collapses AV to 1.11%). Destroys all temporal order.
  mel_block_shuffle partition the 99 frames into ~10-frame segments and permute SEGMENT
                    ORDER (keeps local phonetic structure, destroys global order). This is
                    the temporal analog of E5 time_shuffle's logic — NOT of E5's SPATIAL
                    8x8 pixel block_shuffle (do not conflate).
  phase_scramble    randomize the phase of the waveform's rFFT, keep the magnitude
                    spectrum, inverse-FFT, then compute the mel. Destroys temporal
                    structure, preserves the long-term power spectrum.

Pinned val sha 03c5a87a (N=5244, 180 classes, .eval(), seed 0). For AV the scrambled
audio is paired with CLEAN video so the contrast is "can vision rescue scrambled audio".
Reference rows place the results next to the clean baseline and the additive-noise sweep
at sigma=0.05 (A-only 0.134249, AV 0.609268) so structured-corruption robustness is
auditable against unstructured-noise robustness.
"""
import csv
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analyze_av_msi import (
    BATCH_SIZE,
    T_STRIDE,
    _accuracy,
    _forward_A,
    _forward_AV,
    _load_models,
)
from dataset_raw_noisy import RawNoisyAVDataset
from paired_dataset import _pad_audio, _read_wav, _wav_to_log_mel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "analysis", "msi", "E5b_audio_scramble.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
PIN = "03c5a87a"
# clean + matched additive-noise anchors (EVIDENCE_MAP H / noise-sweep CSVs)
AV_CLEAN, A_CLEAN = 0.956712, 0.926964
AV_NOISE_S05, A_NOISE_S05 = 0.609268, 0.134249   # additive Gaussian, sigma=0.05*rms
SEG_LEN = 10                                      # ~10-frame segments for block shuffle


class _ScrambledAudioView(Dataset):
    """Val partition with the audio stream deterministically SCRAMBLED (clean video).

    Modeled on _NoisyAudioView (raw-wav read + recomputed mel, for phase_scramble) and on
    the mode-enumeration pattern of _PerturbedAVView. Yields (mel[80,99], video[1,T,88,88],
    label) exactly like the clean views, so it drops into _forward_AV / _forward_A."""

    MODES = ("none", "mel_time_shuffle", "mel_block_shuffle", "phase_scramble")

    def __init__(self, base: RawNoisyAVDataset, indices, mode: str,
                 seed: int = 0, seg_len: int = SEG_LEN):
        assert mode in self.MODES, mode
        assert base.noise is False
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mode = mode
        self.seed = int(seed)
        self.seg_len = int(seg_len)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = int(self.indices[k])
        if self.mode == "phase_scramble":
            return self._phase_scramble(idx)

        mel_t, vid, label = self.base[idx]          # clean log-mel [80,99], video, label
        if self.mode == "none":
            return mel_t, vid, label

        T = mel_t.shape[1]                           # 99 frames
        rng = np.random.default_rng(self.seed + idx)
        if self.mode == "mel_time_shuffle":
            perm = rng.permutation(T)
        elif self.mode == "mel_block_shuffle":
            segs = np.array_split(np.arange(T), int(np.ceil(T / self.seg_len)))
            order = rng.permutation(len(segs))
            perm = np.concatenate([segs[i] for i in order])
        else:
            raise ValueError(self.mode)
        perm_t = torch.from_numpy(perm.astype(np.int64))
        return mel_t.index_select(1, perm_t).contiguous(), vid, label

    def _phase_scramble(self, idx: int):
        """Phase-randomize the raw waveform (keep magnitude spectrum), then re-mel."""
        audio = _read_wav(self.base.audio_paths[idx])           # float32 in [-1, 1]
        rng = np.random.default_rng(self.seed + idx)
        spec = np.fft.rfft(audio)
        mag = np.abs(spec)
        phase = rng.uniform(-np.pi, np.pi, size=spec.shape)
        scram = mag * np.exp(1j * phase)
        scram[0] = mag[0]                                       # DC stays real
        if len(audio) % 2 == 0:
            scram[-1] = mag[-1]                                 # Nyquist stays real
        audio_s = np.fft.irfft(scram, n=len(audio)).astype(np.float32)
        audio_p = _pad_audio(audio_s, int(self.base.pad_offsets[idx]))
        mel_t = torch.from_numpy(_wav_to_log_mel(audio_p).astype(np.float32))
        # clean video, identical preprocessing to RawNoisyAVDataset.__getitem__
        v = np.array(self.base._videos[idx])
        if self.base.t_stride > 1:
            v = v[:: self.base.t_stride]
        v = torch.from_numpy(v).unsqueeze(0).float() / 255.0
        return mel_t, v, int(self.base.labels[idx])


def _val_sha(val_idx) -> str:
    arr = val_idx.numpy() if hasattr(val_idx, "numpy") else np.asarray(val_idx)
    return hashlib.sha256(bytes(arr.astype("int64").tobytes())).hexdigest()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    splits = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                        weights_only=False)
    val_idx = splits["val_idx"]
    sha = _val_sha(val_idx)
    assert sha.startswith(PIN), f"VAL PIN MISMATCH: {sha[:16]}"
    val_np = val_idx.numpy() if hasattr(val_idx, "numpy") else np.asarray(val_idx)
    assert len(val_np) == 5244, len(val_np)
    print(f"  val_idx sha256[:16]={sha[:16]} (OK)  N={len(val_np)}")

    models = _load_models(device)
    av_model, av_ckpt = models["AV"]
    a_model = models["A"][0]
    print(f"  AV ckpt best_val_acc={av_ckpt.get('best_val_acc', float('nan')):.4%} "
          f"(loading final-weights av_fused.pt -> clean anchor {AV_CLEAN:.6f})")

    print(f"\n{'mode':>18s} {'AV_acc':>9s} {'A_acc':>9s}")
    rows = []
    for mode in _ScrambledAudioView.MODES:
        view = _ScrambledAudioView(base, val_np, mode=mode, seed=0)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        av_out = _forward_AV(av_model, loader, device,
                             video_kind="real", audio_kind="real")
        av_acc = _accuracy(av_out["preds"], av_out["labels"])
        a_preds, _, a_labels = _forward_A(a_model, loader, device)
        a_acc = _accuracy(a_preds, a_labels)
        rows.append((mode, av_acc, a_acc))
        print(f"{mode:>18s} {av_acc*100:8.4f}% {a_acc*100:8.4f}%")

    # reference rows (clean baseline + matched additive-noise sigma=0.05) in the SAME file
    rows.append(("ref_clean_baseline", AV_CLEAN, A_CLEAN))
    rows.append(("ref_additive_noise_sigma0.05", AV_NOISE_S05, A_NOISE_S05))

    # self-check: 'none' reproduces clean within ~0.1pp (acceptance criterion)
    none = {m: (av, a) for m, av, a in rows}["none"]
    print(f"\n[self-check] none AV={none[0]:.6f} (anchor {AV_CLEAN}); "
          f"A={none[1]:.6f} (anchor {A_CLEAN})")
    assert abs(none[0] - AV_CLEAN) < 1e-3, f"none AV {none[0]} != clean"
    assert abs(none[1] - A_CLEAN) < 1e-3, f"none A {none[1]} != clean"

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["mode", "AV_acc", "A_acc"])
        for mode, av, a in rows:
            w.writerow([mode, f"{av:.6f}", f"{a:.6f}"])
    print(f"\nwrote {OUT}")

    # headline contrast: is AV (clean vision) more scramble-robust than A-only?
    print("\n[robustness contrast: AV with clean video vs A-only, per scramble mode]")
    for mode, av, a in rows[:4]:
        if mode == "none":
            continue
        print(f"    {mode:>18s}: AV {av*100:6.2f}%  A {a*100:6.2f}%  "
              f"vision-rescue Δ={(av - a) * 100:+6.2f}pp")
    print("DONE")


if __name__ == "__main__":
    main()
