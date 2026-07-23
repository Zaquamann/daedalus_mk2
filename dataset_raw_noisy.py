#!/usr/bin/env python3
"""Raw-audio noise injection + on-the-fly mel computation. Loads each .wav
fresh, optionally adds Gaussian noise to the waveform (σ_a = u·audio_rms with
u uniform in `noise_range`), then computes the log-mel using the same
parameters as `preprocess.py` / `paired_dataset.py` so σ_a=0 reproduces the
cached mel byte-for-byte modulo float32 noise."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from paired_dataset import (
    CACHE_DIR,
    DATASET_AV_PATH,
    PROCESSED_DIR,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    T_FRAMES,
    VIDEO_CACHE_NAME,
    VIDEO_HW,
    _pad_audio,
    _read_wav,
    _wav_to_log_mel,
)


PAD_OFFSETS_PATH = os.path.join(PROCESSED_DIR, "pad_offsets.pt")


# Pad-offset recovery: the pad applied at cache-build time was per-sample
# deterministic, so leading all-zero video frames recover pad_left_frames
# without re-running the RNG.

def _derive_pad_left_frames(video_clip: np.ndarray) -> int:
    """Count leading frames that are entirely zero in a (T, H, W) uint8 clip."""
    pad_left = 0
    for t in range(video_clip.shape[0]):
        if video_clip[t].max() > 0:
            break
        pad_left += 1
    return pad_left


def build_pad_offsets_cache(force: bool = False) -> str:
    """Walk the video memmap once and dump per-sample pad_left_frames.

    Output: `processed/pad_offsets.pt` with key `pad_left_frames` (int32 tensor).
    """
    if os.path.exists(PAD_OFFSETS_PATH) and not force:
        return PAD_OFFSETS_PATH

    d = torch.load(DATASET_AV_PATH, weights_only=False)
    n = len(d["labels"])
    cache_path = d["video_cache_path"]
    if not os.path.exists(cache_path):
        cache_path = os.path.join(
            CACHE_DIR, d.get("video_cache_name", VIDEO_CACHE_NAME)
        )
    T, H, W = d["video_shape"]
    videos = np.memmap(cache_path, dtype=np.uint8, mode="r", shape=(n, T, H, W))

    print(f"Deriving pad_left for {n} samples...")
    pad_offsets = np.zeros(n, dtype=np.int32)
    for i in range(n):
        pad_offsets[i] = _derive_pad_left_frames(videos[i])
    print(f"  done. min={pad_offsets.min()}, "
          f"max={pad_offsets.max()}, "
          f"mean={pad_offsets.mean():.2f}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    torch.save(
        {
            "pad_left_frames": torch.from_numpy(pad_offsets),
            "n_samples": n,
            "T_frames": T,
            "samples_per_frame": SAMPLES_PER_FRAME,
        },
        PAD_OFFSETS_PATH,
    )
    print(f"Saved {PAD_OFFSETS_PATH}")
    return PAD_OFFSETS_PATH


class RawNoisyAVDataset(Dataset):
    """Raw-audio + cached-video dataset with Gaussian noise on the waveform.

    Yields:
        if return_video=True:  (mel[80, 99] float32, video[1, T, 88, 88] float32, label int)
        if return_video=False: (mel[80, 99] float32, label int)

    Args:
        dataset_pt_path: path to the AV-paired manifest (default `processed/dataset_av.pt`).
        t_stride:        slice every k-th video frame (e.g. 2 → T=50 from cached T=100).
        noise:           if True, add Gaussian noise to the raw audio at __getitem__.
        noise_range:     (lo, hi) for σ_a sampled uniformly from `[lo, hi] * audio_rms`.
                         Default (0.001, 0.05) per debugger recommendation.
        return_video:    A-only training can pass False to skip the memmap read.
    """

    def __init__(
        self,
        dataset_pt_path: Optional[str] = None,
        t_stride: int = 1,
        noise: bool = True,
        noise_range: tuple[float, float] = (0.001, 0.05),
        return_video: bool = True,
    ):
        path = dataset_pt_path or DATASET_AV_PATH
        d = torch.load(path, weights_only=False)
        self.labels: torch.Tensor = d["labels"]
        self.label_to_idx: dict = d["label_to_idx"]
        self.idx_to_label: dict = d["idx_to_label"]
        self.audio_paths: list[str] = d["audio_paths"]
        self.video_paths: list[str] = d.get("video_paths", [])
        self.config: dict = d["config"]

        self.t_stride: int = max(1, int(t_stride))
        self.noise: bool = bool(noise)
        self.noise_lo: float = float(noise_range[0])
        self.noise_hi: float = float(noise_range[1])
        self.return_video: bool = bool(return_video)

        # video memmap (lazy: only read in __getitem__ when return_video)
        T, H, W = d["video_shape"]
        n = len(self.labels)
        cache_path = d["video_cache_path"]
        if not os.path.exists(cache_path):
            cache_path = os.path.join(
                CACHE_DIR, d.get("video_cache_name", VIDEO_CACHE_NAME)
            )
        if return_video and not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"video cache missing: {cache_path}; "
                f"run `python paired_dataset.py` to rebuild."
            )
        self._video_shape = (T, H, W)
        self._video_cache_path = cache_path
        self._videos = (
            np.memmap(cache_path, dtype=np.uint8, mode="r", shape=(n, T, H, W))
            if return_video and os.path.exists(cache_path)
            else None
        )

        # pad offsets
        if not os.path.exists(PAD_OFFSETS_PATH):
            build_pad_offsets_cache()
        po = torch.load(PAD_OFFSETS_PATH, weights_only=False)
        self.pad_offsets: np.ndarray = po["pad_left_frames"].numpy().astype(np.int64)
        if len(self.pad_offsets) != n:
            raise RuntimeError(
                f"pad_offsets length {len(self.pad_offsets)} != dataset size {n}; "
                f"rebuild with `dataset_raw_noisy.build_pad_offsets_cache(force=True)`."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # 1. Load raw audio
        audio = _read_wav(self.audio_paths[idx])         # float32 in [-1, 1]

        # 2. Optional noise on the waveform (per-sample σ_a)
        if self.noise:
            audio_rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
            # torch.rand keeps DataLoader worker RNG isolation correct
            u = torch.rand(()).item()
            sigma = (self.noise_lo + u * (self.noise_hi - self.noise_lo)) * audio_rms
            noise = torch.randn(len(audio)).numpy().astype(np.float32) * sigma
            audio = audio + noise

        # 3. Apply the same pad as the cache — derived from the video
        pad_left_frames = int(self.pad_offsets[idx])
        audio_p = _pad_audio(audio, pad_left_frames)

        # 4. Mel
        mel = _wav_to_log_mel(audio_p).astype(np.float32)
        mel_t = torch.from_numpy(mel)

        if not self.return_video:
            return mel_t, int(self.labels[idx])

        # 5. Video from memmap
        v = np.array(self._videos[idx])                  # (T, H, W) uint8 (copy)
        if self.t_stride > 1:
            v = v[:: self.t_stride]
        v = torch.from_numpy(v).unsqueeze(0).float() / 255.0
        return mel_t, v, int(self.labels[idx])


# Smoke tests (run as `python dataset_raw_noisy.py`)

def _smoke_clean_match(n_check: int = 100) -> None:
    """σ_a=0 → fresh mel must equal `dataset_av.pt`'s cached mel within float32 noise."""
    print("\n[smoke 1] σ_a=0 fresh mel vs cached `dataset_av.pt` mel")
    d = torch.load(DATASET_AV_PATH, weights_only=False)
    cached = d["spectrograms"]                            # (N, 80, 99) float32

    ds = RawNoisyAVDataset(noise=False, return_video=False)
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=n_check, replace=False)

    diffs = []
    for i in indices:
        mel, _ = ds[int(i)]
        d_max = float((mel - cached[int(i)]).abs().max())
        diffs.append(d_max)
    diffs = np.asarray(diffs)
    print(f"  N={n_check}")
    print(f"  per-sample max abs diff: "
          f"min={diffs.min():.3e}, median={np.median(diffs):.3e}, max={diffs.max():.3e}")
    if diffs.max() < 1e-4:
        print("  [OK] all samples within float32 noise — fresh mel matches cached mel")
    else:
        print(f"  [FAIL] max abs diff {diffs.max():.3e} > 1e-4 — investigate")


def _smoke_noise_distribution(n_check: int = 200) -> None:
    """σ_a samples should be uniform in `[lo, hi] · audio_rms`."""
    print("\n[smoke 2] σ_a sampling distribution (uniform in [0.001, 0.05] · audio_rms)")
    ds = RawNoisyAVDataset(noise=True, return_video=False,
                           noise_range=(0.001, 0.05))
    rng = np.random.default_rng(1)
    indices = rng.choice(len(ds), size=n_check, replace=False)
    sigmas = []
    rmses = []
    snrs_db = []
    for i in indices:
        audio = _read_wav(ds.audio_paths[int(i)])
        rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
        rmses.append(rms)
        # Replay the dataset's own noise sample (call __getitem__ once)
        _ = ds[int(i)]
        # We can't directly observe the σ that was drawn without instrumentation,
        # so just sweep noise_range theoretically and report SNR bounds.
    # Theoretical SNR: signal_power = rms^2; noise_power = sigma^2 = (u·rms)^2
    # SNR_dB = 10 log10(rms^2 / (u·rms)^2) = -20 log10(u)
    snr_max_db = -20 * np.log10(0.001)     # σ=0.001·rms → SNR ≈ +60 dB
    snr_min_db = -20 * np.log10(0.05)      # σ=0.050·rms → SNR ≈ +26 dB
    rmses = np.asarray(rmses)
    print(f"  audio_rms over {n_check} val samples: "
          f"min={rmses.min():.4f}, median={np.median(rmses):.4f}, max={rmses.max():.4f}")
    print(f"  σ_a range across all samples (u={0.001}..{0.05}, rms varies):")
    print(f"    σ_a min: {0.001 * rmses.min():.6f}")
    print(f"    σ_a max: {0.05 * rmses.max():.6f}")
    print(f"  per-sample SNR ranges (uniform in u → log10-uniform in dB):")
    print(f"    SNR_max_dB ≈ {snr_max_db:.1f} dB  (u=0.001, near-clean)")
    print(f"    SNR_min_dB ≈ {snr_min_db:.1f} dB  (u=0.050, audibly noisy)")


def _smoke_dataloader_throughput(n_iter: int = 50, batch: int = 64) -> None:
    """Time how fast the dataset feeds batches — gate for training-time use."""
    import time

    from torch.utils.data import DataLoader
    print(f"\n[smoke 3] DataLoader throughput (B={batch}, on-the-fly mel)")
    ds = RawNoisyAVDataset(noise=True, t_stride=2, return_video=True)
    for workers in (0, 4, 8):
        dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                        pin_memory=True)
        it = iter(dl)
        # warm
        for _ in range(2):
            next(it)
        t0 = time.time()
        for _ in range(n_iter):
            mel, vid, y = next(it)
        dt = time.time() - t0
        per_batch_ms = dt / n_iter * 1000
        per_sample_ms = per_batch_ms / batch
        print(f"  workers={workers:>2d}: {per_batch_ms:6.1f} ms/batch, "
              f"{per_sample_ms:5.2f} ms/sample")


def main() -> None:
    if not os.path.exists(PAD_OFFSETS_PATH):
        build_pad_offsets_cache()
    _smoke_clean_match()
    _smoke_noise_distribution()
    _smoke_dataloader_throughput()
    print("\nAll smoke tests done.")


if __name__ == "__main__":
    main()
