#!/usr/bin/env python3
"""Paired audio + video dataset. Walks both trees, aligns clips by label
sequence within each (speaker, group), and precomputes a log-mel cache plus a
memmap uint8 video cache that share a per-sample 10ms pad-offset.
Run as a script to build caches; import `PairedAVDataset` for training."""

from __future__ import annotations

import os
import subprocess
import time
from collections import defaultdict
from typing import Iterable

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import torch
from torch.utils.data import Dataset


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.expanduser("~/Downloads/audio")
VIDEO_DIR = os.path.join(SCRIPT_DIR, "data", "visual", "video_data")
CACHE_DIR = os.path.join(SCRIPT_DIR, "data", "visual", "cache")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed")
DATASET_AV_PATH = os.path.join(PROCESSED_DIR, "dataset_av.pt")

# Speakers / groups that lack one modality entirely. Excluded from BOTH the
# AV path and the audio-only baseline so the two experiments train and test on
# the same partition.
DROPPED_SPEAKERS: frozenset[str] = frozenset({"speaker-34"})        # no video
DROPPED_SPEAKER_GROUPS: frozenset[tuple[str, int]] = frozenset({
    ("speaker-14", 3),                                              # no -C.wav
})

# Audio config (mirrors preprocess.py)
SAMPLE_RATE = 44100
N_MELS = 80
N_FFT = 2048
HOP_LENGTH = 441       # ~10 ms at 44.1 kHz  (= 1 video frame at 100 fps)
WIN_LENGTH = 1103
MAX_DURATION_S = 1.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_S)
N_MEL_FRAMES = 99      # log-mel time dim produced by the STFT below

# Video config
VIDEO_HW = 88          # lip ROI side after center-crop + scale
T_FRAMES = 100         # 1.0 s at 100 fps
VIDEO_FPS = 100
SAMPLES_PER_FRAME = SAMPLE_RATE // VIDEO_FPS    # 441 — exact

VIDEO_CACHE_NAME = f"videos_{VIDEO_HW}_{T_FRAMES}.uint8"


def parse_av_filename(fname: str, ext: str) -> tuple[int, int, str]:
    """`<spkr>_<grp>-<item>_<LABEL>-C.<ext>` → (group, item, label).

    Drops the `-C` suffix and strips trailing whitespace from LABEL — the audio
    side has occasional trailing-space junk that doesn't appear on video.
    """
    if not fname.endswith(ext):
        raise ValueError(fname)
    stem = fname[: -len(ext)]
    parts = stem.split("_", 2)
    if len(parts) < 3:
        raise ValueError(fname)
    grp_str, _, item_str = parts[1].partition("-")
    label = parts[2].rsplit("-C", 1)[0].strip()
    return int(grp_str), int(item_str), label


def _list_modality(speaker: str, root: str, ext: str) -> list[tuple[int, int, str, str]]:
    """List files for one speaker. Returns sorted [(grp, item, label, abs_path), ...]."""
    spk_dir = os.path.join(root, speaker)
    if not os.path.isdir(spk_dir):
        return []
    out = []
    for fn in os.listdir(spk_dir):
        if (not fn.endswith(ext)
                or fn.startswith("._")
                or "_SENT-END-" in fn):
            continue
        try:
            g, i, lbl = parse_av_filename(fn, ext)
        except ValueError:
            continue
        out.append((g, i, lbl, os.path.join(spk_dir, fn)))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def _lcs_pair(a: list[tuple], v: list[tuple]) -> list[tuple[str, str, str]]:
    """Pair a vs v by longest common subsequence of labels.

    Used as a fallback when len(a) != len(v) inside a (speaker, group): handles
    the off-by-one slate insertions/deletions cleanly.
    """
    A = [r[2] for r in a]
    V = [r[2] for r in v]
    m, n = len(A), len(V)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if A[i] == V[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    out: list[tuple[str, str, str]] = []
    i, j = m, n
    while i > 0 and j > 0:
        if A[i - 1] == V[j - 1]:
            out.append((a[i - 1][3], v[j - 1][3], a[i - 1][2]))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    out.reverse()
    return out


def _pair_within_group(
    audio_seq: list[tuple], video_seq: list[tuple],
) -> list[tuple[str, str, str]]:
    """Pair one (speaker, group) bucket. audio_seq/video_seq are item-sorted."""
    a = [r for r in audio_seq if r[2] != "SENTENCE"]
    v = list(video_seq)
    if not a or not v:
        return []
    if len(a) == len(v) and all(ar[2] == vr[2] for ar, vr in zip(a, v)):
        # Fast path: identical label sequence.
        return [(ar[3], vr[3], ar[2]) for ar, vr in zip(a, v)]
    return _lcs_pair(a, v)


def build_pairs() -> tuple[list[dict], dict[str, int]]:
    """Walk both modalities and return the paired sample list + per-speaker counts.

    Each pair is returned as a dict to keep the downstream call sites readable.
    """
    speakers = sorted(
        d for d in os.listdir(AUDIO_DIR)
        if d.startswith("speaker-") and os.path.isdir(os.path.join(AUDIO_DIR, d))
    )

    pairs: list[dict] = []
    per_spk: dict[str, int] = {}

    for spk in speakers:
        if spk in DROPPED_SPEAKERS:
            continue

        audio_all = _list_modality(spk, AUDIO_DIR, "-C.wav")
        video_all = _list_modality(spk, VIDEO_DIR, "-C.mkv")
        if not video_all:
            continue

        a_by_g: dict[int, list] = defaultdict(list)
        v_by_g: dict[int, list] = defaultdict(list)
        for r in audio_all:
            a_by_g[r[0]].append(r)
        for r in video_all:
            v_by_g[r[0]].append(r)

        kept = 0
        for g in sorted(set(a_by_g) & set(v_by_g)):
            if (spk, g) in DROPPED_SPEAKER_GROUPS:
                continue
            for ap, vp, lbl in _pair_within_group(a_by_g[g], v_by_g[g]):
                pairs.append({
                    "audio_path": ap, "video_path": vp,
                    "speaker": spk, "group": g, "label": lbl,
                })
                kept += 1
        per_spk[spk] = kept

    return pairs, per_spk


# Audio (log-mel)

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank() -> np.ndarray:
    fmin, fmax = 0.0, SAMPLE_RATE / 2.0
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), N_MELS + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((N_FFT + 1) * hz_points / SAMPLE_RATE).astype(int)
    n_freqs = N_FFT // 2 + 1
    fb = np.zeros((N_MELS, n_freqs), dtype=np.float32)
    for i in range(N_MELS):
        L, C, R = bins[i], bins[i + 1], bins[i + 2]
        for j in range(L, C):
            if C != L:
                fb[i, j] = (j - L) / (C - L)
        for j in range(C, R):
            if R != C:
                fb[i, j] = (R - j) / (R - C)
    return fb


_MEL_FB: np.ndarray | None = None


def _mel_fb() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _build_mel_filterbank()
    return _MEL_FB


def _wav_to_log_mel(audio: np.ndarray) -> np.ndarray:
    _, _, Zxx = scipy.signal.stft(
        audio, fs=SAMPLE_RATE, nperseg=WIN_LENGTH,
        noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT, boundary=None,
    )
    power = np.abs(Zxx) ** 2
    mel = _mel_fb() @ power
    return np.log(np.maximum(mel, 1e-10))


def _read_wav(path: str) -> np.ndarray:
    sr, audio = wavfile.read(path)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"unexpected sample rate {sr} for {path}")
    return audio.astype(np.float32) / 32768.0


# Video decode

def decode_video_gray(path: str) -> np.ndarray:
    """Decode an .mkv to (T, VIDEO_HW, VIDEO_HW) uint8.

    Center-crops to a 384×384 square (source is 500×384), scales to VIDEO_HW,
    and keeps the luma channel only.
    """
    crop_x = (500 - 384) // 2  # = 58
    vf = f"crop=384:384:{crop_x}:0,scale={VIDEO_HW}:{VIDEO_HW},format=gray"
    cmd = [
        "ffmpeg", "-loglevel", "error", "-nostdin", "-i", path,
        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True)
    raw = np.frombuffer(out.stdout, dtype=np.uint8)
    frame_size = VIDEO_HW * VIDEO_HW
    if raw.size % frame_size != 0:
        raise RuntimeError(f"raw bytes not divisible by frame size: {path}")
    return raw.reshape(-1, VIDEO_HW, VIDEO_HW)


# Shared pad / truncate

def _pick_pad_left_frames(
    audio_samples: int, video_frames: int, rng: np.random.Generator,
) -> int:
    """Pick a random left-pad offset (in 10 ms frames) shared by both modalities.

    The offset is bounded so neither the audio nor the video overflows its
    fixed-length output buffer.
    """
    audio_frames = (audio_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    natural = max(audio_frames, video_frames)
    pad_total = max(0, T_FRAMES - natural)
    if pad_total <= 0:
        return 0
    return int(rng.integers(0, pad_total + 1))


def _pad_audio(audio: np.ndarray, pad_left_frames: int) -> np.ndarray:
    pad_left = pad_left_frames * SAMPLES_PER_FRAME
    n = len(audio)
    if n + pad_left > MAX_SAMPLES:
        pad_left = max(0, MAX_SAMPLES - n)
    if n < MAX_SAMPLES:
        pad_right = MAX_SAMPLES - n - pad_left
        return np.pad(audio, (pad_left, max(0, pad_right)))[:MAX_SAMPLES]
    return audio[:MAX_SAMPLES]


def _pad_video(frames: np.ndarray, pad_left_frames: int) -> np.ndarray:
    T = frames.shape[0]
    if T + pad_left_frames > T_FRAMES:
        pad_left_frames = max(0, T_FRAMES - T)
    if T < T_FRAMES:
        pad_right = T_FRAMES - T - pad_left_frames
        zeros_l = np.zeros((pad_left_frames, VIDEO_HW, VIDEO_HW), dtype=np.uint8)
        zeros_r = np.zeros((max(0, pad_right), VIDEO_HW, VIDEO_HW), dtype=np.uint8)
        return np.concatenate([zeros_l, frames, zeros_r], axis=0)[:T_FRAMES]
    return frames[:T_FRAMES]


# Cache build

def precompute_av_cache(seed: int = 42, log_every: int = 500) -> dict:
    """Build the paired AV cache. Returns a small summary dict."""
    pairs, per_spk = build_pairs()
    n = len(pairs)
    if n == 0:
        raise RuntimeError("no AV pairs found")

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, VIDEO_CACHE_NAME)
    print(f"Building AV cache for {n} paired samples → {cache_path}")

    rng = np.random.default_rng(seed)
    videos_mm = np.memmap(
        cache_path, dtype=np.uint8, mode="w+",
        shape=(n, T_FRAMES, VIDEO_HW, VIDEO_HW),
    )
    mels = np.zeros((n, N_MELS, N_MEL_FRAMES), dtype=np.float32)
    labels: list[str] = []
    audio_paths: list[str] = []
    video_paths: list[str] = []
    speakers: list[str] = []
    groups: list[int] = []

    t0 = time.time()
    for i, p in enumerate(pairs):
        frames = decode_video_gray(p["video_path"])
        audio = _read_wav(p["audio_path"])

        pad_left = _pick_pad_left_frames(len(audio), frames.shape[0], rng)
        audio_p = _pad_audio(audio, pad_left)
        frames_p = _pad_video(frames, pad_left)

        mels[i] = _wav_to_log_mel(audio_p).astype(np.float32)
        videos_mm[i] = frames_p
        labels.append(p["label"])
        audio_paths.append(p["audio_path"])
        video_paths.append(p["video_path"])
        speakers.append(p["speaker"])
        groups.append(p["group"])

        if (i + 1) % log_every == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            print(f"  [{i + 1:>5d}/{n}]  {rate:5.1f} samples/s  ETA {eta:5.0f}s")

    videos_mm.flush()
    del videos_mm

    uniq = sorted(set(labels))
    label_to_idx = {l: idx for idx, l in enumerate(uniq)}
    idx_to_label = {idx: l for l, idx in label_to_idx.items()}
    label_indices = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    torch.save(
        {
            "spectrograms": torch.from_numpy(mels),
            "labels": label_indices,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "audio_paths": audio_paths,
            "video_paths": video_paths,
            "speakers": speakers,
            "groups": groups,
            "video_cache_path": cache_path,
            "video_cache_name": VIDEO_CACHE_NAME,
            "video_shape": (T_FRAMES, VIDEO_HW, VIDEO_HW),
            "video_dtype": "uint8",
            "config": {
                "sample_rate": SAMPLE_RATE,
                "n_mels": N_MELS, "n_fft": N_FFT,
                "hop_length": HOP_LENGTH, "win_length": WIN_LENGTH,
                "max_duration_s": MAX_DURATION_S,
                "video_hw": VIDEO_HW, "t_frames": T_FRAMES,
                "video_fps": VIDEO_FPS,
                "dropped_speakers": sorted(DROPPED_SPEAKERS),
                "dropped_speaker_groups": sorted(DROPPED_SPEAKER_GROUPS),
            },
        },
        DATASET_AV_PATH,
    )

    cache_size_gb = os.path.getsize(cache_path) / (1024 ** 3)
    pt_size_mb = os.path.getsize(DATASET_AV_PATH) / (1024 ** 2)
    elapsed = time.time() - t0
    summary = {
        "n_pairs": n,
        "n_classes": len(uniq),
        "per_speaker": per_spk,
        "video_cache_path": cache_path,
        "video_cache_size_gb": cache_size_gb,
        "dataset_av_pt_size_mb": pt_size_mb,
        "elapsed_s": elapsed,
    }
    return summary


class PairedAVDataset(Dataset):
    """Yields `(mel[80,99] float32, video[1,T,88,88] float32, label int)`.

    Reads the mel tensor from `dataset_av.pt` (in RAM, ~510 MB) and the video
    tensor from a memmap-backed uint8 file (lazy, no RAM blowup).
    """

    def __init__(
        self,
        dataset_pt_path: str | None = None,
        t_stride: int = 1,
    ):
        """`t_stride=2` returns every other frame (T=100 cache → T=50 output)."""
        path = dataset_pt_path or DATASET_AV_PATH
        d = torch.load(path, weights_only=False)
        self.mels: torch.Tensor = d["spectrograms"]            # (N, 80, 99) float32
        self.labels: torch.Tensor = d["labels"]                # (N,) long
        self.label_to_idx: dict[str, int] = d["label_to_idx"]
        self.idx_to_label: dict[int, str] = d["idx_to_label"]
        self.audio_paths: list[str] = d.get("audio_paths", [])
        self.video_paths: list[str] = d.get("video_paths", [])
        self.speakers: list[str] = d.get("speakers", [])
        self.groups: list[int] = d.get("groups", [])
        self.config: dict = d["config"]
        self.t_stride: int = max(1, int(t_stride))

        T, H, W = d["video_shape"]
        n = len(self.labels)
        cache_path = d["video_cache_path"]
        if not os.path.exists(cache_path):
            # Allow the cache to live alongside the .pt file even if moved.
            alt = os.path.join(CACHE_DIR, d.get("video_cache_name", VIDEO_CACHE_NAME))
            if os.path.exists(alt):
                cache_path = alt
            else:
                raise FileNotFoundError(
                    f"video cache missing: {cache_path}; run "
                    f"`python paired_dataset.py` to rebuild."
                )
        self._videos = np.memmap(
            cache_path, dtype=np.uint8, mode="r", shape=(n, T, H, W),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        mel = self.mels[idx]                                     # (80, 99) float32
        v = np.array(self._videos[idx])                          # (T, H, W) uint8 (copy)
        if self.t_stride > 1:
            v = v[:: self.t_stride]                              # (T//s, H, W)
        v = torch.from_numpy(v).unsqueeze(0).float() / 255.0     # (1, T', H, W) float32
        return mel, v, int(self.labels[idx])


def _print_summary(summary: dict) -> None:
    print("\n=== Paired AV cache summary ===")
    print(f"Total pairs: {summary['n_pairs']}")
    print(f"Unique labels: {summary['n_classes']}")
    print(f"Video cache: {summary['video_cache_path']}")
    print(f"  size: {summary['video_cache_size_gb']:.2f} GiB (uint8)")
    print(f"dataset_av.pt: {summary['dataset_av_pt_size_mb']:.1f} MiB")
    print(f"Elapsed: {summary['elapsed_s']:.0f}s")
    print("\nPer-speaker pair counts:")
    for spk, n in sorted(summary["per_speaker"].items()):
        print(f"  {spk}: {n}")


if __name__ == "__main__":
    summary = precompute_av_cache()
    _print_summary(summary)
