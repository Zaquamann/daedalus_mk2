#!/usr/bin/env python3
"""Train the parallel/late-fusion AV recognizer with a learned reliability gate
(`model_av_latefusion.AVLateFusionReliabilityWordResNet`).

Two things make the reliability gate actually learn to discount noisy audio:

1. AUDIO-NOISE AUGMENTATION ACROSS THE EVAL SIGMA RANGE. Noise is added to the
   raw waveform (per-sample sigma = u * audio_rms, u ~ U[lo,hi]) — byte-identical
   noise model to the d' harness's `_NoisyAudioView`. The committed clean AV
   model (`train_av.py`) used only SpecAugment + video flip (NO additive audio
   noise), and `train_av_rawnoise.py` capped u at 0.05 — so neither ever saw
   audio worse than video and neither learned to fall back. Here u spans
   [0.0, 0.22], covering the full E1c/E1d grid so the gate sees the regime where
   video is the more reliable cue.

2. PER-MODALITY AUXILIARY SUPERVISION. The loss is
       CE(fused) + AUX * CE(logit_a) + AUX * CE(logit_v)
   so each readout is independently driven to classify; the video->logit path
   cannot collapse to a passenger. Validation reports fused / audio-head /
   video-head accuracy and the mean reliability weight every epoch — the live
   "is the video readout learning?" diagnostic (kill if video-head val acc sits
   at chance by ~ep40).

Validation uses clean audio + clean video, so val_acc is comparable to the
committed AV models. Shares `processed/splits.pt` so the partition is identical.

Cheap-prove knobs via env: EPOCHS, NOISE_HI, AUX_W, WORKERS, SEED, MID_GATE.
"""

from __future__ import annotations

import hashlib
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train import spec_augment, stratified_split
from dataset_raw_noisy import RawNoisyAVDataset
from model_av_latefusion import AVLateFusionReliabilityWordResNet


# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(SCRIPT_DIR, "processed", "splits.pt")
# OUT_TAG suffixes ALL output artifacts so a new run never clobbers a validated
# checkpoint (e.g. OUT_TAG="_gradvid" -> av_fused_latefusion_gradvid.pt + its
# _ep{N}.pt and curves). Default "" reproduces the original paths. This is an
# output-path knob only; it does not touch any training math.
OUT_TAG = os.environ.get("OUT_TAG", "")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", f"av_fused_latefusion{OUT_TAG}.pt")
CURVE_PNG = os.path.join(SCRIPT_DIR, "analysis",
                         f"av_fused_latefusion{OUT_TAG}_curves.png")
CURVE_CSV = os.path.join(SCRIPT_DIR, "analysis",
                         f"av_fused_latefusion{OUT_TAG}_curves.csv")

BATCH_SIZE = 64
NUM_EPOCHS = int(os.environ.get("EPOCHS", "60"))   # cheap-prove default
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
TEST_SIZE = 0.33
RANDOM_SEED = int(os.environ.get("SEED", "0"))     # lead: single-seed, seed=0
T_STRIDE = 2
USE_BF16 = True
# torch.compile is a TRAIN-SPEED-ONLY optimization (never enters any eval number).
# It crashes on dev-codex (Triton runtime JIT needs Python.h, absent; uid 1001, no
# sudo) — proven in the Q14 pre-check. Default ON locally; the pod runner sets
# COMPILE=0. Eager produces identical math, so disabling it is numerically safe.
USE_COMPILE = os.environ.get("COMPILE", "1") == "1"
USE_MID_GATE = os.environ.get("MID_GATE", "0") == "1"   # default: pure late fusion
AUX_WEIGHT = float(os.environ.get("AUX_W", "0.5"))      # per-head aux CE weight
# Audio-noise augmentation span (sigma_a / audio_rms), uniform per sample. Covers
# the full E1c (<=0.08) and E1d (<=0.22) eval grids so the gate learns fallback.
NOISE_RANGE = (0.0, float(os.environ.get("NOISE_HI", "0.22")))
# Modality dropout: per-sample probability of a FULLY-dead stream (distinct from
# the sigma-noise regime). The gate must SEE dead streams to drive w->0 on one,
# i.e. to floor at the surviving cue when a stream dies.
P_ADEAD = float(os.environ.get("P_ADEAD", "0.12"))      # audio-dead (video-only)
P_VDEAD = float(os.environ.get("P_VDEAD", "0.12"))      # video-dead (audio-only)

NUM_WORKERS = int(os.environ.get("WORKERS", "16"))


class _AVAugmentedView(torch.utils.data.Dataset):
    """Adds the (1, ...) channel dim to mel; light per-sample augmentation on
    train only (SpecAugment on mel + horizontal lip flip)."""

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray, augment: bool):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = int(self.indices[k])
        mel, video, label = self.base[idx]
        mel = mel.unsqueeze(0)
        if self.augment:
            mel = spec_augment(mel)
            if torch.rand(1).item() < 0.5:
                video = torch.flip(video, dims=[-1])
        return mel, video, label


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _write_curves_csv(history: dict) -> None:
    os.makedirs(os.path.dirname(CURVE_CSV), exist_ok=True)
    with open(CURVE_CSV, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,"
                "val_acc_audio_head,val_acc_video_head,w_a_mean,"
                "wa_anoisy_mean,wa_anoisy_std,wa_clean_mean,wa_clean_std,"
                "wa_vnoisy_mean,wa_vnoisy_std,"
                "epoch_time_s,peak_gpu_gib\n")
        n = len(history["train_loss"])
        for i in range(n):
            f.write(
                f"{i+1},{history['train_loss'][i]:.6f},"
                f"{history['train_acc'][i]:.6f},"
                f"{history['val_loss'][i]:.6f},"
                f"{history['val_acc'][i]:.6f},"
                f"{history['val_acc_audio'][i]:.6f},"
                f"{history['val_acc_video'][i]:.6f},"
                f"{history['w_a_mean'][i]:.6f},"
                f"{history['wa_an_m'][i]:.6f},{history['wa_an_s'][i]:.6f},"
                f"{history['wa_cl_m'][i]:.6f},{history['wa_cl_s'][i]:.6f},"
                f"{history['wa_vn_m'][i]:.6f},{history['wa_vn_s'][i]:.6f},"
                f"{history['epoch_time_s'][i]:.3f},"
                f"{history['peak_gpu_gib'][i]:.3f}\n"
            )


def _save_curves(history: dict, num_classes: int, best_val_acc: float) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy"); ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(epochs, history["val_acc"], label="fused")
    ax.plot(epochs, history["val_acc_audio"], label="audio head")
    ax.plot(epochs, history["val_acc_video"], label="video head")
    ax2 = ax.twinx()
    ax2.plot(epochs, history["w_a_mean"], color="tab:gray", ls="--", label="w_a")
    ax2.set_ylabel("mean w_a (clean val)"); ax2.set_ylim(0, 1)
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
    ax.set_title(f"Val acc (best fused {best_val_acc:.1%}, {num_classes} cls)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(CURVE_PNG), exist_ok=True)
    fig.savefig(CURVE_PNG, dpi=130)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_base = RawNoisyAVDataset(t_stride=T_STRIDE, noise=True,
                                   noise_range=NOISE_RANGE, return_video=True)
    val_base = RawNoisyAVDataset(t_stride=T_STRIDE, noise=False,
                                 return_video=True)
    labels = train_base.labels
    label_to_idx = train_base.label_to_idx
    idx_to_label = train_base.idx_to_label
    config = train_base.config
    num_classes = len(label_to_idx)

    print(f"Loaded {len(train_base)} paired samples, {num_classes} classes")
    print(f"Noise (sigma_a / audio_rms) at TRAIN: uniform "
          f"[{NOISE_RANGE[0]:.4f}, {NOISE_RANGE[1]:.4f}] per sample; VAL clean")
    print(f"seed={RANDOM_SEED}  epochs={NUM_EPOCHS}  aux_w={AUX_WEIGHT}  "
          f"mid_gate={USE_MID_GATE}  workers={NUM_WORKERS}")
    print(f"modality dropout: P(audio-dead)={P_ADEAD:.2f}  "
          f"P(video-dead)={P_VDEAD:.2f}  P(both)={1-P_ADEAD-P_VDEAD:.2f}")
    print(f"OUT_TAG='{OUT_TAG}'  (D313 candidate-c: gate reads logit-confidence; "
          f"NO video aug — training pipeline == GO)")

    assert os.path.exists(SPLITS_PATH), f"missing shared splits: {SPLITS_PATH}"
    s = torch.load(SPLITS_PATH, weights_only=False)
    train_idx, val_idx = s["train_idx"], s["val_idx"]
    print(f"Loaded shared splits from {SPLITS_PATH}")

    train_hash, val_hash = _hash_idx(train_idx), _hash_idx(val_idx)
    print(f"train_idx sha256: {train_hash}")
    print(f"val_idx   sha256: {val_hash}")

    train_ds = _AVAugmentedView(train_base, train_idx, augment=True)
    val_ds = _AVAugmentedView(val_base, val_idx, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = AVLateFusionReliabilityWordResNet(
        num_classes, use_mid_gate=USE_MID_GATE).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_visual = sum(p.numel() for p in model.visual.parameters())
    n_rel = sum(p.numel() for p in model.rel_gate.parameters())
    print(f"Model parameters: total={n_total:,}, visual={n_visual:,}, "
          f"rel_gate={n_rel:,}")
    print(f"bf16={USE_BF16}  compile={USE_COMPILE}")
    autocast_kw = {"device_type": "cuda", "dtype": torch.bfloat16,
                   "enabled": USE_BF16}

    compiled = torch.compile(model, mode="default") if USE_COMPILE else model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc",
                               "val_acc_audio", "val_acc_video", "w_a_mean",
                               "wa_an_m", "wa_an_s", "wa_cl_m", "wa_cl_s",
                               "wa_vn_m", "wa_vn_s",
                               "epoch_time_s", "peak_gpu_gib")}
    best_val_acc, best_val_loss, best_epoch = 0.0, float("inf"), 0
    epoch_times, peak_gpu_gib = [], 0.0

    # ---- Fixed w_a reliability probe (diagnostic; D313 ep40-50 kill-gate) ----
    # A fixed val subset evaluated each epoch under 3 reliability conditions, to
    # watch whether the gate STARTS tracking graded reliability: if it learns SNR
    # routing, w_a(audio-noisy) < w_a(clean) < w_a(video-noisy); if it stays the
    # D313 constant it is flat ~0.375 across all three -> KILL. Built ONCE with a
    # fixed noise realization so only the model changes epoch-to-epoch.
    def _stack_probe(base, indices, want_video):
        mels, vids = [], []
        for i in indices:
            out = base[int(i)]
            mels.append(out[0].unsqueeze(0))
            if want_video:
                vids.append(out[1])
        return torch.stack(mels), (torch.stack(vids) if want_video else None)

    # Build the fixed probe ONCE. Wrap in RNG save/restore so this eval-only
    # diagnostic consumes ZERO global RNG — the training data order is then
    # independent of the probe (the only intended difference vs GO is the model's
    # gate input). probe_vid_vnoisy already uses a private generator (_g).
    _rng_state = torch.get_rng_state()
    probe_sel = [int(x) for x in
                 (val_idx[:512] if len(val_idx) >= 512 else val_idx)]
    probe_mel_clean, probe_vid_clean = _stack_probe(val_base, probe_sel, True)
    _anoise_base = RawNoisyAVDataset(t_stride=T_STRIDE, noise=True,
                                     noise_range=(0.22, 0.22), return_video=False)
    probe_mel_anoisy, _ = _stack_probe(_anoise_base, probe_sel, False)
    _g = torch.Generator().manual_seed(1234)
    probe_vid_vnoisy = probe_vid_clean.clone()
    for _j in range(probe_vid_vnoisy.shape[0]):
        _v = probe_vid_vnoisy[_j]
        probe_vid_vnoisy[_j] = _v + torch.randn(
            _v.shape, generator=_g) * (0.22 * float(_v.std()))
    torch.set_rng_state(_rng_state)
    print(f"w_a probe: {len(probe_sel)} fixed val samples; conditions = "
          f"clean / audio-noisy(s0.22) / video-noisy(s0.22)")

    @torch.no_grad()
    def _probe_wa(mels, vids):
        model.eval()
        out = []
        for s in range(0, mels.shape[0], 128):
            m = mels[s:s + 128].to(device, non_blocking=True)
            v = vids[s:s + 128].to(device, non_blocking=True)
            with torch.autocast(**autocast_kw):
                _f, _la, _lv, w = compiled(m, v, return_parts=True)
            out.append(w[:, 0].float().cpu())
        wa = torch.cat(out)
        return float(wa.mean()), float(wa.std())

    print(f"\n{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'VaLoss':>7} | "
          f"{'Fused':>6} | {'Ahead':>6} | {'Vhead':>6} | {'w_a':>5} | "
          f"{'waAn':>5} | {'waCl':>5} | {'waVn':>5} | "
          f"{'Time':>5} | {'GPU':>4}")
    print("-" * 116)

    def _save(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx, "idx_to_label": idx_to_label,
            "config": config, "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss, "epoch": best_epoch,
            "train_idx": train_idx, "val_idx": val_idx,
            "train_idx_sha256": train_hash, "val_idx_sha256": val_hash,
            "noise_range": NOISE_RANGE, "noise_kind": "raw_audio_gaussian",
            "aux_weight": AUX_WEIGHT, "use_mid_gate": USE_MID_GATE,
            "p_audio_dead": P_ADEAD, "p_video_dead": P_VDEAD,
            "seed": RANDOM_SEED,
            "val_acc_audio_at_best": history["val_acc_audio"][-1],
            "val_acc_video_at_best": history["val_acc_video"][-1],
            "w_a_mean_at_best": history["w_a_mean"][-1],
        }, path)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model.train()
        tr_loss = tr_correct = tr_total = 0
        for mel, video, y in train_loader:
            mel = mel.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # Per-sample modality dropout: r<P_ADEAD -> audio dead (video-only);
            # next P_VDEAD band -> video dead (audio-only); rest both-present.
            r = torch.rand(y.size(0), device=device)
            audio_dead = r < P_ADEAD
            video_dead = (r >= P_ADEAD) & (r < P_ADEAD + P_VDEAD)
            a_present, v_present = ~audio_dead, ~video_dead
            optimizer.zero_grad()
            with torch.autocast(**autocast_kw):
                fused, la, lv, _w = compiled(
                    mel, video, audio_dead=audio_dead,
                    video_dead=video_dead, return_parts=True)
                # Fused CE on ALL samples (teaches the gate to route to the live
                # stream); per-head aux CE only on samples where that stream is
                # PRESENT (never penalize a head for a zeroed stream).
                loss = criterion(fused, y)
                if a_present.any():
                    loss = loss + AUX_WEIGHT * criterion(la[a_present], y[a_present])
                if v_present.any():
                    loss = loss + AUX_WEIGHT * criterion(lv[v_present], y[v_present])
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * y.size(0)
            tr_correct += (fused.argmax(1) == y).sum().item()
            tr_total += y.size(0)
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        model.eval()
        va_loss = 0.0
        va_f = va_a = va_v = va_total = 0
        w_a_sum = 0.0
        with torch.no_grad():
            for mel, video, y in val_loader:
                mel = mel.to(device, non_blocking=True)
                video = video.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(**autocast_kw):
                    fused, la, lv, w = compiled(mel, video, return_parts=True)
                    loss = criterion(fused, y)
                va_loss += loss.item() * y.size(0)
                va_f += (fused.argmax(1) == y).sum().item()
                va_a += (la.argmax(1) == y).sum().item()
                va_v += (lv.argmax(1) == y).sum().item()
                w_a_sum += w[:, 0].float().sum().item()
                va_total += y.size(0)
        va_loss /= va_total
        va_acc = va_f / va_total
        va_acc_a = va_a / va_total
        va_acc_v = va_v / va_total
        w_a_mean = w_a_sum / va_total

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_acc_audio"].append(va_acc_a)
        history["val_acc_video"].append(va_acc_v)
        history["w_a_mean"].append(w_a_mean)

        # w_a reliability probe (3 fixed conditions) — D313 gate-tracking readout.
        # Tracking ⇒ wa_an < wa_cl < wa_vn; flat ≈0.375 across all ⇒ kill.
        wa_an_m, wa_an_s = _probe_wa(probe_mel_anoisy, probe_vid_clean)
        wa_cl_m, wa_cl_s = _probe_wa(probe_mel_clean, probe_vid_clean)
        wa_vn_m, wa_vn_s = _probe_wa(probe_mel_clean, probe_vid_vnoisy)
        history["wa_an_m"].append(wa_an_m); history["wa_an_s"].append(wa_an_s)
        history["wa_cl_m"].append(wa_cl_m); history["wa_cl_s"].append(wa_cl_s)
        history["wa_vn_m"].append(wa_vn_m); history["wa_vn_s"].append(wa_vn_s)

        epoch_t = time.time() - t0
        epoch_times.append(epoch_t)
        epoch_peak = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                      if device.type == "cuda" else 0.0)
        peak_gpu_gib = max(peak_gpu_gib, epoch_peak)
        history["epoch_time_s"].append(epoch_t)
        history["peak_gpu_gib"].append(epoch_peak)

        print(f"{epoch:3d} | {tr_loss:7.4f} | {tr_acc:6.1%} | {va_loss:7.4f} | "
              f"{va_acc:6.1%} | {va_acc_a:6.1%} | {va_acc_v:6.1%} | "
              f"{w_a_mean:5.3f} | {wa_an_m:5.3f} | {wa_cl_m:5.3f} | "
              f"{wa_vn_m:5.3f} | {epoch_t:4.1f}s | {epoch_peak:3.1f}G",
              flush=True)

        if va_acc > best_val_acc:
            best_val_acc, best_val_loss, best_epoch = va_acc, va_loss, epoch
            _save(MODEL_PATH)
        if epoch % 10 == 0:                      # mid-checkpoints for ep40-50 check
            _save(MODEL_PATH.replace(".pt", f"_ep{epoch}.pt"))
        scheduler.step()

    print(f"\nTraining complete. Best fused val acc: {best_val_acc:.1%} "
          f"(epoch {best_epoch})")
    print(f"Video-head val acc at best: {history['val_acc_video'][best_epoch-1]:.1%}  "
          f"Audio-head: {history['val_acc_audio'][best_epoch-1]:.1%}  "
          f"w_a_mean(clean): {history['w_a_mean'][best_epoch-1]:.3f}")
    print(f"Mean time/epoch: {np.mean(epoch_times):.1f}s")
    print(f"Peak GPU: {peak_gpu_gib:.2f} GiB")
    print(f"Saved best model to: {MODEL_PATH}")

    _save_curves(history, num_classes, best_val_acc)
    _write_curves_csv(history)
    print(f"Saved curves to: {CURVE_PNG}")
    print(f"Saved curves CSV to: {CURVE_CSV}")


if __name__ == "__main__":
    main()
