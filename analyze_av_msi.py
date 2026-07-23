#!/usr/bin/env python3
"""Multisensory-integration battery (11 experiments per the MSI metrics plan)
against the shared val partition. Loads A-only, V-only (fair if present),
and AV-fused checkpoints. Outputs in `analysis/msi/`."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from analyze_phoneme_accuracy import get_onset
from dataset_raw_noisy import RawNoisyAVDataset
from model_av import AVWordResNet
from model_v_only import VOnlyWordResNet
from paired_dataset import (
    DATASET_AV_PATH,
    SAMPLES_PER_FRAME,
    T_FRAMES,
    VIDEO_HW,
    _pad_audio,
    _read_wav,
    _wav_to_log_mel,
)
from train import WordResNet


# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "msi")
os.makedirs(OUT_DIR, exist_ok=True)

A_CKPT = os.path.join(SCRIPT_DIR, "models", "audio_only_filtered.pt")
V_CKPT = os.path.join(SCRIPT_DIR, "models", "video_only.pt")
V_FAIR_CKPT = os.path.join(SCRIPT_DIR, "models", "video_only_fair.pt")
AV_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused.pt")

BATCH_SIZE = 64
T_STRIDE = 2
SIGMA_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

VISEME_MAP = {
    "/f/": "labiodental_fv",
    "/b/": "bilabial_bpm", "/p/": "bilabial_bpm", "/m/": "bilabial_bpm",
    "/w/": "labiovelar_w",
    "/t/": "lingual", "/d/": "lingual", "/n/": "lingual",
    "/s/": "lingual", "/r/": "lingual", "/k/": "lingual",
    "/h/": "glottal_h",
    "vowel": "vowel_initial",
    "other": "other",
}


def _viseme(label: str) -> str:
    return VISEME_MAP.get(get_onset(label), "other")


# Dataset views

class _ValAVView(Dataset):
    """Val partition (clean) — yields `(mel[80,99], video[1,T,88,88], label)`."""

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        return self.base[int(self.indices[k])]


class _NoisyAudioView(Dataset):
    """Val partition with deterministic raw-audio noise at fixed σ_a / rms."""

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray,
                 sigma_mult: float, seed: int = 0):
        assert base.noise is False
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sigma_mult = float(sigma_mult)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = int(self.indices[k])
        audio = _read_wav(self.base.audio_paths[idx])
        if self.sigma_mult > 0:
            rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
            sigma = self.sigma_mult * rms
            rng = np.random.default_rng(self.seed + idx)
            noise = rng.standard_normal(len(audio)).astype(np.float32) * sigma
            audio = audio + noise
        pad_left = int(self.base.pad_offsets[idx])
        audio_p = _pad_audio(audio, pad_left)
        mel = torch.from_numpy(_wav_to_log_mel(audio_p).astype(np.float32))

        v = np.array(self.base._videos[idx])
        if self.base.t_stride > 1:
            v = v[:: self.base.t_stride]
        v = torch.from_numpy(v).unsqueeze(0).float() / 255.0
        return mel, v, int(self.base.labels[idx])


# Model loading helpers

def _load_models(device: torch.device):
    """Return dict: name → (model, n_classes, idx_to_label)."""
    out = {}

    a_ckpt = torch.load(A_CKPT, weights_only=False)
    a = WordResNet(len(a_ckpt["label_to_idx"]))
    a.load_state_dict(a_ckpt["model_state_dict"])
    out["A"] = (a.to(device).eval(), a_ckpt)

    v_path = V_FAIR_CKPT if os.path.exists(V_FAIR_CKPT) else V_CKPT
    v_ckpt = torch.load(v_path, weights_only=False)
    if "block2" in next(iter(v_ckpt["model_state_dict"].keys()), "") or any(
        k.startswith("block2") for k in v_ckpt["model_state_dict"].keys()
    ):
        from model_v_only_fair import VOnlyFairWordResNet
        v = VOnlyFairWordResNet(len(v_ckpt["label_to_idx"]))
    else:
        v = VOnlyWordResNet(len(v_ckpt["label_to_idx"]))
    v.load_state_dict(v_ckpt["model_state_dict"])
    out["V"] = (v.to(device).eval(), v_ckpt)
    out["_V_path"] = v_path

    av_ckpt = torch.load(AV_CKPT, weights_only=False)
    av = AVWordResNet(len(av_ckpt["label_to_idx"]))
    av.load_state_dict(av_ckpt["model_state_dict"])
    out["AV"] = (av.to(device).eval(), av_ckpt)
    return out


# Forward helpers (preds + probs + intermediate activations)

@torch.no_grad()
def _forward_A(model: WordResNet, loader, device):
    preds, probs, labels = [], [], []
    for mel, _v, y in loader:
        x = mel.unsqueeze(1).to(device, non_blocking=True)
        logits = model(x)
        p = logits.softmax(dim=1)
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return (np.concatenate(preds), np.concatenate(probs),
            np.concatenate(labels))


@torch.no_grad()
def _forward_V(model, loader, device):
    preds, probs, labels = [], [], []
    for _mel, v, y in loader:
        v = v.to(device, non_blocking=True)
        logits = model(v)
        p = logits.softmax(dim=1)
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return (np.concatenate(preds), np.concatenate(probs),
            np.concatenate(labels))


@torch.no_grad()
def _forward_AV(model: AVWordResNet, loader, device,
                video_kind: str = "real", audio_kind: str = "real",
                video_scale: float = 1.0, audio_scale: float = 1.0,
                return_acts: bool = False):
    """
    Args:
        video_kind: "real" | "zero" | "scaled" (uses video_scale)
        audio_kind: "real" | "zero" | "scaled"
        video_scale: only used when video_kind="scaled"  (β linear interp toward 0-mean)
        audio_scale: same idea for audio
        return_acts: if True, also collect a_mid, v_mid, gate_out, block2_out, penult.
    """
    preds, probs, labels = [], [], []
    a_mids, v_mids, gates, b2s, pens = [], [], [], [], []
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        # apply audio modifiers
        if audio_kind == "zero":
            mel = torch.zeros_like(mel)
        elif audio_kind == "scaled":
            mel = mel * float(audio_scale)
        # video
        if video_kind == "zero":
            vid = torch.zeros_like(vid)
        elif video_kind == "scaled":
            vid = vid * float(video_scale)

        a_mid = model.audio_block1(mel)
        v_mid = (torch.zeros_like(a_mid) if video_kind == "zero"
                 else model.visual(vid))
        a_fused = model.gate(a_mid, v_mid)
        x = model.audio_block2(a_fused)
        pen = model.gap(x).flatten(1)
        logits = model.fc(model.dropout(pen))
        prob = logits.softmax(dim=1)
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(prob.cpu().numpy())
        labels.append(y.numpy())
        if return_acts:
            a_mids.append(a_mid.cpu().numpy())
            v_mids.append(v_mid.cpu().numpy())
            gates.append((a_fused - a_mid).cpu().numpy())
            b2s.append(x.cpu().numpy())
            pens.append(pen.cpu().numpy())
    out = {
        "preds": np.concatenate(preds),
        "probs": np.concatenate(probs),
        "labels": np.concatenate(labels),
    }
    if return_acts:
        out["a_mid"] = np.concatenate(a_mids)
        out["v_mid"] = np.concatenate(v_mids)
        out["gate"] = np.concatenate(gates)
        out["block2"] = np.concatenate(b2s)
        out["penult"] = np.concatenate(pens)
    return out


def _accuracy(preds, labels):
    return float((preds == labels).mean())


# E1. Inverse-effectiveness — σ_a sweep, AV vs A-only

def E1_inverse_effectiveness(models, val_idx, base, device):
    print("\n[E1] Inverse-effectiveness: σ_a sweep on AV-clean and A-only-clean")
    rows = []
    for sigma in SIGMA_LEVELS:
        view = _NoisyAudioView(base, val_idx, sigma_mult=sigma, seed=0)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        a_preds, _, labels = _forward_A(models["A"][0], loader, device)
        acc_a = _accuracy(a_preds, labels)
        av_out = _forward_AV(models["AV"][0], loader, device,
                             video_kind="real", audio_kind="real")
        acc_av = _accuracy(av_out["preds"], labels)
        rows.append((sigma, acc_a, acc_av, acc_av - acc_a))
        print(f"  σ={sigma:6.4f}: A={acc_a:.4%}  AV={acc_av:.4%}  Δ={acc_av-acc_a:+.4%}")

    out_csv = os.path.join(OUT_DIR, "E1_inverse_effectiveness.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["sigma_per_rms", "A_acc", "AV_acc", "AV_minus_A"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}", f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sigmas = [r[0] for r in rows]
    axes[0].plot(sigmas, [r[1] for r in rows], "o-", label="A-only", color="#4477aa")
    axes[0].plot(sigmas, [r[2] for r in rows], "o-", label="AV", color="#cc6677")
    axes[0].set_xscale("symlog", linthresh=0.001)
    axes[0].set_xlabel("σ_a / audio_rms")
    axes[0].set_ylabel("val acc")
    axes[0].set_title("Inverse effectiveness: acc vs noise")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(sigmas, [r[3] for r in rows], "o-", color="black")
    axes[1].axhline(0, ls="--", color="gray", alpha=0.5)
    axes[1].set_xscale("symlog", linthresh=0.001)
    axes[1].set_xlabel("σ_a / audio_rms")
    axes[1].set_ylabel("AV − A (pp)")
    axes[1].set_title("Multisensory enhancement vs noise")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "E1_inverse_effectiveness.png"), dpi=130)
    plt.close(fig)
    return rows


# E2. Graduated modality dropout (β scaling toward zero)

def E2_graduated_dropout(models, val_idx, base, device):
    print("\n[E2] Graduated modality dropout — AV-clean")
    betas = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    rows = []
    for beta in betas:
        # Drop video
        out_v = _forward_AV(models["AV"][0], loader, device,
                            video_kind="scaled", video_scale=beta)
        acc_drop_v = _accuracy(out_v["preds"], out_v["labels"])
        # Drop audio
        out_a = _forward_AV(models["AV"][0], loader, device,
                            audio_kind="scaled", audio_scale=beta)
        acc_drop_a = _accuracy(out_a["preds"], out_a["labels"])
        rows.append((beta, acc_drop_v, acc_drop_a))
        print(f"  β={beta:4.2f}: AV (video×β)={acc_drop_v:.4%}  "
              f"AV (audio×β)={acc_drop_a:.4%}")

    out_csv = os.path.join(OUT_DIR, "E2_graduated_dropout.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["beta", "AV_video_scaled_acc", "AV_audio_scaled_acc"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}", f"{r[1]:.6f}", f"{r[2]:.6f}"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bs = [r[0] for r in rows]
    ax.plot(bs, [r[1] for r in rows], "o-", label="video × β  (audio full)",
            color="#cc6677")
    ax.plot(bs, [r[2] for r in rows], "o-", label="audio × β  (video full)",
            color="#4477aa")
    ax.invert_xaxis()
    ax.set_xlabel("β  (1=full, 0=zero)")
    ax.set_ylabel("val acc")
    ax.set_title("Graduated modality dropout — AV-clean")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "E2_graduated_dropout.png"), dpi=130)
    plt.close(fig)
    return rows


# E3. McGurk-style cross-pair test

class _McGurkView(Dataset):
    """Yields `(mel_X, video_Y, audio_label_X, video_label_Y)` for cross-pair stimuli."""

    def __init__(self, base: RawNoisyAVDataset, pair_indices, idx_to_label):
        # pair_indices: list of (i_audio, j_video) — both global indices into base
        self.base = base
        self.pairs = list(pair_indices)
        self.idx_to_label = idx_to_label

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j = self.pairs[k]
        audio_mel, _vid_i, ylab_i = self.base[int(i)]
        _mel_j, video_j, ylab_j = self.base[int(j)]
        return audio_mel, video_j, int(ylab_i), int(ylab_j)


def _build_mcgurk_pairs(base, val_idx, idx_to_label, n_per_class: int = 30,
                        rng_seed: int = 0):
    """Pair (audio i, video j) where viseme(i.label) ≠ viseme(j.label),
    same speaker, same group."""
    rng = np.random.default_rng(rng_seed)
    val_set = set(val_idx.tolist())
    # bucket val items by (speaker, group, viseme)
    speakers = base.speakers if hasattr(base, "speakers") else None
    groups = base.groups if hasattr(base, "groups") else None
    if speakers is None or groups is None:
        d = torch.load(DATASET_AV_PATH, weights_only=False)
        speakers = d.get("speakers", [None] * len(base))
        groups = d.get("groups", [None] * len(base))

    by_sg: dict[tuple, list[int]] = defaultdict(list)
    for i in val_idx:
        i = int(i)
        spk = speakers[i] if speakers else "?"
        grp = groups[i] if groups else 0
        by_sg[(spk, grp)].append(i)

    distinct, identical = [], []
    for (spk, grp), items in by_sg.items():
        items = list(items)
        rng.shuffle(items)
        if len(items) < 2:
            continue
        for n, i in enumerate(items):
            label_i = idx_to_label[int(base.labels[i])]
            vis_i = _viseme(label_i)
            # try to find a partner in the same speaker-group with a different viseme
            tries = 0
            for j in items:
                if j == i:
                    continue
                label_j = idx_to_label[int(base.labels[j])]
                vis_j = _viseme(label_j)
                if vis_j != vis_i and vis_j != "other" and vis_i != "other":
                    distinct.append((i, j))
                    tries += 1
                    break
            # also collect a viseme-identical pair
            for j in items:
                if j == i:
                    continue
                label_j = idx_to_label[int(base.labels[j])]
                vis_j = _viseme(label_j)
                if vis_j == vis_i and label_j != label_i and vis_i != "other":
                    identical.append((i, j))
                    break
    return distinct[: n_per_class * 50], identical[: n_per_class * 50]


def E3_mcgurk(models, val_idx, base, device, idx_to_label):
    print("\n[E3] McGurk-style cross-pair (viseme-distinct vs viseme-identical)")
    distinct_pairs, identical_pairs = _build_mcgurk_pairs(base, val_idx, idx_to_label)
    print(f"  distinct pairs:  {len(distinct_pairs)}")
    print(f"  identical pairs: {len(identical_pairs)}")
    rows = []
    for label, pairs in [("distinct", distinct_pairs), ("identical", identical_pairs)]:
        view = _McGurkView(base, pairs, idx_to_label)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        # build manually since we need 4-tuple unpacking
        audio_caps = vis_caps = third = total = 0
        with torch.no_grad():
            for mel, vid, ylab_i, ylab_j in loader:
                mel = mel.unsqueeze(1).to(device, non_blocking=True)
                vid = vid.to(device, non_blocking=True)
                ylab_i = ylab_i.numpy()
                ylab_j = ylab_j.numpy()
                # AV-clean prediction
                a_mid = models["AV"][0].audio_block1(mel)
                v_mid = models["AV"][0].visual(vid)
                a_fused = models["AV"][0].gate(a_mid, v_mid)
                x = models["AV"][0].audio_block2(a_fused)
                pen = models["AV"][0].gap(x).flatten(1)
                pred = models["AV"][0].fc(models["AV"][0].dropout(pen)).argmax(1).cpu().numpy()
                # also A-only
                pred_a = models["A"][0](mel).argmax(1).cpu().numpy()

                for k in range(len(pred)):
                    total += 1
                    if pred[k] == ylab_i[k]:
                        audio_caps += 1
                    elif pred[k] == ylab_j[k]:
                        vis_caps += 1
                    else:
                        third += 1
                # store A-only stats too
        # A-only on these pairs (audio dictates)
        # Re-iterate to get A-only capture
        a_caps_a = 0
        with torch.no_grad():
            for mel, vid, ylab_i, ylab_j in loader:
                mel = mel.unsqueeze(1).to(device, non_blocking=True)
                pred_a = models["A"][0](mel).argmax(1).cpu().numpy()
                for k, p in enumerate(pred_a):
                    if p == ylab_i[k].item():
                        a_caps_a += 1

        rows.append({
            "conflict_type": label,
            "n_pairs": total,
            "AV_audio_capture": audio_caps / max(1, total),
            "AV_visual_capture": vis_caps / max(1, total),
            "AV_third_word":   third      / max(1, total),
            "A_only_audio_capture": a_caps_a / max(1, total),
        })
        print(f"  {label}: n={total}  "
              f"audio_cap={audio_caps/total:.2%}  "
              f"visual_cap={vis_caps/total:.2%}  "
              f"third={third/total:.2%}  "
              f"(A-only audio_cap={a_caps_a/total:.2%})")

    out_csv = os.path.join(OUT_DIR, "E3_mcgurk_capture_rates.csv")
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    return rows


# E4. Activation-level MEI per layer

def E4_activation_mei(models, val_idx, base, device):
    print("\n[E4] Activation-level MEI per layer")
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    # Three conditions through AV-clean
    print("  ... AV (real, real)")
    out_av = _forward_AV(models["AV"][0], loader, device,
                        video_kind="real", audio_kind="real", return_acts=True)
    print("  ... A-only via AV (audio real, video zero)")
    out_a = _forward_AV(models["AV"][0], loader, device,
                       video_kind="zero", audio_kind="real", return_acts=True)
    print("  ... V-only via AV (audio zero, video real)")
    out_v = _forward_AV(models["AV"][0], loader, device,
                       video_kind="real", audio_kind="zero", return_acts=True)

    # Per-channel scalar response: mean over batch + spatial-temporal axes.
    sites = ["a_mid", "v_mid", "gate", "block2", "penult"]
    rows = []
    for site in sites:
        ra = np.abs(out_a[site]).reshape(out_a[site].shape[0], out_a[site].shape[1], -1).mean(axis=(0, 2))
        rv = np.abs(out_v[site]).reshape(out_v[site].shape[0], out_v[site].shape[1], -1).mean(axis=(0, 2))
        rav = np.abs(out_av[site]).reshape(out_av[site].shape[0], out_av[site].shape[1], -1).mean(axis=(0, 2))
        max_uni = np.maximum(ra, rv)
        # MEI per channel; guard against tiny denominators
        denom = np.maximum(max_uni, 1e-6)
        mei = (rav - max_uni) / denom
        sa = (rav > (ra + rv)).astype(np.float32)
        rows.append({
            "site": site,
            "n_channels": int(rav.shape[0]),
            "mean_R_A": float(ra.mean()),
            "mean_R_V": float(rv.mean()),
            "mean_R_AV": float(rav.mean()),
            "median_MEI": float(np.median(mei)),
            "frac_MEI_pos": float((mei > 0).mean()),
            "frac_super_additive": float(sa.mean()),
        })
        print(f"  {site:>8s}: med_MEI={np.median(mei):+.3f}  "
              f"frac_MEI>0={float((mei>0).mean()):.2%}  "
              f"frac_SA={float(sa.mean()):.2%}  "
              f"(R_A={ra.mean():.3f}, R_V={rv.mean():.3f}, R_AV={rav.mean():.3f})")

    out_csv = os.path.join(OUT_DIR, "E4_activation_mei.csv")
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sites_ord = [r["site"] for r in rows]
    x = np.arange(len(sites_ord))
    width = 0.35
    ax.bar(x - width/2, [r["frac_MEI_pos"] for r in rows], width=width,
           label="frac MEI > 0", color="#cc6677")
    ax.bar(x + width/2, [r["frac_super_additive"] for r in rows], width=width,
           label="frac SA (R_AV > R_A + R_V)", color="#4477aa")
    ax.set_xticks(x); ax.set_xticklabels(sites_ord)
    ax.set_ylabel("fraction of channels")
    ax.set_title("Per-layer multisensory-enhancement signatures (AV-clean)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "E4_activation_mei.png"), dpi=130)
    plt.close(fig)

    # also save the raw out for reuse in other experiments
    return out_av, out_a, out_v, rows


# E5. Spatial / temporal scrambling

class _PerturbedAVView(Dataset):
    PERT = ("none", "time_shuffle", "freeze_t0",
            "block_shuffle", "random_video", "zero_video")

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray,
                 perturb: str, seed: int = 1):
        assert perturb in self.PERT
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.perturb = perturb
        self.seed = int(seed)
        # for "random_video", pre-shuffle a permutation
        rng = np.random.default_rng(seed + 7)
        self.random_pair = rng.permutation(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = int(self.indices[k])
        mel, vid, label = self.base[idx]                 # vid: (1, T, H, W)
        if self.perturb == "none":
            return mel, vid, label
        if self.perturb == "time_shuffle":
            T = vid.shape[1]
            rng = np.random.default_rng(self.seed + idx)
            perm = rng.permutation(T)
            vid = vid[:, perm]
            return mel, vid, label
        if self.perturb == "freeze_t0":
            T = vid.shape[1]
            vid = vid[:, :1].expand(-1, T, -1, -1).contiguous()
            return mel, vid, label
        if self.perturb == "block_shuffle":
            # 8x8 spatial block permutation, same per frame
            T, H, W = vid.shape[1:]
            block = 8
            nh, nw = H // block, W // block
            rng = np.random.default_rng(self.seed + idx)
            perm = rng.permutation(nh * nw)
            vid_blocks = vid.unfold(2, block, block).unfold(3, block, block)
            # vid_blocks: (1, T, nh, nw, block, block)
            flat = vid_blocks.contiguous().view(1, T, nh * nw, block, block)
            flat = flat[:, :, perm]
            # un-flatten back to (1, T, H, W)
            unflat = flat.view(1, T, nh, nw, block, block)
            unflat = unflat.permute(0, 1, 2, 4, 3, 5).contiguous().view(1, T, H, W)
            return mel, unflat, label
        if self.perturb == "random_video":
            j = int(self.random_pair[k])
            _mj, vj, _yj = self.base[j]
            return mel, vj, label
        if self.perturb == "zero_video":
            return mel, torch.zeros_like(vid), label
        raise ValueError(self.perturb)


def E5_perturbations(models, val_idx, base, device):
    print("\n[E5] Spatial / temporal scrambling — AV-clean")
    rows = []
    for pert in _PerturbedAVView.PERT:
        view = _PerturbedAVView(base, val_idx, perturb=pert)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        out = _forward_AV(models["AV"][0], loader, device)
        acc = _accuracy(out["preds"], out["labels"])
        rows.append((pert, acc))
        print(f"  {pert:>14s}: AV val_acc = {acc:.4%}")

    out_csv = os.path.join(OUT_DIR, "E5_perturbations.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["perturbation", "AV_acc"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.6f}"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    perts = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    ax.bar(perts, accs, color="#cc6677")
    ax.set_xticklabels(perts, rotation=20, ha="right")
    ax.set_ylabel("AV val_acc")
    ax.set_title("AV under video perturbations")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "E5_perturbations.png"), dpi=130)
    plt.close(fig)
    return rows


# E6/E7. Race-model upper bound on accuracy

def E67_race_bound(models, val_idx, base, device):
    print("\n[E6/E7] Race-model bound on per-item accuracy")
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    a_preds, _, labels = _forward_A(models["A"][0], loader, device)
    v_preds, _, _ = _forward_V(models["V"][0], loader, device)
    av_out = _forward_AV(models["AV"][0], loader, device)
    av_preds = av_out["preds"]

    p_a = (a_preds == labels).astype(np.int32)
    p_v = (v_preds == labels).astype(np.int32)
    p_av = (av_preds == labels).astype(np.int32)

    # Bound under independent race: P(AV) ≤ P(A) + P(V) − P(A)·P(V).
    # Per-item, the bound becomes p_av ≤ p_a + p_v − p_a*p_v which is just OR.
    bound = (p_a | p_v).astype(np.int32)
    violations = (p_av > bound).astype(np.int32)

    n = len(labels)
    rows = {
        "n_items": int(n),
        "P_A":  float(p_a.mean()),
        "P_V":  float(p_v.mean()),
        "P_AV": float(p_av.mean()),
        "P_A_or_V":         float(bound.mean()),
        "frac_violations":  float(violations.mean()),
        "n_violations":     int(violations.sum()),
        "frac_AV_correct_alone": float(((p_av == 1) & (bound == 0)).mean()),
    }
    out_csv = os.path.join(OUT_DIR, "E67_race_bound.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(list(rows.keys()))
        w.writerow([f"{v:.6f}" if isinstance(v, float) else v
                    for v in rows.values()])
    print(f"  P(A)={rows['P_A']:.4%}, P(V)={rows['P_V']:.4%}, "
          f"P(AV)={rows['P_AV']:.4%}, P(A∨V)={rows['P_A_or_V']:.4%}")
    print(f"  AV-only-correct items: {rows['n_violations']}/{n} "
          f"({rows['frac_violations']:.4%})")
    return rows


# E8. Cross-modal predictability (linear probe a_mid → v_mid)

def E8_cross_predict(models, val_idx, base, device, av_acts):
    """Train a linear probe to predict V_mid from A_mid (and v.v.).

    Compares within-AV (`av_acts`) against unisensory baselines:
      - A-only: take A_mid from `WordResNet.block1` as the predictor.
      - V-only: take V_mid from `VOnlyWordResNet.visual` as the target.
    """
    print("\n[E8] Cross-modal prediction: a_mid ↔ v_mid")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    av_amid = av_acts["a_mid"].reshape(av_acts["a_mid"].shape[0], -1)
    av_vmid = av_acts["v_mid"].reshape(av_acts["v_mid"].shape[0], -1)

    # Independent unimodal mid features
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    a_only = []
    with torch.no_grad():
        for mel, _v, _y in loader:
            x = mel.unsqueeze(1).to(device)
            a_only.append(models["A"][0].block1(x).cpu().numpy())
    a_only = np.concatenate(a_only).reshape(av_amid.shape[0], -1)
    v_only = []
    with torch.no_grad():
        for _m, vid, _y in loader:
            v = vid.to(device)
            v_only.append(models["V"][0].visual(v).cpu().numpy())
    v_only = np.concatenate(v_only).reshape(av_vmid.shape[0], -1)

    # Subsample to keep memory bounded — Ridge on (5244, ~128k) is heavy.
    # Project each side to 256 random principal directions for the probe.
    def _project(x: np.ndarray, k: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((x.shape[1], k)).astype(np.float32) / np.sqrt(x.shape[1])
        return x.astype(np.float32) @ proj

    av_amid_p = _project(av_amid, seed=0)
    av_vmid_p = _project(av_vmid, seed=1)
    a_only_p = _project(a_only, seed=2)
    v_only_p = _project(v_only, seed=3)

    rows = []
    for label, X, Y in [
        ("AV_a→v", av_amid_p, av_vmid_p),
        ("AV_v→a", av_vmid_p, av_amid_p),
        ("UNI_a→v", a_only_p, v_only_p),
        ("UNI_v→a", v_only_p, a_only_p),
    ]:
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
        clf = Ridge(alpha=1.0)
        clf.fit(Xtr, Ytr)
        r2 = r2_score(Yte, clf.predict(Xte), multioutput="variance_weighted")
        rows.append((label, float(r2)))
        print(f"  {label:>10s}: R² = {r2:.4f}")

    out_csv = os.path.join(OUT_DIR, "E8_cross_predict.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["probe", "r2"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.6f}"])
    return rows


# E9. Gate readout

def E9_gate_readout(models, val_idx, base, device, av_acts, a_only_acts, v_only_acts):
    print("\n[E9] Gate readout — α and gate magnitude across conditions")
    av_g = av_acts["gate"]
    a_g = a_only_acts["gate"]
    v_g = v_only_acts["gate"]

    rows = []
    for cond, g in [("AV_full", av_g), ("audio_only", a_g), ("video_only", v_g)]:
        rows.append({
            "condition": cond,
            "gate_mean_abs": float(np.mean(np.abs(g))),
            "gate_l2_per_sample": float(np.linalg.norm(g.reshape(g.shape[0], -1),
                                                      axis=1).mean()),
            "frac_pos": float((g > 0).mean()),
        })
        print(f"  {cond:>11s}: |g|={np.mean(np.abs(g)):.4f}  "
              f"L2/sample={np.linalg.norm(g.reshape(g.shape[0], -1), axis=1).mean():.4f}  "
              f"frac>0={(g > 0).mean():.4f}")
    alpha = float(models["AV"][0].gate.alpha.detach().item())
    print(f"  α (learned) = {alpha:.4f}")
    rows.append({"condition": "alpha_param", "gate_mean_abs": alpha,
                 "gate_l2_per_sample": float("nan"), "frac_pos": float("nan")})

    out_csv = os.path.join(OUT_DIR, "E9_gate.csv")
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    return rows, alpha


# E10. Bayesian / inverse-variance check (LOOSE)

def E10_bayes(models, val_idx, base, device):
    """Loose Ernst-Banks check on per-item top-1 confidence (max softmax)."""
    print("\n[E10] Loose inverse-variance check on confidence")
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    _, prob_a, _ = _forward_A(models["A"][0], loader, device)
    _, prob_v, _ = _forward_V(models["V"][0], loader, device)
    av_out = _forward_AV(models["AV"][0], loader, device)
    prob_av = av_out["probs"]

    conf_a = prob_a.max(axis=1)
    conf_v = prob_v.max(axis=1)
    conf_av = prob_av.max(axis=1)

    var_a = float(conf_a.var())
    var_v = float(conf_v.var())
    var_av_obs = float(conf_av.var())
    pred_var_av = (var_a * var_v) / max(var_a + var_v, 1e-12)

    rows = {
        "mean_conf_A": float(conf_a.mean()),
        "mean_conf_V": float(conf_v.mean()),
        "mean_conf_AV": float(conf_av.mean()),
        "var_conf_A": var_a,
        "var_conf_V": var_v,
        "var_conf_AV_observed": var_av_obs,
        "var_conf_AV_optimal_pred": pred_var_av,
        "ratio_observed_over_optimal": var_av_obs / max(pred_var_av, 1e-12),
    }
    print(f"  conf mean: A={rows['mean_conf_A']:.3f}, V={rows['mean_conf_V']:.3f}, "
          f"AV={rows['mean_conf_AV']:.3f}")
    print(f"  conf var:  A={var_a:.4f}, V={var_v:.4f}, "
          f"AV(observed)={var_av_obs:.4f}, AV(optimal pred)={pred_var_av:.4f}")
    out_csv = os.path.join(OUT_DIR, "E10_bayes_check.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(list(rows.keys()))
        w.writerow([f"{v:.6f}" for v in rows.values()])
    return rows


# E11. Temporal congruence — Δt video shift

class _TemporalShiftView(Dataset):
    """Shift video by Δ frames (post-T_STRIDE) before passing to the model."""

    def __init__(self, base: RawNoisyAVDataset, indices: np.ndarray, delta_frames: int):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.delta = int(delta_frames)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = int(self.indices[k])
        mel, vid, label = self.base[idx]                  # vid: (1, T, H, W)
        T = vid.shape[1]
        if self.delta == 0:
            return mel, vid, label
        out = torch.zeros_like(vid)
        if self.delta > 0:
            # shift video forward in time → drop last delta frames, prepend zeros
            d = min(self.delta, T)
            out[:, d:] = vid[:, : T - d]
        else:
            d = min(-self.delta, T)
            out[:, : T - d] = vid[:, d:]
        return mel, out, label


def E11_temporal(models, val_idx, base, device):
    print("\n[E11] Temporal congruence (Δt sweep)")
    # 50 fps after t_stride=2 → 1 frame = 20 ms
    delta_ms_list = (-200, -100, -60, -40, -20, 0, 20, 40, 60, 100, 200)
    rows = []
    for delta_ms in delta_ms_list:
        delta_frames = round(delta_ms / 20.0)            # 20 ms per frame at 50 fps
        view = _TemporalShiftView(base, val_idx, delta_frames=delta_frames)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        out = _forward_AV(models["AV"][0], loader, device)
        acc = _accuracy(out["preds"], out["labels"])
        rows.append((delta_ms, delta_frames, acc))
        print(f"  Δt={delta_ms:+5d} ms  ({delta_frames:+d} frames):  AV={acc:.4%}")

    out_csv = os.path.join(OUT_DIR, "E11_temporal_window.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["delta_ms", "delta_frames", "AV_acc"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    deltas = [r[0] for r in rows]
    accs = [r[2] for r in rows]
    ax.plot(deltas, accs, "o-", color="#cc6677")
    ax.axvline(0, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("video shift Δt (ms)  [+ = video lags]")
    ax.set_ylabel("AV val_acc")
    ax.set_title("Temporal congruence: AV-clean")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "E11_temporal_window.png"), dpi=130)
    plt.close(fig)
    return rows


# Main

def _verdict(name: str, finding: str) -> str:
    return f"- **{name}** — {finding}"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Data
    print("Loading dataset + splits + models...")
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    s = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                   weights_only=False)
    val_idx = s["val_idx"]
    idx_to_label = base.idx_to_label
    models = _load_models(device)
    print(f"  V-only ckpt loaded from: {models.get('_V_path', V_CKPT)}")
    for name in ("A", "V", "AV"):
        m, ck = models[name]
        n = sum(p.numel() for p in m.parameters())
        ba = ck.get("best_val_acc", float("nan"))
        print(f"  {name:>2s}: params={n:,}, best_val_acc={ba:.4%}")

    findings = []

    # E1 — inverse effectiveness
    t0 = time.time()
    e1 = E1_inverse_effectiveness(models, val_idx, base, device)
    print(f"  ... E1 took {time.time()-t0:.1f}s")
    deltas = [r[3] for r in e1]
    findings.append(_verdict("E1 inverse-effectiveness",
        f"AV − A gap at σ=0: {deltas[0]*100:+.2f} pp; at σ=0.05: {deltas[5]*100:+.2f} pp; "
        f"at σ=0.5: {deltas[-1]*100:+.2f} pp. "
        f"{'GROWS' if deltas[5] > deltas[0] + 0.02 else 'DOES NOT GROW'} with σ."))

    # E2 — graduated dropout
    t0 = time.time()
    e2 = E2_graduated_dropout(models, val_idx, base, device)
    print(f"  ... E2 took {time.time()-t0:.1f}s")
    findings.append(_verdict("E2 graduated dropout",
        f"AV with video×0={e2[-1][1]*100:.2f}% (reduces to A-only-ish), "
        f"AV with audio×0={e2[-1][2]*100:.2f}% (V-only-ish). "
        f"video×0.5={e2[2][1]*100:.2f}%, audio×0.5={e2[2][2]*100:.2f}%."))

    # E4 — activation MEI (run before E3 so E8/E9 can reuse acts)
    t0 = time.time()
    out_av_acts, out_a_acts, out_v_acts, e4 = E4_activation_mei(
        models, val_idx, base, device,
    )
    print(f"  ... E4 took {time.time()-t0:.1f}s")
    findings.append(_verdict("E4 activation MEI",
        f"frac MEI>0 — a_mid: {e4[0]['frac_MEI_pos']:.2f}, "
        f"gate: {e4[2]['frac_MEI_pos']:.2f}, "
        f"block2: {e4[3]['frac_MEI_pos']:.2f}, "
        f"penult: {e4[4]['frac_MEI_pos']:.2f}; "
        f"frac SA — block2: {e4[3]['frac_super_additive']:.2f}, "
        f"penult: {e4[4]['frac_super_additive']:.2f}."))

    # E9 — gate readout (uses cached acts)
    t0 = time.time()
    e9, alpha = E9_gate_readout(models, val_idx, base, device,
                                 out_av_acts, out_a_acts, out_v_acts)
    print(f"  ... E9 took {time.time()-t0:.1f}s")
    findings.append(_verdict("E9 gate readout",
        f"α = {alpha:.4f}; mean |gate|: AV={e9[0]['gate_mean_abs']:.4f}, "
        f"audio_only={e9[1]['gate_mean_abs']:.4f}, "
        f"video_only={e9[2]['gate_mean_abs']:.4f}."))

    # E5 — perturbations
    t0 = time.time()
    e5 = E5_perturbations(models, val_idx, base, device)
    print(f"  ... E5 took {time.time()-t0:.1f}s")
    findings.append(_verdict("E5 perturbations",
        " | ".join(f"{k}={v*100:.2f}%" for k, v in e5)))

    # E3 — McGurk
    t0 = time.time()
    e3 = E3_mcgurk(models, val_idx, base, device, idx_to_label)
    print(f"  ... E3 took {time.time()-t0:.1f}s")
    findings.append(_verdict("E3 McGurk-style cross-pair",
        " | ".join(
            f"{r['conflict_type']}: AV audio_cap={r['AV_audio_capture']*100:.2f}%, "
            f"vis_cap={r['AV_visual_capture']*100:.2f}%, "
            f"third={r['AV_third_word']*100:.2f}%, "
            f"A_only audio_cap={r['A_only_audio_capture']*100:.2f}%"
            for r in e3)))

    # E11 — temporal
    t0 = time.time()
    e11 = E11_temporal(models, val_idx, base, device)
    print(f"  ... E11 took {time.time()-t0:.1f}s")
    accs_at = {r[0]: r[2] for r in e11}
    findings.append(_verdict("E11 temporal window",
        f"AV at Δt=0: {accs_at[0]*100:.2f}%, Δt=±100ms: "
        f"{accs_at.get(100, 0)*100:.2f}/{accs_at.get(-100, 0)*100:.2f}%, "
        f"Δt=±200ms: {accs_at.get(200, 0)*100:.2f}/{accs_at.get(-200, 0)*100:.2f}%."))

    # E6/E7 — race bound
    t0 = time.time()
    e67 = E67_race_bound(models, val_idx, base, device)
    print(f"  ... E6/E7 took {time.time()-t0:.1f}s")
    findings.append(_verdict("E6/E7 race-bound",
        f"P(A)={e67['P_A']:.2%}, P(V)={e67['P_V']:.2%}, P(AV)={e67['P_AV']:.2%}, "
        f"P(A∨V)={e67['P_A_or_V']:.2%}; AV-only-correct items: "
        f"{e67['n_violations']} ({e67['frac_violations']:.2%})."))

    # E8 — cross-predict (uses cached acts)
    t0 = time.time()
    e8 = E8_cross_predict(models, val_idx, base, device, out_av_acts)
    print(f"  ... E8 took {time.time()-t0:.1f}s")
    findings.append(_verdict("E8 cross-predict",
        " | ".join(f"{k}: R²={v:.3f}" for k, v in e8)))

    # E10 — Bayes (uses fresh forward passes for confidence)
    t0 = time.time()
    e10 = E10_bayes(models, val_idx, base, device)
    print(f"  ... E10 took {time.time()-t0:.1f}s")
    findings.append(_verdict("E10 Bayes (LOOSE)",
        f"conf var: A={e10['var_conf_A']:.4f}, V={e10['var_conf_V']:.4f}, "
        f"AV(obs)={e10['var_conf_AV_observed']:.4f}, "
        f"AV(opt)={e10['var_conf_AV_optimal_pred']:.4f} "
        f"(ratio = {e10['ratio_observed_over_optimal']:.3f})."))

    # Markdown summary
    md_path = os.path.join(SCRIPT_DIR, "analysis", "MSI_RESULTS.md")
    with open(md_path, "w") as f:
        f.write("# Multisensory-Integration Battery — Results\n\n")
        f.write("Run on the trained checkpoints over the shared val partition "
                "(val_idx sha `03c5a87a…cdf07add`). All raw CSVs and plots are "
                "in `analysis/msi/`.\n\n")
        f.write(f"V-only checkpoint used: `{os.path.basename(models['_V_path'])}` "
                f"(best_val_acc = {models['V'][1].get('best_val_acc', float('nan')):.4%}).\n\n")
        f.write("## Findings (one line each)\n\n")
        for line in findings:
            f.write(line + "\n")
        f.write("\nSee `analysis/msi/E*_*.csv` for raw numbers and "
                "`analysis/msi/E*_*.png` for plots.\n")

    print(f"\nAll done. Markdown summary at {md_path}")


if __name__ == "__main__":
    main()
