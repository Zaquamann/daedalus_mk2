#!/usr/bin/env python3
"""Tier-1 cross-variant mechanism comparison. Runs 5 probes (noise sweep,
W_v channel lesion, α-sweep for additive, gate-inhibition, viseme
decodability) across AV-fused, D3.2 late, D3.10 additive, D3.1 early.
Writes CSVs to `analysis/deepdive/` and a writeup at
`analysis/AV_INTEGRATION_TIER1_CROSS_VARIANT.md`."""

from __future__ import annotations

import csv
import hashlib
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from analyze_av_msi import BATCH_SIZE, T_STRIDE, _ValAVView, _NoisyAudioView, _accuracy
from analyze_av_deepdive import _NoisyVideoView, _NoisyAVView
from analyze_av_phonetics import viseme_class as _viseme_from_label
from dataset_raw_noisy import RawNoisyAVDataset

from model_av import AVWordResNet
from model_av_late import AVLateFusionWordResNet
from model_av_additive import AVAdditiveWordResNet
from model_av_early import (
    AVEarlyFusionWordResNet, _video_to_mel_channels, CV, TARGET_H, TARGET_W,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "deepdive")
MD_PATH = os.path.join(SCRIPT_DIR, "analysis",
                        "AV_INTEGRATION_TIER1_CROSS_VARIANT.md")
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_PATHS = {
    "AV_fused":   os.path.join(SCRIPT_DIR, "models", "av_fused.pt"),
    "D3_2_late":  os.path.join(SCRIPT_DIR, "models", "av_fused_late.pt"),
    "D3_10_add":  os.path.join(SCRIPT_DIR, "models", "av_fused_additive.pt"),
    "D3_1_early": os.path.join(SCRIPT_DIR, "models", "av_fused_early.pt"),
}

VARIANT_FLAVOUR = {
    "AV_fused":   "mid_mult",
    "D3_2_late":  "late",
    "D3_10_add":  "mid_add",
    "D3_1_early": "early",
}
GATE_BEARING = {"mid_mult", "mid_add"}

# σ levels for the 1D sweeps. 8 levels each, log-stepped.
SIGMA_A_LEVELS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
SIGMA_V_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20)

# α-sweep for additive variant (D3.10). Lead's spec: {0, .5, 1, 1.31, 2, 3, 5}.
ADDITIVE_ALPHA_LEVELS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)  # trained α added at runtime

VAL_HASH_PREFIX = "03c5a87a"

NUM_WORKERS = 4


# Model loading + val-split sanity

def _load_variant(name: str, device: torch.device):
    ckpt = torch.load(CKPT_PATHS[name], weights_only=False, map_location="cpu")
    n_classes = len(ckpt["label_to_idx"])
    flavour = VARIANT_FLAVOUR[name]
    if flavour == "mid_mult":
        m = AVWordResNet(n_classes)
    elif flavour == "late":
        m = AVLateFusionWordResNet(n_classes)
    elif flavour == "mid_add":
        m = AVAdditiveWordResNet(n_classes)
    elif flavour == "early":
        m = AVEarlyFusionWordResNet(n_classes)
    else:
        raise ValueError(flavour)
    m.load_state_dict(ckpt["model_state_dict"])
    m = m.to(device).eval()
    return m, ckpt


def _val_idx_with_check():
    splits = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)
    val_idx = splits["val_idx"]
    if hasattr(val_idx, "numpy"):
        val_idx = val_idx.numpy()
    h = hashlib.sha256(np.asarray(val_idx, dtype=np.int64).tobytes()).hexdigest()
    assert h.startswith(VAL_HASH_PREFIX), \
        f"val_idx sha drift: {h[:16]} != {VAL_HASH_PREFIX}…"
    return val_idx, h


# Variant-aware forward

@torch.no_grad()
def _forward_variant(name, model, loader, device,
                       video_kind: str = "real",
                       audio_kind: str = "real") -> dict:
    preds, labels = [], []
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        if audio_kind == "zero":
            mel = torch.zeros_like(mel)
        if video_kind == "zero":
            vid = torch.zeros_like(vid)
            vin = vid
        elif video_kind == "none":
            vin = None
        else:
            vin = vid
        logits = model(mel, vin)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return {"preds": np.concatenate(preds),
            "labels": np.concatenate(labels)}


# Penultimate-feature extraction (one cached pass per variant)

@torch.no_grad()
def _extract_penult(name, model, loader, device) -> dict:
    """Return penultimate features (post-GAP, pre-FC) + labels.

    Also caches a_mid, v_mid for gate-bearing variants (for downstream
    lesion / α-sweep speedups), and the V-branch GAP for the late-fusion
    variant.
    """
    flavour = VARIANT_FLAVOUR[name]
    penults, labels = [], []
    a_mids, v_mids, vgaps = [], [], []
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        labels.append(y.numpy())
        if flavour in {"mid_mult", "mid_add"}:
            a_mid = model.audio_block1(mel)
            v_mid = model.visual(vid)
            a_mids.append(a_mid.cpu())
            v_mids.append(v_mid.cpu())
            Wa = model.gate.Wa(a_mid)
            Wv = model.gate.Wv(v_mid)
            g = torch.sigmoid(Wa + Wv)
            if flavour == "mid_mult":
                a_fused = a_mid * (1.0 + model.gate.alpha * g)
            else:
                a_fused = a_mid + model.gate.alpha * g
            x = model.audio_block2(a_fused)
            pen = model.gap(x).flatten(1)               # (B, 128)
        elif flavour == "late":
            a = model.audio_block1(mel)
            a = model.audio_block2(a)
            a = model.audio_gap(a).flatten(1)            # (B, 128)
            v = model.visual(vid)
            vgap = model.visual_gap(v).flatten(1)        # (B, 64)
            vgaps.append(vgap.cpu())
            pen = torch.cat([a, vgap], dim=1)            # (B, 192)
        elif flavour == "early":
            v_proj = _video_to_mel_channels(vid).to(mel.dtype)
            x = torch.cat([mel, v_proj], dim=1)
            x = model.block1(x)
            x = model.block2(x)
            pen = model.gap(x).flatten(1)                # (B, 192)
        else:
            raise ValueError(flavour)
        penults.append(pen.cpu().numpy())
    cache = {
        "penult": np.concatenate(penults, axis=0),
        "labels": np.concatenate(labels, axis=0),
    }
    if a_mids:
        cache["a_mid"] = torch.cat(a_mids, dim=0)
        cache["v_mid"] = torch.cat(v_mids, dim=0)
    if vgaps:
        cache["v_gap"] = torch.cat(vgaps, dim=0)         # (N, 64)
    return cache


@torch.no_grad()
def _gate_eval_from_cache(name, model, a_mid_c, v_mid_c, labels_c, device,
                            batch_size: int = 256,
                            wv_mask: torch.Tensor | None = None,
                            alpha_override: float | None = None) -> dict:
    flavour = VARIANT_FLAVOUR[name]
    assert flavour in GATE_BEARING
    preds, labels = [], []
    n = a_mid_c.shape[0]
    for i in range(0, n, batch_size):
        a_mid = a_mid_c[i:i+batch_size].to(device, non_blocking=True)
        v_mid = v_mid_c[i:i+batch_size].to(device, non_blocking=True)
        Wa_out = model.gate.Wa(a_mid)
        Wv_out = model.gate.Wv(v_mid)
        if wv_mask is not None:
            Wv_out = Wv_out * wv_mask.view(1, -1, 1, 1)
        g = torch.sigmoid(Wa_out + Wv_out)
        alpha = (model.gate.alpha if alpha_override is None
                  else torch.tensor(float(alpha_override),
                                     device=a_mid.device))
        if flavour == "mid_mult":
            a_fused = a_mid * (1.0 + alpha * g)
        else:
            a_fused = a_mid + alpha * g
        x = model.audio_block2(a_fused)
        pen = model.gap(x).flatten(1)
        logits = model.fc(model.dropout(pen))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(labels_c[i:i+batch_size].numpy())
    return {"preds": np.concatenate(preds),
            "labels": np.concatenate(labels)}


@torch.no_grad()
def _late_eval_with_vgap_mask(model, loader, device,
                                vgap_mask: torch.Tensor) -> dict:
    """For D3.2 late fusion: lesion specific channels of the V-branch GAP
    output (the 64-d vector concat'd with audio's 128-d before fc)."""
    preds, labels = [], []
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        a = model.audio_block1(mel)
        a = model.audio_block2(a)
        a = model.audio_gap(a).flatten(1)
        v = model.visual(vid)
        vgap = model.visual_gap(v).flatten(1)
        vgap = vgap * vgap_mask.view(1, -1)
        x = torch.cat([a, vgap], dim=1)
        x = model.dropout(x)
        logits = model.fc(x)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return {"preds": np.concatenate(preds),
            "labels": np.concatenate(labels)}


# 1. D1.1-style noise sweep with AV-gain

def probe_1_noise_sweep(variants, val_idx, base, device) -> dict:
    """For each variant: AV(σ_a sweep, σ_v=0) and AV(σ_a=0, σ_v sweep) plus
    the variant's own A-only baseline (video=None) at each σ_a level.

    AV gain = AV(σ_a, σ_v=0) − AV_A_only(σ_a)  — within-variant unisensory
    baseline. The σ_v sweep doesn't yield a direct gain interpretation (clean
    audio carries most of the signal) so we just report AV acc at each σ_v.

    Output CSV columns:
       variant, sweep, sigma_a_per_rms, sigma_v_per_pixstd,
       AV_acc, A_only_acc, AV_gain_pp
    """
    print("\n[probe 1] D1.1-style σ_a / σ_v noise sweeps ─────────────")
    rows = []
    results = {n: {"sigma_a": {}, "sigma_v": {}} for n in variants.keys()}

    # σ_a sweep (σ_v = 0)
    print("  σ_a sweep (σ_v=0):")
    for sa in SIGMA_A_LEVELS:
        view = _NoisyAVView(base, val_idx, sigma_a_mult=sa,
                              sigma_v_mult=0.0, seed=0)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
        for name, (m, _ck) in variants.items():
            t0 = time.time()
            full = _forward_variant(name, m, loader, device,
                                       video_kind="real", audio_kind="real")
            a_only = _forward_variant(name, m, loader, device,
                                         video_kind="none", audio_kind="real")
            av_acc = _accuracy(full["preds"], full["labels"])
            a_acc = _accuracy(a_only["preds"], a_only["labels"])
            gain = (av_acc - a_acc) * 100.0
            results[name]["sigma_a"][sa] = {"AV": av_acc, "A": a_acc,
                                              "gain_pp": gain}
            rows.append([name, "sigma_a", f"{sa:.4f}", "0.0000",
                          f"{av_acc:.6f}", f"{a_acc:.6f}",
                          f"{gain:+.4f}"])
            print(f"    σ_a={sa:6.4f}  {name:>11s}  AV={av_acc:.3%}  "
                  f"A-only={a_acc:.3%}  gain={gain:+.2f} pp "
                  f"({time.time()-t0:.1f}s)")

    # σ_v sweep (σ_a = 0)
    print("  σ_v sweep (σ_a=0):")
    for sv in SIGMA_V_LEVELS:
        view = _NoisyAVView(base, val_idx, sigma_a_mult=0.0,
                              sigma_v_mult=sv, seed=0)
        loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
        for name, (m, _ck) in variants.items():
            t0 = time.time()
            full = _forward_variant(name, m, loader, device,
                                       video_kind="real", audio_kind="real")
            a_only = _forward_variant(name, m, loader, device,
                                         video_kind="none", audio_kind="real")
            av_acc = _accuracy(full["preds"], full["labels"])
            a_acc = _accuracy(a_only["preds"], a_only["labels"])
            gain = (av_acc - a_acc) * 100.0
            results[name]["sigma_v"][sv] = {"AV": av_acc, "A": a_acc,
                                              "gain_pp": gain}
            rows.append([name, "sigma_v", "0.0000", f"{sv:.4f}",
                          f"{av_acc:.6f}", f"{a_acc:.6f}",
                          f"{gain:+.4f}"])
            print(f"    σ_v={sv:6.4f}  {name:>11s}  AV={av_acc:.3%}  "
                  f"A-only={a_acc:.3%}  gain={gain:+.2f} pp "
                  f"({time.time()-t0:.1f}s)")

    out_csv = os.path.join(OUT_DIR, "D1_cross_variant_noise.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["variant", "sweep",
                    "sigma_a_per_rms", "sigma_v_per_pixstd",
                    "AV_acc", "A_only_acc", "AV_gain_pp"])
        w.writerows(rows)
    print(f"  wrote {out_csv}")
    return results


# 2. D3.5-style channel lesion (per architecture, where applicable)

def probe_2_channel_lesion(variants, val_idx, base, device,
                             penult_caches) -> dict:
    """Channel-by-channel lesion at each variant's appropriate site:

      mid_mult / mid_add → lesion `gate.Wv` output (64 channels)
      late                → lesion `visual_gap.flatten()` output (64 dims)
      early               → N/A (no isolated V stream)
    """
    print("\n[probe 2] D3.5-style W_v / V-branch channel lesion "
          "─────────────")
    rows = []
    impacts_all = {}

    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    for name, (m, _ck) in variants.items():
        flavour = VARIANT_FLAVOUR[name]
        # Baseline from cache
        if flavour in GATE_BEARING:
            cache = penult_caches[name]
            baseline = _accuracy(
                _gate_eval_from_cache(name, m, cache["a_mid"], cache["v_mid"],
                                          torch.from_numpy(cache["labels"]),
                                          device)["preds"],
                cache["labels"],
            )
            n_ch = m.gate.Wv.weight.shape[0]
            print(f"  {name:>11s}  flavour={flavour}  baseline={baseline:.4%}  "
                  f"(lesion W_v 1×1 conv, 64 ch)")
            impacts = np.zeros(n_ch, dtype=np.float64)
            t0 = time.time()
            for c in range(n_ch):
                mask = torch.ones(n_ch, device=device)
                mask[c] = 0.0
                out = _gate_eval_from_cache(
                    name, m, cache["a_mid"], cache["v_mid"],
                    torch.from_numpy(cache["labels"]),
                    device, wv_mask=mask)
                acc = _accuracy(out["preds"], cache["labels"])
                impact = (baseline - acc) * 100.0
                impacts[c] = impact
                rows.append([name, flavour, c, f"{baseline:.6f}",
                              f"{acc:.6f}", f"{impact:+.4f}"])
            print(f"    done in {time.time()-t0:.1f}s; "
                  f"impacts: mean={impacts.mean():+.3f}, "
                  f"max={impacts.max():+.3f} (ch {int(np.argmax(impacts))}), "
                  f"min={impacts.min():+.3f}")
            top = np.argsort(-impacts)[:8]
            print(f"    top-8: " + ", ".join(
                f"ch{c}={impacts[c]:+.3f}" for c in top))
            impacts_all[name] = impacts
        elif flavour == "late":
            cache = penult_caches[name]
            baseline_acc = None  # compute fresh w/ unmasked v_gap
            # baseline = no mask
            mask = torch.ones(64, device=device)
            out_b = _late_eval_with_vgap_mask(m, loader, device, mask)
            baseline = _accuracy(out_b["preds"], out_b["labels"])
            print(f"  {name:>11s}  flavour={flavour}  baseline={baseline:.4%}  "
                  f"(lesion V-branch GAP, 64-d)")
            impacts = np.zeros(64, dtype=np.float64)
            t0 = time.time()
            for c in range(64):
                mask = torch.ones(64, device=device)
                mask[c] = 0.0
                out = _late_eval_with_vgap_mask(m, loader, device, mask)
                acc = _accuracy(out["preds"], out["labels"])
                impact = (baseline - acc) * 100.0
                impacts[c] = impact
                rows.append([name, flavour, c, f"{baseline:.6f}",
                              f"{acc:.6f}", f"{impact:+.4f}"])
                if c % 16 == 0:
                    print(f"    [{c}/64] acc={acc:.4%} Δ={impact:+.2f} pp "
                          f"({time.time()-t0:.1f}s)")
            print(f"    done in {time.time()-t0:.1f}s; "
                  f"impacts: mean={impacts.mean():+.3f}, "
                  f"max={impacts.max():+.3f} (ch {int(np.argmax(impacts))}), "
                  f"min={impacts.min():+.3f}")
            top = np.argsort(-impacts)[:8]
            print(f"    top-8: " + ", ".join(
                f"ch{c}={impacts[c]:+.3f}" for c in top))
            impacts_all[name] = impacts
        elif flavour == "early":
            print(f"  {name:>11s}  flavour={flavour}  → N/A "
                  f"(no isolated V stream; would require lesioning input "
                  f"channels which doubles as architectural ablation)")
            # Still emit one row marking N/A for the CSV
            rows.append([name, flavour, "N/A", "N/A", "N/A", "N/A"])
            impacts_all[name] = None
        else:
            raise ValueError(flavour)

    out_csv = os.path.join(OUT_DIR, "D3_cross_variant_Wv_lesion.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["variant", "flavour", "channel", "AV_acc_baseline",
                    "AV_acc_lesioned", "delta_pp"])
        w.writerows(rows)
    print(f"  wrote {out_csv}")
    return impacts_all


# 3. D3.7-style α-sweep for additive variant

def probe_3_alpha_sweep_additive(variants, penult_caches, device) -> dict:
    print("\n[probe 3] D3.7-style α-sweep (D3.10 additive) ─────────────")
    name = "D3_10_add"
    m, _ck = variants[name]
    cache = penult_caches[name]
    natural_alpha = float(m.gate.alpha.detach().item())
    levels = sorted(set(list(ADDITIVE_ALPHA_LEVELS) + [natural_alpha]))
    rows = []
    accs = []
    for a in levels:
        out = _gate_eval_from_cache(
            name, m, cache["a_mid"], cache["v_mid"],
            torch.from_numpy(cache["labels"]),
            device, alpha_override=a)
        acc = _accuracy(out["preds"], cache["labels"])
        accs.append(acc)
        is_trained = abs(a - natural_alpha) < 1e-6
        rows.append([f"{a:.6f}", f"{acc:.6f}",
                      f"{natural_alpha:.6f}",
                      "1" if is_trained else "0"])
        tag = " (trained)" if is_trained else ""
        print(f"  α={a:6.3f}: acc={acc:.4%}{tag}")
    out_csv = os.path.join(OUT_DIR, "D3_alpha_sweep_AV_additive.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "AV_acc", "alpha_trained", "is_trained_alpha"])
        w.writerows(rows)
    print(f"  wrote {out_csv}")
    return {
        "levels": levels, "accs": accs,
        "trained_alpha": natural_alpha,
        "acc_at_trained": accs[levels.index(natural_alpha)],
        "acc_at_zero": accs[levels.index(0.0)],
        "acc_at_best": max(accs),
        "alpha_best": levels[int(np.argmax(accs))],
    }


# 4. E9-style gate-output stats — gate-bearing variants

@torch.no_grad()
def _collect_gate_output(name, model, loader, device,
                           video_kind: str, audio_kind: str) -> dict:
    flavour = VARIANT_FLAVOUR[name]
    assert flavour in GATE_BEARING
    sums = {"abs_g": 0.0, "abs_ag": 0.0, "g_total": 0,
            "frac_g_gt_half": 0,
            "res_abs": 0.0, "res_pos": 0, "res_total": 0,
            "res_l2": 0.0, "n": 0}
    for mel, vid, _y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        if audio_kind == "zero":
            mel = torch.zeros_like(mel)
        a_mid = model.audio_block1(mel)
        if video_kind == "zero":
            v_mid = torch.zeros_like(a_mid)
        else:
            v_mid = model.visual(vid)
        g = torch.sigmoid(model.gate.Wa(a_mid) + model.gate.Wv(v_mid))
        ag = model.gate.alpha * g
        # residual = a_fused − a_mid
        if flavour == "mid_mult":
            res = a_mid * ag
        else:
            res = ag.clone()
        sums["abs_g"]         += float(g.abs().sum().cpu())
        sums["abs_ag"]        += float(ag.abs().sum().cpu())
        sums["frac_g_gt_half"]+= int((g > 0.5).sum().cpu())
        sums["g_total"]       += g.numel()
        sums["res_abs"]       += float(res.abs().sum().cpu())
        sums["res_pos"]       += int((res > 0).sum().cpu())
        sums["res_total"]     += res.numel()
        sums["res_l2"]        += float(res.flatten(1).norm(dim=1).sum().cpu())
        sums["n"]             += mel.shape[0]
    return {
        "mean_abs_g":      sums["abs_g"]         / sums["g_total"],
        "mean_abs_alpha_g":sums["abs_ag"]        / sums["g_total"],
        "frac_g_gt_half":  sums["frac_g_gt_half"]/ sums["g_total"],
        "mean_abs_res":    sums["res_abs"]       / sums["res_total"],
        "frac_res_pos":    sums["res_pos"]       / sums["res_total"],
        "mean_res_l2":     sums["res_l2"]        / sums["n"],
        "alpha":           float(model.gate.alpha.detach().item()),
    }


def probe_4_inhibition(variants, val_idx, base, device) -> dict:
    print("\n[probe 4] E9-style gate-output (inhibition) probe "
          "─────────────")
    rows = []
    out_all = {}
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
    for name in ("AV_fused", "D3_10_add"):
        m, _ck = variants[name]
        out_all[name] = {}
        for cond_label, vk, ak in [("AV_full",    "real", "real"),
                                       ("audio_only", "zero", "real"),
                                       ("video_only", "real", "zero")]:
            t0 = time.time()
            s = _collect_gate_output(name, m, loader, device,
                                          video_kind=vk, audio_kind=ak)
            print(f"  {name:>11s}  {cond_label:>11s}: "
                  f"|g|={s['mean_abs_g']:.4f}  "
                  f"|α·g|={s['mean_abs_alpha_g']:.4f}  "
                  f"frac(g>.5)={s['frac_g_gt_half']:.3f}  "
                  f"|res|={s['mean_abs_res']:.4f}  "
                  f"frac(res>0)={s['frac_res_pos']:.3f}  "
                  f"α={s['alpha']:.4f}  ({time.time()-t0:.1f}s)")
            rows.append([name, VARIANT_FLAVOUR[name], cond_label,
                          f"{s['mean_abs_g']:.6f}",
                          f"{s['mean_abs_alpha_g']:.6f}",
                          f"{s['frac_g_gt_half']:.6f}",
                          f"{s['mean_abs_res']:.6f}",
                          f"{s['frac_res_pos']:.6f}",
                          f"{s['mean_res_l2']:.6f}",
                          f"{s['alpha']:.6f}"])
            out_all[name][cond_label] = s
    out_csv = os.path.join(OUT_DIR, "D3_inhibition_cross_variant.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["variant", "flavour", "condition",
                    "mean_abs_g", "mean_abs_alpha_g", "frac_g_gt_half",
                    "mean_abs_residual", "frac_residual_pos",
                    "mean_residual_l2_per_sample", "alpha"])
        w.writerows(rows)
    print(f"  wrote {out_csv}")
    return out_all


# 5. D2.5-style viseme decodability (5-fold logreg on penult)

def _probe_viseme_5fold(X, y, max_iter: int = 1500, C: float = 1.0,
                          seed: int = 0) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, bal_accs = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        clf = LogisticRegression(max_iter=max_iter, C=C)
        clf.fit(X_tr, y[tr])
        pred = clf.predict(X_te)
        accs.append(accuracy_score(y[te], pred))
        bal_accs.append(balanced_accuracy_score(y[te], pred))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "bal_acc_mean": float(np.mean(bal_accs)),
        "bal_acc_std": float(np.std(bal_accs)),
    }


def probe_5_viseme(penult_caches, idx_to_label) -> dict:
    print("\n[probe 5] D2.5-style viseme decodability (5-fold LR on penult)"
          " ─────────────")
    rows = []
    out_all = {}
    for name, cache in penult_caches.items():
        feats = cache["penult"].astype(np.float32)
        y_label = cache["labels"]
        visemes = np.asarray([_viseme_from_label(idx_to_label[int(t)])
                                 for t in y_label])
        keep = visemes != "other"
        f = feats[keep]
        v = visemes[keep]
        n_classes = len(set(v))
        t0 = time.time()
        res = _probe_viseme_5fold(f, v)
        print(f"  {name:>11s}  ({VARIANT_FLAVOUR[name]:>9s})  "
              f"n_kept={keep.sum()}  n_viseme_classes={n_classes}  "
              f"d_penult={feats.shape[1]}  "
              f"acc={res['acc_mean']*100:.2f}% (±{res['acc_std']*100:.2f}) "
              f"bal={res['bal_acc_mean']*100:.2f}% (±{res['bal_acc_std']*100:.2f}) "
              f"({time.time()-t0:.1f}s)")
        rows.append([name, VARIANT_FLAVOUR[name],
                      int(keep.sum()), n_classes,
                      int(feats.shape[1]),
                      f"{res['acc_mean']:.6f}",
                      f"{res['acc_std']:.6f}",
                      f"{res['bal_acc_mean']:.6f}",
                      f"{res['bal_acc_std']:.6f}"])
        out_all[name] = res
    out_csv = os.path.join(OUT_DIR, "D2_viseme_probe_cross_variant.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["variant", "flavour", "n_samples", "n_viseme_classes",
                    "penult_dim",
                    "acc_mean", "acc_std", "bal_acc_mean", "bal_acc_std"])
        w.writerows(rows)
    print(f"  wrote {out_csv}")
    return out_all


# Markdown writeup

def _write_markdown(variants, val_sha, noise_results, channel_impacts,
                     alpha_summary, inhibition, viseme_results) -> None:
    with open(MD_PATH, "w") as f:
        f.write("# AV-Integration Tier-1 Cross-Variant Comparison\n\n")
        f.write(f"Run on the shared val partition "
                 f"(sha `{val_sha[:16]}…`).\n")
        f.write("All probes use deterministic seeds. Raw CSVs in "
                 "`analysis/deepdive/`.\n\n")

        f.write("## Models\n\n")
        f.write("| variant | flavour | params (trainable) | best_val_acc |\n")
        f.write("|---|---|---|---|\n")
        PARAMS = {"AV_fused": 522509, "D3_2_late": 525836,
                   "D3_10_add": 522509, "D3_1_early": 530996}
        for name, (_m, ck) in variants.items():
            f.write(f"| {name} | {VARIANT_FLAVOUR[name]} | "
                    f"{PARAMS[name]:,} | "
                    f"{ck.get('best_val_acc', float('nan'))*100:.4f}% |\n")

        f.write("\n## 1. Noise sweep — AV gain\n\n")
        f.write("AV gain = AV_acc − A-only acc (each variant's own A-only "
                 "path: `model(audio, video=None)`).\n\n")
        f.write("**σ_a sweep (σ_v=0):**\n\n")
        f.write("| variant | σ_a=0 | σ_a=0.01 | σ_a=0.05 | σ_a=0.1 | "
                "σ_a=0.2 | σ_a=0.5 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name in variants.keys():
            d = noise_results[name]["sigma_a"]
            picks = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
            cells = [(d[p]["AV"]*100, d[p]["gain_pp"]) for p in picks]
            row = f"| {name} | " + " | ".join(
                f"{av:.1f}% ({g:+.1f})" for av, g in cells) + " |\n"
            f.write(row)

        f.write("\n**σ_v sweep (σ_a=0):**\n\n")
        f.write("| variant | σ_v=0 | σ_v=0.1 | σ_v=0.2 | σ_v=0.4 | "
                "σ_v=0.8 | σ_v=1.6 | σ_v=3.2 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for name in variants.keys():
            d = noise_results[name]["sigma_v"]
            picks = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
            cells = [(d[p]["AV"]*100, d[p]["gain_pp"]) for p in picks]
            row = f"| {name} | " + " | ".join(
                f"{av:.1f}% ({g:+.1f})" for av, g in cells) + " |\n"
            f.write(row)

        f.write("\n*(format: `AV_acc (AV − A-only gain in pp)`)*\n")

        f.write("\n## 2. Channel-by-channel V-side lesion\n\n")
        f.write("Lesion site varies by architecture:\n\n")
        f.write("- `mid_mult` / `mid_add`: zero one channel of "
                 "`gate.Wv` output (64 channels).\n")
        f.write("- `late`: zero one channel of the V-branch GAP output "
                 "(64-d, pre-concat).\n")
        f.write("- `early`: N/A — no isolated V stream.\n\n")
        f.write("| variant | top-8 channel impacts (pp) | "
                 "top-4 share of positive impacts | max impact |\n")
        f.write("|---|---|---|---|\n")
        for name in variants.keys():
            imp = channel_impacts[name]
            if imp is None:
                f.write(f"| {name} | N/A | N/A | N/A |\n")
                continue
            top = np.argsort(-imp)[:8]
            top_str = ", ".join(f"ch{c}={imp[c]:+.2f}" for c in top)
            pos = imp[imp > 0]
            share = (np.sort(imp)[-4:].sum() / (pos.sum() + 1e-12)
                      if len(pos) else float("nan"))
            f.write(f"| {name} | {top_str} | "
                    f"{share*100:.1f}% | {imp.max():+.3f} |\n")

        f.write("\n## 3. α-sweep — D3.10 additive variant (D3.7 analog)\n\n")
        f.write(f"Trained α = **{alpha_summary['trained_alpha']:.4f}** | "
                 f"acc_at_trained = **{alpha_summary['acc_at_trained']*100:.2f}%** | "
                 f"acc_at_α=0 = {alpha_summary['acc_at_zero']*100:.2f}% | "
                 f"acc_at_best_α({alpha_summary['alpha_best']:.2f}) = "
                 f"{alpha_summary['acc_at_best']*100:.2f}%\n\n")
        f.write("| α | AV acc |\n")
        f.write("|---|---|\n")
        for a, acc in zip(alpha_summary["levels"], alpha_summary["accs"]):
            tag = " ← trained" if abs(
                a - alpha_summary['trained_alpha']) < 1e-6 else ""
            f.write(f"| {a:.3f}{tag} | {acc*100:.2f}% |\n")

        f.write("\n## 4. E9 inhibition probe — gate-bearing variants\n\n")
        f.write("| variant | condition | mean\\|g\\| | mean\\|α·g\\| | "
                "frac(g>0.5) | mean\\|res\\| | frac(res>0) | α |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for name in ("AV_fused", "D3_10_add"):
            for cond in ("AV_full", "audio_only", "video_only"):
                s = inhibition[name][cond]
                f.write(f"| {name} | {cond} | "
                        f"{s['mean_abs_g']:.4f} | "
                        f"{s['mean_abs_alpha_g']:.4f} | "
                        f"{s['frac_g_gt_half']:.3f} | "
                        f"{s['mean_abs_res']:.4f} | "
                        f"{s['frac_res_pos']:.3f} | "
                        f"{s['alpha']:.4f} |\n")
        f.write("\n*residual = a_fused − a_mid; for mid_mult that is "
                 "`a_mid · α·g`, for mid_add it is `α·g` directly.*\n")

        f.write("\n## 5. Viseme decodability (5-fold LR on penult)\n\n")
        f.write("| variant | flavour | penult dim | acc | bal acc |\n")
        f.write("|---|---|---|---|---|\n")
        for name, (_m, _ck) in variants.items():
            v = viseme_results[name]
            d = (128 if VARIANT_FLAVOUR[name] in GATE_BEARING else 192)
            f.write(f"| {name} | {VARIANT_FLAVOUR[name]} | {d} | "
                    f"{v['acc_mean']*100:.2f}% ± "
                    f"{v['acc_std']*100:.2f} | "
                    f"{v['bal_acc_mean']*100:.2f}% ± "
                    f"{v['bal_acc_std']*100:.2f} |\n")

        f.write("\n## Headline summary (1-line each)\n\n")
        ### Build a 1-line summary per probe
        # Inverse effectiveness — measure whether AV gain grows with σ_a
        for_av_name = []
        for name in variants:
            d = noise_results[name]["sigma_a"]
            g0 = d[0.0]["gain_pp"]
            g_high = d[0.5]["gain_pp"]
            grew = (g_high > g0 + 1.0)
            for_av_name.append(
                f"{name} σ_a=0 gain={g0:+.2f}pp → σ_a=0.5 gain={g_high:+.2f}pp "
                f"({'GROWS' if grew else 'flat/shrinks'})")
        f.write("- **Inverse effectiveness:** " + " | ".join(for_av_name) + "\n")
        # Channel concentration
        concs = []
        for name in variants:
            imp = channel_impacts[name]
            if imp is None:
                concs.append(f"{name}: N/A")
                continue
            top4 = np.sort(imp)[-4:].sum()
            pos = imp[imp > 0].sum() + 1e-12
            concs.append(f"{name}: top-4={top4/pos*100:.1f}% of pos")
        f.write("- **Channel concentration:** " + " | ".join(concs) + "\n")
        # Inhibition
        inhs = []
        for name in ("AV_fused", "D3_10_add"):
            g_av = inhibition[name]["AV_full"]["mean_abs_alpha_g"]
            g_ao = inhibition[name]["audio_only"]["mean_abs_alpha_g"]
            sign = "quieter under AV" if g_av < g_ao else "louder under AV"
            inhs.append(f"{name}: |αg|_AV={g_av:.3f}, |αg|_A-only={g_ao:.3f} "
                        f"({sign})")
        f.write("- **Gate magnitude AV vs A-only:** " + " | ".join(inhs) + "\n")
        # Viseme
        vis = []
        for name in variants:
            v = viseme_results[name]
            vis.append(f"{name} penult: {v['acc_mean']*100:.2f}%")
        f.write("- **Viseme decodability:** " + " | ".join(vis) + "\n")
    print(f"wrote markdown: {MD_PATH}")


# Main

def main() -> None:
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_idx, val_sha = _val_idx_with_check()
    print(f"  val_idx sha256: {val_sha[:32]}… ([OK] matches expected "
          f"{VAL_HASH_PREFIX}…)")
    print(f"  N val: {len(val_idx)}")

    print("\nLoading variants...")
    variants = {}
    for name in CKPT_PATHS:
        t0 = time.time()
        m, ck = _load_variant(name, device)
        variants[name] = (m, ck)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:>11s}: params={n:,}, "
              f"best_val_acc={ck.get('best_val_acc', float('nan'))*100:.4f}% "
              f"({time.time()-t0:.1f}s)")

    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
    idx_to_label = base.idx_to_label

    # One-time penultimate cache (also stashes a_mid/v_mid for gate-bearing,
    # v_gap for late). Reused by probes 2/3/5.
    print("\nBuilding per-variant penult caches...")
    penult_caches = {}
    for name in CKPT_PATHS:
        t0 = time.time()
        penult_caches[name] = _extract_penult(name, variants[name][0],
                                                  loader, device)
        cache = penult_caches[name]
        print(f"  {name:>11s}  penult.shape={cache['penult'].shape}  "
              f"a_mid in cache={('a_mid' in cache)}  "
              f"v_gap in cache={('v_gap' in cache)}  "
              f"({time.time()-t0:.1f}s)")

    t0_all = time.time()

    # 1. Noise sweep
    noise_results = probe_1_noise_sweep(variants, val_idx, base, device)

    # 2. Channel lesion
    channel_impacts = probe_2_channel_lesion(variants, val_idx, base, device,
                                                penult_caches)

    # 3. α-sweep for additive
    alpha_summary = probe_3_alpha_sweep_additive(variants, penult_caches,
                                                    device)

    # 4. Inhibition probe
    inhibition = probe_4_inhibition(variants, val_idx, base, device)

    # 5. Viseme decodability
    viseme_results = probe_5_viseme(penult_caches, idx_to_label)

    # Markdown writeup
    _write_markdown(variants, val_sha, noise_results, channel_impacts,
                     alpha_summary, inhibition, viseme_results)

    print(f"\nDone in {(time.time()-t0_all)/60:.1f} min. "
          f"Artifacts in {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        if (fn.startswith("D1_cross_variant_")
                or fn.startswith("D2_viseme_probe_cross_variant")
                or fn.startswith("D3_cross_variant_")
                or fn.startswith("D3_alpha_sweep_AV_additive")
                or fn.startswith("D3_inhibition_cross_variant")):
            print(f"  {fn}")
    print(f"  ../{os.path.basename(MD_PATH)}")


if __name__ == "__main__":
    main()
