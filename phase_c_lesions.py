#!/usr/bin/env python3
"""Phase C — gate lesions + α-sweep (D3.5 Wv channel, D3.7 α-sweep,
D3.8 gate-off, D3.9 Wa channel). Writes CSVs to `analysis/deepdive/`.
Run: `python phase_c_lesions.py`."""

from __future__ import annotations

import contextlib
import csv
import os
import time
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_av_msi import (
    BATCH_SIZE, T_STRIDE, _ValAVView, _accuracy, _load_models,
)
from dataset_raw_noisy import RawNoisyAVDataset
from model_av import AVWordResNet


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis", "deepdive")
AV_RAW_CKPT = os.path.join(SCRIPT_DIR, "models", "av_fused_rawnoise.pt")
os.makedirs(OUT_DIR, exist_ok=True)


# Activation cache: a_mid, v_mid one-time forward through audio_block1 +
# visual encoder. Lesions then operate on cached tensors and only re-run
# gate + block2 + FC (~10× speedup).

@torch.no_grad()
def _cache_a_v_mid(model: AVWordResNet, loader, device):
    """Forward all val samples through audio_block1 and visual encoder once."""
    a_mids, v_mids, ys = [], [], []
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        a_mid = model.audio_block1(mel)
        v_mid = model.visual(vid)
        a_mids.append(a_mid.cpu())
        v_mids.append(v_mid.cpu())
        ys.append(y)
    a_mid = torch.cat(a_mids, dim=0)
    v_mid = torch.cat(v_mids, dim=0)
    labels = torch.cat(ys, dim=0)
    return a_mid, v_mid, labels


@torch.no_grad()
def _forward_from_cache(model: AVWordResNet, a_mid_cache, v_mid_cache,
                          labels_cache, device, batch_size: int = 256,
                          wv_mask: torch.Tensor | None = None,
                          wa_mask: torch.Tensor | None = None,
                          alpha_override: float | None = None,
                          block2_mask: torch.Tensor | None = None) -> dict:
    """Run cached a_mid, v_mid through gate + block2 + FC with optional lesions."""
    preds, labels = [], []
    n = a_mid_cache.shape[0]
    for i in range(0, n, batch_size):
        a_mid = a_mid_cache[i:i+batch_size].to(device, non_blocking=True)
        v_mid = v_mid_cache[i:i+batch_size].to(device, non_blocking=True)

        Wa_out = model.gate.Wa(a_mid)
        Wv_out = model.gate.Wv(v_mid)
        if wv_mask is not None:
            Wv_out = Wv_out * wv_mask.view(1, -1, 1, 1)
        if wa_mask is not None:
            Wa_out = Wa_out * wa_mask.view(1, -1, 1, 1)
        g = torch.sigmoid(Wa_out + Wv_out)
        alpha = (model.gate.alpha if alpha_override is None
                  else torch.tensor(float(alpha_override),
                                     device=a_mid.device))
        a_fused = a_mid * (1.0 + alpha * g)

        x = model.audio_block2(a_fused)
        if block2_mask is not None:
            x = x * block2_mask.view(1, -1, 1, 1)
        pen = model.gap(x).flatten(1)
        logits = model.fc(model.dropout(pen))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(labels_cache[i:i+batch_size].numpy())
    return dict(
        preds=np.concatenate(preds),
        labels=np.concatenate(labels),
    )


# Wrapper to keep call signatures compatible with the rest of the script.
class _CachedLoader:
    """Sentinel — passes the (a_mid, v_mid, labels) cache through the
    sub-experiment helpers via shared module-level state."""

    def __init__(self, a_mid, v_mid, labels):
        self.a_mid = a_mid
        self.v_mid = v_mid
        self.labels = labels


_CACHE: _CachedLoader | None = None


@torch.no_grad()
def _forward_AV_masked(model: AVWordResNet, loader, device,
                       wv_mask: torch.Tensor | None = None,
                       wa_mask: torch.Tensor | None = None,
                       alpha_override: float | None = None,
                       block2_mask: torch.Tensor | None = None) -> dict:
    """Lesion-aware forward.

    If `_CACHE` is set, runs the fast cached pipeline (~5s/lesion);
    otherwise re-iterates the loader (~25s/lesion).
    """
    if _CACHE is not None:
        return _forward_from_cache(
            model, _CACHE.a_mid, _CACHE.v_mid, _CACHE.labels,
            device, wv_mask=wv_mask, wa_mask=wa_mask,
            alpha_override=alpha_override, block2_mask=block2_mask)

    preds, labels = [], []
    for mel, vid, y in loader:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        a_mid = model.audio_block1(mel)
        v_mid = model.visual(vid)

        Wa_out = model.gate.Wa(a_mid)
        Wv_out = model.gate.Wv(v_mid)
        if wv_mask is not None:
            Wv_out = Wv_out * wv_mask.view(1, -1, 1, 1)
        if wa_mask is not None:
            Wa_out = Wa_out * wa_mask.view(1, -1, 1, 1)
        g = torch.sigmoid(Wa_out + Wv_out)
        alpha = (model.gate.alpha if alpha_override is None
                  else torch.tensor(float(alpha_override),
                                     device=a_mid.device))
        a_fused = a_mid * (1.0 + alpha * g)

        x = model.audio_block2(a_fused)
        if block2_mask is not None:
            x = x * block2_mask.view(1, -1, 1, 1)
        pen = model.gap(x).flatten(1)
        logits = model.fc(model.dropout(pen))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return dict(
        preds=np.concatenate(preds),
        labels=np.concatenate(labels),
    )


# D3.5 — Per-channel Wv lesion

def D3_5_Wv_channel_lesion(model_name: str, model, loader, device,
                             baseline_acc: float) -> np.ndarray:
    print(f"\n  D3.5 — Per-channel W_v lesion ({model_name}):")
    n_channels = model.gate.Wv.weight.shape[0]  # 64
    impacts = np.zeros(n_channels, dtype=np.float64)
    accs_lesion = np.zeros(n_channels, dtype=np.float64)
    out_csv = os.path.join(
        OUT_DIR, f"D3_Wv_channel_lesion_{model_name}.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["channel", "AV_acc_baseline", "AV_acc_lesioned",
                    "delta_pp"])
        for c in range(n_channels):
            mask = torch.ones(n_channels, device=device)
            mask[c] = 0.0
            t0 = time.time()
            out = _forward_AV_masked(model, loader, device, wv_mask=mask)
            acc = _accuracy(out["preds"], out["labels"])
            impact_pp = (baseline_acc - acc) * 100.0
            impacts[c] = impact_pp
            accs_lesion[c] = acc
            w.writerow([c, f"{baseline_acc:.6f}", f"{acc:.6f}",
                        f"{impact_pp:+.4f}"])
            if c % 8 == 0:
                print(f"    [{c:>2d}/{n_channels}] acc={acc:.4%} "
                      f"Δ={impact_pp:+.2f} pp ({time.time()-t0:.1f}s)")
    print(f"  wrote {out_csv}")
    top = np.argsort(-impacts)[:8]
    print(f"  top-8 Wv channels by impact (pp):")
    for c in top:
        print(f"    ch{c:>2d}: Δ={impacts[c]:+.3f} pp "
              f"(acc {accs_lesion[c]:.4%})")
    return impacts


# D3.6 — Tertile lesion

def D3_6_tertile_lesion(model_name: str, model, loader, device,
                         baseline_acc: float, impacts: np.ndarray) -> None:
    print(f"\n  D3.6 — Tertile lesion ({model_name}):")
    n_channels = len(impacts)
    order = np.argsort(-impacts)        # high → low
    t1 = order[: n_channels // 3]
    t2 = order[n_channels // 3 : 2 * n_channels // 3]
    t3 = order[2 * n_channels // 3 :]
    out_csv = os.path.join(
        OUT_DIR, f"D3_Wv_tertile_lesion_{model_name}.csv")
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["tertile", "n_channels", "channel_ids",
                    "AV_acc_baseline", "AV_acc_lesioned", "delta_pp"])
        for name, idxs in (("high_impact", t1),
                            ("mid_impact",  t2),
                            ("low_impact",  t3)):
            mask = torch.ones(n_channels, device=device)
            mask[idxs] = 0.0
            out = _forward_AV_masked(model, loader, device, wv_mask=mask)
            acc = _accuracy(out["preds"], out["labels"])
            delta = (baseline_acc - acc) * 100.0
            print(f"    {name:>12s}: |c|={len(idxs)} "
                  f"acc={acc:.4%} Δ={delta:+.2f} pp")
            w.writerow([name, len(idxs),
                        ",".join(map(str, sorted(idxs.tolist()))),
                        f"{baseline_acc:.6f}",
                        f"{acc:.6f}", f"{delta:+.4f}"])
        # All zero (full Wv lesion) — should equal α effective baseline
        mask = torch.zeros(n_channels, device=device)
        out = _forward_AV_masked(model, loader, device, wv_mask=mask)
        acc_all = _accuracy(out["preds"], out["labels"])
        delta = (baseline_acc - acc_all) * 100.0
        print(f"    {'all_Wv_zero':>12s}: |c|={n_channels} "
              f"acc={acc_all:.4%} Δ={delta:+.2f} pp")
        w.writerow(["all_Wv_zero", n_channels, "all",
                    f"{baseline_acc:.6f}",
                    f"{acc_all:.6f}", f"{delta:+.4f}"])
    print(f"  wrote {out_csv}")


# D3.7 / D3.8 — α-sweep including α=0

ALPHA_LEVELS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0)


def D3_7_alpha_sweep(model_name: str, model, loader, device,
                      baseline_acc: float) -> None:
    natural_alpha = float(model.gate.alpha.detach().item())
    print(f"\n  D3.7 — α inference sweep ({model_name}, "
          f"trained α={natural_alpha:.4f}):")
    levels = list(ALPHA_LEVELS) + [natural_alpha]
    levels = sorted(set(levels))
    out_csv = os.path.join(OUT_DIR, f"D3_alpha_sweep_{model_name}.csv")
    bit_match = None
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "AV_acc", "delta_from_baseline_pp",
                    "is_trained_alpha"])
        for a in levels:
            t0 = time.time()
            out = _forward_AV_masked(model, loader, device,
                                       alpha_override=a)
            acc = _accuracy(out["preds"], out["labels"])
            delta = (acc - baseline_acc) * 100.0
            is_trained = abs(a - natural_alpha) < 1e-6
            tag = " (trained)" if is_trained else ""
            print(f"    α={a:6.3f}: acc={acc:.4%} "
                  f"Δ={delta:+.2f} pp{tag} ({time.time()-t0:.1f}s)")
            w.writerow([f"{a:.6f}", f"{acc:.6f}",
                        f"{delta:+.4f}",
                        "1" if is_trained else "0"])
            if is_trained:
                bit_match = abs(acc - baseline_acc) < 1e-6
    print(f"  wrote {out_csv}")
    if bit_match is None:
        print("  [WARN] trained α not visited (shouldn't happen)")
    elif bit_match:
        print("  [OK] trained-α override matches natural AV_full bit-exact")
    else:
        print("  [WARN] trained-α override does NOT bit-match natural — investigate")


# D3.9 — Per-channel Wa lesion (audio side; sanity for D3.5)

def D3_9_Wa_channel_lesion(model_name: str, model, loader, device,
                             baseline_acc: float) -> None:
    print(f"\n  D3.9 — Per-channel W_a lesion ({model_name}):")
    n_channels = model.gate.Wa.weight.shape[0]
    out_csv = os.path.join(
        OUT_DIR, f"D3_Wa_channel_lesion_{model_name}.csv")
    impacts = np.zeros(n_channels)
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["channel", "AV_acc_baseline", "AV_acc_lesioned",
                    "delta_pp"])
        for c in range(n_channels):
            mask = torch.ones(n_channels, device=device)
            mask[c] = 0.0
            out = _forward_AV_masked(model, loader, device, wa_mask=mask)
            acc = _accuracy(out["preds"], out["labels"])
            delta = (baseline_acc - acc) * 100.0
            impacts[c] = delta
            w.writerow([c, f"{baseline_acc:.6f}", f"{acc:.6f}",
                        f"{delta:+.4f}"])
    print(f"  wrote {out_csv}")
    top = np.argsort(-impacts)[:5]
    print(f"  top-5 Wa channels by impact: " +
          ", ".join(f"ch{c}={impacts[c]:+.3f}pp" for c in top))


# Main

def _maybe_load_av_rawnoise(device):
    if not os.path.exists(AV_RAW_CKPT):
        return None
    ckpt = torch.load(AV_RAW_CKPT, weights_only=False)
    m = AVWordResNet(len(ckpt["label_to_idx"])).to(device).eval()
    m.load_state_dict(ckpt["model_state_dict"])
    return m


def _baseline_acc(model, loader, device) -> float:
    out = _forward_AV_masked(model, loader, device)
    return _accuracy(out["preds"], out["labels"])


def _run_for_model(name: str, model, loader, device):
    global _CACHE
    print(f"\n──── {name} ────")
    print(f"Building activation cache (one-pass forward)...")
    t0 = time.time()
    a_mid, v_mid, labels = _cache_a_v_mid(model, loader, device)
    print(f"  cache shapes: a_mid={tuple(a_mid.shape)}, "
          f"v_mid={tuple(v_mid.shape)} ({time.time()-t0:.1f}s)")
    _CACHE = _CachedLoader(a_mid, v_mid, labels)

    # Baseline (lesion-free) via cache — should match natural inference.
    out = _forward_AV_masked(model, loader, device)
    baseline = _accuracy(out["preds"], out["labels"])
    print(f"  cached baseline = {baseline:.4%}")

    impacts = D3_5_Wv_channel_lesion(
        name, model, loader, device, baseline)
    D3_6_tertile_lesion(
        name, model, loader, device, baseline, impacts)
    D3_7_alpha_sweep(
        name, model, loader, device, baseline)
    D3_9_Wa_channel_lesion(
        name, model, loader, device, baseline)
    _CACHE = None
    return baseline


def main() -> None:
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = torch.load(os.path.join(SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)
    val_idx = splits["val_idx"]
    if hasattr(val_idx, "numpy"):
        val_idx = val_idx.numpy()

    models = _load_models(device)
    AV_clean = models["AV"][0]

    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    view = _ValAVView(base, val_idx)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

    baseline_clean = _run_for_model("AV_clean", AV_clean, loader, device)
    assert 0.9560 <= baseline_clean <= 0.9590, \
        f"baseline OOD: {baseline_clean:.4%}"
    print("  [OK] sanity OK (AV_clean baseline)")

    AV_raw = _maybe_load_av_rawnoise(device)
    if AV_raw is not None:
        _ = _run_for_model("AV_rawnoise", AV_raw, loader, device)

    print("\nPhase C done. Artifacts:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("D3_") and f.endswith(".csv"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
