#!/usr/bin/env python3
"""VALIDATOR — independent re-derivation of the LOAD-BEARING cells of the Q14
recurrent-vs-feedforward comparison (analysis/deepdive/Q14_recurrent_vs_feedforward.csv),
plus the three scientific-integrity checks.

This is the INDEPENDENT leg (leg 2): own data pipeline (noise / temporal-shift /
frame-mask reimplemented, NOT imported from analyze_av_msi), own forwards. The
recurrent forward UNROLLS the GRU explicitly (visual -> mean-pool the 40 axis ->
permute -> vgru -> vproj -> broadcast -> gate), so it genuinely EXERCISES the GRU
rather than calling model.forward() (which the coder's harness uses). If my manual
unroll AGREES with the coder's native-forward CSV, that cross-validates both that
forward() does what the architecture claims AND that the GRU is in the path.

Reused (allowed): the trained arch classes (AVWordResNet, AVRecurrentWordResNet) +
signal-processing primitives (_read_wav/_wav_to_log_mel/_pad_audio) + the clean
RawNoisyAVDataset data source. NOT reused: any eval helper / View class / the q14
harness itself.

Cells re-derived (eager fp32, pinned val sha 03c5a87a, N=5244):
  clean_val            rec / ffm / ffc
  rawnoise sigma=0.05  rec / ffm / ffc
  E11 delta_ms=-100    rec / ffm / ffc   (delta_frames = round(-100/20) = -5)
  D5 win=[20,30)       rec / ffm / ffc   (zero video frames 20..29)
ffc clean is the self-check anchor (must reproduce 0.956712, my long-standing AV
unroll anchor) -> proves the AV pipeline is trustworthy before any rec/ffm number.

Integrity checks:
  (a) GRU engaged: reconstruct W_hh init (manual_seed(0), same construction order),
      report ||init||, ||trained||, drift=||trained-init||, rel=drift/||init||.
      Anchor: training log ep182 (best-val ckpt) w_hh_rel_drift=2.153921 (~2.15).
  (b) param-match: n_params(rec) vs n_params(ffm), ratio within +-10%.
  (c) recipe wiring: rec seed/arch, ffm seed/arch from ckpt metadata; CSV arithmetic
      (rec-ffm, ffm-ffc) checked against the reproduced cells in the report stage.

Run on dev-codex:
    ./venv/bin/python validator_indep_q14.py --root /scratch/daedelus \
        --out /scratch/daedelus/analysis/validator_indep_q14.csv
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
GUARD = 5e-3            # 0.5 pp guardrail
DRIFT_LOG = 2.153921   # ep182 best-val ckpt w_hh_rel_drift (training log)

# Claim-CSV cells (analysis/deepdive/Q14_recurrent_vs_feedforward.csv) we re-derive.
CLAIM = {
    ("clean_val", "clean", "rec"): 0.948513,
    ("clean_val", "clean", "ffm"): 0.958238,
    ("clean_val", "clean", "ffc"): 0.956712,
    ("rawnoise_sweep", "0.0500", "rec"): 0.717010,
    ("rawnoise_sweep", "0.0500", "ffm"): 0.439169,
    ("rawnoise_sweep", "0.0500", "ffc"): 0.609268,
    ("E11_temporal", "delta_ms=-100", "rec"): 0.719298,
    ("E11_temporal", "delta_ms=-100", "ffm"): 0.460526,
    ("E11_temporal", "delta_ms=-100", "ffc"): 0.475400,
    ("D5_temporal_saliency", "win=[20,30)", "rec"): 0.222349,
    ("D5_temporal_saliency", "win=[20,30)", "ffm"): 0.241037,
    ("D5_temporal_saliency", "win=[20,30)", "ffc"): 0.213768,
}
FFC_CLEAN_ANCHOR = 0.956712  # self-check


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


class CondView(Dataset):
    """Pinned val with one of: clean / rawnoise(sigma) / shift(delta) / mask(win).
    Reimplements the noise/shift/mask semantics; uses the clean RawNoisyAVDataset
    only as the raw data source + signal-processing primitives."""

    def __init__(self, base, val_idx, prims, mode="clean",
                 sigma=0.0, delta=0, win=None, seed=0):
        self.base = base
        self.val_idx = np.asarray(val_idx, dtype=np.int64)
        self._read_wav, self._wav_to_log_mel, self._pad_audio = prims
        self.mode = mode
        self.sigma = float(sigma)
        self.delta = int(delta)
        self.win = win
        self.seed = int(seed)

    def __len__(self):
        return len(self.val_idx)

    def _clean_video(self, idx):
        v = np.array(self.base._videos[idx])
        if self.base.t_stride > 1:
            v = v[:: self.base.t_stride]
        return torch.from_numpy(v).unsqueeze(0).float() / 255.0  # (1,T,H,W)

    def __getitem__(self, k):
        idx = int(self.val_idx[k])
        label = int(self.base.labels[idx])

        if self.mode == "rawnoise":
            audio = self._read_wav(self.base.audio_paths[idx])
            if self.sigma > 0:
                rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
                sig = self.sigma * rms
                rng = np.random.default_rng(self.seed + idx)
                audio = audio + rng.standard_normal(len(audio)).astype(np.float32) * sig
            audio_p = self._pad_audio(audio, int(self.base.pad_offsets[idx]))
            mel = torch.from_numpy(self._wav_to_log_mel(audio_p).astype(np.float32))
            vid = self._clean_video(idx)
            return mel, vid, label

        # clean / shift / mask all start from the clean (mel, vid)
        mel, vid, _ = self.base[idx]          # RawNoisyAVDataset(noise=False)
        if self.mode == "shift" and self.delta != 0:
            T = vid.shape[1]
            out = torch.zeros_like(vid)
            if self.delta > 0:
                d = min(self.delta, T)
                out[:, d:] = vid[:, : T - d]
            else:
                d = min(-self.delta, T)
                out[:, : T - d] = vid[:, d:]
            vid = out
        elif self.mode == "mask" and self.win is not None:
            vid = vid.clone()
            vid[:, self.win[0]:self.win[1], :, :] = 0.0
        return mel, vid, label


@torch.no_grad()
def av_forward(m, mel, vid):
    """Feedforward AV manual unroll (ffm, ffc). dropout is identity in eval."""
    a_mid = m.audio_block1(mel)
    v_mid = m.visual(vid)
    a_fused = m.gate(a_mid, v_mid)
    x = m.gap(m.audio_block2(a_fused)).flatten(1)
    return m.fc(m.dropout(x))


@torch.no_grad()
def rec_forward(m, mel, vid):
    """Recurrent AV manual unroll — EXPLICITLY runs the GRU over the 50-frame axis."""
    a_mid = m.audio_block1(mel)
    v_mid = m.visual(vid)                     # (B,64,40,50)
    B, C, Hh, T = v_mid.shape
    seq = v_mid.mean(dim=2).permute(0, 2, 1)  # (B,50,64)
    out, _ = m.vgru(seq)                      # (B,50,gru_hidden)
    out = m.vproj(out).permute(0, 2, 1)       # (B,64,50)
    v_ctx = out.unsqueeze(2).expand(B, C, Hh, T).contiguous()
    a_fused = m.gate(a_mid, v_ctx)
    x = m.gap(m.audio_block2(a_fused)).flatten(1)
    return m.fc(m.dropout(x))


def eval_acc(model, fwd, view, device, workers=4):
    dl = DataLoader(view, batch_size=64, shuffle=False,
                    num_workers=workers, pin_memory=True)
    preds, labs = [], []
    for mel, vid, y in dl:
        mel = mel.unsqueeze(1).to(device, non_blocking=True)   # (B,1,80,99)
        vid = vid.to(device, non_blocking=True)
        logits = fwd(model, mel, vid)
        preds.append(logits.argmax(1).cpu().numpy())
        labs.append(y.numpy())
    p = np.concatenate(preds)
    l = np.concatenate(labs)
    return float((p == l).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    sys.path.insert(0, args.root)
    from model_av import AVWordResNet
    from model_av_recurrent import AVRecurrentWordResNet
    from dataset_raw_noisy import RawNoisyAVDataset
    from paired_dataset import _read_wav, _wav_to_log_mel, _pad_audio
    prims = (_read_wav, _wav_to_log_mel, _pad_audio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = os.path.join(args.root, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    sha = _hash_idx(val_idx)
    print(f"[val] N={len(val_idx)} sha256={sha}", flush=True)
    assert sha == EXPECT_SHA, "VAL SHA MISMATCH"
    assert len(val_idx) == 5244

    base = RawNoisyAVDataset(noise=False, t_stride=2, return_video=True)
    mdir = os.path.join(args.root, "models")

    def load(cls, name, gru_hidden=None):
        ck = torch.load(os.path.join(mdir, name), weights_only=False, map_location="cpu")
        sd = ck["model_state_dict"]
        n = int(sd["fc.weight"].shape[0])
        if cls is AVRecurrentWordResNet:
            gh = int(sd["vproj.weight"].shape[1])
            m = cls(n, gru_hidden=gh)
        else:
            m = cls(n)
        m.load_state_dict(sd)
        return m.to(device).float().eval(), ck

    rec, ck_rec = load(AVRecurrentWordResNet, "av_fused_recurrent.pt")
    ffm, ck_ffm = load(AVWordResNet, "av_fused_ff_baseline_seed0.pt")
    ffc, ck_ffc = load(AVWordResNet, "av_fused.pt")
    n_classes = int(ck_rec["model_state_dict"]["fc.weight"].shape[0])
    print(f"[ckpt] rec arch={ck_rec.get('arch')} seed={ck_rec.get('seed')} "
          f"best_val={ck_rec.get('best_val_acc'):.6f} gru_hidden={ck_rec.get('gru_hidden')}",
          flush=True)
    print(f"[ckpt] ffm arch={ck_ffm.get('arch')} seed={ck_ffm.get('seed')} "
          f"best_val={ck_ffm.get('best_val_acc'):.6f}", flush=True)
    print(f"[ckpt] ffc arch={ck_ffc.get('arch','AVWordResNet')} seed={ck_ffc.get('seed','?(42)')} "
          f"best_val={ck_ffc.get('best_val_acc'):.6f}", flush=True)

    # ---- integrity (b): param-match ----
    n_rec = sum(p.numel() for p in rec.parameters())
    n_ffm = sum(p.numel() for p in ffm.parameters())
    n_gru = sum(p.numel() for p in rec.vgru.parameters()) + \
        sum(p.numel() for p in rec.vproj.parameters())
    ratio = n_rec / n_ffm
    param_ok = 0.90 <= ratio <= 1.10
    print(f"\n[integrity b: params] rec={n_rec:,} ffm={n_ffm:,} ratio={ratio:.4f} "
          f"(GRU+proj={n_gru:,}) {'OK' if param_ok else '**FAIL'}", flush=True)

    # ---- integrity (a): GRU drift from init ----
    torch.manual_seed(0)
    np.random.seed(0)
    init_model = AVRecurrentWordResNet(n_classes, gru_hidden=int(ck_rec.get("gru_hidden", 64)))
    w_init = init_model.vgru.weight_hh_l0.detach().clone()
    w_trained = rec.vgru.weight_hh_l0.detach().cpu()
    init_norm = float(torch.linalg.norm(w_init))
    trained_norm = float(torch.linalg.norm(w_trained))
    drift = float(torch.linalg.norm(w_trained - w_init))
    rel = drift / max(init_norm, 1e-12)
    drift_ok = abs(rel - DRIFT_LOG) < 0.05 and rel > 1.0  # ~2.15 and clearly engaged
    print(f"[integrity a: drift] ||init||={init_norm:.4f} ||trained||={trained_norm:.4f} "
          f"drift={drift:.4f} rel={rel:.4f} (log {DRIFT_LOG}) "
          f"{'OK' if drift_ok else '**FAIL'}", flush=True)

    # ---- forwards / cells ----
    specs = [
        ("clean_val", "clean", dict(mode="clean")),
        ("rawnoise_sweep", "0.0500", dict(mode="rawnoise", sigma=0.05, seed=0)),
        ("E11_temporal", "delta_ms=-100", dict(mode="shift", delta=round(-100 / 20.0))),
        ("D5_temporal_saliency", "win=[20,30)", dict(mode="mask", win=(20, 30))),
    ]
    models = [("rec", rec, rec_forward), ("ffm", ffm, av_forward), ("ffc", ffc, av_forward)]

    rows = []
    flags = []
    print("\n[cells — reproduced vs claim]", flush=True)
    for axis, cond, kw in specs:
        view = CondView(base, val_idx, prims, **kw)
        for tag, m, fwd in models:
            acc = eval_acc(m, fwd, view, device, workers=args.workers)
            claim = CLAIM[(axis, cond, tag)]
            d = acc - claim
            ok = abs(d) <= GUARD
            if not ok:
                flags.append((axis, cond, tag, acc, claim))
            print(f"  {axis:>22s} {cond:>14s} {tag}: {acc:.6f} (claim {claim:.6f}) "
                  f"d={d*100:+.4f}pp {'OK' if ok else '**FLAG'}", flush=True)
            rows.append([axis, cond, tag, f"{acc:.6f}", f"{claim:.6f}", f"{d*100:+.4f}",
                         "OK" if ok else "FLAG"])

    # self-check: ffc clean
    ffc_clean = next(float(r[3]) for r in rows
                     if r[0] == "clean_val" and r[2] == "ffc")
    sc_ok = abs(ffc_clean - FFC_CLEAN_ANCHOR) < GUARD

    out = args.out or os.path.join(args.root, "analysis", "validator_indep_q14.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["axis", "condition", "model", "reproduced", "claim", "delta_pp", "FLAG"])
        for r in rows:
            w.writerow(r)
        w.writerow([])
        w.writerow(["integrity", "param_ratio_rec_over_ffm", "", f"{ratio:.4f}",
                    "0.90-1.10", "", "OK" if param_ok else "FAIL"])
        w.writerow(["integrity", "gru_whh_rel_drift", "", f"{rel:.4f}",
                    f"{DRIFT_LOG}", "", "OK" if drift_ok else "FAIL"])
        w.writerow(["integrity", "gru_whh_init_norm", "", f"{init_norm:.4f}", "", "", ""])
        w.writerow(["integrity", "gru_whh_trained_norm", "", f"{trained_norm:.4f}", "", "", ""])
        w.writerow(["integrity", "ffc_clean_selfcheck", "", f"{ffc_clean:.6f}",
                    f"{FFC_CLEAN_ANCHOR}", "", "OK" if sc_ok else "FAIL"])
    print(f"\n[out] wrote {out}", flush=True)

    all_ok = sc_ok and param_ok and drift_ok and not flags
    print("\n[VERDICT]", flush=True)
    print(f"  ffc clean self-check ........ {'OK' if sc_ok else 'FAIL'} "
          f"({ffc_clean:.6f} vs {FFC_CLEAN_ANCHOR})", flush=True)
    print(f"  all 12 cells within 0.5pp ... {'OK' if not flags else f'FLAG {flags}'}", flush=True)
    print(f"  param ratio in [0.90,1.10] .. {'OK' if param_ok else 'FAIL'} ({ratio:.4f})", flush=True)
    print(f"  GRU rel-drift ~2.15 ......... {'OK' if drift_ok else 'FAIL'} ({rel:.4f})", flush=True)
    print(f"[{'GO' if all_ok else 'NO-GO/FLAG'}] Q14 independent re-derivation.", flush=True)


if __name__ == "__main__":
    main()
