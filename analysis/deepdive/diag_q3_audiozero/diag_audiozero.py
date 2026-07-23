#!/usr/bin/env python3
"""DEBUGGER diagnostic — Q3 audio-zero 2-sample cache-vs-fresh discrepancy.

Reproduces the validator's fresh-forward audio-zero (mel:=0) AV path, captures
per-stage GAP + logits, and:
  (1) localizes the FIRST stage where fresh diverges from the cached activations,
  (2) runs the decisive single-variable test: toggle ONLY cudnn.deterministic
      (cache build used True; validator used default False) on ONE fixed GPU,
  (3) tests run-to-run determinism within a fixed mode,
  (4) characterizes the flipped samples (top1-top2 margin vs logit diff).

No production code touched. Instrumentation/diagnosis only.
"""
import argparse, hashlib, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = "/home/vishnu/coding_proj/daedalus/daedalus"
EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"


def _hash_idx(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def build():
    sys.path.insert(0, ROOT)
    from train import WordResNet
    from model_v_only_fair import VOnlyFairWordResNet
    from model_av import AVWordResNet
    proc = os.path.join(ROOT, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    assert _hash_idx(val_idx) == EXPECT_SHA, "val sha mismatch"
    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels = dav["spectrograms"]
    mels_np = mels.numpy() if hasattr(mels, "numpy") else np.asarray(mels)
    labels_all = np.asarray(dav["labels"]).astype(np.int64)
    n_all = len(labels_all)
    T, H, W = dav["video_shape"]
    cache_path = dav.get("video_cache_path")
    videos = np.memmap(cache_path, dtype=np.uint8, mode="r", shape=(n_all, T, H, W))
    return WordResNet, VOnlyFairWordResNet, AVWordResNet, val_idx, mels_np, labels_all, videos


def load_av(AVWordResNet, device):
    ck = torch.load(os.path.join(ROOT, "models", "av_fused.pt"),
                    weights_only=False, map_location="cpu")
    m = AVWordResNet(len(ck["label_to_idx"]))
    m.load_state_dict(ck["model_state_dict"])
    return m.to(device).eval()


def make_loader(val_idx, mels_np, labels_all, videos, stride=2, batch=64, workers=4):
    class Vw(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k])
            mel = torch.from_numpy(mels_np[g]).unsqueeze(0)
            v = np.array(videos[g])
            if stride > 1: v = v[::stride]
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, vid, int(labels_all[g])
    return DataLoader(Vw(), batch_size=batch, shuffle=False,
                      num_workers=workers, pin_memory=True)


@torch.no_grad()
def forward_audiozero(AV, loader, device):
    """Replicate _extract_AV / validator audio-zero: mel:=0, real video."""
    amg, vmg, gog, b2g, LG, PR = [], [], [], [], [], []
    for mel, vid, y in loader:
        mel = mel.to(device, non_blocking=True)
        vid = vid.to(device, non_blocking=True)
        a_mid = AV.audio_block1(torch.zeros_like(mel))
        v_mid = AV.visual(vid)
        a_fused = AV.gate(a_mid, v_mid)
        b2 = AV.audio_block2(a_fused)
        pen = AV.gap(b2).flatten(1)
        lg = AV.fc(AV.dropout(pen))
        amg.append(a_mid.mean(dim=(2, 3)).cpu().numpy())
        vmg.append(v_mid.mean(dim=(2, 3)).cpu().numpy())
        gog.append(a_fused.mean(dim=(2, 3)).cpu().numpy())
        b2g.append(b2.mean(dim=(2, 3)).cpu().numpy())
        LG.append(lg.float().cpu().numpy())
        PR.append(lg.argmax(1).cpu().numpy())   # torch-GPU argmax
    return (np.concatenate(amg), np.concatenate(vmg), np.concatenate(gog),
            np.concatenate(b2g), np.concatenate(LG), np.concatenate(PR))


def set_cudnn(det):
    torch.backends.cudnn.deterministic = bool(det)
    torch.backends.cudnn.benchmark = False


def run_mode(AV, val_idx, mels_np, labels_all, videos, device, det, tag):
    set_cudnn(det)
    loader = make_loader(val_idx, mels_np, labels_all, videos)
    amg, vmg, gog, b2g, LG, PR = forward_audiozero(AV, loader, device)
    print(f"  [{tag}] cudnn.deterministic={det}  acc={ (PR==labels_all_global).mean():.6f}"
          f"  n_correct={int((PR==labels_all_global).sum())}")
    return dict(amg=amg, vmg=vmg, gog=gog, b2g=b2g, LG=LG, PR=PR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"[env] torch {torch.__version__}  device cuda:{args.gpu} "
          f"= {torch.cuda.get_device_name(args.gpu)}")

    WR, VR, AVR, val_idx, mels_np, labels_all, videos = build()
    global labels_all_global
    # labels in val order
    labels_val = labels_all[val_idx]
    labels_all_global = labels_val
    AV = load_av(AVR, device)

    # cache reference
    c = torch.load(os.path.join(ROOT, "processed", "deepdive_act_cache.pt"),
                   weights_only=False)
    az = c["AV_clean_audio_zero"]
    cache = {k: np.asarray(az[k]) for k in ["a_mid_gap", "v_mid_gap",
             "gate_out_gap", "block2_gap", "logits"]}
    cache_pred = cache["logits"].argmax(1)
    y = labels_val
    assert (np.asarray(c["labels"]).astype(np.int64) == y).all(), "label order mismatch"
    print(f"[cache] audio-zero acc={ (cache_pred==y).mean():.6f} "
          f"n_correct={int((cache_pred==y).sum())}  (q4=0.444699)")

    print("\n=== (A) two fresh runs, SAME mode (validator: deterministic=False) — run-to-run determinism ===")
    rF1 = run_mode(AV, val_idx, mels_np, labels_all, videos, device, False, "freshF#1")
    rF2 = run_mode(AV, val_idx, mels_np, labels_all, videos, device, False, "freshF#2")
    print(f"  run-to-run preds differ at {int((rF1['PR']!=rF2['PR']).sum())} samples; "
          f"logits max|d|={np.abs(rF1['LG']-rF2['LG']).max():.3e}")

    print("\n=== (B) fresh run, cache mode (deterministic=True) ===")
    rT1 = run_mode(AV, val_idx, mels_np, labels_all, videos, device, True, "freshT#1")
    rT2 = run_mode(AV, val_idx, mels_np, labels_all, videos, device, True, "freshT#2")
    print(f"  run-to-run (T) preds differ at {int((rT1['PR']!=rT2['PR']).sum())} samples; "
          f"logits max|d|={np.abs(rT1['LG']-rT2['LG']).max():.3e}")

    print("\n=== (C) single-variable: deterministic False vs True (same GPU, same inputs) ===")
    print(f"  preds(F) != preds(T) at {int((rF1['PR']!=rT1['PR']).sum())} samples; "
          f"logits max|d|={np.abs(rF1['LG']-rT1['LG']).max():.3e}")
    print(f"  acc(F)={ (rF1['PR']==y).mean():.6f}  acc(T)={ (rT1['PR']==y).mean():.6f}")

    print("\n=== (D) per-STAGE divergence: cache vs fresh (localize origin) ===")
    for mode, r in [("F", rF1), ("T", rT1)]:
        print(f"  --- fresh mode {mode} vs cache (max|d| per stage) ---")
        for st, fk in [("a_mid_gap", "amg"), ("v_mid_gap", "vmg"),
                       ("gate_out_gap", "gog"), ("block2_gap", "b2g"),
                       ("logits", "LG")]:
            d = np.abs(cache[st] - r[fk]).max()
            print(f"    {st:14s} max|cache-fresh|={d:.3e}")

    print("\n=== (E) flipped samples vs cache + near-tie margin analysis ===")
    for mode, r in [("F", rF1), ("T", rT1)]:
        flip = np.where(r["PR"] != cache_pred)[0]
        print(f"  fresh {mode}: {len(flip)} samples disagree with cache; "
              f"acc(fresh)={ (r['PR']==y).mean():.6f} cache={ (cache_pred==y).mean():.6f}")
        for i in flip[:12]:
            cl, fl = cache["logits"][i], r["LG"][i]
            cs = np.sort(cl)[::-1]; fs = np.sort(fl)[::-1]
            print(f"    idx {i:5d} y={y[i]:3d} | cache top1={cl.argmax():3d}(m={cs[0]-cs[1]:.3e}) "
                  f"fresh top1={fl.argmax():3d}(m={fs[0]-fs[1]:.3e}) "
                  f"| max|d_logit|={np.abs(cl-fl).max():.3e} "
                  f"cache_correct={cl.argmax()==y[i]} fresh_correct={fl.argmax()==y[i]}")

    # global margin-vs-noise: is every disagreement explained by margin<noise?
    print("\n=== (F) margin < cache-fresh logit noise predicts the flips? ===")
    for mode, r in [("F", rF1), ("T", rT1)]:
        cl = cache["logits"]; fl = r["LG"]
        cs = np.sort(cl, axis=1)
        margin = cs[:, -1] - cs[:, -2]
        noise = np.abs(cl - fl).max(axis=1)
        flip = (r["PR"] != cache_pred)
        below = margin < noise
        print(f"  mode {mode}: #flips={int(flip.sum())}  #(margin<noise)={int(below.sum())}  "
              f"flips ⊆ (margin<noise)? {bool(np.all(below[flip]))}  "
              f"median noise={np.median(noise):.3e}  max noise={noise.max():.3e}")

    np.savez("/tmp/diag_audiozero_out.npz",
             cache_logits=cache["logits"], cache_pred=cache_pred,
             freshF_logits=rF1["LG"], freshF_pred=rF1["PR"],
             freshT_logits=rT1["LG"], freshT_pred=rT1["PR"], y=y)
    print("\nsaved /tmp/diag_audiozero_out.npz")


if __name__ == "__main__":
    labels_all_global = None
    main()
