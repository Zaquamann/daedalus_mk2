#!/usr/bin/env python3
"""Causal proof for the Q3 audio-zero discrepancy.

(1) ENVIRONMENT AXIS: run the byte-identical audio-zero construction on the OTHER
    local GPU (cuda:1, different arch) -> different tiny fp noise -> boundary flips,
    while construction is unchanged. Isolates 'compute environment' as the causal axis.
(2) MECHANISM: inject a controlled perturbation into the cache logits.
    delta = fresh - cache (the real cross-env difference). Scale alpha 0->1 and show
    flips switch on exactly at margin<alpha*|delta|. Then random Gaussian noise of the
    SAME magnitude flips a comparable small set, ALL low-margin -> the perturbation is
    generic fp noise, nothing semantic.
"""
import os, sys, hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = "/home/vishnu/coding_proj/daedalus/daedalus"
sys.path.insert(0, ROOT)
EXPECT_SHA = "03c5a87acdcf07add81937906636be99cbbb04779c9fd497a2dce5a6c4565533"


def make_loader(val_idx, mels_np, labels_all, videos, stride=2, batch=64):
    class Vw(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k])
            mel = torch.from_numpy(mels_np[g]).unsqueeze(0)
            v = np.array(videos[g])
            if stride > 1: v = v[::stride]
            vid = torch.from_numpy(v).unsqueeze(0).float() / 255.0
            return mel, vid, int(labels_all[g])
    return DataLoader(Vw(), batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def fwd_az(AV, loader, device):
    LG, A1 = [], []
    for mel, vid, y in loader:
        mel = mel.to(device); vid = vid.to(device)
        a_mid = AV.audio_block1(torch.zeros_like(mel))
        v_mid = AV.visual(vid)
        b2 = AV.audio_block2(AV.gate(a_mid, v_mid))
        lg = AV.fc(AV.dropout(AV.gap(b2).flatten(1)))
        LG.append(lg.float().cpu().numpy())
        A1.append(a_mid.mean(dim=(2, 3)).cpu().numpy())
    return np.concatenate(LG), np.concatenate(A1)


def main():
    from model_av import AVWordResNet
    proc = os.path.join(ROOT, "processed")
    s = torch.load(os.path.join(proc, "splits.pt"), weights_only=False)
    val_idx = np.asarray(s["val_idx"], dtype=np.int64)
    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels_np = np.asarray(dav["spectrograms"])
    labels_all = np.asarray(dav["labels"]).astype(np.int64)
    T, H, W = dav["video_shape"]
    videos = np.memmap(dav["video_cache_path"], dtype=np.uint8, mode="r",
                       shape=(len(labels_all), T, H, W))
    y = labels_all[val_idx]

    d = np.load("/tmp/diag_audiozero_out.npz")
    cache_logits = d["cache_logits"]; cache_pred = d["cache_pred"]
    fresh0_logits = d["freshF_logits"]
    n0 = int((cache_pred == y).sum())
    print(f"[ref] cache n_correct={n0}/5244 acc={n0/5244:.6f}")
    print(f"[ref] fresh cuda:0 n_correct={int((fresh0_logits.argmax(1)==y).sum())}/5244")

    def load_av(dev):
        ck = torch.load(os.path.join(ROOT, "models", "av_fused.pt"),
                        weights_only=False, map_location="cpu")
        m = AVWordResNet(len(ck["label_to_idx"]))
        m.load_state_dict(ck["model_state_dict"])
        return m.to(dev).eval()

    # ---- (1) environment axis: cuda:1 (different GPU arch), identical construction ----
    print("\n=== (1) ENVIRONMENT AXIS — same construction, cuda:1 (different GPU) ===")
    if torch.cuda.device_count() > 1:
        dev1 = torch.device("cuda:1")
        print(f"  cuda:1 = {torch.cuda.get_device_name(1)}")
        AV1 = load_av(dev1)
        torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = False
        lg1, a1_1 = fwd_az(AV1, make_loader(val_idx, mels_np, labels_all, videos), dev1)
        p1 = lg1.argmax(1)
        print(f"  cuda:1 n_correct={int((p1==y).sum())}/5244 acc={(p1==y).mean():.6f}")
        print(f"  cuda:1 vs cache: preds differ at {int((p1!=cache_pred).sum())} ; "
              f"logits max|d|={np.abs(lg1-cache_logits).max():.3e}")
        print(f"  cuda:1 vs cuda:0: preds differ at {int((p1!=fresh0_logits.argmax(1)).sum())} ; "
              f"logits max|d|={np.abs(lg1-fresh0_logits).max():.3e}")
        # a_mid(zeros): identical input (zeros) + identical weights -> only the machine differs
        cc = torch.load(os.path.join(ROOT, "processed", "deepdive_act_cache.pt"),
                        weights_only=False)
        cache_a1 = np.asarray(cc["AV_clean_audio_zero"]["a_mid_gap"])[0]  # constant row
        print(f"  audio_block1(zeros) GAP cuda:1 vs CACHE max|d|={np.abs(a1_1[0]-cache_a1).max():.3e} "
              f"(same zero input + same weights; nonzero => cross-machine conv fp diff)")
    else:
        print("  only one GPU visible; skipping cross-GPU")

    # ---- (2) mechanism: perturbation injection on cache logits ----
    print("\n=== (2) MECHANISM — perturb cache logits by delta=fresh-cache, scale alpha ===")
    delta = fresh0_logits - cache_logits
    cs = np.sort(cache_logits, axis=1)
    margin = cs[:, -1] - cs[:, -2]
    print(f"  |delta| max={np.abs(delta).max():.3e} median(per-sample max)={np.median(np.abs(delta).max(1)):.3e}")
    for a in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        pa = (cache_logits + a * delta).argmax(1)
        nf = int((pa != cache_pred).sum())
        nc = int((pa == y).sum())
        # are all flips low-margin?
        flips = pa != cache_pred
        below = margin < (a * np.abs(delta).max(1) + 1e-12)
        ok = bool(np.all(below[flips])) if flips.any() else True
        print(f"  alpha={a:4.2f}: n_correct={nc}/5244 acc={nc/5244:.6f}  flips_vs_cache={nf}  "
              f"all_flips_margin<alpha|delta|? {ok}")

    print("\n  -- random Gaussian noise of matched scale (mechanism is generic fp noise) --")
    sigma = float(np.std(delta))
    rng = np.random.default_rng(0)
    for seed in range(5):
        eps = rng.normal(0.0, sigma, size=cache_logits.shape).astype(np.float32)
        pe = (cache_logits + eps).argmax(1)
        flips = pe != cache_pred
        nf = int(flips.sum()); nc = int((pe == y).sum())
        below = margin < np.abs(eps).max(1)
        ok = bool(np.all(below[flips])) if flips.any() else True
        print(f"  seed {seed}: sigma={sigma:.3e} n_correct={nc}/5244 flips={nf} all_flips_low_margin? {ok}")

    # ---- (3) clean separation: high-margin samples NEVER disagree ----
    print("\n=== (3) margin threshold: above the noise, cache & fresh AGREE exactly ===")
    noise_max = np.abs(delta).max()
    hi = margin > noise_max
    agree_hi = int((cache_pred[hi] == fresh0_logits.argmax(1)[hi]).sum())
    print(f"  noise_max(|fresh-cache|)={noise_max:.3e}")
    print(f"  #samples margin>noise_max = {int(hi.sum())} ; cache==fresh among them = {agree_hi}/{int(hi.sum())}")
    lo = ~hi
    dis_lo = int((cache_pred[lo] != fresh0_logits.argmax(1)[lo]).sum())
    print(f"  #samples margin<=noise_max = {int(lo.sum())} ; ALL {int((cache_pred!=fresh0_logits.argmax(1)).sum())} disagreements live here = {dis_lo==int((cache_pred!=fresh0_logits.argmax(1)).sum())}")


if __name__ == "__main__":
    main()
