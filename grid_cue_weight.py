"""Cue-reliability grid: how the AV model weights audio vs reliability of each cue.

Axes: A reliability (set by sigma_a) x V reliability (set by sigma_v).
Heat: audio weight w_A = follow_A / (follow_A + follow_V) on DISAGREEMENT trials
      (trials where the A-only and V-only models predict different words). This is
      the behavioural read-out of cue weighting: how often the fused model sides
      with audio vs video when the two cues conflict.

Noise is reproduced bit-identically to _NoisyAVView (audio RNG seed+idx, video RNG
seed+idx+10_000_000), so v_pred is measured on exactly the video AV sees. Efficient:
noisy mels cached per sigma_a; the visual feature v_mid is computed once per batch per
sigma_v and reused across all sigma_a. Pinned val set, seed 0, eager fp32.
"""
import os, csv, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from analyze_av_msi import BATCH_SIZE, T_STRIDE, _load_models
from dataset_raw_noisy import RawNoisyAVDataset
from paired_dataset import _pad_audio, _read_wav, _wav_to_log_mel

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
NW = 8

# sigma grids -> spread of unisensory reliability (from D1_iso_perf_lookup targets + extras)
SIG_A = [0.0, 0.005867, 0.008487, 0.013027, 0.023780, 0.040]   # A-only ~ 93,85,75,65,50,~20%
SIG_V = [0.0, 0.114367, 0.212586, 0.246782, 0.298076, 0.400]   # V-only ~ 86,85,75,65,50,~20%


class MelView(Dataset):
    def __init__(self, base, idx, sigma_a):
        self.base, self.idx, self.sa = base, np.asarray(idx, np.int64), float(sigma_a)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        idx = int(self.idx[k])
        audio = _read_wav(self.base.audio_paths[idx])
        if self.sa > 0:
            rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
            rng = np.random.default_rng(SEED + idx)
            audio = audio + rng.standard_normal(len(audio)).astype(np.float32) * (self.sa * rms)
        pad_left = int(self.base.pad_offsets[idx])
        mel = _wav_to_log_mel(_pad_audio(audio, pad_left)).astype(np.float32)
        return torch.from_numpy(mel), int(self.base.labels[idx]), k


class VidView(Dataset):
    def __init__(self, base, idx, sigma_v):
        self.base, self.idx, self.sv = base, np.asarray(idx, np.int64), float(sigma_v)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        idx = int(self.idx[k])
        v = np.array(self.base._videos[idx])
        if self.base.t_stride > 1:
            v = v[:: self.base.t_stride]
        v = v.astype(np.float32)[np.newaxis, ...] / 255.0
        if self.sv > 0:
            rng = np.random.default_rng(SEED + idx + 10_000_000)
            v = v + rng.standard_normal(v.shape).astype(np.float32) * (self.sv * float(v.std()))
        return torch.from_numpy(v), int(self.base.labels[idx]), k


@torch.no_grad()
def main():
    device = torch.device("cuda")
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    val_idx = torch.load(os.path.join(HERE, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    val_idx = np.asarray(val_idx, np.int64)
    N = len(val_idx)
    m = _load_models(device)
    A_model, V_model, AV_model = m["A"][0], m["V"][0], m["AV"][0]

    labels = np.empty(N, np.int64)

    # --- cache noisy mels + A-only preds per sigma_a ---
    mel_cache, a_pred, a_acc = [], [], []
    for sa in SIG_A:
        t0 = time.time()
        mc = torch.empty(N, 80, 99, dtype=torch.float32)
        loader = DataLoader(MelView(base, val_idx, sa), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NW, pin_memory=True)
        ap = np.empty(N, np.int64)
        for mel, y, k in loader:
            k = k.numpy()
            mc[k] = mel
            labels[k] = y.numpy()
            logits = A_model(mel.unsqueeze(1).to(device, non_blocking=True))
            ap[k] = logits.argmax(1).cpu().numpy()
        mel_cache.append(mc)
        a_pred.append(ap)
        a_acc.append(float((ap == labels).mean()))
        print(f"[A] sigma_a={sa:.6f}  A_acc={a_acc[-1]*100:6.3f}%  ({time.time()-t0:.0f}s)", flush=True)

    # --- per sigma_v: V preds + AV preds for every sigma_a (reuse v_mid) ---
    v_pred, v_acc = [], []
    av_pred = [[np.empty(N, np.int64) for _ in SIG_A] for _ in SIG_V]
    for j, sv in enumerate(SIG_V):
        t0 = time.time()
        vp = np.empty(N, np.int64)
        loader = DataLoader(VidView(base, val_idx, sv), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NW, pin_memory=True)
        for vid, y, k in loader:
            k = k.numpy()
            vid = vid.to(device, non_blocking=True)
            v_mid = AV_model.visual(vid)                       # computed once, reused
            vp[k] = V_model(vid).argmax(1).cpu().numpy()
            for i in range(len(SIG_A)):
                mel = mel_cache[i][k].unsqueeze(1).to(device, non_blocking=True)
                a_mid = AV_model.audio_block1(mel)
                a_fused = AV_model.gate(a_mid, v_mid)
                x = AV_model.audio_block2(a_fused)
                pen = AV_model.gap(x).flatten(1)
                av_pred[j][i][k] = AV_model.fc(pen).argmax(1).cpu().numpy()
        v_pred.append(vp)
        v_acc.append(float((vp == labels).mean()))
        print(f"[V] sigma_v={sv:.6f}  V_acc={v_acc[-1]*100:6.3f}%  ({time.time()-t0:.0f}s)", flush=True)

    # --- cue weight per cell ---
    out = os.path.join(HERE, "analysis", "msi", "E1f_cue_weight_grid.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_a", "sigma_v", "a_acc", "v_acc",
                    "n_disagree", "follow_a", "follow_v", "neither", "w_audio"])
        for j, sv in enumerate(SIG_V):
            for i, sa in enumerate(SIG_A):
                dis = a_pred[i] != v_pred[j]
                fa = int(((av_pred[j][i] == a_pred[i]) & dis).sum())
                fv = int(((av_pred[j][i] == v_pred[j]) & dis).sum())
                nd = int(dis.sum())
                wa = fa / (fa + fv) if (fa + fv) else float("nan")
                w.writerow([f"{sa:.6f}", f"{sv:.6f}", f"{a_acc[i]:.6f}", f"{v_acc[j]:.6f}",
                            nd, fa, fv, nd - fa - fv, f"{wa:.6f}"])
    print(f"\nwrote {out}", flush=True)
    print("a_acc grid:", [f"{x*100:.1f}" for x in a_acc], flush=True)
    print("v_acc grid:", [f"{x*100:.1f}" for x in v_acc], flush=True)


if __name__ == "__main__":
    main()
