#!/usr/bin/env python3
"""Cross-GPU (5090 vs A6000) flip counts for ALL THREE AV conditions, identical
construction. Shows audio-zero is uniquely fp-fragile; full-AV/video-zero robust."""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
ROOT = "/home/vishnu/coding_proj/daedalus/daedalus"; sys.path.insert(0, ROOT)


def make_loader(val_idx, mels_np, labels_all, videos, stride=2, batch=64):
    class Vw(Dataset):
        def __len__(self): return len(val_idx)
        def __getitem__(self, k):
            g = int(val_idx[k]); mel = torch.from_numpy(mels_np[g]).unsqueeze(0)
            v = np.array(videos[g])
            if stride > 1: v = v[::stride]
            return mel, torch.from_numpy(v).unsqueeze(0).float() / 255.0, int(labels_all[g])
    return DataLoader(Vw(), batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def fwd_all(AV, loader, device):
    full, vz, az = [], [], []
    for mel, vid, y in loader:
        mel = mel.to(device); vid = vid.to(device)
        a_mid = AV.audio_block1(mel)
        v_mid = AV.visual(vid)
        full.append(AV.fc(AV.dropout(AV.gap(AV.audio_block2(AV.gate(a_mid, v_mid))).flatten(1))).float().cpu().numpy())
        v0 = torch.zeros_like(a_mid)
        vz.append(AV.fc(AV.dropout(AV.gap(AV.audio_block2(AV.gate(a_mid, v0))).flatten(1))).float().cpu().numpy())
        a0 = AV.audio_block1(torch.zeros_like(mel))
        az.append(AV.fc(AV.dropout(AV.gap(AV.audio_block2(AV.gate(a0, v_mid))).flatten(1))).float().cpu().numpy())
    return np.concatenate(full), np.concatenate(vz), np.concatenate(az)


def main():
    from model_av import AVWordResNet
    proc = os.path.join(ROOT, "processed")
    val_idx = np.asarray(torch.load(os.path.join(proc, "splits.pt"), weights_only=False)["val_idx"], dtype=np.int64)
    dav = torch.load(os.path.join(proc, "dataset_av.pt"), weights_only=False)
    mels_np = np.asarray(dav["spectrograms"]); labels_all = np.asarray(dav["labels"]).astype(np.int64)
    T, H, W = dav["video_shape"]
    videos = np.memmap(dav["video_cache_path"], dtype=np.uint8, mode="r", shape=(len(labels_all), T, H, W))
    y = labels_all[val_idx]

    def load(dev):
        ck = torch.load(os.path.join(ROOT, "models", "av_fused.pt"), weights_only=False, map_location="cpu")
        m = AVWordResNet(len(ck["label_to_idx"])); m.load_state_dict(ck["model_state_dict"]); return m.to(dev).eval()

    torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = False
    res = {}
    for g in [0, 1]:
        dev = torch.device(f"cuda:{g}")
        res[g] = fwd_all(load(dev), make_loader(val_idx, mels_np, labels_all, videos), dev)
        print(f"cuda:{g} = {torch.cuda.get_device_name(g)}")
    print("\ncondition       5090-vs-A6000 logits max|d|   preds-differ   acc5090   accA6000")
    for i, nm in enumerate(["AV_full", "AV_video_zero", "AV_audio_zero"]):
        a, b = res[0][i], res[1][i]
        print(f"{nm:14s}  {np.abs(a-b).max():.3e}                  {int((a.argmax(1)!=b.argmax(1)).sum()):4d}"
              f"          {(a.argmax(1)==y).mean():.6f}  {(b.argmax(1)==y).mean():.6f}")


if __name__ == "__main__":
    main()
