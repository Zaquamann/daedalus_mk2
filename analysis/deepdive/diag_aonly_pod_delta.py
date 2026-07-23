"""TEMP DEBUG INSTRUMENT (debugger) — isolate the A-only local-vs-pod delta.

Run identically on local (reuben-ML) and pod (dev-codex); diff the printed
fingerprints. Each block isolates ONE candidate variable:
  [V] versions
  [W] model weights        -> sha256 of concatenated state_dict bytes
  [R] noise RNG            -> sha256 + sum of the per-idx noise for fixed idx
  [M] mel model-input      -> sha256 + float64 stats of the full stacked mel
  [F] forward (cpu/cuda)   -> A-only acc + predictions sha256 + logits sum

If [W],[R],[M] all match across machines and CPU [F] matches but CUDA [F]
differs -> the delta is the CUDA kernel. Otherwise the differing block names
the cause.
"""
import hashlib
import os
import sys

import numpy as np
import scipy
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from analyze_av_msi import BATCH_SIZE, T_STRIDE, _forward_A  # noqa: E402
from analyze_av_deepdive import _NoisyAVView  # noqa: E402
from dataset_raw_noisy import RawNoisyAVDataset  # noqa: E402
from paired_dataset import _read_wav, _pad_audio, _wav_to_log_mel  # noqa: E402
from train import WordResNet  # noqa: E402

SIG_A, SIG_V = 0.008487, 0.212586
A_CKPT = os.path.join(ROOT, "models", "audio_only_filtered.pt")


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def main():
    print("=" * 64)
    print("[V] versions")
    print(f"  torch={torch.__version__} numpy={np.__version__} "
          f"scipy={scipy.__version__}")
    print(f"  cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"  gpu={dev_name}")
    print(f"  py={sys.version.split()[0]}")

    # ---- [W] weights -------------------------------------------------------
    a_ckpt = torch.load(A_CKPT, weights_only=False, map_location="cpu")
    sd = a_ckpt["model_state_dict"]
    blob = b"".join(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes()
                    for _, v in sorted(sd.items()))
    print("[W] model weights")
    print(f"  state_dict sha256[:16] = {hashlib.sha256(blob).hexdigest()[:16]}")
    print(f"  n_tensors={len(sd)}  best_val_acc={a_ckpt.get('best_val_acc')}")

    model = WordResNet(len(a_ckpt["label_to_idx"]))
    model.load_state_dict(sd)
    model.eval()

    # ---- data --------------------------------------------------------------
    base = RawNoisyAVDataset(noise=False, t_stride=T_STRIDE, return_video=True)
    val_idx = torch.load(os.path.join(ROOT, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    val_idx = np.asarray(val_idx, dtype=np.int64)
    print(f"  val_idx sha={sha(val_idx)} N={len(val_idx)}")

    # ---- [R] noise RNG (isolated, 3 fixed val indices) ---------------------
    print("[R] noise RNG (per-idx, isolated)")
    for k in (0, 100, 5000):
        idx = int(val_idx[k])
        audio = _read_wav(base.audio_paths[idx])
        rms = float(np.sqrt(float((audio ** 2).mean()) + 1e-12))
        sigma = SIG_A * rms
        rng = np.random.default_rng(0 + idx)
        noise = rng.standard_normal(len(audio)).astype(np.float32) * sigma
        print(f"  k={k:>4d} idx={idx:>5d} len={len(audio)} "
              f"audio_sha={sha(audio)} noise_sha={sha(noise)} "
              f"noise_sum={noise.astype(np.float64).sum():.8e}")

    # ---- [M] mel model-input (full stacked tensor) -------------------------
    view = _NoisyAVView(base, val_idx, sigma_a_mult=SIG_A,
                        sigma_v_mult=SIG_V, seed=0)
    mels = np.zeros((len(val_idx), 80, 99), dtype=np.float32)
    labels = np.zeros(len(val_idx), dtype=np.int64)
    for k in range(len(val_idx)):
        mel, _v, y = view[k]
        mels[k] = mel.numpy()
        labels[k] = y
    print("[M] mel model-input (stacked, all val)")
    print(f"  mel sha={sha(mels)} shape={mels.shape}")
    print(f"  mel f64 sum={mels.astype(np.float64).sum():.10e} "
          f"min={mels.min():.6f} max={mels.max():.6f} "
          f"mean={mels.astype(np.float64).mean():.8f}")

    # ---- [F] forward: CPU then CUDA ---------------------------------------
    def run(device):
        m = model.to(device).eval()
        preds, logit_sum, n = [], 0.0, len(val_idx)
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                xb = torch.from_numpy(mels[i:i + BATCH_SIZE]).unsqueeze(1).to(device)
                lo = m(xb)
                logit_sum += float(lo.double().sum().item())
                preds.append(lo.argmax(1).cpu().numpy())
        preds = np.concatenate(preds)
        acc = float((preds == labels).mean())
        return preds, acc, logit_sum

    print("[F] forward")
    p_cpu, a_cpu, s_cpu = run(torch.device("cpu"))
    print(f"  CPU : acc={a_cpu*100:.4f}%  preds_sha={sha(p_cpu)}  "
          f"logit_f64_sum={s_cpu:.6e}")
    if torch.cuda.is_available():
        p_cu, a_cu, s_cu = run(torch.device("cuda"))
        print(f"  CUDA: acc={a_cu*100:.4f}%  preds_sha={sha(p_cu)}  "
              f"logit_f64_sum={s_cu:.6e}")
        print(f"  CPU-vs-CUDA pred disagreements (this machine): "
              f"{int((p_cpu != p_cu).sum())} / {len(p_cpu)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
