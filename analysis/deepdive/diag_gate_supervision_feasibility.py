#!/usr/bin/env python3
"""TEMP DEBUG (task #20, candidate-(b) feasibility) — the transfer probe showed
NO input-space video augmentation (pixel/blur/drop/mixup/tshuffle) moves v_pen
into the E1d-confusable region (all transfer-signals ~0), yet that region IS
linearly present in v_pen (clean-E1d vs clean-nonE1d AUC 0.813). So candidate (a)
[input degradation] can't teach the gate to route on E1d. Candidate (b) instead
SUPERVISES the rel_gate directly from per-trial head-correctness. This probe tests
(b)'s premise on the CURRENT model, clean val:

  D1  decode  video-head-correct  from v_pen            (CV ROC-AUC)
      -> can a gate LEARN, from the vector it already reads, when video is
         unreliable? high AUC => candidate (b)'s target is learnable.
  D2  decode  "audio is the correct head"  from [a_pen,v_pen]
         on DISAGREEMENT trials (exactly one head right)  (CV ROC-AUC)
      -> the exact per-trial ROUTING target a gate-supervision loss would use.
  context: video-head-correct rate & audio-head-correct rate on E1d-pair classes
         vs the rest (confirms E1d = audio-strong / video-weak regime).

READ-ONLY / CHEAP. No training of the model, no production edits.

Run: CUDA_VISIBLE_DEVICES=1 python analysis/deepdive/diag_gate_supervision_feasibility.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

import dprime_latefusion as dlf  # noqa: E402
from analyze_av_msi import _NoisyAudioView, BATCH_SIZE  # noqa: E402

dlf.NW = 6
SEED = 0


def auc(X, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X, y, cv=4, scoring="roc_auc").mean()


def main():
    device = torch.device("cuda")
    base = dlf.RawNoisyAVDataset(noise=False, t_stride=dlf.T_STRIDE,
                                 return_video=True)
    val_idx = torch.load(os.path.join(dlf.SCRIPT_DIR, "processed", "splits.pt"),
                         weights_only=False)["val_idx"]
    models = dlf._load_models(device)
    A, V = models["A"][0], models["V"][0]
    ck = torch.load(dlf.LATE_CKPT, weights_only=False)
    AVl = dlf.AVLateFusionReliabilityWordResNet(
        len(ck["label_to_idx"]), use_mid_gate=ck.get("use_mid_gate", False))
    AVl.load_state_dict(ck["model_state_dict"])
    AVl = AVl.to(device).eval()
    print(f"ckpt={os.path.basename(dlf.LATE_CKPT)} acc={ck.get('best_val_acc')}",
          flush=True)

    pair_ids, _dV = dlf._select_pairs("e1d", A, V, base, val_idx, device)
    pair_classes = sorted({c for p in pair_ids for c in p})

    store = {}
    AVl.audio_gap.register_forward_hook(
        lambda m, i, o: store.__setitem__("a", o.flatten(1).detach().cpu().numpy()))
    AVl.visual_gap.register_forward_hook(
        lambda m, i, o: store.__setitem__("v", o.flatten(1).detach().cpu().numpy()))

    loader = DataLoader(_NoisyAudioView(base, val_idx, sigma_mult=0.0, seed=SEED),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=dlf.NW, pin_memory=True)

    apen, vpen, vcorr, acorr, ys = [], [], [], [], []
    with torch.no_grad():
        for mel, vid, y in loader:
            m1 = mel.unsqueeze(1).to(device, non_blocking=True)
            vd = vid.to(device, non_blocking=True)
            _logits, la, lv, _w = AVl(m1, vd, return_parts=True)
            apen.append(store["a"])
            vpen.append(store["v"])
            yn = y.numpy()
            vcorr.append((lv.argmax(1).cpu().numpy() == yn))
            acorr.append((la.argmax(1).cpu().numpy() == yn))
            ys.append(yn)
    apen = np.concatenate(apen)
    vpen = np.concatenate(vpen)
    vcorr = np.concatenate(vcorr).astype(int)
    acorr = np.concatenate(acorr).astype(int)
    ys = np.concatenate(ys)
    e1d = np.isin(ys, pair_classes)

    print(f"\nclean val n={len(ys)}  video-head acc={vcorr.mean():.3f}  "
          f"audio-head acc={acorr.mean():.3f}", flush=True)
    print(f"E1d-pair classes (n_trials={int(e1d.sum())}): "
          f"video-head acc={vcorr[e1d].mean():.3f}  "
          f"audio-head acc={acorr[e1d].mean():.3f}   <- E1d = video weak/audio "
          f"strong, as designed", flush=True)
    print(f"non-E1d (n={int((~e1d).sum())}): video-head acc={vcorr[~e1d].mean():.3f}"
          f"  audio-head acc={acorr[~e1d].mean():.3f}", flush=True)

    # D1: decode video-head-correct from v_pen (the gate-supervision target)
    a1 = auc(vpen, vcorr)
    print(f"\n[D1] decode video-head-CORRECT from v_pen        : CV ROC-AUC = "
          f"{a1:.3f}  (>>0.5 => a supervised gate CAN learn video-reliability)",
          flush=True)
    # D1b: same but restricted to where it matters most (clean both-present)
    a1b = auc(vpen, (vcorr == 1).astype(int))
    # D2: routing target on DISAGREEMENT trials — is audio the correct head?
    dis = (vcorr != acorr)
    feat = np.concatenate([apen, vpen], axis=1)
    a2 = auc(feat[dis], acorr[dis])
    a2v = auc(vpen[dis], acorr[dis])
    print(f"[D2] decode 'audio is right' on DISAGREEMENT trials "
          f"(n={int(dis.sum())}):", flush=True)
    print(f"        from [a_pen,v_pen]: CV ROC-AUC = {a2:.3f}   "
          f"from v_pen only: {a2v:.3f}", flush=True)
    print(f"        (this is the exact per-trial routing label candidate (b) "
          f"would supervise w_a with)", flush=True)
    print("FEAS_RC=0", flush=True)


if __name__ == "__main__":
    main()
