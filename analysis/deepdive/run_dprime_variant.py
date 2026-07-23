"""TEMP DEBUG INSTRUMENT (debugger, task #5).

Run the EXACT E1c (dprime_precision_test) or E1d (dprime_precision_balanced)
main(), swapping ONLY the AV model between the multiplicative gate
(models/av_fused.pt, AVWordResNet) and the additive gate
(models/av_fused_additive.pt, AVAdditiveWordResNet). Everything else --
pair selection, A/V models, noise grid, seed -- is byte-identical, so d'_A and
d'_V are identical across fusions and ONLY d'_AV (and derived gain) can move.

Usage (run from project root on the pod):
    python run_dprime_variant.py {e1c|e1d} {mult|additive}
Writes the design's default CSV, then copies it to <name>.<fusion>.csv.

Single variable proven: changing the fusion (and nothing else) changes the dip.
"""
import os
import shutil
import sys

import torch

import analyze_av_msi as M

design = sys.argv[1]
fusion = sys.argv[2]
assert design in ("e1c", "e1d") and fusion in ("mult", "additive")

if design == "e1c":
    import dprime_precision_test as D
    default_csv = os.path.join(M.SCRIPT_DIR, "analysis", "msi",
                               "E1c_dprime_precision_sweep.csv")
else:
    import dprime_precision_balanced as D
    default_csv = os.path.join(M.SCRIPT_DIR, "analysis", "msi",
                               "E1d_dprime_precision_balanced.csv")


def _report_av(tag, av_ckpt, model):
    a = float(model.gate.alpha.detach().item())
    print(f"[AV:{tag}] class={type(model).__name__} "
          f"alpha={a:.4f} best_val_acc={av_ckpt.get('best_val_acc')}")


if fusion == "additive":
    from model_av_additive import AVAdditiveWordResNet
    _orig = D._load_models

    def _patched(device):
        models = _orig(device)                    # A, V, AV(multiplicative)
        ck = torch.load(os.path.join(M.SCRIPT_DIR, "models",
                                     "av_fused_additive.pt"),
                        weights_only=False)
        av = AVAdditiveWordResNet(len(ck["label_to_idx"]))
        av.load_state_dict(ck["model_state_dict"])   # strict: raises on mismatch
        av = av.to(device).eval()
        _report_av("additive", ck, av)
        models["AV"] = (av, ck)
        return models

    D._load_models = _patched
else:
    _orig = D._load_models

    def _patched(device):
        models = _orig(device)
        _report_av("mult", models["AV"][1], models["AV"][0])
        return models

    D._load_models = _patched

D.main()

target = default_csv[:-4] + f".{fusion}.csv"
shutil.copy(default_csv, target)
print(f"[saved] {target}")
