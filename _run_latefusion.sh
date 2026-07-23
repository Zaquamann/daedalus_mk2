#!/usr/bin/env bash
# Pod-side runner for task #7 (late-fusion reliability fix). Stages: lib check →
# seed-0 cheap-prove retrain (logged per-epoch; backgroundable) → ALL_DONE
# sentinel. Eval is run SEPARATELY after the per-epoch video-head val-acc is
# confirmed healthy (kill if at chance by ~ep40):
#     python analysis/deepdive/dprime_latefusion.py
# Pod dir spelling is "daedelus" (pod) vs "daedalus" (local) — confirm at push
# time with: runai workspace exec dev-codex -- bash -lc 'ls /scratch/daedelus'
set -uo pipefail
cd /scratch/daedelus
source venv/bin/activate

echo "=== LIB / FILE CHECK ==="
python - <<'PY'
import importlib, os
for m in ["numpy", "torch"]:
    mod = importlib.import_module(m)
    print(m, getattr(mod, "__version__", "?"))
print("cuda", __import__("torch").cuda.is_available())
for f in ["model_av_latefusion.py", "train_av_latefusion.py",
          "analysis/deepdive/dprime_latefusion.py",
          "processed/splits.pt", "models/av_fused.pt",
          "models/audio_only_filtered.pt"]:
    print(("OK  " if os.path.exists(f) else "MISS"), f)
PY

echo "=== RETRAIN (seed=0, late-fusion + noise aug) ==="
# Env knobs (defaults baked in the script): EPOCHS=60 NOISE_HI=0.22 AUX_W=0.5 WORKERS=16 SEED=0
# COMPILE=0: dev-codex has no Python.h for Triton JIT (proven in Q14 pre-check);
# compile is train-speed-only and never enters an eval number, so eager is safe.
COMPILE="${COMPILE:-0}" EPOCHS="${EPOCHS:-60}" python train_av_latefusion.py
rc=$?
echo "=== ALL_DONE rc=${rc} ==="
