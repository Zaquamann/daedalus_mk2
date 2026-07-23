#!/usr/bin/env bash
# Pod-side runner: verify libs (install umap-learn if missing), then build the
# three validation figures. Writes a single log; ends with an ALL_DONE sentinel.
set -uo pipefail
cd /scratch/daedelus
source venv/bin/activate

echo "=== LIB CHECK ==="
python - <<'PY'
import importlib
for m in ["numpy", "torch", "sklearn", "matplotlib"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "IMPORT_FAIL", repr(e))
try:
    import umap
    print("umap", umap.__version__)
except Exception as e:
    print("UMAP_MISSING", repr(e))
PY

if ! python -c "import umap" 2>/dev/null; then
    echo "=== installing umap-learn (umap missing) ==="
    pip install -q umap-learn 2>&1 | tail -5
    python -c "import umap; print('umap now', umap.__version__)" 2>&1
fi

echo "=== RUN FIGURES ==="
python make_validation_figs.py
rc=$?
echo "=== ALL_DONE rc=${rc} ==="
