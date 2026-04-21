#!/usr/bin/env bash
# ── Runway Inclusive — Brev GPU Instance Setup ────────────────────────────────
# Run once after cloning on the Brev instance:
#   chmod +x setup_brev.sh && ./setup_brev.sh
#
# Then start the agent:
#   source .venv/bin/activate
#   export NVIDIA_API_KEY=<your-key>
#   nat serve --config_file workflow-config.yml
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO_DIR/.venv"

echo "════════════════════════════════════════════════════════"
echo "  Runway Inclusive — Brev Setup"
echo "  $(date)"
echo "════════════════════════════════════════════════════════"

# ── 1. Python venv ─────────────────────────────────────────────────────────────
echo ""
echo "[1] Creating virtual environment …"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel

# ── 2. Core requirements ───────────────────────────────────────────────────────
echo "[2] Installing core requirements …"
pip install -q -r "$REPO_DIR/requirements.txt"

# ── 3. Install this repo as an editable NAT plugin ────────────────────────────
echo "[3] Installing runway-nat-plugin (editable) …"
pip install -q -e "$REPO_DIR"

# ── 4. RAPIDS GPU detection and install ────────────────────────────────────────
echo "[4] Checking for NVIDIA GPU / CUDA …"
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    echo "    GPU detected — CUDA $CUDA_VER"

    if [[ "$CUDA_MAJOR" == "12" ]]; then
        echo "    Installing RAPIDS cuDF + cuML for CUDA 12 …"
        pip install -q --extra-index-url=https://pypi.nvidia.com \
            "cudf-cu12==25.2.*" "cuml-cu12==25.2.*"
        echo "    ✓ RAPIDS installed — GPU acceleration active"
    elif [[ "$CUDA_MAJOR" == "11" ]]; then
        echo "    Installing RAPIDS cuDF + cuML for CUDA 11 …"
        pip install -q --extra-index-url=https://pypi.nvidia.com \
            "cudf-cu11==25.2.*" "cuml-cu11==25.2.*"
        echo "    ✓ RAPIDS installed — GPU acceleration active"
    else
        echo "    ⚠  CUDA $CUDA_VER — no matching RAPIDS build, skipping GPU libs"
    fi
else
    echo "    No GPU detected — running CPU-only mode"
fi

# ── 5. Run ML engine to (re)generate clustered data with GPU ──────────────────
echo ""
echo "[5] Running ML engine (Condé Nast Accelerated Intelligence Layer) …"
python "$REPO_DIR/ml_engine.py"

# ── 6. Verify data files ───────────────────────────────────────────────────────
echo ""
echo "[6] Data verification …"
for f in catalog.json schedule.json telemetry.json designers_clustered.json \
          tribe_manifest.json knn_index.pkl met_gala_themes.json; do
    path="$REPO_DIR/data/$f"
    if [[ -f "$path" ]]; then
        size=$(du -sh "$path" | cut -f1)
        echo "    ✓  $f  ($size)"
    else
        echo "    ✗  $f  MISSING — run prep_data.py if raw TMDB/MovieLens data is available"
    fi
done

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Setup complete. Start the agent with:"
echo ""
echo "    source .venv/bin/activate"
echo "    export NVIDIA_API_KEY=<your-key>"
echo "    nat serve --config_file workflow-config.yml"
echo "════════════════════════════════════════════════════════"
