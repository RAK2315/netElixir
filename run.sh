#!/usr/bin/env bash
# =============================================================================
# AIgnition 2026 — single entry point for the automated scoring pipeline.
#
#   ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
#
# Runs end to end with one invocation: (1) generate features from whatever is in
# DATA_DIR, then (2) load the committed pickled model and write predictions.
# No prompts, no network, fails loudly.
# =============================================================================
set -euo pipefail

# Resolve repo root so the script works from any CWD, and make `src` importable
# (required so the pickled ForecastModel class resolves at load time).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# Accept arguments, fall back to sensible defaults for local runs.
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Prefer python3, fall back to python.
PY="$(command -v python3 || command -v python)"

FEATURES_FILE="features.parquet"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "[run.sh] DATA_DIR=$DATA_DIR  MODEL_PATH=$MODEL_PATH  OUTPUT_PATH=$OUTPUT_PATH"

# 1. Generate the features the model expects from the data.
"$PY" src/generate_features.py --data-dir "$DATA_DIR" --out "$FEATURES_FILE"

# 2. Load the pickled model and produce predictions.
"$PY" src/predict.py --features "$FEATURES_FILE" --model "$MODEL_PATH" --output "$OUTPUT_PATH"

echo "[run.sh] Done. Predictions written to $OUTPUT_PATH"
