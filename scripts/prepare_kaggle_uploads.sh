#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/kaggle_uploads"
mkdir -p "$DIST"

cd "$ROOT"

# Code package excludes local env, outputs, and processed data.
zip -r "$DIST/project_code.zip" \
  "src" \
  "config" \
  "scripts" \
  "notebooks" \
  "requirements.txt" \
  "README.md" \
  "Makefile" \
  -x "*/__pycache__/*" "*.pyc" "*.pyo" "*/.venv/*" "*/outputs/*" "*/data/processed/*" "*/data/predictions/*"

zip -r "$DIST/m5_data.zip" "m5-forecasting-accuracy"
zip -r "$DIST/walmart_data.zip" "archive"

echo "Created Kaggle upload files in: $DIST"
echo "- project_code.zip"
echo "- m5_data.zip"
echo "- walmart_data.zip"
