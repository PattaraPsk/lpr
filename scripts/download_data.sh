#!/usr/bin/env bash
# ==============================================================================
# Script: download_data.sh
# Description: Prepares dataset directories and verifies raw computer vision data
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "=========================================================="
echo " Computer Vision Project - Dataset Setup & Ingestion"
echo "=========================================================="

# Create required directories
mkdir -p data/raw data/interim data/processed data/splits

echo "[1/2] Checking raw data directory..."
RAW_COUNT=$(find data/raw -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l || echo "0")
echo "      Found ${RAW_COUNT} raw images in data/raw"

if [ "${RAW_COUNT}" -eq 0 ]; then
    echo "      [NOTE] No raw images found in data/raw."
    echo "      Place images in data/raw/<label>/<image>.jpg or data/raw/<label>_xxx.jpg"
fi

echo "[2/2] Generating dataset splits (train / val / test)..."
python -m src.data.prepare --source-dir data/raw --splits-dir data/splits

echo "=========================================================="
echo " Dataset setup completed successfully."
echo "=========================================================="
