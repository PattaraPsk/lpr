#!/usr/bin/env bash
# ==============================================================================
# Script: run_training.sh
# Description: Executes model training pipeline with YAML configuration
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

DATASET_CFG="${1:-configs/dataset.yaml}"
MODEL_CFG="${2:-configs/model.yaml}"
TRAIN_CFG="${3:-configs/train.yaml}"

echo "=========================================================="
echo " Starting Computer Vision Training Pipeline"
echo " Dataset Config : ${DATASET_CFG}"
echo " Model Config   : ${MODEL_CFG}"
echo " Train Config   : ${TRAIN_CFG}"
echo "=========================================================="

# Ensure output directories exist
mkdir -p outputs/checkpoints outputs/logs outputs/predictions

python -m src.training.train \
    --dataset-config "${DATASET_CFG}" \
    --model-config "${MODEL_CFG}" \
    --train-config "${TRAIN_CFG}"

echo "=========================================================="
echo " Training process finished."
echo " Checkpoints stored in : outputs/checkpoints"
echo " Logs stored in        : outputs/logs"
echo "=========================================================="
