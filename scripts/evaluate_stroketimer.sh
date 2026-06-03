#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

CONFIG="${CONFIG:-configs/stroketimer_best.yaml}"
DATA_ROOT="${DATA_ROOT:-data/center_data}"
CSV_PATH="${CSV_PATH:-data/meta_with_phase_center_stratified_balanced.csv}"
CLASS_FREQ_JSON="${CLASS_FREQ_JSON:-configs/ncct_class_frequency.json}"
CHECKPOINT="${CHECKPOINT:-checkpoints/best/final_model_checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_stroketimer}"
SEED="${SEED:-42}"

export DATA_ROOT CSV_PATH CLASS_FREQ_JSON CHECKPOINT OUTPUT_DIR

mkdir -p "${OUTPUT_DIR}"

python -m stroketimer.visualize \
  --cfg "${CONFIG}" \
  --ckpt "${CHECKPOINT}" \
  --out_dir "${OUTPUT_DIR}" \
  --seed "${SEED}"
