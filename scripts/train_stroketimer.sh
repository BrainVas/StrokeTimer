#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

CONFIG="${CONFIG:-configs/stroketimer_best.yaml}"
DATA_ROOT="${DATA_ROOT:-data/center_data}"
CSV_PATH="${CSV_PATH:-data/meta_with_phase_center_stratified_balanced.csv}"
CLASS_FREQ_JSON="${CLASS_FREQ_JSON:-configs/ncct_class_frequency.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stroketimer_best}"
SEED="${SEED:-42}"

export DATA_ROOT CSV_PATH CLASS_FREQ_JSON OUTPUT_DIR

mkdir -p "${OUTPUT_DIR}"

args=(--cfg "${CONFIG}" --seed "${SEED}")
if [[ -n "${BATCH_SIZE:-}" ]]; then
  args+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${TRIAL:-}" ]]; then
  args+=(--trial "${TRIAL}")
fi

python -m stroketimer.train "${args[@]}"
