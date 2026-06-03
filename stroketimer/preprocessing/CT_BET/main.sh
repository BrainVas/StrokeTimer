#!/usr/bin/env bash
#SBATCH --job-name=ct_bet
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs image_data mask_data results_folder weights_folder

if [[ -n "${CONDA_ENV:-}" ]]; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
  else
    source ~/.bashrc 2>/dev/null || true
  fi
  conda activate "${CONDA_ENV}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
ulimit -s unlimited

IMAGE_FOLDER="${IMAGE_FOLDER:-image_data}"
OUT_DIR="${OUT_DIR:-results_folder}"
WEIGHTS="${WEIGHTS:-weights_folder/unet_CT_SS_3D_201843_163521.h5}"
GPU="${GPU:-0}"

echo "========== CT_BET Started: $(date) =========="
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output dir:   ${OUT_DIR}"
echo "Weights:      ${WEIGHTS}"
echo "Python:       $(python --version) @ $(which python)"
echo "============================================="

python unet_CT_SS.py \
  --mode predict3d \
  --image_folder "${IMAGE_FOLDER}" \
  --out_dir "${OUT_DIR}" \
  --weights "${WEIGHTS}" \
  --img_row 256 \
  --img_col 256 \
  --gpu "${GPU}" \
  --save_pred_mask
