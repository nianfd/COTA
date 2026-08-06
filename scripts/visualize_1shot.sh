#!/bin/bash
set -euo pipefail

# One-shot paper visualization for OTM/COTA-SubspaceAD.
#
# Required:
#   --normal_path  one normal support image
#   --anomaly_path one test image
#
# Example:
#   bash scripts/visualize_1shot.sh \
#       --normal_path ./datasets/mvtec-ad/bottle/train/good/000.png \
#       --anomaly_path ./datasets/mvtec-ad/bottle/test/broken_large/000.png \
#       --output_dir ./visualization_results
#
# Optional environment overrides use the same defaults as benchmark_few_shot.sh:
#   MODEL_CKPT, IMAGE_RES, LAYERS, AGG_METHOD, AUG_COUNT, PCA_EV,
#   OTM_BANK_SIZE, OTM_BANK_CHUNK, OTM_BANK_SEED, SEED, PYTHON_BIN.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_CKPT="${MODEL_CKPT:-../facebook/dinov2-with-registers-giant}"
IMAGE_RES="${IMAGE_RES:-672}"
LAYERS="${LAYERS:--12,-13,-14,-15,-16,-17,-18}"
AGG_METHOD="${AGG_METHOD:-mean}"
AUG_COUNT="${AUG_COUNT:-30}"
PCA_EV="${PCA_EV:-0.99}"
OTM_BANK_SIZE="${OTM_BANK_SIZE:-8192}"
OTM_BANK_CHUNK="${OTM_BANK_CHUNK:-2048}"
OTM_BANK_SEED="${OTM_BANK_SEED:-42}"
SEED="${SEED:-42}"

exec "${PYTHON_BIN}" -u tools/visualize_1shot.py \
    --model_ckpt "${MODEL_CKPT}" \
    --image_res "${IMAGE_RES}" \
    --layers="${LAYERS}" \
    --agg_method "${AGG_METHOD}" \
    --aug_count "${AUG_COUNT}" \
    --pca_ev "${PCA_EV}" \
    --otm_bank_size "${OTM_BANK_SIZE}" \
    --otm_bank_chunk "${OTM_BANK_CHUNK}" \
    --otm_bank_seed "${OTM_BANK_SEED}" \
    --seed "${SEED}" \
    "$@"
