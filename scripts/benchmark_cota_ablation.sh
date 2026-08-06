#!/bin/bash
set -euo pipefail

# One-command image/pixel decoupling and COTA ablation.
#
# Full MVTec, selected diagnostic categories:
#   bash scripts/benchmark_cota_ablation.sh full_shot mvtec \
#       "cable grid transistor metal_nut pill zipper"
#
# MVTec 1-shot:
#   K_VALUES=1 bash scripts/benchmark_cota_ablation.sh few_shot mvtec \
#       "cable grid transistor metal_nut pill zipper"
#
# All MVTec categories:
#   bash scripts/benchmark_cota_ablation.sh full_shot mvtec
#
# A shorter first screen:
#   COTA_VARIANTS="baseline otm_decoupled cota cota_safe" \
#       bash scripts/benchmark_cota_ablation.sh full_shot mvtec "cable pill"

PROTOCOL="${1:-full_shot}"
DATASET_ALIAS="${2:-mvtec}"
CATEGORY_STRING="${3:-}"
COTA_ABLATION_ROOT="${COTA_ABLATION_ROOT:-./cota_ablation_results}"
OTM_BANK_SIZE="${OTM_BANK_SIZE:-8192}"
OTM_BANK_CHUNK="${OTM_BANK_CHUNK:-2048}"
OTM_BANK_SEED="${OTM_BANK_SEED:-42}"
COTA_VARIANTS="${COTA_VARIANTS:-baseline otm_full otm_decoupled calibrated_q calibrated_sum cota cota_safe}"
read -r -a VARIANTS <<< "${COTA_VARIANTS}"

case "${DATASET_ALIAS}" in
    mvtec|mvtec_ad)
        RUN_DATASET_VALUE="mvtec"
        ;;
    visa)
        RUN_DATASET_VALUE="visa"
        ;;
    all)
        RUN_DATASET_VALUE="all"
        ;;
    *)
        echo "Dataset must be mvtec, visa, or all."
        exit 2
        ;;
esac

case "${PROTOCOL}" in
    full_shot)
        BENCHMARK_SCRIPT="scripts/benchmark_full_shot.sh"
        OUTPUT_ENV="FULL_OUT_ROOT"
        ;;
    few_shot)
        BENCHMARK_SCRIPT="scripts/benchmark_few_shot.sh"
        OUTPUT_ENV="FEW_OUT_ROOT"
        ;;
    batched0shot)
        BENCHMARK_SCRIPT="scripts/benchmark_batched0shot.sh"
        OUTPUT_ENV="ZERO_OUT_ROOT"
        ;;
    *)
        echo "Protocol must be full_shot, few_shot, or batched0shot."
        exit 2
        ;;
esac

RUN_ROOT="${COTA_ABLATION_ROOT}/${PROTOCOL}/${DATASET_ALIAS}"
mkdir -p "${RUN_ROOT}" logs

for VARIANT in "${VARIANTS[@]}"; do
    VARIANT_OUT="${RUN_ROOT}/${VARIANT}"
    echo "===== ${PROTOCOL} ${DATASET_ALIAS} variant=${VARIANT} ====="
    env \
        RUN_DATASET="${RUN_DATASET_VALUE}" \
        CATEGORIES="${CATEGORY_STRING}" \
        OTM_VARIANT="${VARIANT}" \
        OTM_BANK_SIZE="${OTM_BANK_SIZE}" \
        OTM_BANK_CHUNK="${OTM_BANK_CHUNK}" \
        OTM_BANK_SEED="${OTM_BANK_SEED}" \
        "${OUTPUT_ENV}=${VARIANT_OUT}" \
        bash "${BENCHMARK_SCRIPT}"
done

python tools/summarize_cota_ablation.py --root "${RUN_ROOT}"
echo "===== COTA ablation complete ====="
