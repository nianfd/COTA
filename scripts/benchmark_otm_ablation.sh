#!/bin/bash
set -euo pipefail

# One-command comparison of every OTM module.
#
# Examples:
#   bash scripts/benchmark_otm_ablation.sh full_shot mvtec "cable screw transistor"
#   bash scripts/benchmark_otm_ablation.sh few_shot visa "pcb1 pcb2 pcb3 pcb4"
#   K_VALUES="1" bash scripts/benchmark_otm_ablation.sh few_shot mvtec "cable"
#   bash scripts/benchmark_otm_ablation.sh batched0shot all

PROTOCOL="${1:-full_shot}"
DATASET_ALIAS="${2:-mvtec}"
CATEGORY_STRING="${3:-}"
OTM_ABLATION_ROOT="${OTM_ABLATION_ROOT:-./otm_ablation_results}"
OTM_BANK_SIZE="${OTM_BANK_SIZE:-8192}"
OTM_BANK_CHUNK="${OTM_BANK_CHUNK:-2048}"
OTM_BANK_SEED="${OTM_BANK_SEED:-42}"
VARIANTS=(baseline tangent raw_knn otm_full)

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

RUN_ROOT="${OTM_ABLATION_ROOT}/${PROTOCOL}/${DATASET_ALIAS}"
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

python tools/summarize_otm_ablation.py --root "${RUN_ROOT}"
echo "===== OTM ablation complete ====="

